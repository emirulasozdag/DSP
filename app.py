"""
Facial Image Warping, Aging, and Expression Transformation — Backend API
=========================================================================

Architecture (layered DSP pipeline)
-------------------------------------
  Input Layer        : Image upload, MIME validation, byte-size guard, decode.
  Preprocessing Layer: Face detection / crop, resize, normalisation, grayscale.
  DSP Processing     : Three independent pipelines —
      • Landmark Detection  — MediaPipe FaceMesh (468 pts) or Haar/skin fallback.
      • Expression Warping  — Muscle-based dense flow field; Delaunay triangulation
                              + per-triangle affine transforms; TPS spline warp.
      • Aging / De-Aging    — 2-D FFT frequency processing:
                                Aging   → high-pass filter (HPF) amplifies high-freq
                                          components (skin texture, wrinkles).
                                De-Aging → low-pass filter (LPF) smooths fine detail.
  Output Layer       : JPEG encode, Base-64, file save, JSON response.
  Evaluation Layer   : MSE, PSNR, SSIM, FFT energy spectrum comparison.

Key DSP concepts used
----------------------
  - 2-D Discrete Fourier Transform (DFT / FFT) via numpy.fft.fft2
  - Frequency-domain filtering: Gaussian, Butterworth (order-2), Ideal brick-wall
  - Parseval's theorem for energy-preserving normalisation
  - Delaunay triangulation (cv2.Subdiv2D) for mesh-based warping
  - Per-triangle affine transform: 3-point correspondence → 2×3 matrix → warpAffine
  - Thin Plate Spline (TPS) interpolation for smooth global deformation
  - Dense inverse displacement (flow) field remapped with cv2.remap

Performance target: ≤ 5 seconds end-to-end on standard hardware.
All matrix operations use NumPy vectorised routines (no Python-level pixel loops).
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import time
import uuid
import traceback
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Startup environment check — runs once when the server starts.
# Uses __file__ so the path is always relative to app.py, not the CWD.
# This prevents Flask's reloader child process from missing the .env file.
# The actual key value is never printed for security reasons.
# ---------------------------------------------------------------------------
print("CWD:", os.getcwd())
_env_path = Path(__file__).resolve().parent / ".env"  # always next to app.py
print(".env exists:", _env_path.exists())
load_dotenv(str(_env_path), override=True)  # override=True ensures reloader re-applies it
_replicate_key_check = os.getenv("REPLICATE_API_TOKEN", "").strip()
print("REPLICATE_API_TOKEN loaded:", bool(_replicate_key_check))
if not _replicate_key_check:
    print("WARNING: REPLICATE_API_TOKEN is not set or not loaded!")
del _replicate_key_check  # don't leave the key value dangling in module scope
# ---------------------------------------------------------------------------

try:
    import requests as _requests  # used as fallback / helper
except ImportError:  # pragma: no cover
    _requests = None

try:
    import replicate as _replicate
except ImportError:  # pragma: no cover
    _replicate = None

try:
    from google import genai as _genai
    from google.genai import types as _genai_types
except ImportError:  # pragma: no cover
    _genai = None
    _genai_types = None

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from skimage.metrics import structural_similarity

try:
    # aging.py — classical DSP aging module (Laplacian HPF + CLAHE + hair graying)
    from aging import apply_aging_effect as _classical_aging_effect
except Exception as _aging_import_err:  # pragma: no cover
    _classical_aging_effect = None
    print(f"aging.py import failed (classical method unavailable): {_aging_import_err}")

try:
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover - optional dependency path
    mp = None

try:
    import dlib as _dlib  # type: ignore
    _DLIB_DETECTOR = _dlib.get_frontal_face_detector()
    _DLIB_PREDICTOR_PATH = Path(__file__).resolve().parent / "models" / "shape_predictor_68_face_landmarks.dat"
    _DLIB_PREDICTOR = _dlib.shape_predictor(str(_DLIB_PREDICTOR_PATH)) if _DLIB_PREDICTOR_PATH.is_file() else None
    DLIB_AVAILABLE = _DLIB_PREDICTOR is not None
    print(f"Dlib available: detector=ready, predictor={'loaded' if DLIB_AVAILABLE else 'missing shape_predictor_68_face_landmarks.dat'}")
except Exception as _dlib_exc:
    _DLIB_DETECTOR = None
    _DLIB_PREDICTOR = None
    DLIB_AVAILABLE = False
    print(f"Dlib not available: {_dlib_exc}")

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
UPLOAD_DIR = ROOT / "uploads"
RESULTS_DIR = ROOT / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# results/ klasörünü uygulama başlangıcında temizle
for _f in RESULTS_DIR.iterdir():
    if _f.is_file():
        _f.unlink()

app = Flask(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_AGING_METHODS = {"freq", "ai", "classical"}
ALLOWED_HPF_FILTERS = {"gaussian_hpf", "butterworth_hpf", "ideal_hpf"}
ALLOWED_LPF_FILTERS = {"gaussian_lpf", "butterworth_lpf", "ideal_lpf"}


# Approximate mapping from MediaPipe FaceMesh indices to a dlib-style 68 landmark set.
MEDIAPIPE_TO_68 = [
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397,
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    168, 6, 197, 195,
    5, 4, 1, 19, 94,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    78, 95, 88, 178, 87, 14, 317, 402,
]

def _init_face_mesh():
    """Initialize MediaPipe FaceMesh if the installed package exposes it."""
    if mp is None:
        print("MediaPipe import unavailable: package could not be imported.")
        return None

    solutions = getattr(mp, "solutions", None)
    if solutions is not None:
        try:
            mesh = solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
            print("MediaPipe FaceMesh initialized successfully via mp.solutions.")
            return {"backend": "solutions", "runner": mesh}
        except Exception as exc:
            print(f"MediaPipe FaceMesh (mp.solutions) initialization failed: {exc}")
            print(traceback.format_exc())
    else:
        print("MediaPipe note: 'mp.solutions' is missing, trying Tasks API.")

    # Fallback for newer mediapipe builds that only expose Tasks API.
    tasks = getattr(mp, "tasks", None)
    vision = getattr(tasks, "vision", None) if tasks is not None else None
    if vision is None:
        print("MediaPipe initialization failed: neither 'mp.solutions' nor 'mp.tasks.vision' is available.")
        return None

    env_model = os.environ.get("MEDIAPIPE_FACE_LANDMARKER_MODEL", "").strip()
    candidates = []
    if env_model:
        candidates.append(Path(env_model))
    candidates.extend(
        [
            ROOT / "models" / "face_landmarker.task",
            ROOT / "models" / "face_landmarker_v2_with_blendshapes.task",
            ROOT / "face_landmarker.task",
        ]
    )
    model_path = next((p for p in candidates if p and p.is_file()), None)
    if model_path is None:
        print(
            "MediaPipe Tasks API is available but no FaceLandmarker model file was found. "
            "Set MEDIAPIPE_FACE_LANDMARKER_MODEL or place a model at models/face_landmarker.task"
        )
        return None

    try:
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        print(f"MediaPipe FaceLandmarker initialized successfully via Tasks API: {model_path}")
        return {"backend": "tasks", "runner": landmarker}
    except Exception as exc:
        print(f"MediaPipe FaceLandmarker (Tasks API) initialization failed: {exc}")
        print(traceback.format_exc())
        return None


FACE_MESH = _init_face_mesh()
print("Using MediaPipe:", FACE_MESH is not None)


def _get_mediapipe_points(image: np.ndarray):
    if FACE_MESH is None:
        return []

    h, w = image.shape[:2]
    backend = FACE_MESH.get("backend")
    runner = FACE_MESH.get("runner")

    if backend == "solutions":
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = runner.process(rgb)
        if not result.multi_face_landmarks:
            return []
        landmarks = result.multi_face_landmarks[0].landmark
    elif backend == "tasks":
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = runner.detect(mp_image)
        if not result.face_landmarks:
            return []
        landmarks = result.face_landmarks[0]
    else:
        print(f"Unknown MediaPipe backend: {backend}")
        return []

    points = []
    for idx, lm in enumerate(landmarks):
        x = int(lm.x * w)
        y = int(lm.y * h)
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        points.append({"id": idx, "x": x, "y": y})

    # Normalize to the canonical FaceMesh set (468). Some runtimes return 478 with iris points.
    if len(points) > 468:
        points = points[:468]
        for idx, p in enumerate(points):
            p["id"] = idx
    return points


def _reduce_mediapipe_to_68(points_468):
    if len(points_468) < 468:
        return []

    reduced = []
    for reduced_idx, mp_idx in enumerate(MEDIAPIPE_TO_68):
        if 0 <= mp_idx < len(points_468):
            p = points_468[mp_idx]
            reduced.append({"id": reduced_idx, "x": p["x"], "y": p["y"], "sourceId": mp_idx})
    return reduced


def _error(message: str, status_code: int = 400):
    return jsonify({"ok": False, "error": message}), status_code


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _get_dlib_landmarks(image: np.ndarray) -> list:
    """Return list of (x, y) pixel tuples using dlib 68-point predictor, or empty list."""
    if not DLIB_AVAILABLE or _DLIB_DETECTOR is None or _DLIB_PREDICTOR is None:
        return []
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dets = _DLIB_DETECTOR(gray, 1)
        if not dets:
            return []
        det = max(dets, key=lambda d: (d.right() - d.left()) * (d.bottom() - d.top()))
        shape = _DLIB_PREDICTOR(gray, det)
        h, w = image.shape[:2]
        return [
            (max(0, min(w - 1, shape.part(i).x)), max(0, min(h - 1, shape.part(i).y)))
            for i in range(68)
        ]
    except Exception:
        return []


def _parse_bool(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _read_image(file_storage):
    if file_storage is None or not file_storage.filename:
        raise ValueError("Image file is required.")

    mime = (getattr(file_storage, "mimetype", "") or "").lower()
    if mime and mime not in ALLOWED_MIME_TYPES:
        raise ValueError("Unsupported file type. Please upload JPG or PNG.")

    raw = file_storage.read()
    if not raw:
        raise ValueError("Uploaded image is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("Uploaded image exceeds the 10 MB limit.")

    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode uploaded image.")

    h, w = image.shape[:2]
    if h < 64 or w < 64:
        raise ValueError("Image resolution is too small. Minimum is 64x64.")
    if h > 4096 or w > 4096:
        raise ValueError("Image resolution is too large. Maximum is 4096x4096.")

    return image


def _read_image_from_bytes(raw: bytes):
    """Decode BGR image from raw bytes (already read from a file/stream)."""
    if not raw:
        raise ValueError("Image bytes are empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("Image exceeds the 10 MB limit.")
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image.")
    h, w = image.shape[:2]
    if h < 64 or w < 64:
        raise ValueError("Image resolution is too small. Minimum is 64x64.")
    if h > 4096 or w > 4096:
        raise ValueError("Image resolution is too large. Maximum is 4096x4096.")
    return image


def _save_image(image: np.ndarray, prefix: str) -> str:
    name = f"{prefix}-{uuid.uuid4().hex[:8]}.jpg"
    path = RESULTS_DIR / name
    ok = cv2.imwrite(str(path), image)
    if not ok:
        return ""
    return str(path.relative_to(ROOT))


def _encode_base64_jpeg(image: np.ndarray, quality: int = 92) -> str:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("Failed to encode image.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _make_deterministic_seed(image: np.ndarray, *tokens) -> int:
    # Build a stable seed from image content + parameters for reproducible demo outputs.
    digest = hashlib.sha256()
    digest.update(image.tobytes())
    for token in tokens:
        digest.update(str(token).encode("utf-8"))
        digest.update(b"\x1f")
    return int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)


def _resize_like(image: np.ndarray, target_shape) -> np.ndarray:
    h, w = target_shape[:2]
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


def _normalize_for_preprocess(image: np.ndarray, mode: str) -> np.ndarray:
    src = image.astype(np.float32)
    mode = (mode or "1").strip().lower()

    if mode == "255":
        return image

    if mode == "bipolar":
        mean = float(np.mean(src))
        std = float(np.std(src)) + 1e-6
        z = (src - mean) / std
        # Clamp to a practical range, then map back to uint8 display space.
        z = np.clip(z, -2.5, 2.5)
        out = ((z + 2.5) / 5.0) * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)

    # Default "0..1" mode: stretch dynamic range in display space.
    mn = float(np.min(src))
    mx = float(np.max(src))
    if mx - mn < 1e-6:
        return image
    out = (src - mn) / (mx - mn)
    out = out * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _crop_face_region(image: np.ndarray, padding_percent: float, use_detector: bool) -> tuple[np.ndarray, bool, str]:
    if not use_detector:
        return image, False, "none"

    det = _detect_face_bbox(image)
    if det is None:
        return image, False, "fallback-none"

    x, y, w, h, method = det
    pad_ratio = max(0.0, min(0.45, float(padding_percent) / 100.0))
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(image.shape[1], x + w + pad_x)
    y1 = min(image.shape[0], y + h + pad_y)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return image, False, "fallback-empty"
    return crop, True, method


def _fft_energy_components(gray: np.ndarray):
    """
    Decompose the 2-D power spectrum of a grayscale image into low-frequency
    and high-frequency energy bands.

    Power Spectral Density (PSD)
    ----------------------------
    By Parseval's theorem, the total signal energy is conserved between the
    spatial and frequency domains:

        Σ |f(x,y)|²  =  (1/N²) Σ |F(u,v)|²

    where N = image_height × image_width.  We normalise each |F(u,v)|² by N²
    so that energy values are independent of image resolution (otherwise a
    larger image would trivially produce more "energy").

    Low-frequency energy  : sum of PSD inside a circular radius of (min(H,W)/8)
                            around the DC centre.  Represents smooth luminance,
                            skin tone, and slow-varying features.
    High-frequency energy : sum of PSD outside that circle.  Represents edges,
                            fine texture, wrinkles — the components modified by
                            the aging / de-aging pipeline.

    hfLfRatio  = high_energy / low_energy
        • A high ratio indicates a texture-rich, aged-looking face.
        • This ratio is compared before/after processing in the Analysis tab.
    """
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.abs(spectrum)
    # Normalise FFT energy with Parseval scaling so values are comparable
    # across image sizes instead of growing with the square of pixel count.
    norm = float(gray.shape[0] * gray.shape[1]) ** 2
    energy_map = (magnitude ** 2) / max(norm, 1.0)

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    # Low-frequency circle radius: 1/8 of the shorter image dimension
    radius = max(8, min(h, w) // 8)
    low_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius

    low_energy  = float(np.sum(energy_map[low_mask]))
    high_energy = float(np.sum(energy_map[~low_mask]))
    total_energy = low_energy + high_energy
    # Ratio guard: +1e-9 avoids division by zero on flat (zero-variance) images
    hf_lf_ratio = float(high_energy / (low_energy + 1e-9))

    return {
        "spectrum":  spectrum,
        "magnitude": magnitude,
        "energyMap": energy_map,
        "energy": {
            "total":      total_energy,
            "high":       high_energy,
            "low":        low_energy,
            "hfLfRatio":  hf_lf_ratio,
        },
    }


def _gray_image_stats(gray: np.ndarray):
    gray_f = gray.astype(np.float32)
    return {
        "mean": float(np.mean(gray_f)),
        "std": float(np.std(gray_f)),
        "min": float(np.min(gray_f)),
        "max": float(np.max(gray_f)),
    }


def _percent_delta(before: float, after: float) -> float:
    return float(((after - before) / (abs(before) + 1e-9)) * 100.0)


def _fft_visuals_b64(gray: np.ndarray):
    spec = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(spec))
    phase = np.angle(spec)

    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    phase_norm = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    mag_vis = cv2.applyColorMap(mag_norm, cv2.COLORMAP_INFERNO)
    phase_vis = cv2.applyColorMap(phase_norm, cv2.COLORMAP_TWILIGHT)

    return {
        "magnitude": _encode_base64_jpeg(mag_vis, quality=90),
        "phase": _encode_base64_jpeg(phase_vis, quality=90),
    }


def _odd_kernel(size: int, minimum: int = 3) -> int:
    size = max(minimum, int(size))
    return size if size % 2 == 1 else size + 1


def _round_energy(energy: dict) -> dict:
    return {key: round(float(value), 4) for key, value in energy.items()}


def _skin_color_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    mask_hsv = cv2.inRange(hsv, (0, 18, 45), (28, 210, 255))
    mask_ycrcb = cv2.inRange(ycrcb, (0, 132, 72), (255, 178, 136))
    mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _face_bbox_or_default(image: np.ndarray) -> tuple[int, int, int, int, str]:
    h, w = image.shape[:2]
    det = _detect_face_bbox(image)
    if det is not None:
        x, y, bw, bh, method = det
        return int(x), int(y), int(bw), int(bh), method
    return int(w * 0.18), int(h * 0.10), int(w * 0.64), int(h * 0.80), "fallback-geo"


def _fill_feature_masks(mask: np.ndarray, face_box: tuple[int, int, int, int]):
    x, y, w, h = face_box

    def ellipse(cx, cy, ax, ay, angle=0.0, value=255):
        cv2.ellipse(
            mask,
            (int(round(cx)), int(round(cy))),
            (max(1, int(round(ax))), max(1, int(round(ay)))),
            float(angle),
            0,
            360,
            int(value),
            -1,
            lineType=cv2.LINE_AA,
        )

    ellipse(x + 0.31 * w, y + 0.40 * h, 0.16 * w, 0.075 * h, -5)
    ellipse(x + 0.69 * w, y + 0.40 * h, 0.16 * w, 0.075 * h, 5)
    ellipse(x + 0.31 * w, y + 0.31 * h, 0.19 * w, 0.045 * h, -7)
    ellipse(x + 0.69 * w, y + 0.31 * h, 0.19 * w, 0.045 * h, 7)
    ellipse(x + 0.50 * w, y + 0.59 * h, 0.13 * w, 0.16 * h, 0)
    ellipse(x + 0.50 * w, y + 0.76 * h, 0.26 * w, 0.095 * h, 0)


def _point_from_landmarks(points, idx: int):
    if points and 0 <= idx < len(points):
        p = points[idx]
        return np.array([float(p["x"]), float(p["y"])], dtype=np.float32)
    return None


def _mean_landmark_point(points, indices):
    selected = [_point_from_landmarks(points, idx) for idx in indices]
    selected = [p for p in selected if p is not None]
    if not selected:
        return None
    return np.mean(np.vstack(selected), axis=0)


def _fill_feature_masks_from_landmarks(mask: np.ndarray, points) -> bool:
    if not points or len(points) < 468:
        return False

    h, w = mask.shape[:2]

    def add_poly(indices, expand=5):
        pts = [_point_from_landmarks(points, i) for i in indices]
        pts = [p for p in pts if p is not None]
        if len(pts) < 3:
            return
        arr = np.array(pts, dtype=np.int32)
        hull = cv2.convexHull(arr)
        # Draw the new region on a temp mask so dilation only affects the new
        # polygon, not the already-accumulated mask from previous add_poly calls.
        temp = np.zeros_like(mask)
        cv2.fillConvexPoly(temp, hull, 255, lineType=cv2.LINE_AA)
        if expand > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd_kernel(expand), _odd_kernel(expand)))
            temp = cv2.dilate(temp, kernel, iterations=1)
        mask[:] = cv2.bitwise_or(mask, temp)

    add_poly([33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173], 5)
    add_poly([362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398], 5)
    add_poly([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95], 7)
    add_poly([1, 4, 5, 195, 197, 98, 327, 2, 94, 19], 5)

    brow_left = _mean_landmark_point(points, [70, 63, 105, 66, 107])
    brow_right = _mean_landmark_point(points, [336, 296, 334, 293, 300])
    if brow_left is not None and brow_right is not None:
        brow_w = max(8, int(np.linalg.norm(brow_right - brow_left) * 0.24))
        for center in (brow_left, brow_right):
            cv2.ellipse(mask, tuple(np.round(center).astype(int)), (brow_w, max(4, brow_w // 3)), 0, 0, 360, 255, -1, lineType=cv2.LINE_AA)

    return np.count_nonzero(mask) > h * w * 0.01


def _build_age_region_masks(image: np.ndarray, preserve_edges: bool = True) -> dict:
    h, w = image.shape[:2]
    face_mask = np.zeros((h, w), dtype=np.uint8)
    raw_points = []
    mask_method = "fallback-geo"

    if FACE_MESH is not None:
        try:
            raw_points = _get_mediapipe_points(image)
        except Exception:
            raw_points = []

    if len(raw_points) >= 32:
        pts = np.array([[p["x"], p["y"]] for p in raw_points], dtype=np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(face_mask, hull, 255, lineType=cv2.LINE_AA)
        x, y, bw, bh = cv2.boundingRect(hull)
        mask_method = "mediapipe"

        # FaceMesh does not fully include the forehead, so add a soft oval above the brow.
        cv2.ellipse(
            face_mask,
            (int(x + bw * 0.5), int(y + bh * 0.18)),
            (max(1, int(bw * 0.42)), max(1, int(bh * 0.20))),
            0,
            0,
            360,
            255,
            -1,
            lineType=cv2.LINE_AA,
        )
    else:
        x, y, bw, bh, mask_method = _face_bbox_or_default(image)
        cv2.ellipse(
            face_mask,
            (int(x + bw * 0.5), int(y + bh * 0.52)),
            (max(1, int(bw * 0.50)), max(1, int(bh * 0.56))),
            0,
            0,
            360,
            255,
            -1,
            lineType=cv2.LINE_AA,
        )

    skin_mask = _skin_color_mask(image)
    face_area = max(1, int(np.count_nonzero(face_mask)))
    skin_in_face = cv2.bitwise_and(skin_mask, face_mask)
    if np.count_nonzero(skin_in_face) < face_area * 0.12:
        skin_in_face = face_mask.copy()

    # Feature-protection mask: only eyes, nose, mouth, eyebrows.
    # Canny edges are intentionally NOT included here — adding them would make
    # the protect mask cover most textured skin and kill the processing_float,
    # leaving wrinkle / HPF effects invisible.  Edge-aware output restoration
    # is handled separately inside _frequency_age_pipeline.
    protect = np.zeros((h, w), dtype=np.uint8)
    if not _fill_feature_masks_from_landmarks(protect, raw_points):
        _fill_feature_masks(protect, (x, y, bw, bh))

    blur_size = _odd_kernel(min(h, w) // 12, 15)
    skin_float = cv2.GaussianBlur(skin_in_face.astype(np.float32) / 255.0, (blur_size, blur_size), 0)
    face_float = cv2.GaussianBlur(face_mask.astype(np.float32) / 255.0, (blur_size, blur_size), 0)
    skin_float = np.clip(np.maximum(skin_float, face_float * 0.52), 0.0, 1.0)

    protect_size = _odd_kernel(min(h, w) // 32, 5)
    protected_float = cv2.GaussianBlur(protect.astype(np.float32) / 255.0, (protect_size, protect_size), 0)
    protected_float = np.clip(protected_float, 0.0, 1.0)

    # processing_float: where wrinkles / HPF effects are applied.
    # Exclude only the compact facial-feature regions (eyes, nose, mouth) —
    # the wide skin/forehead/cheek area stays fully active.
    processing_float = skin_float * (1.0 - 0.72 * protected_float)
    processing_float = np.clip(processing_float, 0.0, 1.0)

    return {
        "skinMask": skin_float,
        "processingMask": processing_float,
        "protectedMask": protected_float,
        "faceBox": (int(x), int(y), int(bw), int(bh)),
        "landmarks": raw_points,
        "maskMethod": mask_method,
    }


def _frequency_mask(shape: tuple[int, int], filter_name: str) -> np.ndarray:
    """
    Build a 2-D frequency-domain filter mask in the *shifted* spectrum space,
    where the DC component (zero frequency) sits at the image centre.

    Radial normalisation
    --------------------
    All distances are normalised to [0, 1] by dividing by the half-diagonal of
    the spectrum, so the same cutoff fraction D₀ works for any image resolution.

    Filter designs (Low-Pass, then HPF = 1 − LPF)
    -----------------------------------------------
    Gaussian LPF   H(r) = exp(−r² / (2·D₀²))
        • Infinitely smooth roll-off; the spatial-domain PSF is also Gaussian.
        • No ringing artefacts — preferred for natural-looking smoothing.

    Butterworth LPF (order n=2)   H(r) = 1 / (1 + (r / D₀)^(2n))
        • Steeper transition band than Gaussian, still no hard discontinuity.
        • Balances sharpness of cut with freedom from Gibbs ringing.

    Ideal (brick-wall) LPF   H(r) = 1 if r ≤ D₀ else 0
        • Perfect frequency separation but the sharp discontinuity in the
          frequency domain produces strong ringing (Gibbs phenomenon) in the
          reconstructed spatial image — useful mainly for analysis.

    HPF variants
    ------------
    Return (1 − LPF) so that only high-frequency components (edges, texture,
    wrinkles) are passed and low-frequency (smooth luminance) is suppressed.
    This is the core mechanism for the *aging* path: amplifying high-frequency
    content makes the skin appear more textured / aged.
    """
    h, w = shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    # Euclidean distance from DC centre for every pixel in the spectrum
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    # Normalise so the corner of the spectrum is radius ≈ 1.0
    radius = radius / (np.sqrt(cy * cy + cx * cx) + 1e-6)

    is_high = filter_name.endswith("_hpf")
    # Cutoff frequencies D₀ chosen so visual effect is meaningful on face crops:
    # LPF cutoffs are slightly higher (keep more detail); HPF cutoffs are lower
    # (remove more smooth background and keep fine texture).
    low_cutoffs = {
        "gaussian_lpf":    0.28,
        "butterworth_lpf": 0.20,
        "ideal_lpf":       0.13,
        "gaussian_hpf":    0.22,
        "butterworth_hpf": 0.17,
        "ideal_hpf":       0.12,
    }
    cutoff = low_cutoffs.get(filter_name, 0.20)

    if filter_name.startswith("gaussian"):
        # Gaussian LPF: smooth frequency-domain envelope
        low = np.exp(-(radius ** 2) / (2.0 * cutoff * cutoff))
    elif filter_name.startswith("butterworth"):
        # Butterworth LPF order-2: steeper roll-off, minimal ringing
        low = 1.0 / (1.0 + (radius / (cutoff + 1e-6)) ** 4)
    else:
        # Ideal (brick-wall) LPF: binary mask at cutoff radius
        low = (radius <= cutoff).astype(np.float32)

    # For HPF: invert the LPF mask (high = 1 − low)
    return (1.0 - low if is_high else low).astype(np.float32)


def _frequency_reconstruct(channel: np.ndarray, filter_name: str) -> np.ndarray:
    """
    Apply a frequency-domain filter to a single-channel (grayscale) image and
    return the filtered result in the spatial domain.

    2-D FFT filtering pipeline
    --------------------------
    Step 1 — Forward transform:   F = fft2(channel)
        Convert the spatial image to the complex frequency domain.
        Each coefficient F[u,v] represents the amplitude and phase of the
        sinusoidal component at spatial frequency (u, v).

    Step 2 — Centre the spectrum: F_s = fftshift(F)
        Rearrange quadrants so that DC (u=0, v=0, the mean intensity) sits at
        the array centre.  This makes the radial filter mask trivial to compute.

    Step 3 — Multiply by mask:    F_filtered = F_s * H(u,v)
        Element-wise multiplication in the frequency domain is equivalent to
        *convolution* with the filter's impulse response in the spatial domain
        (Convolution Theorem).  H(u,v) is the filter mask from _frequency_mask.

    Step 4 — Unshift:             F_back = ifftshift(F_filtered)
        Undo the quadrant swap before applying the inverse transform.

    Step 5 — Inverse transform:   recon = Re{ ifft2(F_back) }
        Convert back to the spatial domain.  We retain only the real part;
        any imaginary residual is purely due to floating-point rounding and is
        numerically negligible for real-valued inputs.

    The output is in the same pixel-intensity scale as the input channel.
    """
    # Step 1+2: FFT and shift DC to centre
    spec = np.fft.fftshift(np.fft.fft2(channel.astype(np.float32)))
    # Step 3: build and apply frequency-domain filter mask
    mask = _frequency_mask(channel.shape, filter_name)
    # Steps 4+5: un-shift and inverse FFT; keep real part only
    recon = np.fft.ifft2(np.fft.ifftshift(spec * mask)).real
    return recon.astype(np.float32)


def _masked_high_frequency_energy(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    high = _frequency_reconstruct(gray, "gaussian_hpf")
    if mask is None:
        mask = np.ones_like(gray, dtype=np.float32)
    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    denom = float(np.sum(mask)) + 1e-6
    return float(np.sum((high * mask) ** 2) / denom)


def _draw_wrinkle_line(layer: np.ndarray, p1, p2, thickness: int):
    p1 = (int(round(p1[0])), int(round(p1[1])))
    p2 = (int(round(p2[0])), int(round(p2[1])))
    cv2.line(layer, p1, p2, -1.0, thickness, lineType=cv2.LINE_AA)
    cv2.line(
        layer,
        (p1[0], p1[1] + max(1, thickness)),
        (p2[0], p2[1] + max(1, thickness)),
        0.32,
        max(1, thickness),
        lineType=cv2.LINE_AA,
    )


def _draw_wrinkle_curve(layer: np.ndarray, points, thickness: int):
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(layer, [pts], False, -1.0, thickness, lineType=cv2.LINE_AA)
    if len(points) >= 3:
        highlight = np.array([[p[0], p[1] + max(1, thickness)] for p in points], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(layer, [highlight], False, 0.24, max(1, thickness), lineType=cv2.LINE_AA)


def _generate_landmark_wrinkle_layer(shape: tuple[int, int], landmarks, face_box: tuple[int, int, int, int], rng, semantic_boost: float) -> np.ndarray | None:
    if not landmarks or len(landmarks) < 468:
        return None

    h, w = shape
    layer = np.zeros((h, w), dtype=np.float32)
    thickness = max(1, int(round(min(w, h) / 280.0)))

    left_eye_outer = _point_from_landmarks(landmarks, 33)
    left_eye_inner = _point_from_landmarks(landmarks, 133)
    right_eye_inner = _point_from_landmarks(landmarks, 362)
    right_eye_outer = _point_from_landmarks(landmarks, 263)
    brow_left = _mean_landmark_point(landmarks, [70, 63, 105, 66, 107])
    brow_right = _mean_landmark_point(landmarks, [336, 296, 334, 293, 300])
    mouth_left = _point_from_landmarks(landmarks, 61)
    mouth_right = _point_from_landmarks(landmarks, 291)
    nose_left = _point_from_landmarks(landmarks, 98)
    nose_right = _point_from_landmarks(landmarks, 327)

    anchors = [left_eye_outer, left_eye_inner, right_eye_inner, right_eye_outer, brow_left, brow_right, mouth_left, mouth_right]
    if any(p is None for p in anchors):
        return None

    eye_span = max(12.0, float(np.linalg.norm(right_eye_outer - left_eye_outer)))
    face_width = max(1.0, float(face_box[2]))
    forehead_left = brow_left * 0.76 + left_eye_outer * 0.24
    forehead_right = brow_right * 0.76 + right_eye_outer * 0.24
    brow_y = min(float(brow_left[1]), float(brow_right[1]))
    top_y = max(0.0, brow_y - face_box[3] * 0.24)

    for frac in (0.32, 0.51, 0.70):
        y = top_y * (1.0 - frac) + brow_y * frac + float(rng.uniform(-1.2, 1.2))
        left = forehead_left + np.array([face_width * 0.03, y - forehead_left[1]], dtype=np.float32)
        right = forehead_right + np.array([-face_width * 0.03, y - forehead_right[1]], dtype=np.float32)
        mid = (left + right) * 0.5 + np.array([float(rng.uniform(-6.0, 6.0)), float(rng.uniform(-2.2, 2.2))], dtype=np.float32)
        _draw_wrinkle_curve(layer, [left, mid, right], thickness)

    for outer, direction in ((left_eye_outer, -1.0), (right_eye_outer, 1.0)):
        for dy, length in ((-0.030, 0.105), (0.008, 0.125), (0.046, 0.100)):
            start = outer + np.array([direction * eye_span * 0.015, eye_span * dy], dtype=np.float32)
            end = start + np.array([direction * eye_span * length, eye_span * (dy * 0.22)], dtype=np.float32)
            _draw_wrinkle_line(layer, start, end, thickness)

    if nose_left is not None and nose_right is not None:
        for mouth, nose, direction in ((mouth_left, nose_left, -1.0), (mouth_right, nose_right, 1.0)):
            start = nose * 0.72 + mouth * 0.28
            mid = nose * 0.38 + mouth * 0.62 + np.array([direction * eye_span * 0.025, 0.0], dtype=np.float32)
            end = mouth + np.array([direction * eye_span * 0.050, eye_span * 0.040], dtype=np.float32)
            _draw_wrinkle_curve(layer, [start, mid, end], thickness)

    for under_left, under_right in ((left_eye_outer, left_eye_inner), (right_eye_inner, right_eye_outer)):
        base_y = max(float(under_left[1]), float(under_right[1])) + eye_span * 0.055
        left = np.array([under_left[0], base_y], dtype=np.float32)
        right = np.array([under_right[0], base_y + float(rng.uniform(-1.0, 1.0))], dtype=np.float32)
        mid = (left + right) * 0.5 + np.array([0.0, eye_span * 0.020], dtype=np.float32)
        _draw_wrinkle_curve(layer, [left, mid, right], thickness)

    # Blur radius scales with image size: ~min(h,w)/30 px ensures that drawn
    # wrinkle strokes become wide, soft shadows rather than sharp black lines.
    blur_k = _odd_kernel(max(7, int(min(h, w) / 30)))
    layer = cv2.GaussianBlur(layer, (blur_k, blur_k), blur_k * 0.35)
    return layer * float(semantic_boost)


def _generate_wrinkle_layer(shape: tuple[int, int], face_box: tuple[int, int, int, int], rng, semantic_boost: float, landmarks=None) -> np.ndarray:
    landmark_layer = _generate_landmark_wrinkle_layer(shape, landmarks, face_box, rng, semantic_boost)
    if landmark_layer is not None:
        return landmark_layer

    h, w = shape
    x, y, bw, bh = face_box
    layer = np.zeros((h, w), dtype=np.float32)
    thickness = max(1, int(round(min(w, h) / 280.0)))

    forehead_y = [0.250, 0.305, 0.360]
    for frac in forehead_y:
        jitter = int(rng.integers(-max(1, bh // 90), max(2, bh // 90 + 1)))
        yy = y + bh * frac + jitter
        x1 = x + bw * (0.33 + float(rng.uniform(-0.015, 0.015)))
        x2 = x + bw * (0.67 + float(rng.uniform(-0.015, 0.015)))
        mid = ((x1 + x2) * 0.5, yy + float(rng.uniform(-2.0, 2.0)))
        _draw_wrinkle_curve(layer, [(x1, yy), mid, (x2, yy + float(rng.uniform(-2.0, 2.0)))], thickness)

    eye_lines = [
        ((0.31, 0.41), (-0.06, -0.020)),
        ((0.31, 0.44), (-0.07, 0.015)),
        ((0.31, 0.47), (-0.055, 0.045)),
        ((0.69, 0.41), (0.06, -0.020)),
        ((0.69, 0.44), (0.07, 0.015)),
        ((0.69, 0.47), (0.055, 0.045)),
    ]
    for (sx, sy), (dx, dy) in eye_lines:
        start = (x + bw * sx, y + bh * sy)
        end = (x + bw * (sx + dx), y + bh * (sy + dy))
        _draw_wrinkle_line(layer, start, end, thickness)

    cv2.ellipse(layer, (int(x + bw * 0.35), int(y + bh * 0.49)), (int(bw * 0.10), int(bh * 0.035)), 0, 8, 172, -0.55, thickness, lineType=cv2.LINE_AA)
    cv2.ellipse(layer, (int(x + bw * 0.65), int(y + bh * 0.49)), (int(bw * 0.10), int(bh * 0.035)), 0, 8, 172, -0.55, thickness, lineType=cv2.LINE_AA)

    _draw_wrinkle_line(layer, (x + bw * 0.40, y + bh * 0.58), (x + bw * 0.33, y + bh * 0.76), thickness)
    _draw_wrinkle_line(layer, (x + bw * 0.60, y + bh * 0.58), (x + bw * 0.67, y + bh * 0.76), thickness)

    for side in (-1, 1):
        for _ in range(8):
            cx = x + bw * (0.37 if side < 0 else 0.63) + float(rng.uniform(-0.045, 0.045)) * bw
            cy = y + bh * 0.62 + float(rng.uniform(-0.07, 0.09)) * bh
            length = bw * float(rng.uniform(0.045, 0.095))
            slope = bh * float(rng.uniform(-0.018, 0.018))
            _draw_wrinkle_line(layer, (cx, cy), (cx + side * length, cy + slope), thickness)

    # Blur radius scales with image size so drawn strokes become soft shadows
    blur_k = _odd_kernel(max(7, int(min(h, w) / 30)))
    layer = cv2.GaussianBlur(layer, (blur_k, blur_k), blur_k * 0.35)
    return layer * float(semantic_boost)


def _generate_age_shadow_layer(shape: tuple[int, int], face_box: tuple[int, int, int, int], landmarks, rng) -> np.ndarray:
    h, w = shape
    x, y, bw, bh = face_box
    layer = np.zeros((h, w), dtype=np.float32)

    def soft_ellipse(center, axes, angle=0.0, value=-1.0):
        temp = np.zeros_like(layer)
        cv2.ellipse(
            temp,
            tuple(np.round(center).astype(int)),
            (max(1, int(round(axes[0]))), max(1, int(round(axes[1])))),
            float(angle),
            0,
            360,
            float(value),
            -1,
            lineType=cv2.LINE_AA,
        )
        return cv2.GaussianBlur(temp, (_odd_kernel(max(7, int(min(w, h) * 0.025))), _odd_kernel(max(7, int(min(w, h) * 0.025)))), 0)

    if landmarks and len(landmarks) >= 468:
        left_outer = _point_from_landmarks(landmarks, 33)
        left_inner = _point_from_landmarks(landmarks, 133)
        right_inner = _point_from_landmarks(landmarks, 362)
        right_outer = _point_from_landmarks(landmarks, 263)
        mouth_left = _point_from_landmarks(landmarks, 61)
        mouth_right = _point_from_landmarks(landmarks, 291)
        nose_left = _point_from_landmarks(landmarks, 98)
        nose_right = _point_from_landmarks(landmarks, 327)

        if all(p is not None for p in (left_outer, left_inner, right_inner, right_outer)):
            eye_span = max(12.0, float(np.linalg.norm(right_outer - left_outer)))
            for a, b in ((left_outer, left_inner), (right_inner, right_outer)):
                center = (a + b) * 0.5 + np.array([0.0, eye_span * 0.045], dtype=np.float32)
                layer += soft_ellipse(center, (eye_span * 0.105, eye_span * 0.028), 0, -0.75)

        if all(p is not None for p in (mouth_left, mouth_right, nose_left, nose_right)):
            thickness = max(2, int(round(min(w, h) / 190.0)))
            for mouth, nose, direction in ((mouth_left, nose_left, -1.0), (mouth_right, nose_right, 1.0)):
                pts = np.array(
                    [
                        nose * 0.78 + mouth * 0.22,
                        nose * 0.46 + mouth * 0.54 + np.array([direction * bw * 0.035, bh * 0.025], dtype=np.float32),
                        mouth + np.array([direction * bw * 0.050, bh * 0.050], dtype=np.float32),
                    ],
                    dtype=np.int32,
                ).reshape((-1, 1, 2))
                cv2.polylines(layer, [pts], False, -1.0, thickness, lineType=cv2.LINE_AA)
    else:
        layer += soft_ellipse((x + bw * 0.35, y + bh * 0.49), (bw * 0.12, bh * 0.035), 0, -0.65)
        layer += soft_ellipse((x + bw * 0.65, y + bh * 0.49), (bw * 0.12, bh * 0.035), 0, -0.65)
        cv2.line(layer, (int(x + bw * 0.40), int(y + bh * 0.59)), (int(x + bw * 0.34), int(y + bh * 0.76)), -1.0, max(2, int(min(w, h) / 190)), lineType=cv2.LINE_AA)
        cv2.line(layer, (int(x + bw * 0.60), int(y + bh * 0.59)), (int(x + bw * 0.66), int(y + bh * 0.76)), -1.0, max(2, int(min(w, h) / 190)), lineType=cv2.LINE_AA)

    # Use a larger blur (~3% of image) so nasolabial/shadow lines look like
    # soft depth shading rather than sharp geometric strokes.
    blur_k = _odd_kernel(max(11, int(min(w, h) * 0.030)))
    layer = cv2.GaussianBlur(layer, (blur_k, blur_k), blur_k * 0.40)
    return np.clip(layer, -1.0, 0.35)


def _apply_age_hair_graying(out: np.ndarray, original: np.ndarray, masks: dict, intensity: float, rng) -> np.ndarray:
    h, w = original.shape[:2]
    x, y, bw, bh = masks["faceBox"]
    region = np.zeros((h, w), dtype=np.float32)
    x0 = max(0, int(x - bw * 0.12))
    x1 = min(w, int(x + bw * 1.12))
    y0 = max(0, int(y - bh * 0.10))
    y1 = min(h, int(y + bh * 0.92))
    region[y0:y1, x0:x1] = 1.0

    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    dark = ((gray < 88) & (hsv[:, :, 1] < 150)).astype(np.float32)
    non_skin = 1.0 - np.clip(masks["skinMask"], 0.0, 1.0)
    protect = np.clip(masks["protectedMask"], 0.0, 1.0)
    hair_mask = dark * non_skin * region * (1.0 - protect * 0.75)

    if np.count_nonzero(hair_mask > 0.2) < 32:
        return out

    strand_noise = rng.normal(0.0, 1.0, size=gray.shape).astype(np.float32)
    strand_noise = cv2.GaussianBlur(strand_noise, (1, _odd_kernel(max(5, int(h * 0.018)))), 0)
    strand_noise = cv2.normalize(strand_noise, None, 0.0, 1.0, cv2.NORM_MINMAX)
    hair_mask *= np.clip(0.40 + 0.85 * strand_noise, 0.0, 1.0)
    hair_mask = cv2.GaussianBlur(hair_mask, (_odd_kernel(max(3, int(min(w, h) * 0.006))), _odd_kernel(max(3, int(min(w, h) * 0.006)))), 0)
    hair_mask = np.clip(hair_mask * (0.18 + 0.28 * intensity), 0.0, 0.42)

    target = np.full_like(out, 150.0, dtype=np.float32)
    return out * (1.0 - hair_mask[:, :, None]) + target * hair_mask[:, :, None]


def _apply_lab_color_tone(lab: np.ndarray, skin_mask: np.ndarray, mode: str, intensity: float, tone: float):
    tone_shift = (tone - 0.5) * 22.0
    if mode == "aging":
        lab[:, :, 0] += skin_mask * (tone_shift - 3.0 * intensity)
        lab[:, :, 1] += skin_mask * (0.8 + 1.3 * intensity)
        lab[:, :, 2] += skin_mask * (1.4 + 3.0 * intensity)
    else:
        lab[:, :, 0] += skin_mask * (tone_shift + 4.0 * intensity)
        lab[:, :, 1] -= skin_mask * (1.0 + 2.0 * intensity)
        lab[:, :, 2] -= skin_mask * (1.0 + 2.5 * intensity)


def _apply_skin_contrast(luminance: np.ndarray, skin_mask: np.ndarray, mode: str, intensity: float, contrast: float) -> np.ndarray:
    if mode == "aging":
        gain = 1.0 + 0.22 * intensity + (contrast - 0.5) * 0.70
    else:
        gain = 1.0 - 0.16 * intensity + (contrast - 0.5) * 0.48
    adjusted = (luminance - 128.0) * gain + 128.0
    return luminance * (1.0 - skin_mask) + adjusted * skin_mask


def _age_intensity(strength: float, years: int, semantic_boost: float) -> float:
    year_gain = min(1.0, abs(float(years)) / 50.0)
    return _clamp(strength * (0.65 + 0.35 * year_gain) * semantic_boost, 0.0, 1.35)


def _build_hair_exclusion_mask(
    image: np.ndarray,
    skin_mask: np.ndarray,
    face_box: tuple,
) -> np.ndarray:
    """
    Build a float32 [0..1] mask of hair pixels so they can be excluded from
    wrinkle enhancement, noise injection, and the final processed-image blend.

    Strategy:
      - Restrict to the upper-face region plus a small lateral extension (where
        hair typically sits above and beside the face bounding box).
      - Within that region take non-skin pixels.
      - Refine with HSV: dark hair (low value) OR light/grey hair (low saturation).
      - Smooth with a Gaussian blur to feather the mask edges naturally.
    """
    h, w = image.shape[:2]
    x, y, bw, bh = face_box

    # Region: upper face + slight lateral pad, stopping before the chin area
    region = np.zeros((h, w), dtype=np.float32)
    x0 = max(0, int(x - bw * 0.12))
    x1 = min(w, int(x + bw * 1.12))
    y0 = max(0, int(y - bh * 0.08))
    y1 = min(h, int(y + bh * 0.88))
    region[y0:y1, x0:x1] = 1.0

    non_skin = 1.0 - np.clip(skin_mask, 0.0, 1.0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2].astype(np.float32) / 255.0
    sat = hsv[:, :, 1].astype(np.float32) / 255.0

    # Dark hair (val < 0.62) OR light / grey hair (sat < 0.22)
    colour_hair = np.clip(
        (val < 0.62).astype(np.float32) + (sat < 0.22).astype(np.float32),
        0.0, 1.0,
    )

    hair = non_skin * region * colour_hair
    blur_k = _odd_kernel(max(9, min(w, h) // 18))
    hair = cv2.GaussianBlur(hair, (blur_k, blur_k), 0)
    return np.clip(hair, 0.0, 1.0)


def _frequency_age_pipeline(
    image: np.ndarray,
    mode: str,
    strength: float,
    years: int,
    contrast: float,
    tone: float,
    noise: float,
    detail: float,
    deterministic: bool,
    hpf: str,
    lpf: str,
    preserve_edges: bool,
    semantic_boost: float = 1.0,
    save_debug: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Frequency-domain face aging / de-aging pipeline.

    Core DSP principle
    ------------------
    Human skin *ages* by accumulating fine-scale surface texture (wrinkles,
    pores, micro-shadows) — which correspond to *high spatial frequencies*.
    Conversely, *younger* skin is smoother, dominated by *low frequencies*.

    Aging path  (mode == 'aging')
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. Extract the high-frequency residual from the luminance channel via HPF:
           high = HPF(L)  ≈  L − LPF(L)
       This isolates the texture layer that aging adds.
    2. Amplify and add it back to the luminance:
           L' = L + α · high
       α is controlled by `intensity`, `detail`, and the selected HPF type.
    3. Overlay a synthetic wrinkle layer (procedural geometry drawn on a float
       canvas) — adds anatomically-placed crease geometry not present in HPF.
    4. Add soft depth shadows under eyes and around nasolabial folds.
    5. Apply CLAHE (local contrast enhancement) to simulate loss of skin
       elasticity; aged skin shows higher intra-region contrast variation.

    De-aging path  (mode == 'deaging')
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. Extract the smooth low-frequency component via LPF:
           smooth = LPF(L)
    2. Blend toward the smooth signal:
           L' = L · (1 − β) + smooth · β
       β is controlled by `intensity` and `detail`.
    3. Mix with bilateral filter output for edge-aware smoothing that preserves
       the eye / lip contour while softening the skin (gaussian blur would blur
       important features too).
    4. If `preserve_edges` is True, restore high-frequency detail at protected
       landmarks (eyes, nose, mouth) to prevent the face from looking blurry.

    Both paths
    ~~~~~~~~~~
    - All processing is performed in CIE L*a*b* colour space so that the
      luminance (L*) modifications do not introduce unintended hue or chroma
      shifts.  Only a subtle colour-tone adjustment (a*, b* channels) is added
      via _apply_lab_color_tone.
    - The skin mask gates all modifications so the background, eyes, teeth,
      and hair are not affected by the texture / smoothness edits.
    - Optional photon-noise grain is added in the aging path to simulate the
      appearance of skin micro-surface roughness on camera.
    """
    seed = _make_deterministic_seed(image, mode, strength, years, contrast, tone, noise, detail, hpf, lpf, preserve_edges, semantic_boost)
    rng = np.random.default_rng(seed) if deterministic else np.random.default_rng()
    masks = _build_age_region_masks(image, preserve_edges=preserve_edges)
    skin = masks["skinMask"].astype(np.float32)
    processing = masks["processingMask"].astype(np.float32)
    protected = masks["protectedMask"].astype(np.float32)

    # Work in L*a*b* to decouple luminance edits from colour
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_orig = lab[:, :, 0].copy()
    l_mod = l_orig.copy()
    intensity = _age_intensity(strength, years, semantic_boost)

    if mode == "aging":
        # ── Step 1: High-Pass Filter on luminance ────────────────────────
        # The HPF residual isolates micro-texture (edges, pores, fine lines).
        # Clipping to ±45 prevents extreme pixel spikes on high-contrast edges
        # from creating black halos when the residual is amplified.
        high = np.clip(_frequency_reconstruct(l_orig, hpf), -45.0, 45.0)

        # ── Step 2: Texture amplification ────────────────────────────────
        # hpf_gain accounts for the different transition-band steepness of each
        # filter type — Ideal HPF passes more high-freq energy than Gaussian,
        # so its gain is higher to produce a comparable visual strength at the
        # same `strength` slider value.
        hpf_gain = {"gaussian_hpf": 0.85, "butterworth_hpf": 1.08, "ideal_hpf": 1.34}.get(hpf, 1.0)
        detail_gain = 0.70 + detail * 0.95
        l_mod += processing * high * (0.78 + 1.58 * intensity) * hpf_gain * detail_gain

        # ── Step 3: Procedural wrinkle geometry ──────────────────────────
        # Draws anatomically-placed crease strokes on a float layer, then adds
        # them to luminance.  Strokes are heavily blurred (~min(h,w)/30 px) so
        # they appear as soft shadow-highlight pairs, not hard geometric lines.
        wrinkle_layer = _generate_wrinkle_layer(l_orig.shape, masks["faceBox"], rng, semantic_boost, masks.get("landmarks"))
        wrinkle_gain = (6.0 + 20.0 * intensity) * (0.58 + detail * 0.46)
        l_mod += processing * wrinkle_layer * wrinkle_gain

        # ── Step 4: Soft shadow layer ────────────────────────────────────
        # Adds subtle darkening under the eyes and along nasolabial folds to
        # simulate the depth shadows that become more pronounced with age.
        shadow_layer = _generate_age_shadow_layer(l_orig.shape, masks["faceBox"], masks.get("landmarks"), rng)
        l_mod += skin * shadow_layer * (6.0 + 24.0 * intensity)

        # ── Step 5: Local contrast enhancement (CLAHE) ───────────────────
        # CLAHE (Contrast Limited Adaptive Histogram Equalisation) operates on
        # small 8×8 tiles.  Higher clipLimit → more micro-contrast, simulating
        # the uneven skin surface texture of aged skin.
        clahe = cv2.createCLAHE(clipLimit=1.35 + 1.45 * intensity, tileGridSize=(8, 8))
        local_contrast = clahe.apply(np.clip(l_orig, 0, 255).astype(np.uint8)).astype(np.float32)
        l_mod = l_mod * (1.0 - processing * 0.13 * intensity) + local_contrast * (processing * 0.13 * intensity)
    else:
        # ── De-aging: Low-Pass Filter smoothing ──────────────────────────
        # The LPF retains only the low-frequency (smooth) component of L*.
        # Blending toward this signal selectively removes high-frequency texture
        # (wrinkles, pores) while preserving overall face shape and shading.
        low = _frequency_reconstruct(l_orig, lpf)

        # ── Edge-aware smoothing via bilateral filter ─────────────────────
        # The bilateral filter smooths luminance while preserving sharp edges
        # (eye contours, lip boundary).  sigmaColor / sigmaSpace are proportional
        # to `strength` so the user controls the smoothing degree.
        bilateral = cv2.bilateralFilter(
            np.clip(l_orig, 0, 255).astype(np.uint8),
            d=9,
            sigmaColor=34 + 54 * strength,
            sigmaSpace=34 + 44 * strength,
        ).astype(np.float32)

        # Blend FFT-LPF result with bilateral output: FFT handles global low-freq
        # shape; bilateral handles local edge preservation.
        smooth = low * 0.46 + bilateral * 0.54
        lpf_gain = {"gaussian_lpf": 0.78, "butterworth_lpf": 1.00, "ideal_lpf": 1.24}.get(lpf, 1.0)
        smooth_strength = _clamp(intensity * (0.56 + (1.0 - detail) * 0.30) * lpf_gain, 0.0, 0.88)
        l_mod = l_mod * (1.0 - processing * smooth_strength) + smooth * (processing * smooth_strength)

        if preserve_edges:
            # Re-inject HPF residual at protected landmarks (eyes, nose, mouth)
            # so that facial features stay sharp even after strong smoothing.
            protected_detail = np.clip(_frequency_reconstruct(l_orig, "gaussian_hpf"), -28.0, 28.0)
            l_mod += protected * protected_detail * (0.22 + 0.18 * detail)

    l_mod = _apply_skin_contrast(l_mod, skin, mode, intensity, contrast)
    lab[:, :, 0] = np.clip(l_mod, 0, 255)
    _apply_lab_color_tone(lab, skin, mode, intensity, tone)
    lab = np.clip(lab, 0, 255)
    processed = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

    if noise > 0.001:
        noise_scale = (18.0 if mode == "aging" else 5.5) * noise * (0.55 + 0.45 * intensity)
        grain = rng.normal(0.0, noise_scale, size=processed.shape).astype(np.float32)
        # noise_mask = processing (already excludes hair) intersected with inverse of
        # protected (excludes lips, eyes, teeth) — background is zero by construction.
        noise_mask = np.clip(processing * (1.0 - np.clip(protected, 0.0, 1.0)), 0.0, 1.0)
        processed += grain * noise_mask[:, :, None]

    blend = np.clip(skin * (0.70 + 0.30 * strength), 0.0, 1.0)
    out = image.astype(np.float32) * (1.0 - blend[:, :, None]) + processed * blend[:, :, None]

    if preserve_edges:
        # Restore original pixel values at sharp feature boundaries (eyes, nose,
        # mouth) using the landmark-based protect mask — not Canny edges, so that
        # only compact feature regions are restored and skin wrinkles remain.
        restore = protected * np.clip(0.40 + 0.20 * (1.0 - strength), 0.0, 0.60)
        out = out * (1.0 - restore[:, :, None]) + image.astype(np.float32) * restore[:, :, None]

    if mode == "aging" and intensity > 0.1:
        out = _apply_age_hair_graying(out, image, masks, intensity, rng)

    meta = {
        "maskMethod": masks["maskMethod"],
        "skinHighFrequencyEnergy": round(_masked_high_frequency_energy(np.clip(out, 0, 255).astype(np.uint8), masks["skinMask"]), 4),
    }
    if save_debug:
        def _mask_b64(m: np.ndarray) -> str:
            vis = np.clip(m * 255.0, 0, 255).astype(np.uint8)
            return _encode_base64_jpeg(cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR), quality=88)
        meta["debugMasks"] = {
            "skin":       _mask_b64(masks["skinMask"].astype(np.float32)),
            "processing": _mask_b64(processing),
            "protected":  _mask_b64(protected),
            "hair":       _mask_b64(hair_excl),
        }
    return np.clip(out, 0, 255).astype(np.uint8), meta


def _find_aging_onnx_model() -> Path | None:
    if not MODELS_DIR.is_dir():
        return None
    candidates = sorted(MODELS_DIR.glob("aging*.onnx"))
    return candidates[0] if candidates else None


def _try_onnx_age_transform(image: np.ndarray, model_path: Path) -> np.ndarray | None:
    try:
        net = cv2.dnn.readNetFromONNX(str(model_path))
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0 / 255.0, size=(256, 256), mean=(0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        pred = net.forward()
        if pred.ndim == 4:
            pred = pred[0]
        if pred.ndim == 3 and pred.shape[0] in {1, 3}:
            pred = np.transpose(pred, (1, 2, 0))
        if pred.ndim == 2:
            pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
        if pred.ndim != 3 or pred.shape[2] not in {1, 3}:
            return None
        pred = pred.astype(np.float32)
        if float(np.max(pred)) <= 2.0:
            pred *= 255.0
        if pred.shape[2] == 1:
            pred = cv2.cvtColor(pred[:, :, 0], cv2.COLOR_GRAY2BGR)
        else:
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        pred = cv2.resize(np.clip(pred, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
        return pred
    except Exception:
        return None


def _ai_age_pipeline(
    image: np.ndarray,
    mode: str,
    strength: float,
    years: int,
    contrast: float,
    tone: float,
    noise: float,
    detail: float,
    deterministic: bool,
    hpf: str,
    lpf: str,
    preserve_edges: bool,
) -> tuple[np.ndarray, str, dict]:
    model_path = _find_aging_onnx_model()
    if model_path is not None:
        onnx_result = _try_onnx_age_transform(image, model_path)
        if onnx_result is not None:
            masks = _build_age_region_masks(image, preserve_edges=preserve_edges)
            blend = np.clip(masks["skinMask"] * (0.55 + 0.35 * strength), 0.0, 1.0)
            result = image.astype(np.float32) * (1.0 - blend[:, :, None]) + onnx_result.astype(np.float32) * blend[:, :, None]
            return np.clip(result, 0, 255).astype(np.uint8), "onnx", {"maskMethod": masks["maskMethod"]}

    result, meta = _frequency_age_pipeline(
        image=image,
        mode=mode,
        strength=_clamp(strength * 1.10, 0.0, 1.0),
        years=years,
        contrast=contrast,
        tone=tone,
        noise=noise,
        detail=_clamp(detail + 0.14, 0.0, 1.0),
        deterministic=deterministic,
        hpf=hpf,
        lpf=lpf,
        preserve_edges=preserve_edges,
        semantic_boost=1.28,
    )
    return result, "fallback-demo", meta


def _thumbnail_b64(image: np.ndarray, max_side: int = 192) -> str:
    h, w = image.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale < 1.0:
        thumb = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    else:
        thumb = image
    return _encode_base64_jpeg(thumb, quality=82)


def _estimate_base_age(image: np.ndarray) -> int:
    masks = _build_age_region_masks(image, preserve_edges=False)
    texture = _masked_high_frequency_energy(image, masks["skinMask"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    skin_mean = float(np.sum(gray * masks["skinMask"]) / (np.sum(masks["skinMask"]) + 1e-6))
    estimate = 24.0 + min(14.0, texture / 95.0) + max(0.0, (138.0 - skin_mean) / 18.0)
    return int(round(_clamp(estimate, 18.0, 58.0)))


def _apparent_age_after(base_age: int, mode: str, years: int, strength: float) -> int:
    requested = abs(int(years))
    signed = requested if mode == "aging" else -requested
    return int(round(_clamp(base_age + signed * (0.38 + 0.62 * strength), 1.0, 99.0)))


def _detect_face_bbox(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Try progressively more lenient parameters to handle cartoon / low-contrast faces
    attempts = [
        dict(scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)),
        dict(scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)),
        dict(scaleFactor=1.03, minNeighbors=2, minSize=(30, 30)),
    ]
    for params in attempts:
        faces = cascade.detectMultiScale(gray, **params)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            return int(x), int(y), int(w), int(h), 'haar'

    # ── Skin-colour fallback (works on synthetic/cartoon faces) ──────
    # Convert to HSV and YCrCb; take union of typical skin ranges
    hsv   = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    mask_hsv   = cv2.inRange(hsv,   (0, 20, 60),  (25, 200, 255))
    mask_ycrcb = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 0.02 * image.shape[0] * image.shape[1]:
            x, y, w, h = cv2.boundingRect(largest)
            # Add 10% padding so we don't clip the hairline / chin
            pad_x = int(w * 0.10)
            pad_y = int(h * 0.08)
            x = max(0, x - pad_x)
            y = max(0, y - pad_y)
            w = min(image.shape[1] - x, w + 2 * pad_x)
            h = min(image.shape[0] - y, h + 2 * pad_y)
            return int(x), int(y), int(w), int(h), 'skin'

    return None


def _clip_point(x: float, y: float, width: int, height: int):
    return max(0, min(width - 1, int(round(x)))), max(0, min(height - 1, int(round(y))))


def _synthetic_68_landmarks_from_bbox(x: int, y: int, w: int, h: int, width: int, height: int):
    """Generate dlib-like 68 points from detected face box as a robust fallback."""
    cx = x + w * 0.5
    cy = y + h * 0.55
    rx = w * 0.45
    ry = h * 0.52

    points = []

    # Jawline: indices 0-16
    jaw_angles = np.linspace(np.pi * 0.08, np.pi * 0.92, 17)
    for a in jaw_angles:
        px = cx + rx * np.cos(a)
        py = cy + ry * np.sin(a)
        points.append(_clip_point(px, py, width, height))

    # Left eyebrow: 17-21
    for t in np.linspace(0, 1, 5):
        px = cx - rx * 0.62 + t * (rx * 0.42)
        py = cy - ry * 0.43 - np.sin(t * np.pi) * ry * 0.05
        points.append(_clip_point(px, py, width, height))

    # Right eyebrow: 22-26
    for t in np.linspace(0, 1, 5):
        px = cx + rx * 0.20 + t * (rx * 0.42)
        py = cy - ry * 0.43 - np.sin(t * np.pi) * ry * 0.05
        points.append(_clip_point(px, py, width, height))

    # Nose bridge: 27-30
    for t in np.linspace(0, 1, 4):
        px = cx
        py = cy - ry * 0.26 + t * (ry * 0.33)
        points.append(_clip_point(px, py, width, height))

    # Nose bottom: 31-35
    nose_offsets = [(-0.16, 0.12), (-0.08, 0.16), (0.0, 0.18), (0.08, 0.16), (0.16, 0.12)]
    for ox, oy in nose_offsets:
        points.append(_clip_point(cx + rx * ox, cy + ry * oy, width, height))

    # Left eye: 36-41
    left_eye_cx = cx - rx * 0.30
    left_eye_cy = cy - ry * 0.12
    for a in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        px = left_eye_cx + rx * 0.11 * np.cos(a)
        py = left_eye_cy + ry * 0.07 * np.sin(a)
        points.append(_clip_point(px, py, width, height))

    # Right eye: 42-47
    right_eye_cx = cx + rx * 0.30
    right_eye_cy = cy - ry * 0.12
    for a in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        px = right_eye_cx + rx * 0.11 * np.cos(a)
        py = right_eye_cy + ry * 0.07 * np.sin(a)
        points.append(_clip_point(px, py, width, height))

    # Outer mouth: 48-59
    mouth_cx = cx
    mouth_cy = cy + ry * 0.32
    for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        px = mouth_cx + rx * 0.24 * np.cos(a)
        py = mouth_cy + ry * 0.12 * np.sin(a)
        points.append(_clip_point(px, py, width, height))

    # Inner mouth: 60-67
    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        px = mouth_cx + rx * 0.15 * np.cos(a)
        py = mouth_cy + ry * 0.07 * np.sin(a)
        points.append(_clip_point(px, py, width, height))

    return points[:68]


# ═══════════════════════════════════════════════════════════════════════════════
# EXPRESSION / FACE WARPING  v2 – natural deformation engine
# ═══════════════════════════════════════════════════════════════════════════════

# ── Landmark group definitions for MediaPipe FaceMesh 468 points ────────────
_MP468_GROUPS: dict = {
    # Mouth corners
    "mouth_left":          [61],
    "mouth_right":         [291],
    # Upper outer lip arc
    "upper_lip":           [40, 37, 0, 267, 270, 409],
    # Lower outer lip arc
    "lower_lip":           [146, 91, 181, 84, 17, 314, 405, 321, 375],
    # Inner lip ring
    "inner_upper":         [78, 191, 80, 81, 82, 13, 312, 311, 310, 308],
    "inner_lower":         [78, 95, 88, 178, 87, 14, 317, 402, 318, 308],
    # Cheeks (rise with smile)
    "cheek_left":          [116, 123, 147, 213, 192],
    "cheek_right":         [345, 352, 376, 433, 411],
    # Lower eye border (very subtle crinkle with big smile)
    "lower_eye_left":      [158, 153, 144, 145],
    "lower_eye_right":     [385, 380, 373, 374],
    # Eyebrows (10 pts each to cover full arch)
    "brow_left":           [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "brow_right":          [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    # Forehead – weak secondary drift with brow raise
    "forehead_left":       [54, 103, 67, 109],
    "forehead_right":      [284, 332, 297, 338],
    # Upper eyelid – tiny secondary (anatomically follows brow)
    "upper_eyelid_left":   [159, 160, 161],
    "upper_eyelid_right":  [386, 387, 388],
    # Full jawline halves for slimming
    "jaw_left":            [234, 93, 132, 58, 172, 136, 150, 149, 176],
    "jaw_right":           [454, 323, 361, 288, 397, 365, 379, 378, 400],
    # Chin bottom row (subtle upward tuck with slimming)
    "chin":                [148, 152, 377],
    # Nose tip / face centre reference (kept stable)
    "nose_tip":            [1, 5, 4],
    "center":              [1],
}

# ── Landmark group definitions for synthetic 68-point fallback ───────────────
_MP68_GROUPS: dict = {
    "mouth_left":          [48],
    "mouth_right":         [54],
    "upper_lip":           [49, 50, 51, 52, 53],
    "lower_lip":           [55, 56, 57, 58, 59],
    "inner_upper":         [61, 62, 63, 64, 65],
    "inner_lower":         [67, 60, 66],
    "cheek_left":          [2, 3, 4],
    "cheek_right":         [12, 13, 14],
    "lower_eye_left":      [41, 40],
    "lower_eye_right":     [47, 46],
    "brow_left":           [17, 18, 19, 20, 21],
    "brow_right":          [22, 23, 24, 25, 26],
    "forehead_left":       [18, 19],
    "forehead_right":      [23, 24],
    "upper_eyelid_left":   [37, 38],
    "upper_eyelid_right":  [43, 44],
    "jaw_left":            [1, 2, 3, 4, 5, 6, 7],
    "jaw_right":           [9, 10, 11, 12, 13, 14, 15],
    "chin":                [8],
    "nose_tip":            [29, 30],
    "center":              [30],
}


def _extract_landmarks_for_warp(image: np.ndarray):
    """Return (src_pts [N,2] float32, groups dict) using 468-pt MP or synth-68 fallback."""
    h, w = image.shape[:2]

    if FACE_MESH is not None:
        mp_points = _get_mediapipe_points(image)
        if mp_points:
            pts = np.array([[float(p["x"]), float(p["y"])] for p in mp_points], dtype=np.float32)
            return pts, _MP468_GROUPS

    # Synthetic 68-point fallback
    det = _detect_face_bbox(image)
    if det is None:
        bx, by, bw, bh = int(w * 0.18), int(h * 0.12), int(w * 0.64), int(h * 0.76)
    else:
        bx, by, bw, bh, _ = det

    pts68 = _synthetic_68_landmarks_from_bbox(bx, by, bw, bh, w, h)
    pts = np.array([[float(x), float(y)] for x, y in pts68], dtype=np.float32)
    return pts, _MP68_GROUPS


def _get_68_pts_for_warp(image: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Return (pts68 [68,2] float32, method_str) for expression warping.
    Priority: MediaPipe 468→68 reduction > Dlib 68-point > synthetic fallback.
    Always returns exactly 68 landmarks.
    """
    h, w = image.shape[:2]

    # 1. MediaPipe (best accuracy)
    if FACE_MESH is not None:
        mp_points = _get_mediapipe_points(image)
        if len(mp_points) >= 468:
            reduced = _reduce_mediapipe_to_68(mp_points)
            if len(reduced) == 68:
                pts = np.array([[float(p["x"]), float(p["y"])] for p in reduced],
                               dtype=np.float32)
                return pts, "mediapipe"

    # 2. Dlib (if model file is present)
    dlib_pts = _get_dlib_landmarks(image)
    if len(dlib_pts) == 68:
        pts = np.array([[float(x), float(y)] for x, y in dlib_pts], dtype=np.float32)
        return pts, "dlib"

    # 3. Synthetic 68-point fallback from Haar/skin bbox
    det = _detect_face_bbox(image)
    if det is None:
        bx, by, bw, bh = int(w * 0.18), int(h * 0.12), int(w * 0.64), int(h * 0.76)
    else:
        bx, by, bw, bh, _ = det
    pts68 = _synthetic_68_landmarks_from_bbox(bx, by, bw, bh, w, h)
    pts = np.array([[float(x), float(y)] for x, y in pts68], dtype=np.float32)
    return pts, "fallback"


def _estimate_expression_sliders_from_68(pts68: np.ndarray, strength: float = 0.6) -> dict:
    """
    Estimate only expression-style controls from a 68-point reference face.

    This intentionally avoids transferring jaw, eye, nose, or full landmark
    identity offsets. Different people and poses should not be interpreted as
    expression, otherwise reference-only mode can wildly reshape the target.
    """
    if pts68 is None or len(pts68) < 68:
        return {"smile": 0.0, "brow": 0.0, "lip": 0.0, "slim": 0.0}

    pts = pts68.astype(np.float32)
    min_x = float(np.min(pts[:, 0]))
    max_x = float(np.max(pts[:, 0]))
    min_y = float(np.min(pts[:, 1]))
    max_y = float(np.max(pts[:, 1]))
    face_w = max(max_x - min_x, 1.0)
    face_h = max(max_y - min_y, 1.0)

    def point(i: int) -> np.ndarray:
        return pts[min(max(i, 0), len(pts) - 1)]

    mouth = pts[48:68]
    mouth_outer = pts[48:60]
    mouth_left = point(48)
    mouth_right = point(54)
    mouth_width = float(np.linalg.norm(mouth_right - mouth_left) / face_w)
    corner_y = float((mouth_left[1] + mouth_right[1]) * 0.5)
    mouth_mid_y = float(np.median(mouth_outer[:, 1]))
    corner_lift = (mouth_mid_y - corner_y) / face_h

    upper_lip_y = float(np.mean([point(i)[1] for i in (50, 51, 52, 61, 62, 63)]))
    lower_lip_y = float(np.mean([point(i)[1] for i in (56, 57, 58, 65, 66, 67)]))
    mouth_open = max(0.0, (lower_lip_y - upper_lip_y) / face_h)

    brow_y = float(np.mean(pts[17:27, 1]))
    eye_y = float(np.mean(pts[36:48, 1]))
    brow_gap = max(0.0, (eye_y - brow_y) / face_h)

    smile = _clamp(((corner_lift - 0.010) / 0.075) + max(0.0, mouth_open - 0.045) * 1.6, 0.0, 0.75)
    lip = _clamp((mouth_width - 0.34) / 0.22, 0.0, 0.45)
    brow = _clamp((brow_gap - 0.18) / 0.12, 0.0, 0.55)

    gain = _clamp(float(strength), 0.1, 1.0)
    params = {
        "smile": float(smile * gain),
        "brow": float(brow * gain),
        "lip": float(lip * gain),
        "slim": 0.0,
    }
    for key, value in list(params.items()):
        if abs(value) < 0.025:
            params[key] = 0.0
    return params


def _build_reference_expression_dst_68(
    ref_pts: np.ndarray,
    tgt_pts: np.ndarray,
    strength: float,
    target_w: int,
    target_h: int,
) -> tuple[np.ndarray, int, float, list[int]]:
    """
    Transfer only expression-bearing 68-point landmark regions.

    Full 68-point displacement copies identity, head pose, jaw width, and other
    person-specific structure. This helper keeps the target identity stable by
    applying reference-vs-neutral displacement only to brows, lips, and a tiny
    cheek/eyelid support set.
    """
    dst = tgt_pts.astype(np.float32).copy()
    ref = ref_pts.astype(np.float32)
    tgt = tgt_pts.astype(np.float32)

    ref_min = ref.min(axis=0)
    ref_max = ref.max(axis=0)
    ref_face_w = max(float(ref_max[0] - ref_min[0]), 1.0)
    ref_face_h = max(float(ref_max[1] - ref_min[1]), 1.0)

    tgt_min = tgt.min(axis=0)
    tgt_max = tgt.max(axis=0)
    tgt_face_w = max(float(tgt_max[0] - tgt_min[0]), 1.0)
    tgt_face_h = max(float(tgt_max[1] - tgt_min[1]), 1.0)
    target_face_size = max(tgt_face_w, tgt_face_h)

    det_ref = _detect_face_bbox_from_points(ref)
    neutral_ref = np.array(
        [[float(x), float(y)]
         for x, y in _synthetic_68_landmarks_from_bbox(
             det_ref[0], det_ref[1], det_ref[2], det_ref[3],
             int(max(ref_max[0] + 2, ref_face_w + ref_min[0] + 2)),
             int(max(ref_max[1] + 2, ref_face_h + ref_min[1] + 2)),
         )],
        dtype=np.float32,
    )

    strength = _clamp(float(strength), 0.1, 1.0)
    moved: set[int] = set()

    def apply_indices(indices: list[int], gain: float, cap_frac: float) -> None:
        cap = cap_frac * target_face_size
        for i in indices:
            if i >= len(ref) or i >= len(tgt) or i >= len(neutral_ref):
                continue
            disp_ref = ref[i] - neutral_ref[i]
            disp = np.array([
                (disp_ref[0] / ref_face_w) * tgt_face_w,
                (disp_ref[1] / ref_face_h) * tgt_face_h,
            ], dtype=np.float32) * strength * gain
            mag = float(np.linalg.norm(disp))
            if mag > cap and mag > 1e-6:
                disp *= cap / mag
            dst[i] = tgt[i] + disp
            moved.add(int(i))

    mouth_outer = list(range(48, 60))
    mouth_inner = list(range(60, 68))
    brows = list(range(17, 27))
    eyelid_support = [37, 38, 43, 44]
    cheek_support = [3, 4, 12, 13]

    apply_indices(mouth_outer, gain=1.10, cap_frac=0.18)
    apply_indices(mouth_inner, gain=1.25, cap_frac=0.20)
    apply_indices(brows, gain=0.75, cap_frac=0.09)
    apply_indices(eyelid_support, gain=0.22, cap_frac=0.035)
    apply_indices(cheek_support, gain=0.18, cap_frac=0.035)

    # Keep identity-critical anchors pinned even with expressive references.
    for i in list(range(27, 36)) + list(range(36, 48)) + list(range(0, 17)):
        if i not in cheek_support and i < len(dst):
            dst[i] = tgt[i]
            moved.discard(int(i))

    dst[:, 0] = np.clip(dst[:, 0], 1.0, float(target_w - 2))
    dst[:, 1] = np.clip(dst[:, 1], 1.0, float(target_h - 2))
    disp_mag = np.linalg.norm(dst - tgt, axis=1)
    moved_indices = [int(i) for i in sorted(moved) if disp_mag[i] > 0.25]
    moved_count = len(moved_indices)
    max_disp = float(np.max(disp_mag)) if moved_count else 0.0
    return dst, moved_count, max_disp, moved_indices


def _detect_face_bbox_from_points(points: np.ndarray) -> tuple[int, int, int, int]:
    min_xy = np.min(points, axis=0)
    max_xy = np.max(points, axis=0)
    x = int(np.floor(min_xy[0]))
    y = int(np.floor(min_xy[1]))
    w = int(np.ceil(max_xy[0] - min_xy[0]))
    h = int(np.ceil(max_xy[1] - min_xy[1]))
    return max(0, x), max(0, y), max(1, w), max(1, h)


def _build_dst_68_direct(
    src68: np.ndarray,
    smile: float,
    brow: float,
    lip: float,
    slim: float,
) -> tuple[np.ndarray, int, float, list]:
    """
    Build displaced 68-pt destination landmarks using *direct* per-point
    rules.  Only the explicitly listed points are moved; everything else is
    left unchanged (no Gaussian propagation, no falloff).

    68-point index conventions (dlib / standard):
      0-16  : jawline       (slim)
      17-21 : left brow     (brow)
      22-26 : right brow    (brow)
      27-35 : nose bridge + tip
      36-41 : left eye
      42-47 : right eye
      48    : left mouth corner  (smile x, lip x)
      54    : right mouth corner (smile x, lip x)
      48-67 : full mouth ring
    """
    dst = src68.copy()
    h_dim, w_dim = float(np.max(src68[:, 1])), float(np.max(src68[:, 0]))

    min_x = float(np.min(src68[:, 0]))
    max_x = float(np.max(src68[:, 0]))
    min_y = float(np.min(src68[:, 1]))
    max_y = float(np.max(src68[:, 1]))
    face_w = max(max_x - min_x, 1.0)
    face_h = max(max_y - min_y, 1.0)

    # Nose bridge centre — used as slim reference
    # Points 27-30 = nose bridge in dlib 68; take x mean as center_x.
    bridge_idxs = [i for i in range(27, 31) if i < len(src68)]
    center_x = float(np.mean(src68[bridge_idxs, 0])) if bridge_idxs else (min_x + face_w / 2)

    moved_mask = np.zeros(len(src68), dtype=bool)

    # ── SMILE ────────────────────────────────────────────────────────────
    # Corners 48 (left) + 54 (right) get full displacement.
    # Adjacent lip ring points get proportional motion to prevent mesh tearing.
    if abs(smile) > 1e-4 and len(src68) > 54:
        dx_c = 0.050 * face_w * smile   # horizontal: corners move outward
        dy_c = 0.060 * face_h * smile   # vertical:   corners move upward

        dst[48, 0] -= dx_c
        dst[48, 1] -= dy_c
        dst[54, 0] += dx_c
        dst[54, 1] -= dy_c
        moved_mask[[48, 54]] = True

        # Left neighbours: 45% close, 20% far
        for i, w in [(49, 0.45), (59, 0.45), (60, 0.45),
                     (50, 0.20), (58, 0.20), (61, 0.20), (67, 0.20)]:
            if i < len(src68):
                dst[i, 0] -= dx_c * w
                dst[i, 1] -= dy_c * w
                moved_mask[i] = True
        # Right neighbours: 45% close, 20% far
        for i, w in [(55, 0.45), (65, 0.45),
                     (56, 0.20), (64, 0.20), (62, 0.20)]:
            if i < len(src68):
                dst[i, 0] += dx_c * w
                dst[i, 1] -= dy_c * w
                moved_mask[i] = True

    # ── BROW RAISE ───────────────────────────────────────────────────────
    # Move brow points 17-26 upward. Upper eyelid points follow at 40%/20%
    # to prevent unnatural skin stretching between brow and eye.
    if abs(brow) > 1e-4 and len(src68) > 46:
        brow_dy = 0.090 * face_h * brow
        brow_idxs = list(range(17, 27))
        for i in brow_idxs:
            dst[i, 1] -= brow_dy
        moved_mask[brow_idxs] = True
        # Upper eyelid follow-through: 40% close, 20% far
        for i, w in [(37, 0.40), (38, 0.40), (43, 0.40), (44, 0.40)]:
            if i < len(src68):
                dst[i, 1] -= brow_dy * w
                moved_mask[i] = True
        for i, w in [(36, 0.20), (39, 0.20), (42, 0.20), (45, 0.20)]:
            if i < len(src68):
                dst[i, 1] -= brow_dy * w
                moved_mask[i] = True

    # ── LIP WIDENING ─────────────────────────────────────────────────────
    # Corners 48 / 54 move outward; same neighbour distribution as smile,
    # horizontal axis only.
    if abs(lip) > 1e-4 and len(src68) > 54:
        dx_l = 0.080 * face_w * lip   # corners outward
        dy_l = 0.030 * face_h * lip   # slight vertical separation (upper up, lower down)
        dst[48, 0] -= dx_l
        dst[54, 0] += dx_l
        moved_mask[[48, 54]] = True
        # Left-side neighbours — horizontal spread
        for i, w in [(49, 0.50), (59, 0.50), (60, 0.50),
                     (50, 0.25), (58, 0.25), (61, 0.25), (67, 0.25)]:
            if i < len(src68):
                dst[i, 0] -= dx_l * w
                moved_mask[i] = True
        # Right-side neighbours — horizontal spread
        for i, w in [(55, 0.50), (65, 0.50),
                     (56, 0.25), (64, 0.25), (62, 0.25)]:
            if i < len(src68):
                dst[i, 0] += dx_l * w
                moved_mask[i] = True
        # Vertical separation: upper lip up, lower lip down
        for i in [48, 49, 50, 54, 55, 56]:   # upper lip outline
            if i < len(src68):
                dst[i, 1] -= dy_l
        for i in [58, 59, 60, 61, 62, 64, 65, 67]:  # lower lip outline
            if i < len(src68):
                dst[i, 1] += dy_l

    # ── FACE SLIM ────────────────────────────────────────────────────────
    # Move ONLY jawline points 0-16 toward center_x.
    if abs(slim) > 1e-4 and len(src68) > 16:
        jaw_idxs = list(range(17))
        for i in jaw_idxs:
            dst[i, 0] += (center_x - src68[i, 0]) * 0.130 * slim
        moved_mask[jaw_idxs] = True

    # ── Hard displacement clamp: <= 12% of face_size ─────────────────────
    face_size = max(face_w, face_h)
    max_allowed = 0.12 * face_size
    disp_raw = dst - src68
    norms = np.linalg.norm(disp_raw, axis=1, keepdims=True)
    overflow = (norms > max_allowed) & (norms > 0)
    if np.any(overflow):
        scale = np.where(overflow, max_allowed / norms, 1.0)
        dst = src68 + disp_raw * scale

    # Measure how many points actually moved and max displacement
    disp = np.linalg.norm(dst - src68, axis=1)
    moved_count = int(np.sum(disp > 0.3))
    max_disp = float(np.max(disp)) if moved_count > 0 else 0.0
    moved_indices = [int(i) for i in np.where(moved_mask)[0]]

    return dst, moved_count, max_disp, moved_indices


# ── Muscle-based dense flow-field engine (v3) ────────────────────────────────

def _group_centroid(src_pts: np.ndarray, groups: dict, *keys) -> np.ndarray | None:
    """Return the mean position of all landmarks belonging to the given group keys."""
    idxs: list = []
    for k in keys:
        idxs.extend(groups.get(k, []))
    valid = [src_pts[i] for i in idxs if 0 <= i < len(src_pts)]
    if not valid:
        return None
    return np.mean(valid, axis=0).astype(np.float32)


def _muscle_add_force(
    flow: np.ndarray,
    cx: float, cy: float,
    rx: float, ry: float,
    angle_deg: float,
    fdx: float, fdy: float,
    strength: float,
) -> None:
    """
    Accumulate an anisotropic Gaussian force into the inverse flow field.

    flow[y, x, 0/1] is the *inverse* displacement: output pixel (x, y) samples
    source at (x + flow_x, y + flow_y).  A forward push (fdx, fdy) is stored as
    its negative so that cv2.remap moves tissue in the correct direction.
    """
    h, w = flow.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    dx_g = xx.astype(np.float32) - cx
    dy_g = yy.astype(np.float32) - cy
    rad = float(np.deg2rad(angle_deg))
    cos_a, sin_a = float(np.cos(rad)), float(np.sin(rad))
    lx =  dx_g * cos_a + dy_g * sin_a
    ly = -dx_g * sin_a + dy_g * cos_a
    weight = np.exp(-(lx ** 2 / (rx ** 2 + 1e-6) + ly ** 2 / (ry ** 2 + 1e-6))) * strength
    flow[:, :, 0] -= weight * fdx
    flow[:, :, 1] -= weight * fdy


def _build_muscle_flow(
    h: int,
    w: int,
    src_pts: np.ndarray,
    groups: dict,
    smile: float,
    brow: float,
    lip: float,
    slim: float,
) -> np.ndarray:
    """
    Isolated per-slider displacement field [h, w, 2].

    Design rules:
    - Each slider ONLY displaces its own landmark group.  Zero coupling between
      regions (no cheek lift from smile, no eyelid from brow, etc.).
    - Forces use tight Gaussian radii (≤5% face width) so neighbouring regions
      are not affected — effectively direct landmark displacement.
    - Sign convention: flow -= weight * fdx/fdy  (inverse warp for cv2.remap).
      Forward push → store as negative fdx/fdy so the pixel samples the correct
      source location.
    """
    flow = np.zeros((h, w, 2), dtype=np.float32)
    face_w = max(float(np.max(src_pts[:, 0]) - np.min(src_pts[:, 0])), 1.0)
    face_h = max(float(np.max(src_pts[:, 1]) - np.min(src_pts[:, 1])), 1.0)
    _cx = _group_centroid(src_pts, groups, "center")
    cx_pt = _cx if _cx is not None else src_pts.mean(axis=0)

    def ctr(*keys):
        return _group_centroid(src_pts, groups, *keys)

    def force(pt, rx, ry, fdx, fdy, s=1.0, angle=0.0):
        if pt is None:
            return
        _muscle_add_force(flow, float(pt[0]), float(pt[1]),
                          float(rx), float(ry), float(angle),
                          float(fdx), float(fdy), float(s))

    def curve(v: float) -> float:
        """Power-1.5 curve — used only for sliders that need it (slim)."""
        return float(np.sign(v) * abs(v) ** 1.5)

    # Moderate local radii — visible effect while keeping each slider
    # confined to its own anatomical region.
    # ~8-10% face_w ≈ 20-30 px on a typical crop (enough to see, too small to bleed).
    R_C  = face_w * 0.085   # corner radius (mouth corners, brow)
    R_L  = face_w * 0.095   # lip body centroid (slightly wider)
    R_CV = face_h * 0.075   # vertical complement for corners
    R_LV = face_h * 0.065   # vertical complement for lip body

    # ── SMILE ──────────────────────────────────────────────────────────────
    # Moves ONLY mouth corners (outward + upward) + lips (vertical only).
    # No cheeks, no eyes, no nose, no jaw — zero cross-region coupling.
    #   corner dx = ±0.030 × face_w × smile
    #   corner dy = -0.040 × face_h × smile  (negative = upward in image coords)
    #   lip    dy = ±0.010 × face_h × smile
    if abs(smile) > 0.01:
        mc_l = ctr("mouth_left")
        mc_r = ctr("mouth_right")
        ul   = ctr("upper_lip")
        ll   = ctr("lower_lip")

        cdx = 0.030 * face_w * smile
        cdy = 0.040 * face_h * smile
        ldy = min(abs(smile), 1.0) * 0.010 * face_h

        # flow -= weight * fdy; for tissue to rise, flow_y > 0, so fdy < 0.
        force(mc_l, R_C, R_CV, -cdx, -cdy, 1.0)   # left corner:  left  + up
        force(mc_r, R_C, R_CV,  cdx, -cdy, 1.0)   # right corner: right + up
        force(ul,   R_L, R_LV,  0.0, -ldy, 0.60)  # upper lip:    up only
        force(ll,   R_L, R_LV,  0.0,  ldy, 0.60)  # lower lip:    down only

    # ── BROW RAISE ─────────────────────────────────────────────────────────
    # Moves ONLY the brow landmark centroids — no forehead, no eyelid.
    if abs(brow) > 0.01:
        bl   = ctr("brow_left")
        br_r = ctr("brow_right")
        bdy  = 0.045 * face_h * brow
        force(bl,   R_C, R_CV, 0.0, -bdy, 1.0)
        force(br_r, R_C, R_CV, 0.0, -bdy, 1.0)

    # ── LIP WIDENING ───────────────────────────────────────────────────────
    # Mouth corners horizontal only — no vertical, no cheek/nose side effects.
    if abs(lip) > 0.01:
        mc_l = ctr("mouth_left")
        mc_r = ctr("mouth_right")
        ldx  = 0.032 * face_w * lip
        force(mc_l, R_C, R_CV, -ldx, 0.0, 1.0)
        force(mc_r, R_C, R_CV,  ldx, 0.0, 1.0)

    # ── FACE SLIMMING (masseter + jaw geometry) ────────────────────────────
    # Jaw requires a wider radius than the other sliders because the masseter
    # covers a larger anatomical area.  Force is centered 28% inward from the
    # jaw border to prevent background contamination at the skin edge.
    if abs(slim) > 0.01:
        sl      = curve(slim)
        jaw_l   = ctr("jaw_left")
        jaw_r   = ctr("jaw_right")
        chin    = ctr("chin")
        face_cx = float(cx_pt[0])
        face_cy = float(cx_pt[1])
        if jaw_l is not None:
            inward = float(np.clip((face_cx - jaw_l[0]) * 0.20 * sl, -face_w * 0.10, face_w * 0.10))
            jaw_l_inner = jaw_l + (np.array([face_cx, face_cy], np.float32) - jaw_l) * 0.28
            force(jaw_l_inner, face_w * 0.13, face_h * 0.22, inward, 0.0, 1.0)
        if jaw_r is not None:
            inward = float(np.clip((face_cx - jaw_r[0]) * 0.20 * sl, -face_w * 0.10, face_w * 0.10))
            jaw_r_inner = jaw_r + (np.array([face_cx, face_cy], np.float32) - jaw_r) * 0.28
            force(jaw_r_inner, face_w * 0.13, face_h * 0.22, inward, 0.0, 1.0)
        force(chin, face_w * 0.20, face_h * 0.09, 0.0, -sl * face_h * 0.011, 0.70)

    return flow


def _lock_identity_regions(
    flow: np.ndarray, src_pts: np.ndarray, groups: dict,
) -> np.ndarray:
    """
    Suppress flow near identity-critical anchors (nose tip, eye centres) so
    they remain perfectly stable regardless of expression slider values.
    """
    h, w = flow.shape[:2]
    face_w = max(float(np.max(src_pts[:, 0]) - np.min(src_pts[:, 0])), 1.0)
    yy, xx = np.ogrid[:h, :w]
    lock_mask = np.zeros((h, w), dtype=np.float32)
    lock_specs = [
        (_group_centroid(src_pts, groups, "nose_tip", "center"), face_w * 0.065),
        (_group_centroid(src_pts, groups, "lower_eye_left"),     face_w * 0.046),
        (_group_centroid(src_pts, groups, "lower_eye_right"),    face_w * 0.046),
    ]
    for pt, radius in lock_specs:
        if pt is None:
            continue
        d2 = (xx.astype(np.float32) - float(pt[0])) ** 2 + \
             (yy.astype(np.float32) - float(pt[1])) ** 2
        lock_mask = np.maximum(lock_mask, np.exp(-d2 / (radius ** 2 + 1e-6)))
    lock_mask = np.clip(lock_mask, 0.0, 1.0)
    flow[:, :, 0] *= 1.0 - lock_mask
    flow[:, :, 1] *= 1.0 - lock_mask
    return flow


def _lock_slimming_identity_regions(
    flow: np.ndarray, src_pts: np.ndarray, groups: dict,
) -> np.ndarray:
    """
    Keep expression features stable during face slimming.

    Slimming should reshape the outer lower-face contour.  The mouth, nose and
    eyes are identity/expression features, so they get a stronger lock than the
    general expression-flow lock above.
    """
    h, w = flow.shape[:2]
    face_w = max(float(np.max(src_pts[:, 0]) - np.min(src_pts[:, 0])), 1.0)
    yy, xx = np.ogrid[:h, :w]
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)
    lock_mask = np.zeros((h, w), dtype=np.float32)

    lock_specs = [
        (_group_centroid(src_pts, groups, "nose_tip", "center"), face_w * 0.090, 1.00),
        (_group_centroid(src_pts, groups, "lower_eye_left"),     face_w * 0.060, 0.95),
        (_group_centroid(src_pts, groups, "lower_eye_right"),    face_w * 0.060, 0.95),
        (_group_centroid(src_pts, groups, "mouth_left"),         face_w * 0.105, 1.00),
        (_group_centroid(src_pts, groups, "mouth_right"),        face_w * 0.105, 1.00),
        (_group_centroid(src_pts, groups, "upper_lip", "inner_upper"), face_w * 0.135, 1.00),
        (_group_centroid(src_pts, groups, "lower_lip", "inner_lower"), face_w * 0.135, 1.00),
    ]

    for pt, radius, strength in lock_specs:
        if pt is None:
            continue
        d2 = (xx - float(pt[0])) ** 2 + (yy - float(pt[1])) ** 2
        lock_mask = np.maximum(
            lock_mask,
            np.exp(-d2 / (radius ** 2 + 1e-6)) * float(strength),
        )

    lock_mask = np.clip(lock_mask, 0.0, 1.0)
    flow[:, :, 0] *= 1.0 - lock_mask
    flow[:, :, 1] *= 1.0 - lock_mask
    return flow


def _build_face_slimming_flow(
    h: int,
    w: int,
    src_pts: np.ndarray,
    groups: dict,
    slim: float,
) -> np.ndarray:
    """
    Dense inverse-remap field for natural face slimming.

    Instead of a single pull near the mouth, this applies several small,
    horizontally inward forces along both jaw/masseter contours.  The strongest
    pull is around the lower cheek and jaw angle; chin movement is subtle.
    """
    flow = np.zeros((h, w, 2), dtype=np.float32)
    if abs(slim) < 1e-4:
        return flow

    min_x = float(np.min(src_pts[:, 0]))
    max_x = float(np.max(src_pts[:, 0]))
    min_y = float(np.min(src_pts[:, 1]))
    max_y = float(np.max(src_pts[:, 1]))
    face_w = max(max_x - min_x, 1.0)
    face_h = max(max_y - min_y, 1.0)
    center_pt = _group_centroid(src_pts, groups, "center", "nose_tip")
    face_cx = float(center_pt[0]) if center_pt is not None else (min_x + face_w * 0.5)

    sl = float(np.sign(slim) * abs(slim) ** 1.15)
    max_pull = face_w * 0.120

    def add_jaw_forces(key: str):
        idxs = [i for i in groups.get(key, []) if 0 <= i < len(src_pts)]
        for i in idxs:
            p = src_pts[i].astype(np.float32)
            y_norm = float(np.clip((p[1] - min_y) / face_h, 0.0, 1.0))
            x_dist = abs(float(face_cx - p[0])) / face_w

            # Upper cheek gets a lighter pull; lower cheek/jaw angle carries
            # the visible slimming.  Inner points near the mouth contribute less.
            vertical_strength = float(np.clip((y_norm - 0.34) / 0.46, 0.0, 1.0))
            outer_strength = float(np.clip((x_dist - 0.13) / 0.22, 0.0, 1.0))
            strength = 0.34 + 0.66 * vertical_strength
            strength *= 0.45 + 0.55 * outer_strength
            if strength <= 0.05:
                continue

            raw_pull = (face_cx - float(p[0])) * 0.205 * sl * strength
            pull_x = float(np.clip(raw_pull, -max_pull, max_pull))
            if abs(pull_x) < 0.15:
                continue

            # Keep the force on the contour/lower cheek rather than drifting
            # toward the mouth centre.
            anchor = p.copy()
            anchor[0] += pull_x * 0.10
            rx = face_w * (0.085 + 0.040 * vertical_strength)
            ry = face_h * (0.100 + 0.065 * vertical_strength)
            _muscle_add_force(flow, float(anchor[0]), float(anchor[1]), rx, ry, 0.0, pull_x, 0.0, 1.0)

    add_jaw_forces("jaw_left")
    add_jaw_forces("jaw_right")

    for cheek_key in ("cheek_left", "cheek_right"):
        cheek = _group_centroid(src_pts, groups, cheek_key)
        if cheek is None:
            continue
        y_norm = float(np.clip((cheek[1] - min_y) / face_h, 0.0, 1.0))
        if y_norm < 0.34:
            continue
        cheek_pull = float(np.clip((face_cx - float(cheek[0])) * 0.085 * sl, -max_pull * 0.55, max_pull * 0.55))
        if abs(cheek_pull) >= 0.15:
            _muscle_add_force(
                flow,
                float(cheek[0] + cheek_pull * 0.10),
                float(cheek[1] + face_h * 0.045),
                face_w * 0.115,
                face_h * 0.140,
                0.0,
                cheek_pull,
                0.0,
                0.80,
            )

    chin = _group_centroid(src_pts, groups, "chin")
    if chin is not None:
        _muscle_add_force(
            flow,
            float(chin[0]),
            float(chin[1]),
            face_w * 0.16,
            face_h * 0.07,
            0.0,
            0.0,
            -sl * face_h * 0.006,
            0.45,
        )

    return _lock_slimming_identity_regions(flow, src_pts, groups)


def _apply_slimming_flow(
    image: np.ndarray,
    src_pts: np.ndarray,
    groups: dict,
    slim: float,
) -> np.ndarray:
    """
    Apply jaw slimming as a dense local inverse-remap field.

    Moving the jaw with Delaunay triangles creates skinny triangles between the
    inward jaw and the original face hull; on real photos this can show up as a
    bitten cheek or coloured stripe.  A smooth remap field keeps the jaw/cheek
    tissue coherent while naturally revealing the unchanged background.
    """
    if abs(slim) < 1e-4:
        return image

    h, w = image.shape[:2]
    flow = _build_face_slimming_flow(h, w, src_pts, groups, slim)

    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    return cv2.remap(
        image,
        xx + flow[:, :, 0],
        yy + flow[:, :, 1],
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Grid overlay (drawn on input BEFORE warp so TPS bends the lines)
# ──────────────────────────────────────────────────────────────────────────────
def _draw_grid(img: np.ndarray, cols: int = 20, rows: int = 20, alpha: float = 0.42) -> np.ndarray:
    """Return a copy of *img* with a semi-transparent white grid burned in.

    Drawing the grid BEFORE the TPS/Delaunay remap lets the warp bend the
    grid lines naturally, making the non-linear deformation visible.
    """
    h, w = img.shape[:2]
    overlay = img.copy()
    white = (255, 255, 255)
    for c in range(cols + 1):
        x = int(round(c * w / cols))
        cv2.line(overlay, (x, 0), (x, h - 1), white, 1, cv2.LINE_AA)
    for r in range(rows + 1):
        y = int(round(r * h / rows))
        cv2.line(overlay, (0, y), (w - 1, y), white, 1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)


def _apply_double_chin_flow(
    image: np.ndarray,
    src_pts: np.ndarray,
    groups: dict,
    double_chin: float,
) -> np.ndarray:
    """
    Simulate submental fat (double chin) as a smooth dense inverse-remap field.

    Anchor strategy
    ---------------
    Vertical push  (Y ↓): MP-468 indices 152, 175, 199 — bottom chin points.
    Lateral left   (X ←): MP-468 indices 172, 132 — left jaw lower rim.
    Lateral right  (X →): MP-468 indices 397, 361 — right jaw lower rim.
    Neck extension       : synthetic point 20 % of face_h below the chin
                           centroid so the submental volume spreads into the
                           upper neck region naturally.

    All anchors fall back to group centroids when running on the 68-pt
    fallback model (indices out of range).

    Design features
    ---------------
    * Per-anchor Gaussian kernels, max-blended → no inter-blob valley.
    * Separate flow_y (vertical) and flow_x (lateral) fields built from their
      own anchor sets, then summed.
    * dc = double_chin^0.65 — non-linear: subtle at low values, capped high.
    * Sigmoid identity lock above the mouth line (band = 16 % face_h).
    * GaussianBlur on the final flow fields before cv2.remap.
    """
    if abs(double_chin) < 1e-4:
        return image

    h, w      = image.shape[:2]
    n_pts     = len(src_pts)
    face_w    = max(float(np.max(src_pts[:, 0]) - np.min(src_pts[:, 0])), 1.0)
    face_h    = max(float(np.max(src_pts[:, 1]) - np.min(src_pts[:, 1])), 1.0)
    dc        = float(double_chin) ** 0.65   # non-linear strength

    xx, yy = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    # ── helpers ──────────────────────────────────────────────────────────
    def _g2d(px, py, rx, ry):
        return np.exp(
            -((xx - px) / (rx + 1e-4)) ** 2
            -((yy - py) / (ry + 1e-4)) ** 2
        ).astype(np.float32)

    def _safe_pts(indices, fallback_key):
        """Return list of (x, y) from specific MP-468 indices; fall back to
        group centroid when indices exceed src_pts length (68-pt model)."""
        result = [src_pts[i] for i in indices if i < n_pts]
        if not result:
            idxs = [i for i in groups.get(fallback_key, []) if i < n_pts]
            if idxs:
                result = [src_pts[idxs].mean(axis=0)]
        return result

    # ── anchor point sets ─────────────────────────────────────────────────
    # Vertical: 152 (chin bottom), 175 (lower mandible), 199 (very bottom)
    vert_pts  = _safe_pts([152, 175, 199], "chin")
    # Lateral left:  172 (jaw lower-left), 132 (jaw mid-left)
    lat_l_pts = _safe_pts([172, 132], "jaw_left")
    # Lateral right: 397 (jaw lower-right), 361 (jaw mid-right)
    lat_r_pts = _safe_pts([397, 361], "jaw_right")

    if not vert_pts:
        return image

    # ── synthetic submental volume centre (neck extension) ────────────────
    chin_cx = float(np.mean([p[0] for p in vert_pts]))
    chin_cy = float(np.mean([p[1] for p in vert_pts]))
    # 20 % below the chin cluster → upper neck / submental space
    sub_cx  = chin_cx
    sub_cy  = chin_cy + face_h * 0.20

    # ── VERTICAL FLOW (flow_y) — inverse remap: negative value → push down ─
    env_vert = np.zeros((h, w), dtype=np.float32)
    for p in vert_pts:
        # Wide horizontal radius so push spans full chin width
        env_vert = np.maximum(
            env_vert,
            _g2d(float(p[0]), float(p[1]), face_w * 0.70, face_h * 0.40),
        )
    # Neck extension blob — slightly lighter weight, wider spread
    env_vert = np.maximum(
        env_vert,
        _g2d(sub_cx, sub_cy, face_w * 0.65, face_h * 0.38) * 0.65,
    )
    env_vert = np.clip(env_vert, 0.0, 1.0)
    flow_y   = -env_vert * dc * face_h * 0.080

    # ── LATERAL FLOW (flow_x) — separate envelopes for left / right ───────
    env_lat_l = np.zeros((h, w), dtype=np.float32)
    for p in lat_l_pts:
        env_lat_l = np.maximum(
            env_lat_l,
            _g2d(float(p[0]), float(p[1]), face_w * 0.42, face_h * 0.34),
        )
    env_lat_r = np.zeros((h, w), dtype=np.float32)
    for p in lat_r_pts:
        env_lat_r = np.maximum(
            env_lat_r,
            _g2d(float(p[0]), float(p[1]), face_w * 0.42, face_h * 0.34),
        )

    dx_max  = dc * face_w * 0.055
    # left side pushes further left (negative x), right side further right
    flow_x  = (-env_lat_l + env_lat_r) * dx_max

    # ── SIGMOID IDENTITY LOCK — fade to zero above the mouth line ────────
    mouth_idxs = [i for i in groups.get("inner_upper", []) if i < n_pts]
    if mouth_idxs:
        mouth_y = float(src_pts[mouth_idxs, 1].mean())
    else:
        mouth_y = float(np.min(src_pts[:, 1])) + face_h * 0.55

    band    = face_h * 0.22   # wider transition = softer jaw boundary
    sig_arg = np.clip((yy - mouth_y) / (band + 1e-4), -7.0, 7.0)
    lock    = (1.0 / (1.0 + np.exp(-sig_arg))).astype(np.float32)

    flow_x *= lock
    flow_y *= lock

    # ── GAUSSIAN BLUR on flow fields — removes residual discontinuities ───
    ksize      = max(int(face_h * 0.13) | 1, 11)   # odd, ~13 % face height
    sigma_blur = ksize * 0.50
    flow_x = cv2.GaussianBlur(flow_x, (ksize, ksize), sigma_blur)
    flow_y = cv2.GaussianBlur(flow_y, (ksize, ksize), sigma_blur)

    return cv2.remap(
        image,
        xx + flow_x,
        yy + flow_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


# ── Legacy point-displacement helpers (kept for backward compatibility) ────────


def _apply_face_fullness_flow(
    image: np.ndarray,
    src_pts: np.ndarray,
    groups: dict,
    face_fullness: float,
) -> np.ndarray:
    """
    Simulate overall facial fullness ("chubby / fat face") as a dense
    inverse-remap field.  Pushes cheek and lower-face landmarks radially
    outward from the face centroid.

    Anchor landmarks (MP-468)
    -------------------------
    Cheek left:   50, 123, 117, 118, 101
    Cheek right: 280, 352, 347, 346, 330
    These fall back to jaw_left / jaw_right group centroids on the 68-pt model.

    Algorithm
    ---------
    1. Build per-anchor Gaussian kernels; max-blend into one "fullness envelope".
    2. For each pixel compute its outward unit vector from the face centroid.
    3. Multiply envelope × dc × max_push × unit_vector → forward displacement.
    4. Apply as INVERSE remap (subtract displacement from sample coords).
    5. Sigmoid identity lock above the nose tip so eyes/brows/forehead stay still.
    6. GaussianBlur flow field for smooth transitions.
    """
    if abs(face_fullness) < 1e-4:
        return image

    h, w   = image.shape[:2]
    n_pts  = len(src_pts)
    face_w = max(float(np.max(src_pts[:, 0]) - np.min(src_pts[:, 0])), 1.0)
    face_h = max(float(np.max(src_pts[:, 1]) - np.min(src_pts[:, 1])), 1.0)

    # Non-linear: more visible at low values than the chin slider
    dc = float(face_fullness) ** 0.60

    xx, yy = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    def _g2d(px, py, rx, ry):
        return np.exp(
            -((xx - px) / (rx + 1e-4)) ** 2
            -((yy - py) / (ry + 1e-4)) ** 2
        ).astype(np.float32)

    def _safe_pts(indices, fallback_key):
        result = [src_pts[i] for i in indices if i < n_pts]
        if not result:
            idxs = [i for i in groups.get(fallback_key, []) if i < n_pts]
            if idxs:
                result = [src_pts[idxs].mean(axis=0)]
        return result

    # ── Cheek anchor sets ─────────────────────────────────────────────────
    cheek_l_pts = _safe_pts([50, 123, 117, 118, 101], "jaw_left")
    cheek_r_pts = _safe_pts([280, 352, 347, 346, 330], "jaw_right")
    all_cheek   = cheek_l_pts + cheek_r_pts
    if not all_cheek:
        return image

    # Face centroid (used as the radial expansion origin)
    face_cx = float(np.mean(src_pts[:, 0]))
    face_cy = float(np.mean(src_pts[:, 1]))

    # ── Fullness envelope (max-blend all cheek anchors) ───────────────────
    # Horizontal radius wide enough to cover full cheek + some neck; vertical
    # radius covers from cheekbone down to jowl.
    env = np.zeros((h, w), dtype=np.float32)
    for p in all_cheek:
        env = np.maximum(
            env,
            _g2d(float(p[0]), float(p[1]), face_w * 0.52, face_h * 0.42),
        )
    # Additional wide base so the lower face (jowl area) also fills in
    chin_pts = _safe_pts([152, 175, 199], "chin")
    env_chin = np.zeros((h, w), dtype=np.float32)  # kept separate for volume coupling
    for p in chin_pts:
        blob = _g2d(float(p[0]), float(p[1]), face_w * 0.65, face_h * 0.38) * 0.60
        env      = np.maximum(env,      blob)
        env_chin = np.maximum(env_chin, blob)
    env = np.clip(env, 0.0, 1.0)
    env_chin = np.clip(env_chin, 0.0, 1.0)

    # ── Radial outward displacement ───────────────────────────────────────
    # Formula inspired by x = x * (1 + scale/200): push proportional to
    # distance from centroid, capped so distant pixels don't fly away.
    dx_raw  = xx - face_cx
    dy_raw  = yy - face_cy
    dist    = np.sqrt(dx_raw ** 2 + dy_raw ** 2).astype(np.float32) + 1e-4
    ux      = (dx_raw / dist).astype(np.float32)   # unit outward X
    uy      = (dy_raw / dist).astype(np.float32)   # unit outward Y

    max_push_x = dc * face_w * 0.065
    max_push_y = dc * face_h * 0.034   # less vertical: cheeks expand sideways

    flow_x = -env * ux * max_push_x   # inverse remap: source is inward
    flow_y = -env * uy * max_push_y

    # ── Volume coupling: cheek expansion → slight chin drop ───────────────
    # When the cheeks push outward, real fat redistributes and the
    # submental region sags slightly downward.  Coupling weight: 35 %.
    flow_y += -env_chin * dc * face_h * 0.022   # additional downward pull

    # ── Sigmoid identity lock — fade to zero above the nose tip ──────────
    nose_idxs = [i for i in groups.get("nose_tip", []) if i < n_pts]
    if nose_idxs:
        nose_y = float(src_pts[nose_idxs, 1].mean())
    else:
        nose_y = float(np.min(src_pts[:, 1])) + face_h * 0.42

    band    = face_h * 0.18   # wider transition = smoother blend into forehead
    sig_arg = np.clip((yy - nose_y) / (band + 1e-4), -7.0, 7.0)
    lock    = (1.0 / (1.0 + np.exp(-sig_arg))).astype(np.float32)

    flow_x *= lock
    flow_y *= lock

    # ── Nose anchor lock ──────────────────────────────────────────────────
    # MP-468 indices 1, 2, 4, 5, 168 mark the nose bridge, alae and tip.
    # Build an inverse-Gaussian mask that suppresses flow to zero near these
    # fixed points so the nose never widens regardless of face_fullness value.
    # Suppression is applied BEFORE the Gaussian blur so the blur naturally
    # feathers the boundary — no hard edge artifact.
    _NOSE_ANCHORS = [1, 2, 4, 5, 168]
    nose_pts = _safe_pts(_NOSE_ANCHORS, "nose_tip")
    if nose_pts:
        nose_env = np.zeros((h, w), dtype=np.float32)
        for p in nose_pts:
            nose_env = np.maximum(
                nose_env,
                _g2d(float(p[0]), float(p[1]), face_w * 0.13, face_h * 0.10),
            )
        nose_free = (1.0 - np.clip(nose_env, 0.0, 1.0)).astype(np.float32)
        flow_x *= nose_free
        flow_y *= nose_free

    # ── Gaussian blur on flow fields ─────────────────────────────────────
    ksize      = max(int(face_h * 0.13) | 1, 11)   # odd, ~13 % face height
    sigma_blur = ksize * 0.50
    flow_x = cv2.GaussianBlur(flow_x, (ksize, ksize), sigma_blur)
    flow_y = cv2.GaussianBlur(flow_y, (ksize, ksize), sigma_blur)

    return cv2.remap(
        image,
        xx + flow_x,
        yy + flow_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )

def _falloff_weights(src_points: np.ndarray, anchor_idxs: list, sigma: float) -> np.ndarray:
    """Per-point Gaussian weight: distance from nearest anchor point."""
    valid = [i for i in anchor_idxs if 0 <= i < len(src_points)]
    if not valid:
        return np.zeros(len(src_points), dtype=np.float32)
    anchors = src_points[valid]                                   # (K, 2)
    diffs = src_points[:, None, :] - anchors[None, :, :]         # (N, K, 2)
    min_d2 = np.min(np.sum(diffs ** 2, axis=2), axis=1)          # (N,)
    return np.exp(-min_d2 / (2.0 * sigma * sigma)).astype(np.float32)


def _build_dst_landmarks_v2(
    src_points: np.ndarray,
    groups: dict,
    smile: float,
    brow: float,
    lip: float,
    slim: float,
    w: int,
    h: int,
    double_chin: float = 0.0,
) -> np.ndarray:
    """
    Build displaced target landmark set with per-expression Gaussian falloff.
    All magnitudes are scaled relative to the face bounding box, never raw pixels.
    """
    min_x = float(np.min(src_points[:, 0]))
    max_x = float(np.max(src_points[:, 0]))
    min_y = float(np.min(src_points[:, 1]))
    max_y = float(np.max(src_points[:, 1]))
    face_w = max(max_x - min_x, 1.0)
    face_h = max(max_y - min_y, 1.0)
    ref = max(face_w, face_h)

    displacements = np.zeros_like(src_points)  # (N, 2)

    def add_disp(keys, dx: float, dy: float, sigma_frac: float = 0.12):
        idxs: list = []
        for k in keys:
            idxs.extend(groups.get(k, []))
        sigma = sigma_frac * ref
        w_arr = _falloff_weights(src_points, idxs, sigma)
        displacements[:, 0] += dx * w_arr
        displacements[:, 1] += dy * w_arr

    # ── SMILE ──────────────────────────────────────────────────────────────
    if abs(smile) > 1e-4:
        s = float(smile)
        # Corners: outward and upward — wide sigma propagates to cheek skin
        add_disp(["mouth_left"],         dx=-s * 0.022 * face_w, dy=-s * 0.038 * face_h, sigma_frac=0.13)
        add_disp(["mouth_right"],        dx= s * 0.022 * face_w, dy=-s * 0.038 * face_h, sigma_frac=0.13)
        # Inner upper lip: slight lift (natural widening around the arc)
        add_disp(["inner_upper"],        dx=0, dy=-s * 0.006 * face_h, sigma_frac=0.05)
        # Upper outer lip arc lift
        add_disp(["upper_lip"],          dx=0, dy=-s * 0.010 * face_h, sigma_frac=0.07)
        # Inner lower lip: slight drop (open-mouth smile depth)
        add_disp(["inner_lower"],        dx=0, dy= s * 0.005 * face_h, sigma_frac=0.05)
        # Lower outer lip: slight drop
        add_disp(["lower_lip"],          dx=0, dy= s * 0.008 * face_h, sigma_frac=0.07)
        # Cheek rise — large sigma so surrounding skin tissue follows
        add_disp(["cheek_left"],         dx=-s * 0.005 * face_w, dy=-s * 0.020 * face_h, sigma_frac=0.18)
        add_disp(["cheek_right"],        dx= s * 0.005 * face_w, dy=-s * 0.020 * face_h, sigma_frac=0.18)
        # Subtle lower-eye crinkle (anatomy: zygomatic muscle)
        add_disp(["lower_eye_left"],     dx=0, dy=-s * 0.008 * face_h, sigma_frac=0.08)
        add_disp(["lower_eye_right"],    dx=0, dy=-s * 0.008 * face_h, sigma_frac=0.08)

    # ── BROW RAISE ─────────────────────────────────────────────────────────
    if abs(brow) > 1e-4:
        b = float(brow)
        # Brows move up — sigma large enough to pull brow body uniformly
        add_disp(["brow_left"],          dx=0, dy=-b * 0.052 * face_h, sigma_frac=0.10)
        add_disp(["brow_right"],         dx=0, dy=-b * 0.052 * face_h, sigma_frac=0.10)
        # Forehead stretches upward — wider sigma for skin continuity
        add_disp(["forehead_left"],      dx=0, dy=-b * 0.022 * face_h, sigma_frac=0.16)
        add_disp(["forehead_right"],     dx=0, dy=-b * 0.022 * face_h, sigma_frac=0.16)
        # Upper eyelid follows very slightly (anatomically coupled)
        add_disp(["upper_eyelid_left"],  dx=0, dy=-b * 0.006 * face_h, sigma_frac=0.07)
        add_disp(["upper_eyelid_right"], dx=0, dy=-b * 0.006 * face_h, sigma_frac=0.07)

    # ── LIP WIDENING ───────────────────────────────────────────────────────
    if abs(lip) > 1e-4:
        li = float(lip)
        # Corners spread outward; centre stays fixed (sigma small to preserve philtrum)
        add_disp(["mouth_left"],         dx=-li * 0.030 * face_w, dy=0, sigma_frac=0.08)
        add_disp(["mouth_right"],        dx= li * 0.030 * face_w, dy=0, sigma_frac=0.08)
        # Full outer lip contour — scale outward proportionally from lip centre
        add_disp(["upper_lip"],          dx=0, dy=-li * 0.006 * face_h, sigma_frac=0.06)
        add_disp(["lower_lip"],          dx=0, dy= li * 0.006 * face_h, sigma_frac=0.06)
        # Inner lip ring also expands slightly (keeps lip plump, not flat)
        add_disp(["inner_upper"],        dx=0, dy=-li * 0.003 * face_h, sigma_frac=0.04)
        add_disp(["inner_lower"],        dx=0, dy= li * 0.003 * face_h, sigma_frac=0.04)

    # ── FACE SLIMMING ──────────────────────────────────────────────────────
    if abs(slim) > 1e-4:
        sl = float(slim)
        ctr_x = float(
            np.mean([src_points[i, 0] for i in groups.get("center", [1]) if 0 <= i < len(src_points)])
        )
        # Per-point inward pull scaled by distance from centre; use add_disp so
        # Gaussian falloff propagates the displacement to neighbouring skin points.
        # sigma = 0.060 * ref  (tight radius — falloff to near-zero within ~80 px,
        # preventing the cumulative effect of 18 jaw points from reaching the nose).
        max_pull = 0.13 * face_w
        sigma = 0.060 * ref
        for side_key in ("jaw_left", "jaw_right"):
            for i in groups.get(side_key, []):
                if 0 <= i < len(src_points):
                    raw_pull = (ctr_x - src_points[i, 0]) * sl * 0.20
                    pull_x = float(np.clip(raw_pull, -max_pull, max_pull))
                    w_arr = _falloff_weights(src_points, [i], sigma)
                    displacements[:, 0] += pull_x * w_arr
        # Chin: very subtle upward tuck (avoid an excessively sharp chin)
        chin_pull = sl * 0.010 * face_h
        chin_sigma = 0.055 * ref
        chin_idxs = [i for i in groups.get("chin", []) if 0 <= i < len(src_points)]
        if chin_idxs:
            w_arr_chin = _falloff_weights(src_points, chin_idxs, chin_sigma)
            displacements[:, 1] -= chin_pull * w_arr_chin

        # Hard-zero identity regions so slim NEVER moves the nose, eyes, or mouth
        # even when multiple jaw points contribute small falloff values that add up.
        identity_keys = (
            "nose_tip", "center",
            "lower_eye_left", "lower_eye_right",
            "upper_eyelid_left", "upper_eyelid_right",
            "inner_upper", "inner_lower",
            "upper_lip", "lower_lip",
            "brow_left", "brow_right",
        )
        for key in identity_keys:
            for i in groups.get(key, []):
                if 0 <= i < len(src_points):
                    displacements[i] = 0.0

    # ── DOUBLE CHIN ────────────────────────────────────────────────────────
    # Push chin / jaw-bottom landmarks downward to simulate submental fat.
    # We use a narrow sigma so only the lower face tissue moves and the neck
    # boundary (outside the face hull) stays intact.
    if abs(double_chin) > 1e-4:
        dc = float(double_chin)
        # Primary: chin tip down
        add_disp(["chin"],      dx=0, dy= dc * 0.055 * face_h, sigma_frac=0.10)
        # Secondary: jaw bottom fills out laterally + down
        add_disp(["jaw_left"],  dx=-dc * 0.018 * face_w, dy= dc * 0.030 * face_h, sigma_frac=0.09)
        add_disp(["jaw_right"], dx= dc * 0.018 * face_w, dy= dc * 0.030 * face_h, sigma_frac=0.09)
        # Hard-zero identity regions so double_chin never touches eyes/nose/mouth
        for key in ("nose_tip", "center", "lower_eye_left", "lower_eye_right",
                    "upper_eyelid_left", "upper_eyelid_right",
                    "inner_upper", "inner_lower", "upper_lip", "lower_lip",
                    "brow_left", "brow_right"):
            for i in groups.get(key, []):
                if 0 <= i < len(src_points):
                    displacements[i] = 0.0

    # Global displacement clamp — 12 % cap keeps deformation in natural range
    max_disp = 0.12 * ref
    disp_mag = np.linalg.norm(displacements, axis=1, keepdims=True)
    scale = np.where(disp_mag > max_disp, max_disp / (disp_mag + 1e-8), 1.0)
    displacements *= scale

    dst = src_points + displacements
    dst[:, 0] = np.clip(dst[:, 0], 1, w - 2)
    dst[:, 1] = np.clip(dst[:, 1], 1, h - 2)
    return dst


# ── Delaunay triangulation helpers ──────────────────────────────────────────

def _hull_fence_points(src_pts: np.ndarray, n: int = 60) -> np.ndarray:
    """
    Return N equally arc-length-spaced points along the convex hull perimeter
    of src_pts.  These points are added at IDENTICAL positions to both
    src_full and dst_full, creating a rigid "fence" around the face boundary.

    Effect: Delaunay triangles inside the face can only connect face landmarks
    to each other or to fence points — never directly to the oval/border ring.
    This prevents background pixels leaking into face-interior triangles when
    the jaw is slimmed inward, eliminating the blue-stripe artifact.
    """
    hull = cv2.convexHull(src_pts.astype(np.float32))  # (k, 1, 2)
    hull = hull[:, 0, :]                                 # (k, 2)
    hull_closed = np.vstack([hull, hull[:1]])            # close the loop

    diffs   = np.diff(hull_closed, axis=0)               # (k, 2) edge vectors
    seg_len = np.linalg.norm(diffs, axis=1)              # edge lengths
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    total   = cum_len[-1]
    if total < 1.0:
        return hull

    targets = np.linspace(0.0, total, n, endpoint=False)
    pts: list = []
    for t in targets:
        idx = int(np.searchsorted(cum_len, t, side='right')) - 1
        idx = min(idx, len(hull_closed) - 2)
        seg_t = (t - cum_len[idx]) / (seg_len[idx] + 1e-9)
        pts.append(hull_closed[idx] + seg_t * diffs[idx])
    return np.array(pts, dtype=np.float32)


def _add_boundary_points(points: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Return a merged point set of face landmarks + stable anchors.

    Two rings of anchors are added so the Delaunay mesh covers the full canvas
    without degenerate boundary triangles:

    1. Image-border ring — 16 points per edge (64 total), clamped 1 px inside
       the canvas so cv2.Subdiv2D never raises an out-of-bounds error.
    2. Mid-frame oval ring — 24 points on an ellipse centred on the face
       bounding box, placed between the face landmarks and the image border.
       These prevent very large, floppy triangles from spanning the whole frame
       when the face is small relative to the image.

    Anchors appear at identical positions in *both* the source and destination
    point sets, so they produce identity triangles outside the face region and
    the background never moves.
    """
    n_edge = 16           # border ring: points per edge
    n_oval = 24           # mid-frame oval: total points

    # —— Image border ring ——
    margin = 1            # 1-px inset so all points are strictly inside (0, w) x (0, h)
    xs = np.linspace(margin, w - 1 - margin, n_edge)
    ys = np.linspace(margin, h - 1 - margin, n_edge)
    bnd: list = []
    for x in xs:
        bnd.append([float(x), float(margin)])
        bnd.append([float(x), float(h - 1 - margin)])
    for y in ys[1:-1]:
        bnd.append([float(margin), float(y)])
        bnd.append([float(w - 1 - margin), float(y)])

    # —— Mid-frame oval ring ——
    if len(points) >= 4:
        cx = float(np.mean(points[:, 0]))
        cy = float(np.mean(points[:, 1]))
        # Oval radii ≈ midpoint between face extent and canvas edge
        rx = max(float(np.max(points[:, 0]) - np.min(points[:, 0])) * 0.78,
                 min(cx, w - cx) * 0.60)
        ry = max(float(np.max(points[:, 1]) - np.min(points[:, 1])) * 0.78,
                 min(cy, h - cy) * 0.60)
    else:
        cx, cy = w / 2.0, h / 2.0
        rx, ry = w * 0.40, h * 0.40
    for i in range(n_oval):
        angle = 2.0 * np.pi * i / n_oval
        ox = float(np.clip(cx + rx * np.cos(angle), margin, w - 1 - margin))
        oy = float(np.clip(cy + ry * np.sin(angle), margin, h - 1 - margin))
        bnd.append([ox, oy])

    return np.vstack([points, np.array(bnd, dtype=np.float32)])


def _triangle_indices_from_subdiv(points: np.ndarray, w: int, h: int) -> list:
    """
    Compute Delaunay triangles and return valid (i, j, k) index triples.

    Triangles are rejected when they are:
    - degenerate  (any vertex maps to None)
    - duplicate   (same index triple already seen)
    - tiny        (area < 2 px²)
    - extreme aspect ratio  (longest edge > 8 × shortest edge)
    - oversized   (any edge longer than 55 % of the larger image dimension,
                   which catches huge background triangles that should not
                   be warped and that would otherwise cause visible tearing)
    """
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    pts_int = [
        (max(0, min(w - 1, int(round(float(p[0]))))),
         max(0, min(h - 1, int(round(float(p[1]))))))
        for p in points
    ]
    point_dict: dict = {}
    for idx, (x, y) in enumerate(pts_int):
        point_dict.setdefault((x, y), idx)
    for x, y in pts_int:
        try:
            subdiv.insert((x, y))
        except cv2.error:
            pass

    tri_list = subdiv.getTriangleList()
    if tri_list is None or len(tri_list) == 0:
        return []

    pts_arr = np.array(points, dtype=np.float32)
    max_edge_len = max(w, h) * 0.55   # reject triangles spanning more than 55 % of the frame

    def idx_of(px: float, py: float):
        key = (int(round(px)), int(round(py)))
        if key in point_dict:
            return point_dict[key]
        arr = pts_arr - np.array([px, py], dtype=np.float32)
        d2 = np.sum(arr * arr, axis=1)
        i = int(np.argmin(d2))
        return i if d2[i] < 9.0 else None

    seen: set = set()
    tris: list = []
    for t in tri_list:
        x1, y1, x2, y2, x3, y3 = t
        if not (0 <= x1 <= w and 0 <= y1 <= h and
                0 <= x2 <= w and 0 <= y2 <= h and
                0 <= x3 <= w and 0 <= y3 <= h):
            continue
        i1 = idx_of(x1, y1)
        i2 = idx_of(x2, y2)
        i3 = idx_of(x3, y3)
        if i1 is None or i2 is None or i3 is None:
            continue
        if i1 == i2 or i2 == i3 or i1 == i3:
            continue
        key = tuple(sorted((i1, i2, i3)))
        if key in seen:
            continue

        # —— Geometric quality filters ——
        verts = np.array([[x1, y1], [x2, y2], [x3, y3]], dtype=np.float32)
        area = abs(_triangle_area(verts))
        if area < 2.0:
            continue
        # Edge lengths
        e0 = float(np.linalg.norm(verts[1] - verts[0]))
        e1 = float(np.linalg.norm(verts[2] - verts[1]))
        e2 = float(np.linalg.norm(verts[0] - verts[2]))
        longest  = max(e0, e1, e2)
        shortest = min(e0, e1, e2)
        # Reject triangles that span too large a region (background / border artefacts)
        if longest > max_edge_len:
            continue
        # Reject degenerate slivers: aspect ratio > 8:1
        if shortest < 1e-3 or longest / shortest > 8.0:
            continue

        seen.add(key)
        tris.append(key)
    return tris


def _triangle_area(t: np.ndarray) -> float:
    """Signed area of a 2-D triangle."""
    return 0.5 * float(
        (t[1, 0] - t[0, 0]) * (t[2, 1] - t[0, 1])
        - (t[2, 0] - t[0, 0]) * (t[1, 1] - t[0, 1])
    )


def _warp_triangle(
    src: np.ndarray,
    dst: np.ndarray,
    t_src: np.ndarray,
    t_dst: np.ndarray,
    *,
    interp: int = cv2.INTER_CUBIC,
) -> None:
    """
    Warp one Delaunay triangle from the source image into the destination using
    an affine transformation, then alpha-blend it into the destination buffer.

    Affine Transformation Theory
    -----------------------------
    An affine transform maps triangle t_src → t_dst via a 2×3 matrix M:

        [x']   [m00 m01 m02] [x]
        [y'] = [m10 m11 m12] [y]
                              [1]

    Given exactly 3 non-collinear point correspondences, M has a unique
    solution (cv2.getAffineTransform).  The mapping preserves:
      • Straight lines (collinearity)
      • Parallel lines (parallelism)
      • Ratios of distances along lines

    It does NOT preserve angles or lengths in general — that is what enables
    the natural-looking local deformation needed for expression warping.

    Algorithm
    ---------
    1. Compute bounding rectangles r1 (source) and r2 (destination) to work on
       cropped patches rather than the full image — avoids allocating a full-size
       warp for every triangle (significant performance saving).

    2. Express triangle vertices relative to their bounding-rect origins so the
       cv2.warpAffine call operates on the small patch only.

    3. Create an anti-aliased binary fill mask for the destination triangle and
       apply a 1-pixel Gaussian feather to soften triangle-boundary seams.

    4. Blend the warped patch into the destination:
           dst_roi = dst_roi * (1 − mask) + warped * mask
       The soft mask eliminates the hard edges that would otherwise appear at
       the boundary between adjacent triangles (seam artefacts).

    Degenerate triangle guard
    -------------------------
    Triangles with area < 4 px² or with a dst/src area ratio > 5 (extreme
    distortion) are skipped to prevent ringing or black-fill artefacts.
    """
    area_src = abs(_triangle_area(t_src))
    area_dst = abs(_triangle_area(t_dst))
    if area_src < 4.0 or area_dst < 4.0:
        return
    if area_src > 0 and (area_dst / area_src > 5.0 or area_src / area_dst > 5.0):
        return

    # Aspect-ratio guard on destination triangle
    e0 = float(np.linalg.norm(t_dst[1] - t_dst[0]))
    e1 = float(np.linalg.norm(t_dst[2] - t_dst[1]))
    e2 = float(np.linalg.norm(t_dst[0] - t_dst[2]))
    longest  = max(e0, e1, e2)
    shortest = min(e0, e1, e2)
    if shortest < 1e-3 or longest / shortest > 8.0:
        return

    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))
    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return

    # Shift vertices to be relative to their bounding-rect top-left corners
    t1_rect = np.float32([[t_src[i][0] - r1[0], t_src[i][1] - r1[1]] for i in range(3)])
    t2_rect = np.float32([[t_dst[i][0] - r2[0], t_dst[i][1] - r2[1]] for i in range(3)])

    # Feather radius: 1 px for tiny triangles, up to 3 px for large ones.
    # Using an odd kernel so GaussianBlur sigma is well-defined.
    tri_side = max(r2[2], r2[3])
    feather_r = 1 if tri_side < 20 else (2 if tri_side < 60 else 3)
    feather_k = feather_r * 2 + 1   # always odd: 3, 5, or 7
    feather_sigma = 0.5 + feather_r * 0.4

    mask = np.zeros((r2[3], r2[2]), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), 1.0, lineType=cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (feather_k, feather_k), feather_sigma)

    src_patch = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    if src_patch.size == 0:
        return

    # Compute and apply the 2×3 affine matrix for this triangle pair
    mat = cv2.getAffineTransform(t1_rect, t2_rect)
    warped = cv2.warpAffine(
        src_patch, mat, (r2[2], r2[3]),
        flags=interp,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    dst_roi = dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    if dst_roi.shape[:2] != warped.shape[:2]:
        return

    # Alpha-blend warped patch into the destination using the feathered mask
    mask3 = mask[:, :, None]
    dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = (
        dst_roi * (1.0 - mask3) + warped * mask3
    )


# ── Thin Plate Spline (TPS) warping ─────────────────────────────────────────

def _tps_build(dst_ctrl: np.ndarray, src_ctrl: np.ndarray, lam: float = 1e-3):
    """
    Solve TPS coefficients for the inverse mapping (dst → src), so that
    cv2.remap can sample source pixels for every destination pixel.

    ``lam`` is the Tikhonov regularisation strength.  Larger values produce
    smoother (less over-fitted) warps; 1e-3 is a good default for facial
    expression work.  Returns (W, A, ctrl) or None on failure.
    """
    n = len(dst_ctrl)
    if n < 4:
        return None
    ctrl = dst_ctrl.astype(np.float64)
    out  = src_ctrl.astype(np.float64)

    # TPS kernel matrix K — U(r) = r² log r²
    ctrl_sq = np.sum(ctrl ** 2, axis=1)
    r2 = ctrl_sq[:, None] + ctrl_sq[None, :] - 2.0 * (ctrl @ ctrl.T)  # (n,n)
    np.fill_diagonal(r2, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        K = np.where(r2 > 0, r2 * np.log(np.maximum(r2, 1e-14)), 0.0)

    P = np.column_stack([np.ones(n), ctrl])          # (n, 3)
    Z = np.zeros((3, 3), np.float64)
    L = np.block([[K, P], [P.T, Z]])                 # (n+3, n+3)
    L[:n, :n] += np.eye(n) * lam                    # Tikhonov regularisation

    rhs_x = np.concatenate([out[:, 0], np.zeros(3)])
    rhs_y = np.concatenate([out[:, 1], np.zeros(3)])
    try:
        cx = np.linalg.solve(L, rhs_x)
        cy = np.linalg.solve(L, rhs_y)
    except np.linalg.LinAlgError:
        return None

    W = np.column_stack([cx[:n], cy[:n]])   # (n, 2) weights
    A = np.column_stack([cx[n:], cy[n:]])   # (3, 2) affine
    return W, A, ctrl


def _tps_eval(W: np.ndarray, A: np.ndarray, ctrl: np.ndarray,
              query_pts: np.ndarray) -> np.ndarray:
    """Evaluate TPS at query_pts (M, 2); returns mapped (M, 2)."""
    n = len(ctrl)
    K = np.zeros((len(query_pts), n), np.float64)
    for i in range(n):
        diff = query_pts.astype(np.float64) - ctrl[i]
        r2 = np.sum(diff ** 2, axis=1)
        nz = r2 > 0
        K[nz, i] = r2[nz] * np.log(r2[nz])
    P = np.column_stack([np.ones(len(query_pts)), query_pts.astype(np.float64)])
    return (K @ W + P @ A).astype(np.float32)


def _tps_remap(
    image: np.ndarray,
    src_ctrl: np.ndarray,
    dst_ctrl: np.ndarray,
    subsample: int = 4,
    lam: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Warp image using TPS.  Evaluates the dense inverse field on a coarser grid
    then bicubically upsamples it before calling cv2.remap.

    Returns (warped_image, map_x, map_y) so the caller can post-process the
    displacement maps (e.g. with _lock_identity_regions) before committing.
    """
    h, w = image.shape[:2]
    result = _tps_build(dst_ctrl, src_ctrl, lam=lam)
    if result is None:
        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)
        id_x, id_y = np.meshgrid(xs, ys)
        return image.copy(), id_x, id_y
    W, A, ctrl = result

    ys = np.arange(0, h, subsample, dtype=np.float64)
    xs = np.arange(0, w, subsample, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.column_stack([XX.ravel(), YY.ravel()])

    mapped = _tps_eval(W, A, ctrl, grid)
    map_x_s = mapped[:, 0].reshape(len(ys), len(xs)).astype(np.float32)
    map_y_s = mapped[:, 1].reshape(len(ys), len(xs)).astype(np.float32)

    map_x = cv2.resize(map_x_s, (w, h), interpolation=cv2.INTER_CUBIC)
    map_y = cv2.resize(map_y_s, (w, h), interpolation=cv2.INTER_CUBIC)

    warped = cv2.remap(image, map_x, map_y,
                       interpolation=cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_REFLECT_101)
    return warped, map_x, map_y


# ── Main expression warp orchestrator ────────────────────────────────────────


def _blend_warped_face_mask(
    original: np.ndarray,
    warped: np.ndarray,
    src_pts: np.ndarray,
    w: int,
    h: int,
) -> np.ndarray:
    """
    Blend the warped face back onto the original using a Gaussian-feathered
    convex hull face mask.  Hair, neck, and background come from the unmodified
    original; only the face interior receives the warp result.
    """
    hull = cv2.convexHull(src_pts.astype(np.int32))
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, hull, 255, lineType=cv2.LINE_AA)

    # Slight erosion prevents hard blending artefacts at the convex hull edge
    erode_r = max(3, min(w, h) // 55)
    erode_k = _odd_kernel(erode_r)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
    face_mask = cv2.erode(face_mask, erode_kernel, iterations=2)

    # Gaussian feathering — radius ≈ 4 % of the shorter image dimension
    feather_r = max(15, min(w, h) // 25)
    feather_k = _odd_kernel(feather_r)
    alpha = cv2.GaussianBlur(
        face_mask.astype(np.float32) / 255.0, (feather_k, feather_k), 0
    )
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]

    result = (
        original.astype(np.float32) * (1.0 - alpha)
        + warped.astype(np.float32) * alpha
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_expression_warp(
    image: np.ndarray,
    smile: float,
    brow: float,
    lip: float,
    slim: float,
    warp_mode: str = "delaunay",
    save_debug: bool = False,
    interp: int = cv2.INTER_CUBIC,
    double_chin: float = 0.0,
    face_fullness: float = 0.0,
) -> tuple:
    """
    Anatomical Delaunay expression warp using MediaPipe 468-point mesh.

    Pipeline:
      1. Extract full 468-pt MediaPipe landmarks with anatomical group definitions
         (jaw, eyebrows, eyes, nose, mouth, cheeks, forehead).  Synthetic 68-pt
         fallback used when MediaPipe is unavailable.
      2. Displace only expression-relevant landmark groups using Gaussian falloff.
         Cheek support points are included for smile so the mouth region does not
         look detached.  Identity anchors (nose tip, inner eye corners) are not moved.
      3. Dense boundary anchors added at identical src & dst positions — background
         stays completely still.
      4. Delaunay triangulate the full landmark + boundary set.
      5. Affine-warp each triangle via cv2.getAffineTransform + cv2.warpAffine.
      6. Blend back using a Gaussian-feathered convex hull face mask so hair,
         neck, and background come from the unmodified original.

    Returns (result, debug_paths, used_lm_count, moved_pts, max_disp, tri_count).
    """
    h, w = image.shape[:2]
    if abs(smile) < 1e-4 and abs(brow) < 1e-4 and abs(lip) < 1e-4 and abs(slim) < 1e-4 and abs(double_chin) < 1e-4 and abs(face_fullness) < 1e-4:
        return image.copy(), {}, 0, 0, 0.0, 0, "none"

    # 1. Anatomical landmark extraction: full 468-pt MediaPipe mesh with group defs
    #    (or synthetic 68-pt fallback with matching group defs when MP unavailable).
    #    _extract_landmarks_for_warp is always used — _get_68_pts_for_warp is NOT used
    #    here; it is only used by the expression-transfer endpoint.
    src_pts, groups = _extract_landmarks_for_warp(image)
    n_lm = len(src_pts)
    # Determine source for debug / response field
    lm_source = "mediapipe468" if n_lm >= 468 else "fallback68"

    # 2. Build displaced destinations via per-group Gaussian falloff.
    #    Smile/brow/lip stay on the anatomical mesh.  Slimming is applied later
    #    as a dense remap because jaw-only Delaunay movement creates visible
    #    cheek-edge artefacts on high-contrast backgrounds.
    dst_pts_stats = _build_dst_landmarks_v2(src_pts, groups, smile, brow, lip, slim, w, h, double_chin=double_chin)
    # double_chin is intentionally excluded from dst_pts_mesh: it is applied
    # exclusively as a post-pass dense remap in _apply_double_chin_flow.
    # Including it here too caused double-displacement → stepping artefacts.
    dst_pts_mesh = _build_dst_landmarks_v2(src_pts, groups, smile, brow, lip, 0.0, w, h, double_chin=0.0)

    # 3. Per-point displacement stats used by response fields and debug output
    disp = np.linalg.norm(dst_pts_stats - src_pts, axis=1)
    moved_pts = int(np.sum(disp > 0.3))
    max_disp = float(np.max(disp)) if moved_pts > 0 else 0.0
    moved_indices = [int(i) for i in np.where(disp > 0.3)[0]]

    # 4. Face-hull fence + outer boundary anchors.
    #
    #    The fence is a dense ring of points sampled along the convex hull of
    #    src_pts.  Added at IDENTICAL positions to both src_full and dst_full
    #    (identity = no warping) it acts as a rigid barrier:
    #       - Inside fence  → triangles only connect face landmarks; background
    #                         pixels can never leak into face-interior triangles.
    #       - Outside fence → identity triangles; background stays perfectly still.
    #    Without this fence, Delaunay creates large triangles from the slimmed
    #    jaw all the way to the oval/border anchors; those triangles sample the
    #    exposed background gap and produce the blue-stripe artefact on the cheeks.
    debug_paths: dict = {}
    has_mesh_expression = (
        abs(smile) >= 1e-4 or abs(brow) >= 1e-4 or abs(lip) >= 1e-4 or abs(double_chin) >= 1e-4 or abs(face_fullness) >= 1e-4
    )

    if has_mesh_expression:
        fence = _hull_fence_points(src_pts, n=60)   # same coords in src and dst
        src_full = _add_boundary_points(np.vstack([src_pts, fence]), w, h)
        dst_full = _add_boundary_points(np.vstack([dst_pts_mesh, fence]), w, h)
    else:
        src_full = dst_full = None

    if has_mesh_expression and warp_mode == "tps":
        try:
            # Use boundary-anchored control sets (src_full / dst_full) so the
            # background anchor points keep the warp to zero outside the face.
            warped, tps_map_x, tps_map_y = _tps_remap(
                image, src_full, dst_full, subsample=4, lam=4e-3
            )

            # Apply identity lock: suppress displacement near nose tip & eye
            # centres so those landmarks stay perfectly stable even under TPS.
            _img_h, _img_w = image.shape[:2]
            _xs = np.arange(_img_w, dtype=np.float32)
            _ys = np.arange(_img_h, dtype=np.float32)
            _XX, _YY = np.meshgrid(_xs, _ys)
            flow = np.stack([tps_map_x - _XX, tps_map_y - _YY], axis=2)
            flow = _lock_identity_regions(flow, src_pts, groups)
            tps_map_x = _XX + flow[:, :, 0]
            tps_map_y = _YY + flow[:, :, 1]
            warped = cv2.remap(image, tps_map_x, tps_map_y,
                               interpolation=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REFLECT_101)

            mse = float(np.mean((warped.astype(np.float32) - image.astype(np.float32)) ** 2))
            if mse > 3000.0:
                raise ValueError(f"TPS MSE too high ({mse:.0f})")
        except Exception:
            warp_mode = "delaunay"

    if has_mesh_expression and warp_mode != "tps":
        # 5. Delaunay triangulation on full landmark + boundary set
        tris = _triangle_indices_from_subdiv(src_full, w, h)
        tri_count = len(tris)

        # 6. Affine-warp each triangle (cv2.getAffineTransform + cv2.warpAffine)
        src_f32 = image.astype(np.float32)
        warped_f32 = src_f32.copy()
        for (i, j, k) in tris:
            t_src = np.array([src_full[i], src_full[j], src_full[k]], dtype=np.float32)
            t_dst = np.array([dst_full[i], dst_full[j], dst_full[k]], dtype=np.float32)
            _warp_triangle(src_f32, warped_f32, t_src, t_dst, interp=interp)
        warped = np.clip(warped_f32, 0, 255).astype(np.uint8)
    elif has_mesh_expression:
        tri_count = 0
    else:
        warped = image.copy()
        tri_count = 0

    # 7. Blend warped face back via Gaussian-feathered convex hull mask.
    #    We use dst_pts (the moved/slimmed landmarks) for the hull so that the
    #    convex hull follows the NEW face boundary rather than the original one.
    #    This prevents stretched boundary triangles from sampling the background
    #    (e.g. a solid-blue studio background) and projecting those pixels back
    #    into the face area as coloured stripes or grid artefacts.
    #    Outside the dst-hull the original image shows through, which for any
    #    background is exactly the correct content.
    if has_mesh_expression:
        result = _blend_warped_face_mask(image, warped, dst_pts_mesh, w, h)
    else:
        result = warped

    # 8. Apply face slimming as a dense local remap after the mesh expression
    #    pass.  This keeps the jaw/cheek boundary smooth and avoids the
    #    Delaunay fold-over artefact that made slim output look visually broken.
    result = _apply_slimming_flow(result, src_pts, groups, slim)

    # 9. Apply double chin as a separate dense-remap pass — chin/jaw-bottom
    #    pushed down with a smooth falloff so neck tissue stretches naturally.
    if abs(double_chin) >= 1e-4:
        result = _apply_double_chin_flow(result, src_pts, groups, double_chin)

    # 10. Apply face fullness — cheek anchors pushed radially outward from
    #     the face centroid, locked above the nose line.
    if abs(face_fullness) >= 1e-4:
        result = _apply_face_fullness_flow(result, src_pts, groups, face_fullness)

    if save_debug:
        moved_set = set(moved_indices)
        vis_src = image.copy()
        for i, pt in enumerate(src_pts):
            colour = (0, 0, 255) if i in moved_set else (0, 255, 0)
            cv2.circle(vis_src, (int(pt[0]), int(pt[1])), 2, colour, -1)
        debug_paths["source_landmarks"] = _save_image(vis_src, "debug_src_lm")

        vis_dst = result.copy()
        for pt in dst_pts_stats:
            cv2.circle(vis_dst, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)
        debug_paths["dest_landmarks"] = _save_image(vis_dst, "debug_dst_lm")

        vis_vec = image.copy()
        for i, (p_s, p_d) in enumerate(zip(src_pts, dst_pts_stats)):
            if i in moved_set:
                cv2.arrowedLine(
                    vis_vec,
                    (int(p_s[0]), int(p_s[1])),
                    (int(p_d[0]), int(p_d[1])),
                    (0, 255, 255), 1, tipLength=0.3,
                )
        debug_paths["displacement_vectors"] = _save_image(vis_vec, "debug_disp")
        debug_paths["before_after"] = _save_image(
            np.hstack([image, result]), "debug_before_after"
        )

    return result, debug_paths, n_lm, moved_pts, max_disp, tri_count, lm_source


@app.get("/")
def index_page():
    resp = send_from_directory(ROOT, "index.html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.get("/style.css")
def style_page():
    return send_from_directory(ROOT, "style.css")


@app.get("/app.js")
def script_page():
    resp = send_from_directory(ROOT, "app.js")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.get("/results/<path:filename>")
def results_file(filename):
    return send_from_directory(RESULTS_DIR, filename)


@app.get("/uploads/<path:filename>")
def uploads_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.get("/api/health")
def api_health():
    return jsonify({"ok": True, "service": "FaceWarp DSP API", "timestamp": time.time()})


@app.post("/api/preprocess")
def api_preprocess():
    started = time.perf_counter()

    try:
        image = _read_image(request.files.get("image"))
    except ValueError as exc:
        return _error(str(exc), 400)

    target_size = int(request.form.get("size", 256))
    target_size = max(64, min(target_size, 1024))
    grayscale = request.form.get("grayscale", "false").lower() == "true"
    norm_mode = request.form.get("norm", "1")
    detector_mode = (request.form.get("detector", "haar") or "haar").strip().lower()
    detector_enabled = detector_mode != "none"
    try:
        padding_percent = float(request.form.get("padding", 20))
    except ValueError:
        padding_percent = 20.0

    # Optional extra controls (interpolation is applied)
    interp_map = {
        "bicubic":  cv2.INTER_CUBIC,
        "bilinear": cv2.INTER_LINEAR,
        "nearest":  cv2.INTER_NEAREST,
    }
    interp = interp_map.get(request.form.get("interp", "bicubic"), cv2.INTER_CUBIC)

    face_crop, face_detected, face_detector = _crop_face_region(
        image=image,
        padding_percent=padding_percent,
        use_detector=detector_enabled,
    )

    face = cv2.resize(face_crop, (target_size, target_size), interpolation=interp)
    face = _normalize_for_preprocess(face, norm_mode)

    if grayscale:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)

    output_path = _save_image(face, "preprocess")
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    return jsonify(
        {
            "ok": True,
            "width": int(face.shape[1]),
            "height": int(face.shape[0]),
            "grayscale": grayscale,
            "normalization": norm_mode,
            "faceDetected": bool(face_detected),
            "faceDetector": face_detector,
            "saved": output_path,
            "durationMs": round(elapsed_ms, 2),
        }
    )


@app.post("/api/landmarks")
def api_landmarks():
    started = time.perf_counter()

    try:
        image = _read_image(request.files.get("image"))
    except ValueError as exc:
        return _error(str(exc), 400)

    h, w = image.shape[:2]
    points = []
    raw_mediapipe = []
    include_reduced68 = request.form.get("includeReduced68", "false").lower() == "true"
    response_mode = request.form.get("landmarkMode", "raw").strip().lower()  # raw | reduced68 | both

    confidence = 72.0  # default: last-resort fallback

    used_method = "fallback-geo"

    if FACE_MESH is not None:
        raw_mediapipe = _get_mediapipe_points(image)
        if raw_mediapipe:
            used_method = "mediapipe"
            confidence = 99.2

            reduced_68 = _reduce_mediapipe_to_68(raw_mediapipe)
            if response_mode == "reduced68":
                points = reduced_68
            else:
                points = raw_mediapipe

            print(f"/api/landmarks: using MediaPipe ({len(raw_mediapipe)} points)")
        else:
            print("/api/landmarks: MediaPipe available but no face landmarks detected, falling back.")

    if not points and used_method != "mediapipe":
        det = _detect_face_bbox(image)
        if det is None:
            # Last-resort geometric fallback: assume centred portrait
            bx, by, bw, bh = int(w * 0.18), int(h * 0.12), int(w * 0.64), int(h * 0.76)
            confidence = 72.0
            used_method = "fallback-geo"
        else:
            bx, by, bw, bh, det_method = det
            confidence = 85.0 if det_method == 'haar' else 78.0
            used_method = f"fallback-{det_method}"
        fallback_points = _synthetic_68_landmarks_from_bbox(bx, by, bw, bh, w, h)
        for idx, (px, py) in enumerate(fallback_points):
            points.append({"id": idx, "x": int(px), "y": int(py)})
        print(f"/api/landmarks: using fallback synthetic landmarks ({len(points)} points)")

    overlay = image.copy()
    for p in points:
        cv2.circle(overlay, (p["x"], p["y"]), 1, (255, 0, 255), -1)

    saved = _save_image(overlay, "landmarks")
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    return jsonify(
        {
            "ok": True,
            "width": int(w),
            "height": int(h),
            "pointCount": len(points),
            "confidence": round(confidence, 1),
            "method": used_method,
            "landmarks": points,
            **({
                "rawLandmarks": raw_mediapipe,
            } if response_mode == "both" else {}),
            **({
                "reduced68": _reduce_mediapipe_to_68(raw_mediapipe),
            } if (used_method == "mediapipe" and (include_reduced68 or response_mode == "both")) else {}),
            "saved": saved,
            "durationMs": round(elapsed_ms, 2),
        }
    )


@app.post("/api/analysis")
def api_analysis():
    """
    Evaluation Layer — quantitative comparison of original vs. transformed image.

    Metrics computed
    ----------------
    MSE  (Mean Squared Error)
        MSE = (1/N) Σ (f(x,y) − g(x,y))²
        Measures average pixel-level deviation.  Range [0, ∞); lower = more
        similar.  Works in the spatial domain on grayscale luminance.

    PSNR  (Peak Signal-to-Noise Ratio)
        PSNR = 20 · log₁₀(MAX / √MSE)   where MAX = 255 for uint8 images.
        Expresses similarity in decibels.  Higher = better quality.
        • > 40 dB : virtually indistinguishable  (de-aging / subtle expression)
        • 30–40 dB: good quality               (moderate aging)
        • < 30 dB : visible differences        (strong aging / heavy warp)
        A flat (identical) image reports 99 dB as a sentinel for MSE = 0.

    SSIM  (Structural Similarity Index)
        SSIM ∈ [−1, 1]; 1 = identical.  Computed with scikit-image using a
        sliding Gaussian window.  Unlike MSE/PSNR, SSIM models human perception
        by comparing luminance, contrast, AND structure simultaneously.

    FFT Energy comparison
        For each image the power spectrum is decomposed into low-frequency and
        high-frequency energy bands (see _fft_energy_components).  The delta
        shows how much the transformation changed the frequency distribution:
        • Aging increases hfLfRatio (more texture energy)
        • De-aging decreases hfLfRatio (more smooth energy)
        This confirms the DSP algorithm is working as intended.
    """
    started = time.perf_counter()

    try:
        original = _read_image(request.files.get("original") or request.files.get("image"))
    except ValueError as exc:
        return _error(f"Original image: {exc}", 400)

    transformed_file = request.files.get("transformed")
    if transformed_file is None or not transformed_file.filename:
        return _error("Transformed image is required for analysis.", 400)

    try:
        transformed = _read_image(transformed_file)
    except ValueError as exc:
        return _error(f"Transformed image: {exc}", 400)

    transformed = _resize_like(transformed, original.shape)

    gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_trans = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY).astype(np.float32)

    mse = float(np.mean((gray_orig - gray_trans) ** 2))
    if mse == 0.0:
        psnr = 99.0
    else:
        psnr = float(20.0 * np.log10(255.0 / np.sqrt(mse)))
    ssim = float(structural_similarity(gray_orig, gray_trans, data_range=255.0))

    orig_fft = _fft_energy_components(gray_orig)
    trans_fft = _fft_energy_components(gray_trans)
    orig_stats = _gray_image_stats(gray_orig)
    trans_stats = _gray_image_stats(gray_trans)
    orig_spectrum = _fft_visuals_b64(gray_orig)
    trans_spectrum = _fft_visuals_b64(gray_trans)

    energy_delta = {
        "totalPct": _percent_delta(orig_fft["energy"]["total"], trans_fft["energy"]["total"]),
        "highPct": _percent_delta(orig_fft["energy"]["high"], trans_fft["energy"]["high"]),
        "lowPct": _percent_delta(orig_fft["energy"]["low"], trans_fft["energy"]["low"]),
        "hfLfRatioPct": _percent_delta(orig_fft["energy"]["hfLfRatio"], trans_fft["energy"]["hfLfRatio"]),
    }
    luminance_delta_pct = _percent_delta(orig_stats["mean"], trans_stats["mean"])

    compare = np.hstack([original, transformed])
    cv2.putText(compare, f"MSE: {mse:.2f}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(compare, f"PSNR: {psnr:.2f} dB", (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(compare, f"SSIM: {ssim:.3f}", (16, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    saved = _save_image(compare, "analysis")

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    return jsonify(
        {
            "ok": True,
            "mse": round(mse, 4),
            "psnr": round(psnr, 4),
            "ssim": round(ssim, 4),
            "energy": trans_fft["energy"],
            "energyComparison": {
                "original": orig_fft["energy"],
                "transformed": trans_fft["energy"],
                "delta": energy_delta,
            },
            "luminance": {
                "original": orig_stats,
                "transformed": trans_stats,
                "deltaMeanPct": luminance_delta_pct,
            },
            "spectrum": {
                "original": orig_spectrum,
                "transformed": trans_spectrum,
            },
            "saved": saved,
            "durationMs": round(elapsed_ms, 2),
        }
    )


@app.post("/api/aging")
def api_aging():
    started = time.perf_counter()

    try:
        image = _read_image(request.files.get("image"))
    except ValueError as exc:
        return _error(str(exc), 400)

    mode = (request.form.get("mode", "aging") or "aging").strip().lower()
    if mode not in {"aging", "deaging"}:
        return _error("Invalid mode. Use 'aging' or 'deaging'.", 400)

    method = (request.form.get("method", "freq") or "freq").strip().lower()
    if method not in ALLOWED_AGING_METHODS:
        return _error("Invalid method. Use 'freq', 'ai', or 'classical'.", 400)

    hpf = (request.form.get("hpf", "gaussian_hpf") or "gaussian_hpf").strip().lower()
    if hpf not in ALLOWED_HPF_FILTERS:
        return _error("Invalid high-pass filter. Use gaussian_hpf, butterworth_hpf, or ideal_hpf.", 400)

    lpf = (request.form.get("lpf", "gaussian_lpf") or "gaussian_lpf").strip().lower()
    if lpf not in ALLOWED_LPF_FILTERS:
        return _error("Invalid low-pass filter. Use gaussian_lpf, butterworth_lpf, or ideal_lpf.", 400)

    try:
        strength = float(request.form.get("strength", 0.5))
        years = int(float(request.form.get("years", 10)))
        contrast = float(request.form.get("contrast", 0.5))
        tone = float(request.form.get("tone", 0.5))
        noise = float(request.form.get("noise", 0.0))
        detail = float(request.form.get("detail", 0.5))
        deterministic = _parse_bool(request.form.get("deterministic", "true"), default=True)
        preserve_edges = _parse_bool(request.form.get("preserveEdges"), default=True)
        save_debug = _parse_bool(request.form.get("debug", "false"), default=False)
    except ValueError:
        return _error("Invalid aging parameters.", 400)

    strength = _clamp(strength, 0.0, 1.0)
    contrast = _clamp(contrast, 0.0, 1.0)
    tone = _clamp(tone, 0.0, 1.0)
    noise = _clamp(noise, 0.0, 1.0)
    detail = _clamp(detail, 0.0, 1.0)
    years = max(-50, min(50, years))

    print("=" * 60, flush=True)
    print("AGING ENDPOINT CALLED", flush=True)
    print(f"  method   = {method!r}", flush=True)
    print(f"  mode     = {mode!r}", flush=True)
    print(f"  strength = {strength:.3f}  (forced to 1.0 for debug)", flush=True)
    print(f"  years    = {years}", flush=True)
    print(f"  image    = {image.shape}  dtype={image.dtype}", flush=True)
    print("=" * 60, flush=True)

    if method == "classical":
        # ── Classical DSP path: aging.py (DoG HPF + CLAHE + hair greying) ──
        if _classical_aging_effect is None:
            return _error("Classical aging module (aging.py) could not be loaded.", 500)
        # Fetch MediaPipe landmarks so aging.py can use region-specific masks.
        classical_lm: list = []
        try:
            classical_lm = _get_mediapipe_points(image)
        except Exception:
            pass
        print(f"  apply_aging_effect called: landmarks={len(classical_lm)} pts  "
              f"intensity={strength:.3f}", flush=True)
        out = _classical_aging_effect(
            image,
            landmarks=classical_lm or None,
            intensity=strength,
        )
        diff = float(np.abs(out.astype(np.int32) - image.astype(np.int32)).mean())
        print(f"  apply_aging_effect DONE: mean_abs_diff={diff:.3f}", flush=True)
        if diff < 2.0:
            print("  WARNING: Aging effect too weak or not applied. "
                  f"mean_abs_diff={diff:.3f} < 2.0", flush=True)
        model_status = "classical-dsp"
        transform_meta = {"maskMethod": "classical"}
        freq_variant = out
        ai_variant = out
    elif method == "ai":
        out, model_status, transform_meta = _ai_age_pipeline(
            image=image,
            mode=mode,
            strength=strength,
            years=years,
            contrast=contrast,
            tone=tone,
            noise=noise,
            detail=detail,
            deterministic=deterministic,
            hpf=hpf,
            lpf=lpf,
            preserve_edges=preserve_edges,
        )
        ai_variant = out
        # Compute standard frequency result separately for comparison
        freq_variant, _ = _frequency_age_pipeline(
            image=image,
            mode=mode,
            strength=strength,
            years=years,
            contrast=contrast,
            tone=tone,
            noise=noise,
            detail=detail,
            deterministic=deterministic,
            hpf=hpf,
            lpf=lpf,
            preserve_edges=preserve_edges,
            semantic_boost=1.0,
        )
    else:
        out, transform_meta = _frequency_age_pipeline(
            image=image,
            mode=mode,
            strength=strength,
            years=years,
            contrast=contrast,
            tone=tone,
            noise=noise,
            detail=detail,
            deterministic=deterministic,
            hpf=hpf,
            lpf=lpf,
            preserve_edges=preserve_edges,
            semantic_boost=1.0,
            save_debug=save_debug,
        )
        diff = float(np.abs(out.astype(np.int32) - image.astype(np.int32)).mean())
        print(f"  freq pipeline DONE: mean_abs_diff={diff:.3f}", flush=True)
        if diff < 2.0:
            print("  WARNING: Aging effect too weak or not applied. "
                  f"mean_abs_diff={diff:.3f} < 2.0", flush=True)
        model_status = "frequency-dsp"
        freq_variant = out
        ai_variant = out

    # Generate classical variant thumbnail for all methods (side-by-side compare)
    if _classical_aging_effect is not None and method != "classical":
        try:
            classical_out = _classical_aging_effect(image, intensity=strength)
        except Exception:
            classical_out = out
    else:
        classical_out = out

    gray_before = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_after = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
    energy_before = _round_energy(_fft_energy_components(gray_before)["energy"])
    energy_after = _round_energy(_fft_energy_components(gray_after)["energy"])
    apparent_age_before = _estimate_base_age(image)
    apparent_age_after = _apparent_age_after(apparent_age_before, mode, years, strength)

    try:
        encoded = _encode_base64_jpeg(out)
    except ValueError as exc:
        return _error(str(exc), 500)

    saved = _save_image(out, "aging")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return jsonify(
        {
            "ok": True,
            "image": encoded,
            "mode": mode,
            "method": method,
            "years": years,
            "deterministic": deterministic,
            "preserveEdges": preserve_edges,
            "hpf": hpf,
            "lpf": lpf,
            "modelStatus": model_status,
            "apparentAgeBefore": apparent_age_before,
            "apparentAgeAfter": apparent_age_after,
            "energyBefore": energy_before,
            "energyAfter": energy_after,
            "maskMethod": transform_meta.get("maskMethod"),
            "skinHighFrequencyEnergy": transform_meta.get("skinHighFrequencyEnergy"),
            "debugMasks": transform_meta.get("debugMasks"),
            "variants": {
                "frequency":  _thumbnail_b64(freq_variant),
                "ai":         _thumbnail_b64(ai_variant),
                "classical":  _thumbnail_b64(classical_out),
            },
            "saved": saved,
            "durationMs": round(elapsed_ms, 2),
        }
    )


# ---------------------------------------------------------------------------
# AI aging — Replicate image-editing helper
# ---------------------------------------------------------------------------
# Frequency mode (method="freq") uses the local classical DSP pipeline:
#   HPF amplification, CLAHE, hair greying — runs entirely locally.
# AI mode (method="ai") sends the image to the Replicate API for AI-based aging.
#   Requires REPLICATE_API_TOKEN in the .env file (never hardcoded).
#   Uses FLUX Kontext Pro on Replicate for high-quality image-to-image face aging.

# ---------------------------------------------------------------------------
# Replicate aging — FLUX Kontext Pro (image-to-image editing)
# ---------------------------------------------------------------------------

# Model identifier on Replicate — supports image-to-image editing with a prompt.
_REPLICATE_MODEL = "black-forest-labs/flux-kontext-pro"

# Prompt template; {degree} is substituted based on intensity.
_AGING_PROMPT_TEMPLATE = (
    "Make this person look {degree}. "
    "Add realistic facial wrinkles, aged skin texture, age spots, and gray hair. "
    "Keep the same identity, pose, background, and lighting. "
    "Do not change the face shape dramatically."
)


def call_replicate_aging(image_bytes: bytes, intensity: float = 0.5) -> bytes:
    """
    Sends a portrait to the Replicate FLUX Kontext Pro model and returns
    the aged image as raw bytes.

    The API token is read from the REPLICATE_API_TOKEN environment variable
    (loaded from .env via load_dotenv at startup).
    Do NOT hardcode secrets.

    Parameters
    ----------
    image_bytes : bytes
        JPEG-encoded input portrait.
    intensity : float
        0.0–1.0 scale controlling how much older the result should look.

    Returns
    -------
    bytes
        Raw bytes of the output image (JPEG or PNG, as returned by Replicate).

    Raises
    ------
    EnvironmentError : REPLICATE_API_TOKEN not set.
    RuntimeError     : replicate package unavailable, API error, or no image returned.
    """
    # Re-load .env in case the Flask reloader spawned a fresh process.
    _env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(str(_env_path), override=True)

    api_token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    print(f"[call_replicate_aging] REPLICATE_API_TOKEN present: {bool(api_token)}", flush=True)
    if not api_token:
        raise EnvironmentError(
            "REPLICATE_API_TOKEN is not configured. "
            "Add it to your .env file and restart the server."
        )

    if _replicate is None:
        raise RuntimeError(
            "replicate is not installed. Run: pip install replicate"
        )

    # Vary the aging degree based on the requested intensity.
    if intensity >= 0.7:
        degree = "20-30 years older"
    elif intensity >= 0.4:
        degree = "10-20 years older"
    else:
        degree = "5-10 years older"

    prompt = _AGING_PROMPT_TEMPLATE.format(degree=degree)

    # Encode the image as a base64 data URI so Replicate accepts it directly.
    img_b64 = base64.b64encode(image_bytes).decode()
    img_data_uri = f"data:image/jpeg;base64,{img_b64}"

    print(f"[call_replicate_aging] calling model: {_REPLICATE_MODEL}", flush=True)

    # Instantiate a client with an explicit token — never rely on global state.
    client = _replicate.Client(api_token=api_token)

    # Run the model.  The output is either a URL string or a list of URL strings.
    output = client.run(
        _REPLICATE_MODEL,
        input={
            "input_image": img_data_uri,
            "prompt": prompt,
        },
    )

    # Resolve the output URL.
    if isinstance(output, list):
        result_url = str(output[0])
    else:
        result_url = str(output)

    if not result_url:
        raise RuntimeError("Replicate returned an empty output URL.")

    print(f"[call_replicate_aging] output URL: {result_url[:80]}...", flush=True)

    # Download the resulting image.
    if _requests is None:
        raise RuntimeError("requests is not installed. Run: pip install requests")

    resp = _requests.get(result_url, timeout=120)
    resp.raise_for_status()

    result_bytes = resp.content
    if not result_bytes:
        raise RuntimeError("Replicate returned an empty image body.")

    return result_bytes


@app.post("/api/aging-ai")
def api_aging_ai():
    """
    AI aging endpoint — routes to Replicate FLUX Kontext Pro (image-to-image).

    This endpoint is separate from /api/aging (classical DSP / Frequency mode)
    so that the two modes can be tested, deployed, and rate-limited independently.

    Flow
    ----
    1. Read and validate the uploaded image.
    2. Check REPLICATE_API_TOKEN env var — return 503 immediately if missing.
    3. Forward the image to Replicate via call_replicate_aging().
    4. Decode the returned bytes, save to results/, return JSON with base64 image.

    Error handling
    --------------
    - Missing API key          → 503 + clear message.
    - API timeout / network    → 502 + message.
    - Invalid/empty response   → 502 + message.
    - Bad upload / image       → 400.
    """
    print("AGING AI ENDPOINT CALLED - REPLICATE MODE", flush=True)
    print("REPLICATE_API_TOKEN available:", bool(os.getenv("REPLICATE_API_TOKEN")), flush=True)
    started = time.perf_counter()

    # ── 1. Read uploaded image ────────────────────────────────────────────────
    raw_file = request.files.get("image")
    if raw_file is None:
        return _error("No image uploaded.", 400)

    # Keep raw bytes for forwarding; also decode for local metadata.
    image_bytes = raw_file.read()
    if not image_bytes:
        return _error("Uploaded image is empty.", 400)

    try:
        image = _read_image_from_bytes(image_bytes)
    except Exception as exc:
        return _error(f"Could not decode image: {exc}", 400)

    # ── 2. Check API token ────────────────────────────────────────────────────
    _key_val = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    print(f"[api_aging_ai] key check: {bool(_key_val)}", flush=True)
    if not _key_val:
        msg = "REPLICATE_API_TOKEN is not configured. Add it to your .env file."
        print(f"[api_aging_ai] returning 503: {msg}", flush=True)
        return jsonify({"ok": False, "error": msg}), 503

    try:
        strength = _clamp(float(request.form.get("strength", 0.5)), 0.0, 1.0)
        mode     = (request.form.get("mode", "aging") or "aging").strip().lower()
        years    = max(-50, min(50, int(float(request.form.get("years", 10)))))
    except ValueError:
        return _error("Invalid aging parameters.", 400)

    if mode not in {"aging", "deaging"}:
        return _error("Invalid mode. Use 'aging' or 'deaging'.", 400)

    # ── 3. Call Replicate image-editing API ───────────────────────────────────
    try:
        result_bytes = call_replicate_aging(image_bytes, intensity=strength)
    except EnvironmentError as exc:
        print(f"[api_aging_ai] EnvironmentError: {exc}", flush=True)
        return jsonify({"ok": False, "error": str(exc)}), 503
    except RuntimeError as exc:
        print(f"[api_aging_ai] RuntimeError: {exc}", flush=True)
        return jsonify({"ok": False, "error": str(exc)}), 502
    except Exception as exc:
        print(f"[api_aging_ai] Unexpected error: {exc}", flush=True)
        return jsonify({"ok": False, "error": f"Unexpected Replicate aging error: {exc}"}), 500

    # ── 4. Decode result, save, return ────────────────────────────────────────
    try:
        arr = np.frombuffer(result_bytes, dtype=np.uint8)
        out = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if out is None:
            raise ValueError("Could not decode AI aging result as an image.")
    except Exception as exc:
        return jsonify({"ok": False, "error": f"AI aging result is not a valid image: {exc}"}), 502

    # Resize to match input dimensions if the API changed them.
    if out.shape[:2] != image.shape[:2]:
        out = cv2.resize(out, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

    try:
        encoded = _encode_base64_jpeg(out)
    except ValueError as exc:
        return _error(str(exc), 500)

    saved      = _save_image(out, "aging-ai")
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    # Build image_url from saved path (relative to RESULTS_DIR).
    image_url = f"/results/{Path(saved).name}" if saved else None

    apparent_age_before = _estimate_base_age(image)
    apparent_age_after  = _apparent_age_after(apparent_age_before, mode, years, strength)

    return jsonify({
        "ok":                True,
        "image":             encoded,
        "image_url":         image_url,
        "method":            "ai",
        "mode":              mode,
        "years":             years,
        "modelStatus":       "replicate-flux-kontext-pro",
        "apparentAgeBefore": apparent_age_before,
        "apparentAgeAfter":  apparent_age_after,
        "saved":             saved,
        "durationMs":        round(elapsed_ms, 2),
    })


@app.post("/api/transform")
def api_transform():
    started = time.perf_counter()

    try:
        image = _read_image(request.files.get("image"))
    except ValueError as exc:
        return _error(str(exc), 400)

    try:
        smile       = float(request.form.get("smile",        0))
        brow        = float(request.form.get("brow",         0))
        lip         = float(request.form.get("lip",          0))
        slim        = float(request.form.get("slim",         0))
        face_fullness = float(request.form.get("face_fullness", 0))
    except ValueError:
        return _error("Invalid slider values.", 400)

    # Accept either [0..100] or normalised [0..1].
    def _norm(v: float) -> float:
        if abs(v) > 1.0:
            v = v / 100.0
        return float(max(-1.0, min(1.0, v)))

    smile         = _norm(smile)
    brow          = _norm(brow)
    lip           = _norm(lip)
    slim          = _norm(slim)
    face_fullness = _norm(face_fullness)

    warp_mode = (request.form.get("mode", "delaunay") or "delaunay").strip().lower()
    if warp_mode not in {"delaunay", "tps", "flow"}:
        warp_mode = "delaunay"
    if warp_mode == "flow":
        warp_mode = "delaunay"   # flow-field retired; map to stable Delaunay

    interp_name = (request.form.get("interpolation", "bicubic") or "bicubic").strip().lower()
    interp_flag = {
        "bicubic":  cv2.INTER_CUBIC,
        "bilinear": cv2.INTER_LINEAR,
        "nearest":  cv2.INTER_NEAREST,
    }.get(interp_name, cv2.INTER_CUBIC)
    save_debug = request.form.get("debug", "false").strip().lower() in {"1", "true", "yes"}
    show_grid = request.form.get("show_grid", "0").strip() in {"1", "true", "yes"}

    # ── Fast path: all sliders at zero → return original immediately ──────
    if smile == 0.0 and brow == 0.0 and lip == 0.0 and slim == 0.0 and face_fullness == 0.0:
        h_orig, w_orig = image.shape[:2]
        try:
            encoded_orig = _encode_base64_jpeg(image)
        except ValueError as exc:
            return _error(str(exc), 500)
        saved_orig = _save_image(image, "transform")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return jsonify({
            "ok": True,
            "image": encoded_orig,
            "width": int(w_orig),
            "height": int(h_orig),
            "method": warp_mode,
            "interpolation": interp_name,
            "landmarkSource": "none",
            "usedLandmarkCount": 0,
            "movedPoints": 0,
            "maxDisplacement": 0.0,
            "triangleCount": 0,
            "maskApplied": False,
            "saved": saved_orig,
            "debug": {},
            "durationMs": round(elapsed_ms, 2),
        })

    # Burn a grid into the input BEFORE warping so the TPS remap bends the
    # grid lines — giving a direct visual proof of non-linear deformation.
    if show_grid and abs(face_fullness) >= 1e-4:
        image = _draw_grid(image)

    result, debug_paths, n_lm, moved_pts, max_disp, tri_count, lm_source = _apply_expression_warp(
        image, smile, brow, lip, slim,
        warp_mode=warp_mode,
        save_debug=save_debug,
        interp=interp_flag,
        double_chin=face_fullness * 0.30,   # secondary chin coupling: 30 % of volume
        face_fullness=face_fullness,
    )

    h, w = result.shape[:2]
    try:
        encoded = _encode_base64_jpeg(result)
    except ValueError as exc:
        return _error(str(exc), 500)

    saved = _save_image(result, "transform")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return jsonify(
        {
            "ok":                True,
            "image":             encoded,
            "width":             int(w),
            "height":            int(h),
            "method":            warp_mode,
            "interpolation":     interp_name,
            "landmarkSource":    lm_source,
            "usedLandmarkCount": int(n_lm),
            "movedPoints":       int(moved_pts),
            "maxDisplacement":   round(float(max_disp), 2),
            "triangleCount":     int(tri_count),
            "maskApplied":       True,
            "saved":             saved,
            "debug":             debug_paths,
            "durationMs":        round(elapsed_ms, 2),
            "facialVolumeApplied": abs(face_fullness) >= 1e-4,
        }
    )


@app.post("/api/expression-transfer")
def api_expression_transfer():
    """
    Transfer the expression from a reference face image onto a target face image.
    Both faces must be detectable. Uses 68-point landmark displacement (Delaunay warping).
    """
    started = time.perf_counter()

    try:
        target = _read_image(request.files.get("target") or request.files.get("image"))
    except ValueError as exc:
        return _error(str(exc), 400)

    ref_file = request.files.get("reference")
    if ref_file is None or not ref_file.filename:
        return _error("Reference image is required.", 400)
    try:
        reference = _read_image(ref_file)
    except ValueError as exc:
        return _error(f"Reference image: {exc}", 400)

    try:
        strength = float(request.form.get("strength", 0.6))
        strength = _clamp(strength, 0.1, 1.0)
    except ValueError:
        strength = 0.6

    # ---------- Landmarks ----------
    ref_pts, ref_method = _get_68_pts_for_warp(reference)
    tgt_pts, tgt_method = _get_68_pts_for_warp(target)

    if len(ref_pts) < 68 or len(tgt_pts) < 68:
        return _error("Could not detect sufficient landmarks in one or both images.", 422)

    h, w = target.shape[:2]
    params = _estimate_expression_sliders_from_68(ref_pts, strength=strength)
    target_dst, moved_pts, max_disp, moved_indices = _build_reference_expression_dst_68(
        ref_pts, tgt_pts, strength, w, h
    )

    src_full = _add_boundary_points(tgt_pts, w, h)
    dst_full = _add_boundary_points(target_dst, w, h)
    tris = _triangle_indices_from_subdiv(src_full, w, h)

    src_f32 = target.astype(np.float32)
    warped_f32 = src_f32.copy()
    for (i, j, k) in tris:
        t_src = np.array([src_full[i], src_full[j], src_full[k]], dtype=np.float32)
        t_dst = np.array([dst_full[i], dst_full[j], dst_full[k]], dtype=np.float32)
        _warp_triangle(src_f32, warped_f32, t_src, t_dst, interp=cv2.INTER_CUBIC)
    warped = np.clip(warped_f32, 0, 255).astype(np.uint8)
    result = _blend_warped_face_mask(target, warped, target_dst, w, h)
    n_lm = len(tgt_pts)
    tri_count = len(tris)
    lm_source = f"target-{tgt_method}"

    try:
        encoded = _encode_base64_jpeg(result)
    except ValueError as exc:
        return _error(str(exc), 500)

    saved = _save_image(result, "expr-transfer")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return jsonify({
        "ok": True,
        "image": encoded,
        "width": int(w),
        "height": int(h),
        "refMethod": ref_method,
        "targetMethod": tgt_method,
        "landmarkSource": lm_source,
        "usedLandmarkCount": int(n_lm),
        "movedPoints": int(moved_pts),
        "maxDisplacement": round(float(max_disp), 2),
        "triangleCount": int(tri_count),
        "transferMode": "reference-expression-only",
        "expressionParams": {
            "smile": round(params["smile"], 4),
            "brow": round(params["brow"], 4),
            "lip": round(params["lip"], 4),
            "slim": 0.0,
        },
        "saved": saved,
        "durationMs": round(elapsed_ms, 2),
    })

    # ---------- Neutral reference: synthetic 68pts from detected bbox ----------
    det_ref = _detect_face_bbox(reference)
    if det_ref:
        rbx, rby, rbw, rbh, _ = det_ref
    else:
        rh, rw = reference.shape[:2]
        rbx, rby, rbw, rbh = int(rw * 0.18), int(rh * 0.12), int(rw * 0.64), int(rh * 0.76)
    neutral_ref = np.array(
        [[float(x), float(y)]
         for x, y in _synthetic_68_landmarks_from_bbox(rbx, rby, rbw, rbh,
                                                        reference.shape[1], reference.shape[0])],
        dtype=np.float32,
    )

    # ---------- Compute normalised displacements from reference → neutral ----------
    ref_min = ref_pts.min(axis=0)
    ref_max = ref_pts.max(axis=0)
    ref_face_w = max(float(ref_max[0] - ref_min[0]), 1.0)
    ref_face_h = max(float(ref_max[1] - ref_min[1]), 1.0)

    tgt_min = tgt_pts.min(axis=0)
    tgt_max = tgt_pts.max(axis=0)
    tgt_face_w = max(float(tgt_max[0] - tgt_min[0]), 1.0)
    tgt_face_h = max(float(tgt_max[1] - tgt_min[1]), 1.0)

    ref_disp_norm = ref_pts - neutral_ref
    ref_disp_norm[:, 0] /= ref_face_w
    ref_disp_norm[:, 1] /= ref_face_h

    # Apply scaled displacements to target landmarks
    target_dst = tgt_pts.copy()
    target_dst[:, 0] += ref_disp_norm[:, 0] * tgt_face_w * strength
    target_dst[:, 1] += ref_disp_norm[:, 1] * tgt_face_h * strength

    # Clamp displacement per-point to <= 15% of face size (prevent wild warping)
    face_size = max(tgt_face_w, tgt_face_h)
    disp_raw = target_dst - tgt_pts
    norms = np.linalg.norm(disp_raw, axis=1, keepdims=True)
    max_allowed = 0.15 * face_size
    overflow = (norms > max_allowed) & (norms > 0)
    if np.any(overflow):
        scale = np.where(overflow, max_allowed / (norms + 1e-9), 1.0)
        target_dst = tgt_pts + disp_raw * scale

    target_dst[:, 0] = np.clip(target_dst[:, 0], 1.0, float(w - 2))
    target_dst[:, 1] = np.clip(target_dst[:, 1], 1.0, float(h - 2))

    # ---------- Delaunay warp ----------
    src_full = _add_boundary_points(tgt_pts, w, h)
    dst_full = _add_boundary_points(target_dst, w, h)
    tris = _triangle_indices_from_subdiv(src_full, w, h)

    src_f32 = target.astype(np.float32)
    result = src_f32.copy()
    for (i, j, k) in tris:
        t_src = np.array([src_full[i], src_full[j], src_full[k]], dtype=np.float32)
        t_dst = np.array([dst_full[i], dst_full[j], dst_full[k]], dtype=np.float32)
        _warp_triangle(src_f32, result, t_src, t_dst, interp=cv2.INTER_CUBIC)
    result = np.clip(result, 0, 255).astype(np.uint8)

    try:
        encoded = _encode_base64_jpeg(result)
    except ValueError as exc:
        return _error(str(exc), 500)

    saved = _save_image(result, "expr-transfer")
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    return jsonify({
        "ok": True,
        "image": encoded,
        "width": int(w),
        "height": int(h),
        "refMethod": ref_method,
        "targetMethod": tgt_method,
        "triangleCount": len(tris),
        "saved": saved,
        "durationMs": round(elapsed_ms, 2),
    })


if __name__ == "__main__":
    import threading
    import webbrowser

    host = os.environ.get("DSP_HOST", "127.0.0.1")
    port = int(os.environ.get("DSP_PORT", "8081"))
    debug = (os.environ.get("DSP_DEBUG", "false") or "false").strip().lower() in {"1", "true", "yes", "on"}
    use_reloader = (os.environ.get("DSP_RELOADER", "false") or "false").strip().lower() in {"1", "true", "yes", "on"}

    # Automatically open the browser once the server is ready.
    # A short delay (1.2 s) ensures Flask has bound the socket before the
    # browser tries to connect.  The flag prevents double-open when the
    # reloader forks a child process (WERKZEUG_RUN_MAIN is set in the child).
    no_reloader_child = not use_reloader or os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    if not debug or no_reloader_child:
        url = f"http://{host}:{port}"
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
        print(f" * Opening browser at {url}")

    app.run(host=host, port=port, debug=debug, use_reloader=use_reloader)
