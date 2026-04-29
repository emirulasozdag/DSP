from __future__ import annotations

import base64
import hashlib
import os
import time
import uuid
import traceback
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from skimage.metrics import structural_similarity

try:
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover - optional dependency path
    mp = None

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
ALLOWED_AGING_METHODS = {"freq", "ai"}
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
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.abs(spectrum)
    # Normalize FFT energy with Parseval scaling so values are comparable
    # across image sizes instead of growing with the square of pixel count.
    norm = float(gray.shape[0] * gray.shape[1]) ** 2
    energy_map = (magnitude ** 2) / max(norm, 1.0)

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = max(8, min(h, w) // 8)
    low_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius

    low_energy = float(np.sum(energy_map[low_mask]))
    high_energy = float(np.sum(energy_map[~low_mask]))
    total_energy = low_energy + high_energy
    hf_lf_ratio = float(high_energy / (low_energy + 1e-9))

    return {
        "spectrum": spectrum,
        "magnitude": magnitude,
        "energyMap": energy_map,
        "energy": {
            "total": total_energy,
            "high": high_energy,
            "low": low_energy,
            "hfLfRatio": hf_lf_ratio,
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

    protect = np.zeros((h, w), dtype=np.uint8)
    if not _fill_feature_masks_from_landmarks(protect, raw_points):
        _fill_feature_masks(protect, (x, y, bw, bh))

    if preserve_edges:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 150)
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, edge_kernel, iterations=1)
        protect = cv2.max(protect, edges)

    blur_size = _odd_kernel(min(h, w) // 12, 15)
    skin_float = cv2.GaussianBlur(skin_in_face.astype(np.float32) / 255.0, (blur_size, blur_size), 0)
    face_float = cv2.GaussianBlur(face_mask.astype(np.float32) / 255.0, (blur_size, blur_size), 0)
    skin_float = np.clip(np.maximum(skin_float, face_float * 0.52), 0.0, 1.0)

    protect_size = _odd_kernel(min(h, w) // 32, 5)
    protected_float = cv2.GaussianBlur(protect.astype(np.float32) / 255.0, (protect_size, protect_size), 0)
    protected_float = np.clip(protected_float, 0.0, 1.0)

    if preserve_edges:
        processing_float = skin_float * (1.0 - 0.82 * protected_float)
    else:
        processing_float = skin_float
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
    h, w = shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = radius / (np.sqrt(cy * cy + cx * cx) + 1e-6)

    is_high = filter_name.endswith("_hpf")
    low_cutoffs = {
        "gaussian_lpf": 0.28,
        "butterworth_lpf": 0.20,
        "ideal_lpf": 0.13,
        "gaussian_hpf": 0.22,
        "butterworth_hpf": 0.17,
        "ideal_hpf": 0.12,
    }
    cutoff = low_cutoffs.get(filter_name, 0.20)

    if filter_name.startswith("gaussian"):
        low = np.exp(-(radius ** 2) / (2.0 * cutoff * cutoff))
    elif filter_name.startswith("butterworth"):
        low = 1.0 / (1.0 + (radius / (cutoff + 1e-6)) ** 4)
    else:
        low = (radius <= cutoff).astype(np.float32)

    return (1.0 - low if is_high else low).astype(np.float32)


def _frequency_reconstruct(channel: np.ndarray, filter_name: str) -> np.ndarray:
    spec = np.fft.fftshift(np.fft.fft2(channel.astype(np.float32)))
    mask = _frequency_mask(channel.shape, filter_name)
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

    layer = cv2.GaussianBlur(layer, (3, 3), 0)
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

    layer = cv2.GaussianBlur(layer, (3, 3), 0)
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

    layer = cv2.GaussianBlur(layer, (_odd_kernel(max(5, int(min(w, h) * 0.012))), _odd_kernel(max(5, int(min(w, h) * 0.012)))), 0)
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
) -> tuple[np.ndarray, dict]:
    seed = _make_deterministic_seed(image, mode, strength, years, contrast, tone, noise, detail, hpf, lpf, preserve_edges, semantic_boost)
    rng = np.random.default_rng(seed) if deterministic else np.random.default_rng()
    masks = _build_age_region_masks(image, preserve_edges=preserve_edges)
    skin = masks["skinMask"].astype(np.float32)
    processing = masks["processingMask"].astype(np.float32)
    protected = masks["protectedMask"].astype(np.float32)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_orig = lab[:, :, 0].copy()
    l_mod = l_orig.copy()
    intensity = _age_intensity(strength, years, semantic_boost)

    if mode == "aging":
        high = np.clip(_frequency_reconstruct(l_orig, hpf), -45.0, 45.0)
        hpf_gain = {"gaussian_hpf": 0.85, "butterworth_hpf": 1.08, "ideal_hpf": 1.34}.get(hpf, 1.0)
        detail_gain = 0.70 + detail * 0.95
        l_mod += processing * high * (0.78 + 1.58 * intensity) * hpf_gain * detail_gain

        wrinkle_layer = _generate_wrinkle_layer(l_orig.shape, masks["faceBox"], rng, semantic_boost, masks.get("landmarks"))
        wrinkle_gain = (9.0 + 31.0 * intensity) * (0.58 + detail * 0.46)
        l_mod += processing * wrinkle_layer * wrinkle_gain

        shadow_layer = _generate_age_shadow_layer(l_orig.shape, masks["faceBox"], masks.get("landmarks"), rng)
        l_mod += skin * shadow_layer * (6.0 + 24.0 * intensity)

        clahe = cv2.createCLAHE(clipLimit=1.35 + 1.45 * intensity, tileGridSize=(8, 8))
        local_contrast = clahe.apply(np.clip(l_orig, 0, 255).astype(np.uint8)).astype(np.float32)
        l_mod = l_mod * (1.0 - processing * 0.13 * intensity) + local_contrast * (processing * 0.13 * intensity)
    else:
        low = _frequency_reconstruct(l_orig, lpf)
        bilateral = cv2.bilateralFilter(
            np.clip(l_orig, 0, 255).astype(np.uint8),
            d=9,
            sigmaColor=34 + 54 * strength,
            sigmaSpace=34 + 44 * strength,
        ).astype(np.float32)
        smooth = low * 0.46 + bilateral * 0.54
        lpf_gain = {"gaussian_lpf": 0.78, "butterworth_lpf": 1.00, "ideal_lpf": 1.24}.get(lpf, 1.0)
        smooth_strength = _clamp(intensity * (0.56 + (1.0 - detail) * 0.30) * lpf_gain, 0.0, 0.88)
        l_mod = l_mod * (1.0 - processing * smooth_strength) + smooth * (processing * smooth_strength)

        if preserve_edges:
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
        processed += grain * skin[:, :, None]

    blend = np.clip(skin * (0.70 + 0.30 * strength), 0.0, 1.0)
    if preserve_edges:
        blend *= 1.0 - protected * 0.68
    out = image.astype(np.float32) * (1.0 - blend[:, :, None]) + processed * blend[:, :, None]

    if preserve_edges:
        restore = protected * np.clip(0.48 + 0.25 * (1.0 - strength), 0.0, 0.72)
        out = out * (1.0 - restore[:, :, None]) + image.astype(np.float32) * restore[:, :, None]

    if mode == "aging" and intensity > 0.1:
        out = _apply_age_hair_graying(out, image, masks, intensity, rng)

    meta = {
        "maskMethod": masks["maskMethod"],
        "skinHighFrequencyEnergy": round(_masked_high_frequency_energy(np.clip(out, 0, 255).astype(np.uint8), masks["skinMask"]), 4),
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


# ── Legacy point-displacement helpers (kept for backward compatibility) ────────

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
        max_pull = 0.13 * face_w
        for side_key in ("jaw_left", "jaw_right"):
            for i in groups.get(side_key, []):
                if 0 <= i < len(src_points):
                    raw_pull = (ctr_x - src_points[i, 0]) * sl * 0.20
                    pull_x = float(np.clip(raw_pull, -max_pull, max_pull))
                    # Inject a tiny point displacement then spread with falloff
                    sigma = 0.10 * ref
                    w_arr = _falloff_weights(src_points, [i], sigma)
                    displacements[:, 0] += pull_x * w_arr
        # Chin: very subtle upward tuck (avoid an excessively sharp chin)
        chin_pull = sl * 0.010 * face_h
        chin_sigma = 0.08 * ref
        chin_idxs = [i for i in groups.get("chin", []) if 0 <= i < len(src_points)]
        if chin_idxs:
            w_arr_chin = _falloff_weights(src_points, chin_idxs, chin_sigma)
            displacements[:, 1] -= chin_pull * w_arr_chin

    # Global displacement clamp (prevents wild deformation at slider extremes)
    max_disp = 0.15 * ref
    disp_mag = np.linalg.norm(displacements, axis=1, keepdims=True)
    scale = np.where(disp_mag > max_disp, max_disp / (disp_mag + 1e-8), 1.0)
    displacements *= scale

    dst = src_points + displacements
    dst[:, 0] = np.clip(dst[:, 0], 1, w - 2)
    dst[:, 1] = np.clip(dst[:, 1], 1, h - 2)
    return dst


# ── Delaunay triangulation helpers ──────────────────────────────────────────

def _add_boundary_points(points: np.ndarray, w: int, h: int) -> np.ndarray:
    """Dense border ring so the Delaunay mesh covers the full canvas without gaps."""
    n = 8  # points per border edge
    xs = np.linspace(0, w - 1, n)
    ys = np.linspace(0, h - 1, n)
    bnd: list = []
    for x in xs:
        bnd.append([x, 0])
        bnd.append([x, h - 1])
    for y in ys[1:-1]:
        bnd.append([0, y])
        bnd.append([w - 1, y])
    return np.vstack([points, np.array(bnd, dtype=np.float32)])


def _triangle_indices_from_subdiv(points: np.ndarray, w: int, h: int) -> list:
    """Compute Delaunay triangles; return list of (i, j, k) index triples."""
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
    """Affine-warp one Delaunay triangle from src into dst with feathered mask."""
    area_src = abs(_triangle_area(t_src))
    area_dst = abs(_triangle_area(t_dst))
    # Skip degenerate, tiny, or badly distorted triangles
    if area_src < 4.0 or area_dst < 4.0:
        return
    if area_src > 0 and area_dst / area_src > 5.0:
        return

    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))
    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return

    t1_rect = np.float32([[t_src[i][0] - r1[0], t_src[i][1] - r1[1]] for i in range(3)])
    t2_rect = np.float32([[t_dst[i][0] - r2[0], t_dst[i][1] - r2[1]] for i in range(3)])

    # Anti-aliased fill mask, then 1-pixel feather to reduce seams
    mask = np.zeros((r2[3], r2[2]), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), 1.0, lineType=cv2.LINE_AA)
    if mask.shape[0] > 4 and mask.shape[1] > 4:
        mask = cv2.GaussianBlur(mask, (3, 3), 0.6)

    src_patch = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    if src_patch.size == 0:
        return

    mat = cv2.getAffineTransform(t1_rect, t2_rect)
    warped = cv2.warpAffine(
        src_patch, mat, (r2[2], r2[3]),
        flags=interp,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    dst_roi = dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    if dst_roi.shape[:2] != warped.shape[:2]:
        return

    mask3 = mask[:, :, None]
    dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = (
        dst_roi * (1.0 - mask3) + warped * mask3
    )


# ── Thin Plate Spline (TPS) warping ─────────────────────────────────────────

def _tps_build(dst_ctrl: np.ndarray, src_ctrl: np.ndarray):
    """
    Solve TPS coefficients for the inverse mapping (dst → src), so that
    cv2.remap can sample source pixels for every destination pixel.
    Returns (W, A, ctrl) or None on failure.
    """
    n = len(dst_ctrl)
    if n < 4:
        return None
    ctrl = dst_ctrl.astype(np.float64)
    out  = src_ctrl.astype(np.float64)

    # TPS kernel matrix K
    K = np.zeros((n, n), np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                r2 = float(np.sum((ctrl[i] - ctrl[j]) ** 2))
                K[i, j] = r2 * np.log(max(r2, 1e-14))

    P = np.column_stack([np.ones(n), ctrl])          # (n, 3)
    Z = np.zeros((3, 3), np.float64)
    L = np.block([[K, P], [P.T, Z]])                 # (n+3, n+3)
    L[:n, :n] += np.eye(n) * 1e-4                   # regularisation

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


def _tps_remap(image: np.ndarray, src_ctrl: np.ndarray, dst_ctrl: np.ndarray,
               subsample: int = 4) -> np.ndarray:
    """
    Warp image using TPS.  Evaluates the dense inverse field on a coarser grid
    then bicubically upsamples it before calling cv2.remap.
    """
    h, w = image.shape[:2]
    result = _tps_build(dst_ctrl, src_ctrl)
    if result is None:
        return image.copy()
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

    return cv2.remap(image, map_x, map_y,
                     interpolation=cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REFLECT_101)


# ── Main expression warp orchestrator (v3) ────────────────────────────────────

def _apply_expression_warp(
    image: np.ndarray,
    smile: float,
    brow: float,
    lip: float,
    slim: float,
    warp_mode: str = "flow",   # parameter kept for API compatibility
    save_debug: bool = False,
) -> tuple:
    """
    Muscle-based dense flow-field expression warping.

    Pipeline:
      1. Build per-pixel inverse displacement field from anatomical muscle forces.
      2. Lock identity-critical regions (nose tip, eye centres).
      3. Smooth with a wide Gaussian — eliminates all spatial discontinuities.
      4. Clamp extreme displacements.
      5. Apply via cv2.remap — no triangle seams, no alpha blending.
    """
    h, w = image.shape[:2]

    if smile == 0.0 and brow == 0.0 and lip == 0.0 and slim == 0.0:
        return image.copy(), {}

    src_pts, groups = _extract_landmarks_for_warp(image)

    # 1. Muscle-driven dense flow field
    flow = _build_muscle_flow(h, w, src_pts, groups, smile, brow, lip, slim)

    # 2. Identity preservation
    flow = _lock_identity_regions(flow, src_pts, groups)

    # 2.5 Hull masking — zero the flow outside the face convex hull so that
    # background pixels keep zero displacement (no background colour bleed).
    hull_pts_warp = cv2.convexHull(src_pts.astype(np.int32))
    hull_f = np.zeros((h, w), dtype=np.float32)
    cv2.fillConvexPoly(hull_f, hull_pts_warp, 1.0)
    kh = max(21, int(min(h, w) * 0.07)) | 1
    hull_f = cv2.GaussianBlur(hull_f, (kh, kh), kh * 0.35)
    flow[:, :, 0] *= hull_f
    flow[:, :, 1] *= hull_f

    # 3. Localised Gaussian smoothing — smooths the sharp Gaussian edges at
    #    landmark boundaries without spreading into adjacent regions.
    #    ~5% of image size keeps effects visible but still locally confined.
    ksize = max(17, int(min(h, w) * 0.05)) | 1
    sigma = float(ksize) * 0.42
    flow[:, :, 0] = cv2.GaussianBlur(flow[:, :, 0], (ksize, ksize), sigma)
    flow[:, :, 1] = cv2.GaussianBlur(flow[:, :, 1], (ksize, ksize), sigma)

    # 4. Anti-artifact displacement clamp
    mag = np.hypot(flow[:, :, 0], flow[:, :, 1])
    max_disp = min(h, w) * 0.13
    over = mag > max_disp
    if over.any():
        scale = np.where(over, max_disp / (mag + 1e-8), 1.0)
        flow[:, :, 0] *= scale
        flow[:, :, 1] *= scale

    # 5. Inverse remap: dst(x, y) ← src(x + flow_x, y + flow_y)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    map_x = np.clip(xx + flow[:, :, 0], 0.0, w - 1.0)
    map_y = np.clip(yy + flow[:, :, 1], 0.0, h - 1.0)
    result = cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    debug_paths: dict = {}
    if save_debug:
        mag_u8  = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mag_vis = cv2.applyColorMap(mag_u8, cv2.COLORMAP_JET)
        debug_paths["flow_magnitude"] = _save_image(mag_vis, "debug_flow_magnitude")
        vis_lm = image.copy()
        for pt in src_pts:
            cv2.circle(vis_lm, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
        debug_paths["landmarks"] = _save_image(vis_lm, "debug_landmarks")
        debug_paths["before_after"] = _save_image(
            np.hstack([image, result]), "debug_before_after"
        )

    return result, debug_paths


@app.get("/")
def index_page():
    return send_from_directory(ROOT, "index.html")


@app.get("/style.css")
def style_page():
    return send_from_directory(ROOT, "style.css")


@app.get("/app.js")
def script_page():
    return send_from_directory(ROOT, "app.js")


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
        return _error("Invalid method. Use 'freq' or 'ai'.", 400)

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
    except ValueError:
        return _error("Invalid aging parameters.", 400)

    strength = _clamp(strength, 0.0, 1.0)
    contrast = _clamp(contrast, 0.0, 1.0)
    tone = _clamp(tone, 0.0, 1.0)
    noise = _clamp(noise, 0.0, 1.0)
    detail = _clamp(detail, 0.0, 1.0)
    years = max(-50, min(50, years))

    if method == "ai":
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
        freq_variant = out
        ai_variant = out
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
        )
        model_status = "frequency-dsp"
        freq_variant = out
        ai_variant = out

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
            "variants": {
                "frequency": _thumbnail_b64(freq_variant),
                "ai": _thumbnail_b64(ai_variant),
            },
            "saved": saved,
            "durationMs": round(elapsed_ms, 2),
        }
    )


@app.post("/api/transform")
def api_transform():
    started = time.perf_counter()

    try:
        image = _read_image(request.files.get("image"))
    except ValueError as exc:
        return _error(str(exc), 400)

    try:
        smile = float(request.form.get("smile", 0))
        brow  = float(request.form.get("brow",  0))
        lip   = float(request.form.get("lip",   0))
        slim  = float(request.form.get("slim",  0))
    except ValueError:
        return _error("Invalid slider values.", 400)

    # Accept either [0..100] or normalised [0..1].
    def _norm(v: float) -> float:
        if abs(v) > 1.0:
            v = v / 100.0
        return float(max(-1.0, min(1.0, v)))

    smile = _norm(smile)
    brow  = _norm(brow)
    lip   = _norm(lip)
    slim  = _norm(slim)

    warp_mode = (request.form.get("mode", "flow") or "flow").strip().lower()
    if warp_mode not in {"delaunay", "tps", "flow"}:
        warp_mode = "flow"

    save_debug = request.form.get("debug", "false").strip().lower() in {"1", "true", "yes"}

    # ── Fast path: all sliders at zero → return original immediately ──────
    if smile == 0.0 and brow == 0.0 and lip == 0.0 and slim == 0.0:
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
            "warpMode": warp_mode,
            "saved": saved_orig,
            "debug": {},
            "durationMs": round(elapsed_ms, 2),
        })

    result, debug_paths = _apply_expression_warp(
        image, smile, brow, lip, slim,
        warp_mode=warp_mode,
        save_debug=save_debug,
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
            "ok":        True,
            "image":     encoded,
            "width":     int(w),
            "height":    int(h),
            "warpMode":  warp_mode,
            "saved":     saved,
            "debug":     debug_paths,
            "durationMs": round(elapsed_ms, 2),
        }
    )


if __name__ == "__main__":
    host = os.environ.get("DSP_HOST", "127.0.0.1")
    port = int(os.environ.get("DSP_PORT", "8081"))
    debug = (os.environ.get("DSP_DEBUG", "false") or "false").strip().lower() in {"1", "true", "yes", "on"}
    use_reloader = (os.environ.get("DSP_RELOADER", "false") or "false").strip().lower() in {"1", "true", "yes", "on"}
    app.run(host=host, port=port, debug=debug, use_reloader=use_reloader)
