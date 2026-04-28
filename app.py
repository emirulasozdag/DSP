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
UPLOAD_DIR = ROOT / "uploads"
RESULTS_DIR = ROOT / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg"}


# Approximate mapping from MediaPipe FaceMesh indices to a dlib-style 68 landmark set.
MEDIAPIPE_TO_68 = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
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


def _make_deterministic_seed(image: np.ndarray, mode: str, strength: float, years: int, contrast: float, tone: float, noise: float, detail: float) -> int:
    # Build a stable seed from image content + parameters for reproducible demo outputs.
    digest = hashlib.sha256()
    digest.update(image.tobytes())
    digest.update(f"{mode}|{strength:.6f}|{years}|{contrast:.6f}|{tone:.6f}|{noise:.6f}|{detail:.6f}".encode("utf-8"))
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
    energy_map = magnitude ** 2

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


def _fft_age_transform(image: np.ndarray, mode: str, strength: float, years: int, detail: float):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2

    # Smaller sigma -> stronger high-pass emphasis.
    sigma_base = max(10.0, min(h, w) * 0.16)
    sigma = sigma_base * (1.15 - 0.55 * strength)
    low_pass = np.exp(-d2 / (2.0 * sigma * sigma))
    high_pass = 1.0 - low_pass

    spec = np.fft.fftshift(np.fft.fft2(gray))
    year_gain = min(1.0, abs(years) / 50.0)
    detail_gain = max(0.0, min(1.0, detail))

    if mode == "deaging":
        amp = 1.0 + 0.95 * strength * low_pass
        amp *= (1.0 - (0.45 + 0.35 * detail_gain) * strength * high_pass)
    else:
        amp = 1.0 + (0.55 + 0.30 * detail_gain) * strength * high_pass
        amp *= (1.0 - 0.25 * strength * low_pass * (0.7 + 0.3 * year_gain))

    mod_spec = spec * amp
    recon = np.fft.ifft2(np.fft.ifftshift(mod_spec)).real
    recon = np.clip(recon, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_ch = lab[:, :, 0]
    alpha = 0.25 + 0.55 * strength
    lab[:, :, 0] = np.clip((1.0 - alpha) * l_ch + alpha * recon, 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out


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


def _extract_landmarks_for_warp(image: np.ndarray):
    h, w = image.shape[:2]

    # Prefer MediaPipe FaceMesh for denser, semantically stable points.
    if FACE_MESH is not None:
        mp_points = _get_mediapipe_points(image)
        if mp_points:
            pts = []
            for p in mp_points:
                pts.append([float(p["x"]), float(p["y"])])

            groups = {
                "mouth_left": [61],
                "mouth_right": [291],
                "mouth_upper": [13],
                "mouth_lower": [14],
                "brow_left": [70, 63, 105],
                "brow_right": [300, 293, 334],
                "slim_left": [234, 172, 136],
                "slim_right": [454, 397, 365],
                "center": [1],
            }
            return np.array(pts, dtype=np.float32), groups

    # Robust fallback: synthetic 68 from detected bbox.
    det = _detect_face_bbox(image)
    if det is None:
        bx, by, bw, bh = int(w * 0.18), int(h * 0.12), int(w * 0.64), int(h * 0.76)
    else:
        bx, by, bw, bh, _ = det

    pts68 = _synthetic_68_landmarks_from_bbox(bx, by, bw, bh, w, h)
    pts = np.array([[float(x), float(y)] for x, y in pts68], dtype=np.float32)
    groups = {
        "mouth_left": [48],
        "mouth_right": [54],
        "mouth_upper": [51],
        "mouth_lower": [57],
        "brow_left": [17, 18, 19, 20, 21],
        "brow_right": [22, 23, 24, 25, 26],
        "slim_left": [1, 2, 3, 4, 5],
        "slim_right": [11, 12, 13, 14, 15],
        "center": [30],
    }
    return pts, groups


def _apply_group_shift(points: np.ndarray, idxs, dx: float = 0.0, dy: float = 0.0):
    for i in idxs:
        if 0 <= i < len(points):
            points[i, 0] += dx
            points[i, 1] += dy


def _build_dst_landmarks(src_points: np.ndarray, groups: dict, smile: float, brow: float, lip: float, slim: float, w: int, h: int):
    dst = src_points.copy()

    # Smile and lip widening: move mouth corners outward + slightly upward.
    smile_x = float(smile) * 0.055 * w
    smile_y = float(smile) * 0.030 * h
    lip_x = float(lip) * 0.045 * w
    lip_y = float(lip) * 0.012 * h
    _apply_group_shift(dst, groups["mouth_left"], dx=-(smile_x + lip_x), dy=-(smile_y))
    _apply_group_shift(dst, groups["mouth_right"], dx=(smile_x + lip_x), dy=-(smile_y))
    _apply_group_shift(dst, groups["mouth_upper"], dy=-lip_y)
    _apply_group_shift(dst, groups["mouth_lower"], dy=lip_y)

    # Brow raise.
    brow_y = float(brow) * 0.060 * h
    _apply_group_shift(dst, groups["brow_left"], dy=-brow_y)
    _apply_group_shift(dst, groups["brow_right"], dy=-brow_y)

    # Face slimming: move cheek/jaw points toward center line.
    center_x = float(np.mean([dst[i, 0] for i in groups["center"] if 0 <= i < len(dst)]))
    slim_gain = float(slim) * 0.18
    for i in groups["slim_left"]:
        if 0 <= i < len(dst):
            dst[i, 0] += (center_x - dst[i, 0]) * slim_gain
    for i in groups["slim_right"]:
        if 0 <= i < len(dst):
            dst[i, 0] += (center_x - dst[i, 0]) * slim_gain

    dst[:, 0] = np.clip(dst[:, 0], 0, w - 1)
    dst[:, 1] = np.clip(dst[:, 1], 0, h - 1)
    return dst


def _add_boundary_points(points: np.ndarray, w: int, h: int):
    boundary = np.array(
        [
            [0, 0], [w // 2, 0], [w - 1, 0],
            [0, h // 2], [w - 1, h // 2],
            [0, h - 1], [w // 2, h - 1], [w - 1, h - 1],
            [w // 4, 0], [(3 * w) // 4, 0],
            [w // 4, h - 1], [(3 * w) // 4, h - 1],
            [0, h // 4], [0, (3 * h) // 4],
            [w - 1, h // 4], [w - 1, (3 * h) // 4],
        ],
        dtype=np.float32,
    )
    return np.vstack([points, boundary])


def _triangle_indices_from_subdiv(points: np.ndarray, w: int, h: int):
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    pts_int = [(int(round(p[0])), int(round(p[1]))) for p in points]
    for x, y in pts_int:
        subdiv.insert((x, y))

    tri_list = subdiv.getTriangleList()
    if tri_list is None:
        return []

    point_dict = {}
    for idx, (x, y) in enumerate(pts_int):
        point_dict.setdefault((x, y), idx)

    def idx_of(px: float, py: float):
        key = (int(round(px)), int(round(py)))
        if key in point_dict:
            return point_dict[key]
        # Fallback: nearest point lookup.
        arr = points - np.array([px, py], dtype=np.float32)
        d2 = np.sum(arr * arr, axis=1)
        i = int(np.argmin(d2))
        return i if d2[i] < 4.0 else None

    tris = []
    for t in tri_list:
        x1, y1, x2, y2, x3, y3 = t
        if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h and 0 <= x3 < w and 0 <= y3 < h):
            continue
        i1 = idx_of(x1, y1)
        i2 = idx_of(x2, y2)
        i3 = idx_of(x3, y3)
        if i1 is None or i2 is None or i3 is None:
            continue
        if i1 == i2 or i2 == i3 or i1 == i3:
            continue
        tri = tuple(sorted((i1, i2, i3)))
        tris.append(tri)

    # Preserve unique triangles only.
    return list(dict.fromkeys(tris))


def _warp_triangle(src: np.ndarray, dst: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray):
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))

    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return

    t1_rect = np.array([[t_src[i][0] - r1[0], t_src[i][1] - r1[1]] for i in range(3)], dtype=np.float32)
    t2_rect = np.array([[t_dst[i][0] - r2[0], t_dst[i][1] - r2[1]] for i in range(3)], dtype=np.float32)

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), lineType=cv2.LINE_AA)

    src_patch = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    if src_patch.size == 0:
        return

    mat = cv2.getAffineTransform(t1_rect, t2_rect)
    warped = cv2.warpAffine(
        src_patch,
        mat,
        (r2[2], r2[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    dst_roi = dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    if dst_roi.size == 0:
        return

    dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = dst_roi * (1.0 - mask) + warped * mask


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
            "rawLandmarks": raw_mediapipe if response_mode == "both" else None,
            "reduced68": (
                _reduce_mediapipe_to_68(raw_mediapipe) if (used_method == "mediapipe" and (include_reduced68 or response_mode == "both")) else None
            ),
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

    fft_energy = _fft_energy_components(gray_trans)
    orig_spectrum = _fft_visuals_b64(gray_orig)
    trans_spectrum = _fft_visuals_b64(gray_trans)

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
            "energy": fft_energy["energy"],
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

    try:
        strength = float(request.form.get("strength", 0.5))
        years = int(float(request.form.get("years", 10)))
        contrast = float(request.form.get("contrast", 0.5))
        tone = float(request.form.get("tone", 0.5))
        noise = float(request.form.get("noise", 0.0))
        detail = float(request.form.get("detail", 0.5))
        deterministic = (request.form.get("deterministic", "true") or "true").strip().lower() in {"1", "true", "yes", "on"}
    except ValueError:
        return _error("Invalid aging parameters.", 400)

    strength = max(0.0, min(1.0, strength))
    contrast = max(0.0, min(1.0, contrast))
    tone = max(0.0, min(1.0, tone))
    noise = max(0.0, min(1.0, noise))
    detail = max(0.0, min(1.0, detail))
    years = max(-50, min(50, years))

    transformed = _fft_age_transform(image, mode, strength, years, detail)

    # Post-tone controls to keep UI sliders meaningful.
    t_shift = (tone - 0.5) * 26.0
    c_gain = 0.85 + contrast * 0.55
    out = transformed.astype(np.float32)
    out = (out - 128.0) * c_gain + 128.0
    out[:, :, 2] += t_shift * 0.8
    out[:, :, 0] -= t_shift * 0.6

    if noise > 0.001:
        if deterministic:
            seed = _make_deterministic_seed(image, mode, strength, years, contrast, tone, noise, detail)
            rng = np.random.default_rng(seed)
            grain = rng.normal(0.0, 18.0 * noise, size=out.shape).astype(np.float32)
        else:
            grain = np.random.normal(0.0, 18.0 * noise, size=out.shape).astype(np.float32)
        if mode == "deaging":
            grain *= 0.45
        out += grain

    out = np.clip(out, 0, 255).astype(np.uint8)

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
            "years": years,
            "deterministic": deterministic,
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
        brow = float(request.form.get("brow", 0))
        lip = float(request.form.get("lip", 0))
        slim = float(request.form.get("slim", 0))
    except ValueError:
        return _error("Invalid slider values.", 400)

    # Accept either [0..100] or normalized [0..1].
    def _norm(v: float) -> float:
        if abs(v) > 1.0:
            v = v / 100.0
        return float(max(-1.0, min(1.0, v)))

    smile = _norm(smile)
    brow = _norm(brow)
    lip = _norm(lip)
    slim = _norm(slim)

    h, w = image.shape[:2]
    src_pts, groups = _extract_landmarks_for_warp(image)
    dst_pts = _build_dst_landmarks(src_pts, groups, smile, brow, lip, slim, w, h)

    src_all = _add_boundary_points(src_pts, w, h)
    dst_all = _add_boundary_points(dst_pts, w, h)
    triangles = _triangle_indices_from_subdiv(src_all, w, h)

    output = image.astype(np.float32).copy()
    src_f = image.astype(np.float32)
    for i1, i2, i3 in triangles:
        t_src = np.float32([src_all[i1], src_all[i2], src_all[i3]])
        t_dst = np.float32([dst_all[i1], dst_all[i2], dst_all[i3]])
        _warp_triangle(src_f, output, t_src, t_dst)

    warped = np.clip(output, 0, 255).astype(np.uint8)
    # Mild bilateral pass reduces tiny triangle seam artifacts.
    warped = cv2.bilateralFilter(warped, d=5, sigmaColor=20, sigmaSpace=7)

    try:
        encoded = _encode_base64_jpeg(warped)
    except ValueError as exc:
        return _error(str(exc), 500)

    saved = _save_image(warped, "transform")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return jsonify(
        {
            "ok": True,
            "image": encoded,
            "width": int(w),
            "height": int(h),
            "saved": saved,
            "durationMs": round(elapsed_ms, 2),
        }
    )


if __name__ == "__main__":
    host = os.environ.get("DSP_HOST", "127.0.0.1")
    port = int(os.environ.get("DSP_PORT", "8080"))
    debug = (os.environ.get("DSP_DEBUG", "false") or "false").strip().lower() in {"1", "true", "yes", "on"}
    use_reloader = (os.environ.get("DSP_RELOADER", "false") or "false").strip().lower() in {"1", "true", "yes", "on"}
    app.run(host=host, port=port, debug=debug, use_reloader=use_reloader)
