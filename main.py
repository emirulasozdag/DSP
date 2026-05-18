"""
Real-Time Snapchat-Like Virtual Try-On
======================================

This desktop app reads webcam frames, tracks one face with MediaPipe Face Mesh,
and overlays selectable transparent PNG glasses and hats in real time.

DSP / image-processing concepts shown here:
  - landmark tracking: MediaPipe estimates stable facial points every frame
  - affine transforms: hats rotate and scale with head roll
  - perspective warping: glasses are trapezoid-warped for left/right yaw
  - alpha blending: transparent PNG pixels are mixed onto the camera frame
  - low-pass filtering: exponential smoothing reduces landmark jitter
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np


ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"

WINDOW_NAME = "Snapchat-Like DSP Virtual Try-On - Q/ESC to quit"
SMOOTHING_ALPHA = 0.32
CATALOG_WIDTH = 360
CARD_W = 92
CARD_H = 68
CARD_GAP = 14

GLASSES_WIDTH_FACTOR = 2.45
HAT_WIDTH_FACTOR = 1.55
HAT_Y_OFFSET_FACTOR = -0.45
HAT_BOTTOM_ALIGNMENT = 0.55

# MediaPipe Face Mesh landmarks used for stable measurements.
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
FOREHEAD = 10
FOREHEAD_CENTER = 151
LEFT_FOREHEAD = 109
RIGHT_FOREHEAD = 338
CHIN = 152
NOSE_TIP = 1
NOSE_BRIDGE_POINTS = (168, 6, 197)


@dataclass
class Accessory:
    name: str
    path: Path
    image: np.ndarray
    thumb: np.ndarray


@dataclass
class TryOnState:
    glasses_index: int = 0
    hat_index: int = 0
    show_glasses: bool = True
    show_hat: bool = True
    mirror: bool = False
    card_rects: list[tuple[int, int, int, int, str, int]] | None = None


def load_png_with_alpha(path: str | Path) -> np.ndarray:
    """Load a PNG as BGRA so OpenCV preserves transparent pixels."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        alpha = np.full(image.shape[:2] + (1,), 255, dtype=image.dtype)
        image = np.concatenate([image, alpha], axis=2)
    elif image.shape[2] != 4:
        raise ValueError(f"Expected 1, 3, or 4 channels in {path}, got {image.shape[2]}")

    return image


def overlay_transparent(
    background: np.ndarray,
    overlay: np.ndarray,
    x: int,
    y: int,
) -> np.ndarray:
    """
    Alpha-blend a BGRA overlay onto a BGR frame.

    Alpha blending mixes visible accessory pixels with the camera frame:
    output = alpha * accessory + (1 - alpha) * background.
    """
    if overlay.size == 0:
        return background

    bg_h, bg_w = background.shape[:2]
    ov_h, ov_w = overlay.shape[:2]

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + ov_w, bg_w)
    y2 = min(y + ov_h, bg_h)

    if x1 >= x2 or y1 >= y2:
        return background

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    overlay_crop = overlay[oy1:oy2, ox1:ox2]
    alpha = overlay_crop[:, :, 3:4].astype(np.float32) / 255.0
    overlay_rgb = overlay_crop[:, :, :3].astype(np.float32)
    background_crop = background[y1:y2, x1:x2].astype(np.float32)

    blended = alpha * overlay_rgb + (1.0 - alpha) * background_crop
    background[y1:y2, x1:x2] = blended.astype(np.uint8)
    return background


def rotate_and_scale_image(image: np.ndarray, angle: float, scale: float) -> np.ndarray:
    """
    Rotate and scale a BGRA image using a 2x3 affine transform.

    This keeps hats aligned with head roll while transparent borders stay clear.
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0 or scale <= 0:
        return image

    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = max(1, int((h * sin) + (w * cos)))
    new_h = max(1, int((h * cos) + (w * sin)))

    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]

    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def perspective_warp_glasses(image: np.ndarray, target_width: float, yaw: float) -> np.ndarray:
    """
    Resize and perspective-warp glasses for a light 3-D try-on effect.

    Yaw is estimated from nose position. As yaw increases, one side is slightly
    compressed and shifted so the glasses feel attached during left/right turns.
    """
    if target_width <= 1:
        return image

    scale = target_width / image.shape[1]
    target_height = max(1, int(image.shape[0] * scale))
    target_width_i = max(1, int(target_width))
    resized = cv2.resize(image, (target_width_i, target_height), interpolation=cv2.INTER_AREA)

    h, w = resized.shape[:2]
    yaw_clamped = float(np.clip(yaw, -0.35, 0.35))
    inset = abs(yaw_clamped) * 0.30 * w
    skew = abs(yaw_clamped) * 0.10 * h

    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    if yaw_clamped >= 0:
        dst = np.float32([
            [inset * 0.15, skew * 0.20],
            [w - 1 - inset, skew],
            [w - 1 - inset, h - 1 - skew],
            [inset * 0.15, h - 1 - skew * 0.20],
        ])
    else:
        dst = np.float32([
            [inset, skew],
            [w - 1 - inset * 0.15, skew * 0.20],
            [w - 1 - inset * 0.15, h - 1 - skew * 0.20],
            [inset, h - 1 - skew],
        ])

    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        resized,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def smooth_value(previous: Any, current: Any, alpha: float = SMOOTHING_ALPHA) -> Any:
    """
    Exponential moving average / low-pass filter.

    The filter reduces high-frequency landmark jitter while preserving normal
    head motion. A higher alpha reacts faster; a lower alpha is steadier.
    """
    if previous is None:
        return current

    if isinstance(current, dict):
        return {
            key: smooth_value(previous.get(key), value, alpha)
            for key, value in current.items()
        }

    prev_array = np.asarray(previous, dtype=np.float32)
    curr_array = np.asarray(current, dtype=np.float32)
    smoothed = (1.0 - alpha) * prev_array + alpha * curr_array

    if np.isscalar(current):
        return float(smoothed)
    return smoothed


def _landmark_to_point(landmarks: list[Any], index: int, width: int, height: int) -> np.ndarray:
    landmark = landmarks[index]
    return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)


def _mean_landmark_point(
    landmarks: list[Any],
    indices: tuple[int, ...],
    width: int,
    height: int,
) -> np.ndarray:
    points = [_landmark_to_point(landmarks, idx, width, height) for idx in indices]
    return np.mean(points, axis=0).astype(np.float32)


def get_face_measurements(
    landmarks: list[Any],
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    """
    Convert Face Mesh landmarks into stable try-on measurements.

    Glasses use eye corners plus nose bridge, not a rough face center. Hats use
    upper-face landmarks only as a reference and are then moved upward.
    """
    left_eye = (
        _landmark_to_point(landmarks, LEFT_EYE_OUTER, frame_width, frame_height)
        + _landmark_to_point(landmarks, LEFT_EYE_INNER, frame_width, frame_height)
    ) / 2.0
    right_eye = (
        _landmark_to_point(landmarks, RIGHT_EYE_INNER, frame_width, frame_height)
        + _landmark_to_point(landmarks, RIGHT_EYE_OUTER, frame_width, frame_height)
    ) / 2.0
    left_outer = _landmark_to_point(landmarks, LEFT_EYE_OUTER, frame_width, frame_height)
    right_outer = _landmark_to_point(landmarks, RIGHT_EYE_OUTER, frame_width, frame_height)
    nose_bridge = _mean_landmark_point(landmarks, NOSE_BRIDGE_POINTS, frame_width, frame_height)
    nose_tip = _landmark_to_point(landmarks, NOSE_TIP, frame_width, frame_height)
    left_cheek = _landmark_to_point(landmarks, LEFT_CHEEK, frame_width, frame_height)
    right_cheek = _landmark_to_point(landmarks, RIGHT_CHEEK, frame_width, frame_height)
    forehead = _mean_landmark_point(
        landmarks,
        (FOREHEAD, FOREHEAD_CENTER, LEFT_FOREHEAD, RIGHT_FOREHEAD),
        frame_width,
        frame_height,
    )
    chin = _landmark_to_point(landmarks, CHIN, frame_width, frame_height)

    raw_eye_center = (left_eye + right_eye) / 2.0
    eye_center = (raw_eye_center * 0.70) + (nose_bridge * 0.30)
    face_center = (left_cheek + right_cheek) / 2.0
    eye_vector = right_eye - left_eye
    cheek_vector = right_cheek - left_cheek

    eye_distance = float(np.linalg.norm(right_outer - left_outer))
    face_width = float(np.linalg.norm(cheek_vector))
    face_height = float(np.linalg.norm(chin - forehead))
    roll_angle = math.degrees(math.atan2(float(eye_vector[1]), float(eye_vector[0])))

    # Approximate yaw from nose shift relative to cheek center. Positive/negative
    # values drive small horizontal and perspective corrections for glasses.
    yaw = float((nose_tip[0] - face_center[0]) / max(face_width, 1.0))

    return {
        "left_eye": left_eye,
        "right_eye": right_eye,
        "eye_center": eye_center,
        "nose_bridge": nose_bridge,
        "forehead": forehead,
        "face_center": face_center,
        "eye_distance": eye_distance,
        "face_width": face_width,
        "face_height": face_height,
        "roll_angle": roll_angle,
        "yaw": yaw,
    }


def make_thumbnail(image: np.ndarray, width: int = 76, height: int = 48) -> np.ndarray:
    """Create a small BGR thumbnail with alpha composited onto a dark card."""
    thumb = np.zeros((height, width, 3), dtype=np.uint8)
    thumb[:] = (31, 35, 50)

    h, w = image.shape[:2]
    scale = min((width - 10) / max(w, 1), (height - 10) / max(h, 1))
    resized = cv2.resize(
        image,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    overlay_transparent(thumb, resized, x, y)
    return thumb


def load_catalog() -> tuple[list[Accessory], list[Accessory]]:
    """Load all catalog accessories once so the webcam loop stays fast."""
    glasses = []
    hats = []
    for idx in range(1, 4):
        path = ASSETS_DIR / f"glasses_{idx}.png"
        image = load_png_with_alpha(path)
        glasses.append(Accessory(f"Glasses {idx}", path, image, make_thumbnail(image)))

    for idx in range(1, 4):
        path = ASSETS_DIR / f"hat_{idx}.png"
        image = load_png_with_alpha(path)
        hats.append(Accessory(f"Hat {idx}", path, image, make_thumbnail(image)))

    return glasses, hats


def draw_label(
    image: np.ndarray,
    text: str,
    pos: tuple[int, int],
    scale: float = 0.58,
    color: tuple[int, int, int] = (232, 238, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_catalog_panel(
    frame_height: int,
    glasses: list[Accessory],
    hats: list[Accessory],
    state: TryOnState,
    fps: float,
    measurements: dict[str, Any] | None,
) -> np.ndarray:
    """Draw the right-side virtual try-on catalog panel."""
    panel = np.zeros((frame_height, CATALOG_WIDTH, 3), dtype=np.uint8)
    panel[:] = (14, 18, 30)

    cv2.rectangle(panel, (0, 0), (CATALOG_WIDTH - 1, frame_height - 1), (47, 56, 82), 1)
    draw_label(panel, "Virtual Try-On", (22, 38), 0.78, (255, 255, 255), 2)
    draw_label(panel, "Snapchat-like DSP demo", (22, 64), 0.46, (150, 163, 196), 1)

    selected_glasses = glasses[state.glasses_index].name if state.show_glasses else "Off"
    selected_hat = hats[state.hat_index].name if state.show_hat else "Off"
    draw_label(panel, f"FPS: {fps:4.1f}", (22, 102), 0.55, (125, 222, 255), 1)
    yaw_text = "No face"
    if measurements is not None:
        yaw_text = f"Yaw: {float(measurements['yaw']):+.2f}"
    draw_label(panel, yaw_text, (132, 102), 0.55, (172, 151, 255), 1)
    draw_label(panel, f"Glasses: {selected_glasses}", (22, 130), 0.52, (236, 240, 255), 1)
    draw_label(panel, f"Hat: {selected_hat}", (22, 154), 0.52, (236, 240, 255), 1)
    draw_label(panel, f"Mirror: {'On' if state.mirror else 'Off'}", (22, 178), 0.52, (236, 240, 255), 1)

    rects: list[tuple[int, int, int, int, str, int]] = []

    def draw_section(title: str, items: list[Accessory], kind: str, selected: int, y: int) -> int:
        draw_label(panel, title, (22, y), 0.66, (255, 255, 255), 2)
        y += 16
        for i, item in enumerate(items):
            x = 22 + i * (CARD_W + CARD_GAP)
            active = (kind == "glasses" and state.show_glasses and i == selected) or (
                kind == "hat" and state.show_hat and i == selected
            )
            fill = (29, 35, 54) if not active else (42, 62, 104)
            border = (79, 91, 127) if not active else (80, 190, 255)
            cv2.rectangle(panel, (x, y), (x + CARD_W, y + CARD_H), fill, -1)
            cv2.rectangle(panel, (x, y), (x + CARD_W, y + CARD_H), border, 2)
            thumb = item.thumb
            tx = x + (CARD_W - thumb.shape[1]) // 2
            ty = y + 8
            panel[ty:ty + thumb.shape[0], tx:tx + thumb.shape[1]] = thumb
            key = str(i + 1 if kind == "glasses" else i + 4)
            draw_label(panel, f"{key}", (x + 8, y + CARD_H - 10), 0.50, (194, 205, 255), 1)
            rects.append((x, y, CARD_W, CARD_H, kind, i))
        return y + CARD_H + 42

    y_next = draw_section("Glasses", glasses, "glasses", state.glasses_index, 204)
    draw_section("Hats", hats, "hat", state.hat_index, y_next)

    controls_y = max(frame_height - 78, y_next + 100)
    cv2.line(panel, (22, controls_y - 22), (CATALOG_WIDTH - 22, controls_y - 22), (47, 56, 82), 1)
    draw_label(panel, "Controls", (22, controls_y), 0.64, (255, 255, 255), 2)
    controls = [
        "1-3 Glasses   4-6 Hats",
        "G/H Toggle    R Mirror",
        "0 Clear       Click cards",
        "Q/ESC Quit",
    ]
    for row, text in enumerate(controls):
        draw_label(panel, text, (22, controls_y + 24 + row * 20), 0.50, (176, 188, 220), 1)

    state.card_rects = rects
    return panel


def draw_status(frame: np.ndarray, fps: float, measurements: dict[str, Any] | None) -> None:
    """Draw compact status text on the live preview."""
    if measurements is None:
        text = "No face detected"
    else:
        text = f"FPS {fps:4.1f} | yaw {float(measurements['yaw']):+.2f}"

    cv2.putText(frame, text, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 248, 255), 1, cv2.LINE_AA)


def apply_glasses(frame: np.ndarray, glasses: np.ndarray, measurements: dict[str, Any]) -> None:
    """Warp and place glasses using eye corners, nose bridge, roll, and yaw."""
    eye_distance = max(float(measurements["eye_distance"]), 1.0)
    roll_angle = float(measurements["roll_angle"])
    yaw = float(measurements["yaw"])
    eye_center = np.asarray(measurements["eye_center"], dtype=np.float32)
    nose_bridge = np.asarray(measurements["nose_bridge"], dtype=np.float32)

    target_width = eye_distance * GLASSES_WIDTH_FACTOR
    warped = perspective_warp_glasses(glasses, target_width, yaw)
    transformed = rotate_and_scale_image(warped, roll_angle, 1.0)

    anchor = (eye_center * 0.62) + (nose_bridge * 0.38)
    yaw_shift = yaw * eye_distance * 0.28
    x = int(anchor[0] - transformed.shape[1] / 2 + yaw_shift)
    y = int(anchor[1] - transformed.shape[0] * 0.52)
    overlay_transparent(frame, transformed, x, y)


def apply_hat(frame: np.ndarray, hat: np.ndarray, measurements: dict[str, Any]) -> None:
    """Place the hat above the forehead so the bottom edge reaches the hairline."""
    face_width = max(float(measurements["face_width"]), 1.0)
    face_height = max(float(measurements["face_height"]), face_width * 1.15)
    roll_angle = float(measurements["roll_angle"])
    forehead = np.asarray(measurements["forehead"], dtype=np.float32)

    target_width = face_width * HAT_WIDTH_FACTOR
    scale = target_width / hat.shape[1]
    transformed = rotate_and_scale_image(hat, roll_angle, scale)

    x = int(forehead[0] - transformed.shape[1] / 2)
    y = int(
        forehead[1]
        - transformed.shape[0] * HAT_BOTTOM_ALIGNMENT
        + face_height * HAT_Y_OFFSET_FACTOR
    )
    overlay_transparent(frame, transformed, x, y)


def handle_key(key: int, state: TryOnState) -> bool:
    """Return True when the app should quit."""
    if key in (27, ord("q"), ord("Q")):
        return True
    if key in (ord("1"), ord("2"), ord("3")):
        state.glasses_index = key - ord("1")
        state.show_glasses = True
    elif key in (ord("4"), ord("5"), ord("6")):
        state.hat_index = key - ord("4")
        state.show_hat = True
    elif key in (ord("g"), ord("G")):
        state.show_glasses = not state.show_glasses
    elif key in (ord("h"), ord("H")):
        state.show_hat = not state.show_hat
    elif key in (ord("r"), ord("R")):
        state.mirror = not state.mirror
    elif key == ord("0"):
        state.show_glasses = False
        state.show_hat = False
    return False


def on_mouse(event: int, x: int, y: int, _flags: int, state: TryOnState) -> None:
    """Select catalog items by clicking their cards."""
    if event != cv2.EVENT_LBUTTONDOWN or not state.card_rects:
        return
    # Card rectangles are in the right-side panel coordinate system.
    panel_x = x - (state.card_rects_frame_offset if hasattr(state, "card_rects_frame_offset") else 0)
    for rx, ry, rw, rh, kind, index in state.card_rects:
        if rx <= panel_x <= rx + rw and ry <= y <= ry + rh:
            if kind == "glasses":
                state.glasses_index = index
                state.show_glasses = True
            else:
                state.hat_index = index
                state.show_hat = True
            return


def main() -> None:
    glasses_catalog, hat_catalog = load_catalog()
    state = TryOnState(card_rects=[])

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam with cv2.VideoCapture(0).")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, state)

    smoothed: dict[str, Any] | None = None
    previous_tick = cv2.getTickCount()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read from webcam.")
                break

            if state.mirror:
                frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            state.card_rects_frame_offset = frame_w

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            current_measurements = None
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                current_measurements = get_face_measurements(landmarks, frame_w, frame_h)
                smoothed = smooth_value(smoothed, current_measurements, SMOOTHING_ALPHA)

                if state.show_glasses:
                    apply_glasses(frame, glasses_catalog[state.glasses_index].image, smoothed)
                if state.show_hat:
                    apply_hat(frame, hat_catalog[state.hat_index].image, smoothed)

            now = cv2.getTickCount()
            dt = (now - previous_tick) / cv2.getTickFrequency()
            previous_tick = now
            if dt > 0:
                instant_fps = 1.0 / dt
                fps = instant_fps if fps == 0.0 else (0.88 * fps + 0.12 * instant_fps)

            draw_status(frame, fps, smoothed if current_measurements is not None else None)
            panel = draw_catalog_panel(
                frame_h,
                glasses_catalog,
                hat_catalog,
                state,
                fps,
                smoothed if current_measurements is not None else None,
            )
            display = np.hstack([frame, panel])
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if handle_key(key, state):
                break
    finally:
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
