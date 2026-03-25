import os
import io
import csv
import base64
import uuid
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from flask import Flask, render_template, request, jsonify, send_file
from scipy.fft import fft2, ifft2, fftshift
from skimage.metrics import structural_similarity as ssim
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
STANDARD_SIZE = (512, 512)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')


def allowed_file(filename):
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    return ext in ALLOWED_EXTENSIONS


def encode_image(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


def decode_image(data_uri):
    if ',' in data_uri:
        data_uri = data_uri.split(',')[1]
    img_bytes = base64.b64decode(data_uri)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


# ─── FR-3.1: Image Input & Preprocessing ───

def validate_and_preprocess(image):
    """FR-3.1.2 to FR-3.1.7: Validate, detect face, crop, normalize, resize."""
    if image is None:
        return None, None, "Invalid image data"

    h, w = image.shape[:2]
    if h < 50 or w < 50:
        return None, None, "Image resolution too low"

    # FR-3.1.4: Detect facial region
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    if len(faces) == 0:
        # Use full image if no face detected
        face_region = image.copy()
        face_box = (0, 0, w, h)
    else:
        x, y, fw, fh = faces[0]
        # Add padding
        pad = int(max(fw, fh) * 0.3)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + fw + pad)
        y2 = min(h, y + fh + pad)
        face_region = image[y1:y2, x1:x2]
        face_box = (x1, y1, x2 - x1, y2 - y1)

    # FR-3.1.5 & FR-3.1.7: Normalize and resize
    resized = cv2.resize(face_region, STANDARD_SIZE, interpolation=cv2.INTER_AREA)

    return resized, face_box, None


# ─── FR-3.2: Facial Landmark Detection ───

def detect_landmarks(image):
    """FR-3.2.1 to FR-3.2.3: Detect and return 478 facial landmarks using MediaPipe Tasks API."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    landmarks = []

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.face_landmarks:
            for lm in result.face_landmarks[0]:
                landmarks.append({
                    'x': int(lm.x * w),
                    'y': int(lm.y * h),
                    'nx': lm.x,
                    'ny': lm.y
                })

    return landmarks


def draw_landmarks(image, landmarks, indices=None):
    """Draw landmark points on image."""
    img = image.copy()
    for i, lm in enumerate(landmarks):
        if indices is None or i in indices:
            cv2.circle(img, (lm['x'], lm['y']), 1, (0, 255, 0), -1)
    return img


# ─── MediaPipe landmark index groups ───

LANDMARK_GROUPS = {
    'mouth': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 61],
    'left_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    'right_eyebrow': [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    'lips_outer': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78],
    'lips_inner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61],
    'jaw': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    'left_cheek': [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152],
    'right_cheek': [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152],
    'mouth_corners': [61, 291],
    'mouth_top': [13, 14],
    'mouth_bottom': [17],
}


# ─── FR-3.3: Expression Manipulation ───

def apply_expression_warp(image, landmarks, transform_type, intensity):
    """
    FR-3.3.1 to FR-3.3.9: Apply geometric warping for expression manipulation.
    Uses moving least squares deformation.
    """
    if not landmarks:
        return image

    h, w = image.shape[:2]
    src_points = []
    dst_points = []

    intensity_factor = intensity / 100.0

    if transform_type == 'smile':
        # FR-3.3.1: Move mouth corners up and outward
        indices = [61, 291, 13, 14, 17, 78, 308]
        for idx in indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                sx, sy = lm['x'], lm['y']
                src_points.append([sx, sy])
                if idx in [61, 78]:  # left corner
                    dst_points.append([sx - int(8 * intensity_factor), sy - int(12 * intensity_factor)])
                elif idx in [291, 308]:  # right corner
                    dst_points.append([sx + int(8 * intensity_factor), sy - int(12 * intensity_factor)])
                elif idx in [13, 14]:  # top lip
                    dst_points.append([sx, sy - int(4 * intensity_factor)])
                else:
                    dst_points.append([sx, sy + int(4 * intensity_factor)])

    elif transform_type == 'eyebrow':
        # FR-3.3.2: Raise eyebrows
        for idx in LANDMARK_GROUPS['left_eyebrow'] + LANDMARK_GROUPS['right_eyebrow']:
            if idx < len(landmarks):
                lm = landmarks[idx]
                sx, sy = lm['x'], lm['y']
                src_points.append([sx, sy])
                dst_points.append([sx, sy - int(15 * intensity_factor)])

    elif transform_type == 'lips':
        # FR-3.3.3: Widen lips
        for idx in LANDMARK_GROUPS['lips_outer']:
            if idx < len(landmarks):
                lm = landmarks[idx]
                sx, sy = lm['x'], lm['y']
                cx = w // 2
                src_points.append([sx, sy])
                direction = 1 if sx > cx else -1
                dst_points.append([sx + int(direction * 10 * intensity_factor), sy])

    elif transform_type == 'slim':
        # FR-3.3.4: Face slimming
        for idx in LANDMARK_GROUPS['jaw']:
            if idx < len(landmarks):
                lm = landmarks[idx]
                sx, sy = lm['x'], lm['y']
                cx = w // 2
                src_points.append([sx, sy])
                direction = -1 if sx > cx else 1
                offset = int(abs(sx - cx) * 0.08 * intensity_factor)
                dst_points.append([sx + direction * offset, sy])

    if len(src_points) < 3:
        return image

    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)

    # FR-3.3.5 & FR-3.3.6: Triangle-based geometric deformation
    result = warp_image_triangles(image, src_points, dst_points)
    return result


def warp_image_triangles(image, src_pts, dst_pts):
    """
    FR-3.3.6 & FR-3.3.7: Triangle-based deformation with interpolation.
    """
    h, w = image.shape[:2]
    output = image.copy()

    # Add corner points to ensure full image coverage
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1],
                         [w // 2, 0], [0, h // 2], [w - 1, h // 2], [w // 2, h - 1]],
                        dtype=np.float32)

    all_src = np.vstack([src_pts, corners])
    all_dst = np.vstack([dst_pts, corners])

    # Delaunay triangulation
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)

    for pt in all_dst:
        x_c = max(0, min(w - 1, int(pt[0])))
        y_c = max(0, min(h - 1, int(pt[1])))
        subdiv.insert((x_c, y_c))

    triangles = subdiv.getTriangleList()
    output = np.zeros_like(image)

    for tri in triangles:
        pts_dst = np.array([[tri[0], tri[1]], [tri[2], tri[3]], [tri[4], tri[5]]], dtype=np.float32)

        # Check if triangle is inside image
        if (pts_dst[:, 0].min() < 0 or pts_dst[:, 0].max() >= w or
                pts_dst[:, 1].min() < 0 or pts_dst[:, 1].max() >= h):
            continue

        # Find corresponding source points
        pts_src = np.zeros_like(pts_dst)
        for i, pt in enumerate(pts_dst):
            dists = np.linalg.norm(all_dst - pt, axis=1)
            idx = np.argmin(dists)
            pts_src[i] = all_src[idx]

        # Compute affine transform and warp
        try:
            M = cv2.getAffineTransform(pts_src, pts_dst)
            warped = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            # Create mask for this triangle
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, pts_dst.astype(np.int32), 255)

            mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            output = (output * (1 - mask3) + warped * mask3).astype(np.uint8)
        except cv2.error:
            continue

    # Blend to minimize artifacts (FR-3.3.8)
    mask_total = cv2.cvtColor(cv2.cvtColor(output, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    mask_total = (mask_total > 0).astype(np.float32)
    output = np.where(mask_total > 0, output, image).astype(np.uint8)

    return output


# ─── FR-3.4: Aging & De-Aging ───

def apply_aging(image, intensity, mode='aging'):
    """
    FR-3.4.1 to FR-3.4.7: Aging/de-aging using frequency domain manipulation.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape
    intensity_factor = intensity / 100.0

    # FR-3.5.1: Apply 2D FFT
    f_transform = fft2(gray)
    f_shifted = fftshift(f_transform)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    if mode == 'aging':
        # FR-3.4.1 & FR-3.4.2: Enhance high-frequency for wrinkles
        mask = np.ones((rows, cols), dtype=np.float64)
        # Create high-pass enhancement
        for i in range(rows):
            for j in range(cols):
                dist = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                radius = min(crow, ccol)
                if dist < radius * 0.1:
                    mask[i, j] = 1.0
                else:
                    boost = 1.0 + (intensity_factor * 2.0 * (dist / radius))
                    mask[i, j] = min(boost, 1.0 + intensity_factor * 3.0)

        f_modified = f_shifted * mask
        img_back = np.real(ifft2(fftshift(f_modified)))

        # FR-3.4.7: Contrast adjustment for realism
        img_back = np.clip(img_back, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0 + intensity_factor * 2.0, tileGridSize=(8, 8))
        img_back = clahe.apply(img_back)

        # Add subtle texture for wrinkle effect
        noise = np.random.normal(0, intensity_factor * 15, (rows, cols)).astype(np.float64)
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150).astype(np.float64) / 255.0
        noise = noise * (0.3 + edges * 0.7)
        img_back = np.clip(img_back.astype(np.float64) + noise, 0, 255).astype(np.uint8)

        # Yellow/warm tint for aged look
        result = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)
        result[:, :, 0] = np.clip(result[:, :, 0].astype(np.float64) * (1 - 0.1 * intensity_factor), 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2].astype(np.float64) * (1 + 0.05 * intensity_factor), 0, 255)

    else:
        # FR-3.4.4 & FR-3.4.5: Low-pass filter for de-aging
        mask = np.zeros((rows, cols), dtype=np.float64)
        radius = int(min(crow, ccol) * (0.3 + 0.5 * (1 - intensity_factor)))
        for i in range(rows):
            for j in range(cols):
                dist = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                if dist <= radius:
                    mask[i, j] = 1.0
                else:
                    # Gaussian falloff
                    mask[i, j] = np.exp(-((dist - radius) ** 2) / (2 * (radius * 0.3) ** 2))

        # FR-3.4.6: Preserve edges
        edges = cv2.Canny(gray.astype(np.uint8), 30, 100).astype(np.float64) / 255.0
        edge_mask_freq = fftshift(fft2(edges))
        edge_preserve = np.abs(edge_mask_freq) / (np.max(np.abs(edge_mask_freq)) + 1e-10)
        mask = np.maximum(mask, edge_preserve * 0.3)

        f_modified = f_shifted * mask
        img_back = np.real(ifft2(fftshift(f_modified)))
        img_back = np.clip(img_back, 0, 255).astype(np.uint8)

        # Smooth skin while preserving structure
        bilateral = cv2.bilateralFilter(img_back, 9, 75 * intensity_factor, 75 * intensity_factor)
        img_back = cv2.addWeighted(img_back, 0.3, bilateral, 0.7, 0)

        result = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)
        # Slight brightness boost
        result = np.clip(result.astype(np.float64) * (1 + 0.05 * intensity_factor), 0, 255).astype(np.uint8)

    return result


# ─── FR-3.5: Frequency Domain Analysis ───

def compute_frequency_analysis(image):
    """FR-3.5.1 to FR-3.5.6: Full frequency domain analysis."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # FR-3.5.1: 2D Fourier Transform
    f_transform = fft2(gray)
    f_shifted = fftshift(f_transform)

    # FR-3.5.2: Magnitude spectrum
    magnitude = np.log1p(np.abs(f_shifted))
    magnitude_normalized = ((magnitude - magnitude.min()) /
                            (magnitude.max() - magnitude.min() + 1e-10) * 255).astype(np.uint8)

    # Apply colormap for visualization
    magnitude_colored = cv2.applyColorMap(magnitude_normalized, cv2.COLORMAP_JET)

    # FR-3.5.3: Total spectral energy
    power_spectrum = np.abs(f_shifted) ** 2
    total_energy = float(np.sum(power_spectrum))

    # FR-3.5.4: High and low frequency energy
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    radius = min(crow, ccol) // 4

    low_mask = np.zeros((rows, cols), dtype=bool)
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) <= radius:
                low_mask[i, j] = True

    low_energy = float(np.sum(power_spectrum[low_mask]))
    high_energy = float(np.sum(power_spectrum[~low_mask]))

    # FR-3.5.5: Frequency energy ratios
    ratio = low_energy / (high_energy + 1e-10)

    return {
        'spectrum_image': encode_image(magnitude_colored),
        'total_energy': total_energy,
        'low_energy': low_energy,
        'high_energy': high_energy,
        'lf_hf_ratio': ratio,
        'low_pct': low_energy / (total_energy + 1e-10) * 100,
        'high_pct': high_energy / (total_energy + 1e-10) * 100
    }


# ─── FR-3.6: Quantitative Evaluation ───

def compute_metrics(original, transformed):
    """FR-3.6.1 to FR-3.6.3: Compute MSE, PSNR, SSIM."""
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    trans_gray = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)

    # Ensure same size
    if orig_gray.shape != trans_gray.shape:
        trans_gray = cv2.resize(trans_gray, (orig_gray.shape[1], orig_gray.shape[0]))

    # FR-3.6.1: MSE
    mse_val = float(np.mean((orig_gray.astype(np.float64) - trans_gray.astype(np.float64)) ** 2))

    # FR-3.6.2: PSNR
    if mse_val == 0:
        psnr_val = float('inf')
    else:
        psnr_val = float(10 * np.log10((255.0 ** 2) / mse_val))

    # FR-3.6.3: SSIM
    ssim_val = float(ssim(orig_gray, trans_gray))

    return {
        'mse': round(mse_val, 4),
        'psnr': round(psnr_val, 4),
        'ssim': round(ssim_val, 4)
    }


# ─── Routes ───

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_image():
    """FR-3.1.1: Upload and preprocess image."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Only JPG and PNG are supported.'}), 400

    # Read image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': 'Could not decode image'}), 400

    # FR-3.1.2 to FR-3.1.7: Preprocess
    processed, face_box, error = validate_and_preprocess(image)
    if error:
        return jsonify({'error': error}), 400

    # FR-3.2.1: Detect landmarks
    landmarks = detect_landmarks(processed)

    # FR-3.1.3: Return processed image for display
    return jsonify({
        'original': encode_image(processed),
        'landmarks': landmarks,
        'face_detected': face_box is not None,
        'image_size': list(STANDARD_SIZE),
        'landmark_count': len(landmarks)
    })


@app.route('/api/landmarks', methods=['POST'])
def get_landmarks_image():
    """FR-3.2.3 & FR-3.2.4: Get image with landmarks drawn."""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    image = decode_image(data['image'])
    landmarks = data.get('landmarks', [])
    show = data.get('show', True)

    if show and landmarks:
        result = draw_landmarks(image, landmarks)
    else:
        result = image

    return jsonify({'image': encode_image(result)})


@app.route('/api/export_landmarks', methods=['POST'])
def export_landmarks():
    """FR-3.2.5: Export landmark coordinates."""
    data = request.get_json()
    landmarks = data.get('landmarks', [])

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Index', 'X', 'Y', 'Normalized_X', 'Normalized_Y'])
    for i, lm in enumerate(landmarks):
        writer.writerow([i, lm['x'], lm['y'], round(lm['nx'], 6), round(lm['ny'], 6)])

    mem = io.BytesIO()
    mem.write(output.getvalue().encode('utf-8'))
    mem.seek(0)

    return send_file(mem, mimetype='text/csv', as_attachment=True, download_name='landmarks.csv')


@app.route('/api/transform', methods=['POST'])
def transform_image():
    """FR-3.3: Apply expression transformation."""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    image = decode_image(data['image'])
    landmarks = data.get('landmarks', [])
    transform_type = data.get('type', 'smile')
    intensity = data.get('intensity', 50)

    # FR-3.3.1 to FR-3.3.9
    result = apply_expression_warp(image, landmarks, transform_type, intensity)

    # Compute metrics
    metrics = compute_metrics(image, result)

    return jsonify({
        'transformed': encode_image(result),
        'metrics': metrics
    })


@app.route('/api/aging', methods=['POST'])
def aging_transform():
    """FR-3.4: Apply aging/de-aging."""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    image = decode_image(data['image'])
    mode = data.get('mode', 'aging')
    intensity = data.get('intensity', 50)

    result = apply_aging(image, intensity, mode)

    # Resize result to match original if needed
    if result.shape[:2] != image.shape[:2]:
        result = cv2.resize(result, (image.shape[1], image.shape[0]))

    metrics = compute_metrics(image, result)

    return jsonify({
        'transformed': encode_image(result),
        'metrics': metrics
    })


@app.route('/api/frequency', methods=['POST'])
def frequency_analysis():
    """FR-3.5: Frequency domain analysis."""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    image = decode_image(data['image'])
    analysis = compute_frequency_analysis(image)

    # If transformed image provided, do comparative analysis (FR-3.5.6)
    if 'transformed' in data:
        trans_image = decode_image(data['transformed'])
        trans_analysis = compute_frequency_analysis(trans_image)
        analysis['transformed_spectrum'] = trans_analysis['spectrum_image']
        analysis['transformed_energy'] = trans_analysis['total_energy']
        analysis['transformed_low_energy'] = trans_analysis['low_energy']
        analysis['transformed_high_energy'] = trans_analysis['high_energy']
        analysis['transformed_lf_hf_ratio'] = trans_analysis['lf_hf_ratio']
        analysis['transformed_low_pct'] = trans_analysis['low_pct']
        analysis['transformed_high_pct'] = trans_analysis['high_pct']

    return jsonify(analysis)


@app.route('/api/metrics', methods=['POST'])
def get_metrics():
    """FR-3.6: Compute evaluation metrics."""
    data = request.get_json()
    if not data or 'original' not in data or 'transformed' not in data:
        return jsonify({'error': 'Both original and transformed images required'}), 400

    original = decode_image(data['original'])
    transformed = decode_image(data['transformed'])
    metrics = compute_metrics(original, transformed)

    return jsonify(metrics)


@app.route('/api/export_metrics', methods=['POST'])
def export_metrics():
    """FR-3.6.5: Export metrics as CSV."""
    data = request.get_json()
    metrics_list = data.get('metrics', [])

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Transformation', 'MSE', 'PSNR', 'SSIM'])
    for m in metrics_list:
        writer.writerow([m.get('name', ''), m.get('mse', ''), m.get('psnr', ''), m.get('ssim', '')])

    mem = io.BytesIO()
    mem.write(output.getvalue().encode('utf-8'))
    mem.seek(0)

    return send_file(mem, mimetype='text/csv', as_attachment=True, download_name='metrics_report.csv')


@app.route('/api/save_image', methods=['POST'])
def save_image():
    """FR-3.7.3: Save processed image."""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    image = decode_image(data['image'])
    filename = f"processed_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(RESULTS_FOLDER, filename)
    cv2.imwrite(filepath, image)

    return jsonify({'filename': filename, 'path': filepath})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
