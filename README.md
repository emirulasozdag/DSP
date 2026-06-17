# Real-Time DSP Virtual Try-On

This project is a live Snapchat-like virtual try-on demo for a DSP / image-processing course. It opens the webcam, tracks a face with MediaPipe Face Mesh, and overlays selectable accessories and makeup in real time.

The virtual try-on is integrated into the existing FaceWarp website webcam flow. The old static upload/image-processing pipeline is preserved.

## Features

- Live webcam input inside the FaceWarp website.
- MediaPipe Face Mesh landmark tracking for live browser frames.
- Right-side category menu with Accessories and Makeup.
- Six premium glasses styles, five cartoon hat models, lipstick, and blush filters.
- Yaw-aware glasses placement for left/right head turns.
- Perspective-like glasses warping for a more 3-D try-on feel.
- Hat placement anchored to upper-face landmarks and shifted upward so it sits near the hairline.
- Lipstick anchored to lip polygons and blush anchored to cheek landmark groups.
- Alpha blending for transparent PNG accessories.
- Exponential moving average smoothing to reduce jitter without too much lag.

## Project Structure

```text
main.py
assets/
  glasses.png
  glasses_1.png
  glasses_2.png
  glasses_3.png
  glasses_4.png
  glasses_5.png
  glasses_6.png
  hat.png
  hat_1.png
  hat_2.png
  hat_3.png
  hat_5.png
  hat_6.png
requirements.txt
README.md
```

## Setup

MediaPipe supports Python 3.9-3.12 on PyPI. This workspace's default `python` may be Python 3.13, so start the website with the Python 3.10 MediaPipe environment:

```bash
env DSP_PORT=8082 /opt/anaconda3/envs/dsp310/bin/python app.py
```

Then open:

```text
http://127.0.0.1:8082
```

Website try-on menu:

- Main categories: Accessories, Makeup
- Accessories: Glasses, Hats
- Makeup: Lipstick, Blush
- `A`: open Accessories
- `M`: open Makeup
- `B`: go back
- `0`: clear filters
- `D`: toggle makeup anchor debug points
- `R`: toggle optional mirror mode; default is Mirror Off. On browsers that deliver the camera mirrored, Mirror Off counter-flips the display so the preview looks non-mirrored.
- `Q` or `ESC`: close webcam
- The panel also shows render FPS, landmark latency, smoothing alpha, and whether landmarks are fresh or stale.

For a fresh environment:

```bash
python3.10 -m venv .venv-face-filter
source .venv-face-filter/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

### Local AI Aging & dlib Setup

To ensure that the Local AI Aging and dlib landmark tracking features run smoothly on another computer, you need to configure the large files and repositories that are ignored in `.gitignore`:

1. **PyTorch & Torchvision Installation:**
   PyTorch is not included in `requirements.txt` because the installation command depends on the operating system and CUDA (GPU) support. Run one of the following commands in the terminal of the target computer based on your hardware:
   * **CPU Only (if no GPU or CUDA is not needed):**
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
     ```
   * **NVIDIA GPU (for fast CUDA execution - CUDA 12.8 / 12.1 recommended):**
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
     ```

2. **dlib Installation (Windows):**
   To avoid compilation errors on Windows (especially with Python 3.14+), download and install the precompiled dlib wheel:
   ```bash
   pip install https://github.com/sachadee/Dlib/raw/main/dlib-20.0.0-cp314-cp314-win_amd64.whl
   ```

3. **Download Model Weights (aging.pt & dlib shape predictor):**
   Due to GitHub's file size limits, large models (`models/aging.pt` - 2.27 GB and `models/shape_predictor_68_face_landmarks.dat` - 99.6 MB) are not included in the repository.
   
   You can download all required models automatically by running:
   ```bash
   python download_models.py
   ```
   *(Note: `face_landmarker.task` is already included directly in the Git repository as it is small).*

4. **`sam_repo` Automatic Setup:**
   Ensure that `git` is installed on the target computer. When the project runs for the first time, `face_aging/aging_service.py` will automatically clone the `sam_repo` directory from GitHub. If the target machine is offline, you can manually zip your local `sam_repo` folder and extract it to the project root directory.

If macOS blocks the webcam, allow camera access for your terminal or IDE:

```text
System Settings > Privacy & Security > Camera
```

## How It Works

Each browser webcam frame is captured in a non-mirrored coordinate system and sent to the Flask `/api/landmarks` endpoint. The site requests MediaPipe Face Mesh landmarks and uses eye corners, nose bridge, cheeks, forehead, chin, lips, and cheek regions to estimate:

- eye center and nose bridge for glasses anchor
- eye distance for glasses scale
- eye-line angle for head roll
- face width and face height for hat scale and position
- nose offset from face center for approximate yaw
- lip polygons for lipstick
- cheek landmark groups for blush

The glasses are resized, yaw-adjusted, rotated, and alpha-blended onto the browser canvas. The yaw estimate slightly compresses and shifts the glasses when the head turns left or right.

The hat is scaled by face width, rotated by head roll, and anchored from upper-face landmarks. It is intentionally shifted upward:

```text
hat_y = forehead_y - hat_height * HAT_BOTTOM_ALIGNMENT + face_height * HAT_Y_OFFSET_FACTOR
```

The default `HAT_Y_OFFSET_FACTOR = -0.45` places the bottom edge near the hairline/top forehead area instead of covering the forehead.

Lipstick uses only these MediaPipe Face Mesh lip landmarks:

```text
outer: 61,146,91,181,84,17,314,405,321,375,291,308
inner: 78,95,88,178,87,14,317,402,318,324
```

Blush uses only these cheek landmarks:

```text
left: 50,117,118,119,123,187
right: 280,346,347,348,352,411
```

Both makeup effects are drawn on soft offscreen canvas masks with alpha blending and Gaussian-style blur. Press `D` in the webcam modal to show debug anchor points and verify the regions.

The DSP smoothing step is an exponential moving average:

```text
smoothed = (1 - alpha) * previous + alpha * current
```

This behaves like a low-pass filter: it suppresses fast landmark jitter while keeping normal head motion responsive.

## Tuning

Useful constants:

- `FAST_SMOOTHING_ALPHA`: faster tracking for lips, eyes, glasses, cheeks, roll, and yaw.
- `SLOW_SMOOTHING_ALPHA`: steadier tracking for hat anchors.
- `LANDMARK_POLL_MS`: interval for live Face Mesh landmark requests.
- `GLASSES_WIDTH_FACTOR_WEB`: glasses size relative to eye distance.
- `HAT_WIDTH_FACTOR_WEB`: hat size relative to face width.
- `HAT_Y_OFFSET_FACTOR_WEB`: negative values move the hat upward.
- `HAT_BOTTOM_ALIGNMENT_WEB`: controls which vertical part of the hat aligns near the forehead reference.
- `LIPSTICK_OPACITY`, `LIPSTICK_BLUR`: lipstick softness and strength.
- `BLUSH_OPACITY`, `BLUSH_BLUR`, `BLUSH_X_OFFSET`, `BLUSH_Y_OFFSET`, `BLUSH_SCALE`: blush placement and softness.

Precise lipstick and blush require MediaPipe Face Mesh with 468+ landmarks. If the site is run with an unsupported Python version, the backend may fall back to a 68-point approximation; accessory filters still work, but precise makeup is disabled because those lip and cheek indices do not exist.

## Manual Acceptance Test

Run:

```bash
env DSP_PORT=8082 /opt/anaconda3/envs/dsp310/bin/python app.py
```

Confirm that:

- the website opens at `http://127.0.0.1:8082`
- clicking `Use Webcam` opens the large 70/30 webcam modal
- the category menu shows Accessories and Makeup
- clicking catalog cards changes accessories and makeup
- `A`, `M`, `B`, `0`, `D`, `Q`, and `ESC` work
- Mirror mode is Off by default; the video and accessory canvas are flipped together only when needed to counter-correct a mirrored camera stream
- glasses stay aligned when moving left/right, tilting, and moving closer/farther
- the hat sits on top of the head instead of covering the forehead
- lipstick sits on the lips and blush sits on both cheeks
