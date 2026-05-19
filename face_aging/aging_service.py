import os
import sys
import cv2
import numpy as np
import PIL.Image
import scipy.ndimage
import torch
import torchvision.transforms as transforms
from argparse import Namespace

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Append sam_repo to sys.path so we can import its modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM_REPO_PATH = os.path.join(PROJECT_ROOT, "sam_repo")

if not os.path.exists(SAM_REPO_PATH):
    print("[AgingService] SAM repository not found. Cloning...")
    import subprocess
    subprocess.run(["git", "clone", "https://github.com/yuval-alaluf/SAM.git", SAM_REPO_PATH], check=True)

if SAM_REPO_PATH not in sys.path:
    sys.path.append(SAM_REPO_PATH)

try:
    from models.psp import pSp
    from datasets.augmentations import AgeTransformer
    SAM_AVAILABLE = True
except ImportError as e:
    print(f"[AgingService] Could not import SAM dependencies: {e}")
    SAM_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# FFHQ-style face alignment using MediaPipe Tasks API
# ─────────────────────────────────────────────────────────────────────────────

def align_face_mediapipe(img_bgr):
    """
    FFHQ face alignment using MediaPipe Tasks API.
    Returns: (PIL.Image of size 256x256, original_quad for unwarping, landmark list)
    """
    model_asset_path = os.path.join(PROJECT_ROOT, 'models', 'face_landmarker.task')
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        res = landmarker.detect(mp_image)
        if not res.face_landmarks:
            return None, None, None

        lm = res.face_landmarks[0]
        h, w, _ = img_bgr.shape

        # Viewer's Left (smaller X): indices 33/133
        # Viewer's Right (larger X): indices 263/362
        eye_left  = np.array([(lm[33].x  + lm[133].x) / 2 * w, (lm[33].y  + lm[133].y) / 2 * h])
        eye_right = np.array([(lm[263].x + lm[362].x) / 2 * w, (lm[263].y + lm[362].y) / 2 * h])
        mouth_left  = np.array([lm[78].x  * w, lm[78].y  * h])
        mouth_right = np.array([lm[308].x * w, lm[308].y * h])

        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        original_quad = quad.copy()   # save BEFORE any crop/pad mutations
        qsize = np.hypot(*x) * 2

        img         = PIL.Image.fromarray(img_rgb)
        output_size = 256
        transform_size = 256

        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)),
                     int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, getattr(PIL.Image, 'Resampling', PIL.Image).LANCZOS)
            quad /= shrink
            qsize /= shrink

        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
                int(np.ceil(max(quad[:, 0]))),  int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
                min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img  = img.crop(crop)
            quad -= crop[0:2]

        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
               int(np.ceil(max(quad[:, 0]))),  int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0),
               max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img_arr = np.pad(np.float32(img),
                             ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            _h, _w, _ = img_arr.shape
            _y, _x, _ = np.ogrid[:_h, :_w, :1]
            mask = np.maximum(
                1.0 - np.minimum(np.float32(_x) / pad[0], np.float32(_w - 1 - _x) / pad[2]),
                1.0 - np.minimum(np.float32(_y) / pad[1], np.float32(_h - 1 - _y) / pad[3]))
            blur = qsize * 0.02
            img_arr += (scipy.ndimage.gaussian_filter(img_arr, [blur, blur, 0]) - img_arr) \
                       * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img_arr += (np.median(img_arr, axis=(0, 1)) - img_arr) * np.clip(mask, 0.0, 1.0)
            img   = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img_arr), 0, 255)), 'RGB')
            quad += pad[:2]

        BILINEAR = getattr(PIL.Image, 'Resampling', PIL.Image).BILINEAR
        LANCZOS  = getattr(PIL.Image, 'Resampling', PIL.Image).LANCZOS
        img = img.transform((transform_size, transform_size),
                            PIL.Image.QUAD, (quad + 0.5).flatten(), BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), LANCZOS)

        return img, original_quad, lm


# ─────────────────────────────────────────────────────────────────────────────
# Eye-region protection mask (MediaPipe landmark-based)
# ─────────────────────────────────────────────────────────────────────────────

# MediaPipe eye-outline landmark indices (left eye viewer / right eye viewer)
_LEFT_EYE_IDX  = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                  246, 161, 160, 159, 158, 157, 173]
_RIGHT_EYE_IDX = [263, 249, 390, 373, 374, 380, 381, 382, 362,
                  466, 388, 387, 386, 385, 384, 398]


def build_eye_protection_mask(lm, img_h: int, img_w: int,
                              padding_ratio: float = 0.45) -> np.ndarray:
    """
    Creates a soft float32 mask (0..1) in which the eye regions are 1.0
    (to be preserved) and the rest is 0.0.

    The mask will be used to composite ORIGINAL eye pixels back over the
    aged-face output, so that eye colour is never changed.

    padding_ratio : fraction of eye width added around each eye as soft margin
    """
    mask = np.zeros((img_h, img_w), dtype=np.float32)

    for eye_indices in (_LEFT_EYE_IDX, _RIGHT_EYE_IDX):
        pts = np.array([[int(lm[i].x * img_w), int(lm[i].y * img_h)]
                        for i in eye_indices], dtype=np.int32)
        x, y, bw, bh = cv2.boundingRect(pts)
        pad_x = int(bw * padding_ratio)
        pad_y = int(bh * padding_ratio)
        cx, cy = x + bw // 2, y + bh // 2
        axes = (bw // 2 + pad_x, bh // 2 + pad_y)
        cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1.0, -1)

    # Soft feathered edges — blur proportional to image size
    k = max(int(min(img_h, img_w) * 0.03) | 1, 5)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Soft face blend mask (ellipse on FFHQ canonical space)
# ─────────────────────────────────────────────────────────────────────────────

def _build_skin_mask(size: int = 1024) -> np.ndarray:
    """Soft elliptical mask centred on the skin region (ignores hair)."""
    mask = np.zeros((size, size), dtype=np.float32)
    centre = (size // 2, int(size * 0.57))
    axes   = (int(size * 0.35), int(size * 0.41))
    cv2.ellipse(mask, centre, axes, 0, 0, 360, 1.0, -1)
    k = (size // 7) | 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def _build_compositing_mask(size: int = 1024) -> np.ndarray:
    """
    Full-face + full-hair compositing mask for FFHQ canonical space.

    Two-ellipse approach:
      • Top ellipse  : wide, centred at y≈0.38 (above the nose) — gives
                       full coverage of hair crown all the way to top edge.
      • Bottom ramp  : a graduated, *asymmetric* feather that dissolves the
                       chin/neck edge downward over ~20% of the canvas height
                       so the seam is invisible after warpPerspective.

    The bottom is feathered much more aggressively than the top/sides:
    we multiply the mask row-by-row with a 1-D sigmoid ramp so the
    bottom 25% of the canvas fades from 1.0 → 0.0 smoothly.
    """
    # ── 1. Primary face+hair ellipse ─────────────────────────────────────
    mask = np.zeros((size, size), dtype=np.float32)
    # Tall upward-shifted ellipse: covers from crown (y=0) to chin (y≈0.80)
    centre_face = (size // 2, int(size * 0.40))
    axes_face   = (int(size * 0.46), int(size * 0.52))
    cv2.ellipse(mask, centre_face, axes_face, 0, 0, 360, 1.0, -1)

    # Light uniform blur to soften all edges equally
    k_light = (size // 20) | 1
    mask = cv2.GaussianBlur(mask, (k_light, k_light), 0)

    # ── 2. Asymmetric bottom feather ─────────────────────────────────────
    # Build a 1-D vertical ramp that is 1.0 from y=0 down to y=75% of the
    # canvas, then falls smoothly to 0.0 by y=100%.  This dissolves the
    # bottom neck edge without touching the hair crown at all.
    ys = np.linspace(0.0, 1.0, size, dtype=np.float32)  # 0 = top, 1 = bottom
    fade_start = 0.72   # full opacity above this fraction
    fade_end   = 0.96   # fully transparent below this fraction
    ramp = np.clip((fade_end - ys) / (fade_end - fade_start), 0.0, 1.0)
    # Apply a 1-D Gaussian smooth to the ramp so it has no kink
    k_ramp = (size // 12) | 1
    ramp = cv2.GaussianBlur(ramp.reshape(-1, 1), (1, k_ramp), 0).reshape(-1)
    # Broadcast ramp across all columns
    ramp_2d = np.tile(ramp.reshape(-1, 1), (1, size))
    mask = mask * ramp_2d

    # ── 3. Side feather (subtle) ─────────────────────────────────────────
    # Fade the left and right edges so the warp boundary doesn't show
    xs = np.linspace(0.0, 1.0, size, dtype=np.float32)  # 0 = left, 1 = right
    side_fade_w = 0.04   # feather width fraction on each side
    side_ramp = np.clip(
        np.minimum(xs / side_fade_w, (1.0 - xs) / side_fade_w), 0.0, 1.0)
    mask = mask * side_ramp.reshape(1, -1)

    return np.clip(mask, 0.0, 1.0)


def _build_hair_compositing_mask(size: int = 1024) -> np.ndarray:
    """
    Mask covering ONLY the upper hair-crown region of the FFHQ canvas.
    Used as a fallback to force AI-generated hair pixels directly onto
    the output, bypassing any accidental suppression by the face mask.

    In FFHQ canonical space the hair crown occupies the top ~35% of the
    image.  We paint a wide ellipse whose centre is pushed above the top
    edge so the ellipse only exposes its bottom half inside the canvas.
    """
    mask = np.zeros((size, size), dtype=np.float32)
    # Centre above the canvas so we only see the bottom arc of the ellipse
    centre = (size // 2, int(size * 0.10))
    axes   = (int(size * 0.48), int(size * 0.42))
    cv2.ellipse(mask, centre, axes, 0, 0, 360, 1.0, -1)
    k = (size // 10) | 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic SAM age mapping
# ─────────────────────────────────────────────────────────────────────────────

def compute_sam_target_age(current_age: int, target_age: int) -> int:
    """
    Map the requested age difference to SAM's 0-100 target scale.
    Keeps the target closer to the requested biological age, with only a
    mild boost so the effect is visible without being extreme.
    """
    delta = max(0, target_age - current_age)

    if delta <= 0:
        return target_age

    # Çok hafif bir destek (boost) ekliyoruz ki değişim fark edilsin, 
    # ama biyolojik yaşa (target_age) çok yakın kalsın.
    if delta <= 10:
        boost = 2
    elif delta <= 20:
        boost = 4
    elif delta <= 30:
        boost = 6
    else:
        boost = 8

    sam_age = target_age + boost
    return min(int(sam_age), 95)


# ─────────────────────────────────────────────────────────────────────────────
# Main service class
# ─────────────────────────────────────────────────────────────────────────────

class LocalAgingService:
    """
    Local face-aging inference using PyTorch + SAM (sam_ffhq_aging.pt).

    Identity-preservation strategy
    --------------------------------
    1. Dynamic age mapping   : SAM age target is computed proportionally
                               from (current_age → target_age) delta,
                               avoiding one-size-fits-all over-aging.
    2. Eye region protection : MediaPipe landmarks locate both eyes precisely.
                               After compositing the aged face, the original
                               eye pixels (with a soft feather edge) are
                               blended back — preserving eye colour exactly.
    3. Skin-tone correction  : Reinhard LAB colour transfer normalises the
                               AI output skin tone to match the source,
                               removing greenish/yellowish cast artefacts.
    """

    def __init__(self, model_path: str = "models/aging.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[AgingService] Loading model from '{model_path}' on {self.device}...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[AgingService] Model file not found: {model_path}."
            )

        if not SAM_AVAILABLE:
            raise RuntimeError(
                "[AgingService] SAM architecture modules unavailable. "
                "Ensure sam_repo/ is present."
            )

        ckpt = torch.load(model_path, map_location="cpu")
        opts = ckpt["opts"]
        opts["checkpoint_path"] = model_path
        opts["device"]          = str(self.device)
        self.opts = Namespace(**opts)

        self.net = pSp(self.opts)
        self.net.eval()
        if self.device.type == "cuda":
            self.net.to(self.device)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    # ── Private helpers ───────────────────────────────────────────────────────

    def _preprocess(self, pil_img: PIL.Image.Image, sam_age: int) -> torch.Tensor:
        """Normalize 256×256 PIL image and add SAM's age conditioning channel."""
        tensor = self.transform(pil_img)
        tensor = AgeTransformer(target_age=sam_age)(tensor.cpu())
        return tensor.unsqueeze(0).to(self.device).float()

    def _get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the raw W+ latent codes from the pSp encoder without decoding."""
        with torch.no_grad():
            codes = self.net.encoder(x)
            if self.net.opts.start_from_latent_avg:
                codes = codes + self.net.latent_avg
            elif self.net.opts.start_from_encoded_w_plus:
                encoded_latents = self.net.pretrained_encoder(x[:, :-1, :, :])
                encoded_latents = encoded_latents + self.net.latent_avg
                codes = codes + encoded_latents
            return codes

    @staticmethod
    def _postprocess(tensor: torch.Tensor) -> np.ndarray:
        """Convert SAM output tensor [-1,1] → uint8 BGR image."""
        t = tensor.squeeze(0).cpu().detach()
        t = torch.clamp(t, -1.0, 1.0)
        t = (t + 1.0) / 2.0
        arr = t.permute(1, 2, 0).numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _skin_tone_correction(original_bgr: np.ndarray,
                              aged_bgr: np.ndarray,
                              face_mask: np.ndarray = None) -> np.ndarray:
        """
        Reinhard LAB colour transfer: copy skin-tone statistics from
        original face → aged face. Uses a mask so hair/background
        are NOT colour-corrected (allowing hair whitening to persist).
        """
        orig_lab = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
        aged_lab = cv2.cvtColor(aged_bgr,     cv2.COLOR_BGR2LAB).astype(np.float64)

        result = aged_lab.copy()
        for c in range(3):
            if face_mask is not None:
                o_vals = orig_lab[:, :, c][face_mask > 0.1]
                a_vals = aged_lab[:, :, c][face_mask > 0.1]
                o_mean, o_std = (o_vals.mean(), o_vals.std()) if o_vals.size else (0, 1)
                a_mean, a_std = (a_vals.mean(), a_vals.std()) if a_vals.size else (0, 1)
            else:
                o_mean, o_std = orig_lab[:, :, c].mean(), orig_lab[:, :, c].std()
                a_mean, a_std = aged_lab[:, :, c].mean(), aged_lab[:, :, c].std()
                
            eps = 1e-6
            result[:, :, c] = (aged_lab[:, :, c] - a_mean) * (o_std / (a_std + eps)) + o_mean

        result = np.clip(result, 0, 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        
        if face_mask is not None:
            mask_3ch = np.stack([face_mask] * 3, axis=-1)
            output = result_bgr.astype(np.float32) * mask_3ch + aged_bgr.astype(np.float32) * (1.0 - mask_3ch)
            return np.clip(output, 0, 255).astype(np.uint8)
            
        return result_bgr

    # ── Public API ────────────────────────────────────────────────────────────

    def run_inference(self, image: np.ndarray,
                      target_age: int = 50,
                      current_age: int = 30) -> np.ndarray:
        """
        Age *image* (BGR ndarray) and return a BGR ndarray.
        """
        # 1. FFHQ alignment + landmark detection ──────────────────────────
        aligned_pil, original_quad, lm = align_face_mediapipe(image)
        if aligned_pil is None:
            print("[AgingService] No face detected. Returning original image.")
            return image

        # 2. Dynamic SAM age computation ──────────────────────────────────
        sam_age = compute_sam_target_age(current_age, target_age)
        print(f"[AgingService] current_age={current_age}, target_age={target_age}, "
              f"-> sam_age={sam_age}", flush=True)

        # 3. Explicit W+ Latent Space Manipulation (Identity Lock & Wrinkle Amplification)
        # Get baseline latent for current age
        input_orig = self._preprocess(aligned_pil, sam_age=current_age)
        latent_orig = self._get_latent(input_orig)
        
        # Get target latent for requested age
        input_aged = self._preprocess(aligned_pil, sam_age=sam_age)
        latent_aged = self._get_latent(input_aged)
        
        # Calculate the aging direction vector
        direction = latent_aged - latent_orig
        
        # We start with the original latent to strictly enforce the Identity Lock
        final_latent = latent_orig.clone()
        
        delta = max(0, target_age - current_age)
        
        # Layers 5-7 (Medium): Skin structure, deeper wrinkles
        medium_mult = 1.15 if delta > 15 else 1.05
        
        # Layers 8-11 (Fine): Micro-texture, pores
        fine_mult = 1.25 if delta > 15 else 1.10
        
        import math

        # 1. Aggressive Scaling on Color Layers (14-17)
        # Threshold-Based Application: activates > 45, scales exponentially
        if target_age >= 50:
            excess = target_age - 50
            # Exponentially scales from 2.5x at age 50 up to max 4.0x
            hair_mult = min(4.0, 2.5 * math.exp(excess * 0.04))
        elif target_age > 45:
            excess = target_age - 45
            # Linear ramp from 1.0 at age 45 to 2.5 at age 50
            hair_mult = 1.0 + (1.5 / 5.0) * excess
        else:
            hair_mult = 1.0
            
        # Apply the explicit layer restrictions and amplifications
        # Layers 0-4 (Coarse): STRICTLY FROZEN (Identity, shape, pose, eye color geometry)
        
        # Layers 5-7 (Medium): Scaled up for better wrinkles
        for i in range(5, 8):
            final_latent[:, i, :] += direction[:, i, :] * medium_mult
            
        # Layers 8-11 (Fine): Scaled up for texture
        for i in range(8, 12):
            final_latent[:, i, :] += direction[:, i, :] * fine_mult
            
        # Layers 12-13 (Some color/texture): Moderate scale
        for i in range(12, 14):
            final_latent[:, i, :] += direction[:, i, :] * (1.2 if delta > 15 else 1.0)
            
        # Layers 12-17 (Hair Pigmentation/Color): Aggressive whitening
        # NOTE: We do NOT pull toward latent_avg — the FFHQ population mean
        # skews toward dark hair, so that would bias the result darker.
        # Instead we amplify the model's own aging direction (which genuinely
        # encodes gray/white pigment loss) and add an explicit positive
        # lightness offset derived from how far the direction already moves.
        for i in range(12, 18):
            # Pick the right multiplier: layers 12-13 get moderate scale,
            # layers 14-17 (primary pigment layers) get the full hair_mult
            layer_mult = hair_mult if i >= 14 else (1.2 if delta > 15 else 1.0)
            final_latent[:, i, :] += direction[:, i, :] * layer_mult

            # Lightness bias for ages > 45 on all hair/color layers
            if target_age > 45:
                # Scale: 0.0 at age 45, ramps to max 1.5 at age 80+
                lightness_strength = min(1.5, (target_age - 45) * 0.05)

                # The aging direction at these layers already encodes
                # the shift from current pigment → older pigment.
                # Amplify *only its positive components* (lightness-increasing
                # dimensions) to push hair lighter without darkening anything.
                dir_layer = direction[:, i, :]
                positive_dir = torch.clamp(dir_layer, min=0.0)
                norm = positive_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                positive_unit = positive_dir / norm

                # Inject a scaled positive-lightness nudge into the latent
                final_latent[:, i, :] += positive_unit * lightness_strength

        # 4. SAM Generator Inference ──────────────────────────────────────
        with torch.no_grad():
            # Pass the manipulated W+ latent code directly to the decoder
            out = self.net(final_latent, input_code=True, input_is_full=True, 
                           randomize_noise=False, resize=False)
            output_tensor = out[0] if isinstance(out, tuple) else out

        # aged_face_1024: 1024×1024 BGR (SAM upscales internally 256→1024)
        aged_face_1024 = self._postprocess(output_tensor)

        # ── LANCZOS ref for skin-tone correction ────────────────────────
        LANCZOS = getattr(PIL.Image, 'Resampling', PIL.Image).LANCZOS
        aligned_1024 = np.array(aligned_pil.resize((1024, 1024), LANCZOS))
        aligned_1024_bgr = cv2.cvtColor(aligned_1024, cv2.COLOR_RGB2BGR)

        # 5. Skin-tone correction (face region only, excludes hair) ──────
        # skin_mask covers the cheeks/forehead but NOT the hair crown, so
        # the LAB colour-transfer does not undo the AI's gray/white hair.
        skin_mask_1024 = _build_skin_mask(size=1024)
        aged_face_corrected = self._skin_tone_correction(
            aligned_1024_bgr, aged_face_1024, skin_mask_1024)

        # 6. Fallback: explicitly paste AI hair pixels over the corrected face
        # This is the critical step: even if the compositing mask later
        # attenuates the hair region, the AI hair is already baked into
        # aged_face_corrected before we unwarp.
        if target_age > 45:
            hair_mask_1024 = _build_hair_compositing_mask(size=1024)
            # Strength scales from 0.6 at age 46 → 1.0 at age 70+
            hair_strength = min(1.0, 0.6 + (target_age - 45) * 0.016)
            hair_mask_3ch = np.stack([hair_mask_1024 * hair_strength] * 3, axis=-1)
            aged_face_corrected = (
                aged_face_1024.astype(np.float32) * hair_mask_3ch
                + aged_face_corrected.astype(np.float32) * (1.0 - hair_mask_3ch)
            )
            aged_face_corrected = np.clip(aged_face_corrected, 0, 255).astype(np.uint8)
            print(f"[AgingService] Hair override applied: strength={hair_strength:.2f}",
                  flush=True)

        # 7. Full face+hair compositing mask ─────────────────────────────
        # New mask reaches the very top of the FFHQ canvas so hair is
        # never excluded from the final composite.
        face_mask_1024    = _build_compositing_mask(size=1024)
        face_mask_1024_3ch = np.stack([face_mask_1024] * 3, axis=-1)

        # 8. Inverse perspective transform: unwarp → original image space ─
        dst_pts  = np.array([[0, 0], [0, 1024], [1024, 1024], [1024, 0]],
                            dtype=np.float32)
        quad_pts = np.array(original_quad, dtype=np.float32)
        M = cv2.getPerspectiveTransform(dst_pts, quad_pts)

        h, w = image.shape[:2]
        unwarped_face = cv2.warpPerspective(
            aged_face_corrected, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        unwarped_mask = cv2.warpPerspective(
            face_mask_1024_3ch, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        # 9. Alpha composite aged face over original background ───────────
        orig_f  = image.astype(np.float32)
        aged_f  = unwarped_face.astype(np.float32)
        blended = orig_f * (1.0 - unwarped_mask) + aged_f * unwarped_mask
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # 10. ── EYE COLOUR PROTECTION ────────────────────────────────────
        # Composite the *original* eye pixels back to preserve iris colour.
        eye_mask = build_eye_protection_mask(lm, h, w, padding_ratio=0.40)
        eye_mask_3ch = np.stack([eye_mask] * 3, axis=-1)

        blended = (orig_f * eye_mask_3ch
                   + blended.astype(np.float32) * (1.0 - eye_mask_3ch))
        return np.clip(blended, 0, 255).astype(np.uint8)
