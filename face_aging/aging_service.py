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
    Creates a larger mask that covers the face AND the hair region, 
    ensuring AI-generated gray hair is composited over the original image.
    """
    mask = np.zeros((size, size), dtype=np.float32)
    centre = (size // 2, int(size * 0.48))  # Moved up to include hair
    axes   = (int(size * 0.42), int(size * 0.50))  # Wider & taller
    cv2.ellipse(mask, centre, axes, 0, 0, 360, 1.0, -1)
    k = (size // 5) | 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
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
        
        # Layers 12-17 (Color/Hair): Specifically controls hair pigment/whitening
        if delta > 30:
            hair_mult = 2.20  # Very prominent gray/white hair for old age
        elif delta > 15:
            hair_mult = 1.60
        else:
            hair_mult = 1.15
            
        # Apply the explicit layer restrictions and amplifications
        # Layers 0-4 (Coarse): STRICTLY FROZEN (Identity, shape, pose, eye color geometry)
        
        # Layers 5-7 (Medium): Scaled up for better wrinkles
        for i in range(5, 8):
            final_latent[:, i, :] += direction[:, i, :] * medium_mult
            
        # Layers 8-11 (Fine): Scaled up for texture
        for i in range(8, 12):
            final_latent[:, i, :] += direction[:, i, :] * fine_mult
            
        # Layers 12-17 (Hair/Pigment): Scaled up heavily for hair whitening
        for i in range(12, 18):
            final_latent[:, i, :] += direction[:, i, :] * hair_mult

        # 4. SAM Generator Inference ──────────────────────────────────────
        with torch.no_grad():
            # Pass the manipulated W+ latent code directly to the decoder
            out = self.net(final_latent, input_code=True, input_is_full=True, 
                           randomize_noise=False, resize=False)
            output_tensor = out[0] if isinstance(out, tuple) else out

        # aged_face_1024: 1024×1024 BGR (SAM upscales internally 256→1024)
        aged_face_1024 = self._postprocess(output_tensor)

        # 5. Soft face blend mask (for compositing onto original photo) ───
        face_mask_1024    = _build_compositing_mask(size=1024)
        face_mask_1024_3ch = np.stack([face_mask_1024] * 3, axis=-1)

        # 4. Skin-tone correction ─────────────────────────────────────────
        # Resize original aligned face to 1024 for reference
        LANCZOS = getattr(PIL.Image, 'Resampling', PIL.Image).LANCZOS
        aligned_1024 = np.array(aligned_pil.resize((1024, 1024), LANCZOS))
        aligned_1024_bgr = cv2.cvtColor(aligned_1024, cv2.COLOR_RGB2BGR)

        skin_mask_1024 = _build_skin_mask(size=1024)
        aged_face_1024 = self._skin_tone_correction(aligned_1024_bgr, aged_face_1024, skin_mask_1024)

        # 6. Inverse perspective transform: unwarp → original image space ─
        dst_pts  = np.array([[0, 0], [0, 1024], [1024, 1024], [1024, 0]],
                            dtype=np.float32)
        quad_pts = np.array(original_quad, dtype=np.float32)
        M = cv2.getPerspectiveTransform(dst_pts, quad_pts)

        h, w = image.shape[:2]
        unwarped_face = cv2.warpPerspective(
            aged_face_1024, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        unwarped_mask = cv2.warpPerspective(
            face_mask_1024_3ch, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        # 7. Alpha composite aged face over original background ───────────
        orig_f  = image.astype(np.float32)
        aged_f  = unwarped_face.astype(np.float32)
        blended = orig_f * (1.0 - unwarped_mask) + aged_f * unwarped_mask
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # 8. ── EYE COLOUR PROTECTION ──────────────────────────────────────
        # Build a soft mask around both eye regions using MediaPipe landmarks.
        # Composite the *original* eye pixels back over the blended result.
        # This guarantees eye colour is never changed by the AI.
        eye_mask = build_eye_protection_mask(lm, h, w, padding_ratio=0.40)
        eye_mask_3ch = np.stack([eye_mask] * 3, axis=-1)

        blended = (orig_f * eye_mask_3ch
                   + blended.astype(np.float32) * (1.0 - eye_mask_3ch))
        return np.clip(blended, 0, 255).astype(np.uint8)
