"""
DSP Aging Module
================
Standalone classical DSP / image-processing aging simulation.
No deep learning required.

Architecture layer  : DSP Processing Layer
Depends on          : OpenCV, NumPy  (no Flask, no MediaPipe)

Public API
----------
    enhance_wrinkles(image, intensity)          -> np.ndarray
    apply_aged_skin_tone(image, intensity)      -> np.ndarray
    whiten_hair_approximation(image, intensity) -> np.ndarray
    apply_aging_effect(image, intensity=0.5)    -> np.ndarray  # main entry point

Design constraints
------------------
- All array operations use NumPy vectorised routines.
- No Python-level pixel loops (no for-row / for-column iteration).
- Single standard image (≤ 1024×1024) processes in well under 5 seconds.
- Input/output: BGR uint8, same spatial shape as the input.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _odd(n: int, minimum: int = 3) -> int:
    """Return n rounded up to the nearest odd integer, at least `minimum`."""
    n = max(minimum, int(n))
    return n if n % 2 == 1 else n + 1


def _soft_face_mask(h: int, w: int) -> np.ndarray:
    """
    Build a soft elliptical mask that covers the central face area.

    The ellipse is centred at (w/2, h*0.50) with radii ~40% of width and
    ~50% of height — empirically good for frontal portrait crops.
    A large Gaussian blur feathers the boundary so blended effects integrate
    seamlessly with the background / hair region.

    Returns a float32 mask in [0, 1].
    """
    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = w // 2, int(h * 0.50)
    cv2.ellipse(
        mask,
        (cx, cy),
        (int(w * 0.40), int(h * 0.50)),
        0, 0, 360,
        1.0, -1,
        lineType=cv2.LINE_AA,
    )
    # Feather radius ~12.5% of the shorter dimension for a smooth fade
    blur_k = _odd(min(h, w) // 8)
    return cv2.GaussianBlur(mask, (blur_k, blur_k), blur_k * 0.38)


# ---------------------------------------------------------------------------
# 1. Wrinkle enhancement
# ---------------------------------------------------------------------------

def enhance_wrinkles(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Enhance wrinkle-like high-frequency facial texture using Gaussian HPF
    blended with a Laplacian second-derivative layer.

    DSP theory
    ----------
    Unsharp-mask / Gaussian HPF
        HPF(x,y)  =  f(x,y) − G_σ * f(x,y)

        Subtracting a Gaussian-blurred version of the grayscale image from
        the original isolates the high-spatial-frequency residual: edges,
        fine lines, pores, and existing wrinkle creases.  This is equivalent
        to convolution with the high-pass kernel (δ − G_σ).

    Laplacian (∇² f)
        The Laplacian is the sum of second-order spatial derivatives:
            ∇²f = ∂²f/∂x² + ∂²f/∂y²
        It responds maximally to intensity ridges and valleys (wrinkle
        edges appear as a dark-bright double-band in the second derivative).
        Mixing a small Laplacian contribution adds geometric crease structure
        that the first-order HPF alone does not fully capture.

    Blending
        The combined texture layer is added to each BGR channel through NumPy
        broadcasting (no channel loop), gated by a soft face ellipse mask so
        the background is untouched.

    Parameters
    ----------
    image     : BGR uint8 input image.
    intensity : Aging strength in [0, 1].  At 0.5 the effect is visible but
                natural; at 1.0 it is strong.

    Returns
    -------
    BGR uint8 image with amplified skin texture.
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))
    h, w = image.shape[:2]

    # --- Step 1: Convert to grayscale — texture analysis is luminance-only ---
    # Working in grayscale halves memory use and avoids spurious colour edges.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # --- Step 2: Low-pass filter (Gaussian blur) to get the smooth baseline ---
    # σ = 2.0 removes detail at spatial periods shorter than ~12 px, which
    # corresponds to fine wrinkle/pore-scale texture on a typical face crop.
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0, sigmaY=2.0)

    # --- Step 3: First-order HPF residual (unsharp-mask) ----------------------
    # Values in approximately [−60, +60]; positive peaks at bright edges,
    # negative troughs at dark grooves (wrinkle shadows).
    hpf = gray - smooth

    # --- Step 4: Second-order Laplacian layer ---------------------------------
    # ksize=3 is the standard 3×3 discrete Laplacian kernel; larger kernels
    # produce a coarser, more global edge map that smears fine texture.
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

    # --- Step 5: Mix HPF and Laplacian ----------------------------------------
    # 65% first-order (overall texture) + 20% second-order (crease geometry).
    # Empirically this ratio produces the most natural-looking wrinkle sharpening
    # without introducing haloing around broader face features.
    texture = hpf * 0.65 + lap * 0.20

    # --- Step 6: Build a soft face-region mask --------------------------------
    # The feathered ellipse prevents the texture boost from bleeding into the
    # background, hair, or neck region.
    face_mask = _soft_face_mask(h, w)

    # --- Step 7: Add texture to all BGR channels via NumPy broadcasting -------
    # texture shape (H, W) → (H, W, 1) broadcasts across 3 channels at once.
    # strength = 0.85 × intensity prevents over-sharpening at intensity = 1.
    strength = intensity * 0.85
    result = image.astype(np.float32)
    result += texture[:, :, np.newaxis] * strength * face_mask[:, :, np.newaxis]

    return np.clip(result, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Aged skin tone
# ---------------------------------------------------------------------------

def apply_aged_skin_tone(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Simulate the colour and contrast characteristics of aged skin.

    Three effects are applied sequentially, all gated by a soft face mask:

    Effect 1 — Saturation reduction (HSV colour space)
        Aged skin has fewer active melanocytes and thinner dermis, resulting
        in paler, less saturated hues.  We scale the S channel down by up to
        25% at maximum intensity.

    Effect 2 — Brightness reduction (HSV colour space)
        A subtle 10% V-channel reduction on the face area mimics the slightly
        muted appearance of aged skin (reduced blood perfusion, less sebum).
        The effect is blended on the face mask only, preserving background.

    Effect 3 — Local contrast boost (CLAHE in L*a*b* space)
        CLAHE (Contrast-Limited Adaptive Histogram Equalisation) divides the
        image into small 8×8 tiles and equalises each locally.  This amplifies
        micro-contrast variations (pigmentation spots, uneven tone) that
        characterise aged skin, without globally blowing out bright regions
        (the clip-limit cap prevents over-amplification of noise).
        Processing in the L* channel of L*a*b* keeps hue and chroma unchanged.

    Parameters
    ----------
    image     : BGR uint8 input.
    intensity : Aging strength in [0, 1].

    Returns
    -------
    BGR uint8 image with aged skin colour and contrast.
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))
    h, w = image.shape[:2]

    # Shared face mask used across all three effects
    face_mask = _soft_face_mask(h, w)   # float32 in [0, 1]

    # ── Effect 1 & 2: Saturation + brightness in HSV ──────────────────────
    # HSV decouples colour (H), vibrancy (S), and lightness (V), making these
    # adjustments numerically independent of each other.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Saturation: uniform scale across the whole image.
    # scale_s ∈ [0.75, 1.0] — maximum 25% desaturation at intensity = 1.
    scale_s = 1.0 - 0.25 * intensity
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale_s, 0.0, 255.0)

    # Brightness: blend a scaled-down V channel on the face area only.
    # Outside the face mask (background) V is unchanged.
    bright_factor = 1.0 - 0.10 * intensity        # e.g. 0.95 at intensity=0.5
    # Per-pixel V scale: 1.0 outside mask, bright_factor at mask centre
    v_scale = 1.0 - face_mask * (1.0 - bright_factor)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0.0, 255.0)

    # Convert back to BGR for the CLAHE step
    bgr_after_hsv = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ── Effect 3: Local contrast boost via CLAHE on L* ────────────────────
    # clipLimit grows from 1.0 (subtle) to 2.8 (strong) with intensity.
    clip_limit = 1.0 + 1.8 * intensity
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

    lab = cv2.cvtColor(bgr_after_hsv, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])   # enhance L* only, keep a*, b*
    bgr_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)

    # Blend CLAHE result onto the HSV result, scaled by intensity and face mask.
    # Maximum 40% CLAHE contribution at intensity = 1 on the face centre —
    # keeps the effect natural and avoids an over-processed appearance.
    clahe_weight = face_mask * 0.40 * intensity           # shape (H, W)
    result = (
        bgr_after_hsv.astype(np.float32) * (1.0 - clahe_weight[:, :, np.newaxis])
        + bgr_clahe * clahe_weight[:, :, np.newaxis]
    )

    return np.clip(result, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. Hair whitening approximation
# ---------------------------------------------------------------------------

def whiten_hair_approximation(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Approximate hair greying without landmark or segmentation models.

    Why an approximation?
    ---------------------
    MediaPipe FaceMesh tracks only facial surface landmarks (468 points on
    the face, not scalp or hair).  For a purely classical DSP pipeline the
    practical alternative is a heuristic spatial-chromatic detector:

    Algorithm
    ---------
    Step 1 — Spatial prior (upper portrait zone)
        Hair is most likely to appear in the upper ~38% of a frontal portrait
        frame.  A linear fade ramp at the bottom of this zone softens the
        boundary so the effect doesn't cut off sharply mid-forehead.

    Step 2 — Chromatic hair detector
        Two conditions combined with element-wise multiplication (no loops):

        Dark-pixel mask  : pixels with grayscale value < 95 are candidate hair.
            Scale: soft ramp [0 → 1] over the range gray ∈ [95 → 20].
            Avoids hard thresholds that produce block artefacts.

        Low-saturation mask : pixels with HSV-S < 140 are less likely to be
            skin (skin in the HSV model has relatively high saturation).
            Combining this with the dark mask rejects tanned-skin pixels that
            happen to be dark.

        Combined: hair_mask = dark × low_sat × upper_zone

    Step 3 — Morphological cleanup
        MORPH_CLOSE fills small gaps within hair strands and removes isolated
        noise pixels (MORPH_CLOSE = dilation then erosion with a 9×9 ellipse).
        A subsequent Gaussian blur feathers the mask edges.

    Step 4 — Colour transform in HSV (vectorised)
        Desaturate  : S  → S × (1 − 0.90 × mask)   → grey / silver
        Lift value  : V  → V + (155 − V) × 0.70 × mask → brightens dark hair
        Both use NumPy arithmetic broadcasting — no per-pixel Python calls.

    Step 5 — Alpha blend
        result = original × (1 − blend) + grey × blend
        blend = clip(hair_mask × intensity, 0, 1)
        The user-facing `intensity` slider controls subtlety.

    Parameters
    ----------
    image     : BGR uint8 input.
    intensity : Greying strength in [0, 1].

    Returns
    -------
    BGR uint8 image with approximate hair greying.
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))
    h, w = image.shape[:2]

    # --- Step 1: Spatial prior — upper portrait zone -------------------------
    hair_zone_rows = int(h * 0.38)
    hair_zone = np.zeros((h, w), dtype=np.float32)
    hair_zone[:hair_zone_rows, :] = 1.0

    # Linear fade ramp over the bottom ~17% of the hair zone so the effect
    # doesn't create a hard horizontal cut-off at the forehead boundary.
    fade_rows = max(20, hair_zone_rows // 6)
    fade_start = hair_zone_rows - fade_rows
    fade_ramp = np.linspace(1.0, 0.0, fade_rows, dtype=np.float32)
    hair_zone[fade_start:hair_zone_rows, :] = fade_ramp[:, np.newaxis]

    # --- Step 2: Chromatic hair detector (vectorised) -------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Dark-pixel soft mask: 1.0 at gray ≤ 20, 0.0 at gray ≥ 95.
    # np.clip ensures values outside the ramp saturate cleanly.
    dark_mask = np.clip((95.0 - gray) / 75.0, 0.0, 1.0)

    # Low-saturation soft mask: 1.0 at S = 0, 0.0 at S = 140.
    low_sat_mask = np.clip((140.0 - hsv[:, :, 1]) / 140.0, 0.0, 1.0)

    # Combine conditions: element-wise product = logical AND with soft weights
    hair_mask = dark_mask * low_sat_mask * hair_zone

    # --- Step 3: Morphological cleanup + edge feathering ---------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    hair_u8 = np.clip(hair_mask * 255.0, 0, 255).astype(np.uint8)
    hair_u8 = cv2.morphologyEx(hair_u8, cv2.MORPH_CLOSE, kernel)
    hair_mask = hair_u8.astype(np.float32) / 255.0

    blur_k = _odd(min(h, w) // 16)
    hair_mask = cv2.GaussianBlur(hair_mask, (blur_k, blur_k), blur_k * 0.40)

    # --- Step 4: Colour transform — desaturate and brighten in HSV ----------
    hsv_grey = hsv.copy()

    # Desaturate: multiplied by (1 − 0.90 × mask); full desaturation at mask=1
    # produces neutral grey; partial desaturation yields a silver-grey tone.
    hsv_grey[:, :, 1] = np.clip(
        hsv[:, :, 1] * (1.0 - hair_mask * 0.90),
        0.0, 255.0
    )

    # Lift brightness: interpolate V toward target_brightness = 155 (mid-grey).
    # The factor 0.70 prevents fully blowing out very dark strands in one pass.
    target_brightness = 155.0
    v_lifted = hsv[:, :, 2] + (target_brightness - hsv[:, :, 2]) * hair_mask * 0.70
    hsv_grey[:, :, 2] = np.clip(v_lifted, 0.0, 255.0)

    grey_bgr = cv2.cvtColor(hsv_grey.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # --- Step 5: Alpha-blend greyed result onto original ----------------------
    # `intensity` scales the blend weight so the user controls subtlety.
    # shape (H, W) → (H, W, 1) broadcasts across BGR channels.
    blend = np.clip(hair_mask * intensity, 0.0, 1.0)[:, :, np.newaxis]
    result = image.astype(np.float32) * (1.0 - blend) + grey_bgr * blend

    return np.clip(result, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 4. Main orchestrator
# ---------------------------------------------------------------------------

def apply_aging_effect(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Apply all three aging effects in sequence.

    Pipeline (DSP Processing Layer)
    --------------------------------
    Stage 1 — enhance_wrinkles
        Gaussian HPF + Laplacian amplification of skin texture.
        Gated by a soft face-region ellipse mask.

    Stage 2 — apply_aged_skin_tone
        HSV saturation / brightness reduction then CLAHE local contrast boost
        in L*a*b* space.  Keeps the result natural — not exaggerated.

    Stage 3 — whiten_hair_approximation
        Upper-region dark-pixel detection → HSV desaturation + value lift →
        alpha-blend.  Subtly shifts dark hair toward grey/white.

    The same `intensity` value in [0, 1] drives all three stages so a single
    slider gives coherent control over the overall aging strength.

    Parameters
    ----------
    image     : BGR uint8 input image (H × W × 3).
    intensity : Overall aging strength.  0.0 = no change, 1.0 = maximum.
                Typical usage: 0.3 – 0.7 for realistic-looking results.

    Returns
    -------
    BGR uint8 image of the same shape, with combined aging applied.

    Example
    -------
    >>> import cv2
    >>> from dsp_aging import apply_aging_effect
    >>> img = cv2.imread("face.jpg")
    >>> aged = apply_aging_effect(img, intensity=0.6)
    >>> cv2.imwrite("aged.jpg", aged)
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))

    # Stage 1: High-frequency texture enhancement (wrinkles)
    result = enhance_wrinkles(image, intensity)

    # Stage 2: Colour and contrast transformation (aged skin tone)
    result = apply_aged_skin_tone(result, intensity)

    # Stage 3: Hair greying approximation (upper region heuristic)
    result = whiten_hair_approximation(result, intensity)

    return result
