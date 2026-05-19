"""
aging.py — Classical DSP Aging Module  (v2)
============================================
Improved standalone aging / skin-simulation module for the FaceWarp DSP
project.  Called from Flask /api/aging when method == "classical".

Key improvements over v1
------------------------
* Region-specific wrinkle masks driven by MediaPipe landmarks (or spatial
  priors when landmarks are unavailable):
    forehead, crow's-feet, under-eye, nasolabial folds, mouth corners.
* Multi-scale Difference-of-Gaussians (DoG) replaces single-scale Laplacian.
  DoG captures wrinkle-scale texture without the harsh ring artefacts
  produced by a small-ksize Laplacian on a uniform skin surface.
* Multi-octave fractal / fBm noise adds organic fine-surface roughness where
  input skin is already smooth — no drawn geometric lines, no grid artefacts.
* Soft Gaussian-feathered masks everywhere — no hard mask edges visible.
* Yellow/brown tone shift in CIE L*a*b* (b* +5, a* -3 at max intensity)
  replicates the jaundice-like hue change of aged skin.
* Hair whitening uses a two-tier dark-pixel detector (V<80 strong, V<130 weak)
  with face-ellipse exclusion and a wider spatial zone (up to 42% height),
  giving visible greying without false-positive skin contamination.

Public API
----------
    enhance_wrinkles(image, landmarks=None, intensity)       -> BGR uint8
    apply_aged_skin_tone(image, intensity)                   -> BGR uint8
    whiten_hair_approximation(image, landmarks=None, intensity) -> BGR uint8
    apply_aging_effect(image, landmarks=None, intensity=0.6) -> BGR uint8

Notes
-----
* No geometric drawing (cv2.line / cv2.polylines) is ever applied to the
  output image — landmark and triangle overlays stay in the visualisation
  pipeline only.
* Fully vectorised — no Python pixel loops.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Internal utilities
# ─────────────────────────────────────────────────────────────────────────────

def _odd(n: int, minimum: int = 3) -> int:
    """Round n up to the nearest odd integer (cv2 kernel-size requirement)."""
    n = max(minimum, int(n))
    return n if n % 2 == 1 else n + 1


def _face_ellipse_mask(h: int, w: int) -> np.ndarray:
    """
    Soft float32 ellipse mask [0, 1] for the central face region.

    Centre (w/2, h*0.52), axes 42% × 50% of image dimensions.
    A large-sigma Gaussian feathers the boundary so effects fade out smoothly.
    """
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(
        mask,
        (w // 2, int(h * 0.52)),
        (int(w * 0.42), int(h * 0.50)),
        0, 0, 360, 1.0, -1, lineType=cv2.LINE_AA,
    )
    k = _odd(min(h, w) // 8)
    return cv2.GaussianBlur(mask, (k, k), k * 0.38)


def _lm(landmarks, idx: int) -> Optional[np.ndarray]:
    """Pixel (x, y) for a MediaPipe 468-point landmark index, or None."""
    if landmarks is None or len(landmarks) <= idx:
        return None
    p = landmarks[idx]
    return np.array([float(p["x"]), float(p["y"])], dtype=np.float32)


def _mean_lm(landmarks, indices: list) -> Optional[np.ndarray]:
    """Mean pixel position of several landmark indices, or None if any missing."""
    pts = [_lm(landmarks, i) for i in indices]
    if any(p is None for p in pts):
        return None
    return np.mean(pts, axis=0).astype(np.float32)


def _soft_circle(h: int, w: int, cx: float, cy: float, radius: float) -> np.ndarray:
    """Float32 Gaussian-feathered circular mask."""
    mask = np.zeros((h, w), dtype=np.float32)
    r = max(2, int(round(radius)))
    cv2.circle(mask, (int(round(cx)), int(round(cy))), r, 1.0, -1, lineType=cv2.LINE_AA)
    k = _odd(max(5, r))
    return cv2.GaussianBlur(mask, (k, k), k * 0.42)


def _soft_hband(h: int, w: int, y_top: float, y_bot: float,
                feather: float = 0.08) -> np.ndarray:
    """
    Horizontal band mask with linear fade at both edges.
    y_top / y_bot are fractions of image height [0, 1].
    """
    yy   = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, np.newaxis]
    up   = np.clip((yy - (y_top - feather)) / (feather + 1e-6), 0.0, 1.0)
    down = np.clip(((y_bot + feather) - yy) / (feather + 1e-6), 0.0, 1.0)
    return np.broadcast_to(np.minimum(up, down), (h, w)).copy()


def _fractal_noise(h: int, w: int, seed: int = 0) -> np.ndarray:
    """
    Multi-octave fBm noise in [-1, 1].

    Four octaves of Gaussian-blurred Gaussian random noise are combined with
    geometrically decreasing amplitudes.  This approximates fractional
    Brownian motion: natural roughness where fine detail has less energy
    than coarse features.  Normalised to exactly [-1, 1].
    """
    rng  = np.random.default_rng(seed)
    base = rng.standard_normal((h, w)).astype(np.float32)
    result = np.zeros((h, w), dtype=np.float32)
    for sigma, amp in [(0.8, 0.45), (2.0, 0.30), (5.0, 0.16), (10.0, 0.09)]:
        k = _odd(max(3, int(sigma * 5)))
        result += cv2.GaussianBlur(base, (k, k), sigma) * amp
    mn, mx = result.min(), result.max()
    if mx > mn:
        result = (result - mn) / (mx - mn) * 2.0 - 1.0
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Region mask builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_wrinkle_region_masks(h: int, w: int, landmarks) -> dict:
    """
    Build per-region soft float32 wrinkle-placement masks.

    Returns dict keys: forehead, eye_corners, under_eye, nasolabial,
    mouth_corners.  Each value is (H, W) float32 in [0, 1].

    When >= 468 landmark points are provided the regions are computed from
    actual landmark pixel positions.  Otherwise spatial priors are used.
    """
    use_lm = landmarks is not None and len(landmarks) >= 468

    if use_lm:
        l_outer = _lm(landmarks, 33)
        l_inner = _lm(landmarks, 133)
        r_inner = _lm(landmarks, 362)
        r_outer = _lm(landmarks, 263)
        eye_span = 0.0
        if l_outer is not None and r_outer is not None:
            eye_span = float(np.linalg.norm(r_outer - l_outer))

        # ── Forehead ───────────────────────────────────────────────────────
        brow_l = _mean_lm(landmarks, [70, 63, 105, 66, 107])
        brow_r = _mean_lm(landmarks, [336, 296, 334, 293, 300])
        top_lm = _lm(landmarks, 10)
        if brow_l is not None and brow_r is not None and top_lm is not None:
            brow_y   = min(float(brow_l[1]), float(brow_r[1])) / h
            top_y    = max(0.0, float(top_lm[1]) / h - 0.03)
            forehead = _soft_hband(h, w, top_y, brow_y - 0.01, feather=0.05)
        else:
            forehead = _soft_hband(h, w, 0.06, 0.36, feather=0.05)

        # ── Crow's feet ────────────────────────────────────────────────────
        r_crow     = max(14.0, eye_span * 0.13)
        eye_corners = np.zeros((h, w), dtype=np.float32)
        for pt in (l_outer, r_outer):
            if pt is not None:
                eye_corners = np.maximum(
                    eye_corners,
                    _soft_circle(h, w, float(pt[0]), float(pt[1]), r_crow),
                )

        # ── Under-eye crescents ────────────────────────────────────────────
        under_eye = np.zeros((h, w), dtype=np.float32)
        for outer_pt, inner_pt in ((l_outer, l_inner), (r_inner, r_outer)):
            if outer_pt is not None and inner_pt is not None:
                cx  = float(outer_pt[0] + inner_pt[0]) * 0.5
                cy  = float(max(outer_pt[1], inner_pt[1])) + eye_span * 0.06
                ew  = abs(float(outer_pt[0]) - float(inner_pt[0]))
                tmp = np.zeros((h, w), dtype=np.float32)
                cv2.ellipse(
                    tmp,
                    (int(round(cx)), int(round(cy))),
                    (max(2, int(ew * 0.52)), max(2, int(eye_span * 0.055))),
                    0, 0, 360, 1.0, -1, lineType=cv2.LINE_AA,
                )
                k = _odd(max(5, int(eye_span * 0.05)))
                under_eye = np.maximum(
                    under_eye, cv2.GaussianBlur(tmp, (k, k), k * 0.40)
                )

        # ── Nasolabial folds ───────────────────────────────────────────────
        nose_l  = _lm(landmarks, 98)
        nose_r  = _lm(landmarks, 327)
        mouth_l = _lm(landmarks, 61)
        mouth_r = _lm(landmarks, 291)
        nasolabial = np.zeros((h, w), dtype=np.float32)
        r_naso = max(10.0, eye_span * 0.10)
        for s_pt, e_pt in ((nose_l, mouth_l), (nose_r, mouth_r)):
            if s_pt is not None and e_pt is not None:
                cx     = float(s_pt[0] + e_pt[0]) * 0.5
                cy     = float(s_pt[1] + e_pt[1]) * 0.5
                length = float(np.linalg.norm(e_pt - s_pt))
                angle  = float(np.degrees(
                    np.arctan2(float(e_pt[1]) - float(s_pt[1]),
                               float(e_pt[0]) - float(s_pt[0]))))
                tmp = np.zeros((h, w), dtype=np.float32)
                cv2.ellipse(
                    tmp,
                    (int(round(cx)), int(round(cy))),
                    (max(2, int(length * 0.52)), max(2, int(r_naso))),
                    angle, 0, 360, 1.0, -1, lineType=cv2.LINE_AA,
                )
                k = _odd(max(7, int(r_naso * 1.6)))
                nasolabial = np.maximum(
                    nasolabial, cv2.GaussianBlur(tmp, (k, k), k * 0.40)
                )

        # ── Mouth corners ─────────────────────────────────────────────────
        r_mouth = max(10.0, eye_span * 0.07)
        mouth_corners = np.zeros((h, w), dtype=np.float32)
        for pt in (mouth_l, mouth_r):
            if pt is not None:
                mouth_corners = np.maximum(
                    mouth_corners,
                    _soft_circle(h, w, float(pt[0]), float(pt[1]), r_mouth),
                )

    else:
        # ── Spatial priors for generic frontal portrait ────────────────────
        forehead = _soft_hband(h, w, 0.08, 0.37, feather=0.05)

        r_crow = max(12.0, min(h, w) * 0.075)
        eye_corners = np.maximum(
            _soft_circle(h, w, w * 0.24, h * 0.41, r_crow),
            _soft_circle(h, w, w * 0.76, h * 0.41, r_crow),
        )

        under_eye = _soft_hband(h, w, 0.41, 0.50, feather=0.035)
        eye_zone  = np.zeros((h, w), dtype=np.float32)
        cv2.rectangle(eye_zone, (0, 0),             (int(w * 0.44), h), 1.0, -1)
        cv2.rectangle(eye_zone, (int(w * 0.56), 0), (w, h),             1.0, -1)
        k_ez  = _odd(w // 14)
        eye_zone  = cv2.GaussianBlur(eye_zone, (k_ez, k_ez), k_ez * 0.40)
        under_eye = under_eye * eye_zone

        nasolabial = _soft_hband(h, w, 0.54, 0.73, feather=0.055)
        ctr_excl   = np.zeros((h, w), dtype=np.float32)
        cv2.rectangle(ctr_excl, (int(w * 0.38), 0), (int(w * 0.62), h), 1.0, -1)
        k_ctr = _odd(w // 10)
        ctr_excl   = cv2.GaussianBlur(ctr_excl, (k_ctr, k_ctr), k_ctr * 0.40)
        nasolabial = nasolabial * (1.0 - ctr_excl)

        r_mouth = max(10.0, min(h, w) * 0.055)
        mouth_corners = np.maximum(
            _soft_circle(h, w, w * 0.37, h * 0.68, r_mouth),
            _soft_circle(h, w, w * 0.63, h * 0.68, r_mouth),
        )

    return {
        "forehead":      np.clip(forehead,      0.0, 1.0),
        "eye_corners":   np.clip(eye_corners,   0.0, 1.0),
        "under_eye":     np.clip(under_eye,     0.0, 1.0),
        "nasolabial":    np.clip(nasolabial,    0.0, 1.0),
        "mouth_corners": np.clip(mouth_corners, 0.0, 1.0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1.  enhance_wrinkles
# ─────────────────────────────────────────────────────────────────────────────

def enhance_wrinkles(image: np.ndarray,
                    landmarks=None,
                    intensity: float = 0.5) -> np.ndarray:
    """
    Amplify wrinkle-like texture using multi-scale DoG and procedural fBm
    noise, applied only to anatomically relevant facial regions.

    Design principle: texture amplification without drawn lines
    -----------------------------------------------------------
    Instead of drawing geometric strokes (which look grid-like), this function
    amplifies texture that is *already present* in the image using Difference-
    of-Gaussians (DoG) high-pass filtering, then supplements with fractal noise
    where skin may be too smooth.  All contributions pass through soft
    Gaussian-feathered region masks — no hard boundaries anywhere.

    DSP Pipeline
    ────────────
    Step 1 — Work in L*a*b* luminance (L*) only; a* and b* are unchanged.
    Step 2 — Multi-scale DoG (Difference of Gaussians):
               fine_DoG   = L − Gauss(L, σ=0.9)  captures pores / fine lines
               medium_DoG = L − Gauss(L, σ=2.8)  captures wrinkle ridges
               Both are clipped to ±20 L* units to suppress halo spikes.
    Step 3 — Fractal fBm noise (see _fractal_noise); amplitude ±3.5 L* units.
    Step 4 — Per-region weighted mix (forehead: medium; crow's-feet: fine+noise;
               under-eye: medium+noise; nasolabial: medium; mouth corners: mixed).
    Step 5 — Composite multiplied by face ellipse mask (gates background).
    Step 6 — Additive blend: L*_out = L*_in + composite × 9 × intensity.
               max_gain=9 L* units prevents harsh lines.
    Step 7 — Reconstruct BGR from modified L*a*b*.
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))
    if intensity < 0.01:
        return image.copy()

    h, w = image.shape[:2]

    # Step 1: L*a*b* luminance channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L   = lab[:, :, 0]

    # Step 2: multi-scale DoG — clip ±20 to prevent halo spikes at hard edges
    fine_dog   = np.clip(L - cv2.GaussianBlur(L, (0, 0), sigmaX=0.9), -20.0, 20.0)
    medium_dog = np.clip(L - cv2.GaussianBlur(L, (0, 0), sigmaX=2.8), -20.0, 20.0)

    # Step 3: reproducible fractal noise seeded by image content.
    # Amplitude 10 L* units so noise adds visible skin roughness on smooth faces.
    img_seed = int(image.flat[len(image.flat) // 2]) ^ int(h * 397 + w * 149)
    noise    = _fractal_noise(h, w, seed=img_seed) * 10.0

    # Step 4: region masks and per-region texture
    # (fine_weight, medium_weight, noise_weight, amplitude_scale)
    # Amplitudes increased ~30% so wrinkles and age texture are clearly visible.
    regions = _build_wrinkle_region_masks(h, w, landmarks)
    region_params = {
        "forehead":      (0.25, 0.55, 0.20, 1.82),
        "eye_corners":   (0.50, 0.28, 0.22, 2.34),
        "under_eye":     (0.35, 0.42, 0.23, 1.69),
        "nasolabial":    (0.22, 0.62, 0.16, 2.08),
        "mouth_corners": (0.40, 0.38, 0.22, 1.56),
    }
    composite = np.zeros((h, w), dtype=np.float32)
    for name, mask in regions.items():
        wf, wm, wn, amp = region_params[name]
        composite += mask * (fine_dog * wf + medium_dog * wm + noise * wn) * amp

    # Step 5: gate to face ellipse
    composite *= _face_ellipse_mask(h, w)

    # Step 6: additive blend.
    # Gain raised to 26 L* units so wrinkles and age texture are visibly stronger.
    lab[:, :, 0] = np.clip(L + composite * (26.0 * intensity), 0.0, 255.0)

    # Step 7: reconstruct BGR
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  apply_aged_skin_tone
# ─────────────────────────────────────────────────────────────────────────────

def apply_aged_skin_tone(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Simulate aged skin colour and contrast in three stages.

    Stage A — Saturation reduction (HSV)
        Aged skin contains fewer active melanocytes and shows reduced blood
        perfusion, both of which lower colour saturation.  We scale the S
        channel of the HSV representation by (1 − 0.25 × intensity), giving
        up to 25% desaturation at maximum intensity.

    Stage B — Brightness reduction on the face region (HSV)
        The V channel is reduced proportionally to `intensity` (up to 10%)
        inside the face ellipse, simulating the slightly muted tone of aged
        skin.  The reduction is spatially weighted by the face mask so the
        background is not affected.

    Stage C — CLAHE local contrast boost (L*a*b* → L* channel only)
        CLAHE (Contrast-Limited Adaptive Histogram Equalisation) divides the
        image into non-overlapping 8×8 tiles and equalises each tile's
        histogram independently, with a clip limit that prevents noise
        amplification.  Applied to the L* (luminance) channel of CIE L*a*b*,
        it increases micro-contrast (uneven pigmentation, age spots) without
        any hue shift.  The CLAHE output is alpha-blended onto the HSV result
        using the face mask so that only the skin area is affected.

    Parameters
    ----------
    image     : BGR uint8 source image.
    intensity : Effect strength in [0, 1].

    Returns
    -------
    BGR uint8 image with aged skin colour and contrast.
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))
    h, w = image.shape[:2]
    face_mask = _face_ellipse_mask(h, w)   # (H, W) float32 [0, 1]

    # ── Stage A & B: saturation + brightness adjustment in HSV ───────────────
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Saturation: uniform scale across the whole image (skin and background).
    # Raised to 50% max desaturation for more visible color shift at higher intensity.
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 - 0.50 * intensity), 0.0, 255.0)

    # Brightness: reduce only inside the face mask (background stays unchanged)
    # v_weight blends from 1.0 (background) to bright_factor (face centre)
    bright_factor = 1.0 - 0.13 * intensity
    v_weight = 1.0 - face_mask * (1.0 - bright_factor)     # shape (H, W)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_weight, 0.0, 255.0)

    bgr_hsv = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ── Stage B-ii: yellow/brown tone shift in L*a*b* ────────────────────────
    # Aged skin has a slightly yellow/jaundiced hue: a* (green-red) shifts
    # slightly warm and b* (blue-yellow) shifts toward yellow.
    # Applied inside the face ellipse only. Increased shift for more visible aging.
    lab_tone = cv2.cvtColor(bgr_hsv, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_tone[:, :, 1] = np.clip(
        lab_tone[:, :, 1] + face_mask * (3.8 * intensity), 0.0, 255.0
    )  # a* shift: more warmth
    lab_tone[:, :, 2] = np.clip(
        lab_tone[:, :, 2] + face_mask * (6.5 * intensity), 0.0, 255.0
    )  # b* shift: more yellow-brown
    bgr_hsv = cv2.cvtColor(lab_tone.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ── Stage C: CLAHE local contrast on L* channel ──────────────────────────
    # clipLimit: 1.0 (subtle) → 3.0 (strong) with intensity, for more age spots
    clahe = cv2.createCLAHE(clipLimit=1.0 + 2.0 * intensity, tileGridSize=(8, 8))

    lab = cv2.cvtColor(bgr_hsv, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])   # modify L* only; a* b* unchanged
    bgr_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)

    # Alpha-blend CLAHE result onto the HSV result using face mask.
    # Maximum 50% CLAHE weight at intensity = 1 for stronger age spot effect.
    clahe_w = face_mask * 0.50 * intensity                  # (H, W)
    result = (
        bgr_hsv.astype(np.float32) * (1.0 - clahe_w[:, :, np.newaxis])
        + bgr_clahe             * clahe_w[:, :, np.newaxis]
    )

    return np.clip(result, 0.0, 255.0).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 3a.  apply_whitening  — low-level utility: blend hair → gray given a mask
# ─────────────────────────────────────────────────────────────────────────────

def apply_whitening(img: np.ndarray,
                    mask: np.ndarray,
                    intensity: float = 1.0) -> np.ndarray:
    """Blend *img* toward gray-hair using *mask* as per-pixel alpha.

    Parameters
    ----------
    img       : BGR uint8 image.
    mask      : float32 (H, W) in [0, 1] — 1 = full effect, 0 = no change.
    intensity : unused in the blend formula; kept for API compatibility.

    Returns
    -------
    BGR uint8 image — only the masked hair region changes color.
    """
    img_f  = img.astype(np.float32)
    mask   = np.clip(mask, 0.0, 1.0)
    mask_3 = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Build gray-hair target in HSV: desaturate + subtle brightness lift.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    s_ch = s_ch * 0.4
    v_ch = np.minimum(v_ch + 40.0, 255.0)
    hsv_white = cv2.merge([h_ch, s_ch, v_ch])
    whitened  = cv2.cvtColor(
        np.clip(hsv_white, 0, 255).astype(np.uint8),
        cv2.COLOR_HSV2BGR,
    ).astype(np.float32)

    # Alpha scales with intensity so the caller controls blend strength.
    alpha  = mask_3 * float(intensity) * 0.45
    result = img_f * (1.0 - alpha) + whitened * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 3b.  whiten_hair_approximation
# ─────────────────────────────────────────────────────────────────────────────

def whiten_hair_approximation(image: np.ndarray,
                              landmarks=None,
                              intensity: float = 0.5,
                              _debug: bool = False) -> np.ndarray:
    """
    Hair greying that works for dark, light, AND colored hair.

    Detection strategy
    ──────────────────
    Primary: spatial zone (upper 60%) + skin exclusion.
             Everything in the zone that is NOT skin is treated as potential
             hair.  Texture / darkness signals then BOOST the mask weight but
             do not gate it — so uniform-colored fills (and real light/blonde
             hair with low texture) are still detected.

    Signals that boost confidence (0..1, combined via soft-OR):
      • Texture  — |Laplacian| catches hair strands (dark AND light).
      • Darkness — V < 180 catches black / brown hair.
      • Gray     — low-S + high-V catches already-gray / blonde hair.

    HSV modification (applied only where hair_mask > 0)
    ────────────────────────────────────────────────────
      S  : adaptive desaturation — base 65%, +20% for vivid colored hair.
      V  : ONLY lifted (never lowered).  Target = 175 for dark hair, unchanged
           for hair that is already brighter than 175.

    Debug
    ─────
    _debug=True → print pixel counts + save debug_mask.jpg.
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))
    if intensity < 0.01:
        return image.copy()

    h, w = image.shape[:2]
    total_px = h * w

    # ── Step 1: spatial zone (upper 60%, 30% fade ramp) ─────────────────────
    hair_zone_frac = 0.60
    if landmarks is not None and len(landmarks) >= 468:
        top_lm = _lm(landmarks, 10)
        if top_lm is not None:
            hair_zone_frac = float(np.clip(float(top_lm[1]) / h + 0.20, 0.45, 0.70))

    zone_rows  = max(1, int(h * hair_zone_frac))
    hair_zone  = np.zeros((h, w), dtype=np.float32)
    hair_zone[:zone_rows, :] = 1.0
    fade_rows  = max(12, int(zone_rows * 0.30))
    fade_start = max(0, zone_rows - fade_rows)
    hair_zone[fade_start:zone_rows, :] = np.linspace(
        1.0, 0.0, zone_rows - fade_start, dtype=np.float32
    )[:, np.newaxis]

    # ── Step 2: HSV channels ─────────────────────────────────────────────────
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    V   = hsv[:, :, 2]
    S   = hsv[:, :, 1]
    H   = hsv[:, :, 0]

    # ── Step 3: confidence-boosting signals ──────────────────────────────────
    # 3a. Texture via Laplacian (works for any color hair)
    gray       = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k_pre      = _odd(max(3, min(h, w) // 80))
    gray_sm    = cv2.GaussianBlur(gray, (k_pre, k_pre), 0)
    lap        = cv2.Laplacian(gray_sm, cv2.CV_32F, ksize=3)
    tex_abs    = np.abs(lap)
    tex_p98    = float(np.percentile(tex_abs, 98)) + 1e-6
    tex_norm   = np.clip(tex_abs / tex_p98, 0.0, 1.0)
    # Ramp 0.15→0.35.  Texture is a MULTIPLIER, not an independent signal, so
    # this threshold only needs to separate hair strands from smooth surfaces.
    texture_sig = np.clip((tex_norm - 0.15) / 0.20, 0.0, 1.0)

    # 3b. Dark-hair signal: V < 180  (black / brown hair)
    dark_sig = np.clip(1.0 - np.clip((V - 140.0) / 40.0, 0.0, 1.0), 0.0, 1.0)

    # 3c. Already-gray / blonde: low S, V in 80-185.
    # V ceiling at 185 explicitly excludes bright walls (V ≥ 185 → gray_sig → 0).
    gray_hair_s = np.clip(1.0 - np.clip((S - 15.0) / 20.0, 0.0, 1.0), 0.0, 1.0)
    gray_hair_v = (np.clip((V - 80.0)  / 60.0, 0.0, 1.0)    # rise  V ∈ 80-140
                   * np.clip((185.0 - V) / 15.0, 0.0, 1.0))  # fall  V ∈ 170-185 → 0
    gray_sig     = gray_hair_s * gray_hair_v

    # Combine: color signals (dark + gray) form the BASE; texture BOOSTS the base
    # but CANNOT fire on its own.  This prevents a textured wall from registering
    # as hair even when V / S values are near-background range.
    base       = np.clip(dark_sig * 0.60 + gray_sig * 0.50, 0.0, 1.0)
    confidence = np.clip(base * (1.0 + texture_sig * 0.80), 0.0, 1.0)
    boost      = confidence

    if _debug:
        print(f"[hair] confidence>0.3: {int(np.sum(confidence>0.3))} ({100*np.sum(confidence>0.3)/total_px:.1f}%)")

    # ── Step 4: face-center ellipse (computed first — used in skin exclusion) ─
    # Center (w/2, h*0.62), semi-axes (w*0.32, h*0.30) → top at h*0.32
    face_oval = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(
        face_oval,
        (w // 2, int(h * 0.62)),
        (max(1, int(w * 0.32)), max(1, int(h * 0.30))),
        0, 0, 360, 1.0, -1, lineType=cv2.LINE_AA,
    )
    k_fo = _odd(max(11, min(h, w) // 12))
    face_oval_soft = cv2.GaussianBlur(face_oval, (k_fo, k_fo), k_fo * 0.40) * 0.65
    face_weight = np.clip(1.0 - face_oval_soft, 0.0, 1.0)

    # ── Step 5: position-aware skin exclusion ────────────────────────────────
    # In the hair zone OUTSIDE the face oval, warm pixels are more likely
    # auburn/red hair than skin → reduce exclusion strength by up to 55%.
    # Inside the face oval → full exclusion (prevents forehead/cheek leakage).
    skin_hue  = np.clip(1.0 - np.abs(H - 13.0) / 15.0, 0.0, 1.0)
    skin_sat  = np.clip((S - 20.0) / 50.0, 0.0, 1.0)
    skin_val  = np.clip((V - 40.0) / 60.0, 0.0, 1.0)
    skin_prob = skin_hue * skin_sat * skin_val
    k_sk = _odd(max(7, min(h, w) // 18))
    skin_prob = cv2.GaussianBlur(skin_prob, (k_sk, k_sk), k_sk * 0.40)

    hair_outside_face = hair_zone * np.clip(1.0 - face_oval_soft, 0.0, 1.0)
    skin_excl_str = np.clip(1.10 - hair_outside_face * 0.60, 0.55, 1.10)
    non_skin = np.clip(1.0 - skin_prob * skin_excl_str, 0.0, 1.0)

    # ── Combine ───────────────────────────────────────────────────────────────
    raw_mask = np.clip(boost * hair_zone * non_skin * face_weight, 0.0, 1.0)

    if _debug:
        print(f"[hair] raw_mask>0.1: {int(np.sum(raw_mask>0.1))} ({100*np.sum(raw_mask>0.1)/total_px:.1f}%)")

    # Auto-relax: if fewer than 0.5% of pixels survive all gates, relax only
    # the pixel signals (lower thresholds) — do NOT fall back to a rectangular
    # spatial-only mask, which would create a visible gray rectangle artifact.
    if np.sum(raw_mask > 0.1) < total_px * 0.005:
        # Re-compute with relaxed thresholds, keeping texture as multiplier only.
        dark_sig2  = np.clip(1.0 - np.clip((V - 160.0) / 40.0, 0.0, 1.0), 0.0, 1.0)
        gray_sig2  = np.clip(1.0 - np.clip((S - 60.0)  / 40.0, 0.0, 1.0), 0.0, 1.0) * gray_hair_v
        base2      = np.clip(dark_sig2 * 0.60 + gray_sig2 * 0.50, 0.0, 1.0)
        conf2      = np.clip(base2 * (1.0 + texture_sig * 0.80), 0.0, 1.0)
        raw_mask   = np.clip(conf2 * hair_zone * non_skin * face_weight, 0.0, 1.0)
        if _debug:
            print(f"[hair] AUTO-RELAX applied. raw_mask>0.1 now: {int(np.sum(raw_mask>0.1))}")

    # ── Step 6: morphological fill + dilation ────────────────────────────────
    mask_u8 = np.clip(raw_mask * 255.0, 0, 255).astype(np.uint8)
    k_close = max(9, min(h, w) // 22)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    k_dil   = max(5, min(h, w) // 28)
    dil_k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dil, k_dil))
    # No dilation — keep the mask tight to the detected hair region.
    hair_mask = mask_u8.astype(np.float32) / 255.0

    # ── Step 7: large Gaussian feather (≥31 px) ──────────────────────────────
    blur_k    = _odd(max(21, min(h, w) // 22))
    hair_mask = cv2.GaussianBlur(hair_mask, (blur_k, blur_k), blur_k * 0.45)
    hair_mask = np.clip(hair_mask, 0.0, 1.0)

    if _debug:
        print(f"[hair] hair_mask>0.1: {int(np.sum(hair_mask>0.1))}")
        try:
            import os
            dbg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_mask.jpg")
            cv2.imwrite(dbg, (hair_mask * 255).astype(np.uint8))
            print(f"[hair] mask saved: {dbg}")
        except Exception as _e:
            print(f"[hair] mask save failed: {_e}")

    # ── Steps 8-9: HSV modification + alpha blend via apply_whitening ─────────
    # apply_whitening desaturates S × 0.4 and lifts V + 80 in HSV space, then
    # blends using hair_mask * intensity as the per-pixel alpha.
    return apply_whitening(image, hair_mask, intensity=intensity)



# ─────────────────────────────────────────────────────────────────────────────
# 4.  apply_aging_effect  — main entry point
# ─────────────────────────────────────────────────────────────────────────────

def apply_aging_effect(image: np.ndarray,
                      landmarks=None,
                      intensity: float = 0.6) -> np.ndarray:
    """
    Apply all three classical DSP aging effects in sequence.

    Processing order
    ────────────────
    1. apply_aged_skin_tone  — HSV desaturation + L*a*b* yellow/brown tone
                               shift + face-gated CLAHE contrast boost.
    2. enhance_wrinkles      — multi-scale DoG + fBm noise, region-specific
                               Gaussian-feathered masks.  No geometric lines.
    3. whiten_hair_approximation — upper-zone dark-pixel hair greying.

    The final output contains NO Delaunay triangle lines, NO landmark dot
    overlays, and NO drawn geometric strokes of any kind.

    Parameters
    ----------
    image     : BGR uint8 input (H × W × 3).
    landmarks : Optional list of dicts {"x": int, "y": int} from MediaPipe
                468-point FaceMesh.  Enables landmark-guided region masks.
                Pass None to use spatial priors.
    intensity : Overall aging strength in [0, 1].  Recommended: 0.3–0.7.

    Returns
    -------
    BGR uint8 aged image, same shape as input.

    Flask usage
    -----------
        from aging import apply_aging_effect
        aged = apply_aging_effect(image, landmarks=lm_points, intensity=strength)
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))
    if intensity < 0.01:
        return image.copy()

    result = apply_aged_skin_tone(image, intensity)
    result = enhance_wrinkles(result, landmarks, intensity)
    result = whiten_hair_approximation(result, landmarks, intensity)
    return result
