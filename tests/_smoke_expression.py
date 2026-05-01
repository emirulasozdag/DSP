"""Smoke test for the final stable expression engine (Delaunay 68-pt / TPS).

Run with:
    /opt/anaconda3/envs/dsp310/bin/python tests/_smoke_expression.py
"""
import os
import importlib.util

import cv2
import numpy as np

os.environ["DSP_PORT"] = "18080"

spec = importlib.util.spec_from_file_location(
    "app", os.path.join(os.path.dirname(__file__), "..", "app.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("Import OK")

# ── 1. Required symbols ───────────────────────────────────────────────────────
REQUIRED = [
    "_get_68_pts_for_warp",
    "_build_dst_68_direct",
    "_apply_expression_warp",
    "_tps_build",
    "_tps_remap",
    "_tps_eval",
]
missing = [f for f in REQUIRED if not hasattr(mod, f)]
assert not missing, f"Missing symbols: {missing}"
print(f"All {len(REQUIRED)} required symbols present")

# ── 2. Synthetic image ────────────────────────────────────────────────────────
img = np.full((256, 256, 3), 128, dtype=np.uint8)
img[100:140, 100:140] = [200, 160, 140]

# ── 3. _apply_expression_warp — return shape & dtype ─────────────────────────
ret = mod._apply_expression_warp(img, 0.5, 0.3, 0.2, 0.3, warp_mode="delaunay", save_debug=False)
assert len(ret) == 7, f"Expected 7-tuple, got {len(ret)}"
r_dl, debug_dl, n_lm, moved, max_d, tri_c, lm_src = ret
assert r_dl.shape == img.shape, f"delaunay shape mismatch: {r_dl.shape}"
assert r_dl.dtype == np.uint8, "delaunay result must be uint8"
assert isinstance(debug_dl, dict), "debug must be a dict"
assert lm_src in {"mediapipe468", "fallback68", "none"}, f"unexpected landmark source {lm_src}"
print(f"_apply_expression_warp delaunay: OK — n_lm={n_lm} moved={moved} tri={tri_c}")

# TPS mode accepted without crashing
r_tps, _dbg, *_ = mod._apply_expression_warp(
    img, 0.5, 0.0, 0.0, 0.0, warp_mode="tps", save_debug=False)
assert r_tps.shape == img.shape and r_tps.dtype == np.uint8
print("_apply_expression_warp tps: OK")

# ── 4. Zero sliders → identity ────────────────────────────────────────────────
r0, dbg0, lm0, mv0, md0, tc0, src0 = mod._apply_expression_warp(img, 0, 0, 0, 0)
assert np.array_equal(r0, img), "Zero sliders must return original unchanged"
assert lm0 == 0, f"usedLandmarkCount must be 0 for zero sliders, got {lm0}"
assert mv0 == 0, f"movedPoints must be 0 for zero sliders, got {mv0}"
assert md0 == 0.0, f"maxDisplacement must be 0 for zero sliders, got {md0}"
assert tc0 == 0, f"triangleCount must be 0 for zero sliders, got {tc0}"
assert src0 == "none", f"landmark source must be none for zero sliders, got {src0}"
print("Zero sliders identity: OK")

# ── 5. _build_dst_68_direct — per-slider isolation ───────────────────────────
# Synthetic 68-pt landmark array spread across a 256x256 canvas
np.random.seed(42)
src68 = np.column_stack([
    np.random.uniform(40, 216, 68),
    np.random.uniform(40, 216, 68),
]).astype(np.float32)

# Lip region: corners (48, 54) + adjacent points that move to prevent mesh tearing
LIP_REGION = {48, 49, 50, 54, 55, 56, 58, 59, 60, 61, 62, 64, 65, 67}
# Brow region: brow points 17-26 + upper eyelid follow-through (36-39, 42-45)
BROW_REGION = set(range(17, 27)) | {36, 37, 38, 39, 42, 43, 44, 45}
JAW_REGION  = set(range(0, 17))

# ── 5a. Smile only ────────────────────────────────────────────────────────────
dst_sm, cnt_sm, disp_sm, moved_sm = mod._build_dst_68_direct(
    src68, smile=0.5, brow=0.0, lip=0.0, slim=0.0)
assert dst_sm.shape == src68.shape
moved_set_sm = set(moved_sm)
# Primary corners must move
assert 48 in moved_set_sm and 54 in moved_set_sm, \
    f"Smile must move corners 48 and 54; got {moved_set_sm}"
# Only lip region may move — brow and jaw must be untouched
for i in BROW_REGION | JAW_REGION:
    assert np.allclose(dst_sm[i], src68[i], atol=1e-5), \
        f"Smile: point {i} (brow/jaw) must not move; delta={dst_sm[i] - src68[i]}"
# All movers must belong to the lip region
assert moved_set_sm <= LIP_REGION, \
    f"Smile: unexpected movers outside lip region: {moved_set_sm - LIP_REGION}"
print(f"_build_dst_68_direct smile: OK — moved {sorted(moved_set_sm)}")

# ── 5b. Brow only ────────────────────────────────────────────────────────────
dst_br, _, _, moved_br = mod._build_dst_68_direct(src68, 0, 0.5, 0, 0)
moved_set_br = set(moved_br)
# All brow points (17-26) must move
for i in range(17, 27):
    assert i in moved_set_br, f"Brow: point {i} should have moved"
# Only BROW_REGION (brow + eyelid) points may move
for i in range(68):
    if i not in BROW_REGION:
        assert np.allclose(dst_br[i], src68[i], atol=1e-5), \
            f"Brow: point {i} must not move; delta={dst_br[i] - src68[i]}"
print(f"_build_dst_68_direct brow: OK — moved {sorted(moved_set_br)}")

# ── 5c. Lip only ─────────────────────────────────────────────────────────────
dst_lp, _, _, moved_lp = mod._build_dst_68_direct(src68, 0, 0, 0.5, 0)
moved_set_lp = set(moved_lp)
# Primary corners must move
assert 48 in moved_set_lp and 54 in moved_set_lp, \
    f"Lip: must move corners 48 and 54; got {moved_set_lp}"
# Only lip region may move
assert moved_set_lp <= LIP_REGION, \
    f"Lip: unexpected movers outside lip region: {moved_set_lp - LIP_REGION}"
# Brow and jaw must be untouched
for i in BROW_REGION | JAW_REGION:
    assert np.allclose(dst_lp[i], src68[i], atol=1e-5), \
        f"Lip: point {i} (brow/jaw) must not move; delta={dst_lp[i] - src68[i]}"
print(f"_build_dst_68_direct lip: OK — moved {sorted(moved_set_lp)}")

# ── 5d. Slim only ────────────────────────────────────────────────────────────
dst_sl, _, _, moved_sl = mod._build_dst_68_direct(src68, 0, 0, 0, 0.5)
moved_set_sl = set(moved_sl)
# All jaw points (0-16) must move
for i in JAW_REGION:
    assert i in moved_set_sl, f"Slim: jawline point {i} should have moved"
# Non-jaw points must be unchanged
for i in range(68):
    if i not in JAW_REGION:
        assert np.allclose(dst_sl[i], src68[i], atol=1e-5), \
            f"Slim: point {i} must not move; delta={dst_sl[i] - src68[i]}"
print(f"_build_dst_68_direct slim: OK — moved {sorted(moved_set_sl)}")

# ── 6. TPS build / eval roundtrip ────────────────────────────────────────────
ctrl_src = np.array([[50, 50], [150, 50], [50, 150], [150, 150]], dtype=np.float32)
ctrl_dst = ctrl_src + 5
result_tps = mod._tps_build(ctrl_dst, ctrl_src)
assert result_tps is not None, "_tps_build returned None"
W, A, ctrl = result_tps
mapped = mod._tps_eval(W, A, ctrl, np.array([[100, 100]], dtype=np.float32))
assert mapped.shape == (1, 2), f"_tps_eval output shape {mapped.shape}"
print("_tps_build / _tps_eval roundtrip: OK")

print("\n=== ALL SMOKE CHECKS PASSED ===")
