"""Smoke test for the rewritten expression/face-warping engine."""
import numpy as np
import cv2
import sys
import os
import importlib
import importlib.util

os.environ["DSP_PORT"] = "18080"

spec = importlib.util.spec_from_file_location(
    "app", os.path.join(os.path.dirname(__file__), "..", "app.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("Import OK")

REQUIRED = [
    "_falloff_weights",
    "_build_dst_landmarks_v2",
    "_tps_build",
    "_tps_remap",
    "_tps_eval",
    "_build_expression_mask",
    "_save_debug_images",
    "_apply_expression_warp",
    "_extract_landmarks_for_warp",
    "_warp_triangle",
    "_triangle_area",
    "_MP468_GROUPS",
    "_MP68_GROUPS",
]
missing = [f for f in REQUIRED if not hasattr(mod, f)]
assert not missing, f"Missing: {missing}"
print(f"All {len(REQUIRED)} symbols present")

# --- Synthetic image (no real face) ---
img = np.full((256, 256, 3), 128, dtype=np.uint8)
img[100:140, 100:140] = [200, 160, 140]  # fake skin patch

# Delaunay mode
r_dl, _ = mod._apply_expression_warp(img, 0.5, 0.3, 0.2, 0.3,
                                      warp_mode="delaunay", save_debug=False)
assert r_dl.shape == img.shape and r_dl.dtype == np.uint8, "Delaunay shape/dtype"
print("Delaunay:", r_dl.shape, r_dl.dtype)

# TPS mode
r_tps, _ = mod._apply_expression_warp(img, 0.5, 0.3, 0.2, 0.3,
                                        warp_mode="tps", save_debug=False)
assert r_tps.shape == img.shape and r_tps.dtype == np.uint8, "TPS shape/dtype"
print("TPS:", r_tps.shape, r_tps.dtype)

# Slider at 0 → identical to input
r0, _ = mod._apply_expression_warp(img, 0, 0, 0, 0)
assert np.array_equal(r0, img), "Zero-slider must return original unchanged"
print("Zero-slider identity: OK")

# Falloff weights are monotone with distance
pts = np.array([[0, 0], [10, 0], [20, 0]], dtype=np.float32)
w = mod._falloff_weights(pts, [0], sigma=5.0)
assert w[0] > w[1] > w[2], f"Expected monotone falloff; got {w}"
print("Falloff monotone: OK")

# build_dst_landmarks_v2 returns same shape, dtype float32
src48 = np.random.rand(468, 2).astype(np.float32) * 200 + 28
dst48 = mod._build_dst_landmarks_v2(src48, mod._MP468_GROUPS, 0.5, 0.3, 0.2, 0.3, 256, 256)
assert dst48.shape == src48.shape and dst48.dtype == np.float32, "dst shape/dtype"
# All displaced points within image bounds
assert dst48[:, 0].min() >= 1 and dst48[:, 0].max() <= 254
assert dst48[:, 1].min() >= 1 and dst48[:, 1].max() <= 254
print("build_dst_landmarks_v2: OK")

# TPS build/eval roundtrip: 4 ctrl pts
ctrl_src = np.array([[50, 50], [150, 50], [50, 150], [150, 150]], dtype=np.float32)
ctrl_dst = ctrl_src + 5  # slight shift
result_tps = mod._tps_build(ctrl_dst, ctrl_src)
assert result_tps is not None, "TPS build failed"
W, A, ctrl = result_tps
q = np.array([[100, 100]], dtype=np.float32)
mapped = mod._tps_eval(W, A, ctrl, q)
assert mapped.shape == (1, 2), "TPS eval output shape"
print("TPS build/eval: OK")

# build_expression_mask returns float32 in [0,1]
for expr in ("smile", "brow", "lip", "slim"):
    m = mod._build_expression_mask(256, 256, src48, mod._MP468_GROUPS, expr, 180.0, 180.0)
    assert m.shape == (256, 256) and m.dtype == np.float32
    assert 0.0 <= m.min() and m.max() <= 1.0 + 1e-6
print("Expression masks: OK")

print("\n=== ALL SMOKE CHECKS PASSED ===")
