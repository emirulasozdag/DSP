"""Smoke test for the muscle-based flow-field expression engine (v3)."""
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
    # New muscle-engine symbols
    "_group_centroid",
    "_muscle_add_force",
    "_build_muscle_flow",
    "_lock_identity_regions",
    # Orchestrator
    "_apply_expression_warp",
    # TPS utilities (kept)
    "_tps_build",
    "_tps_remap",
    "_tps_eval",
    # Landmark infrastructure
    "_extract_landmarks_for_warp",
    "_MP468_GROUPS",
    "_MP68_GROUPS",
    # Legacy point helpers (backward compat)
    "_falloff_weights",
    "_build_dst_landmarks_v2",
]
missing = [f for f in REQUIRED if not hasattr(mod, f)]
assert not missing, f"Missing: {missing}"
print(f"All {len(REQUIRED)} symbols present")

# --- Synthetic image (no real face) ---
img = np.full((256, 256, 3), 128, dtype=np.uint8)
img[100:140, 100:140] = [200, 160, 140]  # fake skin patch

# Flow-field warp (new default)
r_fl, _ = mod._apply_expression_warp(img, 0.5, 0.3, 0.2, 0.3,
                                      warp_mode="flow", save_debug=False)
assert r_fl.shape == img.shape and r_fl.dtype == np.uint8, "flow shape/dtype"
print("flow:", r_fl.shape, r_fl.dtype)

# warp_mode arg is now ignored (always flow); other values are accepted silently
r_dl, _ = mod._apply_expression_warp(img, 0.5, 0.3, 0.2, 0.3,
                                      warp_mode="delaunay", save_debug=False)
assert r_dl.shape == img.shape and r_dl.dtype == np.uint8, "delaunay compat shape/dtype"
print("delaunay compat:", r_dl.shape, r_dl.dtype)

# Slider at 0 → identical to input
r0, _ = mod._apply_expression_warp(img, 0, 0, 0, 0)
assert np.array_equal(r0, img), "Zero-slider must return original unchanged"
print("Zero-slider identity: OK")

# _group_centroid returns a point within the src_pts bounds
src48 = np.random.rand(468, 2).astype(np.float32) * 200 + 28
ctr = mod._group_centroid(src48, mod._MP468_GROUPS, "mouth_left", "mouth_right")
assert ctr is not None and ctr.shape == (2,), "group_centroid shape"
assert float(src48[:, 0].min()) <= float(ctr[0]) <= float(src48[:, 0].max()), "centroid x in bounds"
print("group_centroid: OK")

# _build_muscle_flow returns correct shape and non-zero flow for non-zero slider
flow = mod._build_muscle_flow(256, 256, src48, mod._MP468_GROUPS, 0.5, 0.0, 0.0, 0.0)
assert flow.shape == (256, 256, 2), f"flow shape {flow.shape}"
assert flow.dtype == np.float32, "flow dtype"
assert np.any(flow != 0), "flow field should be non-zero for non-zero smile"
print("build_muscle_flow: OK")

# _lock_identity_regions suppresses flow near the nose/eye lock regions
flow_ones = np.ones((256, 256, 2), dtype=np.float32)
locked = mod._lock_identity_regions(flow_ones.copy(), src48, mod._MP468_GROUPS)
assert locked.shape == (256, 256, 2), "locked shape"
# Pixels near lock regions must have suppressed flow (min < 0.5)
assert float(locked.min()) < 0.5, "lock should significantly suppress flow near anchor regions"
print("lock_identity_regions: OK")

# Legacy: _falloff_weights still present and monotone
pts = np.array([[0, 0], [10, 0], [20, 0]], dtype=np.float32)
w = mod._falloff_weights(pts, [0], sigma=5.0)
assert w[0] > w[1] > w[2], f"Expected monotone falloff; got {w}"
print("Falloff monotone: OK")

# Legacy: build_dst_landmarks_v2 still callable and returns correct shape
dst48 = mod._build_dst_landmarks_v2(src48, mod._MP468_GROUPS, 0.5, 0.3, 0.2, 0.3, 256, 256)
assert dst48.shape == src48.shape and dst48.dtype == np.float32, "dst shape/dtype"
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

print("\n=== ALL SMOKE CHECKS PASSED ===")
