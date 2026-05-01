"""
Assertion-based smoke test for all FaceWarp DSP API endpoints.

Run with:
    /opt/anaconda3/envs/dsp310/bin/python _check_endpoints.py
"""
import base64
import importlib
import importlib.util
import io
import os
import sys

import cv2
import numpy as np

os.environ.setdefault("DSP_PORT", "19090")

spec = importlib.util.spec_from_file_location(
    "app", os.path.join(os.path.dirname(__file__), "app.py")
)
dsp_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dsp_app)

client = dsp_app.app.test_client()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_jpeg(size: int = 256) -> tuple[np.ndarray, bytes]:
    img = np.full((size, size, 3), (74, 82, 92), dtype=np.uint8)
    cv2.ellipse(img, (size // 2, int(size * 0.54)), (int(size * 0.28), int(size * 0.36)),
                0, 0, 360, (180, 160, 140), -1, cv2.LINE_AA)
    cv2.ellipse(img, (int(size * 0.38), int(size * 0.42)), (12, 8), 0, 0, 360, (20, 22, 28), -1)
    cv2.ellipse(img, (int(size * 0.62), int(size * 0.42)), (12, 8), 0, 0, 360, (20, 22, 28), -1)
    _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    return img, buf.tobytes()


def _decode_b64_image(payload: dict) -> np.ndarray:
    b64 = payload.get("image", "")
    assert b64, "Response missing 'image' field"
    arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert img is not None and img.ndim == 3, "base64 image failed to decode"
    return img


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def _post(path: str, jpeg: bytes, **fields) -> tuple[int, dict]:
    data = {"image": (io.BytesIO(jpeg), "face.jpg")}
    data.update({k: str(v) for k, v in fields.items()})
    r = client.post(path, data=data, content_type="multipart/form-data")
    return r.status_code, r.get_json()


def _ok(label: str):
    print(f"  OK  {label}")


# ── Build test image ──────────────────────────────────────────────────────────

img, jpeg = _make_jpeg(256)

# ── /api/health ───────────────────────────────────────────────────────────────

r = client.get("/api/health")
assert r.status_code == 200, f"/api/health status {r.status_code}"
j = r.get_json()
assert j["ok"] is True, f"/api/health ok={j.get('ok')}"
_ok("/api/health")

# ── /api/preprocess ───────────────────────────────────────────────────────────

status, j = _post("/api/preprocess", jpeg, size=128)
assert status == 200, f"/api/preprocess status {status}"
assert j["ok"] is True, f"/api/preprocess ok={j.get('ok')}"
assert j["width"] == 128, f"/api/preprocess width={j.get('width')}, expected 128"
_ok("/api/preprocess (size=128)")

# ── /api/landmarks ────────────────────────────────────────────────────────────

status, j = _post("/api/landmarks", jpeg)
assert status == 200, f"/api/landmarks status {status}"
assert j["ok"] is True, f"/api/landmarks ok={j.get('ok')}"
count = j.get("pointCount", len(j.get("landmarks", [])))
assert count >= 60, f"/api/landmarks pointCount={count}, expected >=60"
_ok(f"/api/landmarks (pointCount={count})")

# ── /api/transform — zero sliders ────────────────────────────────────────────

status, j = _post("/api/transform", jpeg, smile=0, brow=0, lip=0, slim=0)
assert status == 200, f"/api/transform (zeros) status {status}"
assert j["ok"] is True, f"/api/transform (zeros) ok={j.get('ok')}"
out_img = _decode_b64_image(j)
assert out_img.shape == img.shape, \
    f"/api/transform (zeros) shape {out_img.shape} != {img.shape}"
mse_val = _mse(img, out_img)
assert mse_val < 20.0, \
    f"/api/transform (zeros) MSE={mse_val:.2f} should be <20 (near-identity)"
_ok(f"/api/transform zero sliders (MSE={mse_val:.2f})")

# ── /api/transform — non-zero sliders ────────────────────────────────────────

status, j = _post("/api/transform", jpeg, smile=50, brow=30, lip=0, slim=0)
assert status == 200, f"/api/transform (smile50) status {status}"
assert j["ok"] is True, f"/api/transform (smile50) ok={j.get('ok')}"
assert j["movedPoints"] > 0, \
    f"/api/transform (smile50) movedPoints={j.get('movedPoints')}, expected >0"
assert j["maxDisplacement"] > 0, \
    f"/api/transform (smile50) maxDisplacement={j.get('maxDisplacement')}, expected >0"
transformed_img = _decode_b64_image(j)
assert transformed_img.shape == img.shape, \
    f"/api/transform (smile50) output shape {transformed_img.shape} != {img.shape}"
_ok(f"/api/transform smile=50 (moved={j['movedPoints']} maxD={j['maxDisplacement']:.1f})")

# ── /api/transform — TPS mode ─────────────────────────────────────────────────

status, j = _post("/api/transform", jpeg, smile=50, mode="tps")
assert status == 200, f"/api/transform (tps) status {status}"
assert j["ok"] is True, f"/api/transform (tps) ok={j.get('ok')}"
assert j["method"] in {"delaunay", "tps"}, \
    f"/api/transform (tps) method={j.get('method')}"
_decode_b64_image(j)
_ok(f"/api/transform TPS (method={j['method']})")

# ── /api/analysis — missing transformed → 400 ────────────────────────────────

r = client.post("/api/analysis",
                data={"image": (io.BytesIO(jpeg), "orig.jpg")},
                content_type="multipart/form-data")
assert r.status_code in (400, 422), \
    f"/api/analysis (no transformed) status {r.status_code}, expected 400"
assert r.get_json()["ok"] is False, "/api/analysis (no transformed) should return ok=False"
_ok("/api/analysis missing transformed → 400")

# ── /api/analysis — valid call ────────────────────────────────────────────────

mod_img = np.clip(img.astype(np.float32) * 0.92 + 8, 0, 255).astype(np.uint8)
_, mod_jpeg = cv2.imencode(".jpg", mod_img, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
r = client.post("/api/analysis", data={
    "image":       (io.BytesIO(jpeg),              "original.jpg"),
    "transformed": (io.BytesIO(mod_jpeg.tobytes()), "transformed.jpg"),
}, content_type="multipart/form-data")
assert r.status_code == 200, f"/api/analysis status {r.status_code}"
j = r.get_json()
assert j["ok"] is True, f"/api/analysis ok={j.get('ok')}"
assert "mse"  in j, "/api/analysis missing 'mse'"
assert "ssim" in j, "/api/analysis missing 'ssim'"
assert isinstance(j["mse"],  (int, float)), "/api/analysis mse not numeric"
assert isinstance(j["ssim"], (int, float)), "/api/analysis ssim not numeric"
_ok(f"/api/analysis (mse={j['mse']:.2f} ssim={j['ssim']:.3f})")

# ── /api/aging — valid ────────────────────────────────────────────────────────

status, j = _post("/api/aging", jpeg, mode="aging", method="freq",
                  years=20, strength=0.7, deterministic="true")
assert status == 200, f"/api/aging status {status}"
assert j["ok"] is True, f"/api/aging ok={j.get('ok')}"
aged_img = _decode_b64_image(j)
assert aged_img.shape == img.shape, \
    f"/api/aging output shape {aged_img.shape} != {img.shape}"
_ok("/api/aging valid")

# ── /api/aging — valid deaging ───────────────────────────────────────────────

status, j = _post("/api/aging", jpeg, mode="deaging", method="freq",
                  years=20, strength=0.7, deterministic="true")
assert status == 200, f"/api/deaging status {status}"
assert j["ok"] is True, f"/api/deaging ok={j.get('ok')}"
deaged_img = _decode_b64_image(j)
assert deaged_img.shape == img.shape, \
    f"/api/deaging output shape {deaged_img.shape} != {img.shape}"
_ok("/api/aging deaging valid")

# ── /api/aging — bad mode → 400 ──────────────────────────────────────────────

status, j = _post("/api/aging", jpeg, mode="older")
assert status == 400, f"/api/aging bad mode status {status}, expected 400"
assert j["ok"] is False, "/api/aging bad mode should return ok=False"
_ok("/api/aging bad mode → 400")

# ── /api/aging — bad hpf → 400 ───────────────────────────────────────────────

status, j = _post("/api/aging", jpeg, hpf="sharpen")
assert status == 400, f"/api/aging bad hpf status {status}, expected 400"
assert j["ok"] is False, "/api/aging bad hpf should return ok=False"
_ok("/api/aging bad hpf → 400")

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== ALL ENDPOINT CHECKS PASSED ===")
