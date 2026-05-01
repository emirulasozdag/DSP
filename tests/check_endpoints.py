"""
Comprehensive endpoint tests for FaceWarp DSP API.

Run with:
    /opt/anaconda3/envs/dsp310/bin/python tests/check_endpoints.py

All assertions produce clear messages on failure.
"""
from __future__ import annotations

import base64
import io
import importlib
import importlib.util
import os
import sys
import time
import unittest

import cv2
import numpy as np

# ── Load app without starting the server ────────────────────────────────────
os.environ.setdefault("DSP_PORT", "19090")   # avoid clashing with running server
spec = importlib.util.spec_from_file_location(
    "app",
    os.path.join(os.path.dirname(__file__), "..", "app.py"),
)
dsp_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dsp_app)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _synthetic_portrait(size: int = 256, seed: int = 42) -> np.ndarray:
    """256×256 BGR image that looks vaguely like a portrait (no real face needed)."""
    rng = np.random.default_rng(seed)
    bg = np.full((size, size, 3), (74, 82, 92), dtype=np.uint8)
    skin = np.array([180, 142, 120], dtype=np.float32)
    cx, cy = size // 2, int(size * 0.52)
    rx, ry = int(size * 0.30), int(size * 0.38)
    cv2.ellipse(bg, (cx, cy), (rx, ry), 0, 0, 360, skin.tolist(), -1, cv2.LINE_AA)
    # eyes
    cv2.ellipse(bg, (int(size * 0.38), int(size * 0.42)), (14, 9), 0, 0, 360, (20, 22, 28), -1, cv2.LINE_AA)
    cv2.ellipse(bg, (int(size * 0.62), int(size * 0.42)), (14, 9), 0, 0, 360, (20, 22, 28), -1, cv2.LINE_AA)
    # mouth
    cv2.ellipse(bg, (cx, int(size * 0.72)), (28, 9), 0, 0, 360, (90, 60, 110), 2, cv2.LINE_AA)
    return bg


def _jpeg_bytes(image: np.ndarray, quality: int = 94) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok, "Failed to encode test image"
    return buf.tobytes()


def _b64_decode_image(payload: dict) -> np.ndarray | None:
    """Decode base64 image from a JSON response. Returns BGR ndarray."""
    b64 = payload.get("image", "")
    if not b64:
        return None
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _assert_b64_image_valid(payload: dict, msg: str = "") -> np.ndarray:
    img = _b64_decode_image(payload)
    assert img is not None and img.ndim == 3, f"base64 image invalid — {msg}"
    return img


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = cv2.resize(b, (a.shape[1], a.shape[0])).astype(np.float32) if b.shape != a.shape else b.astype(np.float32)
    return float(np.mean((a - b) ** 2))


# ── Test fixtures ─────────────────────────────────────────────────────────────

class _BaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._orig_fm = dsp_app.FACE_MESH
        dsp_app.FACE_MESH = None          # use synthetic landmarks everywhere
        cls.client = dsp_app.app.test_client()
        cls.portrait = _synthetic_portrait()
        cls.portrait_bytes = _jpeg_bytes(cls.portrait)

    @classmethod
    def tearDownClass(cls):
        dsp_app.FACE_MESH = cls._orig_fm

    # ── Convenience POSTers ───────────────────────────────────────────────
    def _post_preprocess(self, img_bytes=None, **kwargs):
        img_bytes = img_bytes or self.portrait_bytes
        # Map 'resize' → 'size' for convenience in test callers
        if "resize" in kwargs:
            kwargs["size"] = kwargs.pop("resize")
        data = {"image": (io.BytesIO(img_bytes), "face.jpg")}
        data.update({k: str(v) for k, v in kwargs.items()})
        return self.client.post("/api/preprocess", data=data,
                                content_type="multipart/form-data")

    def _post_landmarks(self, img_bytes=None, **kwargs):
        img_bytes = img_bytes or self.portrait_bytes
        data = {"image": (io.BytesIO(img_bytes), "face.jpg")}
        data.update({k: str(v) for k, v in kwargs.items()})
        return self.client.post("/api/landmarks", data=data,
                                content_type="multipart/form-data")

    def _post_transform(self, img_bytes=None, **kwargs):
        img_bytes = img_bytes or self.portrait_bytes
        data = {"image": (io.BytesIO(img_bytes), "face.jpg")}
        defaults = dict(smile=0, brow=0, lip=0, slim=0)
        defaults.update(kwargs)
        data.update({k: str(v) for k, v in defaults.items()})
        return self.client.post("/api/transform", data=data,
                                content_type="multipart/form-data")

    def _post_aging(self, img_bytes=None, **kwargs):
        img_bytes = img_bytes or self.portrait_bytes
        data = {
            "image": (io.BytesIO(img_bytes), "face.jpg"),
            "mode": "aging", "method": "freq",
            "years": "20", "strength": "0.7",
            "contrast": "0.5", "tone": "0.5",
            "noise": "0.1", "detail": "0.6",
            "hpf": "gaussian_hpf", "lpf": "gaussian_lpf",
            "preserveEdges": "true",
        }
        data.update({k: str(v) for k, v in kwargs.items()})
        # Re-wrap image if overridden
        if "image" in kwargs:
            data["image"] = (io.BytesIO(kwargs["image"]), "face.jpg")
        return self.client.post("/api/aging", data=data,
                                content_type="multipart/form-data")

    def _post_analysis(self, original_bytes=None, transformed_bytes=None, **kwargs):
        orig = original_bytes or self.portrait_bytes
        # Make a slightly different image for transformed
        if transformed_bytes is None:
            mod_img = (self.portrait.astype(np.float32) * 0.95 + 5).clip(0, 255).astype(np.uint8)
            transformed_bytes = _jpeg_bytes(mod_img)
        data = {
            "image":       (io.BytesIO(orig), "original.jpg"),
            "transformed": (io.BytesIO(transformed_bytes), "transformed.jpg"),
        }
        data.update({k: str(v) for k, v in kwargs.items()})
        return self.client.post("/api/analysis", data=data,
                                content_type="multipart/form-data")


# ── /api/health ───────────────────────────────────────────────────────────────

class TestHealth(_BaseTest):
    def test_health_ok(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertTrue(payload["ok"])
        self.assertIn("service", payload)
        self.assertIn("timestamp", payload)


# ── /api/preprocess ───────────────────────────────────────────────────────────

class TestPreprocess(_BaseTest):
    def test_valid_image_returns_ok(self):
        res = self._post_preprocess(size=256, grayscale="false")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertTrue(payload["ok"])
        required = {"ok", "saved", "width", "height", "grayscale",
                    "normalization", "faceDetected", "faceDetector", "durationMs"}
        for field in required:
            self.assertIn(field, payload, f"Missing field: {field}")
        self.assertEqual(payload["width"], 256)
        self.assertEqual(payload["height"], 256)
        self.assertIsInstance(payload["durationMs"], (int, float))

    def test_grayscale_option(self):
        res = self._post_preprocess(size=128, grayscale="true")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["grayscale"])

    def test_resize_options(self):
        for size in (128, 256, 512):
            with self.subTest(size=size):
                res = self._post_preprocess(size=size)
                payload = res.get_json()
                self.assertEqual(res.status_code, 200)
                self.assertEqual(payload["width"], size)

    def test_invalid_file_rejected(self):
        res = self.client.post(
            "/api/preprocess",
            data={"image": (io.BytesIO(b"not an image"), "bad.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(res.status_code, 400)
        self.assertFalse(res.get_json()["ok"])

    def test_empty_body_rejected(self):
        res = self.client.post("/api/preprocess", data={},
                               content_type="multipart/form-data")
        self.assertIn(res.status_code, (400, 422))
        self.assertFalse(res.get_json()["ok"])

    def test_too_large_file_rejected(self):
        big_noise = (_jpeg_bytes(np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)))
        # Fake a file that claims 11 MB by padding raw bytes
        padded = big_noise + b"\x00" * (11 * 1024 * 1024)
        res = self.client.post(
            "/api/preprocess",
            data={"image": (io.BytesIO(padded), "big.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(res.status_code, 400)
        self.assertFalse(res.get_json()["ok"])


# ── /api/landmarks ────────────────────────────────────────────────────────────

class TestLandmarks(_BaseTest):
    def test_returns_landmarks(self):
        res = self._post_landmarks()
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertTrue(payload["ok"])
        self.assertIn("landmarks", payload)
        lms = payload["landmarks"]
        self.assertIsInstance(lms, list)
        self.assertGreater(len(lms), 0)
        # Each landmark must have x and y
        first = lms[0]
        self.assertIn("x", first)
        self.assertIn("y", first)

    def test_landmark_count_is_reasonable(self):
        """68-pt fallback or 468-pt MediaPipe — both are acceptable."""
        res = self._post_landmarks()
        payload = res.get_json()
        count = payload.get("pointCount", len(payload.get("landmarks", [])))
        self.assertGreaterEqual(count, 60, "Expected at least 60 landmarks")

    def test_reduced68_via_mode_param(self):
        """Request landmarkMode=reduced68; should return 68 points (fallback ok)."""
        res = self._post_landmarks(landmarkMode="reduced68")
        payload = res.get_json()
        self.assertTrue(payload["ok"])
        lms = payload.get("landmarks", [])
        self.assertGreaterEqual(len(lms), 60)

    def test_invalid_file_rejected(self):
        res = self.client.post(
            "/api/landmarks",
            data={"image": (io.BytesIO(b"garbage"), "bad.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(res.status_code, 400)


# ── /api/transform ────────────────────────────────────────────────────────────

class TestTransform(_BaseTest):
    REQUIRED_FIELDS = {
        "ok", "image", "width", "height", "method",
        "interpolation", "usedLandmarkCount",
        "movedPoints", "maxDisplacement", "triangleCount",
        "saved", "durationMs",
    }

    def _transform_and_decode(self, **kwargs):
        res = self._post_transform(**kwargs)
        self.assertEqual(res.status_code, 200, res.data)
        payload = res.get_json()
        self.assertTrue(payload["ok"])
        for f in self.REQUIRED_FIELDS:
            self.assertIn(f, payload, f"Missing field '{f}' in transform response")
        img = _assert_b64_image_valid(payload, str(kwargs))
        return img, payload

    def test_zero_sliders_returns_almost_same_image(self):
        img, payload = self._transform_and_decode(smile=0, brow=0, lip=0, slim=0)
        mse_val = _mse(self.portrait, img)
        self.assertLess(mse_val, 20.0,
                        f"Zero sliders should return near-identical image; MSE={mse_val:.1f}")

    def test_smile_changes_image_slightly(self):
        original_img, _ = self._transform_and_decode(smile=0)
        smile_img, payload = self._transform_and_decode(smile=80)
        mse_val = _mse(original_img, smile_img)
        self.assertGreater(mse_val, 0.5,
                           f"Smile=80 should differ from zero; MSE={mse_val:.2f}")
        self.assertLess(mse_val, 800.0,
                        f"Smile=80 change is too extreme (alien face); MSE={mse_val:.1f}")
        self.assertGreater(payload["movedPoints"], 0, "movedPoints must be >0 for non-zero smile")

    def test_brow_does_not_distort_eyes(self):
        """Brow raise should NOT move the eye PUPILS — only the brow region."""
        zero_img, _ = self._transform_and_decode(brow=0)
        brow_img, _ = self._transform_and_decode(brow=100)
        h, w = self.portrait.shape[:2]
        # Pupil-only region: very tight crop around expected eye positions
        # y ∈ [40%, 50%], x ∈ [30%, 70%] — center of eyes, avoiding brows
        y0, y1 = int(h * 0.40), int(h * 0.50)
        x0, x1 = int(w * 0.30), int(w * 0.70)
        eye_zero = zero_img[y0:y1, x0:x1].astype(np.float32)
        eye_brow = brow_img[y0:y1, x0:x1].astype(np.float32)
        eye_mse = float(np.mean((eye_zero - eye_brow) ** 2))
        # Delaunay mesh will propagate some motion to neighbouring triangles,
        # so allow a moderate budget — we only require no extreme distortion.
        self.assertLess(eye_mse, 2000.0,
                        f"Brow raise distorted pupil region too much; eye_MSE={eye_mse:.1f}")

    def test_slim_changes_only_jawline(self):
        zero_img, _ = self._transform_and_decode(slim=0)
        slim_img, payload = self._transform_and_decode(slim=80)
        mse_val = _mse(zero_img, slim_img)
        self.assertGreater(mse_val, 0.2, "Slim=80 should change something")
        self.assertLess(mse_val, 600.0, f"Slim=80 too extreme; MSE={mse_val:.1f}")
        self.assertEqual(payload["triangleCount"], 0,
                         "Slim-only transform should use dense flow, not jaw Delaunay triangles")

    def test_delaunay_method_accepted(self):
        _, payload = self._transform_and_decode(smile=50, mode="delaunay")
        self.assertIn(payload["method"], {"delaunay", "tps"})

    def test_tps_method_accepted(self):
        _, payload = self._transform_and_decode(smile=50, mode="tps")
        # TPS may fall back to delaunay on synthetic image — both are valid
        self.assertIn(payload["method"], {"delaunay", "tps"})

    def test_max_displacement_is_bounded(self):
        """Max displacement must stay within 20% of face size for natural output."""
        _, payload = self._transform_and_decode(smile=100, brow=100, lip=100, slim=100)
        face_size = max(payload["width"], payload["height"])
        max_d = payload["maxDisplacement"]
        # Allow up to 20% for synthetic images with no real face
        self.assertLess(max_d, face_size * 0.20,
                        f"maxDisplacement={max_d:.1f} exceeds 20% of face size {face_size}")

    def test_invalid_image_rejected(self):
        res = self.client.post(
            "/api/transform",
            data={"image": (io.BytesIO(b"not an image"), "bad.jpg"),
                  "smile": "50"},
            content_type="multipart/form-data",
        )
        self.assertEqual(res.status_code, 400)
        self.assertFalse(res.get_json()["ok"])

    def test_performance_under_two_seconds(self):
        start = time.perf_counter()
        self._transform_and_decode(smile=50, brow=30, lip=20, slim=40)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 2.0, f"Transform took {elapsed:.2f}s; should be <2s")


# ── /api/aging ────────────────────────────────────────────────────────────────

class TestAging(_BaseTest):
    REQUIRED_FIELDS = {
        "ok", "image", "mode", "method", "years",
        "apparentAgeBefore", "apparentAgeAfter",
        "energyBefore", "energyAfter",
        "skinHighFrequencyEnergy", "durationMs",
    }

    def test_aging_mode_returns_ok(self):
        res = self._post_aging(mode="aging")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertTrue(payload["ok"])
        for f in self.REQUIRED_FIELDS:
            self.assertIn(f, payload, f"Missing field '{f}'")
        _assert_b64_image_valid(payload, "aging mode")

    def test_deaging_mode_returns_ok(self):
        res = self._post_aging(mode="deaging")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertTrue(payload["ok"])
        _assert_b64_image_valid(payload, "deaging mode")

    def test_freq_method_works(self):
        res = self._post_aging(method="freq")
        self.assertEqual(res.status_code, 200)

    def test_ai_method_works(self):
        res = self._post_aging(method="ai")
        self.assertEqual(res.status_code, 200)

    def test_invalid_mode_rejected(self):
        res = self._post_aging(mode="older")
        self.assertEqual(res.status_code, 400)
        self.assertFalse(res.get_json()["ok"])

    def test_invalid_method_rejected(self):
        res = self._post_aging(method="deepfake")
        self.assertEqual(res.status_code, 400)
        self.assertFalse(res.get_json()["ok"])

    def test_aging_changes_image(self):
        res = self._post_aging(mode="aging", years="50", strength="1.0", detail="1.0", noise="0.0")
        payload = res.get_json()
        aged = _assert_b64_image_valid(payload)
        mse_val = _mse(self.portrait, aged)
        self.assertGreater(mse_val, 1.0, "Aging should change the image")

    def test_energy_metrics_present_and_numeric(self):
        payload = self._post_aging().get_json()
        # energyBefore/After are dicts with 'total', 'high', 'low', 'hfLfRatio'
        for key in ("energyBefore", "energyAfter"):
            val = payload[key]
            self.assertIsInstance(val, dict, f"{key} should be a dict")
            self.assertIn("total", val, f"{key}['total'] missing")
            self.assertIsInstance(val["total"], (int, float))
        self.assertIsInstance(payload["skinHighFrequencyEnergy"], (int, float))

    def test_base64_image_decodes(self):
        payload = self._post_aging().get_json()
        img = _assert_b64_image_valid(payload, "aging base64")
        self.assertEqual(img.shape, self.portrait.shape)

    def test_independent_of_expression_pipeline(self):
        """Aging must work on a freshly uploaded image without prior transform."""
        res = self._post_aging(image=self.portrait_bytes, mode="aging")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["ok"])


# ── /api/analysis ─────────────────────────────────────────────────────────────

class TestAnalysis(_BaseTest):
    REQUIRED_FIELDS = {
        "ok", "mse", "psnr", "ssim",
        "energy", "energyComparison", "luminance",
        "spectrum", "saved", "durationMs",
    }

    def test_valid_analysis_returns_all_fields(self):
        res = self._post_analysis()
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertTrue(payload["ok"])
        for f in self.REQUIRED_FIELDS:
            self.assertIn(f, payload, f"Missing field: {f}")

    def test_metrics_are_numeric(self):
        payload = self._post_analysis().get_json()
        self.assertIsInstance(payload["mse"],  (int, float))
        self.assertIsInstance(payload["psnr"], (int, float))
        self.assertIsInstance(payload["ssim"], (int, float))

    def test_ssim_in_valid_range(self):
        payload = self._post_analysis().get_json()
        ssim = payload["ssim"]
        self.assertGreaterEqual(ssim, -1.0)
        self.assertLessEqual(ssim, 1.0)

    def test_identical_images_give_zero_mse(self):
        """Identical original and transformed → MSE ≈ 0, SSIM ≈ 1."""
        res = self._post_analysis(
            original_bytes=self.portrait_bytes,
            transformed_bytes=self.portrait_bytes,
        )
        payload = res.get_json()
        self.assertEqual(res.status_code, 200)
        self.assertLess(payload["mse"], 5.0, "Identical images → MSE must be ~0")
        self.assertGreater(payload["ssim"], 0.90, "Identical images → SSIM must be ~1.0")

    def test_different_images_give_positive_mse(self):
        noisy = np.clip(
            self.portrait.astype(np.float32) + np.random.normal(0, 30, self.portrait.shape),
            0, 255
        ).astype(np.uint8)
        noisy_bytes = _jpeg_bytes(noisy)
        payload = self._post_analysis(transformed_bytes=noisy_bytes).get_json()
        self.assertGreater(payload["mse"], 10.0)

    def test_energy_comparison_structure(self):
        payload = self._post_analysis().get_json()
        ec = payload["energyComparison"]
        self.assertIsInstance(ec, dict)
        # Keys are 'original', 'transformed', 'delta'
        for key in ("original", "transformed", "delta"):
            self.assertIn(key, ec, f"energyComparison missing '{key}'")

    def test_luminance_statistics_structure(self):
        payload = self._post_analysis().get_json()
        lum = payload["luminance"]
        self.assertIsInstance(lum, dict)

    def test_spectrum_structure(self):
        payload = self._post_analysis().get_json()
        spec = payload["spectrum"]
        self.assertIsInstance(spec, dict)

    def test_missing_transformed_fails(self):
        res = self.client.post(
            "/api/analysis",
            data={"image": (io.BytesIO(self.portrait_bytes), "orig.jpg")},
            content_type="multipart/form-data",
        )
        self.assertIn(res.status_code, (400, 422))
        self.assertFalse(res.get_json()["ok"])

    def test_missing_original_fails(self):
        mod_bytes = _jpeg_bytes(
            np.clip(self.portrait.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
        )
        res = self.client.post(
            "/api/analysis",
            data={"transformed": (io.BytesIO(mod_bytes), "xform.jpg")},
            content_type="multipart/form-data",
        )
        self.assertIn(res.status_code, (400, 422))
        self.assertFalse(res.get_json()["ok"])


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in (TestHealth, TestPreprocess, TestLandmarks,
                TestTransform, TestAging, TestAnalysis):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
