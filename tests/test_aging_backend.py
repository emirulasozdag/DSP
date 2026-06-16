import base64
import io
import importlib.util
import os
import time
import unittest

import cv2
import numpy as np


spec = importlib.util.spec_from_file_location(
    "app",
    os.path.join(os.path.dirname(__file__), "..", "app.py"),
)
dsp_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dsp_app)


def _synthetic_face(size=256):
    rng = np.random.default_rng(1234)
    image = np.full((size, size, 3), (74, 82, 92), dtype=np.uint8)
    face_color = np.array([142, 178, 218], dtype=np.float32)
    cv2.ellipse(image, (size // 2, int(size * 0.52)), (int(size * 0.31), int(size * 0.39)), 0, 0, 360, face_color.tolist(), -1, lineType=cv2.LINE_AA)

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask, (size // 2, int(size * 0.52)), (int(size * 0.31), int(size * 0.39)), 0, 0, 360, 255, -1, lineType=cv2.LINE_AA)
    texture = rng.normal(0, 6, image.shape).astype(np.float32)
    image = np.where(mask[:, :, None] > 0, np.clip(image.astype(np.float32) + texture, 0, 255), image).astype(np.uint8)

    cv2.ellipse(image, (int(size * 0.39), int(size * 0.42)), (13, 8), 0, 0, 360, (24, 26, 32), -1, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (int(size * 0.61), int(size * 0.42)), (13, 8), 0, 0, 360, (24, 26, 32), -1, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (int(size * 0.50), int(size * 0.63)), (9, 17), 0, 0, 360, (105, 128, 160), 2, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (int(size * 0.50), int(size * 0.73)), (31, 11), 0, 0, 180, (56, 54, 142), 3, lineType=cv2.LINE_AA)
    cv2.line(image, (int(size * 0.31), int(size * 0.35)), (int(size * 0.45), int(size * 0.33)), (35, 39, 48), 3, lineType=cv2.LINE_AA)
    cv2.line(image, (int(size * 0.55), int(size * 0.33)), (int(size * 0.69), int(size * 0.35)), (35, 39, 48), 3, lineType=cv2.LINE_AA)
    return image


def _jpeg_bytes(image):
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    if not ok:
        raise RuntimeError("Could not encode test image.")
    return buf.tobytes()


def _decode_response_image(payload):
    arr = np.frombuffer(base64.b64decode(payload["image"]), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _edge_energy(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edges = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    return float(np.sum(edges * mask) / (np.sum(mask) + 1e-6))


class AgingBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_face_mesh = dsp_app.FACE_MESH
        dsp_app.FACE_MESH = None
        cls.client = dsp_app.app.test_client()
        cls.image = _synthetic_face()

    @classmethod
    def tearDownClass(cls):
        dsp_app.FACE_MESH = cls.original_face_mesh

    def _post_aging(self, **overrides):
        params = {
            "mode": "aging",
            "method": "freq",
            "years": "30",
            "strength": "0.85",
            "contrast": "0.55",
            "tone": "0.50",
            "noise": "0.20",
            "detail": "0.70",
            "deterministic": "true",
            "hpf": "gaussian_hpf",
            "lpf": "gaussian_lpf",
            "preserveEdges": "true",
        }
        params.update({key: str(value) for key, value in overrides.items()})
        params["image"] = (io.BytesIO(_jpeg_bytes(self.image)), "face.jpg")
        return self.client.post("/api/aging", data=params, content_type="multipart/form-data")

    def test_accepts_all_modes_methods_and_filter_options(self):
        for method in ("freq", "ai"):
            with self.subTest(method=method):
                res = self._post_aging(method=method)
                self.assertEqual(res.status_code, 200)
                payload = res.get_json()
                self.assertTrue(payload["ok"])
                self.assertEqual(payload["method"], method)
                self.assertIn(payload["modelStatus"], {"frequency-dsp", "fallback-demo", "onnx"})
                self.assertIn("frequency", payload["variants"])
                self.assertIn("ai", payload["variants"])

        for hpf in ("gaussian_hpf", "butterworth_hpf", "ideal_hpf"):
            with self.subTest(hpf=hpf):
                res = self._post_aging(mode="aging", hpf=hpf)
                self.assertEqual(res.status_code, 200)
                self.assertEqual(res.get_json()["hpf"], hpf)

        for lpf in ("gaussian_lpf", "butterworth_lpf", "ideal_lpf"):
            with self.subTest(lpf=lpf):
                res = self._post_aging(mode="deaging", lpf=lpf)
                self.assertEqual(res.status_code, 200)
                self.assertEqual(res.get_json()["lpf"], lpf)

    def test_rejects_invalid_aging_options(self):
        invalid_cases = [
            {"mode": "older"},
            {"method": "paint"},
            {"hpf": "sharpen"},
            {"lpf": "blur"},
        ]
        for overrides in invalid_cases:
            with self.subTest(overrides=overrides):
                res = self._post_aging(**overrides)
                self.assertEqual(res.status_code, 400)
                self.assertFalse(res.get_json()["ok"])

    def test_deterministic_requests_return_identical_image_payloads(self):
        first = self._post_aging(method="freq", noise="0.35").get_json()
        second = self._post_aging(method="freq", noise="0.35").get_json()
        self.assertEqual(first["image"], second["image"])

    def test_slider_changes_produce_measurably_different_images(self):
        subtle = _decode_response_image(self._post_aging(years="5", strength="0.15", noise="0.0").get_json())
        strong = _decode_response_image(self._post_aging(years="50", strength="1.0", noise="0.35", detail="1.0").get_json())
        mse = float(np.mean((subtle.astype(np.float32) - strong.astype(np.float32)) ** 2))
        self.assertGreater(mse, 8.0)

    def test_aging_increases_skin_high_frequency_energy(self):
        payload = self._post_aging(mode="aging", years="50", strength="1.0", detail="1.0", noise="0.0", hpf="ideal_hpf").get_json()
        self.assertIn("image", payload, "Response must contain base64 image")
        aged = _decode_response_image(payload)
        self.assertIsNotNone(aged, "Aging response image must decode successfully")
        masks = dsp_app._build_age_region_masks(self.image, preserve_edges=True)
        before = dsp_app._masked_high_frequency_energy(self.image, masks["skinMask"])
        after = dsp_app._masked_high_frequency_energy(aged, masks["skinMask"])
        self.assertGreater(
            after, before * 1.03,
            f"Aging should increase skin HF energy (DSP direction); "
            f"before={before:.4f} after={after:.4f} ratio={after/(before+1e-9):.3f}"
        )

    def test_deaging_smooths_skin_and_preserves_feature_edges(self):
        payload = self._post_aging(mode="deaging", years="50", strength="1.0", detail="0.0", noise="0.0", lpf="ideal_lpf").get_json()
        self.assertIn("image", payload, "Response must contain base64 image")
        deaged = _decode_response_image(payload)
        self.assertIsNotNone(deaged, "Deaging response image must decode successfully")
        self.assertEqual(deaged.shape, self.image.shape, "Deaged image shape must match input")

        masks = dsp_app._build_age_region_masks(self.image, preserve_edges=True)
        before_hf = dsp_app._masked_high_frequency_energy(self.image, masks["skinMask"])
        after_hf = dsp_app._masked_high_frequency_energy(deaged, masks["skinMask"])
        self.assertLess(
            after_hf, before_hf * 0.97,
            f"Deaging should reduce skin HF energy (DSP direction); "
            f"before={before_hf:.4f} after={after_hf:.4f} ratio={after_hf/(before_hf+1e-9):.3f}"
        )

        before_edges = _edge_energy(self.image, masks["protectedMask"])
        after_edges = _edge_energy(deaged, masks["protectedMask"])
        self.assertGreater(
            after_edges, before_edges * 0.40,
            f"Protected feature edges must not collapse; "
            f"before={before_edges:.4f} after={after_edges:.4f} ratio={after_edges/(before_edges+1e-9):.3f}"
        )

    def test_standard_request_finishes_under_five_seconds(self):
        started = time.perf_counter()
        res = self._post_aging()
        elapsed = time.perf_counter() - started
        payload = res.get_json()
        self.assertEqual(res.status_code, 200)
        self.assertLess(elapsed, 5.0)
        self.assertLess(payload["durationMs"], 5000.0)
        self.assertIn("energyBefore", payload)
        self.assertIn("energyAfter", payload)
        self.assertIn("apparentAgeBefore", payload)
        self.assertIn("apparentAgeAfter", payload)


class AgingBackendTestsWithMediaPipe(AgingBackendTests):
    """Re-runs the same tests with the real MediaPipe FaceMesh if available."""

    @classmethod
    def setUpClass(cls):
        cls.original_face_mesh = dsp_app.FACE_MESH
        if cls.original_face_mesh is None:
            raise unittest.SkipTest("MediaPipe FaceMesh not available — skipping real-landmark tests")
        # Leave FACE_MESH untouched so real landmarks are used
        cls.client = dsp_app.app.test_client()
        cls.image = _synthetic_face()

    @classmethod
    def tearDownClass(cls):
        dsp_app.FACE_MESH = cls.original_face_mesh


if __name__ == "__main__":
    unittest.main()
