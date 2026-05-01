import base64
import io
import importlib
import unittest

import cv2
import numpy as np


dsp_app = importlib.import_module("app")


def _synthetic_face(size=256, smile=0.0, brow=0.0, lip=0.0):
    image = np.full((size, size, 3), (66, 74, 90), dtype=np.uint8)

    cx = size // 2
    cy = int(size * 0.53)
    cv2.ellipse(image, (cx, cy), (int(size * 0.31), int(size * 0.39)), 0, 0, 360, (154, 186, 224), -1, lineType=cv2.LINE_AA)

    brow_shift = int(round(12 * brow))
    cv2.line(image, (int(size * 0.31), int(size * 0.35) - brow_shift), (int(size * 0.45), int(size * 0.33) - brow_shift), (30, 36, 46), 3, lineType=cv2.LINE_AA)
    cv2.line(image, (int(size * 0.55), int(size * 0.33) - brow_shift), (int(size * 0.69), int(size * 0.35) - brow_shift), (30, 36, 46), 3, lineType=cv2.LINE_AA)

    cv2.ellipse(image, (int(size * 0.39), int(size * 0.43)), (13, 8), 0, 0, 360, (24, 27, 33), -1, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (int(size * 0.61), int(size * 0.43)), (13, 8), 0, 0, 360, (24, 27, 33), -1, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (int(size * 0.50), int(size * 0.62)), (9, 17), 0, 0, 360, (108, 130, 160), 2, lineType=cv2.LINE_AA)

    mouth_y = int(round(size * 0.74 - (smile * 18)))
    mouth_w = int(round(31 + (lip * 8)))
    mouth_h = int(round(10 + (smile * 5) + (lip * 2)))
    cv2.ellipse(image, (cx, mouth_y), (mouth_w, mouth_h), 0, 0, 180, (58, 56, 150), 3, lineType=cv2.LINE_AA)

    return image


def _jpeg_bytes(image):
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("Could not encode test image.")
    return buf.tobytes()


def _decode_response_image(payload):
    arr = np.frombuffer(base64.b64decode(payload["image"]), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


class TransformReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_face_mesh = dsp_app.FACE_MESH
        dsp_app.FACE_MESH = None
        cls.client = dsp_app.app.test_client()
        cls.source = _synthetic_face(smile=0.05, brow=0.05, lip=0.0)
        cls.reference = _synthetic_face(smile=1.0, brow=0.65, lip=0.7)

    @classmethod
    def tearDownClass(cls):
        dsp_app.FACE_MESH = cls.original_face_mesh

    def _post_transform(
        self,
        mode="slider_only",
        include_reference=False,
        source_image=None,
        reference_image=None,
        **overrides,
    ):
        src = source_image if source_image is not None else self.source
        ref = reference_image if reference_image is not None else self.reference
        data = {
            "smile": "0",
            "brow": "0",
            "lip": "0",
            "slim": "0",
            "mode": mode,
            "referenceStrength": "0.9",
            "hybridBlend": "0.8",
            "image": (io.BytesIO(_jpeg_bytes(src)), "source.jpg"),
        }
        data.update({k: str(v) for k, v in overrides.items()})
        if include_reference:
            data["reference"] = (io.BytesIO(_jpeg_bytes(ref)), "reference.jpg")
        return self.client.post("/api/transform", data=data, content_type="multipart/form-data")

    def test_slider_only_returns_expected_metadata(self):
        res = self._post_transform(mode="slider_only")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["transferMode"], "slider_only")
        self.assertFalse(payload["referenceUsed"])
        self.assertIn("quality", payload)
        self.assertIn("warnings", payload)

    def test_reference_only_uses_reference_and_changes_output(self):
        base_payload = self._post_transform(mode="slider_only").get_json()
        ref_payload = self._post_transform(mode="reference_only", include_reference=True).get_json()

        self.assertEqual(ref_payload["transferMode"], "reference_only")
        self.assertTrue(ref_payload["referenceUsed"])
        self.assertGreaterEqual(ref_payload["quality"]["referenceConfidence"], 0.0)

        baseline = _decode_response_image(base_payload)
        referenced = _decode_response_image(ref_payload)
        mse = float(np.mean((baseline.astype(np.float32) - referenced.astype(np.float32)) ** 2))
        self.assertGreater(mse, 1.0)

    def test_hybrid_without_reference_falls_back_to_slider_only(self):
        payload = self._post_transform(mode="hybrid", include_reference=False).get_json()
        self.assertEqual(payload["transferMode"], "slider_only")
        self.assertFalse(payload["referenceUsed"])
        self.assertTrue(payload["warnings"])

    def test_reference_only_without_reference_returns_400(self):
        res = self._post_transform(mode="reference_only", include_reference=False)
        self.assertEqual(res.status_code, 400)
        payload = res.get_json()
        self.assertFalse(payload["ok"])
        self.assertIn("Reference image is required", payload["error"])

    def test_reference_only_low_resolution_still_changes_output(self):
        source_small = cv2.resize(self.source, (176, 120), interpolation=cv2.INTER_AREA)
        reference_small = cv2.resize(self.reference, (176, 120), interpolation=cv2.INTER_AREA)

        base_payload = self._post_transform(mode="slider_only", source_image=source_small).get_json()
        ref_payload = self._post_transform(
            mode="reference_only",
            include_reference=True,
            source_image=source_small,
            reference_image=reference_small,
            referenceStrength="1.0",
        ).get_json()

        self.assertEqual(ref_payload["transferMode"], "reference_only")
        self.assertTrue(ref_payload["referenceUsed"])

        baseline = _decode_response_image(base_payload)
        referenced = _decode_response_image(ref_payload)
        mse = float(np.mean((baseline.astype(np.float32) - referenced.astype(np.float32)) ** 2))
        self.assertGreater(mse, 0.9)

    def test_rejects_invalid_mode(self):
        res = self._post_transform(mode="unknown")
        self.assertEqual(res.status_code, 400)
        payload = res.get_json()
        self.assertFalse(payload["ok"])


if __name__ == "__main__":
    unittest.main()
