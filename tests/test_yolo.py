import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import morning
import yolo_detect

NAMES = ["top-01", "bottom-01", "shoes-01"]
ITEMS = [{"id": n, "name": n, "category": n.split("-")[0], "color": "black"} for n in NAMES]


def raw_output(rows, num_classes=3, anchors=50):
    """Fake (1, 4 + classes, anchors) output. rows: (cx, cy, w, h, [class scores])."""
    out = np.zeros((1, 4 + num_classes, anchors), np.float32)
    for n, (cx, cy, w, h, cls_scores) in enumerate(rows):
        out[0, :4, n] = [cx, cy, w, h]
        out[0, 4:, n] = cls_scores
    return out


class FakeModel:
    def __init__(self, scores, trained=NAMES, names=NAMES):
        self.names = names
        self.trained = set(trained)
        self.scores = scores

    def run(self, img):
        classes = np.arange(len(self.scores))
        return np.zeros((len(classes), 4)), np.array(self.scores, float), classes


def cfg(**yolo):
    return {"yolo": {"model": "x.onnx", "present": 0.5, "unsure": 0.25, "api_fallback": True, **yolo},
            "detect": {"method": "yolo"}}


class TestDecode(unittest.TestCase):
    def test_letterbox_landscape_photo(self):
        img = np.full((1080, 1920, 3), 200, np.uint8)
        out, scale, (left, top) = yolo_detect.letterbox(img, 640)
        self.assertEqual(out.shape, (640, 640, 3))
        self.assertAlmostEqual(scale, 1 / 3)
        self.assertEqual((left, top), (0, 140))
        self.assertEqual(out[0, 0, 0], yolo_detect.PAD)    # padding above
        self.assertEqual(out[320, 320, 0], 200)            # photo in the middle

    def test_boxes_map_back_to_photo_pixels(self):
        out = raw_output([(320, 320, 100, 200, [0.9, 0.1, 0.0])])
        boxes, scores, classes = yolo_detect.decode(out, 3, 1 / 3, (0, 140))
        np.testing.assert_allclose(boxes[0], [810, 240, 1110, 840], atol=1e-3)
        self.assertEqual(classes[0], 0)
        self.assertAlmostEqual(scores[0], 0.9, places=5)

    def test_nms_free_output(self):
        out = np.zeros((1, 300, 6), np.float32)
        out[0, 0] = [10, 150, 110, 250, 0.8, 2]
        boxes, scores, classes = yolo_detect.decode(out, 3, 0.5, (0, 10))
        np.testing.assert_allclose(boxes[0], [20, 280, 220, 480])
        self.assertEqual(classes[0], 2)

    def test_unexpected_shape(self):
        with self.assertRaises(RuntimeError):
            yolo_detect.decode(np.zeros((1, 9, 50)), 3, 1, (0, 0))

    def test_item_scores_keeps_best_per_item(self):
        scores = np.array([0.3, 0.8, 0.6, 0.2])
        classes = np.array([0, 0, 1, 0])
        best = yolo_detect.item_scores(scores, classes, NAMES)
        self.assertEqual(best, {"top-01": 0.8, "bottom-01": 0.6, "shoes-01": 0.0})

    def test_names_from_onnx_metadata(self):
        session = mock.Mock()
        session.get_modelmeta.return_value.custom_metadata_map = {"names": "{0: 'top-01', 1: 'shoes-01'}"}
        meta = yolo_detect.load_meta(Path("/nonexistent/closet.onnx"), session)
        self.assertEqual(meta["names"], ["top-01", "shoes-01"])


class TestDetect(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.photo = Path(self.tmp.name) / "closet.jpg"
        cv2.imwrite(str(self.photo), np.zeros((90, 160, 3), np.uint8))

    def run_detect(self, model, conf, env=None):
        with mock.patch.object(yolo_detect, "Model", return_value=model), \
             mock.patch.dict(os.environ, env or {}, clear=False):
            if env is None:
                os.environ.pop("ANTHROPIC_API_KEY", None)
            return yolo_detect.detect(self.photo, ITEMS, conf)

    def test_thresholds(self):
        result = self.run_detect(FakeModel([0.9, 0.3, 0.1]), cfg())
        self.assertEqual(result["present"], ["top-01"])
        self.assertEqual(result["unsure"], ["bottom-01"])

    def test_untrained_items_assumed_there_without_api(self):
        model = FakeModel([0.9, 0.0, 0.0], trained=["top-01", "bottom-01"])
        result = self.run_detect(model, cfg())
        self.assertIn("shoes-01", result["unsure"])
        self.assertNotIn("bottom-01", result["present"] + result["unsure"])

    def test_untrained_items_go_to_api(self):
        model = FakeModel([0.9, 0.0, 0.0], trained=["top-01", "bottom-01"])
        api = {"present": ["shoes-01", "bottom-01"], "unsure": [], "unknown": ["grey hoodie"]}
        with mock.patch("detect.detect", return_value=api) as call:
            result = self.run_detect(model, cfg(), env={"ANTHROPIC_API_KEY": "k"})
        call.assert_called_once()
        # the API only decides about the item the model doesn't know
        self.assertEqual(result["present"], ["top-01", "shoes-01"])
        self.assertEqual(result["unknown"], ["grey hoodie"])

    def test_api_fallback_can_be_turned_off(self):
        model = FakeModel([0.9, 0.0, 0.0], trained=["top-01"])
        with mock.patch("detect.detect") as call:
            self.run_detect(model, cfg(api_fallback=False), env={"ANTHROPIC_API_KEY": "k"})
        call.assert_not_called()

    def test_morning_uses_yolo_when_configured(self):
        with mock.patch.object(yolo_detect, "detect", return_value={"present": [], "unsure": [], "unknown": []}) as y, \
             mock.patch.object(morning.detect, "detect") as api:
            morning.run_detection(self.photo, ITEMS, cfg())
        y.assert_called_once()
        api.assert_not_called()

    def test_missing_model_file(self):
        with self.assertRaises(RuntimeError):
            yolo_detect.Model(Path(self.tmp.name) / "nope.onnx")


if __name__ == "__main__":
    unittest.main()
