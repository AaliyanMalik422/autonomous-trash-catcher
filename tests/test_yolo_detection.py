"""Unit tests for YOLO result conversion; do not require model weights or a camera."""
from types import SimpleNamespace

import pytest

from vision.trash_detection import extract_detections, resolve_class_ids


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class Box:
    def __init__(self, xyxy, class_id, confidence):
        self.xyxy = [SimpleNamespace(tolist=lambda: xyxy)]
        self.cls = [Scalar(class_id)]
        self.conf = [Scalar(confidence)]


def test_extract_detections_converts_boxes_and_centres():
    result = SimpleNamespace(
        boxes=[Box([10.2, 20.4, 50.8, 80.1], 32, 0.91)]
    )
    detections = extract_detections(result, {32: "sports ball"})
    assert detections == [{
        "class_id": 32,
        "class": "sports ball",
        "confidence": pytest.approx(0.91),
        "bbox": (10, 20, 51, 80),
        "center": (30, 50),
    }]


def test_extract_detections_handles_no_boxes():
    assert extract_detections(SimpleNamespace(boxes=None), {0: "person"}) == []


def test_resolve_class_ids_is_case_insensitive():
    assert resolve_class_ids({32: "sports ball", 39: "bottle"}, ["Sports Ball", "BOTTLE"]) == [32, 39]


def test_resolve_class_ids_reports_unsupported_classes():
    with pytest.raises(ValueError, match="paper ball"):
        resolve_class_ids({32: "sports ball"}, ["paper ball"])
