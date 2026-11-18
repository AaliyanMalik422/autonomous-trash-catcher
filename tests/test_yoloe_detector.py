"""Unit tests for YOLOE result conversion and stereo correspondence."""
from types import SimpleNamespace

import numpy as np

from vision.yoloe_detector import (
    DEFAULT_TRASH_CLASSES, ObjectDetection, extract_detections, object_family,
)
from vision.stereo_tracker import StereoCalibration, StereoTracker


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


def detection(label, cx, cy, area=500, confidence=0.8):
    return ObjectDetection(
        0, label, confidence,
        (int(cx - 10), int(cy - 10), int(cx + 10), int(cy + 10)),
        (cx, cy), area,
    )


def test_default_prompts_cover_project_targets():
    prompts = set(DEFAULT_TRASH_CLASSES)
    assert {"ball", "crumpled paper ball", "plastic bottle", "drink can",
            "disposable cup", "plastic wrapper"} <= prompts


def test_extract_detections_uses_segmentation_centroid():
    box = SimpleNamespace(
        xyxy=[np.asarray([10, 20, 50, 60], dtype=float)],
        cls=[Scalar(1)], conf=[Scalar(0.91)],
    )
    mask = np.zeros((1, 80, 100), dtype=np.float32)
    mask[0, 30:40, 20:30] = 1.0
    result = SimpleNamespace(boxes=[box], masks=SimpleNamespace(data=mask))
    found = extract_detections(result, {1: "crumpled paper ball"}, (80, 100, 3))
    assert len(found) == 1
    assert found[0].label == "crumpled paper ball"
    assert found[0].center == (24.5, 34.5)
    assert found[0].area == 100.0
    assert found[0].confidence == 0.91


def test_extract_detections_handles_empty_result():
    assert extract_detections(SimpleNamespace(boxes=None, masks=None), {}, (80, 100, 3)) == []


def test_object_families_normalize_prompt_synonyms():
    assert object_family("crumpled paper ball") == object_family("ball")
    assert object_family("plastic bottle") == object_family("clear bottle")
    assert object_family("plastic wrapper") == object_family("piece of rubbish")


def test_stereo_pair_selection_matches_same_object_and_epipolar_row():
    left = [detection("crumpled paper ball", 320, 220)]
    right = [detection("ball", 300, 221)]
    pair = StereoTracker._select_yolo_pair(left, right)
    assert pair == (left[0], right[0])


def test_stereo_pair_selection_rejects_wrong_class_and_bad_row():
    left = [detection("ball", 320, 220)]
    assert StereoTracker._select_yolo_pair(left, [detection("plastic bottle", 300, 220)]) is None
    assert StereoTracker._select_yolo_pair(left, [detection("ball", 300, 260)]) is None


def test_simulation_backend_does_not_load_yolo_weights():
    tracker = StereoTracker(StereoCalibration.ideal())
    assert tracker.detector_backend == "blob"
    assert tracker.yolo_detector is None
