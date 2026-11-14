"""Pretrained open-vocabulary YOLOE detector used by the live vision pipeline.

Lazy-loaded so simulation and unit tests do not require PyTorch or model weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np


DEFAULT_TRASH_CLASSES = (
    "ball",
    "crumpled paper ball",
    "plastic bottle",
    "drink can",
    "disposable cup",
    "plastic wrapper",
    "plastic bag",
    "cardboard carton",
    "piece of rubbish",
)


@dataclass(frozen=True)
class ObjectDetection:
    class_id: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    area: float


def _as_float(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def extract_detections(result: Any, names: Any, image_shape: tuple[int, ...]) -> list[ObjectDetection]:
    """Convert an Ultralytics Results object into mask-centred detections."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    height, width = image_shape[:2]
    masks = getattr(result, "masks", None)
    mask_data = getattr(masks, "data", None) if masks is not None else None
    detections: list[ObjectDetection] = []

    for index, box in enumerate(boxes):
        coords = _to_numpy(box.xyxy[0]).reshape(-1)
        if coords.size < 4:
            continue
        x1, y1, x2, y2 = (int(round(float(v))) for v in coords[:4])
        x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            continue

        class_id = int(_as_float(box.cls[0]))
        confidence = _as_float(box.conf[0])
        label = str(names.get(class_id, class_id)) if isinstance(names, dict) else str(names[class_id])

        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        area = float((x2 - x1) * (y2 - y1))
        if mask_data is not None and index < len(mask_data):
            mask = _to_numpy(mask_data[index]).squeeze()
            if mask.ndim == 2:
                if mask.shape != (height, width):
                    mask = cv2.resize(mask.astype(np.float32), (width, height),
                                      interpolation=cv2.INTER_NEAREST)
                ys, xs = np.nonzero(mask > 0.5)
                if len(xs) >= 5:
                    center_x, center_y = float(xs.mean()), float(ys.mean())
                    area = float(len(xs))

        detections.append(ObjectDetection(
            class_id, label, confidence, (x1, y1, x2, y2),
            (center_x, center_y), area,
        ))
    return detections


def object_family(label: str) -> str:
    """Normalize prompt synonyms so stereo views can agree on the same object."""
    value = label.casefold()
    if "ball" in value or "wad" in value:
        return "ball"
    if "bottle" in value:
        return "bottle"
    if "can" in value:
        return "can"
    if "cup" in value:
        return "cup"
    if any(word in value for word in ("wrapper", "bag", "carton", "rubbish", "trash", "waste")):
        return "litter"
    return value


class YOLOETrashDetector:
    """YOLOE-26n open-vocabulary instance detector with project-specific prompts."""

    def __init__(
        self,
        model_path: str = "yoloe-26n-seg.pt",
        confidence: float = 0.18,
        image_size: int = 640,
        device: str = "cpu",
        classes: Iterable[str] = DEFAULT_TRASH_CLASSES,
    ):
        if not 0.0 < confidence <= 1.0:
            raise ValueError("confidence must be in (0, 1]")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        try:
            from ultralytics import YOLOE
        except ImportError as exc:
            raise RuntimeError(
                "YOLOE requires optional vision dependencies. Run: "
                "python -m pip install -r requirements-vision.txt"
            ) from exc

        self.classes = tuple(classes)
        if not self.classes:
            raise ValueError("At least one YOLOE class prompt is required")
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.device = device
        self.model = YOLOE(model_path)
        self.model.set_classes(list(self.classes))

    def detect(self, frame: np.ndarray) -> list[ObjectDetection]:
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        return extract_detections(results[0], self.model.names, frame.shape)
