"""One-camera YOLO object-detection preview.

Run:
    python -m vision.trash_detection --source 2

This is a detector bring-up tool. It does not send commands to the robot and does
not claim to track objects or predict trajectories.
"""
from __future__ import annotations

import argparse
import time
from typing import Any

import cv2

from .cameras import open_capture, parse_source


def _scalar(value: Any) -> float:
    """Convert a scalar tensor / NumPy scalar / Python number to float."""
    return float(value.item() if hasattr(value, "item") else value)


def extract_detections(result: Any, names: Any) -> list[dict]:
    """Convert one Ultralytics Results object into plain, testable detections."""
    detections = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        coords = box.xyxy[0].tolist()
        x1, y1, x2, y2 = (int(round(float(v))) for v in coords)
        class_id = int(_scalar(box.cls[0]))
        confidence = _scalar(box.conf[0])
        if isinstance(names, dict):
            class_name = str(names.get(class_id, class_id))
        else:
            class_name = str(names[class_id])
        detections.append({
            "class_id": class_id,
            "class": class_name,
            "confidence": confidence,
            "bbox": (x1, y1, x2, y2),
            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
        })
    return detections


def resolve_class_ids(names: Any, requested: list[str] | None) -> list[int] | None:
    """Resolve optional class-name filters against the model's class labels."""
    if not requested:
        return None
    pairs = names.items() if isinstance(names, dict) else enumerate(names)
    by_name = {str(name).casefold(): int(class_id) for class_id, name in pairs}
    missing = [name for name in requested if name.casefold() not in by_name]
    if missing:
        available = ", ".join(sorted(str(v) for v in (names.values() if isinstance(names, dict) else names)))
        raise ValueError(
            f"Unknown class name(s): {', '.join(missing)}. "
            f"This model provides: {available}"
        )
    return [by_name[name.casefold()] for name in requested]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live single-camera YOLO detection (no robot commands)."
    )
    parser.add_argument("--source", default="2",
                        help="OpenCV camera index or video URL (default: 2, configured DroidCam)")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="Ultralytics model weights; yolo11n.pt downloads on first use")
    parser.add_argument("--confidence", type=float, default=0.35,
                        help="Minimum detection confidence (0-1; default: 0.35)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu",
                        help="Inference device, e.g. cpu or 0 (default: cpu)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Optional class-name allowlist; default shows all model classes")
    return parser


def run(args: argparse.Namespace) -> int:
    if not 0.0 < args.confidence <= 1.0:
        raise ValueError("--confidence must be greater than 0 and at most 1")
    if args.imgsz <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("--imgsz, --width and --height must be positive")

    source = parse_source(args.source)
    camera = open_capture(source, args.width, args.height, fps=30)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera source {args.source!r}. "
            "Check the DroidCam connection and try --source with the correct camera index or URL."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        camera.release()
        raise RuntimeError(
            "Ultralytics is not installed. Install detector dependencies with: "
            "python -m pip install -r requirements.txt -r requirements-vision.txt"
        ) from exc

    try:
        print(f"Loading YOLO weights: {args.model}")
        model = YOLO(args.model)  # Ultralytics downloads yolo11n.pt on first use if needed.
        class_ids = resolve_class_ids(model.names, args.classes)
        print(f"Camera connected: {args.source} | model: {args.model} | device: {args.device}")
        print("Live detection started. Press P to print detections, Q or Esc to quit.")

        fps_ema = 0.0
        last_frame_time = time.monotonic()
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Camera frame read failed; stopping preview.")
                break

            results = model.predict(
                source=frame,
                conf=args.confidence,
                imgsz=args.imgsz,
                device=args.device,
                classes=class_ids,
                verbose=False,
            )
            result = results[0]
            detections = extract_detections(result, model.names)
            annotated = result.plot()

            for detection in detections:
                cx, cy = detection["center"]
                cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(
                    annotated,
                    f"({cx}, {cy})",
                    (cx + 7, max(cy - 7, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            now = time.monotonic()
            dt = now - last_frame_time
            last_frame_time = now
            if dt > 0:
                instant_fps = 1.0 / dt
                fps_ema = instant_fps if fps_ema == 0 else 0.85 * fps_ema + 0.15 * instant_fps

            cv2.putText(
                annotated,
                f"YOLO | {len(detections)} detections | {fps_ema:.1f} FPS | Q: quit",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Autonomous Trash Catcher - YOLO detection", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("p"):
                print("\n--- CURRENT DETECTIONS ---")
                if not detections:
                    print("No objects detected.")
                for item in detections:
                    print(
                        f"{item['class']} ({item['confidence']:.2f}) "
                        f"center={item['center']} bbox={item['bbox']}"
                    )
            elif key in (ord("q"), 27):
                break

        return 0
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = build_parser().parse_args()
    try:
        raise SystemExit(run(args))
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
