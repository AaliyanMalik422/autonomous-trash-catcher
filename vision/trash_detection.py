"""Live one-camera YOLOE detection preview.

Run: python -m vision.trash_detection --source 2
This perception-only mode never connects to or commands the robot.
"""
from __future__ import annotations

import argparse
import time

import cv2

from .cameras import open_capture, parse_source
from .yoloe_detector import DEFAULT_TRASH_CLASSES, YOLOETrashDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live YOLOE trash-object detection (no robot commands)."
    )
    parser.add_argument("--source", default="2",
                        help="OpenCV camera index or stream URL (default: DroidCam index 2)")
    parser.add_argument("--model", default="yoloe-26n-seg.pt")
    parser.add_argument("--confidence", type=float, default=0.18)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_TRASH_CLASSES),
                        help="Text prompts used by the open-vocabulary detector")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0 < args.confidence <= 1:
        raise SystemExit("ERROR: --confidence must be in (0, 1]")
    camera = open_capture(parse_source(args.source), args.width, args.height, fps=30)
    if not camera.isOpened():
        camera.release()
        raise SystemExit(
            f"ERROR: cannot open camera {args.source!r}; connect DroidCam and verify its OpenCV index"
        )

    try:
        print(f"Loading {args.model} with prompts: {', '.join(args.classes)}")
        detector = YOLOETrashDetector(
            args.model, args.confidence, args.imgsz, args.device, args.classes
        )
        print("Detection running. P = print detections, Q/Esc = quit.")
        fps = 0.0
        previous = time.monotonic()
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError("Camera stopped delivering frames")

            detections = detector.detect(frame)
            view = frame.copy()
            for item in detections:
                x1, y1, x2, y2 = item.bbox
                cx, cy = (int(round(v)) for v in item.center)
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.circle(view, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(
                    view, f"{item.label} {item.confidence:.2f}",
                    (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0, 220, 0), 1, cv2.LINE_AA,
                )
                cv2.putText(
                    view, f"({cx}, {cy})", (cx + 7, cy - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA,
                )

            now = time.monotonic()
            instant = 1.0 / max(now - previous, 1e-6)
            previous = now
            fps = instant if fps == 0 else 0.85 * fps + 0.15 * instant
            cv2.putText(
                view, f"YOLOE | {len(detections)} detections | {fps:.1f} FPS",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2,
            )
            cv2.imshow("Autonomous Trash Catcher - YOLOE", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("p"):
                print("\n--- DETECTIONS ---")
                for item in detections:
                    print(
                        f"{item.label}: {item.confidence:.3f}, "
                        f"centre={item.center}, bbox={item.bbox}"
                    )
                if not detections:
                    print("No target objects detected.")
            elif key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
