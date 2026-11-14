"""Single-camera object-motion preview; never connects to or commands the robot."""
from __future__ import annotations

import argparse
import math
import time

import cv2

from .cameras import open_capture, parse_source
from .stereo_tracker import BlobDetector, DetectorConfig
from .yoloe_detector import DEFAULT_TRASH_CLASSES, YOLOETrashDetector


def build_parser():
    parser = argparse.ArgumentParser(description="Live object detection without robot control.")
    parser.add_argument("--source", default="2", help="Camera index or stream URL")
    parser.add_argument("--backend", choices=("blob", "yoloe"), default="blob")
    parser.add_argument("--model", default="yoloe-26n-seg.pt")
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_TRASH_CLASSES))
    return parser


def main():
    args = build_parser().parse_args()
    camera = open_capture(parse_source(args.source), args.width, args.height, fps=30)
    if not camera.isOpened():
        camera.release()
        raise SystemExit(f"Cannot open camera {args.source!r}")

    try:
        if args.backend == "blob":
            detector = BlobDetector(DetectorConfig(
                min_area=60.0, max_area=15000.0, warmup_frames=30,
                person_exclusion_px=45,
            ))
            print("OpenCV motion detector: keep the scene still for 1 second, then move an object.")
        else:
            detector = YOLOETrashDetector(
                args.model, args.confidence, args.imgsz, args.device, args.classes
            )
        previous = time.monotonic()
        fps = 0.0
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError("Camera stopped delivering frames")
            view = frame.copy()
            found = []
            if args.backend == "blob":
                item = detector.detect(frame)
                if item is not None:
                    cx, cy, area = item
                    center = (int(cx), int(cy))
                    radius = max(6, min(90, int(math.sqrt(area / math.pi))))
                    cv2.circle(view, center, radius, (0, 220, 0), 2)
                    cv2.circle(view, center, 4, (0, 0, 255), -1)
                    found.append((center, area))
            else:
                found = detector.detect(frame)
                for item in found:
                    x1, y1, x2, y2 = item.bbox
                    cx, cy = (int(v) for v in item.center)
                    cv2.rectangle(view, (x1, y1), (x2, y2), (0, 220, 0), 2)
                    cv2.putText(view, f"{item.label} {item.confidence:.2f}",
                                (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.48, (0, 220, 0), 1, cv2.LINE_AA)
            now = time.monotonic()
            instant = 1.0 / max(now - previous, 1e-6)
            previous = now
            fps = instant if fps == 0 else 0.85 * fps + 0.15 * instant
            warm = args.backend == "blob" and detector.frames <= detector.cfg.warmup_frames
            status = f"warm-up {detector.frames}/{detector.cfg.warmup_frames}" if warm else f"{len(found)} detections"
            cv2.putText(view, f"{args.backend.upper()} | {status} | {fps:.1f} FPS",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
            cv2.imshow("Autonomous Trash Catcher - Detection", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("p"):
                if args.backend == "blob":
                    print("Motion candidate:", found if found else "none")
                else:
                    print([(d.label, round(d.confidence, 3), d.center) for d in found] or "No detections")
            elif key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
