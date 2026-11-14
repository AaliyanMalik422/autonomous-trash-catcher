"""Single-camera detection and image-plane tracking demo; no robot control."""
from __future__ import annotations

import argparse
from collections import deque
import math
import os
import time

import cv2

from .cameras import open_capture, parse_source
from .stereo_tracker import BlobDetector, DetectorConfig
from .yoloe_detector import DEFAULT_TRASH_CLASSES, YOLOETrashDetector


def build_parser():
    parser = argparse.ArgumentParser(
        description="Single-camera moving-object detection and 2D tracking demo."
    )
    parser.add_argument("--source", default="2", help="Camera index or stream URL")
    parser.add_argument("--backend", choices=("blob", "yoloe"), default="blob")
    parser.add_argument("--model", default="yoloe-26n-seg.pt")
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--record", nargs="?", const="logs/single_camera_demo.mp4",
                        help="save annotated demo video (optional path)")
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_TRASH_CLASSES))
    return parser


def main():
    args = build_parser().parse_args()
    camera = open_capture(parse_source(args.source), args.width, args.height, fps=30)
    if not camera.isOpened():
        camera.release()
        raise SystemExit(f"Cannot open camera {args.source!r}")

    writer = None
    try:
        if args.backend == "blob":
            detector = BlobDetector(DetectorConfig(
                min_area=60.0, max_area=15000.0, warmup_frames=30,
                person_exclusion_px=45,
            ))
            print("OpenCV motion detection: hold the scene still for 1 second, then move/throw an object.")
        else:
            detector = YOLOETrashDetector(
                args.model, args.confidence, args.imgsz, args.device, args.classes
            )

        if args.record:
            folder = os.path.dirname(args.record)
            if folder:
                os.makedirs(folder, exist_ok=True)
            writer = cv2.VideoWriter(
                args.record, cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                (args.width, args.height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create demo recording: {args.record}")
            print(f"Recording annotated demo to: {args.record}")

        trace = deque(maxlen=45)
        recent = deque(maxlen=10)
        last_center = None
        lost_frames = 0
        total_frames = 0
        detected_frames = 0
        previous = time.monotonic()
        fps = 0.0

        print("Demo active: green circle = detected object; line = recent image-plane path.")
        print("Press P to print statistics; Q/Esc to finish and save the recording.")
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError("Camera stopped delivering frames")
            total_frames += 1
            now = time.monotonic()
            view = frame.copy()
            center = None
            area = 0.0
            label = "moving object"

            if args.backend == "blob":
                item = detector.detect(frame, last_center if lost_frames <= 8 else None)
                if item is not None:
                    cx, cy, area = item
                    center = (int(round(cx)), int(round(cy)))
            else:
                items = detector.detect(frame)
                if items:
                    item = max(items, key=lambda d: d.confidence)
                    center = (int(round(item.center[0])), int(round(item.center[1])))
                    area = item.area
                    label = f"{item.label} {item.confidence:.2f}"
                    x1, y1, x2, y2 = item.bbox
                    cv2.rectangle(view, (x1, y1), (x2, y2), (0, 220, 0), 2)

            if center is not None:
                detected_frames += 1
                lost_frames = 0
                last_center = center
                trace.append(center)
                recent.append((now, center[0], center[1]))
                radius = max(7, min(65, int(math.sqrt(max(area, 1.0) / math.pi))))
                cv2.circle(view, center, radius, (0, 220, 0), 2)
                cv2.circle(view, center, 4, (0, 0, 255), -1)
                cv2.putText(view, f"DETECTED: {label}", (10, args.height - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
                cv2.putText(view, f"centre=({center[0]}, {center[1]}) px",
                            (10, args.height - 17), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, (255, 255, 0), 1, cv2.LINE_AA)
            else:
                lost_frames += 1
                if lost_frames == 9:
                    trace.clear()
                    recent.clear()
                if lost_frames > 8:
                    last_center = None

            # The trace is deliberately labelled image-plane motion, not 3D trajectory.
            if len(trace) > 1:
                for index in range(1, len(trace)):
                    thickness = max(1, int(3 * index / len(trace)))
                    cv2.line(view, trace[index - 1], trace[index], (255, 180, 0),
                             thickness, cv2.LINE_AA)

            speed_px_s = 0.0
            if len(recent) >= 2:
                t0, x0, y0 = recent[0]
                t1, x1, y1 = recent[-1]
                if t1 > t0:
                    speed_px_s = math.hypot(x1 - x0, y1 - y0) / (t1 - t0)

            instant = 1.0 / max(now - previous, 1e-6)
            previous = now
            fps = instant if fps == 0 else 0.85 * fps + 0.15 * instant
            warming = args.backend == "blob" and detector.frames <= detector.cfg.warmup_frames
            if warming:
                status = f"BACKGROUND WARM-UP {detector.frames}/{detector.cfg.warmup_frames}"
            elif center is not None:
                status = "OBJECT TRACKED"
            else:
                status = "SEARCHING FOR MOVING OBJECT"

            cv2.putText(view, "AUTONOMOUS TRASH CATCHER | SINGLE-CAMERA DEMO",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
            cv2.putText(view, f"{status} | {fps:.1f} FPS",
                        (10, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 0), 2)
            cv2.putText(view, f"image-plane speed: {speed_px_s:.0f} px/s | frames detected: {detected_frames}",
                        (10, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            cv2.putText(view, "2D camera tracking only - stereo depth / robot disabled",
                        (10, args.height - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (220, 220, 220), 1)

            if writer is not None:
                writer.write(view)
            cv2.imshow("Autonomous Trash Catcher - Single Camera Demo", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("p"):
                ratio = detected_frames / max(total_frames, 1)
                print(f"Frames: {total_frames}; detected: {detected_frames}; "
                      f"detection ratio: {ratio:.1%}; current image speed: {speed_px_s:.1f}px/s")
            elif key in (ord("q"), 27):
                break

        print(f"Demo finished. {detected_frames}/{total_frames} frames contained a motion detection.")
        if writer is not None:
            print(f"Saved recording: {args.record}")
    finally:
        camera.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
