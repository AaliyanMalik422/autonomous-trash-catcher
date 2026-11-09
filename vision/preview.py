"""Single-camera live preview of the ball detector (no stereo calibration needed).

    python -m vision.preview                       # USB webcam 0
    python -m vision.preview --source 1            # another webcam
    python -m vision.preview --source http://192.168.1.20:8080/video   # phone stream
    python -m vision.preview --source clip.mp4     # recorded video

Keep the scene empty for ~1 s while the background is learned, then throw a ball.
Keys: q = quit, r = relearn background, s = save a screenshot.
"""
import argparse
import collections
import time

import cv2
import numpy as np

from .stereo_tracker import BlobDetector, DetectorConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--min-area", type=float, default=60.0)
    ap.add_argument("--max-area", type=float, default=2500.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    a = ap.parse_args()
    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {a.source!r}")
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cfg = DetectorConfig(min_area=a.min_area, max_area=a.max_area)
    det = BlobDetector(cfg)
    trail = collections.deque(maxlen=30)
    last, fps_t, fps = None, time.time(), 0.0
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        res = det.detect(frame, last)
        vis = frame.copy()
        if res is not None:
            cx, cy, area = res
            last = (cx, cy)
            trail.append((int(cx), int(cy)))
            cv2.circle(vis, (int(cx), int(cy)), 12, (0, 255, 0), 2)
            cv2.putText(vis, f"ball area={area:.0f}px", (int(cx) + 14, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            last = None
        for i in range(1, len(trail)):
            cv2.line(vis, trail[i - 1], trail[i], (0, 200, 255), 2)
        learning = det.frames <= cfg.warmup_frames
        if n % 10 == 0:
            now = time.time(); fps = 10 / (now - fps_t); fps_t = now
        cv2.putText(vis, ("LEARNING BACKGROUND - keep still" if learning else "READY - throw!")
                    + f"   {fps:.0f} fps", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if learning else (0, 255, 0), 2)
        mask = det.bg.getBackgroundImage()
        fg = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask) if mask is not None else None
        panel = vis if fg is None else np.hstack([vis, cv2.cvtColor(cv2.threshold(fg, 25, 255, cv2.THRESH_BINARY)[1], cv2.COLOR_GRAY2BGR)])
        cv2.imshow("ball detector preview", panel)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("r"):
            det = BlobDetector(cfg); trail.clear()
        if k == ord("s"):
            cv2.imwrite(f"preview_{int(time.time())}.png", panel)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
