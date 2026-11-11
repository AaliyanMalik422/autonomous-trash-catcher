"""Single-camera live preview of the ball detector (no stereo calibration needed).

    python -m vision.preview                       # USB webcam 0
    python -m vision.preview --source 1            # another webcam
    python -m vision.preview --source http://192.168.1.20:8080/video   # phone stream
    python -m vision.preview --source clip.mp4     # recorded video

Keep the scene empty for ~1 s while the background is learned, then throw a ball.
Shows: camera view with the detected blob + its trail, and the foreground mask.
Keys: q = quit, r = relearn background, s = save a screenshot.
"""
import argparse
import collections
import time

import cv2
import numpy as np

from .stereo_tracker import BlobDetector, DetectorConfig


def throw_stats(track):
    """(n_points, span_s, speed_px_s, acc_px_s2, rms_px) of a 2D track, or None if < 3 points."""
    if len(track) < 3:
        return None
    t = np.array([p[0] for p in track]); x = np.array([p[1] for p in track]); y = np.array([p[2] for p in track])
    span = float(t[-1] - t[0])
    if span <= 0:
        return None
    t = t - t[-1]
    deg = 2 if len(track) >= 4 else 1
    qy = np.polyfit(t, y, deg)
    qx = np.polyfit(t, x, 1)
    rms = float(np.sqrt(np.mean(np.concatenate([y - np.polyval(qy, t), x - np.polyval(qx, t)]) ** 2)))
    speed = float(np.hypot(qx[0], np.polyfit(t, y, 1)[0]))
    acc = 2.0 * qy[0] if deg == 2 else 0.0
    return len(track), span, speed, float(acc), rms


def throw_like(track, min_speed=250.0, acc_range=(300.0, 12000.0), max_rms_px=4.0,
               min_points=5, span_range=(0.2, 0.9)):
    """Is this 2D track a thrown object? Fast, curving DOWNWARD in the image like free fall,
    and well described by a parabola (hand waves / swinging arms are not).
    Works down to ~15 fps (needs >= 5 points spanning 0.2-0.9 s).
    """
    st = throw_stats(track)
    if st is None:
        return False
    n, span, speed, acc, rms = st
    return (n >= min_points and span_range[0] <= span <= span_range[1]
            and speed >= min_speed and acc_range[0] <= acc <= acc_range[1] and rms <= max_rms_px)


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
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # MJPG usually allows 30 fps
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cfg = DetectorConfig(min_area=a.min_area, max_area=a.max_area)
    det = BlobDetector(cfg)
    trail = collections.deque(maxlen=30)
    track = collections.deque(maxlen=24)   # (t, x, y) for the flight check
    thrown_until = 0.0
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
            now_t = time.time()
            track.append((now_t, cx, cy))
            if throw_like(list(track)):
                thrown_until = now_t + 0.5
            is_throw = now_t < thrown_until
            if is_throw:
                trail.append((int(cx), int(cy)))
            else:
                trail.clear()
            colour = (0, 255, 0) if is_throw else (150, 150, 150)
            cv2.circle(vis, (int(cx), int(cy)), 12, colour, 2)
            cv2.putText(vis, ("THROW" if is_throw else "moving") + f" area={area:.0f}px",
                        (int(cx) + 14, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        else:
            last = None
            if track and time.time() - track[-1][0] > 0.35:
                track.clear()
        st = throw_stats(list(track)) if track else None
        if st is not None:
            cv2.putText(vis, f"track pts={st[0]} span={st[1]:.2f}s speed={st[2]:.0f} acc={st[3]:.0f} rms={st[4]:.1f}",
                        (10, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
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
            det = BlobDetector(cfg); trail.clear(); track.clear()
        if k == ord("s"):
            cv2.imwrite(f"preview_{int(time.time())}.png", panel)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
