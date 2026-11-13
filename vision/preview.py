"""Single-camera live preview of the ball detector (no stereo calibration needed).

    python -m vision.preview --source http://PHONE-IP:8080/video      # phone stream
    python -m vision.preview                                          # USB webcam 0
    python -m vision.preview --source clip.mp4

Keep the scene empty for ~1 s while the background is learned, then throw a ball.
Green circle + "THROW" = fast object on a falling arc; grey = other motion.
Every detected throw's image track is saved to logs/real_throws/ for
`python -m vision.check_throw_prediction`.
Keys: q = quit, r = relearn background, s = save a screenshot.
A dead stream is reconnected automatically (the video says NO VIDEO meanwhile).
"""
import argparse
import collections
import json
import os
import time

import cv2
import numpy as np

from .cameras import _Source, parse_source
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


def run(a, capture=None, max_seconds=None, show=True):
    src = parse_source(a.source)
    cam = _Source(src, a.width, a.height, 30, capture=capture)
    cfg = DetectorConfig(min_area=a.min_area, max_area=a.max_area)
    det = BlobDetector(cfg)
    trail = collections.deque(maxlen=30)
    track = collections.deque(maxlen=24)          # (t, x, y) for the flight check
    flight, flight_was_throw, thrown_until = [], False, 0.0
    os.makedirs(a.save_dir, exist_ok=True)
    saved = len([f for f in os.listdir(a.save_dir) if f.endswith(".json")])
    last, seq, stamps, fps = None, -1, collections.deque(maxlen=30), 0.0
    last_frame_wall, n_saved_now = time.monotonic(), 0
    t_start = time.monotonic()
    try:
        while True:
            if max_seconds is not None and time.monotonic() - t_start > max_seconds:
                break
            items = [i for i in cam.snapshot() if i[2] > seq]
            if not items:
                if show:
                    blank = np.zeros((a.height, a.width, 3), np.uint8)
                    msg = "NO VIDEO - reconnecting..." if time.monotonic() - last_frame_wall > 1.0 else "waiting..."
                    cv2.putText(blank, msg, (20, a.height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("ball detector preview", blank)
                    if cv2.waitKey(10) & 0xFF == ord("q"):
                        break
                else:
                    time.sleep(0.005)
                continue
            for t_f, frame, seq in items:                  # process every new frame, oldest first
                last_frame_wall = time.monotonic()
                stamps.append(t_f)
                if len(stamps) >= 5:
                    fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
                res = det.detect(frame, last)
                vis = frame.copy()
                if res is not None:
                    cx, cy, area = res
                    last = (cx, cy)
                    track.append((t_f, cx, cy))
                    if throw_like(list(track)):
                        thrown_until = t_f + 0.5
                    is_throw = t_f < thrown_until
                    flight.append([t_f, float(cx), float(cy), float(area)])
                    flight_was_throw = flight_was_throw or is_throw
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
                    if track and t_f - track[-1][0] > 0.35:
                        track.clear()
                    if flight and t_f - flight[-1][0] > 0.35:          # the flight is over
                        if flight_was_throw and len(flight) >= 6:
                            saved += 1
                            n_saved_now += 1
                            path = os.path.join(a.save_dir, f"throw_{saved:04d}.json")
                            with open(path, "w") as f:
                                json.dump(flight, f)
                            print(f"[saved] {path}  ({len(flight)} points)")
                        flight, flight_was_throw = [], False
                st = throw_stats(list(track)) if track else None
                if st is not None:
                    cv2.putText(vis, f"track pts={st[0]} span={st[1]:.2f}s speed={st[2]:.0f} acc={st[3]:.0f} rms={st[4]:.1f}",
                                (10, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                for i in range(1, len(trail)):
                    cv2.line(vis, trail[i - 1], trail[i], (0, 200, 255), 2)
                learning = det.frames <= cfg.warmup_frames
                cv2.putText(vis, ("LEARNING BACKGROUND - keep still" if learning else "READY - throw!")
                            + f"   {fps:.0f} fps" + (f"   reconnects {cam.reconnects}" if cam.reconnects else ""),
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255) if learning else (0, 255, 0), 2)
            if show:
                bg = det.bg.getBackgroundImage()
                fg = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), bg) if bg is not None else None
                panel = vis if fg is None else np.hstack(
                    [vis, cv2.cvtColor(cv2.threshold(fg, 25, 255, cv2.THRESH_BINARY)[1], cv2.COLOR_GRAY2BGR)])
                cv2.imshow("ball detector preview", panel)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("r"):
                    det = BlobDetector(cfg); trail.clear(); track.clear()
                if k == ord("s"):
                    cv2.imwrite(f"preview_{int(time.time())}.png", panel)
    finally:
        cam.release()
        if show:
            cv2.destroyAllWindows()
    return n_saved_now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--min-area", type=float, default=60.0)
    ap.add_argument("--max-area", type=float, default=2500.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--save-dir", default="logs/real_throws",
                    help="every detected throw (image track) is saved here for check_throw_prediction")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
