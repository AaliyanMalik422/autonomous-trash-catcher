"""Measure how long a camera takes to show what happened (the number for `latency_s`).

Point ONE phone at this laptop's screen (screen filling most of the view, room lights dim).
The tool flashes the screen white at random moments and measures how long until the white
appears in the phone's stream.

    python -m vision.latency_test --source http://PHONE1_IP:8080/video
    python -m vision.latency_test --source 0 --trials 10

The result includes the screen's own delay (~20-40 ms), so it slightly overestimates (safe).
"""
import argparse
import random
import time

import cv2
import numpy as np


def detect_flash_latency(frame_times, brightness, t_flash, low, high):
    """Seconds from t_flash until the first frame whose brightness crosses the midpoint
    between `low` (screen black) and `high` (screen white). None if never seen."""
    thr = 0.5 * (low + high)
    for t, b in zip(frame_times, brightness):
        if t >= t_flash and b >= thr:
            return float(t - t_flash)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--processing_s", type=float, default=0.03,
                    help="extra time for detection + serial to add to the suggestion")
    a = ap.parse_args()
    src = int(a.source) if a.source.isdigit() else a.source
    from vision.cameras import _Source
    cam = _Source(src, 640, 480, 60)
    win = "latency test (press q to abort)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    black = np.zeros((600, 800, 3), np.uint8)
    white = np.full((600, 800, 3), 255, np.uint8)
    seen, ft, fb = -1, [], []

    def pump():
        nonlocal seen
        for t, f, sq in cam.snapshot():
            if sq > seen:
                seen = sq
                ft.append(t)
                fb.append(float(np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))))

    def show(img, secs):
        end = time.monotonic() + secs
        cv2.imshow(win, img)
        while time.monotonic() < end:
            pump()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

    results = []
    try:
        show(black, 1.5)
        low = float(np.median(fb[-20:])) if fb else 0.0
        show(white, 1.0)                              # learn the "white" level
        high = float(np.percentile(fb[-20:], 90)) if fb else 255.0
        if high - low < 15:
            raise SystemExit("The phone barely sees the flash. Point it at the screen, dim the room "
                             "lights and raise the screen brightness.")
        for i in range(a.trials):
            show(black, random.uniform(1.0, 2.2))
            pump()
            cv2.imshow(win, white)
            cv2.waitKey(1)
            t_flash = time.monotonic()
            show(white, 0.8)
            lat = detect_flash_latency(ft, fb, t_flash, low, high)
            if lat is not None:
                results.append(lat)
                print(f"trial {i+1}: {lat*1000:.0f} ms")
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        cv2.destroyAllWindows()
    if len(results) < 3:
        raise SystemExit("Too few valid trials.")
    med, p90 = float(np.median(results)), float(np.percentile(results, 90))
    print(f"\ncamera delay: median {med*1000:.0f} ms, 90th percentile {p90*1000:.0f} ms")
    print(f"=> set  \"latency_s\": {med + a.processing_s:.3f}  in config/robot.json")


if __name__ == "__main__":
    main()
