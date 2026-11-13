"""Step 1 of phone bring-up: are both USB/UVC camera devices alive, fast and steady enough?

    python -m vision.phones_check                    # uses config/robot.json
    python -m vision.phones_check --left URL --right URL --fps 60 --seconds 8

Prints per-camera resolution, real frame rate, jitter and the longest stall, with a
PASS/WARN/FAIL verdict and what to change in the phone app if it is not good enough.
"""
import argparse
import time

import cv2
import numpy as np

from trajectory.robot_config import load_settings


def summarize(times):
    """Frame-arrival times -> dict(fps, jitter_ms, max_gap_ms, n)."""
    t = np.asarray(times, dtype=float)
    if len(t) < 5:
        return {"fps": 0.0, "jitter_ms": float("inf"), "max_gap_ms": float("inf"), "n": len(t)}
    dt = np.diff(t)
    return {"fps": float(1.0 / np.mean(dt)), "jitter_ms": float(np.std(dt) * 1000),
            "max_gap_ms": float(np.max(dt) * 1000), "n": len(t)}


def verdict(stats, target_fps, size, want_size):
    msgs, level = [], "PASS"

    def worse(l):
        nonlocal level
        order = ["PASS", "WARN", "FAIL"]
        level = order[max(order.index(level), order.index(l))]

    if stats["n"] < 5:
        return "FAIL", ["no frames received - check the camera index, USB webcam mode, data cable and OS camera permissions"]
    if stats["fps"] < 0.5 * target_fps:
        worse("FAIL"); msgs.append(f"only {stats['fps']:.0f} fps (target {target_fps}): lower resolution/quality "
                                   "check the USB cable/port and camera output mode, reduce resolution if needed, and improve lighting")
    elif stats["fps"] < 0.85 * target_fps:
        worse("WARN"); msgs.append(f"{stats['fps']:.0f} fps is below target {target_fps}")
    if stats["max_gap_ms"] > 150:
        worse("WARN"); msgs.append(f"stalls up to {stats['max_gap_ms']:.0f} ms: Wi-Fi congestion, move the router closer")
    if stats["jitter_ms"] > 8:
        worse("WARN"); msgs.append(f"uneven frame timing ({stats['jitter_ms']:.1f} ms jitter)")
    if size != want_size:
        worse("WARN"); msgs.append(f"stream is {size[0]}x{size[1]}, expected {want_size[0]}x{want_size[1]}: "
                                   "set the camera output resolution (calibration must use the same size)")
    return level, msgs


class _SingleRig:
    """Looks like StereoCameras but opens ONE stream (the right side mirrors the left)."""

    def __init__(self, src, w, h, fps):
        from vision.cameras import _Source, parse_source
        self.L = _Source(parse_source(src), w, h, fps)
        self.R = self.L

    def release(self):
        self.L.release()


def main():
    s = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", default=s.left)
    ap.add_argument("--right", default=s.right)
    ap.add_argument("--fps", type=int, default=s.fps)
    ap.add_argument("--width", type=int, default=s.width)
    ap.add_argument("--height", type=int, default=s.height)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--single", action="store_true",
                    help="only check the --left stream (one phone; never open the same URL twice)")
    ap.add_argument("--warmup", type=float, default=1.0, help="seconds ignored at the start")
    a = ap.parse_args()
    if "PHONE" in str(a.left) + ("" if a.single else str(a.right)):
        raise SystemExit("Edit config/robot.json: set left/right to the detected USB camera indices or valid camera stream URLs.")
    from vision.cameras import StereoCameras
    if a.single:
        a.right = a.left
    cams = StereoCameras(a.left, a.right, a.width, a.height, a.fps, max_skew_s=1e9) if not a.single \
        else _SingleRig(a.left, a.width, a.height, a.fps)
    times = {"L": [], "R": []}
    seen = {"L": -1, "R": -1}
    size = {"L": None, "R": None}
    last = {"L": None, "R": None}
    end = time.monotonic() + a.seconds
    print(f"Checking both cameras for {a.seconds:.0f} s ...")
    while time.monotonic() < end:
        for name, src in (("L", cams.L), ("R", cams.R)):
            for t, f, sq in src.snapshot():
                if sq > seen[name]:
                    seen[name] = sq
                    times[name].append(t)
                    size[name] = (f.shape[1], f.shape[0])
                    last[name] = f
        if not a.no_window and last["L"] is not None and last["R"] is not None:
            h = min(last["L"].shape[0], last["R"].shape[0])
            cv2.imshow("phones check (q to stop)", np.hstack([last["L"][:h], last["R"][:h]]))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        time.sleep(0.002)
    cams.release()
    cv2.destroyAllWindows()
    worst = "PASS"
    names = (("L", "phone"),) if a.single else (("L", "left "), ("R", "right"))
    for name, label in names:
        t0 = times[name][0] if times[name] else 0.0
        st = summarize([t for t in times[name] if t >= t0 + a.warmup])
        lvl, msgs = verdict(st, a.fps, size[name] or (0, 0), (a.width, a.height))
        print(f"[{lvl}] {label}: {size[name]} {st['fps']:.1f} fps, jitter {st['jitter_ms']:.1f} ms, "
              f"longest stall {st['max_gap_ms']:.0f} ms")
        for m in msgs:
            print("       -", m)
        if lvl == "FAIL" or (lvl == "WARN" and worst == "PASS"):
            worst = lvl
    print("\nOverall:", worst, "- next: python -m vision.sync_tool, then python -m vision.latency_test"
          if worst != "FAIL" else "\nOverall: FAIL - fix the items above, then run this check again")


if __name__ == "__main__":
    main()
