"""Step 1 of phone bring-up: are both streams alive, fast and steady enough?

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
        return "FAIL", ["no frames received - check the URL / same Wi-Fi network / app is streaming"]
    if stats["fps"] < 0.5 * target_fps:
        worse("FAIL"); msgs.append(f"only {stats['fps']:.0f} fps (target {target_fps}): lower resolution/quality "
                                   "in the app, use 5 GHz Wi-Fi or USB tethering, and turn the lights on")
    elif stats["fps"] < 0.85 * target_fps:
        worse("WARN"); msgs.append(f"{stats['fps']:.0f} fps is below target {target_fps}")
    if stats["max_gap_ms"] > 150:
        worse("WARN"); msgs.append(f"stalls up to {stats['max_gap_ms']:.0f} ms: Wi-Fi congestion, move the router closer")
    if stats["jitter_ms"] > 8:
        worse("WARN"); msgs.append(f"uneven frame timing ({stats['jitter_ms']:.1f} ms jitter)")
    if size != want_size:
        worse("WARN"); msgs.append(f"stream is {size[0]}x{size[1]}, expected {want_size[0]}x{want_size[1]}: "
                                   "set the resolution in the phone app (calibration must use the same size)")
    return level, msgs


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
    a = ap.parse_args()
    if "PHONE" in str(a.left) + str(a.right):
        raise SystemExit("Edit config/robot.json first: replace PHONE1_IP / PHONE2_IP with the stream URLs.")
    from vision.cameras import StereoCameras
    cams = StereoCameras(a.left, a.right, a.width, a.height, a.fps, max_skew_s=1e9)
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
    for name, label in (("L", "left "), ("R", "right")):
        st = summarize(times[name])
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
