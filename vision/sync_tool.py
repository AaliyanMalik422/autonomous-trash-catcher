"""Measure the time offset between two independent cameras (two phones).

Two phones stream with different latencies, so frames that arrive together were not taken
together. A ball at 5 m/s moves 5 cm in 10 ms, so a 30 ms mismatch ruins the stereo depth.

Procedure: point both cameras at the zone, run
    python -m vision.sync_tool --left http://PHONE1:8080/video --right http://PHONE2:8080/video
and for 6 seconds wave your arm / a ball back and forth quickly in view of BOTH cameras
(start/stop abruptly several times). It prints the offset to pass to main.py:
    --right-offset-ms <value>
"""
import argparse
import time

import cv2
import numpy as np


def estimate_offset(t_l, s_l, t_r, s_r, max_lag_s=0.5, grid_hz=1000.0):
    """Offset (seconds) to ADD to right timestamps so right events line up with left events.

    t_*: sample times, s_*: activity signals (e.g. motion energy per frame).
    """
    t_l, s_l, t_r, s_r = map(lambda a: np.asarray(a, dtype=float), (t_l, s_l, t_r, s_r))
    lo = max(t_l[0], t_r[0]) + max_lag_s
    hi = min(t_l[-1], t_r[-1]) - max_lag_s
    if hi - lo < 1.0:
        raise ValueError("Need at least ~2 s of overlapping data beyond the lag window")
    grid = np.arange(lo, hi, 1.0 / grid_hz)

    def z(a):
        return (a - a.mean()) / (a.std() + 1e-9)

    L = z(np.interp(grid, t_l, s_l))
    best, best_c = 0.0, -np.inf
    lags = np.arange(-max_lag_s, max_lag_s, 1.0 / grid_hz)
    cs = np.empty(len(lags))
    for i, off in enumerate(lags):
        R = np.interp(grid - off, t_r, s_r)        # R'(t) = R(t - off)
        cs[i] = np.dot(L, z(R)) / len(grid)
    k = int(np.argmax(cs))
    if 0 < k < len(cs) - 1:                          # parabolic refinement
        a, b, c = cs[k - 1], cs[k], cs[k + 1]
        denom = a - 2 * b + c
        frac = 0.5 * (a - c) / denom if denom != 0 else 0.0
        return float(lags[k] + frac / grid_hz), float(cs[k])
    return float(lags[k]), float(cs[k])


def motion_energy(prev_gray, gray):
    return float(np.mean(cv2.absdiff(prev_gray, gray)))


def record(left, right, seconds=6.0, fps=30):
    from .cameras import StereoCameras
    cams = StereoCameras(left, right, fps=fps, max_skew_s=1e9)   # no pairing needed here
    sig = {"L": ([], []), "R": ([], [])}
    prev = {"L": None, "R": None}
    seen = {"L": -1, "R": -1}
    end = time.monotonic() + seconds
    print(f"Recording {seconds:.0f} s: wave a ball/arm back and forth in front of BOTH cameras...")
    while time.monotonic() < end:
        for name, src in (("L", cams.L), ("R", cams.R)):
            for t, f, s in src.snapshot():
                if s <= seen[name]:
                    continue
                seen[name] = s
                g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (9, 9), 0)
                if prev[name] is not None:
                    sig[name][0].append(t)
                    sig[name][1].append(motion_energy(prev[name], g))
                prev[name] = g
        time.sleep(0.002)
    cams.release()
    return sig["L"], sig["R"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", default="0")
    ap.add_argument("--right", default="1")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()
    (tl, sl), (tr, sr) = record(a.left, a.right, a.seconds, a.fps)
    off, score = estimate_offset(tl, sl, tr, sr)
    print(f"offset = {off*1000:+.1f} ms   (match score {score:.2f}; want > 0.5, otherwise redo with sharper motions)")
    print(f"use:  python main.py --live ... --right-offset-ms {off*1000:.1f}")


if __name__ == "__main__":
    main()
