"""Does detection + the projectile model work on REAL throws? (one phone is enough)

`python -m vision.preview` saves each detected throw's image track to logs/real_throws/.
This script fits the first part of every flight with the pinhole-camera view of projectile motion
(valid for throws in any direction, even straight at the camera) and checks how well it predicts the remaining points the camera actually
saw. Small hold-out errors mean: the detector finds the ball, the timestamps are usable and the
physics model describes your real throws.

    python -m vision.check_throw_prediction                  # logs/real_throws/
    python -m vision.check_throw_prediction --dir some/folder --fit-frac 0.55
"""
import argparse
import glob
import json
import os

import numpy as np


def _fit_projective(t, u, v):
    """Image-space model of a ballistic 3D flight seen by a pinhole camera (any direction).

    X(t) and depth are linear in t, height is quadratic, and pixels are ratios by depth:
        u = (c0 + c1 t) / (1 + q t),   v = (b0 + b1 t + b2 t^2) / (1 + q t)
    with ONE shared q. Multiplying out makes it linear in (c0, c1, b0, b1, b2, q).
    """
    n = len(t)
    A = np.zeros((2 * n, 6))
    y = np.zeros(2 * n)
    A[:n, 0] = 1; A[:n, 1] = t; A[:n, 5] = -t * u; y[:n] = u
    A[n:, 2] = 1; A[n:, 3] = t; A[n:, 4] = t * t; A[n:, 5] = -t * v; y[n:] = v
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return sol


def _predict(sol, t):
    c0, c1, b0, b1, b2, q = sol
    den = 1.0 + q * t
    den = np.where(np.abs(den) < 0.05, 0.05, den)
    return (c0 + c1 * t) / den, (b0 + b1 * t + b2 * t * t) / den


def holdout_error(track, fit_frac=0.55, min_fit=6):
    """track: list of [t, u, v, area]. Returns dict or None if the track is too short.

    Fit the projective ballistic model to the first `fit_frac` of the points and predict the
    rest. Errors in pixels.
    """
    tr = np.asarray(track, dtype=float)
    if len(tr) < min_fit + 2:
        return None
    t, u, v = tr[:, 0] - tr[0, 0], tr[:, 1], tr[:, 2]
    m = max(min_fit, int(round(fit_frac * len(t))))
    if len(t) - m < 2:
        return None
    sol = _fit_projective(t[:m], u[:m], v[:m])
    pu, pv = _predict(sol, t)
    err = np.hypot(u[m:] - pu[m:], v[m:] - pv[m:])
    fit_rms = float(np.sqrt(np.mean(np.concatenate([u[:m] - pu[:m], v[:m] - pv[:m]]) ** 2)))
    path_px = float(np.sum(np.hypot(np.diff(u), np.diff(v))))
    fps = float(1.0 / np.median(np.diff(t)))
    return {"n": len(t), "n_fit": m, "fps": fps, "fit_rms_px": fit_rms,
            "holdout_mean_px": float(np.mean(err)), "holdout_max_px": float(np.max(err)),
            "path_px": path_px}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="logs/real_throws")
    ap.add_argument("--fit-frac", type=float, default=0.55)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, "*.json")))
    if not files:
        raise SystemExit(f"No throws in {a.dir}. Run `python -m vision.preview` and throw a ball first.")
    rows = []
    print(f"{'file':<18}{'pts':>4}{'fps':>6}{'fit rms':>9}{'hold-out mean':>15}{'max':>7}{'path':>7}")
    for f in files:
        with open(f) as fh:
            r = holdout_error(json.load(fh), a.fit_frac)
        if r is None:
            print(f"{os.path.basename(f):<18}  too short, skipped")
            continue
        rows.append(r)
        print(f"{os.path.basename(f):<18}{r['n']:>4}{r['fps']:>6.0f}{r['fit_rms_px']:>8.1f}p"
              f"{r['holdout_mean_px']:>14.1f}p{r['holdout_max_px']:>6.1f}p{r['path_px']:>6.0f}p")
    if not rows:
        raise SystemExit("No usable throws.")
    rel = np.median([r["holdout_mean_px"] / max(r["path_px"], 1) for r in rows]) * 100
    med = np.median([r["holdout_mean_px"] for r in rows])
    print(f"\n{len(rows)} throws | median hold-out error {med:.1f} px = {rel:.1f} % of the path length")
    if rel < 4:
        print("PASS: the model predicts your real throws well. Detection, timing and physics are consistent.")
    elif rel < 10:
        print("OK/WARN: usable, but noisy. Likely causes: low or uneven frame rate, motion blur (more light / "
              "faster shutter), or a track that includes non-ball points.")
    else:
        print("FAIL: large errors. Check the frame rate and stalls (phones_check), that the track is the ball "
              "(look at the preview), and that the phone is not moving.")


if __name__ == "__main__":
    main()
