"""Stereo calibration with a printed checkerboard.

    python -m vision.calibrate_stereo capture --left 0 --right 1 --out calib_images
        SPACE = save a pair (only when the board is found in BOTH views), q = quit.
        Take 20-30 pairs: board near/far, tilted, covering the whole image, held still.
    python -m vision.calibrate_stereo solve --images calib_images --cols 9 --rows 6 \
            --square 0.025 --out stereo_calib.npz
        cols/rows = number of INNER corners; square = square side in metres.
Prints the reprojection RMS (aim < 0.6 px) and the baseline (check against a ruler).
"""
import argparse
import glob
import os

import cv2
import numpy as np

from .stereo_tracker import StereoCalibration


def capture(a):
    from .cameras import StereoCameras
    from trajectory.robot_config import load_settings
    st = load_settings()
    a.left = a.left or st.left
    a.right = a.right or st.right
    if "PHONE" in str(a.left) + str(a.right):
        raise SystemExit("Edit config/robot.json first: put the real stream URLs in left/right.")
    os.makedirs(a.out, exist_ok=True)
    cams = StereoCameras(a.left, a.right, st.width, st.height, st.fps, swap=st.swap,
                         max_skew_s=0.05)   # hold the board still while saving
    n, seq = 0, -1
    pattern = (a.cols, a.rows)
    while True:
        r = cams.read(seq)
        if r is None:
            continue
        _, L, R, seq = r
        fl, _ = cv2.findChessboardCorners(cv2.cvtColor(L, cv2.COLOR_BGR2GRAY), pattern,
                                          cv2.CALIB_CB_FAST_CHECK)
        fr, _ = cv2.findChessboardCorners(cv2.cvtColor(R, cv2.COLOR_BGR2GRAY), pattern,
                                          cv2.CALIB_CB_FAST_CHECK)
        vis = np.hstack([L, R]).copy()
        cv2.putText(vis, f"pairs {n}  L:{'OK' if fl else '--'} R:{'OK' if fr else '--'}  SPACE=save q=quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if fl and fr else (0, 0, 255), 2)
        cv2.imshow("stereo calibration", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" ") and fl and fr:
            cv2.imwrite(os.path.join(a.out, f"left_{n:02d}.png"), L)
            cv2.imwrite(os.path.join(a.out, f"right_{n:02d}.png"), R)
            n += 1
    cams.release()
    cv2.destroyAllWindows()


def solve(a):
    if a.cols < 2 or a.rows < 2 or not np.isfinite(a.square) or a.square <= 0:
        raise SystemExit("Checkerboard dimensions must be >= 2 and --square must be a positive metre value.")
    pattern = (a.cols, a.rows)
    objp = np.zeros((a.cols * a.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square

    # Match by the numeric suffix (00, 01, ...) instead of zip(sorted(...)):
    # a missing image must never shift every subsequent left/right pairing.
    lf = {os.path.basename(f)[5:]: f for f in glob.glob(os.path.join(a.images, "left_*.png"))}
    rf = {os.path.basename(f)[6:]: f for f in glob.glob(os.path.join(a.images, "right_*.png"))}
    keys = sorted(set(lf) & set(rf))
    if not keys:
        raise SystemExit(f"No matching left_NN.png/right_NN.png pairs found in {a.images!r}")
    if len(lf) != len(rf) or set(lf) != set(rf):
        print(f"WARNING: unmatched images ignored (left={len(lf)}, right={len(rf)}, matched={len(keys)})")

    obj, pl, pr, size = [], [], [], None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    for key in keys:
        L = cv2.imread(lf[key], cv2.IMREAD_GRAYSCALE)
        R = cv2.imread(rf[key], cv2.IMREAD_GRAYSCALE)
        if L is None or R is None:
            print(f"WARNING: unreadable pair {key}; skipped")
            continue
        if L.shape != R.shape:
            raise SystemExit(f"Pair {key} has different left/right image sizes: {L.shape[::-1]} vs {R.shape[::-1]}")
        pair_size = L.shape[::-1]
        if size is None:
            size = pair_size
        elif pair_size != size:
            raise SystemExit(f"Image resolution changed: expected {size}, got {pair_size} in pair {key}")

        okl, cl = cv2.findChessboardCorners(L, pattern)
        okr, cr = cv2.findChessboardCorners(R, pattern)
        if okl and okr:
            obj.append(objp.copy())
            pl.append(cv2.cornerSubPix(L, cl, (11, 11), (-1, -1), crit))
            pr.append(cv2.cornerSubPix(R, cr, (11, 11), (-1, -1), crit))

    if len(obj) < 10:
        raise SystemExit(f"Only {len(obj)} usable pairs; need >= 10 (20-30 recommended)")
    left_rms, K1, D1, _, _ = cv2.calibrateCamera(obj, pl, size, None, None)
    right_rms, K2, D2, _, _ = cv2.calibrateCamera(obj, pr, size, None, None)
    stereo_rms, K1, D1, K2, D2, R, T, _, _ = cv2.stereoCalibrate(
        obj, pl, pr, K1, D1, K2, D2, size, flags=cv2.CALIB_FIX_INTRINSIC, criteria=crit)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, size, R, T, alpha=0)

    signed_baseline = -float(P2[0, 3] / P2[0, 0])
    if not np.isfinite(signed_baseline) or signed_baseline <= 0:
        raise SystemExit(
            f"Stereo baseline has the wrong sign ({signed_baseline:.4f} m). "
            "Check physical left/right camera order; swap the two sources and recapture if needed."
        )

    m1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_16SC2)
    m2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_16SC2)
    cal = StereoCalibration(size, P1, P2, m1, m2, R1)
    cal.save(a.out)
    print(f"{len(obj)} pairs | stereo RMS {stereo_rms:.3f} px | "
          f"left intrinsic RMS {left_rms:.3f} px | right intrinsic RMS {right_rms:.3f} px | "
          f"baseline {cal.baseline*100:.1f} cm | focal {cal.focal:.0f} px -> saved {a.out}")
    if max(left_rms, right_rms, stereo_rms) > 1.0:
        print("WARNING: RMS > 1 px. Retake pairs: board sharp, flat, and spread across the image.")
    elif max(left_rms, right_rms, stereo_rms) > 0.6:
        print("NOTE: RMS is above the preferred 0.6 px target; validate depth at known distances.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--left")
    c.add_argument("--right")
    c.add_argument("--cols", type=int, default=9)
    c.add_argument("--rows", type=int, default=6)
    c.add_argument("--out", default="calib_images")
    s = sub.add_parser("solve")
    s.add_argument("--images", default="calib_images")
    s.add_argument("--cols", type=int, default=9)
    s.add_argument("--rows", type=int, default=6)
    s.add_argument("--square", type=float, default=0.025)
    s.add_argument("--out", default="stereo_calib.npz")
    args = ap.parse_args()
    capture(args) if args.cmd == "capture" else solve(args)
