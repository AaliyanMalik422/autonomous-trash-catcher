"""Enumerate USB/UVC camera devices that OpenCV can actually read.

Connect both phones by USB in webcam/UVC mode, then run:
    python -m vision.list_cameras --max-index 8

Use the two reported indices as left/right in config/robot.json. This checks
whether the operating system exposes each phone as a camera; it does not
calibrate stereo geometry.
"""
import argparse

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-index", type=int, default=8,
                        help="probe indices from 0 up to (but not including) this value")
    args = parser.parse_args()

    found = 0
    for index in range(max(0, args.max_index)):
        cap = cv2.VideoCapture(index)
        try:
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"index {index}: opened but no frame")
                continue
            h, w = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc_num = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc = "".join(chr((fourcc_num >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")
            print(f"index {index}: {w}x{h}, reported FPS={fps:.1f}, FOURCC={fourcc or 'unknown'}")
            found += 1
        finally:
            cap.release()

    if found == 0:
        raise SystemExit("No readable camera devices found. Check USB webcam mode, cable/data connection, and OS permissions.")
    print(f"Found {found} readable camera device(s).")


if __name__ == "__main__":
    main()
