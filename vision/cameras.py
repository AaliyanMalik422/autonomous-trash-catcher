"""Dual-camera capture for two USB webcams OR two phone cameras (IP / DroidCam / Iriun streams).

Each camera runs in its own thread (a slow Wi-Fi stream must not block the other one) and
keeps a short history of (arrival_time, frame). `read()` returns the newest left frame
paired with the right frame closest in time, after applying a fixed per-camera time offset
(see vision/sync_tool.py to measure it). Pairs further apart than `max_skew_s` are dropped.

Sources: an int (USB index) or a string URL, e.g.
    "http://192.168.1.20:8080/video"      (IP Webcam app, MJPEG)
    "http://192.168.1.21:4747/video"      (DroidCam)
    "rtsp://192.168.1.22:8554/live"
"""
import collections
import threading
import time
from typing import Optional, Tuple, Union

import cv2

Source = Union[int, str]


def parse_source(s) -> Source:
    return int(s) if isinstance(s, (int,)) or str(s).isdigit() else str(s)


def pair_frames(t_left: float, right_hist, right_offset_s: float, max_skew_s: float):
    """Pick the right-camera item whose (t + offset) is closest to t_left.

    right_hist: iterable of (t, frame, seq). Returns the item or None if none is close enough.
    """
    best, best_d = None, None
    for item in right_hist:
        d = abs(item[0] + right_offset_s - t_left)
        if best_d is None or d < best_d:
            best, best_d = item, d
    if best is None or best_d > max_skew_s:
        return None
    return best


class _Source:
    """One camera: background thread + short frame history."""

    def __init__(self, source: Source, width, height, fps, fourcc="MJPG", capture=None):
        self.cap = capture
        if self.cap is None:
            self.cap = cv2.VideoCapture(source, cv2.CAP_ANY)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera source {source!r}")
            if isinstance(source, int):   # network streams ignore these (set them in the phone app)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.hist = collections.deque(maxlen=12)
        self.lock = threading.Lock()
        self.seq = 0
        self._stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self._stop:
            if not self.cap.grab():
                time.sleep(0.005)
                continue
            t = time.monotonic()
            ok, frame = self.cap.retrieve()
            if not ok or frame is None:
                continue
            with self.lock:
                self.seq += 1
                self.hist.append((t, frame, self.seq))

    def snapshot(self):
        with self.lock:
            return list(self.hist)

    def release(self):
        self._stop = True
        self.thread.join(timeout=1.0)
        self.cap.release()


class StereoCameras:
    def __init__(self, left: Source = 0, right: Source = 1, width=640, height=480, fps=30,
                 swap=False, fourcc="MJPG", right_offset_s: float = 0.0,
                 max_skew_s: Optional[float] = None, captures=None):
        left, right = parse_source(left), parse_source(right)
        if swap:
            left, right = right, left
        self.right_offset_s = right_offset_s
        # allow up to ~0.6 of a frame period of skew by default
        self.max_skew_s = max_skew_s if max_skew_s is not None else 0.6 / max(fps, 1)
        caps = captures or (None, None)
        self.L = _Source(left, width, height, fps, fourcc, caps[0])
        self.R = _Source(right, width, height, fps, fourcc, caps[1])
        self.skew_log = collections.deque(maxlen=200)

    def read(self, last_seq: int = -1, timeout: float = 1.0):
        """Newest synchronized pair newer than last_seq: (t, left, right, seq) or None."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            lh = self.L.snapshot()
            if lh and lh[-1][2] != last_seq:
                tl, fl, sl = lh[-1]
                item = pair_frames(tl, self.R.snapshot(), self.right_offset_s, self.max_skew_s)
                if item is not None:
                    self.skew_log.append(item[0] + self.right_offset_s - tl)
                    return tl, fl, item[1], sl
            time.sleep(0.001)
        return None

    def release(self):
        self.L.release()
        self.R.release()
