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
import numpy as np

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


class ClockSmoother:
    """Turns jittery frame-arrival stamps into the camera's true regular frame clock.

    A phone captures frames at a steady rate but they ARRIVE (and get stamped) with network /
    scheduling jitter of several ms. A robust line fit of stamp-vs-frame-number over the last
    frames removes that jitter, so two cameras can be paired by the right frame instead of a
    neighbouring one (a one-frame mix-up at 60 fps is a ~50 cm depth error).
    """

    def __init__(self, n=40, max_dev_frac=0.4):
        self.n, self.max_dev_frac = n, max_dev_frac
        self.k = []
        self.t = []

    def update(self, k: int, t: float) -> float:
        if len(self.k) >= 8:
            K, T = np.array(self.k, float), np.array(self.t, float)
            slopes = np.diff(T) / np.diff(K)
            b = float(np.median(slopes))
            a = float(np.median(T - b * K))
            pred = a + b * k
            if abs(t - pred) <= self.max_dev_frac * b:
                self.k.append(k); self.t.append(pred)       # store the smoothed time
                self.k, self.t = self.k[-self.n:], self.t[-self.n:]
                return pred
            if abs(t - pred) > 4 * b:                       # clock jumped / long stall: resync
                self.k, self.t = [], []
        self.k.append(k); self.t.append(t)
        self.k, self.t = self.k[-self.n:], self.t[-self.n:]
        return t


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
        self.clock = ClockSmoother()
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
                self.hist.append((self.clock.update(self.seq, t), frame, self.seq))

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
        self.last_times = (0.0, 0.0)   # (left, right+offset) capture times of the last pair returned

    def read(self, last_seq: int = -1, timeout: float = 1.0):
        """Newest synchronized pair: (t, left, right, key) or None.

        t is the capture time of the pair on the left camera's clock. The pair is anchored on
        whichever camera's newest frame is OLDER after applying the offset (the slower/more
        delayed stream), so its partner has already arrived. `key` changes for every new pair.
        """
        off = self.right_offset_s
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            lh, rh = self.L.snapshot(), self.R.snapshot()
            if lh and rh:
                tl, tr = lh[-1][0], rh[-1][0] + off
                pair = None
                if tl <= tr:                      # left is the older head: anchor on left
                    item = pair_frames(tl, rh, off, self.max_skew_s)
                    if item is not None:
                        pair = (tl, lh[-1][1], item[1], lh[-1][2], item[2], item[0] + off - tl, (tl, item[0] + off))
                else:                             # right is the older head: anchor on right
                    item = pair_frames(tr, lh, 0.0, self.max_skew_s)
                    if item is not None:
                        pair = (tr, item[1], rh[-1][1], item[2], rh[-1][2], item[0] - tr, (item[0], tr))
                if pair is not None:
                    key = pair[3] * 1000003 + pair[4]
                    if key != last_seq:
                        self.skew_log.append(pair[5])
                        self.last_times = pair[6]
                        return pair[0], pair[1], pair[2], key
            time.sleep(0.001)
        return None

    def release(self):
        self.L.release()
        self.R.release()
