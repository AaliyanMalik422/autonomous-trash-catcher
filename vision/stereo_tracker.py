"""Stereo detection + 3D triangulation (matches the proposal: OpenCV background
subtraction + contour filtering, no pretrained detector).

Pipeline per frame pair:
    rectify L/R -> MOG2 background subtraction -> morphology -> contours ->
    pick the best ball-sized blob in each image -> epipolar check (same row) ->
    disparity -> 3D point in camera frame -> convert to the shared world frame.

World frame (shared with trajectory/):  X right (m), Y forward depth (m), Z up (m).
"""
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


# ----------------------------------------------------------------- calibration
@dataclass
class StereoCalibration:
    """Rectified stereo rig. P1/P2 are the 3x4 rectified projection matrices."""
    image_size: Tuple[int, int]            # (width, height)
    P1: np.ndarray
    P2: np.ndarray
    map1: Optional[Tuple[np.ndarray, np.ndarray]] = None   # left rectify maps (None = already rectified)
    map2: Optional[Tuple[np.ndarray, np.ndarray]] = None

    @classmethod
    def ideal(cls, width=640, height=480, focal_px=600.0, baseline_m=0.12):
        """Already-rectified pinhole pair (used by tests / the simulator)."""
        cx, cy = width / 2.0, height / 2.0
        P1 = np.array([[focal_px, 0, cx, 0], [0, focal_px, cy, 0], [0, 0, 1, 0]], dtype=float)
        P2 = P1.copy()
        P2[0, 3] = -focal_px * baseline_m
        return cls((width, height), P1, P2)

    def save(self, path: str):
        d = dict(size=np.array(self.image_size), P1=self.P1, P2=self.P2)
        if self.map1 is not None:
            d.update(m1x=self.map1[0], m1y=self.map1[1], m2x=self.map2[0], m2y=self.map2[1])
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path: str):
        d = np.load(path)
        maps = ("m1x" in d.files)
        return cls(tuple(int(v) for v in d["size"]), d["P1"], d["P2"],
                   (d["m1x"], d["m1y"]) if maps else None,
                   (d["m2x"], d["m2y"]) if maps else None)

    @property
    def focal(self): return float(self.P1[0, 0])
    @property
    def baseline(self): return float(-self.P2[0, 3] / self.P2[0, 0])

    def rectify(self, left, right):
        if self.map1 is None:
            return left, right
        return (cv2.remap(left, *self.map1, cv2.INTER_LINEAR),
                cv2.remap(right, *self.map2, cv2.INTER_LINEAR))


@dataclass
class RigGeometry:
    """Where the camera pair sits in the world (measure with a tape + phone level)."""
    cam_height_m: float = 1.0      # lens height above the floor
    pitch_deg: float = 0.0         # downward tilt of the optical axis (positive = looking down)
    cam_x_m: float = 0.0           # lateral offset of the rig centre (world X)
    cam_y_m: float = 0.0           # forward offset (world Y) of the rig from the origin

    def cam_to_world(self, xc, yc, zc):
        th = np.radians(self.pitch_deg)
        X = xc + self.cam_x_m
        Y = -yc * np.sin(th) + zc * np.cos(th) + self.cam_y_m
        Z = self.cam_height_m - yc * np.cos(th) - zc * np.sin(th)
        return float(X), float(Y), float(Z)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


# -------------------------------------------------------------------- detector
@dataclass
class DetectorConfig:
    history: int = 300
    var_threshold: float = 28.0
    min_area: float = 60.0         # px; below is sensor noise
    max_area: float = 2500.0       # px; above is a person / large object (matches FilterConfig)
    max_row_diff_px: float = 6.0   # epipolar tolerance after rectification
    min_disparity_px: float = 2.0
    max_depth_m: float = 8.0
    open_kernel: int = 3
    dilate_iter: int = 2
    warmup_frames: int = 30        # background model learns first
    person_exclusion_px: int = 45  # ignore small blobs this close to a large (person-sized) blob


class BlobDetector:
    def __init__(self, cfg: Optional[DetectorConfig] = None):
        self.cfg = cfg or DetectorConfig()
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=self.cfg.history, varThreshold=self.cfg.var_threshold, detectShadows=False)
        k = self.cfg.open_kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.frames = 0

    def detect(self, frame_bgr, last_xy=None):
        """Returns (cx, cy, area) of the best ball-sized blob, or None."""
        c = self.cfg
        gray = cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3
                                else frame_bgr, (5, 5), 0)
        mask = self.bg.apply(gray)
        self.frames += 1
        if self.frames <= c.warmup_frames:
            return None
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=c.dilate_iter)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Person-sized blobs: not candidates themselves, and small blobs touching/near them
        # (a hand or arm, or a ball still in the hand) are not candidates either.
        person_zone = None
        if c.person_exclusion_px > 0:
            big = [k for k in cnts if cv2.contourArea(k) > c.max_area]
            if big:
                person_zone = np.zeros(mask.shape[:2], np.uint8)
                cv2.drawContours(person_zone, big, -1, 255, -1)
                e = 2 * c.person_exclusion_px + 1
                person_zone = cv2.dilate(person_zone, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e)))
        best, best_score = None, -1.0
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if not (c.min_area <= area <= c.max_area):
                continue
            per = cv2.arcLength(cnt, True)
            circ = 4 * np.pi * area / (per * per + 1e-6)          # 1.0 = perfect circle
            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            if person_zone is not None and person_zone[int(cy), int(cx)]:
                continue
            score = circ
            if last_xy is not None:     # prefer continuity with the previous detection
                score += 0.5 / (1.0 + np.hypot(cx - last_xy[0], cy - last_xy[1]) / 40.0)
            if score > best_score:
                best, best_score = (cx, cy, area), score
        return best


# --------------------------------------------------------------------- tracker
class StereoTracker:
    """Frame pair in -> packet dict out (the format trajectory/ expects)."""

    def __init__(self, calib: StereoCalibration, rig: Optional[RigGeometry] = None,
                 det_cfg: Optional[DetectorConfig] = None):
        self.calib = calib
        self.rig = rig or RigGeometry()
        self.cfg = det_cfg or DetectorConfig()
        self.det_l = BlobDetector(self.cfg)
        self.det_r = BlobDetector(self.cfg)
        self._last_l = None
        self._prev = {"L": None, "R": None}     # previous detection (t, u, v) per camera

    def triangulate(self, ul, vl, ur, vr) -> Optional[Tuple[float, float, float]]:
        c = self.cfg
        if abs(vl - vr) > c.max_row_diff_px:
            return None
        d = ul - ur
        if d < c.min_disparity_px:
            return None
        pts = cv2.triangulatePoints(self.calib.P1, self.calib.P2,
                                    np.array([[ul], [vl]], dtype=float),
                                    np.array([[ur], [vr]], dtype=float))
        xc, yc, zc = (pts[:3, 0] / pts[3, 0]).tolist()
        if not (0.2 < zc < c.max_depth_m):
            return None
        return xc, yc, zc

    def _align(self, key, t_cam, u, v, t_ref, max_dt=0.05):
        """Move a detection from its own capture time to t_ref using that camera's own motion.

        Two phones are never exactly in sync. A ball moving sideways at 1.5 m/s with the
        cameras only 8 ms out of step gives ~25 cm depth error (12 cm baseline); moving each
        detection to the common time first brings that down to ~2 cm.
        """
        prev = self._prev[key]
        self._prev[key] = (t_cam, u, v)
        if prev is None or abs(t_ref - t_cam) < 1e-4:
            return u, v
        dt = t_cam - prev[0]
        if not (1e-4 < dt <= max_dt):
            return u, v
        k = (t_ref - t_cam) / dt
        return u + (u - prev[1]) * k, v + (v - prev[2]) * k

    def process(self, left, right, t: Optional[float] = None,
                t_left: Optional[float] = None, t_right: Optional[float] = None) -> dict:
        """t: reference time of the pair. t_left/t_right: each camera's own capture time (same
        clock, offset applied; `StereoCameras.last_times`). If they differ from t, each detection
        is moved to t before triangulation."""
        t = time.monotonic() if t is None else t
        t_left = t if t_left is None else t_left
        t_right = t if t_right is None else t_right
        left, right = self.calib.rectify(left, right)
        dl = self.det_l.detect(left, self._last_l)
        dr = self.det_r.detect(right)
        pkt = {"timestamp": t, "detected": False, "X": 0.0, "Y": 0.0, "Z": 0.0, "blob_area": 0.0}
        if dl is None:
            self._prev["L"] = None
        if dr is None:
            self._prev["R"] = None
        if dl is None or dr is None:
            self._last_l = None
            return pkt
        ul, vl = self._align("L", t_left, dl[0], dl[1], t)
        ur, vr = self._align("R", t_right, dr[0], dr[1], t)
        p = self.triangulate(ul, vl, ur, vr)
        if p is None:
            self._last_l = None
            return pkt
        self._last_l = (dl[0], dl[1])
        X, Y, Z = self.rig.cam_to_world(*p)
        pkt.update(detected=True, X=X, Y=Y, Z=Z, blob_area=float(0.5 * (dl[2] + dr[2])))
        return pkt
