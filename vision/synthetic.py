"""Render synthetic stereo frames of a thrown ball (for tests and `main.py --sim-vision`).

Lets the *whole* vision path (background subtraction -> blob -> stereo -> 3D) be tested
against known ground truth without cameras.
"""
from typing import Optional, Tuple

import cv2
import numpy as np

from .stereo_tracker import RigGeometry, StereoCalibration


class SyntheticRig:
    def __init__(self, calib: Optional[StereoCalibration] = None, rig: Optional[RigGeometry] = None,
                 ball_radius_m: float = 0.035, noise: float = 2.0, seed: int = 0):
        self.calib = calib or StereoCalibration.ideal()
        self.rig = rig or RigGeometry(cam_height_m=1.0, pitch_deg=0.0)
        self.r = ball_radius_m
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        w, h = self.calib.image_size
        base = self.rng.integers(60, 140, (h, w), dtype=np.uint8)
        self.bg = cv2.GaussianBlur(base, (0, 0), 6)           # static textured background

    def world_to_cam(self, X, Y, Z):
        th = np.radians(self.rig.pitch_deg)
        x = X - self.rig.cam_x_m
        y_f = Y - self.rig.cam_y_m
        z_up = Z - self.rig.cam_height_m
        # inverse of RigGeometry.cam_to_world
        yc = -y_f * np.sin(th) - z_up * np.cos(th)
        zc = y_f * np.cos(th) - z_up * np.sin(th)
        return x, yc, zc

    def _project(self, P, xc, yc, zc):
        u = (P[0, 0] * xc + P[0, 2] * zc + P[0, 3]) / zc
        v = (P[1, 1] * yc + P[1, 2] * zc) / zc
        return u, v

    def render(self, point_world: Optional[Tuple[float, float, float]]):
        w, h = self.calib.image_size
        out = []
        for P in (self.calib.P1, self.calib.P2):
            img = self.bg.astype(np.float32).copy()
            if point_world is not None:
                xc, yc, zc = self.world_to_cam(*point_world)
                if zc > 0.2:
                    u, v = self._project(P, xc, yc, zc)
                    rad = max(self.calib.focal * self.r / zc, 1.5)
                    cv2.circle(img, (int(round(u * 16)), int(round(v * 16))),
                               int(round(rad * 16)), 255.0, -1, cv2.LINE_AA, 4)
            img += self.rng.normal(0, self.noise, img.shape).astype(np.float32)
            out.append(cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        return out[0], out[1]
