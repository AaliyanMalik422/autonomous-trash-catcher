"""Distinguish a real throw from walking people, hand waves and sensor noise.

A track is confirmed as a throw only when, after >= min_points valid detections:
  * blob size and height are plausible (not a person, not on the floor),
  * the motion is approaching the robot and within a plausible speed range,
  * X/Y are well described by straight lines and Z by a *gravity-constrained*
    parabola (free flight) -- walkers, waving hands and bouncing things fail this.
"""
from typing import List, Optional, Tuple

import numpy as np

from .config import FilterConfig, PredictorConfig, ZoneConfig


class ThrowFilter:
    def __init__(self, config: Optional[FilterConfig] = None,
                 zone: Optional[ZoneConfig] = None, g: Optional[float] = None):
        self.cfg = config or FilterConfig()
        self.zone = zone or ZoneConfig()
        self.g = g if g is not None else PredictorConfig().g
        self.trajectory_buffer: List[dict] = []
        self.last_reason = ""
        self._last_t: Optional[float] = None
        self._confirmed = False

    # ----------------------------------------------------------------- helpers
    def reset(self):
        self.trajectory_buffer = []
        self._last_t = None
        self._confirmed = False
        self.last_reason = "reset"

    def _in_zone(self, p) -> bool:
        z = self.zone
        return z.x_min <= p["X"] <= z.x_max and z.y_min <= p["Y"] <= z.y_max

    def _reject(self, reason):
        self.last_reason = reason
        return False, self.trajectory_buffer

    def _check_gap(self, t):
        if self._last_t is not None and t - self._last_t > self.cfg.max_gap_s:
            self.trajectory_buffer = []
            self._last_t = None
            self._confirmed = False

    # -------------------------------------------------------------- main entry
    def process_frame(self, pkt: dict) -> Tuple[bool, list]:
        """Returns (is_valid_throw, buffer_of_accepted_points)."""
        c = self.cfg
        t = pkt.get("timestamp", 0.0)
        self._check_gap(t)

        if not pkt.get("detected", False):
            return self._reject("not detected")

        area = pkt.get("blob_area", 0.0)
        if not (c.min_area <= area <= c.max_area):
            return self._reject("blob size")
        if pkt["Z"] < c.min_height_z:
            return self._reject("too low")
        if not self._in_zone(pkt):
            return self._reject("outside zone")

        buf = self.trajectory_buffer
        if buf:
            last = buf[-1]
            dt = t - last["timestamp"]
            if dt <= 0:
                return self._reject("non-monotonic time")
            d = np.sqrt((pkt["X"] - last["X"]) ** 2 + (pkt["Y"] - last["Y"]) ** 2
                        + (pkt["Z"] - last["Z"]) ** 2)
            if d > c.max_jump_m + c.max_speed * dt:
                return self._reject("jump outlier")

        buf.append(pkt)
        self._last_t = t
        if len(buf) > 80:
            del buf[0]

        if self._confirmed:          # latched: keep tracking until the track is lost
            # ...but re-check free-fall curvature on the longer baseline (cancels slow
            # non-throws such as a hand wave that initially looked like a parabola)
            if self._curvature_bad(buf):
                self._confirmed = False
                return self._reject("reconsidered: not free fall")
            return True, buf
        if len(buf) < c.min_points:
            return self._reject(f"collecting ({len(buf)}/{c.min_points})")
        return self._validate(buf)

    # -------------------------------------------------------------- validation
    def _curvature_bad(self, buf) -> bool:
        c = self.cfg
        if len(buf) < 8:
            return False
        t = np.array([p["timestamp"] for p in buf])
        t = t - t[-1]
        if t[-1] - t[0] < c.reconsider_min_duration_s:
            return False
        z = np.array([p["Z"] for p in buf])
        acc = 2.0 * np.polyfit(t, z, 2)[0]
        return abs(acc + self.g) > c.curvature_tol

    def _validate(self, buf):
        c = self.cfg
        g = self.g
        t = np.array([p["timestamp"] for p in buf])
        t = t - t[-1]
        x = np.array([p["X"] for p in buf])
        y = np.array([p["Y"] for p in buf])
        z = np.array([p["Z"] for p in buf])
        if t[-1] - t[0] < c.min_duration_s:
            return self._reject("too short")

        A = np.column_stack([np.ones_like(t), t])
        cx, *_ = np.linalg.lstsq(A, x, rcond=None)
        cy, *_ = np.linalg.lstsq(A, y, rcond=None)
        z_adj = z + 0.5 * g * t * t
        cz, *_ = np.linalg.lstsq(A, z_adj, rcond=None)

        rx = x - A @ cx
        ry = y - A @ cy
        rz = z_adj - A @ cz
        xy_rms = float(np.sqrt(np.mean(np.concatenate([rx, ry]) ** 2)))
        z_rms = float(np.sqrt(np.mean(rz ** 2)))
        vx, vy, vz = cx[1], cy[1], cz[1]
        speed = float(np.sqrt(vx * vx + vy * vy + vz * vz))

        if xy_rms > c.max_xy_rms:
            return self._reject(f"not straight horizontally ({xy_rms:.3f})")
        if z_rms > c.max_z_rms:
            return self._reject(f"not free flight ({z_rms:.3f})")
        if not (c.min_speed <= speed <= c.max_speed):
            return self._reject(f"speed {speed:.2f}")
        if c.approach_sign * vy < c.min_approach_speed:
            return self._reject("not approaching")

        # Free-flight curvature check once there is enough baseline: an unconstrained
        # quadratic fit of Z must show acceleration close to -g (hand waves, bouncing
        # or oscillating things do not).
        if (t[-1] - t[0]) >= c.curvature_min_duration_s and len(t) >= 6:
            q = np.polyfit(t, z, 2)
            acc = 2.0 * q[0]
            if abs(acc + g) > c.curvature_tol:
                return self._reject(f"vertical accel {acc:.1f} != -g")

        self._confirmed = True
        self.last_reason = "throw"
        return True, buf


if __name__ == "__main__":
    ThrowFilter()
    print("ThrowFilter initialized successfully.")
