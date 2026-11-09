"""TrajectoryEngine: packets in (from vision) -> robot commands out (to the ESP32).

    packet = {"timestamp", "detected", "X", "Y", "Z", "blob_area"}   (see README)

States:  IDLE -> TRACKING -> ACTIVE -> COOLDOWN -> IDLE
  IDLE/TRACKING : filter collects points, predictor refines the landing estimate.
  ACTIVE        : a target was sent; keep refining while there is still time,
                  hold position until shortly after the predicted impact, then go home.
  COOLDOWN      : ignore input briefly so the thrown object / catch does not retrigger.
"""
from dataclasses import dataclass
from typing import Optional

from .calibration import LandingCorrection, ThrowLogger
from .config import Config
from .throw_filter import ThrowFilter
from .trajectory_predictor import Prediction, TrajectoryPredictor

IDLE, TRACKING, ACTIVE, COOLDOWN = "IDLE", "TRACKING", "ACTIVE", "COOLDOWN"


@dataclass
class Command:
    kind: str                       # "target" | "home" | "stop"
    x: float = 0.0                  # robot frame (m): +x strafe right, +y forward, from home
    y: float = 0.0
    ttl: float = 0.0                # estimated seconds until the object reaches basket height
    sigma: float = 0.0              # 1-sigma landing uncertainty (m)
    reachable: bool = True          # False: robot is predicted unable to get there in time
    clamped: bool = False           # True: target was clipped to the robot's reach area
    prediction: Optional[Prediction] = None


def reach_distance(ttl: float, max_accel: float, max_speed: float) -> float:
    """Farthest the robot can travel from rest in `ttl` seconds."""
    if ttl <= 0:
        return 0.0
    t_acc = max_speed / max_accel
    if ttl <= t_acc:
        return 0.5 * max_accel * ttl * ttl
    return 0.5 * max_accel * t_acc ** 2 + max_speed * (ttl - t_acc)


class TrajectoryEngine:
    def __init__(self, config: Optional[Config] = None,
                 correction: Optional[LandingCorrection] = None,
                 logger: Optional[ThrowLogger] = None):
        self.cfg = config or Config()
        self.filter = ThrowFilter(self.cfg.filt, self.cfg.zone, self.cfg.pred.g)
        self.predictor = TrajectoryPredictor(config=self.cfg.pred)
        self.correction = correction or LandingCorrection()
        self.logger = logger
        self.state = IDLE
        self.last_prediction: Optional[Prediction] = None
        self.last_command: Optional[Command] = None
        self._t_start = 0.0
        self._t_impact = 0.0
        self._t_cooldown_end = 0.0
        self._sent_xy = None
        self._logged = False

    # ---------------------------------------------------------------- helpers
    def _to_robot_frame(self, cx, cy):
        e = self.cfg.engine
        rx = e.sign_x * (cx - e.home_x)
        ry = e.sign_y * (cy - e.home_y)
        z = self.cfg.zone
        crx = max(-z.reach_x, min(z.reach_x, rx))
        cry = max(-z.reach_y, min(z.reach_y, ry))
        return crx, cry, (crx != rx or cry != ry)

    def _make_command(self, pred: Prediction, ttl_eff: float) -> Command:
        e = self.cfg.engine
        cx, cy = self.correction.apply(pred.x, pred.y)
        rx, ry, clamped = self._to_robot_frame(cx, cy)
        dist = (rx * rx + ry * ry) ** 0.5
        reachable = (not clamped) and dist <= reach_distance(ttl_eff, e.max_accel, e.max_speed)
        return Command("target", rx, ry, ttl_eff, pred.sigma, reachable, clamped, pred)

    def _finish(self, t) -> Optional[Command]:
        self.state = COOLDOWN
        self._t_cooldown_end = t + self.cfg.engine.cooldown_s
        self.filter.reset()
        self._sent_xy = None
        return Command("home") if self.cfg.engine.return_home else None

    # ------------------------------------------------------------------- API
    def tick(self, t: float) -> Optional[Command]:
        """Time-based transitions; call regularly even when no packets arrive."""
        if self.state == ACTIVE and t > self._t_impact + self.cfg.engine.hold_after_impact_s:
            return self._finish(t)
        if self.state == COOLDOWN and t >= self._t_cooldown_end:
            self.state = IDLE
            self.filter.reset()
            self._logged = False
        return None

    def update(self, pkt: dict) -> Optional[Command]:
        t = pkt.get("timestamp", 0.0)
        cmd = self.tick(t)
        if cmd is not None:
            return cmd
        if self.state == COOLDOWN:
            return None

        e = self.cfg.engine
        valid, buf = self.filter.process_frame(pkt)

        if self.state == ACTIVE:
            if self.filter.last_reason.startswith("reconsidered"):
                # turned out not to be a throw after all -> abort and go home
                return self._finish(t)
            if not valid:
                return None
            return self._refine(buf, t)

        # IDLE / TRACKING
        if not valid:
            self.state = TRACKING if buf else IDLE
            return None
        if self.state == IDLE:
            self.state = TRACKING
            self._t_start = buf[0]["timestamp"]
        pred = self.predictor.predict(buf)
        if pred is None:
            return None
        self.last_prediction = pred
        ttl_eff = pred.t_impact - t - e.latency_s
        if ttl_eff < e.min_ttl_s:
            # too late to do anything useful; wait for the next throw
            self.state = COOLDOWN
            self._t_cooldown_end = t + e.cooldown_s
            self.filter.reset()
            return None
        if pred.sigma > e.send_sigma_m and (t - self._t_start) < e.max_wait_s:
            return None
        return self._activate(pred, ttl_eff, t)

    def _activate(self, pred, ttl_eff, t) -> Command:
        cmd = self._make_command(pred, ttl_eff)
        self.state = ACTIVE
        self._t_impact = pred.t_impact
        self._sent_xy = (cmd.x, cmd.y)
        self.last_command = cmd
        if self.logger and not self._logged:
            self.logger.log(pred, ttl_eff)
            self._logged = True
        return cmd

    def _refine(self, buf, t) -> Optional[Command]:
        e = self.cfg.engine
        pred = self.predictor.predict(buf)
        if pred is None:
            return None
        self.last_prediction = pred
        self._t_impact = pred.t_impact
        ttl_eff = pred.t_impact - t - e.latency_s
        if ttl_eff < e.min_ttl_s:
            return None
        cmd = self._make_command(pred, ttl_eff)
        if self._sent_xy is not None:
            dx = cmd.x - self._sent_xy[0]
            dy = cmd.y - self._sent_xy[1]
            if (dx * dx + dy * dy) ** 0.5 < e.min_update_delta_m:
                return None
        self._sent_xy = (cmd.x, cmd.y)
        self.last_command = cmd
        return cmd
