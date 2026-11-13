"""Closed-loop simulation: mock throws -> TrajectoryEngine -> accel-limited robot.

Answers the key feasibility question: *given the robot's real acceleration and the
camera frame rate, what fraction of throws can actually be caught?*

    python -m trajectory.sim --n 300 --fps 30 --accel 3.6 --speed 0.9 --reach 0.5
"""
import argparse
from dataclasses import dataclass

import numpy as np

from .config import Config
from .mock_vision import MockStereoCamera
from .pipeline import TrajectoryEngine


class SimRobot:
    """Point mass in the robot frame, vector-accel and speed limited (like the firmware)."""

    def __init__(self, max_accel, max_speed, cmd_latency=0.04):
        self.a, self.vmax, self.lat = max_accel, max_speed, cmd_latency
        self.p = np.zeros(2)
        self.v = np.zeros(2)
        self.target = np.zeros(2)
        self.pending = []   # (apply_time, target)

    def command(self, t, target):
        self.pending.append((t + self.lat, np.array(target, dtype=float)))

    def step(self, t, dt):
        while self.pending and self.pending[0][0] <= t:
            self.target = self.pending.pop(0)[1]
        # P-D style approach: desired speed limited by braking distance
        err = self.target - self.p
        d = np.linalg.norm(err)
        if d < 1e-6:
            v_des = np.zeros(2)
        else:
            v_mag = min(self.vmax, np.sqrt(2 * self.a * d))
            v_des = err / d * v_mag
        dv = v_des - self.v
        n = np.linalg.norm(dv)
        step = self.a * dt
        self.v = v_des if n <= step else self.v + dv / n * step
        self.p = self.p + self.v * dt


@dataclass
class SimResult:
    n: int
    detected: int
    caught: int
    mean_pred_err: float
    median_pred_err: float
    mean_miss: float


def run_sim(n=300, fps=30, accel=3.6, speed=0.9, reach=0.5, basket_radius=0.15,
            noise=0.025, latency=0.08, seed=0, cfg=None, verbose=False):
    cfg = cfg or Config()
    cfg.engine.max_accel, cfg.engine.max_speed = accel, speed
    cfg.engine.latency_s = latency
    cfg.zone.reach_x = cfg.zone.reach_y = max(reach, 0.3)
    home = (cfg.engine.home_x, cfg.engine.home_y)
    caught = detected = 0
    pred_err, miss = [], []
    dt = 1.0 / 1000
    for i in range(n):
        cam = MockStereoCamera(fps=fps, noise_std_meters=noise, seed=seed * 100000 + i)
        pk, tl = cam.random_throw(home=home, reach=reach)
        eng = TrajectoryEngine(Config(**{k: getattr(cfg, k) for k in ("zone", "filt", "pred", "engine")}))
        robot = SimRobot(accel, speed)
        true_xy = np.array([cfg.engine.sign_x * (tl["X_land"] - home[0]),
                            cfg.engine.sign_y * (tl["Y_land"] - home[1])])
        t = 0.0
        k = 0
        got = False
        while t <= tl["t_land"] + 1e-9:
            while k < len(pk) and pk[k]["timestamp"] + latency <= t:  # frame arrives after latency
                c = eng.update(pk[k])
                k += 1
                if c is not None:
                    if c.kind == "target":
                        robot.command(t, (c.x, c.y))
                        got = True
                    elif c.kind == "home":
                        robot.command(t, (0, 0))
            robot.step(t, dt)
            t += dt
        detected += got
        if eng.last_prediction is not None:
            pr = eng.last_prediction
            cx, cy = eng.correction.apply(pr.x, pr.y)
            px = cfg.engine.sign_x * (cx - home[0])
            py = cfg.engine.sign_y * (cy - home[1])
            pred_err.append(float(np.hypot(px - true_xy[0], py - true_xy[1])))
        m = float(np.linalg.norm(robot.p - true_xy))
        miss.append(m)
        caught += m <= basket_radius
    return SimResult(n, detected, caught,
                     float(np.mean(pred_err)) if pred_err else float("nan"),
                     float(np.median(pred_err)) if pred_err else float("nan"),
                     float(np.mean(miss)))


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--accel", type=float, default=3.6)
    ap.add_argument("--speed", type=float, default=0.9)
    ap.add_argument("--reach", type=float, default=0.5, help="half-size of landing zone around home (m)")
    ap.add_argument("--basket", type=float, default=0.15, help="catch radius (m)")
    ap.add_argument("--noise", type=float, default=0.025)
    ap.add_argument("--latency", type=float, default=0.08)
    a = ap.parse_args()
    r = run_sim(a.n, a.fps, a.accel, a.speed, a.reach, a.basket, a.noise, a.latency)
    print(f"throws {r.n} | acted on {r.detected} | caught {r.caught} ({100*r.caught/r.n:.0f}%) | "
          f"pred err mean {r.mean_pred_err*100:.1f} cm median {r.median_pred_err*100:.1f} cm | "
          f"mean miss {r.mean_miss*100:.1f} cm")


if __name__ == "__main__":
    _main()
