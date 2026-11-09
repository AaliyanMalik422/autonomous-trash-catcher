"""Projectile landing prediction from noisy 3D points.

Model (gravity known, no drag):
    X(t) = x0 + vx*t              (linear)
    Y(t) = y0 + vy*t              (linear)
    Z(t) = z0 + vz*t - g*t^2/2    (gravity-constrained -> fit is linear in vz, z0)
with t measured relative to the *last* observed point, so (x0, y0, z0) is the
current position and extrapolation to impact is well conditioned.

Improvements over a plain least-squares fit:
  * robust outlier rejection (MAD based) before the final fit,
  * optional recency weighting,
  * analytic 1-sigma uncertainty of the landing point (including time error),
  * picks the *descending* crossing of the basket height (the old code could
    pick the ascending crossing for balls thrown from below rim height).
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import PredictorConfig

NOISE_FLOOR_M = 0.012   # never trust a residual estimate smaller than this (few points)


@dataclass
class Prediction:
    x: float
    y: float
    z: float                 # basket height
    t_ref: float             # timestamp of last point used
    ttl: float               # seconds from t_ref until impact
    t_impact: float          # absolute impact time (t_ref + ttl)
    sigma_x: float
    sigma_y: float
    sigma: float             # combined 1-sigma landing error (m)
    n_used: int
    n_total: int
    rms: float               # rms residual of the fit (m)
    vx: float
    vy: float
    vz: float
    x0: float
    y0: float
    z0: float
    g: float

    def curve(self, n: int = 50, t_back: float = 0.0):
        """Fitted path from `t_back` seconds before t_ref to impact (for plotting)."""
        t = np.linspace(-t_back, self.ttl, n)
        return (self.x0 + self.vx * t,
                self.y0 + self.vy * t,
                self.z0 + self.vz * t - 0.5 * self.g * t * t)


def _wls(t: np.ndarray, v: np.ndarray, w: np.ndarray):
    """Weighted LS for v = a + b*t. Returns (a, b, cov_unit) with cov_unit=(A'WA)^-1."""
    A = np.column_stack([np.ones_like(t), t])
    Aw = A * w[:, None]
    M = A.T @ Aw
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return None
    coef = Minv @ (Aw.T @ v)
    return float(coef[0]), float(coef[1]), Minv


class TrajectoryPredictor:
    def __init__(self, target_z: Optional[float] = None, g: Optional[float] = None,
                 config: Optional[PredictorConfig] = None):
        cfg = config or PredictorConfig()
        if target_z is not None:
            cfg.target_z = target_z
        if g is not None:
            cfg.g = g
        self.cfg = cfg
        self.target_z = cfg.target_z
        self.g = cfg.g

    # ------------------------------------------------------------------ core
    def _fit(self, t, x, y, z, w):
        g = self.g
        fx = _wls(t, x, w)
        fy = _wls(t, y, w)
        fz = _wls(t, z + 0.5 * g * t * t, w)
        if fx is None or fy is None or fz is None:
            return None
        return fx, fy, fz

    def _weights(self, t):
        if self.cfg.use_weights and len(t) > 1:
            return 1.0 + 0.5 * (t - t[0]) / max(t[-1] - t[0], 1e-6)   # 1.0 .. 1.5
        return np.ones_like(t)

    @staticmethod
    def _residuals(t, v, a, b):
        return v - (a + b * t)

    def predict(self, points: Sequence[dict]) -> Optional[Prediction]:
        cfg = self.cfg
        if len(points) < max(3, cfg.min_points):
            return None
        g = self.g
        t_abs = np.array([p["timestamp"] for p in points], dtype=float)
        x = np.array([p["X"] for p in points], dtype=float)
        y = np.array([p["Y"] for p in points], dtype=float)
        z = np.array([p["Z"] for p in points], dtype=float)
        order = np.argsort(t_abs)
        t_abs, x, y, z = t_abs[order], x[order], y[order], z[order]
        n_total = len(t_abs)

        keep = np.ones(n_total, dtype=bool)
        # Iteratively drop the single worst point while it is a clear outlier.
        for _ in range(max(1, n_total // 4)):
            if keep.sum() <= cfg.min_points:
                break
            tk = t_abs[keep]
            t_ref = tk[-1]
            t = tk - t_ref
            w = self._weights(t)
            fit = self._fit(t, x[keep], y[keep], z[keep], w)
            if fit is None:
                return None
            (x0, vx, _), (y0, vy, _), (z0, vz, _) = fit
            t_all = t_abs - t_ref
            rs = [self._residuals(t_all, x, x0, vx),
                  self._residuals(t_all, y, y0, vy),
                  self._residuals(t_all, z + 0.5 * g * t_all ** 2, z0, vz)]
            rn = np.zeros(n_total)
            for r in rs:
                med = np.median(r[keep])
                mad = 1.4826 * np.median(np.abs(r[keep] - med))
                rn = np.maximum(rn, np.abs(r - med) / max(mad, NOISE_FLOOR_M))
            rn[~keep] = -1.0
            worst = int(np.argmax(rn))
            if rn[worst] <= cfg.outlier_sigma:
                break
            keep[worst] = False

        tk = t_abs[keep]
        t_ref = tk[-1]
        t = tk - t_ref
        w = self._weights(t)
        fit = self._fit(t, x[keep], y[keep], z[keep], w)
        if fit is None:
            return None
        (x0, vx, covx), (y0, vy, covy), (z0, vz, covz) = fit

        # impact time: z0 + vz*t - g t^2/2 = target  -> take the descending root
        c = z0 - self.target_z
        disc = vz * vz + 2.0 * g * c
        if disc < 0:
            return None
        tau = (vz + np.sqrt(disc)) / g
        if tau <= 0:
            return None  # already below the basket height and falling

        # residual noise estimate (dof-corrected, floored)
        n_used = int(keep.sum())
        res = np.concatenate([
            self._residuals(t, x[keep], x0, vx),
            self._residuals(t, y[keep], y0, vy),
            self._residuals(t, z[keep] + 0.5 * g * t * t, z0, vz),
        ])
        dof = max(3 * n_used - 6, 1)
        sigma0 = max(float(np.sqrt(np.sum(res ** 2) / dof)), NOISE_FLOOR_M)
        rms = float(np.sqrt(np.mean(res ** 2)))

        gvec = np.array([1.0, tau])
        var_x = sigma0 ** 2 * float(gvec @ covx @ gvec)
        var_y = sigma0 ** 2 * float(gvec @ covy @ gvec)
        var_z = sigma0 ** 2 * float(gvec @ covz @ gvec)
        slope_z = max(abs(vz - g * tau), 1.0)
        var_t = var_z / slope_z ** 2
        sx = float(np.sqrt(var_x + vx * vx * var_t))
        sy = float(np.sqrt(var_y + vy * vy * var_t))

        px = x0 + vx * tau
        py = y0 + vy * tau
        return Prediction(
            x=float(px), y=float(py), z=self.target_z,
            t_ref=float(t_ref), ttl=float(tau), t_impact=float(t_ref + tau),
            sigma_x=sx, sigma_y=sy, sigma=float(np.hypot(sx, sy)),
            n_used=n_used, n_total=n_total, rms=rms,
            vx=float(vx), vy=float(vy), vz=float(vz),
            x0=float(x0), y0=float(y0), z0=float(z0), g=g,
        )

    # --------------------------------------------------------- legacy wrapper
    def predict_landing(self, points) -> Optional[Tuple[float, float, float, dict]]:
        """Backwards-compatible API: (pred_x, pred_y, time_to_impact, fit_data)."""
        p = self.predict(points)
        if p is None:
            return None
        cx, cy, cz = p.curve(50, t_back=p.t_ref - points[0]["timestamp"])
        return p.x, p.y, p.ttl, {
            "curve_x": cx, "curve_y": cy, "curve_z": cz,
            "pred_x": p.x, "pred_y": p.y, "pred_z": p.z,
            "time_to_impact": p.ttl, "sigma": p.sigma,
        }


if __name__ == "__main__":
    print("TrajectoryPredictor loaded successfully.")
