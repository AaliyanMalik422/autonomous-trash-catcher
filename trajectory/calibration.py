"""Throw-to-landing calibration model built from logged test throws.

Workflow (matches the proposal's "calibration model from logged test throws"):
  1. Run test throws; ThrowLogger appends the system's final prediction per throw.
  2. After each throw, measure where it REALLY landed (tape measure on the floor,
     in the same X/Y frame, metres) and write it into the CSV (true_x, true_y).
  3. `python -m trajectory.calibration fit logs/throws.csv` fits an affine
     correction  true ~= A @ pred + b  (ridge-regularised towards the identity so
     a handful of throws cannot produce a wild correction) and saves it as JSON.
  4. TrajectoryEngine loads that JSON and applies it to every prediction.
"""
import argparse
import csv
import json
import os
from typing import Optional, Tuple

import numpy as np

CSV_FIELDS = ["throw_id", "pred_x", "pred_y", "ttl_at_first_send", "sigma", "n_points",
              "vx", "vy", "vz", "x0", "y0", "z0", "true_x", "true_y"]


class LandingCorrection:
    def __init__(self, A=None, b=None):
        self.A = np.eye(2) if A is None else np.asarray(A, dtype=float).reshape(2, 2)
        self.b = np.zeros(2) if b is None else np.asarray(b, dtype=float).reshape(2)

    @property
    def is_identity(self) -> bool:
        return np.allclose(self.A, np.eye(2)) and np.allclose(self.b, 0)

    def apply(self, x: float, y: float) -> Tuple[float, float]:
        v = self.A @ np.array([x, y]) + self.b
        return float(v[0]), float(v[1])

    @classmethod
    def fit(cls, pred_xy, true_xy, ridge: float = 3.0) -> "LandingCorrection":
        """pred_xy, true_xy: (N,2). Ridge pulls A->I and b->0 (strength in 'samples')."""
        P = np.asarray(pred_xy, dtype=float).reshape(-1, 2)
        T = np.asarray(true_xy, dtype=float).reshape(-1, 2)
        if len(P) == 0:
            return cls()
        X = np.column_stack([P, np.ones(len(P))])          # [x y 1]
        W0 = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # identity prior (3x2)
        # normalise by position scale so ridge treats metres sensibly
        reg = ridge * np.diag([0.05, 0.05, 1.0])  # slope rows barely regularised vs offset
        W = np.linalg.solve(X.T @ X + reg, X.T @ T + reg @ W0)
        return cls(A=W[:2, :].T, b=W[2, :])

    def to_json(self) -> dict:
        return {"A": self.A.tolist(), "b": self.b.tolist()}

    @classmethod
    def from_json(cls, d: dict) -> "LandingCorrection":
        return cls(d["A"], d["b"])

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LandingCorrection":
        with open(path) as f:
            return cls.from_json(json.load(f))


class ThrowLogger:
    """Appends one row per throw to a CSV (true_x/true_y filled in by hand later)."""

    def __init__(self, path: str = "logs/throws.csv"):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.count = 0
        if os.path.exists(path):
            with open(path, newline="") as f:
                self.count = max(0, sum(1 for _ in f) - 1)
        else:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(CSV_FIELDS)

    def log(self, pred, ttl_at_first_send: float = float("nan")) -> int:
        self.count += 1
        row = [self.count, f"{pred.x:.4f}", f"{pred.y:.4f}", f"{ttl_at_first_send:.3f}",
               f"{pred.sigma:.4f}", pred.n_used, f"{pred.vx:.3f}", f"{pred.vy:.3f}",
               f"{pred.vz:.3f}", f"{pred.x0:.3f}", f"{pred.y0:.3f}", f"{pred.z0:.3f}", "", ""]
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        return self.count


def load_pairs(csv_path: str):
    P, T = [], []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("true_x") not in (None, "") and r.get("true_y") not in (None, ""):
                P.append((float(r["pred_x"]), float(r["pred_y"])))
                T.append((float(r["true_x"]), float(r["true_y"])))
    return np.array(P), np.array(T)


def _main():
    ap = argparse.ArgumentParser(description="Fit landing correction from logged throws")
    ap.add_argument("cmd", choices=["fit"])
    ap.add_argument("csv")
    ap.add_argument("--out", default="calibration.json")
    ap.add_argument("--ridge", type=float, default=3.0)
    a = ap.parse_args()
    P, T = load_pairs(a.csv)
    if len(P) < 3:
        raise SystemExit(f"Need >=3 throws with true_x/true_y filled in, found {len(P)}")
    corr = LandingCorrection.fit(P, T, a.ridge)
    before = np.hypot(*(P - T).T)
    after = np.array([np.hypot(*(np.array(corr.apply(*p)) - t)) for p, t in zip(P, T)])
    print(f"{len(P)} throws | mean error before {before.mean()*100:.1f} cm -> after {after.mean()*100:.1f} cm")
    corr.save(a.out)
    print("saved", a.out, corr.to_json())


if __name__ == "__main__":
    _main()
