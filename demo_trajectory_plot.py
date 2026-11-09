"""Standalone demo of the trajectory engine on synthetic throws (no hardware).

    python demo_trajectory_plot.py            # interactive window
    python demo_trajectory_plot.py --save out.png
"""
import argparse

import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument("--save", help="save figure to this PNG instead of opening a window")
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--fps", type=int, default=30)
args = parser.parse_args()
if args.save:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trajectory.mock_vision import MockStereoCamera
from trajectory.pipeline import TrajectoryEngine

cam = MockStereoCamera(fps=args.fps, seed=args.seed)
pk, tl = cam.random_throw()
eng = TrajectoryEngine()
first_cmd = None
for p in pk:
    c = eng.update(p)
    if c and c.kind == "target" and first_cmd is None:
        first_cmd = (p["timestamp"], c)

pred = eng.last_prediction
fig = plt.figure(figsize=(11, 5))
ax = fig.add_subplot(121, projection="3d")
ax.plot([p["true_X"] for p in pk], [p["true_Y"] for p in pk], [p["true_Z"] for p in pk], "g-", label="true path")
ax.scatter([p["X"] for p in pk], [p["Y"] for p in pk], [p["Z"] for p in pk], s=10, c="gray", label="noisy camera points")
if pred is not None:
    cx, cy, cz = pred.curve(60, t_back=0.4)
    ax.plot(cx, cy, cz, "r--", label="fitted path")
    ax.scatter([pred.x], [pred.y], [pred.z], c="r", s=60, marker="x", label="predicted landing")
ax.scatter([tl["X_land"]], [tl["Y_land"]], [tl["Z_land"]], c="g", s=60, marker="o", label="true landing")
ax.set_xlabel("X (m)"); ax.set_ylabel("Y depth (m)"); ax.set_zlabel("Z height (m)")
ax.legend(fontsize=7)

ax2 = fig.add_subplot(122)
ax2.set_title("Top view (X vs Y)")
ax2.plot([p["true_X"] for p in pk], [p["true_Y"] for p in pk], "g-")
ax2.scatter([p["X"] for p in pk], [p["Y"] for p in pk], s=10, c="gray")
e = eng.cfg.engine
ax2.scatter([e.home_x], [e.home_y], c="b", marker="s", s=80, label="robot home")
if pred is not None:
    ax2.scatter([pred.x], [pred.y], c="r", marker="x", s=80, label="predicted")
    ax2.add_patch(plt.Circle((pred.x, pred.y), pred.sigma, color="r", fill=False, alpha=0.5))
ax2.scatter([tl["X_land"]], [tl["Y_land"]], c="g", s=60, label="true landing")
ax2.set_aspect("equal"); ax2.legend(fontsize=7); ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
if pred is not None:
    err = np.hypot(pred.x - tl["X_land"], pred.y - tl["Y_land"])
    fig.suptitle(f"Landing error {err*100:.1f} cm (sigma {pred.sigma*100:.1f} cm), "
                 f"first command at t={first_cmd[0]:.2f}s with {first_cmd[1].ttl:.2f}s to impact" if first_cmd else "no command")
plt.tight_layout()
if args.save:
    plt.savefig(args.save, dpi=120)
    print("saved", args.save)
else:
    plt.show()
