#!/usr/bin/env python3
"""Autonomous trash-catching robot: end-to-end orchestrator.

    cameras -> StereoTracker (vision/) -> TrajectoryEngine (trajectory/) -> ESP32Link -> robot

Modes
    python main.py --sim                       # synthetic throws -> engine, no hardware (dry-run)
    python main.py --sim --port COM5 --realtime  # replay synthetic throws to the REAL robot in real time
    python main.py --sim-vision                # synthetic stereo IMAGES through the full vision path
    python main.py --live --calib stereo_calib.npz --rig rig.json --port COM5
    python main.py --live ... --dry-run        # real cameras, print commands instead of driving
    python main.py --replay logs/throw_0001.json   # re-run a recorded throw through the engine

Useful flags: --correction calibration.json (landing correction), --show (camera preview),
              --log-throws (append predictions to logs/throws.csv for calibration).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from trajectory.calibration import LandingCorrection, ThrowLogger
from trajectory.config import Config
from trajectory.esp32_serial import ESP32Link
from trajectory.mock_vision import MockStereoCamera
from trajectory.pipeline import ACTIVE, IDLE, TrajectoryEngine


# --------------------------------------------------------------------- helpers
class Recorder:
    """Saves the raw packets of every tracked throw so it can be replayed/debugged offline."""

    def __init__(self, folder="logs", preroll=12):
        self.folder, self.preroll = folder, preroll
        os.makedirs(folder, exist_ok=True)
        self.ring, self.cur, self.n = [], None, 0
        existing = [f for f in os.listdir(folder) if f.startswith("throw_") and f.endswith(".json")]
        self.n = len(existing)

    def feed(self, pkt, state):
        if state == IDLE and self.cur is None:
            self.ring.append(pkt)
            self.ring = self.ring[-self.preroll:]
            return
        if self.cur is None:
            self.cur = list(self.ring)
        self.cur.append(pkt)

    def finish(self):
        if self.cur and len(self.cur) > self.preroll + 3:
            self.n += 1
            path = os.path.join(self.folder, f"throw_{self.n:04d}.json")
            with open(path, "w") as f:
                json.dump(self.cur, f)
            print(f"[rec] saved {path} ({len(self.cur)} packets)")
        self.cur, self.ring = None, []


def build_engine(a):
    cfg = Config()
    cfg.engine.home_x, cfg.engine.home_y = a.home_x, a.home_y
    corr = LandingCorrection.load(a.correction) if a.correction else LandingCorrection()
    logger = ThrowLogger(a.throw_log) if a.log_throws else None
    return TrajectoryEngine(cfg, corr, logger)


def dispatch(cmd, link, verbose=True):
    if cmd is None:
        return
    if cmd.kind == "target":
        link.send_target(cmd.x, cmd.y)
        if verbose:
            flag = "" if cmd.reachable else "  [!] may be out of reach in time"
            print(f"[cmd] target x={cmd.x:+.3f} y={cmd.y:+.3f} ttl={cmd.ttl:.2f}s "
                  f"sigma={cmd.sigma*100:.1f}cm{flag}")
    elif cmd.kind == "home":
        link.home()
        if verbose:
            print("[cmd] home")
    elif cmd.kind == "stop":
        link.stop()


def make_link(a):
    dry = a.dry_run or a.port is None
    return ESP32Link(port=a.port, dry_run=dry, verbose=False,
                     min_interval_s=0.02 if a.realtime else 0.0)


# ----------------------------------------------------------------------- modes
def run_packets(packets, eng, link, realtime=False, quiet=False):
    t0 = time.monotonic()
    base = packets[0]["timestamp"]
    for p in packets:
        if realtime:
            wait = (p["timestamp"] - base) - (time.monotonic() - t0)
            if wait > 0:
                time.sleep(wait)
        dispatch(eng.update(p), link, not quiet)
    # let the engine finish the episode (hold, then go home)
    t_end = packets[-1]["timestamp"]
    for k in range(1, 400):
        c = eng.tick(t_end + k * 0.01)
        if c is not None:
            dispatch(c, link, not quiet)
            break


def mode_sim(a):
    eng = build_engine(a)
    link = make_link(a)
    t = 0.0
    for i in range(a.n):
        cam = MockStereoCamera(fps=a.fps, noise_std_meters=a.noise, seed=a.seed + i, t0=t)
        pk, tl = cam.random_throw(home=(eng.cfg.engine.home_x, eng.cfg.engine.home_y))
        print(f"\n=== throw {i+1}: true landing X={tl['X_land']:+.3f} Y={tl['Y_land']:+.3f} "
              f"(robot frame {tl['X_land']-eng.cfg.engine.home_x:+.3f}, {tl['Y_land']-eng.cfg.engine.home_y:+.3f})")
        run_packets(pk, eng, link, realtime=a.realtime)
        t = pk[-1]["timestamp"] + 5.0
        eng.tick(t)
        eng.tick(t + 1)
    link.close()


def mode_sim_vision(a):
    from vision.stereo_tracker import StereoTracker
    from vision.synthetic import SyntheticRig
    rig = SyntheticRig(seed=a.seed)
    tr = StereoTracker(rig.calib, rig.rig)
    for i in range(40):
        tr.process(*rig.render(None), t=-1 + i * 0.01)
    eng = build_engine(a)
    link = make_link(a)
    cam = MockStereoCamera(fps=a.fps, noise_std_meters=0.0, seed=a.seed)
    pk, tl = cam.random_throw(home=(eng.cfg.engine.home_x, eng.cfg.engine.home_y))
    print(f"true landing X={tl['X_land']:+.3f} Y={tl['Y_land']:+.3f}")
    pkts, errs = [], []
    for p in pk:
        pkt = tr.process(*rig.render((p["true_X"], p["true_Y"], p["true_Z"])), t=p["timestamp"])
        if pkt["detected"]:
            errs.append(np.linalg.norm([pkt["X"] - p["true_X"], pkt["Y"] - p["true_Y"], pkt["Z"] - p["true_Z"]]))
        pkts.append(pkt)
    print(f"3D tracking error: median {np.median(errs)*100:.1f} cm over {len(errs)}/{len(pk)} frames")
    run_packets(pkts, eng, link)


def mode_replay(a):
    with open(a.replay) as f:
        pk = json.load(f)
    run_packets(pk, build_engine(a), make_link(a), realtime=a.realtime)


def mode_live(a):
    import cv2
    from vision.cameras import StereoCameras
    from vision.stereo_tracker import RigGeometry, StereoCalibration, StereoTracker
    calib = StereoCalibration.load(a.calib)
    rig = RigGeometry.load(a.rig) if a.rig else RigGeometry()
    tracker = StereoTracker(calib, rig)
    eng = build_engine(a)
    link = ESP32Link(port=a.port, dry_run=a.dry_run)
    rec = Recorder() if a.record else None
    cams = StereoCameras(a.left, a.right, calib.image_size[0], calib.image_size[1], a.fps, a.swap,
                         right_offset_s=a.right_offset_ms / 1000.0)
    print("Learning background... keep the zone empty for ~2 s. Press q (with --show) or Ctrl+C to quit.")
    link.zero()
    seq, was_active = -1, False
    try:
        while True:
            r = cams.read(seq, timeout=0.5)
            if r is None:
                eng_cmd = eng.tick(time.monotonic())
                dispatch(eng_cmd, link)
                continue
            t, L, R, seq = r
            pkt = tracker.process(L, R, t)
            cmd = eng.update(pkt)
            dispatch(cmd, link)
            if rec:
                rec.feed(pkt, eng.state)
                if was_active and eng.state != ACTIVE:
                    rec.finish()
            was_active = eng.state == ACTIVE
            for line in link.poll():
                if a.verbose:
                    print("[esp32]", line)
            if a.show:
                vis = np.hstack([L, R])
                cv2.putText(vis, f"{eng.state} det={pkt['detected']}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("stereo", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cams.release()
        link.close()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sim", action="store_true")
    g.add_argument("--sim-vision", action="store_true")
    g.add_argument("--live", action="store_true")
    g.add_argument("--replay", metavar="JSON")
    ap.add_argument("--port", help="ESP32 serial port (COM5, /dev/ttyUSB0). Omit to auto-detect (live).")
    ap.add_argument("--dry-run", action="store_true", help="print commands instead of sending")
    ap.add_argument("--calib", default="stereo_calib.npz")
    ap.add_argument("--rig", help="JSON with cam_height_m, pitch_deg, cam_x_m, cam_y_m")
    ap.add_argument("--correction", help="calibration.json from trajectory.calibration fit")
    ap.add_argument("--left", default="0", help="USB index or phone stream URL (http://ip:port/video)")
    ap.add_argument("--right", default="1", help="USB index or phone stream URL")
    ap.add_argument("--right-offset-ms", type=float, default=0.0,
                    help="time offset added to right-camera timestamps (measure with vision.sync_tool)")
    ap.add_argument("--swap", action="store_true", help="swap left/right camera roles")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--home-x", type=float, default=0.0, help="robot home X in world frame (m)")
    ap.add_argument("--home-y", type=float, default=1.3, help="robot home Y in world frame (m)")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--record", action="store_true", help="save raw packets of each throw to logs/")
    ap.add_argument("--log-throws", action="store_true")
    ap.add_argument("--throw-log", default="logs/throws.csv")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--n", type=int, default=3, help="--sim: number of throws")
    ap.add_argument("--noise", type=float, default=0.025)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--realtime", action="store_true")
    a = ap.parse_args()
    if a.sim:
        mode_sim(a)
    elif a.sim_vision:
        mode_sim_vision(a)
    elif a.replay:
        mode_replay(a)
    else:
        mode_live(a)


if __name__ == "__main__":
    main()
