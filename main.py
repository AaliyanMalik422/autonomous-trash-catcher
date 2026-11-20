#!/usr/bin/env python3
"""Autonomous trash-catching robot: end-to-end orchestrator.

    USB cameras -> StereoTracker (vision/) -> TrajectoryEngine (trajectory/) -> ESP32 Wi-Fi UDP -> skid-steer robot

Modes
    python main.py --sim                       # synthetic throws -> engine, no hardware (dry-run)
    python main.py --sim --realtime            # explicitly replay synthetic throws to the REAL robot
    python main.py --sim-vision                # synthetic stereo IMAGES through the full vision path
    python main.py --live --calib stereo_calib.npz --rig rig.json
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
from trajectory.mock_vision import MockStereoCamera
from trajectory.pipeline import ACTIVE, IDLE, TrajectoryEngine
from trajectory.robot_config import DEFAULT_PATH, load_settings


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
    if getattr(a, "latency_s", None) is not None:
        cfg.engine.latency_s = a.latency_s
    if a.correction and os.path.exists(a.correction):
        corr = LandingCorrection.load(a.correction)
        print(f"[init] landing correction loaded from {a.correction}")
    else:
        corr = LandingCorrection()
    logger = ThrowLogger(a.throw_log) if a.log_throws else None
    return TrajectoryEngine(cfg, corr, logger)


def dispatch(cmd, link, verbose=True):
    if cmd is None:
        return
    if cmd.kind == "target":
        # Never dispatch a clipped or physically unreachable interception target.
        # A clamped coordinate is not the predicted landing point.
        if cmd.clamped or not cmd.reachable:
            link.stop()
            if verbose:
                reason = "outside configured reach area" if cmd.clamped else "insufficient time to reach"
                print(f"[safety] target rejected ({reason}); STOP sent")
            return
        link.send_target(cmd.x, cmd.y)
        if verbose:
            print(f"[cmd] target x={cmd.x:+.3f} y={cmd.y:+.3f} ttl={cmd.ttl:.2f}s "
                  f"sigma={cmd.sigma*100:.1f}cm")
    elif cmd.kind == "home":
        link.home()
        if verbose:
            print("[cmd] home")
    elif cmd.kind == "stop":
        link.stop()


def make_link(a, force_dry_run=False):
    """Create the skid-steer UDP link.

    Simulations are dry-run by default. Only --realtime explicitly permits
    synthetic/replayed commands to reach the physical robot.
    """
    from trajectory.esp32_wifi_link import ESP32WifiLink
    return ESP32WifiLink(ip=getattr(a, "wifi_ip", "192.168.4.1"),
                         port=getattr(a, "wifi_port", 4210),
                         dry_run=a.dry_run or force_dry_run,
                         verbose=getattr(a, "verbose", False),
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
    link = make_link(a, force_dry_run=not a.realtime)
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
    # Keep this deterministic scenario aligned with the end-to-end regression fixture.
    rig = SyntheticRig(seed=a.seed + 1)
    tr = StereoTracker(rig.calib, rig.rig)
    for i in range(40):
        tr.process(*rig.render(None), t=-1 + i * 0.01)
    eng = build_engine(a)
    link = make_link(a, force_dry_run=True)
    cam = MockStereoCamera(fps=a.fps, noise_std_meters=0.0, seed=a.seed + 7)
    # Use a representative, near-straight flight that is trackable and reachable
    # from the configured robot home, so this checks the positive command path.
    pk, tl = cam.generate_throw(
        start_pos=(0.2, 2.8, 1.3), velocity=(-0.28, -1.8, 2.2)
    )
    print(f"true landing X={tl['X_land']:+.3f} Y={tl['Y_land']:+.3f}")
    pkts, errs = [], []
    for p in pk:
        pkt = tr.process(*rig.render((p["true_X"], p["true_Y"], p["true_Z"])), t=p["timestamp"])
        if pkt["detected"]:
            errs.append(np.linalg.norm([pkt["X"] - p["true_X"], pkt["Y"] - p["true_Y"], pkt["Z"] - p["true_Z"]]))
        pkts.append(pkt)
    print(f"3D tracking error: median {np.median(errs)*100:.1f} cm over {len(errs)}/{len(pk)} frames")
    run_packets(pkts, eng, link)
    pos_messages = [message for message in link.sent if message.startswith("<POS,")]
    if not pos_messages:
        detected_count = sum(bool(p.get("detected")) for p in pkts)
        raise RuntimeError(
            "Synthetic stereo smoke test did not dispatch an interception target: "
            f"{detected_count}/{len(pkts)} frames detected, engine={eng.state}, "
            f"filter={eng.filter.last_reason!r}, buffered={len(eng.filter.trajectory_buffer)}, "
            f"last_command={eng.last_command!r}. "
            "The full vision-to-robot path must produce a POS command before it passes."
        )
    target_xy = np.array([float(v) for v in pos_messages[0].strip("<>").split(",")[1:3]])
    if np.linalg.norm(target_xy) < 0.05:
        raise RuntimeError(
            f"Synthetic stereo smoke test generated a trivial target {target_xy.tolist()} m; "
            "it must exercise a measurable robot repositioning command."
        )


def mode_replay(a):
    with open(a.replay) as f:
        pk = json.load(f)
    run_packets(pk, build_engine(a), make_link(a, force_dry_run=not a.realtime), realtime=a.realtime)


def wait_for_ack(link, expected, timeout_s=1.0):
    """Wait for a fresh acknowledgement from the ESP32; never reuse an old ACK."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ack = link.last_ack
        if ack is not None and ack[0] == expected:
            return True
        time.sleep(0.01)
    return False


def robot_at_home(link, position_tolerance_m=0.05, speed_tolerance_mps=0.05,
                  max_odo_age_s=0.5):
    """Require fresh ESP32 odometry and a stationary robot at the ZERO origin."""
    odo = getattr(link, "odo", None)
    age_fn = getattr(link, "odo_age_s", None)
    age = age_fn() if callable(age_fn) else None
    if odo is None or age is None or age < 0 or age > max_odo_age_s:
        return False
    x, y, vx, vy, _ = odo
    return (np.hypot(x, y) <= position_tolerance_m
            and np.hypot(vx, vy) <= speed_tolerance_mps)


def wait_for_fresh_odometry(link, timeout_s=1.5, max_age_s=0.5):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        age_fn = getattr(link, "odo_age_s", None)
        age = age_fn() if callable(age_fn) else None
        if getattr(link, "odo", None) is not None and age is not None and 0 <= age <= max_age_s:
            return True
        time.sleep(0.01)
    return False


def handle_missing_camera_pair(eng, link, now=None):
    """Handle a camera-pair timeout without allowing stale motion commands to persist."""
    if eng.state == ACTIVE:
        link.stop()
        raise RuntimeError(
            "Synchronized camera frames were lost during an active interception. "
            "STOP was sent and live autonomy was aborted."
        )
    tick_time = time.monotonic() if now is None else now
    dispatch(eng.tick(tick_time), link)


def mode_live(a):
    import cv2
    from vision.cameras import StereoCameras
    from vision.stereo_tracker import RigGeometry, StereoCalibration, StereoTracker
    for src in (a.left, a.right):
        if "PHONE" in str(src):
            raise SystemExit("Camera sources are still placeholders. Edit config/robot.json "
                             "(or pass --left/--right) with the real stream URLs.")
    if not os.path.exists(a.calib):
        raise SystemExit(f"No stereo calibration at {a.calib}. Run: python -m vision.calibrate_stereo capture / solve "
                         "(see README, phone bring-up step 3).")
    calib = StereoCalibration.load(a.calib)
    if not a.rig or not os.path.isfile(a.rig):
        raise SystemExit(
            f"No measured camera rig geometry at {a.rig!r}. Create the rig JSON with "
            "cam_height_m, pitch_deg, yaw_deg, cam_x_m, and cam_y_m before starting live mode."
        )
    try:
        rig = RigGeometry.load(a.rig)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid camera rig geometry in {a.rig}: {exc}") from exc
    print(f"[init] cameras {a.left} | {a.right}  {a.fps} fps  offset {a.right_offset_ms} ms  latency {a.latency_s} s")
    tracker = StereoTracker(
        calib, rig, detector_backend=a.detector, yolo_model=a.yolo_model,
        yolo_confidence=a.yolo_confidence, yolo_device=a.yolo_device,
    )
    eng = build_engine(a)
    rec = Recorder() if a.record else None
    cams = StereoCameras(a.left, a.right, calib.image_size[0], calib.image_size[1], a.fps, a.swap,
                         right_offset_s=a.right_offset_ms / 1000.0,
                         left_rotate_180=a.left_rotate_180,
                         right_rotate_180=a.right_rotate_180)
    link = make_link(a)
    seq, was_active = -1, False
    try:
        if not a.dry_run:
            link.last_ack = None
            if not link.ping() or not wait_for_ack(link, "PING", timeout_s=1.5):
                raise SystemExit(
                    "No fresh <ACK,PING> from the skid-steer ESP32. Check the TrashBot Wi-Fi connection; "
                    "no motion command was sent."
                )
            print("[init] ESP32 Wi-Fi link verified")

            # ZERO defines the robot-relative origin used by every POS target. Do not
            # start perception/control until the controller confirms this reset.
            link.last_ack = None
            if not link.zero() or not wait_for_ack(link, "ZERO", timeout_s=1.0):
                raise SystemExit(
                    "ESP32 did not acknowledge <ZERO>. Check the controller and restart; "
                    "autonomous commands were not enabled."
                )
            print("[init] robot odometry origin acknowledged")
            link.odo = None
            if not wait_for_fresh_odometry(link, timeout_s=1.5):
                raise SystemExit(
                    "ESP32 acknowledged ZERO but no fresh ODO telemetry arrived. "
                    "Autonomous operation was not enabled."
                )
            print("[init] fresh ESP32 odometry verified")
        else:
            link.zero()
        print("Learning background... keep the zone empty for ~2 s. Press q (with --show) or Ctrl+C to quit.")
        while True:
            r = cams.read(seq, timeout=0.15)
            if r is None:
                handle_missing_camera_pair(eng, link)
                continue
            t, L, R, seq = r
            if not a.dry_run:
                odo_age = link.odo_age_s()
                if odo_age is None or odo_age > 0.5:
                    link.stop()
                    raise RuntimeError(
                        "Fresh ESP32 odometry was lost during live operation. STOP was sent; autonomy aborted."
                    )
                # The reachability model assumes the robot starts at ZERO and at rest.
                # Do not arm another throw until the return-home move has actually completed.
                if eng.state == IDLE and not robot_at_home(link):
                    continue
            pkt = tracker.process(L, R, t, *cams.last_times)
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
    ap.add_argument("--wifi-ip", help="skid-steer ESP32 AP IP (default 192.168.4.1)")
    ap.add_argument("--wifi-port", type=int, help="skid-steer UDP port (default 4210)")
    ap.add_argument("--dry-run", action="store_true", help="print commands instead of sending")
    ap.add_argument("--calib")
    ap.add_argument("--rig", help="required measured JSON: cam_height_m, pitch_deg, yaw_deg, cam_x_m, cam_y_m")
    ap.add_argument("--correction", help="calibration.json from trajectory.calibration fit")
    ap.add_argument("--left", help="USB index or phone stream URL (http://ip:port/video)")
    ap.add_argument("--right", help="USB index or phone stream URL")
    ap.add_argument("--right-offset-ms", type=float, default=None,
                    help="time offset added to right-camera timestamps (measure with vision.sync_tool)")
    ap.add_argument("--swap", action="store_true", default=None, help="swap left/right camera roles")
    ap.add_argument("--fps", type=int)
    ap.add_argument("--latency-s", type=float, help="camera->decision latency (see vision.latency_test)")
    ap.add_argument("--config", help=f"settings file (default {DEFAULT_PATH})")
    ap.add_argument("--home-x", type=float, help="robot home X in world frame (m)")
    ap.add_argument("--home-y", type=float, help="robot home Y in world frame (m)")
    ap.add_argument("--detector", choices=("yoloe", "blob"), default="blob",
                    help="live detector backend (default: fast class-agnostic motion tracking)")
    ap.add_argument("--yolo-model", default="yoloe-26n-seg.pt")
    ap.add_argument("--yolo-confidence", type=float, default=0.18)
    ap.add_argument("--yolo-device", default="cpu")
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
    st = load_settings(a.config)               # config/robot.json (if present) + defaults
    for name, value in (("left", st.left), ("right", st.right), ("fps", st.fps),
                        ("left_rotate_180", st.left_rotate_180),
                        ("right_rotate_180", st.right_rotate_180),
                        ("right_offset_ms", st.right_offset_ms), ("latency_s", st.latency_s),
                        ("home_x", st.home_x), ("home_y", st.home_y), ("calib", st.calib),
                        ("rig", st.rig), ("correction", st.correction), ("swap", st.swap),
                        ("wifi_ip", st.wifi_ip), ("wifi_port", st.wifi_port)):
        if getattr(a, name, None) is None:
            setattr(a, name, value)
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
