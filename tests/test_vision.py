import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trajectory.mock_vision import MockStereoCamera
from vision.stereo_tracker import RigGeometry, StereoCalibration, StereoTracker
from vision.synthetic import SyntheticRig


def _warm(tr, rig, n=40):
    for i in range(n):
        L, R = rig.render(None)
        tr.process(L, R, t=-1.0 + i * 0.01)


def test_triangulation_matches_ground_truth_static_points():
    rig = SyntheticRig(rig=RigGeometry(cam_height_m=1.0, pitch_deg=10.0))
    tr = StereoTracker(rig.calib, rig.rig)
    _warm(tr, rig)
    errs = []
    for (X, Y, Z) in [(0.0, 2.0, 1.0), (0.5, 2.5, 1.3), (-0.4, 3.0, 0.8), (0.2, 1.8, 0.5)]:
        # flash the ball in for a few frames so the background model sees it as foreground
        pkt = None
        for k in range(3):
            L, R = rig.render((X, Y, Z))
            pkt = tr.process(L, R, t=k * 0.03)
        assert pkt["detected"], (X, Y, Z)
        errs.append(np.linalg.norm([pkt["X"] - X, pkt["Y"] - Y, pkt["Z"] - Z]))
        for k in range(30):                       # let the model forget it
            tr.process(*rig.render(None), t=1 + k * 0.01)
    assert max(errs) < 0.06, errs
    assert np.mean(errs) < 0.03


def test_full_vision_path_on_a_throw_feeds_the_trajectory_engine():
    from trajectory.pipeline import TrajectoryEngine
    rig = SyntheticRig(seed=1)
    tr = StereoTracker(rig.calib, rig.rig)
    _warm(tr, rig)
    cam = MockStereoCamera(fps=30, noise_std_meters=0.0, seed=7)
    pk, tl = cam.random_throw()
    eng = TrajectoryEngine()
    cmds, errs = [], []
    for p in pk:
        L, R = rig.render((p["true_X"], p["true_Y"], p["true_Z"]))
        pkt = tr.process(L, R, t=p["timestamp"])
        if pkt["detected"]:
            errs.append(np.linalg.norm([pkt["X"] - p["true_X"], pkt["Y"] - p["true_Y"],
                                        pkt["Z"] - p["true_Z"]]))
        c = eng.update(pkt)
        if c and c.kind == "target":
            cmds.append(c)
    assert len(errs) >= len(pk) * 0.7, "ball lost too often"
    assert np.median(errs) < 0.05
    assert cmds, "engine never produced a target from real image processing"
    e = eng.cfg.engine
    assert abs(cmds[-1].x - (tl["X_land"] - e.home_x)) < 0.15
    assert abs(cmds[-1].y - (tl["Y_land"] - e.home_y)) < 0.15


def test_no_detection_on_static_scene():
    rig = SyntheticRig(seed=2)
    tr = StereoTracker(rig.calib, rig.rig)
    _warm(tr, rig)
    for k in range(20):
        assert not tr.process(*rig.render(None), t=k * 0.03)["detected"]


def test_calibration_save_load(tmp_path):
    cal = StereoCalibration.ideal(focal_px=550, baseline_m=0.1)
    f = str(tmp_path / "c.npz")
    cal.save(f)
    c2 = StereoCalibration.load(f)
    assert abs(c2.baseline - 0.1) < 1e-9 and abs(c2.focal - 550) < 1e-9


# ---------------------------------------------------------- phone / async cameras
def test_pair_frames_picks_closest_with_offset():
    from vision.cameras import pair_frames
    hist = [(1.000, "a", 1), (1.033, "b", 2), (1.066, "c", 3)]
    assert pair_frames(1.040, hist, 0.0, 0.02)[1] == "b"
    # right camera runs 30 ms late -> its stamp must be shifted back by 0.033
    assert pair_frames(1.000, hist, -0.033, 0.01)[1] == "b"
    assert pair_frames(2.0, hist, 0.0, 0.02) is None


def test_estimate_offset_recovers_known_delay():
    from vision.sync_tool import estimate_offset
    rng = np.random.default_rng(0)
    # bursty motion signal sampled at slightly different rates / phases by the two phones
    ev = np.cumsum(rng.uniform(0.15, 0.5, 40))

    def sig(t, delay):
        return sum(np.exp(-((t - (e + delay)) / 0.04) ** 2) for e in ev)

    for true_delay in (0.0, 0.12, -0.08, 0.215):
        tl = np.arange(0.5, 8.0, 1 / 30.0)
        tr = np.arange(0.52, 8.0, 1 / 29.7)
        sl = sig(tl, 0.0) + rng.normal(0, 0.02, len(tl))
        sr = sig(tr, true_delay) + rng.normal(0, 0.02, len(tr))   # right stamps events late by true_delay
        off, score = estimate_offset(tl, sl, tr, sr)
        assert abs(off - (-true_delay)) < 0.012, (true_delay, off)
        assert score > 0.5


def test_async_stereo_cameras_with_fake_captures():
    import time as _t
    import threading
    from vision.cameras import StereoCameras

    class Fake:
        def __init__(self, period, delay, tag):
            self.period, self.delay, self.tag, self.n = period, delay, tag, 0
            self.next = _t.monotonic() + delay
        def isOpened(self): return True
        def set(self, *a): return True
        def grab(self):
            now = _t.monotonic()
            if now < self.next:
                _t.sleep(self.next - now)
            self.next += self.period
            self.n += 1
            return True
        def retrieve(self): return True, np.full((4, 4, 3), self.n % 255, np.uint8)
        def release(self): pass

    cams = StereoCameras("0", "1", fps=30, captures=(Fake(1 / 30, 0.0, "L"), Fake(1 / 30, 0.011, "R")))
    seq, got = -1, 0
    for _ in range(15):
        r = cams.read(seq, timeout=0.5)
        if r:
            seq = r[3]
            got += 1
    skew = np.abs(np.array(cams.skew_log))
    cams.release()
    assert got >= 10
    assert skew.max() < 0.6 / 30 + 1e-6
