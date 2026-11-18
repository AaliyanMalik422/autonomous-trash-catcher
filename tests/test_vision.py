import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trajectory.mock_vision import MockStereoCamera
from vision.stereo_tracker import RigGeometry, StereoCalibration, StereoTracker
from vision.cameras import parse_source
from vision.synthetic import SyntheticRig
from vision.calibrate_stereo import ensure_empty_capture_directory


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


def test_parse_usb_camera_indices_and_stream_urls():
    assert parse_source("0") == 0
    assert parse_source("1") == 1
    assert parse_source("http://192.168.1.20:8080/video") == "http://192.168.1.20:8080/video"


def test_triangulation_returns_original_left_camera_frame_after_rectification():
    # The rectified projection matrices reconstruct R1 @ point. The tracker must
    # rotate that point back before applying RigGeometry.
    from vision.stereo_tracker import DetectorConfig
    angle = np.radians(8.0)
    R1 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(angle), np.sin(angle)],
        [0.0, -np.sin(angle), np.cos(angle)],
    ])
    base = StereoCalibration.ideal(focal_px=600.0, baseline_m=0.12)
    cal = StereoCalibration(base.image_size, base.P1, base.P2, R1=R1)
    tracker = StereoTracker(cal, RigGeometry(), det_cfg=DetectorConfig(min_disparity_px=0.1))
    expected = np.array([0.15, 0.08, 2.5])
    rect = R1 @ expected
    fx, cx, cy = cal.P1[0, 0], cal.P1[0, 2], cal.P1[1, 2]
    ul = fx * rect[0] / rect[2] + cx
    vl = fx * rect[1] / rect[2] + cy
    ur = ul - fx * cal.baseline / rect[2]
    actual = tracker.triangulate(ul, vl, ur, vl)
    assert actual is not None
    assert np.allclose(actual, expected, atol=1e-5)


def test_calibration_rejects_frame_size_mismatch():
    cal = StereoCalibration.ideal(width=640, height=480)
    left = np.zeros((480, 640, 3), dtype=np.uint8)
    wrong = np.zeros((720, 1280, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Camera frame size must match calibration"):
        cal.rectify(left, wrong)


def test_calibration_rejects_invalid_rectification_rotation():
    with pytest.raises(ValueError, match="R1 must be a finite 3x3"):
        StereoCalibration((640, 480), np.eye(3), np.eye(3), R1=np.eye(2))


def test_calibration_capture_rejects_existing_session_images(tmp_path):
    ensure_empty_capture_directory(str(tmp_path))
    (tmp_path / "left_00.png").write_bytes(b"old left capture")
    (tmp_path / "right_00.png").write_bytes(b"old right capture")
    with pytest.raises(SystemExit, match="fresh empty directory"):
        ensure_empty_capture_directory(str(tmp_path))


def test_rig_geometry_yaw_rotates_camera_coordinates_into_robot_frame():
    rig = RigGeometry(cam_height_m=1.0, yaw_deg=90.0, cam_x_m=0.2, cam_y_m=0.3)
    X, Y, Z = rig.cam_to_world(0.0, 0.0, 2.0)
    assert X == pytest.approx(2.2)
    assert Y == pytest.approx(0.3)
    assert Z == pytest.approx(1.0)


def test_rig_geometry_rejects_invalid_pose():
    with pytest.raises(ValueError, match="Camera height"):
        RigGeometry(cam_height_m=0.0)
    with pytest.raises(ValueError, match="pitch"):
        RigGeometry(pitch_deg=90.0)


def test_stereo_calibration_rejects_non_rotation_r1():
    cal = StereoCalibration.ideal()
    with pytest.raises(ValueError, match="orthonormal"):
        StereoCalibration(image_size=cal.image_size, P1=cal.P1, P2=cal.P2,
                          R1=np.zeros((3, 3)))


def test_camera_sync_estimator_rejects_invalid_samples():
    from vision.sync_tool import estimate_offset
    with pytest.raises(ValueError, match="at least three"):
        estimate_offset([], [], [], [])
    t = np.linspace(0.0, 4.0, 401)
    signal = np.sin(2.1 * t) + 0.35 * np.sin(4.7 * t + 0.2)
    with pytest.raises(ValueError, match="strictly increasing"):
        estimate_offset([0.0, 1.0, 0.5], [0.0, 1.0, 0.0],
                        [0.0, 1.0, 2.0], [0.0, 1.0, 0.0])


def test_camera_sync_estimator_returns_strong_score_for_matching_signals():
    from vision.sync_tool import estimate_offset
    t = np.linspace(0.0, 8.0, 801)
    signal = np.sin(1.7 * t) + 0.4 * np.sin(3.1 * t + 0.3) + 0.15 * np.sin(6.3 * t)
    offset, score = estimate_offset(t, signal, t, signal)
    assert abs(offset) < 0.005
    assert score > 0.9
