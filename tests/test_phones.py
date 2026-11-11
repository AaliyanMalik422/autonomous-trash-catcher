"""Tests for the phone bring-up tooling and a simulated 'two phones on Wi-Fi' end-to-end run."""
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trajectory.mock_vision import MockStereoCamera
from trajectory.pipeline import TrajectoryEngine
from trajectory.robot_config import RobotSettings, load_settings
from vision.cameras import StereoCameras
from vision.latency_test import detect_flash_latency
from vision.phones_check import summarize, verdict
from vision.stereo_tracker import StereoTracker
from vision.synthetic import SyntheticRig


def test_settings_roundtrip_and_validation(tmp_path):
    p = tmp_path / "robot.json"
    s = RobotSettings(left="http://1.2.3.4:8080/video", right="http://1.2.3.5:8080/video", fps=60)
    s.save(str(p))
    s2 = load_settings(str(p))
    assert s2.left.endswith("8080/video") and s2.fps == 60 and not s2.has_placeholder_sources()
    assert RobotSettings().has_placeholder_sources()
    p.write_text('{"typo_key": 1}')
    with pytest.raises(ValueError):
        load_settings(str(p))


def test_phones_check_verdicts():
    good = summarize(np.arange(0, 2, 1 / 60))
    assert abs(good["fps"] - 60) < 0.5
    assert verdict(good, 60, (640, 480), (640, 480))[0] == "PASS"
    slow = summarize(np.arange(0, 2, 1 / 15))
    assert verdict(slow, 60, (640, 480), (640, 480))[0] == "FAIL"
    assert verdict(good, 60, (1280, 720), (640, 480))[0] == "WARN"
    assert verdict(summarize([0, 1]), 60, (0, 0), (640, 480))[0] == "FAIL"
    stall = np.concatenate([np.arange(0, 1, 1 / 60), np.arange(1.3, 2.3, 1 / 60)])
    assert verdict(summarize(stall), 60, (640, 480), (640, 480))[0] == "WARN"


def test_flash_latency_detection():
    t = np.arange(0, 2, 1 / 60)
    t_flash = 1.0
    b = np.where(t >= t_flash + 0.145, 200.0, 20.0)         # phone shows white 145 ms later
    lat = detect_flash_latency(t, b, t_flash, low=20.0, high=200.0)
    assert 0.145 <= lat < 0.145 + 1 / 60 + 1e-9
    assert detect_flash_latency(t, np.full_like(t, 20.0), t_flash, 20.0, 200.0) is None


class FakePhone:
    """Looks like cv2.VideoCapture: delivers frames at `fps`; each frame shows the scene at its
    capture time minus `delay_s` (network latency), taken from a fine 240 Hz pre-rendered grid."""

    GRID = 240

    def __init__(self, frames, fps, delay_s, t0, phase=0.0):
        self.frames, self.fps, self.delay, self.t0 = frames, fps, delay_s, t0
        self.n = 0
        self.next = time.monotonic() + phase

    def isOpened(self): return True
    def set(self, *a): return True
    def release(self): pass

    def grab(self):
        now = time.monotonic()
        if now < self.next:
            time.sleep(self.next - now)
        self.cap_t = self.next                         # exact exposure time
        self.next += 1.0 / self.fps
        return True

    def retrieve(self):
        k = int(round((self.cap_t - self.t0 - self.delay) * self.GRID))
        return True, self.frames[min(max(k, 0), len(self.frames) - 1)]


_CACHE = {}


def _scenario(fps, delay_r, right_offset, seed=3, phase_r=0.0083):
    rig = SyntheticRig(seed=seed)
    if seed in _CACHE:
        left, right, truth, tl = _CACHE[seed]
        t0 = time.monotonic() + 0.1
        cams = StereoCameras('0', '1', fps=fps, right_offset_s=right_offset,
                             captures=(FakePhone(left, fps, 0.0, t0), FakePhone(right, fps, delay_r, t0, phase_r)))
        return rig, cams, truth, tl, t0, len(left)
    G = FakePhone.GRID
    cam = MockStereoCamera(fps=G, noise_std_meters=0.0, seed=seed)
    pk, tl = cam.random_throw()
    empty = rig.render(None)
    n_bg = G                                    # 1 s of empty scene first
    left, right, truth = [], [], []
    for _ in range(n_bg):
        left.append(empty[0]); right.append(empty[1]); truth.append(None)
    for p in pk:
        L, R = rig.render((p["true_X"], p["true_Y"], p["true_Z"]))
        left.append(L); right.append(R); truth.append((p["true_X"], p["true_Y"], p["true_Z"]))
    for _ in range(G // 2):
        left.append(empty[0]); right.append(empty[1]); truth.append(None)
    _CACHE[seed] = (left, right, truth, tl)
    t0 = time.monotonic() + 0.1
    cams = StereoCameras("0", "1", fps=fps, right_offset_s=right_offset,
                         captures=(FakePhone(left, fps, 0.0, t0), FakePhone(right, fps, delay_r, t0, phase_r)))
    return rig, cams, truth, tl, t0, len(left)


def _run(fps, delay_r, right_offset):
    rig, cams, truth, tl, t0, n = _scenario(fps, delay_r, right_offset)
    tracker = StereoTracker(rig.calib, rig.rig)
    eng = TrajectoryEngine()
    eng.cfg.engine.latency_s = 0.05
    errs, targets, seq = [], [], -1
    G = FakePhone.GRID
    end = t0 + n / G
    while time.monotonic() < end:
        r = cams.read(seq, timeout=0.2)
        if r is None:
            continue
        t, L, R, seq = r
        pkt = tracker.process(L, R, t, *cams.last_times)
        k = int(round((t - t0) * G))
        if pkt["detected"] and 0 <= k < len(truth) and truth[k] is not None:
            errs.append(np.linalg.norm([pkt["X"] - truth[k][0], pkt["Y"] - truth[k][1], pkt["Z"] - truth[k][2]]))
        c = eng.update(pkt)
        if c and c.kind == "target":
            targets.append(c)
    cams.release()
    return errs, targets, tl, eng


def test_two_async_phones_end_to_end_with_correct_offset():
    """60 fps, right phone 25 ms behind the left one (offset compensated, random frame phase).
    Real-time threads make single runs vary, so judge 5 runs together."""
    med_errs, got_target = [], 0
    for _ in range(5):
        errs, targets, tl, eng = _run(fps=60, delay_r=0.025, right_offset=-0.025)
        assert len(errs) > 12, "ball hardly tracked"
        med_errs.append(float(np.median(errs)))
        got_target += bool(targets)
    assert np.median(med_errs) < 0.12, med_errs
    assert got_target >= 4, got_target
