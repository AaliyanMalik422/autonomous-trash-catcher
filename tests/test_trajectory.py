import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trajectory.calibration import LandingCorrection, ThrowLogger, load_pairs
from trajectory.config import Config
from trajectory.esp32_wifi_link import ESP32WifiLink
from trajectory.mock_vision import MockStereoCamera
from trajectory.pipeline import ACTIVE, COOLDOWN, IDLE, TrajectoryEngine, reach_distance
from trajectory.throw_filter import ThrowFilter
from trajectory.trajectory_predictor import TrajectoryPredictor


# ------------------------------------------------------------------ predictor
def test_predictor_exact_data_exact_landing():
    cam = MockStereoCamera(fps=60, noise_std_meters=0.0)
    pk, tl = cam.generate_throw()
    p = TrajectoryPredictor().predict(pk[:10])
    assert p is not None
    assert abs(p.x - tl["X_land"]) < 1e-3
    assert abs(p.y - tl["Y_land"]) < 1e-3
    assert abs(p.t_impact - tl["t_land"]) < 1e-3


def test_predictor_accuracy_with_noise():
    errs = []
    for s in range(100):
        cam = MockStereoCamera(fps=30, noise_std_meters=0.025, seed=s)
        pk, tl = cam.random_throw()
        p = TrajectoryPredictor().predict(pk[:9])      # ~0.27 s of flight
        errs.append(np.hypot(p.x - tl["X_land"], p.y - tl["Y_land"]))
    assert np.median(errs) < 0.12
    assert np.percentile(errs, 90) < 0.25


def test_predictor_sigma_is_calibrated():
    """~68% of errors per axis should fall inside the reported 1-sigma (loosely)."""
    inside = 0
    N = 200
    for s in range(N):
        cam = MockStereoCamera(fps=30, noise_std_meters=0.025, seed=1000 + s)
        pk, tl = cam.random_throw()
        p = TrajectoryPredictor().predict(pk[:8])
        if np.hypot(p.x - tl["X_land"], p.y - tl["Y_land"]) < 2.0 * p.sigma:
            inside += 1
    assert inside / N > 0.75


def test_predictor_rejects_outliers():
    cam = MockStereoCamera(fps=60, noise_std_meters=0.01, seed=3)
    pk, tl = cam.generate_throw()
    clean = TrajectoryPredictor().predict(pk[:14])
    pk2 = [dict(p) for p in pk[:14]]
    pk2[6]["X"] += 0.7
    pk2[9]["Z"] -= 0.6
    dirty = TrajectoryPredictor().predict(pk2)
    assert np.hypot(dirty.x - clean.x, dirty.y - clean.y) < 0.08


def test_predictor_picks_descending_crossing():
    # thrown upward from below rim height: first crossing is ascending, landing is the 2nd
    cam = MockStereoCamera(fps=60, noise_std_meters=0.0)
    pk, tl = cam.generate_throw(start_pos=(0, 3.0, 0.2), velocity=(0, -1.0, 4.0))
    p = TrajectoryPredictor().predict(pk[3:8])
    assert abs(p.t_impact - tl["t_land"]) < 1e-3


def test_predictor_needs_points():
    assert TrajectoryPredictor().predict([]) is None
    cam = MockStereoCamera()
    pk, _ = cam.generate_throw()
    assert TrajectoryPredictor().predict(pk[:2]) is None


def test_legacy_api():
    cam = MockStereoCamera(fps=60, noise_std_meters=0.0)
    pk, tl = cam.generate_throw()
    px, py, tti, fit = TrajectoryPredictor().predict_landing(pk[:10])
    assert abs(px - tl["X_land"]) < 1e-3 and len(fit["curve_x"]) == 50


# --------------------------------------------------------------------- filter
def _confirmed(pk):
    f = ThrowFilter()
    return any(f.process_frame(p)[0] for p in pk)


@pytest.mark.parametrize("fps", [30, 60])
def test_filter_accepts_throws(fps):
    ok = sum(_confirmed(MockStereoCamera(fps=fps, seed=s).random_throw()[0]) for s in range(60))
    assert ok == 60


def test_filter_rejects_walker():
    for s in range(20):
        assert not _confirmed(MockStereoCamera(seed=s).generate_walker())


def test_filter_rejects_floor_and_noise():
    cam = MockStereoCamera(fps=30, seed=1)
    pk = [{"timestamp": i / 30, "detected": True, "X": 0.0, "Y": 2.0, "Z": 0.05,
           "blob_area": 300.0} for i in range(30)]
    assert not _confirmed(pk)
    pk = [{"timestamp": i / 30, "detected": False, "X": 0, "Y": 0, "Z": 0, "blob_area": 0}
          for i in range(30)]
    assert not _confirmed(pk)


def test_filter_survives_dropouts_and_outliers():
    ok = 0
    for s in range(40):
        cam = MockStereoCamera(fps=30, seed=s, dropout_prob=0.1, outlier_prob=0.05)
        ok += _confirmed(cam.random_throw()[0])
    assert ok >= 30


def test_filter_drops_track_after_gap():
    f = ThrowFilter()
    cam = MockStereoCamera(fps=30, seed=2)
    pk, _ = cam.random_throw()
    for p in pk[:3]:
        f.process_frame(p)
    assert len(f.trajectory_buffer) == 3
    late = dict(pk[3]); late["timestamp"] += 1.0
    f.process_frame(late)
    assert len(f.trajectory_buffer) == 1


# --------------------------------------------------------------------- engine
def _run_engine(pk, eng=None):
    eng = eng or TrajectoryEngine()
    cmds = []
    for p in pk:
        c = eng.update(p)
        if c:
            cmds.append((p["timestamp"], c))
    return eng, cmds


def test_engine_throw_produces_correct_target():
    cam = MockStereoCamera(fps=30, seed=11)
    pk, tl = cam.random_throw()
    eng, cmds = _run_engine(pk)
    targets = [c for _, c in cmds if c.kind == "target"]
    assert targets, "engine never sent a target"
    last = targets[-1]
    e = eng.cfg.engine
    assert abs(last.x - e.sign_x * (tl["X_land"] - e.home_x)) < 0.10
    assert abs(last.y - e.sign_y * (tl["Y_land"] - e.home_y)) < 0.10
    assert targets[0].ttl > 0.1          # sent early enough to be useful


def test_engine_ignores_walker_and_wave_mostly():
    for s in range(10):
        eng, cmds = _run_engine(MockStereoCamera(seed=s).generate_walker())
        assert not [c for _, c in cmds if c.kind == "target"]


def test_engine_returns_home_after_throw_then_cools_down_then_rearms():
    cam = MockStereoCamera(fps=30, seed=21)
    pk, tl = cam.random_throw()
    eng = TrajectoryEngine()
    _, cmds = _run_engine(pk, eng)
    # run time forward past impact + hold
    t = pk[-1]["timestamp"]
    c = eng.tick(t + 2.0)
    kinds = [c.kind for _, c in cmds] + ([c.kind] if c else [])
    assert "home" in kinds
    assert eng.state in (COOLDOWN, IDLE)
    eng.tick(t + 5.0)
    assert eng.state == IDLE
    # second throw (time-shifted) works again
    pk2, _ = MockStereoCamera(fps=30, seed=22, t0=t + 6.0).random_throw()
    _, cmds2 = _run_engine(pk2, eng)
    assert [c for _, c in cmds2 if c.kind == "target"]


def test_engine_clamps_out_of_reach():
    cfg = Config()
    cfg.zone.reach_x = cfg.zone.reach_y = 0.3
    cam = MockStereoCamera(fps=60, noise_std_meters=0.0)
    pk, _ = cam.generate_throw(start_pos=(0.0, 2.6, 1.3), velocity=(2.0, -1.0, 2.2))
    eng = TrajectoryEngine(cfg)
    _, cmds = _run_engine(pk, eng)
    t = [c for _, c in cmds if c.kind == "target"][-1]
    assert t.clamped and not t.reachable
    assert abs(t.x) <= 0.3 + 1e-9 and abs(t.y) <= 0.3 + 1e-9


def test_reach_distance():
    assert reach_distance(0, 3, 2) == 0
    assert reach_distance(0.2, 3, 2) == pytest.approx(0.06)
    assert reach_distance(2.0, 3, 1.8) > reach_distance(1.0, 3, 1.8)


# ---------------------------------------------------------------- calibration
def test_correction_identity_and_fit(tmp_path):
    assert LandingCorrection().apply(1.0, 2.0) == (1.0, 2.0)
    rng = np.random.default_rng(0)
    P = rng.uniform(-0.5, 0.5, (30, 2))
    A = np.array([[1.05, 0.02], [-0.01, 0.97]])
    b = np.array([0.04, -0.03])
    T = P @ A.T + b + rng.normal(0, 0.003, P.shape)
    corr = LandingCorrection.fit(P, T, ridge=1.0)
    after = np.linalg.norm(np.array([corr.apply(*p) for p in P]) - T, axis=1).mean()
    before = np.linalg.norm(P - T, axis=1).mean()
    assert after < 0.3 * before
    f = tmp_path / "c.json"
    corr.save(str(f))
    assert np.allclose(LandingCorrection.load(str(f)).A, corr.A)


def test_correction_few_samples_stays_near_identity():
    corr = LandingCorrection.fit([[0.1, 0.2]], [[0.15, 0.25]], ridge=3.0)
    assert np.allclose(corr.A, np.eye(2), atol=0.1)


def test_logger_roundtrip(tmp_path):
    path = str(tmp_path / "throws.csv")
    lg = ThrowLogger(path)
    pk, _ = MockStereoCamera(fps=30, seed=4).random_throw()
    p = TrajectoryPredictor().predict(pk[:8])
    lg.log(p, 0.3)
    lg.log(p, 0.3)
    assert lg.count == 2 and ThrowLogger(path).count == 2
    assert len(load_pairs(path)[0]) == 0   # true landing not filled in yet


# ---------------------------------------------------------------- skid-steer wifi
def test_wifi_link_dry_run_and_parity():
    link = ESP32WifiLink(dry_run=True, min_interval_s=0.05)
    assert link.send_target(0.123456, -0.5)
    assert not link.send_target(0.2, 0.2)          # rate limited
    link.home()
    link.stop()
    link.ping()
    link.zero()
    assert link.sent == ["<POS,0.123,-0.500>", "<HOME>", "<STOP>", "<PING>", "<ZERO>"]
    link.close()

