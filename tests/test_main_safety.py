import pytest

from main import ACTIVE, handle_missing_camera_pair


class FakeEngine:
    state = ACTIVE


class FakeLink:
    def __init__(self):
        self.stops = 0

    def stop(self):
        self.stops += 1


def test_camera_loss_during_active_interception_stops_and_aborts():
    link = FakeLink()
    with pytest.raises(RuntimeError, match="STOP was sent"):
        handle_missing_camera_pair(FakeEngine(), link, now=1.0)
    assert link.stops == 1


def test_robot_at_home_requires_fresh_stationary_odometry():
    from main import robot_at_home

    class Link:
        odo = (0.01, -0.01, 0.0, 0.02, 10.0)

        def odo_age_s(self):
            return 0.1

    assert robot_at_home(Link())

    moving = Link()
    moving.odo = (0.0, 0.0, 0.2, 0.0, 10.0)
    assert not robot_at_home(moving)

    away = Link()
    away.odo = (0.2, 0.0, 0.0, 0.0, 10.0)
    assert not robot_at_home(away)

    stale = Link()
    stale.odo_age_s = lambda: 1.0
    assert not robot_at_home(stale)
