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
