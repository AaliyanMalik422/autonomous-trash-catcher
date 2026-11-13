"""Regression tests for command-line transport selection."""
from types import SimpleNamespace

import main


def test_make_link_uses_wifi_when_requested(monkeypatch):
    captured = {}

    class FakeWifiLink:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("trajectory.esp32_wifi_link.ESP32WifiLink", FakeWifiLink)
    args = SimpleNamespace(
        wifi=True,
        wifi_ip="192.168.4.1",
        wifi_port=4210,
        dry_run=True,
        verbose=True,
        realtime=False,
    )

    link = main.make_link(args)

    assert isinstance(link, FakeWifiLink)
    assert captured == {
        "ip": "192.168.4.1",
        "port": 4210,
        "dry_run": True,
        "verbose": True,
        "min_interval_s": 0.0,
    }
