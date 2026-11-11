"""One settings file for the real rig (phones, serial port, geometry, calibration files).

Edit config/robot.json once; `python main.py --live` then needs no other flags.
Command-line flags still override anything in the file.
"""
import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Optional

DEFAULT_PATH = "config/robot.json"


@dataclass
class RobotSettings:
    # --- cameras (phones: stream URLs; USB: integer indexes as strings) ---
    left: str = "http://PHONE1_IP:8080/video"
    right: str = "http://PHONE2_IP:8080/video"
    swap: bool = False                 # set true if "left" is actually the right-hand camera
    fps: int = 60
    width: int = 640                   # 640x480 is the sweet spot: ~8 ms/frame-pair on a laptop CPU
    height: int = 480
    right_offset_ms: float = 0.0       # from `python -m vision.sync_tool`
    latency_s: float = 0.12            # from `python -m vision.latency_test` (camera->decision)
    # --- robot ---
    serial_port: Optional[str] = None  # e.g. "/dev/ttyUSB0" or "COM5"; null = auto-detect
    home_x: float = 0.0                # robot rest position in world frame (m)
    home_y: float = 1.3
    # --- files produced during bring-up ---
    calib: str = "stereo_calib.npz"
    rig: str = "rig.json"
    correction: Optional[str] = "calibration.json"

    def has_placeholder_sources(self) -> bool:
        return "PHONE" in str(self.left) or "PHONE" in str(self.right)

    def save(self, path: str = DEFAULT_PATH):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


def load_settings(path: Optional[str] = None) -> RobotSettings:
    path = path or (DEFAULT_PATH if os.path.exists(DEFAULT_PATH) else None)
    if path is None:
        return RobotSettings()
    with open(path) as f:
        raw = json.load(f)
    known = {f.name for f in fields(RobotSettings)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
    return RobotSettings(**raw)
