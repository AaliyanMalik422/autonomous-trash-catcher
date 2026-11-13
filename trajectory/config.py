"""Central tunables for the trajectory pipeline.

Coordinate convention (shared with vision/):
    X  left/right of the camera pair (m, +right)
    Y  forward depth from the cameras (m)
    Z  height above the floor (m)
All timestamps are seconds on one monotonic clock.
"""
from dataclasses import dataclass, field


@dataclass
class ZoneConfig:
    # Region (camera frame, metres) in which a throw may originate / be tracked.
    x_min: float = -2.0
    x_max: float = 2.0
    y_min: float = 0.3
    y_max: float = 5.0
    # Region the robot can physically cover (landing points outside are clamped).
    reach_x: float = 0.8      # +/- metres from robot home
    reach_y: float = 0.8


@dataclass
class FilterConfig:
    min_area: float = 100.0        # px, reject dust
    max_area: float = 2500.0       # px, reject people
    min_height_z: float = 0.30     # m, must be airborne
    max_gap_s: float = 0.20        # no valid detection for this long -> drop track
    max_jump_m: float = 0.60       # per-frame jump beyond this -> outlier point
    min_points: int = 5            # points before a throw can be confirmed
    min_speed: float = 1.0         # m/s total speed
    max_speed: float = 14.0        # m/s total speed
    approach_sign: int = -1        # -1: Y decreases toward the robot/cameras
    min_approach_speed: float = 0.4  # m/s along Y toward robot
    max_z_rms: float = 0.07        # m, rms of gravity-constrained vertical fit
    max_xy_rms: float = 0.07       # m, rms of linear horizontal fit
    min_duration_s: float = 0.16   # track must span at least this long before confirming
    curvature_min_duration_s: float = 0.16  # apply free-fall curvature check beyond this span
    reconsider_min_duration_s: float = 0.30  # re-check curvature on longer baseline
    curvature_tol: float = 18.0    # m/s^2 allowed deviation of fitted Z accel from -g


@dataclass
class PredictorConfig:
    g: float = 9.81
    target_z: float = 0.35         # basket rim height (m)
    min_points: int = 4
    outlier_sigma: float = 3.0
    use_weights: bool = True       # weight recent points slightly higher


@dataclass
class EngineConfig:
    send_sigma_m: float = 0.15     # send first command once 1-sigma landing error < this
    max_wait_s: float = 0.30       # ...or after this much tracking time regardless
    min_ttl_s: float = 0.06        # too late to move below this
    latency_s: float = 0.08        # camera+USB+processing+serial latency to subtract
    min_update_delta_m: float = 0.02   # don't resend if target moved less than this
    hold_after_impact_s: float = 0.40  # stay put this long after predicted impact
    cooldown_s: float = 0.60
    return_home: bool = True
    # robot home position in camera frame + axis mapping to robot frame
    home_x: float = 0.0
    home_y: float = 1.3
    sign_x: float = 1.0            # robot +x (strafe right) = sign_x * camera X
    sign_y: float = 1.0            # robot +y (forward)      = sign_y * camera Y
    # robot dynamics for the reachability estimate
    max_accel: float = 3.6         # m/s^2; conservative firmware braking acceleration (0.6 * MAX_LIN_DECEL)
    max_speed: float = 0.9         # m/s; firmware POS_MAX_V for autonomous position moves


@dataclass
class Config:
    zone: ZoneConfig = field(default_factory=ZoneConfig)
    filt: FilterConfig = field(default_factory=FilterConfig)
    pred: PredictorConfig = field(default_factory=PredictorConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
