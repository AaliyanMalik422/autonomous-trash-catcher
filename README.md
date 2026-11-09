# Autonomous Trash-Catching Robot

Stereo webcams detect a thrown object, a physics model predicts where it will land, and an
ESP32 mecanum robot drives there to catch it in a basket.

```
cameras ─► vision/StereoTracker ─► trajectory/TrajectoryEngine ─► ESP32Link ─► mecanum_locomotion (ESP32)
           (3D packets)            (filter, predict, command)     (serial)      (position control)
```

## Layout
| Path | What it does |
|---|---|
| `vision/stereo_tracker.py` | Rectify → MOG2 background subtraction → blob → stereo triangulation → world-frame packet |
| `vision/cameras.py` | Threaded dual-webcam capture (always the newest frame pair, low latency) |
| `vision/calibrate_stereo.py` | Checkerboard stereo calibration (`capture`, `solve`) |
| `vision/synthetic.py` | Renders fake stereo images of a throw, to test the whole vision path without cameras |
| `vision/trash_detection.py` | Original YOLO single-camera experiment (kept; not used by `main.py`) |
| `trajectory/throw_filter.py` | Rejects people, hand waves and noise (size, height, speed, free-fall curvature) |
| `trajectory/trajectory_predictor.py` | Robust gravity-constrained fit → landing point, impact time and 1-sigma error |
| `trajectory/pipeline.py` | `TrajectoryEngine`: filter + predictor + when/what to command, return-home logic |
| `trajectory/calibration.py` | Landing-correction model fitted from logged real throws |
| `trajectory/esp32_serial.py` | Serial link (auto port detect, rate limit, dry-run) |
| `trajectory/sim.py` | Closed-loop simulation: how many throws can a robot of acceleration *a* catch? |
| `mecanum_locomotion/mecanum_locomotion.ino` | ESP32: PI wheel control, odometry, **position** control, serial parser |
| `tests/` | `pytest` suite + host-side firmware test |

## Coordinate frame (everything uses this)
`X` right (m), `Y` forward depth from the cameras (m), `Z` up (m). The robot's home position in
this frame is `--home-x/--home-y`. Commands to the ESP32 are relative to home: `+x` strafe right,
`+y` forward. If the robot's physical axes are mirrored, flip `sign_x/sign_y` in `trajectory/config.py`.

Packet from vision → trajectory:
`{"timestamp", "detected", "X", "Y", "Z", "blob_area"}` (timestamp: `time.monotonic()` seconds).

ESP32 protocol (115200 baud): `<POS,x,y>`  `<HOME>`  `<STOP>`  `<ZERO>`  `<VEL,vx,vy,w>`;
replies `<ACK,...>` and `<ODO,x,y,vx,vy>` at 20 Hz.

## Quick start
```bash
pip install -r requirements.txt
python -m pytest tests                      # 26 tests
./tests/firmware/run_firmware_test.sh       # compiles the .ino on the PC and drives a simulated robot
python demo_trajectory_plot.py              # trajectory demo plot
python -m trajectory.sim --accel 6 --fps 30 # catch-rate simulation
python main.py --sim                        # synthetic throws through the engine (dry-run)
python main.py --sim-vision                 # synthetic stereo IMAGES through the full vision path
```

## Bring-up on real hardware (in this order)
1. **Flash** `mecanum_locomotion/mecanum_locomotion.ino`. Test with the Serial Monitor:
   `<ZERO>`, `<VEL,0.2,0,0>` (robot should strafe right; fix `MOTOR_INVERT` / `ENC_INVERT` / pins if not),
   then `<POS,0.3,0>` (should stop ~30 cm to the right; check `<ODO,...>` ≈ tape-measure distance,
   adjust `WHEEL_RADIUS_M` / `ENC_COUNTS_PER_WHEEL_REV` if not).
2. **Stereo calibration**: print a 9×6 checkerboard, glue it flat, measure a square (m).
   `python -m vision.calibrate_stereo capture --left 0 --right 1`, then
   `python -m vision.calibrate_stereo solve --square 0.025 --out stereo_calib.npz`. Aim for RMS < 0.6 px.
   Verify: baseline printed ≈ ruler measurement.
3. **Rig geometry**: write `rig.json` = `{"cam_height_m": 1.0, "pitch_deg": 0, "cam_x_m": 0, "cam_y_m": 0}`
   (lens height above floor, downward tilt, rig position in the world frame). Hold a ball at a known
   spot and check the printed X/Y/Z.
4. **Dry run** with real cameras, no motors: `python main.py --live --calib stereo_calib.npz --rig rig.json --dry-run --show --record`
5. **Live**: `python main.py --live --calib stereo_calib.npz --rig rig.json --port COM5 --log-throws`
6. **Calibration model**: after each throw, tape-measure where it landed and fill `true_x,true_y` in
   `logs/throws.csv`; then `python -m trajectory.calibration fit logs/throws.csv --out calibration.json`
   and run with `--correction calibration.json`.
7. Tune in `trajectory/config.py`: zone size, `latency_s` (measure it!), `send_sigma_m`, basket size.

## Using two phones instead of USB webcams
Any app that serves an MJPEG/RTSP stream works (IP Webcam on Android, DroidCam, Iriun, EpocCam).
Pass the stream URLs instead of camera indexes:
```bash
python -m vision.sync_tool --left http://192.168.1.20:8080/video --right http://192.168.1.21:8080/video
python main.py --live --left http://192.168.1.20:8080/video --right http://192.168.1.21:8080/video \
       --right-offset-ms <value from sync_tool> --calib stereo_calib.npz --rig rig.json --port COM5
```
Rules that decide whether this works:
* **Two phones are not frame-synchronised.** Run `vision.sync_tool` (wave a ball in front of both) and pass the printed offset. Re-run it if the Wi-Fi changes.
* **Latency** is the enemy: use a dedicated 5 GHz hotspot/router (not campus Wi-Fi), or USB tethering. Choose 640x480 (or 720p) at 60 fps in the app, MJPEG, lowest quality that still shows the ball. Set `latency_s` in `trajectory/config.py` to the measured value (film a running stopwatch next to the screen showing the stream to measure it).
* In the app lock **exposure, focus and white balance**, disable stabilisation and HDR, and mount both phones rigidly (tripod/bracket, landscape). Do the checkerboard calibration with the phones in their final position and the same app settings; moving a phone invalidates calibration.
* A higher frame rate (60 fps) halves the time needed to see enough points of the throw (see table).

## Feasibility numbers (simulation, 300 throws, landing zone ±0.5 m around home, 15 cm basket radius)
Prediction is accurate (≈2 cm); **the limit is how fast the robot can move**. A throw has ≈0.8 s
of airtime, ≈0.2 s is spent collecting frames, so the robot has ≈0.4–0.5 s to move.

| Camera fps | Pixel noise | Robot accel | Catch rate |
|---|---|---|---|
| 30 | 2.5 cm | 3 m/s² (original firmware limit) | 57 % |
| 30 | 2.5 cm | 6 m/s² | 90 % |
| 30 | 2.5 cm | 10 m/s² | 99 % |
| 60 | 2.5 cm | 6 m/s² | 92 % |
| 30 | 5 cm | 6 m/s² | 70 % |

So: raise `MAX_LIN_ACCEL`/`MAX_LIN_DECEL` in the firmware (needs enough motor torque and wheel grip),
keep the landing zone small, and keep stereo noise low (good calibration, bright even lighting, fixed exposure).
Note: the code assumes **mecanum** wheels (as in the firmware); the proposal text says "omni" – same kinematics idea.

## Not implemented yet
Module 3 (basket fill sensor + tilt/trapdoor servos) is not in this repo's firmware yet.
