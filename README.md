# Autonomous Trash-Catching Robot

A laptop-based stereo-vision system predicts where a thrown object will land. A skid-steer ESP32 robot then turns and drives to that point so the object falls into the open basket.

**Current project scope:** Module 1 (Stereo Vision & Trajectory Prediction) and Module 2 (Skid-Steer Locomotion & Interception). Module 3 (Payload Sensing & Auto-Emptying) is deferred and is not part of the current completion target.

## Current system architecture

```
PHONE CAMERA 1 ─┐
                ├─ USB/UVC ─► LAPTOP ─► Stereo tracking + 3D trajectory prediction
PHONE CAMERA 2 ─┘                              │
                                               │ target <POS,x,y>
                                               │ Wi-Fi UDP, port 4210
                                               ▼
                                      ESP32 SKID-STEER ROBOT
                                      ├─ encoder odometry
                                      ├─ closed-loop motor speed control
                                      └─ turn + drive to target
```

The phones connect to the laptop as two separate USB camera devices. The laptop connects to the ESP32's `TrashBot` Wi-Fi access point. These are two independent connections: **USB for the cameras, Wi-Fi for the robot**.

The current chassis is **skid-steer**, not Mecanum or omni-wheel. It cannot strafe sideways. The firmware accepts a target in its odometry frame, turns toward it, and then drives forward or in reverse while correcting its heading.

## Module 1 — Stereo Vision & Trajectory Prediction (Laptop)

- `vision/cameras.py`: asynchronous capture and timestamp pairing for two camera sources. It accepts USB camera indices (for example, `0` and `1`) or compatible stream URLs.
- `vision/stereo_tracker.py`: stereo rectification, background subtraction, blob detection, stereo correspondence, depth triangulation and camera-to-world transformation.
- `vision/calibrate_stereo.py`: checkerboard stereo calibration.
- `trajectory/throw_filter.py`: filters incidental movement using blob area, height, speed, approach direction and free-fall curvature.
- `trajectory/trajectory_predictor.py`: fits the 3D projectile path using gravity-constrained least squares, rejects outliers and estimates the landing point and uncertainty.
- `trajectory/pipeline.py`: state machine that turns a valid prediction into target/home commands.
- `trajectory/calibration.py`: logs throws and fits an empirical landing correction from measured prediction errors.

No pretrained object detector is required by the current design; detection and trajectory logic are implemented with OpenCV and the project's own physics pipeline.

## Module 2 — Skid-Steer Locomotion & Interception (ESP32)

- `skid_steer_locomotion/skid_steer_locomotion.ino`: selected ESP32 firmware for the 4-wheel skid-steer robot, using two BTS7960 motor drivers and wheel encoders.
- The firmware implements encoder odometry, closed-loop PI wheel-speed control, acceleration limiting, position control, a manual web joystick, and the computer UDP interface.
- `trajectory/esp32_wifi_link.py`: laptop-side UDP client.

The laptop sends:
- `<POS,x,y>`: target position in metres relative to the last `<ZERO>` command; +x is robot-right and +y is robot-forward.
- `<HOME>`: return to the zero position.
- `<ZERO>`: make the current robot position and heading the odometry origin.
- `<STOP>`: stop the robot.
- `<PING>`: check communication.

The ESP32 replies with `<ACK,...>` and periodic `<ODO,x,y,vx,vy>` telemetry. The firmware's position controller performs the turning and driving; the laptop does not send Mecanum wheel speeds.

## Hardware and physical setup

Required for the current two-module target:

- Two phones capable of exposing their rear cameras to the laptop as camera devices over USB.
- Two USB data cables (charging-only cables will not work).
- A rigid side-by-side phone mount. The relative camera positions must not move after calibration.
- Laptop with Python, OpenCV and the project dependencies.
- Existing skid-steer robot with ESP32, BTS7960 drivers, encoders, motors and battery.
- Open-top basket mounted on the chassis.

Mount both phones in landscape orientation, at the same height and with their optical axes aimed at the same throwing zone. Keep the baseline and phone positions fixed. Avoid changing camera resolution, zoom, focus mode or orientation after stereo calibration.

### Phone-to-laptop connection

First connect each phone separately and confirm that the operating system exposes it as a camera.

**Android options:**
- If the phone supports USB webcam/UVC mode, connect with a data-capable USB cable and select `Webcam` under the phone's USB options. Not all Android models provide this feature.
- Otherwise, DroidCam documents USB operation using the phone app, desktop client, USB debugging and (on some Windows devices) phone drivers. On Linux, DroidCam notes that ADB is required.

**iPhone option:**
- Apple Continuity Camera supports a wired USB connection between a compatible iPhone and Mac. It requires device compatibility, Wi-Fi/Bluetooth and trusting the Mac. This is not a general iPhone-to-Windows USB webcam solution.

After the two phones are connected, enumerate the camera devices:

```bash
python -m vision.list_cameras --max-index 8
```

The tool prints each readable OpenCV camera index and its frame size. If the phones appear at indices `0` and `1`, the default configuration is already aligned. If they appear at different indices, edit `left` and `right` in `config/robot.json`.

If no camera devices appear, resolve the phone's USB webcam mode, USB data cable, desktop driver/client and operating-system camera permissions before attempting stereo calibration.

## Computer-to-robot connection

1. Power on the skid-steer robot.
2. Connect the laptop to the robot's `TrashBot` Wi-Fi access point (the firmware's default is `192.168.4.1`, UDP port `4210`).
3. Confirm the laptop is on the robot's network.
4. Run the safe communication check:

```bash
python esp32_wifi_link.py
```

This sends `<PING>` only and should receive `<ACK,PING>`. It does **not** move the robot by default. The optional `--move-test` performs a 30 cm forward-and-return test and should only be used with clear floor space and the robot secured for the test.

## Stereo calibration

Calibration must use the same camera pair, image resolution, orientation and rigid mounting that will be used during operation.

1. Print or display a checkerboard with **9 × 6 inner corners**. Measure the physical side length of one square accurately.
2. With both phones mounted and connected, capture 20–30 synchronized checkerboard pairs with the board at different positions, distances and tilts. Keep the board still for each saved pair.
3. Capture:

```bash
python -m vision.calibrate_stereo capture --left 0 --right 1
```

4. Solve (the example assumes 25 mm squares; replace this if yours differ):

```bash
python -m vision.calibrate_stereo solve --cols 9 --rows 6 --square 0.025 --out stereo_calib.npz
```

The tool reports stereo reprojection RMS and camera baseline. Aim for reprojection RMS below approximately 0.6 px, and check that the reported baseline agrees with the physical separation measured on the mount. Retake poor calibration pairs rather than accepting a bad calibration.

5. Measure the camera rig's height, downward pitch and world-frame position. Save them in `rig.json`, for example:

```json
{
  "cam_height_m": 1.0,
  "pitch_deg": 0.0,
  "cam_x_m": 0.0,
  "cam_y_m": 0.0
}
```

Replace the example values with measurements from the actual installation; do not assume that the example values describe the real rig.

## Coordinate-frame alignment (critical)

The trajectory engine outputs robot-relative targets: +x means right and +y means forward from the robot's zero position. The ESP32 position controller uses the same convention in its odometry frame.

At the start of a live session, the laptop sends `<ZERO>`. Therefore:

- Put the robot at the chosen home/start position before starting live mode.
- Point the robot's forward axis along the camera/world frame's +Y direction.
- Ensure the robot's right side corresponds to +X.
- Set `home_x` and `home_y` in `config/robot.json` to the robot's measured home position in the camera/world coordinate frame.
- Measure the camera height/pitch and configure `rig.json` before relying on 3D heights.

If the robot is rotated relative to the camera coordinate frame, or the home coordinates are wrong, a mathematically valid landing prediction can still produce an incorrect movement target. Confirm the axes and origin using slow, close-range tests before allowing autonomous movement.

## Configuration

The tracked `config/robot.json` defaults to USB camera indices `0` and `1`, 640×480 calibration dimensions, 30 FPS, and the skid-steer ESP32 at `192.168.4.1:4210`.

Edit the left/right indices, FPS, camera time offset, measured home coordinates, and calibration file names as needed. For two camera streams that do not expose USB webcam devices, the existing camera code can also accept stream URLs, but USB/UVC is the selected setup for this build.

## Bring-up sequence

Follow this order so problems can be isolated:

1. **Camera visibility:** run `python -m vision.list_cameras --max-index 8` and confirm both phones produce readable frames.
2. **Camera timing:** run `python -m vision.phones_check` after configuring the two sources. Confirm the frame rate is stable and determine any right-camera timestamp offset.
3. **Stereo geometry:** capture and solve the checkerboard calibration; verify RMS and baseline.
4. **Vision-only dry run:** with the robot powered but wheels clear or the robot otherwise secured, run live vision in dry-run mode:

```bash
python main.py --live --dry-run --show
```

Check that the object is detected in both views, that its 3D position is plausible, and that predicted landing coordinates make sense. Keep the throwing zone empty while the background model warms up.

5. **Robot communication:** run `python esp32_wifi_link.py` and confirm `<ACK,PING>`.
6. **Controlled motion check:** with a clear test area, verify robot zero, forward/reverse, turn direction and measured odometry. Start with a short target and keep the manual remote/physical stop available.
7. **End-to-end trial:** only after camera calibration, coordinate alignment and the motor link have been checked, run:

```bash
python main.py --live --show
```

Keep the first throws close, slow and repeatable. Record prediction-versus-actual landing results so the landing-correction model can be calibrated from real trials.

## Simulation and tests

Install requirements and run the automated suite:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python main.py --sim
python main.py --sim-vision
python demo_trajectory_plot.py
```

**Safety:** `--sim` and `--sim-vision` are dry-run by default and cannot command the physical robot. `--replay` is also dry-run unless `--realtime` is explicitly passed. Live mode is the only normal path that sends targets to the car, and `--live --dry-run` is the recommended first integration check.

## Current status

- **Module 1:** vision and trajectory code, synthetic tests, phone-camera capture support, and stereo calibration tools are present. Real USB phone compatibility, stereo calibration quality, and real-throw prediction accuracy still need to be measured on the team's physical setup.
- **Module 2:** skid-steer firmware and the laptop UDP client are present. The remote joystick has been tested by the team. The next validation target is the laptop-to-ESP32 UDP connection, followed by calibrated autonomous target movement.
- **Module 3:** deferred and excluded from the current scope.
