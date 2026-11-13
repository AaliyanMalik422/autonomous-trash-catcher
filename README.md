# Autonomous Trash-Catching Robot

Stereo cameras (USB webcams or smartphones) detect a thrown object, a physics model predicts where it will land, and an ESP32 robot drives to intercept and catch it in a basket.

```
cameras ─► vision/StereoTracker ─► trajectory/TrajectoryEngine ─► ESP32Link / WifiLink ─► ESP32 Robot
           (3D world packets)      (filter, predict, command)     (Serial / WiFi UDP)     (Closed-loop control)
```

## Repository Architecture

### Module 1: Stereo Vision & Trajectory Prediction (Laptop)
* **Vision Pipeline (`vision/`):**
  * `vision/stereo_tracker.py`: Rectification → OpenCV MOG2 background subtraction → morphology → contour circle-fitting → epipolar constraint check → disparity triangulation → camera-to-world transform.
  * `vision/cameras.py`: Threaded capture for USB webcams and phone IP cameras with network timeout handling, reconnects, and timestamp clock-smoothing.
  * `vision/calibrate_stereo.py`: Stereo calibration using checkerboard pattern (`capture`, `solve`).
  * `vision/preview.py`: Visual detector preview with 2D trajectory flight check.
  * `vision/check_throw_prediction.py`: Validates detection, timing, and projectile model on real throws.
  * `vision/synthetic.py`: Synthetic stereo image generator for full pipeline testing without physical cameras.
* **Trajectory Pipeline (`trajectory/`):**
  * `trajectory/throw_filter.py`: Filters out incidental motion (people walking, hand waving, noise) using blob area, height, velocity, and vertical free-fall curvature ($\ddot{Z} \approx -g$).
  * `trajectory/trajectory_predictor.py`: Gravity-constrained weighted least-squares fit ($Z(t) = z_0 + v_z t - \frac{1}{2} g t^2$, linear $X, Y$), MAD outlier rejection, descending root calculation at basket rim height, and 1-sigma uncertainty propagation.
  * `trajectory/pipeline.py`: `TrajectoryEngine` state machine (IDLE → TRACKING → ACTIVE → COOLDOWN), latency compensation, reachability checks, and return-home logic.
  * `trajectory/calibration.py`: Throw logger and empirical landing-correction polynomial fit.

### Module 2: Locomotion & Interception (ESP32)
* **Skid-Steer Locomotion (WiFi UDP / BTS7960) (`skid_steer_locomotion/`):**
  * `skid_steer_locomotion/skid_steer_locomotion.ino`: Firmware for 4-wheel skid-steer drive powered by 2x BTS7960 motor drivers with wheel encoders (JGB37-520). Features closed-loop PI speed control, odometry, position controller with dynamic braking curve, web joystick UI on port 80, and high-speed UDP command link on port 4210.
  * `trajectory/esp32_wifi_link.py` / `esp32_wifi_link.py`: UDP client link sending `<POS,x,y>`, `<HOME>`, `<STOP>`, `<ZERO>`, `<VEL,vx,vy,w>` and receiving `<ACK,...>` and `<ODO,x,y,vx,vy>` telemetry at 20 Hz.
* **USB serial transport (`trajectory/esp32_serial.py`):** The Python serial client is present, but the matching Mecanum firmware source (`mecanum_locomotion/mecanum_locomotion.ino`) is not in this checkout. Do not use the serial mode with a robot until matching firmware is added and validated.

---

## Layout

| Path | What it does |
|---|---|
| `vision/stereo_tracker.py` | Rectify → MOG2 background subtraction → contour → stereo triangulation → world-frame packet |
| `vision/cameras.py` | Threaded capture for USB webcams & phone streams (MJPEG/RTSP) |
| `vision/calibrate_stereo.py` | Checkerboard stereo calibration (`capture`, `solve`) |
| `vision/preview.py` | Live preview tool for camera calibration & single-camera throw verification |
| `vision/check_throw_prediction.py` | Checks projectile trajectory fits on recorded real throws |
| `vision/synthetic.py` | Renders synthetic stereo throw images to test vision path offline |
| `trajectory/throw_filter.py` | Rejects non-throw motion (people, hand waves, jitter) via speed & gravity curvature |
| `trajectory/trajectory_predictor.py` | Gravity-constrained least-squares fit → landing coordinates, impact time & sigma |
| `trajectory/pipeline.py` | `TrajectoryEngine`: orchestrates filtering, prediction, and robot commands |
| `trajectory/calibration.py` | Systematic landing correction model fitted from logged real throws |
| `trajectory/esp32_wifi_link.py` | WiFi UDP link for skid-steer car (TrashBot AP, port 4210) |
| `trajectory/esp32_serial.py` | USB serial transport client (115200 baud); matching firmware is currently absent |
| `trajectory/sim.py` | Closed-loop simulation measuring catch rate against robot acceleration |
| `skid_steer_locomotion/` | ESP32 firmware for skid-steer drive (2x BTS7960, encoders, WiFi UDP + Web UI) |
| `tests/firmware/` | Legacy C++ host harness; currently skipped because its Mecanum firmware source is absent |
| `tests/` | Python unit tests for vision, trajectory, transport selection, and camera settings |

---

## Coordinate Frame

* `X`: Right from cameras (+X, metres).
* `Y`: Forward depth from cameras (+Y, metres).
* `Z`: Upward from floor (+Z, metres).
* The robot's home position in the world frame is configured by `--home-x` and `--home-y` (default: `(0.0, 1.3)`).
* Commands to the robot are relative to its home position: `+x` right / strafe right, `+y` forward.

---

## Quick Start

### 1. Run Unit Tests & Simulations
```bash
pip install -r requirements.txt
python -m pytest tests                      # 34 tests covering vision, trajectory & links
# The legacy C++ firmware harness is skipped until mecanum_locomotion.ino is restored.
python demo_trajectory_plot.py              # 3D trajectory visualization plot
python main.py --sim                        # dry-run simulation of synthetic throws
python main.py --sim-vision                 # synthetic stereo images through full vision pipeline
```

### 2. Skid-Steer Car Bring-Up (WiFi)
1. Flash `skid_steer_locomotion/skid_steer_locomotion.ino` to the ESP32 using Arduino IDE.
2. Connect your laptop to the car's WiFi access point:
   * **SSID:** `TrashBot`
   * **Password:** `12345678`
3. Verify basic WiFi motion link:
   ```bash
   python esp32_wifi_link.py
   ```
   *(The car should answer `<ACK,PING>` and drive forward 30 cm then return).*
4. Run end-to-end with WiFi link:
   ```bash
   python main.py --live --wifi --calib stereo_calib.npz --rig rig.json
   ```

### 3. USB Serial Transport

The Python serial client (`trajectory/esp32_serial.py`) is retained, but this checkout does not contain its matching Mecanum firmware. Serial robot operation is therefore not a supported bring-up path until that firmware is restored and tested.

### 4. Camera Calibration (Webcams or Phones)
* **For 2 USB Webcams:**
  ```bash
  python -m vision.calibrate_stereo capture --left 0 --right 1
  python -m vision.calibrate_stereo solve --square 0.025 --out stereo_calib.npz
  ```
* **For 2 Phone Streams (IP Webcam / DroidCam):**
  Configure URLs in `config/robot.json`:
  ```json
  {
    "left": "http://192.168.4.2:8080/video",
    "right": "http://192.168.4.3:8080/video",
    "wifi": true
  }
  ```
  Check stream quality:
  ```bash
  python -m vision.phones_check
  ```

---

## Status
* **Module 1 (Stereo Vision & Trajectory Prediction):** Core code and synthetic tests are present. Real-camera calibration and real-throw validation still depend on the physical setup.
* **Module 2 (Locomotion & Interception):** Skid-steer ESP32 firmware and its Wi-Fi Python link are present. Physical robot validation is not demonstrated by the committed automated tests. The serial client exists, but its matching Mecanum firmware is absent.
* **Module 3 (Payload Sensing & Auto-Emptying):** Not implemented in this checkout.
