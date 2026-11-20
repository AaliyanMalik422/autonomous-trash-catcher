[![Tests](https://github.com/AaliyanMalik422/autonomous-trash-catcher/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/AaliyanMalik422/autonomous-trash-catcher/actions/workflows/tests.yml)

# Autonomous Trash Catcher

An engineering prototype for detecting thrown waste, estimating its landing position, and positioning a skid-steer robot to intercept it.

The system uses phone cameras connected to a laptop for vision processing. The laptop estimates the object's motion and landing point, then communicates a target position to an ESP32-based robot over Wi-Fi.

## System overview

~~~text
Phone cameras
     │  USB / camera stream
     ▼
Laptop
  ├─ Camera capture and synchronization
  ├─ Stereo vision and object tracking
  ├─ 3D trajectory estimation
  └─ Landing-point calculation
     │  Wi-Fi / UDP
     ▼
ESP32 controller
     │
     ▼
Skid-steer drive platform and collection basket
~~~

The intended operating sequence is:

1. Capture synchronized images of an object in flight.
2. Estimate its 3D position from the calibrated camera pair.
3. determine whether the observed motion is consistent with a throw.
4. Fit the flight path and estimate the landing point.
5. Send a position target to the robot controller.
6. Turn and drive the skid-steer platform toward the target.

## Capabilities

- Capture from USB cameras and compatible phone-camera streams.
- Stereo calibration, image rectification, and timestamp alignment.
- Motion-based object detection and stereo correspondence.
- 3D position reconstruction and gravity-constrained trajectory fitting.
- Throw filtering and landing-point estimation.
- ESP32 communication over UDP, including position, home, stop, and status messages.
- Simulation and dry-run modes for testing without driving the robot.

## Hardware

The current prototype is designed around:

- Two phones with cameras that can be exposed to the laptop as camera devices.
- A rigid mount that keeps the cameras fixed relative to one another.
- A laptop running Python and OpenCV.
- An ESP32-based skid-steer chassis with motor drivers and wheel encoders.
- A collection basket mounted on the chassis.

The two cameras connect to the laptop independently of the robot link. Camera data is carried over USB or a supported camera stream; robot commands are sent over Wi-Fi.

The chassis is skid-steer. It turns by driving its left and right sides at different speeds and cannot translate sideways like an omni-directional platform.

## Software requirements

- Python 3.11 or 3.14 (both versions are covered by the automated test workflow).
- OpenCV and NumPy.
- PySerial for serial-connected hardware tools.
- Matplotlib for plots.
- Pytest for the automated test suite.

## Installation

Clone the repository and create an isolated Python environment:

~~~bash
git clone https://github.com/AaliyanMalik422/autonomous-trash-catcher.git
cd autonomous-trash-catcher

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

On Ubuntu, install the virtual-environment package first if the venv command is unavailable:

~~~bash
sudo apt install python3-venv
~~~

## Quick start

Run the software simulations:

~~~bash
# Simulated throws through the trajectory engine
python main.py --sim

# Synthetic camera images through the vision pipeline
python main.py --sim-vision
~~~

Run the automated tests:

~~~bash
python -m pytest -q tests
~~~

Simulation and test commands do not require the robot to be connected.

## Camera configuration

Edit **config/robot.json** to set the camera sources and operating parameters. For USB cameras, sources are normally OpenCV device indexes such as 0 and 1. Compatible stream URLs can also be used.

List available camera devices:

~~~bash
python -m vision.list_cameras --max-index 8
~~~

The camera pair must remain rigidly mounted after calibration. Changing the camera spacing, orientation, resolution, or focus can invalidate the calibration.

## Stereo calibration

Stereo depth requires a calibration captured from the actual camera mount.

1. Print a checkerboard with 9 × 6 inner corners.
2. Measure the physical side length of one square.
3. Configure the left and right camera sources in config/robot.json.
4. For each phone mounted upside down, set its corresponding `left_rotate_180` or
   `right_rotate_180` option to `true`. Keep the same setting for calibration and live operation.
5. Capture 20–30 synchronized pairs with the board at varied positions and angles.

~~~bash
python -m vision.calibrate_stereo capture --left 0 --right 1 --out calib_images
~~~

Solve the calibration. Replace 0.025 with the measured square size in metres:

~~~bash
python -m vision.calibrate_stereo solve --images calib_images --cols 9 --rows 6 --square 0.025 --out stereo_calib.npz
~~~

Review the reported reprojection error and baseline. The baseline should agree with the physical camera separation. A low reprojection error alone does not prove that the camera rig or world-coordinate measurements are correct.

Create **rig.json** with measured camera geometry:

~~~json
{
  "cam_height_m": 1.0,
  "pitch_deg": 0.0,
  "yaw_deg": 0.0,
  "cam_x_m": 0.0,
  "cam_y_m": 0.0
}
~~~

These values are examples only. Choose one shared robot-aligned world frame for both the camera pose and the robot home coordinates. The default convention places the left camera lens at X=0, Y=0, so cam_x_m and cam_y_m are normally zero; set home_x/home_y in config/robot.json to the robot's ZERO location measured in that same frame (the defaults are examples, not measured values). yaw_deg is the camera rig's heading offset from robot forward (positive points right); pitch_deg is downward tilt. The current model assumes the mount has no roll. Live mode refuses to start without a valid measured rig file.

## Live vision

After both cameras are configured and a valid stereo calibration and rig configuration are available, run vision without sending movement commands:

~~~bash
python main.py --live --dry-run --show
~~~

The dry-run mode displays live perception and prints the commands that would be sent. It does not command the robot to move.

## Robot interface

The laptop communicates with the ESP32 over UDP. The default controller address is **192.168.4.1**, port **4210**. The laptop must be connected to the robot's Wi-Fi access point before hardware communication is attempted.

The interface uses the following commands:

| Command | Purpose |
|---|---|
| POS | Set a target position in the robot coordinate frame |
| ZERO | Set the current position as the odometry origin |
| HOME | Return to the configured home position |
| STOP | Stop the drive system |
| PING | Check communication with the controller |

The ESP32 reports acknowledgements and odometry data. Live mode requires a fresh PING acknowledgement, a ZERO acknowledgement, and continuing fresh odometry telemetry. It will not re-arm a new throw until odometry confirms that the robot has returned to the ZERO origin and is stationary. The robot's coordinate convention and the camera/world coordinate frame must be aligned before autonomous operation.

Only run live mode without **--dry-run** after checking the camera calibration, measured camera rig pose, coordinate-frame alignment, firmware, motor direction, encoder signs and scale, odometry, and emergency stop. Begin with the robot secured or with the drive wheels raised. Keep a clear test area and an independent means of stopping the motors.

## Validation status

The repository includes automated tests for the trajectory engine, camera timing, stereo geometry, and simulation paths. GitHub Actions runs the test suite on Python 3.11.

Physical performance depends on the installed camera rig and robot hardware. Stereo depth accuracy, real-throw landing prediction, and autonomous interception must be validated on the assembled prototype before being treated as operational.

## Limitations

- A single camera provides image-plane position only; it cannot determine metric depth by itself.
- Stereo reconstruction depends on a fixed, correctly calibrated camera pair.
- Motion-based detection is designed to identify objects moving against a relatively stable background. Stationary objects are not continuously detected by this method.
- Fast motion blur, occlusion, lighting changes, and background movement can reduce detection continuity.
- A skid-steer robot cannot move directly sideways; the controller must turn before driving toward a lateral target.

## Repository contents

The repository contains the vision and calibration pipeline, trajectory estimation and filtering code, robot communication interface, ESP32 firmware, configuration files, simulation tools, and automated tests.

