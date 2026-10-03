# Autonomous Trash-Catching Robot (Module 1: Vision & Trajectory)

Software pipeline for an autonomous robot that detects thrown trash in 3D using stereo webcams, predicts its landing coordinates using projectile physics, and sends target `(X, Y)` coordinates to an ESP32 omni-wheel chassis over serial.

## Repository Structure

* **`vision/` (Part 1 — Stereo Vision & 3D Tracking)**
  * Captures synchronized frames from dual USB webcams.
  * Runs background subtraction and contour detection to locate the thrown object in 2D.
  * Uses stereo calibration and disparity triangulation to output real-world 3D coordinates `(X, Y, Z)` in meters every frame.

* **`trajectory/` (Part 2 — Filtering, Physics Prediction & Serial Comms)**
  * `mock_vision.py`: Generates synthetic 60 FPS 3D throws with Gaussian camera noise for testing without hardware.
  * `throw_filter.py`: Filters out non-throws (e.g., people walking by) using zone bounds, blob size, speed, and parabolic arc checks.
  * `trajectory_predictor.py`: Fits linear horizontal equations and a gravity-constrained vertical parabola ($g = 9.81\text{ m/s}^2$) to predict where the object will intersect the basket height, refining the prediction mid-air.
  * `esp32_serial.py`: Sends formatted `<X, Y>` target packets to the ESP32 over UART (`115200` baud).

* **`main.py`**: Connects `vision/` output directly into `trajectory/`.
* **`logs/`**: Stores predicted vs. actual landing data for empirical calibration.

## Shared Data Interface (`vision` -> `trajectory`)

Each frame from `vision/` passes a Python dictionary in this format to `trajectory/`:

    {
        "timestamp": 0.165,   # seconds elapsed since start
        "detected": True,     # True if object is visible in both cameras
        "X": 0.32,            # Left/Right position (meters)
        "Y": 2.15,            # Forward depth from cameras (meters)
        "Z": 1.40,            # Height off the floor (meters)
        "blob_area": 380.0    # Contour pixel area for size filtering
    }

## Quick Start

1. Install dependencies:

       pip install -r requirements.txt

2. Run the 3D trajectory simulation and visualizer:

       python trajectory/trajectory_predictor.py
