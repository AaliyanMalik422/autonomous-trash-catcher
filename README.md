# Autonomous Trash-Catching Robot

Software and hardware pipeline for an autonomous robot that detects thrown objects in 3D using stereo webcams, predicts landing coordinates using projectile physics, and sends target `(X, Y)` coordinates to an ESP32 omni-wheel chassis over serial.

## Repository Structure

*   **`vision/` (Part 1 — Stereo Vision & 3D Tracking)**
    *   Captures synchronized frames from dual USB webcams.
    *   Runs background subtraction and contour detection to locate the thrown object in 2D.
    *   Uses stereo calibration and disparity triangulation to output real-world 3D coordinates `(X, Y, Z)` in meters every frame.
*   **`trajectory/` (Part 2 — Filtering, Physics Prediction & Serial Comms)**
    *   `mock_vision.py`: Generates synthetic 60 FPS 3D throws with Gaussian camera noise for testing without hardware.
    *   `throw_filter.py`: Filters out non-throws (e.g., people walking by) using zone bounds, blob size, speed, and parabolic arc checks.
    *   `trajectory_predictor.py`: Fits linear horizontal equations and a gravity-constrained vertical parabola ($g = 9.81\text{ m/s}^2$) to predict where the object will intersect the basket height, refining the prediction mid-air.
    *   `esp32_serial.py`: Sends formatted `<POS, X, Y>` target packets to the ESP32 over UART (`115200` baud).
*   **`mecanum_locomotion/` (Part 3 — Hardware Execution)**
    *   ESP32 C++ firmware (`mecanum_locomotion.ino`).
    *   Executes a 200 Hz closed-loop PI velocity controller and inverse kinematics for a 4-wheel Mecanum drive system based on serial target commands.

## Root Execution Scripts

*   **`main.py`**: The primary orchestrator. Connects `vision/` output directly into `trajectory/` and transmits data to the hardware.
*   **`demo_trajectory_plot.py`**: Runs the complete software prototype pipeline using simulated 3D data to generate real-time Matplotlib visualizations of the physical intercept calculation.

## Shared Data Interface (`vision` -> `trajectory`)

Each frame from `vision/` passes a Python dictionary in this format to `trajectory/`:

```python
{
    "timestamp": 0.165,   # seconds elapsed since start
    "detected": True,     # True if object is visible in both cameras
    "X": 0.32,            # Left/Right position (meters)
    "Y": 2.15,            # Forward depth from cameras (meters)
    "Z": 1.40,            # Height off the floor (meters)
    "blob_area": 380.0    # Contour pixel area for size filtering
}
Quick Start
1. Clone and Install Dependencies

Bash
git clone [https://github.com/AaliyanMalik422/autonomous-trash-catcher.git](https://github.com/AaliyanMalik422/autonomous-trash-catcher.git)
cd autonomous-trash-catcher
pip install -r requirements.txt
2. Run the 3D Trajectory Simulation
To test the math and filtering pipeline visually without physical hardware connected:

Bash
python demo_trajectory_plot.py
3. Run the Live Production System
To run the full pipeline with stereo cameras and the ESP32 connected:

Bash
python main.py
