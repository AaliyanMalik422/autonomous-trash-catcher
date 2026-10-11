# Autonomous Trash-Catching Robot

## Project Overview
This project is an autonomous robotics system designed to visually track a thrown object (projectile), calculate its physical trajectory in real time, and dynamically drive a catching basket to the predicted landing coordinate using a Mecanum drivetrain.

## Current Phase: Software & Math Prototype (Hardware-in-the-Loop)
Due to local supply chain delays for specific drivetrain components (12V 400 RPM JGA25-370 metal gear motors and 80mm Mecanum wheels), the current prototype submission focuses on the completed **Perception and Prediction Pipeline**. 

The system successfully demonstrates:
1. **Spatial Filtering (`throw_filter.py`):** Rejects visual noise and isolates valid airborne objects moving toward the robot.
2. **Trajectory Prediction (`trajectory_predictor.py`):** Uses mathematical kinematics to fit a gravity-constrained parabola ($Z(t) = Z_0 + V_{z0}t - \frac{1}{2}gt^2$) to calculate the precise $(X, Y)$ floor intercept.
3. **Data Visualization (`demo_trajectory_plot.py`):** Real-time 3D Matplotlib rendering of the predicted flight path and target intercept.

## System Architecture
* **Perception & Compute (Laptop):** Python, OpenCV, NumPy, Matplotlib.
* **Communication Bridge:** USB Serial transmitting `<POS, X, Y>` coordinates at 115200 baud.
* **Microcontroller:** ESP32 handling inverse kinematics for omnidirectional movement.
* **Actuation (Pending Integration):** 2x L298N Motor Drivers powered by a custom 11.1V 3S 21700 battery pack, driving 4x DC motors.

## Next Milestones
* Integrate dual-camera stereo vision for physical 3D depth acquisition.
* Assemble MDF chassis and mount drivetrain upon hardware arrival.
* Flash ESP32 Mecanum kinematic mixing matrix.
