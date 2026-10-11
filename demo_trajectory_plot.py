import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from trajectory.mock_vision import MockVision
from trajectory.throw_filter import ThrowFilter
from trajectory.trajectory_predictor import TrajectoryPredictor

def run_trajectory_demo():
    mock = MockVision()
    throw_filter = ThrowFilter()
    predictor = TrajectoryPredictor(target_z=0.35)

    print("="*60)
    print(" AUTONOMOUS ROBOT: TRAJECTORY PREDICTION & INTERCEPT DEMO")
    print("="*60)

    tracked_points = []
    final_prediction = None

    # Process simulated vision stream
    for frame in mock.generate_throw_frames():
        valid, buffer = throw_filter.process_frame(frame)
        if valid:
            tracked_points.append(frame)
            # Predict once we have sufficient observation points (6+ frames)
            if len(tracked_points) >= 6 and final_prediction is None:
                res = predictor.predict_landing(tracked_points)
                if res is not None:
                    pred_x, pred_y, t_rem, fit_data = res
                    final_prediction = fit_data
                    print(f"\n[TARGET LOCKED] Inbound Projectile Confirmed!")
                    print(f" -> Predicted Impact Coordinates: X = {pred_x:.3f} m, Y = {pred_y:.3f} m")
                    print(f" -> Basket Intercept Height:    Z = {fit_data['pred_z']:.2f} m")
                    print(f" -> Estimated Time to Intercept: {t_rem:.3f} seconds")
                    print(f" -> Transmitting to ESP32:      <POS,{pred_x:.2f},{pred_y:.2f}>\n")

    # Render 3D Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Observed airborne points
    obs_x = [p["X"] for p in tracked_points]
    obs_y = [p["Y"] for p in tracked_points]
    obs_z = [p["Z"] for p in tracked_points]
    ax.scatter(obs_x, obs_y, obs_z, color='blue', s=45, label='Camera Observations (Stereo)')

    if final_prediction is not None:
        # 2. Fitted physical parabolic trajectory
        ax.plot(final_prediction["curve_x"], 
                final_prediction["curve_y"], 
                final_prediction["curve_z"], 
                'r--', linewidth=2.5, label='Constrained Parabolic Fit')

        # 3. Intercept landing target coordinate
        ax.scatter([final_prediction["pred_x"]], 
                   [final_prediction["pred_y"]], 
                   [final_prediction["pred_z"]], 
                   color='green', s=160, marker='*', label='Predicted Landing Target')

        # 4. Robot catch zone plane
        ax.scatter([final_prediction["pred_x"]], 
                   [final_prediction["pred_y"]], 
                   [0.0], 
                   color='purple', s=80, marker='s', label='Chassis Base Target (Floor)')

    ax.set_title("Autonomous Ball Trajectory & Intercept Prediction", fontsize=13, fontweight='bold')
    ax.set_xlabel("X - Lateral (meters)")
    ax.set_ylabel("Y - Depth Distance (meters)")
    ax.set_zlabel("Z - Height (meters)")
    ax.legend(loc='upper right')
    ax.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_trajectory_demo()
