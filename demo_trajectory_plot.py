import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from trajectory.mock_vision import MockVision
from trajectory.throw_filter import ThrowFilter
from trajectory.trajectory_predictor import TrajectoryPredictor

def run_submission_demo():
    print("="*60)
    print(" SYSTEM BOOT: AUTONOMOUS TRAJECTORY ENGINE")
    print("="*60)

    # Initialize your actual project modules
    mock_camera = MockVision()
    throw_filter = ThrowFilter()
    predictor = TrajectoryPredictor(target_z=0.35) 

    tracked_points = []
    final_prediction = None

    print("[SYSTEM] Receiving simulated 3D spatial data stream...")

    # Feed the data through your actual filter and predictor
    for frame in mock_camera.generate_throw_frames():
        valid, buffer = throw_filter.process_frame(frame)
        
        if valid:
            tracked_points.append(frame)
            
            # Once we have enough valid points, solve the physics
            if len(tracked_points) >= 6 and final_prediction is None:
                res = predictor.predict_landing(tracked_points)
                if res is not None:
                    pred_x, pred_y, t_rem, fit_data = res
                    final_prediction = fit_data
                    
                    print(f"\n[URGENT] INBOUND PROJECTILE DETECTED")
                    print(f" -> Predicted Landing: X = {pred_x:.3f}m, Y = {pred_y:.3f}m")
                    print(f" -> Basket Intercept Height: Z = {fit_data['pred_z']:.2f}m")
                    print(f" -> Estimated Impact Time: {t_rem:.3f} sec")
                    print(f" -> Transmission Packet: <POS,{pred_x:.2f},{pred_y:.2f}>\n")

    # Generate the 3D prototype plot
    if final_prediction is not None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        obs_x = [p["X"] for p in tracked_points]
        obs_y = [p["Y"] for p in tracked_points]
        obs_z = [p["Z"] for p in tracked_points]

        ax.scatter(obs_x, obs_y, obs_z, color='blue', s=45, label='Filtered Camera Data')
        ax.plot(final_prediction["curve_x"], final_prediction["curve_y"], final_prediction["curve_z"], 
                'r--', linewidth=2.5, label='Gravity-Constrained Fit')
        ax.scatter([final_prediction["pred_x"]], [final_prediction["pred_y"]], [final_prediction["pred_z"]], 
                   color='green', s=160, marker='*', label='Intercept Target')

        ax.set_title("Autonomous Ball Trajectory Prediction", fontweight='bold')
        ax.set_xlabel("X - Lateral (m)")
        ax.set_ylabel("Y - Depth (m)")
        ax.set_zlabel("Z - Height (m)")
        ax.legend()
        plt.show()
    else:
        print("[ERROR] Trajectory prediction failed.")

if __name__ == "__main__":
    run_submission_demo()