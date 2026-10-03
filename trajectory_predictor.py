import numpy as np
import matplotlib.pyplot as plt
from mock_vision import MockStereoCamera

class TrajectoryPredictor:
    def __init__(self, basket_height=0.35, min_frames=6):
        self.basket_height = basket_height
        self.min_frames = min_frames
        self.g = 9.81
        self.buffer = []
        self.calib_ax, self.calib_bx = 1.0, 0.0
        self.calib_ay, self.calib_by = 1.0, 0.0

    def reset(self):
        self.buffer.clear()

    def add_frame_and_predict(self, packet):
        if not packet.get("detected", False):
            return None

        self.buffer.append(packet)
        if len(self.buffer) < self.min_frames:
            return None

        t0 = self.buffer[0]["timestamp"]
        t = np.array([p["timestamp"] - t0 for p in self.buffer])
        X = np.array([p["X"] for p in self.buffer])
        Y = np.array([p["Y"] for p in self.buffer])
        Z = np.array([p["Z"] for p in self.buffer])

        vx, x0 = np.polyfit(t, X, 1)
        vy, y0 = np.polyfit(t, Y, 1)

        Z_gravity_adjusted = Z + 0.5 * self.g * (t ** 2)
        vz, z0 = np.polyfit(t, Z_gravity_adjusted, 1)

        a = 0.5 * self.g
        b = -vz
        c = self.basket_height - z0
        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return None

        t_land = (-b + np.sqrt(discriminant)) / (2 * a)
        time_remaining = t_land - t[-1]

        if time_remaining <= 0:
            return None

        x_pred = x0 + vx * t_land
        y_pred = y0 + vy * t_land

        x_final = self.calib_ax * x_pred + self.calib_bx
        y_final = self.calib_ay * y_pred + self.calib_by

        return {
            "frames_used": len(self.buffer),
            "X_target": round(float(x_final), 3),
            "Y_target": round(float(y_final), 3),
            "Z_target": self.basket_height,
            "time_remaining_sec": round(float(time_remaining), 3),
            "t_land": float(t_land),
            "params": (x0, y0, z0, vx, vy, vz)
        }

def run_simulation_and_plot():
    camera = MockStereoCamera(fps=60, noise_std_meters=0.025)
    predictor = TrajectoryPredictor(basket_height=0.35, min_frames=8)

    packets, true_landing = camera.generate_throw()

    early_prediction = None
    for packet in packets:
        pred = predictor.add_frame_and_predict(packet)
        if pred and early_prediction is None:
            early_prediction = pred

    err_x_cm = abs(early_prediction["X_target"] - true_landing["X_land"]) * 100
    err_y_cm = abs(early_prediction["Y_target"] - true_landing["Y_land"]) * 100
    total_err_cm = np.hypot(err_x_cm, err_y_cm)

    print("\n================ 3D TRAJECTORY PREDICTION TEST ================")
    print(f"Frames used for prediction : First {early_prediction['frames_used']} frames (0.13s of flight)")
    print(f"Time left for robot to move: {early_prediction['time_remaining_sec']} seconds")
    print(f"TRUE Landing Coordinates   : X = {true_landing['X_land']:.3f} m, Y = {true_landing['Y_land']:.3f} m")
    print(f"PREDICTED Coordinates      : X = {early_prediction['X_target']:.3f} m, Y = {early_prediction['Y_target']:.3f} m")
    print(f"Prediction Error           : {total_err_cm:.2f} cm (even with 2.5cm camera noise!)")
    print(f"ESP32 Serial Packet Sent   : <X:{early_prediction['X_target']:.2f},Y:{early_prediction['Y_target']:.2f}>")
    print("===============================================================\n")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    true_X = [p["true_X"] for p in packets]
    true_Y = [p["true_Y"] for p in packets]
    true_Z = [p["true_Z"] for p in packets]
    ax.plot(true_X, true_Y, true_Z, color="gray", alpha=0.5, label="Actual True Flight Path")

    used_packets = packets[:early_prediction["frames_used"]]
    ax.scatter(
        [p["X"] for p in used_packets],
        [p["Y"] for p in used_packets],
        [p["Z"] for p in used_packets],
        color="blue", s=50, label=f"First {len(used_packets)} Noisy Camera Points (Input)"
    )

    x0, y0, z0, vx, vy, vz = early_prediction["params"]
    t_curve = np.linspace(0, early_prediction["t_land"], 50)
    ax.plot(
        x0 + vx * t_curve,
        y0 + vy * t_curve,
        z0 + vz * t_curve - 0.5 * 9.81 * (t_curve ** 2),
        "g--", linewidth=2.5, label="Predicted Trajectory Curve"
    )

    ax.scatter(true_landing["X_land"], true_landing["Y_land"], true_landing["Z_land"],
               color="black", marker="o", s=120, label="Actual Basket Intersection")
    ax.scatter(early_prediction["X_target"], early_prediction["Y_target"], early_prediction["Z_target"],
               color="red", marker="*", s=220, label="Predicted Target Sent to ESP32")

    ax.set_title("Autonomous Trash Catcher — 3D Trajectory Prediction (8 Frames)")
    ax.set_xlabel("X (Left/Right Meters)")
    ax.set_ylabel("Y (Depth Distance Meters)")
    ax.set_zlabel("Z (Height Meters)")
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation_and_plot()
