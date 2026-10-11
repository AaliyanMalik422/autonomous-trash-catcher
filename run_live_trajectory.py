import cv2
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from trajectory.throw_filter import ThrowFilter
from trajectory.trajectory_predictor import TrajectoryPredictor

class WebcamTracker:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.start_time = time.time()
        self.KNOWN_BALL_RADIUS_M = 0.033  # Standard ball radius in meters
        self.FOCAL_LENGTH_PX = 600.0      # Standard focal length estimate

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # HSV range for neon green / yellow
        lower_bound = (25, 40, 40)
        upper_bound = (55, 255, 255)

        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        packet = {
            "timestamp": time.time() - self.start_time,
            "detected": False,
            "X": 0.0, "Y": 0.0, "Z": 0.0, "blob_area": 0.0
        }

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            if radius > 12:
                # 3D monocular depth calculation
                depth_y = (self.KNOWN_BALL_RADIUS_M * self.FOCAL_LENGTH_PX) / radius
                pos_x = ((x - 320.0) * depth_y) / self.FOCAL_LENGTH_PX
                pos_z = ((240.0 - y) * depth_y) / self.FOCAL_LENGTH_PX

                packet.update({
                    "detected": True,
                    "X": pos_x,
                    "Y": depth_y,
                    "Z": pos_z,
                    "blob_area": float(cv2.contourArea(c))
                })

                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

        cv2.putText(frame, f"Depth (Y): {packet['Y']:.2f}m | Z: {packet['Z']:.2f}m", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return packet, frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

def plot_intercept(tracked_points, fit_data):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    obs_x = [p["X"] for p in tracked_points]
    obs_y = [p["Y"] for p in tracked_points]
    obs_z = [p["Z"] for p in tracked_points]

    ax.scatter(obs_x, obs_y, obs_z, color='blue', s=40, label='Camera Observations')
    ax.plot(fit_data["curve_x"], fit_data["curve_y"], fit_data["curve_z"], 'r--', linewidth=2, label='Parabolic Fit')
    ax.scatter([fit_data["pred_x"]], [fit_data["pred_y"]], [fit_data["pred_z"]], color='green', s=150, marker='*', label='Intercept Point')

    ax.set_title("Real-Time Calculated Trajectory Intercept", fontsize=12, fontweight='bold')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.tight_layout()
    plt.show()

def main():
    tracker = WebcamTracker(camera_index=0)
    throw_filter = ThrowFilter()
    predictor = TrajectoryPredictor(target_z=0.35)

    print("=" * 60)
    print(" LIVE PERCEPTION & TRAJECTORY PREDICTION ACTIVE")
    print(" Press 'q' in the camera window to exit.")
    print("=" * 60)

    try:
        while True:
            packet, frame = tracker.read_frame()
            if packet is None:
                break

            valid, buffer = throw_filter.process_frame(packet)

            if valid and len(buffer) >= 6:
                res = predictor.predict_landing(buffer)
                if res is not None:
                    pred_x, pred_y, t_impact, fit_data = res
                    print("\n" + "#" * 50)
                    print(f" [TARGET INTERCEPT COMPUTED]")
                    print(f"  -> Predicted Landing: X = {pred_x:.3f} m, Y = {pred_y:.3f} m")
                    print(f"  -> Intercept Height:  Z = {fit_data['pred_z']:.2f} m")
                    print(f"  -> Time to Impact:    {t_impact:.3f} s")
                    print(f"  -> Mecanum Target:    <POS, {pred_x:.2f}, {pred_y:.2f}>")
                    print("#" * 50 + "\n")

                    tracker.release()
                    plot_intercept(buffer, fit_data)
                    break

            cv2.imshow("Robot Tracking Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        tracker.release()

if __name__ == "__main__":
    main()
