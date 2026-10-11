import numpy as np

class TrajectoryPredictor:
    def __init__(self, target_z=0.35, g=9.81):
        """
        target_z: Height of the basket rim in meters (e.g. 0.35m off ground).
        g: Gravitational acceleration (m/s^2).
        """
        self.target_z = target_z
        self.g = g

    def predict_landing(self, points):
        """
        Given a list of dicts with 'timestamp', 'X', 'Y', 'Z',
        fits linear models for X and Y vs time, and a parabolic model for Z vs time.
        Returns: (predicted_x, predicted_y, time_to_impact, fit_points_dict)
        """
        if len(points) < 3:
            return None

        t = np.array([p["timestamp"] for p in points])
        x = np.array([p["X"] for p in points])
        y = np.array([p["Y"] for p in points])
        z = np.array([p["Z"] for p in points])

        # Normalize time so t0 = 0 to avoid large floating point errors
        t0 = t[0]
        t_rel = t - t0

        # Linear fits for horizontal motion: X(t) = vx * t + x0, Y(t) = vy * t + y0
        vx, x0 = np.polyfit(t_rel, x, 1)
        vy, y0 = np.polyfit(t_rel, y, 1)

        # Constrained vertical fit with known gravity:
        # Z(t) = -0.5*g*t^2 + vz0*t + z0  =>  Z(t) + 0.5*g*t^2 = vz0*t + z0
        z_adj = z + 0.5 * self.g * (t_rel ** 2)
        vz0, z0 = np.polyfit(t_rel, z_adj, 1)

        # Solve for impact time at target_z:
        # -0.5*g*t^2 + vz0*t + (z0 - target_z) = 0
        a = -0.5 * self.g
        b = vz0
        c = z0 - self.target_z

        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return None  # Ball trajectory peak does not reach basket height

        # Select the forward-in-time root
        t_impact_1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t_impact_2 = (-b - np.sqrt(discriminant)) / (2 * a)
        future_roots = [r for r in (t_impact_1, t_impact_2) if r > t_rel[-1]]

        if not future_roots:
            return None

        t_impact = min(future_roots)
        pred_x = vx * t_impact + x0
        pred_y = vy * t_impact + y0
        time_to_impact = t_impact - t_rel[-1]

        # Generate smooth curve for 3D plotting
        t_dense = np.linspace(0, t_impact, 50)
        curve_x = vx * t_dense + x0
        curve_y = vy * t_dense + y0
        curve_z = -0.5 * self.g * (t_dense ** 2) + vz0 * t_dense + z0

        fit_data = {
            "curve_x": curve_x,
            "curve_y": curve_y,
            "curve_z": curve_z,
            "pred_x": pred_x,
            "pred_y": pred_y,
            "pred_z": self.target_z,
            "time_to_impact": time_to_impact
        }

        return pred_x, pred_y, time_to_impact, fit_data

if __name__ == "__main__":
    predictor = TrajectoryPredictor()
    print("TrajectoryPredictor loaded successfully.")
