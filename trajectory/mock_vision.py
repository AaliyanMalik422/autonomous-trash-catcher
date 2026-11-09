"""Synthetic stereo-camera data for testing without hardware."""
import numpy as np


class MockStereoCamera:
    def __init__(self, fps=60, noise_std_meters=0.025, seed=42,
                 dropout_prob=0.0, outlier_prob=0.0, t0=0.0):
        self.fps = fps
        self.dt = 1.0 / fps
        self.noise_std = noise_std_meters
        self.g = 9.81
        self.dropout_prob = dropout_prob
        self.outlier_prob = outlier_prob
        self.t0 = t0
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ throws
    def generate_throw(self, start_pos=(0.6, 2.6, 1.3), velocity=(-0.4, -1.8, 2.2),
                       basket_height=0.35):
        x0, y0, z0 = start_pos
        vx, vy, vz = velocity

        a = 0.5 * self.g
        b = -vz
        c = basket_height - z0
        t_land_true = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

        true_landing = {
            "t_land": t_land_true + self.t0,
            "X_land": x0 + vx * t_land_true,
            "Y_land": y0 + vy * t_land_true,
            "Z_land": basket_height,
        }

        packets = []
        k = 0
        while True:
            t = k * self.dt
            k += 1
            true_x = x0 + vx * t
            true_y = y0 + vy * t
            true_z = z0 + vz * t - 0.5 * self.g * t * t
            if true_z < 0.0:
                break
            if self.rng.random() < self.dropout_prob:
                packets.append({"timestamp": round(t + self.t0, 4), "detected": False,
                                "X": 0.0, "Y": 0.0, "Z": 0.0, "blob_area": 0.0})
                continue
            nx, ny, nz = self.rng.normal(0, self.noise_std, 3)
            if self.rng.random() < self.outlier_prob:
                nx += self.rng.choice([-1, 1]) * self.rng.uniform(0.3, 0.8)
                nz += self.rng.choice([-1, 1]) * self.rng.uniform(0.3, 0.8)
            packets.append({
                "timestamp": round(t + self.t0, 4),
                "detected": True,
                "X": float(true_x + nx), "Y": float(true_y + ny), "Z": float(true_z + nz),
                "true_X": float(true_x), "true_Y": float(true_y), "true_Z": float(true_z),
                "blob_area": float(self.rng.uniform(320, 480)),
            })
        return packets, true_landing

    def random_throw(self, home=(0.0, 1.3), reach=0.6, basket_height=0.35):
        """Throw that lands uniformly inside +/-reach of `home` (camera frame)."""
        r = self.rng
        hx, hy = home
        lx = hx + r.uniform(-reach, reach)
        ly = hy + r.uniform(-reach, reach)
        x0 = hx + r.uniform(-0.8, 0.8)
        y0 = r.uniform(2.4, 3.4)
        z0 = r.uniform(1.0, 1.6)
        vz = r.uniform(1.5, 3.0)
        a = 0.5 * self.g
        t_land = (vz + np.sqrt(vz ** 2 + 4 * a * (z0 - basket_height))) / (2 * a)
        vx = (lx - x0) / t_land
        vy = (ly - y0) / t_land
        return self.generate_throw((x0, y0, z0), (vx, vy, vz), basket_height)

    # ------------------------------------------------------------ non-throws
    def generate_walker(self, duration=2.0, speed=1.2, z=1.0, area=6000.0):
        """A person walking sideways: large blob, constant height, no parabola."""
        packets = []
        for k in range(int(duration * self.fps)):
            t = k * self.dt
            nx, ny, nz = self.rng.normal(0, self.noise_std, 3)
            packets.append({"timestamp": round(t + self.t0, 4), "detected": True,
                            "X": -1.0 + speed * t + nx, "Y": 2.5 + ny, "Z": z + nz,
                            "blob_area": area})
        return packets

    def generate_hand_wave(self, duration=1.5):
        """Small blob moving up and down sinusoidally (not free flight)."""
        packets = []
        for k in range(int(duration * self.fps)):
            t = k * self.dt
            nx, ny, nz = self.rng.normal(0, self.noise_std, 3)
            packets.append({"timestamp": round(t + self.t0, 4), "detected": True,
                            "X": 0.3 + nx, "Y": 2.4 - 0.9 * t + ny,
                            "Z": 1.1 + 0.4 * np.sin(2 * np.pi * 1.5 * t) + nz,
                            "blob_area": 400.0})
        return packets
