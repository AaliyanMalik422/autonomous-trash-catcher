import numpy as np

class MockStereoCamera:
    def __init__(self, fps=60, noise_std_meters=0.025, seed=42):
        self.dt = 1.0 / fps
        self.noise_std = noise_std_meters
        self.g = 9.81
        np.random.seed(seed)

    def generate_throw(self, start_pos=(0.6, 2.6, 1.3), velocity=(-0.4, -1.8, 2.2), basket_height=0.35):
        x0, y0, z0 = start_pos
        vx, vy, vz = velocity

        a = 0.5 * self.g
        b = -vz
        c = basket_height - z0
        discriminant = b**2 - 4 * a * c
        t_land_true = (-b + np.sqrt(discriminant)) / (2 * a)

        true_landing = {
            "t_land": t_land_true,
            "X_land": x0 + vx * t_land_true,
            "Y_land": y0 + vy * t_land_true,
            "Z_land": basket_height
        }

        packets = []
        t = 0.0
        while True:
            true_x = x0 + vx * t
            true_y = y0 + vy * t
            true_z = z0 + vz * t - 0.5 * self.g * (t ** 2)

            if true_z < 0.0:
                break

            noisy_x = true_x + np.random.normal(0, self.noise_std)
            noisy_y = true_y + np.random.normal(0, self.noise_std)
            noisy_z = true_z + np.random.normal(0, self.noise_std)

            packet = {
                "timestamp": round(t, 4),
                "detected": True,
                "X": float(noisy_x),
                "Y": float(noisy_y),
                "Z": float(noisy_z),
                "true_X": float(true_x),
                "true_Y": float(true_y),
                "true_Z": float(true_z),
                "blob_area": float(np.random.uniform(320, 480))
            }
            packets.append(packet)
            t += self.dt

        return packets, true_landing
