class ThrowFilter:
    def __init__(self):
        # Physical rejection thresholds
        self.MIN_AREA = 100       # pixels: reject tiny dust/sensor noise
        self.MAX_AREA = 2500      # pixels: reject large bodies (e.g., people walking)
        self.MIN_HEIGHT_Z = 0.35  # meters: must be airborne off the floor
        self.MIN_SPEED_Y = 0.50   # m/s: must have forward velocity toward robot
        
        self.trajectory_buffer = []

    def process_frame(self, detection_packet):
        """
        Filters incoming 3D detection frames.
        Returns: (is_valid: bool, buffer: list)
        """
        if not detection_packet.get("detected", False):
            self.trajectory_buffer.clear()
            return False, []

        area = detection_packet.get("blob_area", 0)
        z = detection_packet.get("Z", 0.0)

        # Filter 1: Area size check
        if not (self.MIN_AREA <= area <= self.MAX_AREA):
            return False, self.trajectory_buffer

        # Filter 2: Airborne height check
        if z < self.MIN_HEIGHT_Z:
            return False, self.trajectory_buffer

        self.trajectory_buffer.append(detection_packet)

        # Need at least 3 points to reliably compute trajectory vectors
        if len(self.trajectory_buffer) < 3:
            return False, self.trajectory_buffer

        oldest = self.trajectory_buffer[0]
        newest = self.trajectory_buffer[-1]

        dt = newest["timestamp"] - oldest["timestamp"]
        if dt <= 0.005:
            return False, self.trajectory_buffer

        # Filter 3: Forward velocity (Y decreasing toward robot)
        speed_y = (oldest["Y"] - newest["Y"]) / dt
        if speed_y < self.MIN_SPEED_Y:
            return False, self.trajectory_buffer

        return True, self.trajectory_buffer

if __name__ == "__main__":
    tf = ThrowFilter()
    print("ThrowFilter initialized successfully.")
