"""WiFi (UDP) link to the skid-steer ESP32 car.

Direct runner wrapper for trajectory.esp32_wifi_link.
"""
from trajectory.esp32_wifi_link import ESP32WifiLink

if __name__ == "__main__":
    import time

    link = ESP32WifiLink(verbose=True)
    link.ping()
    time.sleep(0.5)
    if link.last_ack is None:
        print("No reply from the car. Is the computer on the TrashBot WiFi? Is the car powered?")
    else:
        print("Car answered:", link.last_ack[0])
        link.zero()
        time.sleep(0.2)
        link.pos(0.0, 0.30)           # 30 cm forward
        t0 = time.monotonic()
        while time.monotonic() - t0 < 3.0:
            link.pos(0.0, 0.30)       # keep the target alive (3 s timeout on the car)
            if link.odo:
                print("odo x=%.3f y=%.3f vx=%.2f vy=%.2f" % link.odo[:4])
            time.sleep(0.1)
        link.home()
        time.sleep(2.0)
        link.stop()
    link.close()
