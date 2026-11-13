"""WiFi (UDP) link to the skid-steer ESP32 car.

Direct runner wrapper for trajectory.esp32_wifi_link.
"""
import argparse
import sys
import time

from trajectory.esp32_wifi_link import ESP32WifiLink

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test WiFi link to ESP32 skid-steer car.")
    parser.add_argument("--ip", default="192.168.4.1", help="ESP32 IP address (default: 192.168.4.1)")
    parser.add_argument("--port", type=int, default=4210, help="ESP32 UDP port (default: 4210)")
    args = parser.parse_args()

    print(f"Connecting to car at {args.ip}:{args.port}...")
    link = ESP32WifiLink(ip=args.ip, port=args.port, verbose=True)

    # Ping with retries to allow ARP resolution and WiFi socket settling
    connected = False
    for attempt in range(1, 11):
        link.ping()
        time.sleep(0.3)
        if link.last_ack is not None or link.odo is not None:
            connected = True
            break
        print(f"Waiting for car response... (attempt {attempt}/10)")

    if not connected:
        print("\nERROR: No reply from the car.")
        print("Troubleshooting checklist:")
        print("  1. Is your laptop Wi-Fi connected to 'TrashBot'?")
        print("  2. Run 'ping 192.168.4.1' to check if the ESP32 is reachable.")
        print("  3. Verify the ESP32 power LED is lit and battery is switched ON.")
    else:
        name = link.last_ack[0] if link.last_ack else "CONNECTED"
        print(f"\nSUCCESS! Car answered ({name}).")
        print("Running motion test: driving 30 cm forward...")
        link.zero()
        time.sleep(0.2)
        link.pos(0.0, 0.30)           # 30 cm forward
        t0 = time.monotonic()
        while time.monotonic() - t0 < 3.0:
            link.pos(0.0, 0.30)       # keep target alive
            if link.odo:
                print(f"  odo: x={link.odo[0]:+.3f}m, y={link.odo[1]:+.3f}m, vx={link.odo[2]:+.2f}m/s, vy={link.odo[3]:+.2f}m/s")
            time.sleep(0.1)
        print("Returning home...")
        link.home()
        time.sleep(2.0)
        link.stop()
        print("Test complete. Car stopped.")
    link.close()
