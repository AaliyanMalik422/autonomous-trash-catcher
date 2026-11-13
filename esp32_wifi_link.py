#!/usr/bin/env python3
"""Safe connectivity check for the skid-steer ESP32 Wi-Fi link.

By default this script sends PING only and does not move the robot.
Use --move-test explicitly to run the 30 cm forward/return bench test.
"""
import argparse
import time

from trajectory.esp32_wifi_link import ESP32WifiLink


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="192.168.4.1", help="ESP32 access-point IP")
    parser.add_argument("--port", type=int, default=4210, help="ESP32 UDP port")
    parser.add_argument("--move-test", action="store_true",
                        help="explicitly drive 30 cm forward and return home")
    args = parser.parse_args()

    link = ESP32WifiLink(ip=args.ip, port=args.port, verbose=True)
    try:
        print(f"Checking skid-steer ESP32 at {args.ip}:{args.port} (PING only)...")
        connected = False
        for attempt in range(1, 11):
            link.ping()
            deadline = time.monotonic() + 0.3
            while time.monotonic() < deadline:
                if link.last_ack and link.last_ack[0] == "PING":
                    connected = True
                    break
                time.sleep(0.02)
            if connected:
                break
            print(f"No ACK yet (attempt {attempt}/10)")

        if not connected:
            print("ERROR: no <ACK,PING>. Check the laptop's TrashBot Wi-Fi connection and ESP32 power.")
            return 1

        print("SUCCESS: ESP32 acknowledged PING. No movement command has been sent.")
        if link.odo:
            x, y, vx, vy, _ = link.odo
            print(f"Odometry: x={x:+.3f} m, y={y:+.3f} m, vx={vx:+.2f} m/s, vy={vy:+.2f} m/s")

        if args.move_test:
            print("Running explicitly requested 30 cm movement test...")
            link.zero()
            time.sleep(0.2)
            link.pos(0.0, 0.30)
            t0 = time.monotonic()
            while time.monotonic() - t0 < 3.0:
                link.pos(0.0, 0.30)
                if link.odo:
                    print("  odo: x=%+.3f m, y=%+.3f m, vx=%+.2f m/s, vy=%+.2f m/s" % link.odo[:4])
                time.sleep(0.1)
            link.home()
            time.sleep(2.0)
            print("Movement test complete.")
        return 0
    finally:
        link.close()


if __name__ == "__main__":
    raise SystemExit(main())
