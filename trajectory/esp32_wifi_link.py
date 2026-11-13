"""WiFi (UDP) link to the skid-steer ESP32.

Same text commands as the serial firmware:
    <VEL,vx,vy,w>   <POS,x,y>   <HOME>   <STOP>   <ZERO>   <PING>

x = right [m], y = forward [m] (relative to the last ZERO), w = rad/s (+ = counter-clockwise).
The car replies <ACK,CMD> and sends <ODO,x,y,vx,vy> every 50 ms.

Connect the computer to the car's WiFi network ("TrashBot", password 12345678) first.
The cameras use USB, so the computer does not need any other network.

Run this file directly for a short test (THE CAR WILL MOVE 30 cm, wheels free or clear floor):
    python esp32_wifi_link.py
"""
import socket
import threading
import time
from typing import List, Optional, Tuple


class ESP32WifiLink:
    def __init__(self, ip: str = "192.168.4.1", port: int = 4210, min_interval_s: float = 0.02,
                 dry_run: bool = False, verbose: bool = False):
        self.addr = (ip, port)
        self.min_interval_s = min_interval_s
        self.dry_run = dry_run
        self.verbose = verbose
        self.sent: List[str] = []
        self._lock = threading.Lock()
        self._last_tx = 0.0
        self._running = True
        self.odo: Optional[Tuple[float, float, float, float, float]] = None  # (x, y, vx, vy, rx_time)
        self.last_ack: Optional[Tuple[str, float]] = None                   # (name, rx_time)

        if not dry_run:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(("", 0))
            self.sock.settimeout(0.2)
            self._rx = threading.Thread(target=self._rx_loop, daemon=True)
            self._rx.start()
        else:
            self.sock = None
            self._rx = None

    # ---------------------------------------------------------------- send
    def _send(self, msg: str, force: bool = False) -> bool:
        now = time.monotonic()
        with self._lock:
            if not force and now - self._last_tx < self.min_interval_s:
                return False
            self._last_tx = now
            self.sent.append(msg.strip())

        if self.verbose or self.dry_run:
            print("[ESP32-WiFi ->]", msg.strip())

        if self.sock is not None:
            try:
                self.sock.sendto(msg.encode(), self.addr)
            except Exception as e:
                if self.verbose:
                    print(f"[ESP32-WiFi Error] {e}")
                return False
        return True

    def ping(self) -> bool:
        return self._send("<PING>\n", force=True)

    def zero(self) -> bool:
        return self._send("<ZERO>\n", force=True)

    def home(self) -> bool:
        return self._send("<HOME>\n", force=True)

    def stop(self) -> bool:
        return self._send("<STOP>\n", force=True)

    def pos(self, x: float, y: float) -> bool:
        return self._send(f"<POS,{x:.3f},{y:.3f}>\n")

    def send_target(self, x: float, y: float) -> bool:
        """Alias for pos() to provide drop-in compatibility with ESP32Link."""
        return self.pos(x, y)

    def vel(self, vx: float, vy: float, w: float) -> bool:
        return self._send(f"<VEL,{vx:.3f},{vy:.3f},{w:.3f}>\n")

    # ------------------------------------------------------------- receive
    def _rx_loop(self):
        while self._running:
            if self.sock is None:
                break
            try:
                data, _ = self.sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                break
            now = time.monotonic()
            text = data.decode(errors="ignore")
            for chunk in text.split(">"):
                chunk = chunk.strip()
                if not chunk.startswith("<"):
                    continue
                parts = chunk[1:].split(",")
                if parts[0] == "ODO" and len(parts) >= 5:
                    try:
                        x, y, vx, vy = (float(p) for p in parts[1:5])
                    except ValueError:
                        continue
                    self.odo = (x, y, vx, vy, now)
                elif parts[0] == "ACK" and len(parts) >= 2:
                    self.last_ack = (parts[1], now)

    def odo_age_s(self) -> Optional[float]:
        return None if self.odo is None else time.monotonic() - self.odo[4]

    def poll(self) -> List[str]:
        """Check status or return latest ACK for compatibility."""
        res = []
        if self.last_ack is not None:
            res.append(f"<ACK,{self.last_ack[0]}>")
        if self.odo is not None:
            res.append(f"<ODO,{self.odo[0]:.3f},{self.odo[1]:.3f},{self.odo[2]:.3f},{self.odo[3]:.3f}>")
        return res

    def close(self):
        try:
            self.stop()
        finally:
            self._running = False
            if self.sock is not None:
                try:
                    self.sock.close()
                except Exception:
                    pass
                self.sock = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":
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
