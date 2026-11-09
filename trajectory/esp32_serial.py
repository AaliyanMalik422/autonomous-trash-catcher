"""Serial link to the ESP32 locomotion firmware (115200 baud, newline-terminated).

Protocol (ASCII, one command per line):
    <POS,x,y>   go to (x, y) metres in the robot's HOME frame (+x strafe right, +y forward)
    <HOME>      go back to the home position (0, 0)
    <STOP>      stop immediately and hold position
    <ZERO>      declare the current position to be (0, 0) (re-zero odometry)
The ESP32 answers with lines such as  `<ACK,POS,x,y>` and `<ODO,x,y,vx,vy>`.
"""
import time
from typing import List, Optional


def find_esp32_port() -> Optional[str]:
    """Best-effort auto-detect of an ESP32 USB-serial adapter (CP210x / CH340 / FTDI)."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return None
    for p in list_ports.comports():
        desc = f"{p.description} {p.manufacturer or ''}".lower()
        if any(k in desc for k in ("cp210", "ch340", "ch910", "silicon labs", "ftdi", "usb serial", "esp32")):
            return p.device
    return None


class ESP32Link:
    def __init__(self, port: Optional[str] = None, baud: int = 115200,
                 min_interval_s: float = 0.02, dry_run: bool = False, verbose: bool = False):
        self.port = port
        self.baud = baud
        self.min_interval_s = min_interval_s
        self.dry_run = dry_run
        self.verbose = verbose
        self.sent: List[str] = []      # log of everything sent (useful for tests)
        self._ser = None
        self._last_send = 0.0
        self._rx_buf = b""
        if not dry_run:
            self.open()

    # ------------------------------------------------------------- lifecycle
    def open(self):
        import serial  # pyserial
        port = self.port or find_esp32_port()
        if port is None:
            raise RuntimeError("No ESP32 serial port found. Pass port='COM5' or '/dev/ttyUSB0', "
                               "or use dry_run=True.")
        self.port = port
        self._ser = serial.Serial(port, self.baud, timeout=0, write_timeout=0.2)
        time.sleep(2.0)  # ESP32 auto-resets when the port opens
        self._ser.reset_input_buffer()

    def close(self):
        if self._ser is not None:
            try:
                self.stop()
            finally:
                self._ser.close()
                self._ser = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --------------------------------------------------------------- sending
    @staticmethod
    def format_target(x: float, y: float) -> str:
        return f"<POS,{x:.3f},{y:.3f}>\n"

    def _write(self, text: str, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now - self._last_send < self.min_interval_s:
            return False
        self._last_send = now
        self.sent.append(text.strip())
        if self.verbose or self.dry_run:
            print("[ESP32 ->]", text.strip())
        if self._ser is not None:
            self._ser.write(text.encode("ascii"))
        return True

    def send_target(self, x: float, y: float) -> bool:
        return self._write(self.format_target(x, y))

    def home(self):
        self._write("<HOME>\n", force=True)

    def stop(self):
        self._write("<STOP>\n", force=True)

    def zero(self):
        self._write("<ZERO>\n", force=True)

    # -------------------------------------------------------------- receiving
    def poll(self) -> List[str]:
        """Non-blocking read of any complete lines from the ESP32."""
        if self._ser is None:
            return []
        data = self._ser.read(self._ser.in_waiting or 0)
        if not data:
            return []
        self._rx_buf += data
        *lines, self._rx_buf = self._rx_buf.split(b"\n")
        return [l.decode("ascii", "ignore").strip() for l in lines if l.strip()]
