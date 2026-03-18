#!/usr/bin/env python3
"""
IMU Toolkit
Unified reader, recorder, and visualizer for up to 9 IMUs connected to a single
serial receiver (e.g., ESP32) streaming multiplexed packets.
"""

import argparse
import csv
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import serial
import serial.tools.list_ports

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


DEFAULT_PORT = "/dev/ttyACM0"


class IMUReader:
    """
    Reads IMU data from a serial device (e.g., ESP32)
    Parses roll/pitch/yaw, acceleration, gyro, quaternion, and battery data.
    """

    def __init__(self, baudrate: int = 115200):
        self.baudrate = baudrate
        self.ser = None
        self.pattern = re.compile(
            r"ID:(?P<imu_id>\d+)\s*\|\s*"
            r"r:(?P<roll>-?\d+\.\d+)\s+p:(?P<pitch>-?\d+\.\d+)\s+y:(?P<yaw>-?\d+\.\d+)\s*\|\s*"
            r"ax:(?P<ax>-?\d+\.\d+)\s+ay:(?P<ay>-?\d+\.\d+)\s+az:(?P<az>-?\d+\.\d+)"
            r"(?:\s*\|\s*gx:(?P<gx>-?\d+\.\d+)\s+gy:(?P<gy>-?\d+\.\d+)\s+gz:(?P<gz>-?\d+\.\d+))?"
            r"\s*\|\s*"
            r"qI:(?P<quatI>-?\d+\.\d+)\s+qJ:(?P<quatJ>-?\d+\.\d+)\s+qK:(?P<quatK>-?\d+\.\d+)\s+qW:(?P<quatW>-?\d+\.\d+)"
            r"(?:\s*\|\s*batt:(?P<battery>\d+)%?)?",
            re.IGNORECASE,
        )

    def find_serial_port(self, port: Optional[str] = None) -> str:
        """Auto-detect a USB serial port or use the provided one."""
        if port:
            try:
                self.ser = serial.Serial(port, self.baudrate, timeout=1)
                print(f"✅ Connected to {port}")
                time.sleep(2)
                return port
            except Exception as exc:
                raise IOError(f"❌ Failed to connect to {port}: {exc}") from exc

        ports = list(serial.tools.list_ports.comports())
        if not ports:
            raise IOError("❌ No serial ports found. Is your device plugged in?")

        usb_ports = []
        esp32_ports = []

        for dev in ports:
            device_lower = dev.device.lower()
            desc_lower = (dev.description or "").lower()
            manufacturer_lower = (dev.manufacturer or "").lower()

            if any(keyword in desc_lower or keyword in manufacturer_lower for keyword in ["esp32", "esp", "adafruit", "feather"]):
                esp32_ports.append(dev)
            elif any(pattern in device_lower for pattern in ["ttyusb", "ttyacm", "cu.usbserial", "cu.usbmodem", "com"]):
                usb_ports.append(dev)
            elif dev.description and ("usb" in desc_lower or "serial" in desc_lower):
                usb_ports.append(dev)

        if esp32_ports:
            selected_ports = esp32_ports
            print("✅ ESP32 device detected!")
        elif usb_ports:
            selected_ports = usb_ports
        else:
            print("⚠️  No USB serial port detected, trying first available port...")
            selected_ports = [ports[0]]

        chosen = selected_ports[0]
        try:
            self.ser = serial.Serial(chosen.device, self.baudrate, timeout=1)
            print(f"✅ Connected to {chosen.device} ({chosen.description})")
            time.sleep(2)
            return chosen.device
        except Exception as exc:
            raise IOError(f"❌ Failed to connect to {chosen.device}: {exc}") from exc

    def _parse_line(self, line: str) -> Optional[Dict[str, float]]:
        match = self.pattern.match(line)
        if not match:
            return None

        groups = match.groupdict()
        try:
            reading: Dict[str, float] = {
                "imu_id": int(groups["imu_id"]),
                "roll": float(groups["roll"]),
                "pitch": float(groups["pitch"]),
                "yaw": float(groups["yaw"]),
                "ax": float(groups["ax"]),
                "ay": float(groups["ay"]),
                "az": float(groups["az"]),
                "gx": float(groups["gx"]) if groups.get("gx") is not None else 0.0,
                "gy": float(groups["gy"]) if groups.get("gy") is not None else 0.0,
                "gz": float(groups["gz"]) if groups.get("gz") is not None else 0.0,
                "quatI": float(groups["quatI"]),
                "quatJ": float(groups["quatJ"]),
                "quatK": float(groups["quatK"]),
                "quatW": float(groups["quatW"]),
            }
        except (TypeError, ValueError) as exc:
            print(f"⚠️  Failed to parse IMU line: {line[:100]} ({exc})")
            return None

        battery = groups.get("battery")
        reading["battery_percent"] = int(battery) if battery is not None else None
        return reading

    def read_imu_data(self) -> Optional[Dict[str, float]]:
        """Return the most recent IMU packet (blocking until a line is read)."""
        if self.ser is None:
            raise IOError("Serial port not initialized. Call find_serial_port() first.")

        try:
            raw = self.ser.readline()
            timestamp_ns = time.time_ns()
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                return None
            parsed = self._parse_line(line)
            if parsed is None:
                return None
            parsed.setdefault("utc_timestamp_ns", timestamp_ns)
            return parsed
        except Exception as exc:
            print(f"⚠️ Error reading IMU: {exc}")
            return None

    def read_available_data(self, max_messages: Optional[int] = 50) -> List[Dict[str, float]]:
        """Drain buffered IMU packets without blocking."""
        if self.ser is None:
            raise IOError("Serial port not initialized. Call find_serial_port() first.")

        readings: List[Dict[str, float]] = []
        try:
            while max_messages is None or len(readings) < max_messages:
                if self.ser.in_waiting == 0:
                    break
                raw = self.ser.readline()
                timestamp_ns = time.time_ns()
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    break
                parsed = self._parse_line(line)
                if parsed:
                    parsed.setdefault("utc_timestamp_ns", timestamp_ns)
                    readings.append(parsed)
        except Exception as exc:
            print(f"⚠️ Error draining IMU buffer: {exc}")

        return readings

class IMUDataRecorder:
    """
    Records multiplexed IMU packets to CSV and exposes data snapshots for inspection or visualization.
    """

    def __init__(self, output_dir: str = "imu_recordings", baudrate: int = 115200, num_imus: int = 9, max_batch_reads: int = 64):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.is_recording = False
        self.recording_path = None
        self.csv_file = None
        self.csv_writer = None
        self.latest_readings: Dict[int, Dict[str, float]] = {}
        self.num_imus = num_imus
        self.max_batch_reads = max(1, max_batch_reads)

        self.imu = IMUReader(baudrate=baudrate)
        self.connected = False
        self._latest_lock = threading.Lock()
        self._record_lock = threading.Lock()

    def connect(self, port: Optional[str] = None) -> bool:
        if self.connected:
            print("Already connected to IMU")
            return True

        try:
            resolved = self.imu.find_serial_port(port=port)
            self.connected = True
            print(f"✅ Connected to IMU on {resolved}")
            return True
        except IOError as exc:
            print(f"❌ Connection failed: {exc}")
            return False

    def start_recording(self, recording_path: Optional[Union[str, Path]] = None) -> bool:
        if not self.connected:
            print("Not connected to IMU. Call connect() first.")
            return False

        with self._record_lock:
            if self.is_recording:
                print("Already recording...")
                return True

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if recording_path is None:
                self.recording_path = self.output_dir / f"imu_recording_{timestamp}"
            else:
                self.recording_path = Path(recording_path)
            self.recording_path.mkdir(parents=True, exist_ok=True)

            csv_filename = self.recording_path / "imu_data.csv"
            self.csv_file = open(csv_filename, "w", newline="")
            fieldnames = [
                "utc_timestamp_ns",
                "imu_id",
                "roll",
                "pitch",
                "yaw",
                "ax",
                "ay",
                "az",
                "gx",
                "gy",
                "gz",
                "quatI",
                "quatJ",
                "quatK",
                "quatW",
                "battery_percent",
            ]
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()

            self.is_recording = True
            print(f"✅ IMU recording started: {self.recording_path}")
            return True

    def stop_recording(self):
        with self._record_lock:
            if not self.is_recording:
                print("Not currently recording")
                return
            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None
                self.csv_writer = None
            self.is_recording = False
            print(f"✅ IMU recording stopped. Data saved to: {self.recording_path}")

    def read_and_save_data(self) -> List[Dict[str, float]]:
        if not self.connected:
            return []

        new_readings: List[Dict[str, float]] = []
        primary = self.imu.read_imu_data()
        if primary:
            new_readings.append(primary)

        if len(new_readings) < self.max_batch_reads:
            drained = self.imu.read_available_data(self.max_batch_reads - len(new_readings))
            new_readings.extend(drained)

        if not new_readings:
            return []

        rows_to_write: List[Dict[str, float]] = []
        for reading in new_readings:
            imu_id = reading["imu_id"]
            timestamp_ns = reading.get("utc_timestamp_ns")
            if timestamp_ns is None:
                timestamp_ns = time.time_ns()
                reading["utc_timestamp_ns"] = timestamp_ns

            with self._latest_lock:
                self.latest_readings[imu_id] = reading

            rows_to_write.append(
                {
                    "utc_timestamp_ns": timestamp_ns,
                    "imu_id": imu_id,
                    "roll": reading["roll"],
                    "pitch": reading["pitch"],
                    "yaw": reading["yaw"],
                    "ax": reading["ax"],
                    "ay": reading["ay"],
                    "az": reading["az"],
                    "gx": reading["gx"],
                    "gy": reading["gy"],
                    "gz": reading["gz"],
                    "quatI": reading["quatI"],
                    "quatJ": reading["quatJ"],
                    "quatK": reading["quatK"],
                    "quatW": reading["quatW"],
                    "battery_percent": reading.get("battery_percent"),
                }
            )

        with self._record_lock:
            if self.is_recording and self.csv_writer:
                for row in rows_to_write:
                    self.csv_writer.writerow(row)
                if self.csv_file:
                    self.csv_file.flush()

        return new_readings

    def get_latest_snapshot(self) -> Dict[int, Dict[str, float]]:
        """Return a shallow copy of the latest readings for thread-safe inspection."""
        with self._latest_lock:
            return {imu_id: data.copy() for imu_id, data in self.latest_readings.items()}

def quaternion_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    quat = np.array([qw, qx, qy, qz], dtype=float)
    norm = np.linalg.norm(quat)
    if norm == 0:
        return np.eye(3)
    qw, qx, qy, qz = quat / norm
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )


class IMUGridVisualizer:
    """Matplotlib-based 3x3 grid visualizer showing axes + RPY/quaternions."""

    def __init__(
        self,
        recorder: IMUDataRecorder,
        imu_ids: Optional[List[int]] = None,
        update_hz: float = 20.0,
        *,
        auto_update: bool = True,
        enable_animation: bool = True,
    ):
        if plt is None or FuncAnimation is None:
            raise ImportError("matplotlib is required for visualization. Install it via `pip install matplotlib`.")

        self.recorder = recorder
        default_ids = sorted(range(1, recorder.num_imus + 1))
        self.imu_ids = (imu_ids or default_ids)[:9]
        self.interval_ms = int(1000 / max(1e-3, update_hz))
        self.auto_update = auto_update
        self.enable_animation = enable_animation

        self.fig, axes = plt.subplots(3, 3, figsize=(13, 12), subplot_kw={"projection": "3d"})
        self.axes = axes.flatten()
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.05, wspace=0.32, hspace=0.45)
        self._init_axes()
        self.anim = None

    def _init_axes(self):
        for idx, ax in enumerate(self.axes):
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_box_aspect([1, 1, 1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            imu_label = self.imu_ids[idx] if idx < len(self.imu_ids) else "-"
            ax.set_title(f"IMU {imu_label}")

    @staticmethod
    def _reset_axis(ax, title: str):
        ax.cla()
        ax.set_title(title)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_box_aspect([1, 1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    def _draw_orientation(self, ax, rotation_matrix: np.ndarray):
        origin = np.zeros(3)
        colors = ["r", "g", "b"]
        labels = ["X", "Y", "Z"]
        for idx in range(3):
            direction = rotation_matrix[:, idx]
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                direction[0],
                direction[1],
                direction[2],
                length=1.0,
                color=colors[idx],
                linewidth=2,
            )
            ax.text(direction[0], direction[1], direction[2], labels[idx], color=colors[idx], fontsize=8)

    @staticmethod
    def _format_text(data: Dict[str, float]) -> str:
        return (
            f"RPY  roll={data['roll']:.1f}°  pitch={data['pitch']:.1f}°  yaw={data['yaw']:.1f}°\n"
            f"Quat qI={data['quatI']:.3f}  qJ={data['quatJ']:.3f}  qK={data['quatK']:.3f}  qW={data['quatW']:.3f}"
        )

    def _render_snapshot(self, snapshot: Dict[int, Dict[str, float]]):
        for idx, ax in enumerate(self.axes):
            imu_id = self.imu_ids[idx] if idx < len(self.imu_ids) else None
            title = f"IMU {imu_id}" if imu_id is not None else "Unused"
            self._reset_axis(ax, title)

            if imu_id is None:
                ax.text2D(0.5, 0.2, "No IMU", transform=ax.transAxes, ha="center")
                continue

            data = snapshot.get(imu_id, None)
            if not data:
                ax.text2D(0.5, 0.2, "Waiting for data...", transform=ax.transAxes, ha="center")
                continue

            rotation_matrix = quaternion_to_matrix(data["quatW"], data["quatI"], data["quatJ"], data["quatK"])
            self._draw_orientation(ax, rotation_matrix)
            ax.text2D(
                0.5,
                0.02,
                self._format_text(data),
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2},
                clip_on=False,
            )
        return self.axes

    def _update_frame(self, _frame_idx):
        if self.auto_update:
            self.recorder.read_and_save_data()

        snapshot = self.recorder.get_latest_snapshot()
        return self._render_snapshot(snapshot)

    def refresh(self, snapshot: Optional[Dict[int, Dict[str, float]]] = None):
        """Manually refresh the grid with the provided snapshot (non-animated use)."""
        if snapshot is None:
            snapshot = self.recorder.get_latest_snapshot()
        self._render_snapshot(snapshot)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(1e-3)

    def show(self, block: bool = False):
        """Display the figure (useful when animation is disabled)."""
        if plt is None:
            raise ImportError("matplotlib is required for visualization. Install it via `pip install matplotlib`.")
        if not block:
            plt.ion()
        plt.tight_layout()
        plt.show(block=block)

    def close(self):
        if plt:
            plt.close(self.fig)

    def run(self):
        if not self.enable_animation:
            raise RuntimeError("run() cannot be used when enable_animation=False. Call refresh()/show() manually instead.")

        self.anim = FuncAnimation(
            self.fig,
            self._update_frame,
            interval=self.interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.show()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IMU Data Reader / Recorder / Visualizer")
    parser.add_argument("--output-dir", type=str, default="imu_recordings", help="Directory for IMU recordings")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial port baudrate")
    parser.add_argument(
        "--port",
        type=str,
        default=DEFAULT_PORT,
        help=f"Serial port path (default: {DEFAULT_PORT}). Use 'auto' to scan.",
    )
    parser.add_argument(
        "--mode",
        choices=["visualize", "record"],
        default="visualize",
        help="visualize: show 3x3 orientation grid (default); record: write CSV without GUI",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Recording length in seconds (record mode). Leave empty to run until Ctrl+C.",
    )
    parser.add_argument("--visual-update-hz", type=float, default=100.0, help="Visualizer refresh rate")
    parser.add_argument("--visual-ids", type=int, nargs="*", help="Specific IMU IDs to show (max 9)")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    resolved_port = None if args.port in (None, "auto", "AUTO") else args.port

    recorder = IMUDataRecorder(output_dir=args.output_dir, baudrate=args.baudrate)

    if args.mode == "visualize":
        if not recorder.connect(port=resolved_port):
            return
        try:
            visualizer = IMUGridVisualizer(
                recorder=recorder,
                imu_ids=args.visual_ids,
                update_hz=args.visual_update_hz,
            )
            visualizer.run()
        except ImportError as exc:
            print(f"Visualizer unavailable: {exc}")
        finally:
            if recorder.imu.ser:
                recorder.imu.ser.close()
        return

    # Record mode
    if not recorder.connect(port=resolved_port):
        return
    if not recorder.start_recording():
        return

    start_time = time.time()
    print("📼 Recording IMU data... Press Ctrl+C to stop.")
    try:
        while args.duration is None or time.time() - start_time < args.duration:
            recorder.read_and_save_data()
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")
    finally:
        recorder.stop_recording()
        if recorder.imu.ser:
            recorder.imu.ser.close()
        print("✅ Recording complete.")


if __name__ == "__main__":
    main()
