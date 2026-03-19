#!/usr/bin/env python3
"""
ROSHI Receiver + Calibration Orchestrator

Receives video + metadata from iOS app, optionally runs calibration pipeline:
  1) prepare session (extract frames, camera.json, frames.csv, apriltag summary)
  2) SAM-3D-Body inference -> body_data/*.npz (MHR outputs) + body_vis/*.jpg
  3) MHR -> SMPL(X) conversion -> smpl_output/per_frame/*_smpl.npz
  4) compute bone->sensor offsets -> imu_calibration.json
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import uuid as uuid_mod
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from zeroconf import ServiceInfo, Zeroconf

_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent

# Add paths for local imports
for p in [str(_SRC_DIR), str(_PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from hardware.imu_reader import IMUDataRecorder
from utils.imu_id_mapping import IMU_ID_TO_JOINT
from utils.session_preparation import prepare_session

try:
    from config_local import CLUSTER_HOST, CLUSTER_RECEIVED_RECORDINGS_DIR, DISABLE_ZEROCONF
except ImportError:
    CLUSTER_HOST = ""
    CLUSTER_RECEIVED_RECORDINGS_DIR = ""
    DISABLE_ZEROCONF = False



def _print_imu_joint_map() -> None:
    """Print the expected IMU ID -> joint mapping for this pipeline."""
    print("\nIMU ID → Joint mapping (expected):")
    for imu_id in sorted(IMU_ID_TO_JOINT.keys()):
        print(f"  IMU {imu_id}: {IMU_ID_TO_JOINT[imu_id]}")
    print()


def _print_imu_connection_status(
    recorder: Optional["IMUDataRecorder"],
    *,
    expected_imu_ids: List[int],
    imu_to_joint: Dict[int, str],
    probe_seconds: float = 1.5,
    poll_hz: float = 20.0,
) -> None:
    """
    Probe the IMU stream briefly and print which IMU IDs appear to be active.

    Notes:
    - This is a best-effort "presence" check: an IMU is considered "seen" if any packet
      arrives during the probe window.
    - Requires the background IMU polling thread to be running (so snapshots update).
    """
    if recorder is None:
        print("\nIMU status: IMU recorder not initialized.")
        return
    if not getattr(recorder, "connected", False):
        print("\nIMU status: not connected.")
        return

    probe_seconds = max(0.2, float(probe_seconds))
    poll_hz = max(2.0, float(poll_hz))
    sleep_s = 1.0 / poll_hz

    print(f"\nProbing IMU stream for {probe_seconds:.1f}s to detect active sensors...")
    seen: set[int] = set()
    last_snapshot: Dict[int, Dict[str, Any]] = {}
    deadline = time.time() + probe_seconds
    while time.time() < deadline:
        snap = recorder.get_latest_snapshot()
        last_snapshot = snap
        seen.update(snap.keys())
        time.sleep(sleep_s)

    now_ns = time.time_ns()
    print("IMU connectivity:")
    for imu_id in expected_imu_ids:
        joint = imu_to_joint.get(imu_id, "unknown")
        if imu_id not in seen:
            print(f"  IMU {imu_id}: {joint:14s}  MISSING (no packets seen)")
            continue
        pkt = last_snapshot.get(imu_id, {})
        ts = pkt.get("utc_timestamp_ns")
        age_ms = None
        if isinstance(ts, int):
            age_ms = (now_ns - ts) / 1_000_000.0
        batt = pkt.get("battery_percent")
        batt_str = f"{batt}%" if isinstance(batt, int) else "?"
        age_str = f"{age_ms:6.1f} ms" if isinstance(age_ms, float) else "   ?   ms"
        print(f"  IMU {imu_id}: {joint:14s}  SEEN (age {age_str}, batt {batt_str})")
    print()


@dataclass
class CalibrationConfig:
    auto_calibrate: bool
    prompt: bool
    sam_python: str
    mhr_python: str
    calib_python: str
    sam_checkpoint: Path
    sam_mhr_model: Path
    smplx_model: Path
    device: str
    batch_size: int
    min_samples: int
    save_body_vis: bool = False
    calib_frame_start: Optional[int] = None
    calib_frame_end: Optional[int] = None
    calib_utc_start_ns: Optional[int] = None
    calib_utc_end_ns: Optional[int] = None
    calib_frames_csv: Optional[Path] = None


@dataclass
class ClusterConfig:
    """Configuration for uploading sessions to a remote cluster."""
    enabled: bool = False
    host: str = ""
    remote_dir: str = ""
    auto_upload: bool = False
    delete_local_after_upload: bool = False


def _upload_to_cluster(session: Path, cluster: ClusterConfig) -> bool:
    """Upload a session folder to the remote cluster using rsync.

    Returns True if upload succeeded, False otherwise.
    """
    if not cluster.enabled:
        return False

    remote_path = f"{cluster.host}:{cluster.remote_dir}/"

    print(f"\n📤 Uploading {session.name} to cluster...")
    print(f"   Local:  {session}")
    print(f"   Remote: {remote_path}{session.name}")

    rsync_cmd = [
        "rsync", "-avz", "--progress",
        str(session) + "/",
        f"{remote_path}{session.name}/",
    ]

    try:
        subprocess.run(rsync_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Upload failed (rsync exit {e.returncode}). Local files kept.")
        return False
    except FileNotFoundError:
        print("❌ rsync not found. Please install rsync. Local files kept.")
        return False

    # Verify the remote copy exists before deleting local files.
    verify_cmd = [
        "ssh", cluster.host,
        "test", "-d", f"{cluster.remote_dir}/{session.name}",
    ]
    try:
        subprocess.run(verify_cmd, check=True, timeout=15)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"⚠️  Upload reported success but remote verification failed: {e}")
        print("   Local files kept as a safety measure.")
        return False

    print(f"✅ Upload complete and verified: {session.name}")

    if cluster.delete_local_after_upload:
        import shutil
        print(f"🗑️  Deleting local copy: {session}")
        shutil.rmtree(session)

    return True


def _find_conda_python(env_name: str) -> Optional[str]:
    """Find Python executable from a conda environment."""
    candidate = Path.home() / "miniconda3" / "envs" / env_name / "bin" / "python"
    return str(candidate) if candidate.exists() else None


def _yes_no_prompt(msg: str, default: bool = False) -> bool:
    prompt = " [Y/n] " if default else " [y/N] "
    try:
        ans = input(msg + prompt).strip().lower()
    except EOFError:
        return default
    if not ans:
        return default
    return ans in ("y", "yes")


class ROSHICalibrationReceiver:
    def __init__(
        self,
        port: int = 50000,
        output_dir: str = "received_recordings",
        imu_port: Optional[str] = None,
        num_imus: int = 9,
        calibration: CalibrationConfig = None,
        cluster: ClusterConfig = None,
    ):
        self.port = port
        self.output_path = Path(output_dir).resolve()
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.calibration = calibration
        self.cluster = cluster or ClusterConfig()

        self.zeroconf = None
        self.service_info = None
        self.server_socket = None
        self.running = False
        
        # IMU
        self.imu_recorder = None
        self._imu_thread: Optional[threading.Thread] = None
        self._imu_stop = threading.Event()

        # Session registry: maps UUID hex string -> session directory Path
        self._sessions: Dict[str, Path] = {}
        self._sessions_lock = threading.Lock()

        cache = self.output_path / "_imu_cache"
        cache.mkdir(exist_ok=True)
        self.imu_recorder = IMUDataRecorder(output_dir=str(cache), num_imus=num_imus)
        if imu_port not in (None, "auto"):
            self.imu_recorder.connect(port=imu_port)
        else:
            self.imu_recorder.connect()

    def start(self):
        if self.port == 0:
            with socket.socket() as s:
                s.bind(("", 0))
                self.port = s.getsockname()[1]

        print(f"Starting ROSHI receiver on port {self.port}")
        print(f"Output: {self.output_path}")
        
        self._advertise_service()
        self._start_imu_thread()
        _print_imu_connection_status(
            self.imu_recorder,
            expected_imu_ids=sorted(IMU_ID_TO_JOINT.keys()),
            imu_to_joint=IMU_ID_TO_JOINT,
            probe_seconds=1.5,
        )

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(("", self.port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)
        self.running = True

        print("Waiting for connections...")
        try:
            while self.running:
                try:
                    client, addr = self.server_socket.accept()
                    print(f"\nConnection from {addr[0]}:{addr[1]}")
                    threading.Thread(target=self._handle_client, args=(client,), daemon=True).start()
                except socket.timeout:
                    pass
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        self._imu_stop.set()
        if self.server_socket:
            self.server_socket.close()
        if self.zeroconf and self.service_info:
            self.zeroconf.unregister_service(self.service_info)
            self.zeroconf.close()
        if self.imu_recorder and self.imu_recorder.is_recording:
            self.imu_recorder.stop_recording()
        print("Receiver stopped")

    def _advertise_service(self):
        hostname = socket.gethostname()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            local_ip = socket.gethostbyname(hostname)

        try:
            self.zeroconf = Zeroconf()
            self.service_info = ServiceInfo(
                "_roshi._tcp.local.",
                f"ROSHI Calibration._roshi._tcp.local.",
                addresses=[socket.inet_aton(local_ip)],
                port=self.port,
                properties={},
                server=f"{hostname}.local.",
            )
            self.zeroconf.register_service(self.service_info)
            print(f"Advertised: {local_ip}:{self.port}")
        except OSError as e:
            self.zeroconf = None
            self.service_info = None
            print(f"⚠️  Zeroconf unavailable ({e})")
            print(f"   iOS app won't auto-discover. Connect manually to {local_ip}:{self.port}")

    def _start_imu_thread(self):
        if self.imu_recorder is None or not self.imu_recorder.connected:
            return
        self._imu_stop.clear()
        self._imu_thread = threading.Thread(target=self._imu_loop, daemon=True)
        self._imu_thread.start()

    def _imu_loop(self):
        while not self._imu_stop.is_set():
            if self.imu_recorder and self.imu_recorder.connected:
                self.imu_recorder.read_and_save_data()
            time.sleep(0.002)

    # ─────────────────────────────────────────────────────────────────
    # Calibration Pipeline
    # ─────────────────────────────────────────────────────────────────

    def _run_calibration(self, session: Path) -> None:
        cfg = self.calibration

        # 0) Prepare session
        print("\n[1/4] Preparing session...")
        video_path = session / "video.mp4"
        if video_path.exists() and video_path.stat().st_size == 0:
            raise RuntimeError(
                f"Received empty video file (0 bytes): {video_path}\n"
                "This usually means the iOS app did not successfully write the mp4 before sending.\n"
                "Check the iOS console logs for AVAssetWriter errors and ensure the recording runs long enough."
            )
        prepare_session(session)

        color_dir = session / "color"
        body_data = session / "body_data"
        smpl_out = session / "smpl_output"
        cam_json = session / "meta" / "camera.json"

        for d in [body_data, smpl_out]:
            d.mkdir(exist_ok=True)

        # 1) SAM-3D-Body (MHR)
        print("\n[2/4] Running SAM-3D-Body...")
        sam_demo = _PROJECT_ROOT / "sam-3d-body" / "demo.py"
        sam_env = os.environ.copy()
        sam_env["MOMENTUM_ENABLED"] = ""  # Disable PyMomentum for Python 3.12
        sam_env["PYTHONPATH"] = os.pathsep.join([str(_PROJECT_ROOT), str(_PROJECT_ROOT / "MHR")])

        def _run_sam_3d_body(*, output_folder: Path) -> None:
            sam_cmd = [
                cfg.sam_python, str(sam_demo),
                "--image_folder", str(color_dir),
                "--checkpoint_path", str(cfg.sam_checkpoint),
                "--output_folder", str(output_folder),
                "--data_folder", str(body_data),
                "--mhr_path", str(cfg.sam_mhr_model),
                "--camera_json", str(cam_json),
                "--fov_name", "",  # Use camera.json intrinsics, not FOV estimator
                "--subject_only",
            ]
            subprocess.run(sam_cmd, cwd=str(_PROJECT_ROOT), env=sam_env, check=True)

        if cfg.save_body_vis:
            body_vis = session / "body_vis"
            body_vis.mkdir(exist_ok=True)
            _run_sam_3d_body(output_folder=body_vis)
        else:
            # By default, avoid persisting visualization renders; write them to a temp dir and delete.
            with tempfile.TemporaryDirectory(prefix=f"roshi_body_vis_{session.name}_") as tmp_vis:
                _run_sam_3d_body(output_folder=Path(tmp_vis))

        print(f"  ✅ SAM outputs: {len(list(body_data.glob('*.npz')))} npz files")

        # 2) MHR -> SMPL(X)
        print("\n[3/4] Converting MHR to SMPL...")
        convert_script = _PROJECT_ROOT / "MHR" / "tools" / "mhr_smpl_conversion" / "convert_mhr_to_smpl.py"
        conv_cwd = convert_script.parent  # IMPORTANT: must run from this dir for relative assets

        conv_env = os.environ.copy()
        conv_env["PYTHONPATH"] = os.pathsep.join([str(_PROJECT_ROOT), str(_PROJECT_ROOT / "MHR")])

        conv_cmd = [
            cfg.mhr_python, str(convert_script),
            "--input", str(body_data),
            "--output", str(smpl_out),
            "--smplx", str(cfg.smplx_model),
            "--device", cfg.device,
            "--batch-size", str(cfg.batch_size),
        ]
        subprocess.run(conv_cmd, cwd=str(conv_cwd), env=conv_env, check=True)
        per_frame = smpl_out / "per_frame"
        print(f"  ✅ SMPL outputs: {len(list(per_frame.glob('*.npz')))} npz files")

        # 3) Bone->sensor calibration
        print("\n[4/4] Computing bone->sensor offsets...")
        imu_calib = _PROJECT_ROOT / "02_imu_calibration.py"
        out_json = session / "imu_calibration.json"

        calib_cmd = [
            cfg.calib_python, str(imu_calib),
            str(session),
            "--smpl-model-path", str(cfg.smplx_model),
            "--min-samples", str(cfg.min_samples),
            "--output", str(out_json),
        ]
        if cfg.calib_frame_start is not None:
            calib_cmd += ["--frame-start", str(cfg.calib_frame_start)]
        if cfg.calib_frame_end is not None:
            calib_cmd += ["--frame-end", str(cfg.calib_frame_end)]
        if cfg.calib_utc_start_ns is not None:
            calib_cmd += ["--utc-start-ns", str(cfg.calib_utc_start_ns)]
        if cfg.calib_utc_end_ns is not None:
            calib_cmd += ["--utc-end-ns", str(cfg.calib_utc_end_ns)]
        if cfg.calib_frames_csv is not None:
            calib_cmd += ["--frames-csv", str(cfg.calib_frames_csv)]
        subprocess.run(calib_cmd, cwd=str(_PROJECT_ROOT), check=True)

        # Summary
        try:
            data = json.loads(out_json.read_text())
            joints = data.get("joints", {})
            print(f"\n✅ Calibration complete: {len(joints)} joints calibrated")
            print(f"   Output: {out_json}")
        except Exception:
            print(f"\n✅ Calibration complete: {out_json}")

    # ─────────────────────────────────────────────────────────────────
    # Client Handler
    # ─────────────────────────────────────────────────────────────────

    def _handle_client(self, sock: socket.socket):
        try:
            session: Optional[Path] = None
            session_id: Optional[str] = None
            got_video = got_meta = False
            video_path: Optional[Path] = None
            meta_path: Optional[Path] = None

            while True:
                header = self._recv(sock, 1)
                if not header:
                    break

                msg_type = header[0]

                # Control: START_IMU_RECORDING (legacy)
                if msg_type in (2, 4):
                    if msg_type == 4:
                        ts_data = self._recv(sock, 8)
                        if not ts_data:
                            print("  📡 START_IMU_RECORDING (signal 4, missing timestamp)")
                            break
                        sent_ts_ns = struct.unpack(">Q", ts_data)[0]
                        recv_ts_ns = time.time_ns()
                        delta_ms = (recv_ts_ns - sent_ts_ns) / 1e6
                        sent_iso = datetime.fromtimestamp(sent_ts_ns / 1e9, tz=timezone.utc).isoformat()
                        recv_iso = datetime.fromtimestamp(recv_ts_ns / 1e9, tz=timezone.utc).isoformat()
                        print(
                            "  📡 START_IMU_RECORDING "
                            "(signal 4)"
                        )
                        print(f"      sender_utc_ns={sent_ts_ns} ({sent_iso})")
                        print(f"      receiver_utc_ns={recv_ts_ns} ({recv_iso})")
                        print(f"      delta_ms={delta_ms:+.2f} (receiver - sender; clock skew possible)")
                    else:
                        print("  📡 START_IMU_RECORDING (signal 2, legacy: no timestamp)")
                    if session is None:
                        session = self._get_or_create_session(session_id)
                    if self.imu_recorder:
                        imu_dir = session / "imu"
                        imu_dir.mkdir(exist_ok=True)
                        self.imu_recorder.start_recording(recording_path=imu_dir)
                    continue

                # Control: START_IMU_RECORDING with session UUID (signal 5)
                if msg_type == 5:
                    ts_data = self._recv(sock, 8)
                    if not ts_data:
                        print("  📡 START_IMU_RECORDING (signal 5, missing timestamp)")
                        break
                    uuid_data = self._recv(sock, 16)
                    if not uuid_data:
                        print("  📡 START_IMU_RECORDING (signal 5, missing UUID)")
                        break
                    sent_ts_ns = struct.unpack(">Q", ts_data)[0]
                    recv_ts_ns = time.time_ns()
                    delta_ms = (recv_ts_ns - sent_ts_ns) / 1e6
                    sent_iso = datetime.fromtimestamp(sent_ts_ns / 1e9, tz=timezone.utc).isoformat()
                    recv_iso = datetime.fromtimestamp(recv_ts_ns / 1e9, tz=timezone.utc).isoformat()
                    session_id = uuid_mod.UUID(bytes=uuid_data).hex
                    print(f"  📡 START_IMU_RECORDING (signal 5, session={session_id})")
                    print(f"      sender_utc_ns={sent_ts_ns} ({sent_iso})")
                    print(f"      receiver_utc_ns={recv_ts_ns} ({recv_iso})")
                    print(f"      delta_ms={delta_ms:+.2f} (receiver - sender; clock skew possible)")
                    session = self._get_or_create_session(session_id)
                    if self.imu_recorder:
                        imu_dir = session / "imu"
                        imu_dir.mkdir(exist_ok=True)
                        if not self.imu_recorder.is_recording:
                            self.imu_recorder.start_recording(recording_path=imu_dir)
                        else:
                            print("  (IMU already recording, continuing into same session)")
                    continue

                # Control: STOP_IMU_RECORDING (legacy)
                if msg_type == 3:
                    print("  📡 STOP_IMU_RECORDING")
                    if self.imu_recorder and self.imu_recorder.is_recording:
                        self.imu_recorder.stop_recording()
                    continue

                # Control: STOP_IMU_RECORDING with session UUID (signal 6)
                if msg_type == 6:
                    uuid_data = self._recv(sock, 16)
                    if not uuid_data:
                        print("  📡 STOP_IMU_RECORDING (signal 6, missing UUID)")
                        break
                    stop_sid = uuid_mod.UUID(bytes=uuid_data).hex
                    print(f"  📡 STOP_IMU_RECORDING (signal 6, session={stop_sid})")
                    if self.imu_recorder and self.imu_recorder.is_recording:
                        self.imu_recorder.stop_recording()
                    continue

                # File transfer with session UUID (signals 10, 11)
                if msg_type in (10, 11):
                    uuid_data = self._recv(sock, 16)
                    if not uuid_data:
                        print(f"  File type {msg_type}: missing UUID")
                        break
                    file_sid = uuid_mod.UUID(bytes=uuid_data).hex
                    if session_id is None:
                        session_id = file_sid
                    # Use the session for this UUID
                    session = self._get_or_create_session(file_sid)

                    is_video = (msg_type == 10)
                    file_type = "video" if is_video else "metadata"

                    name_len_data = self._recv(sock, 4)
                    if not name_len_data:
                        break
                    name_len = struct.unpack(">I", name_len_data)[0]
                    filename = self._recv(sock, name_len).decode("utf-8")
                    file_size = struct.unpack(">Q", self._recv(sock, 8))[0]

                    out_name = f"video{os.path.splitext(filename)[1] or '.mp4'}" if is_video else "metadata.json"
                    out_path = session / out_name

                    print(f"  Receiving {file_type} (session={file_sid}): {file_size:,} bytes -> {out_name}")
                    if is_video and file_size == 0:
                        print("  ⚠️  Video payload is 0 bytes. This is not a valid MP4; skipping calibration for this session.")

                    with open(out_path, "wb") as f:
                        remaining = file_size
                        while remaining > 0:
                            chunk = self._recv(sock, min(1024 * 1024, remaining))
                            if not chunk:
                                raise IOError("Connection lost")
                            f.write(chunk)
                            remaining -= len(chunk)

                    print(f"  ✓ Saved: {out_path}")

                    if is_video:
                        video_path = out_path
                        got_video = file_size > 0
                    else:
                        meta_path = out_path
                        got_meta = file_size > 0

                    if got_video and got_meta:
                        print("  ✅ Session complete")
                        break
                    continue

                # File transfer (legacy, signals 0 and 1)
                if msg_type not in (0, 1):
                    print(f"  Unknown type: {msg_type}")
                    break

                file_type = "video" if msg_type == 0 else "metadata"

                # Read header
                name_len = struct.unpack(">I", self._recv(sock, 4))[0]
                filename = self._recv(sock, name_len).decode("utf-8")
                file_size = struct.unpack(">Q", self._recv(sock, 8))[0]

                if session is None:
                    session = self._get_or_create_session(session_id)

                out_name = f"video{os.path.splitext(filename)[1] or '.mp4'}" if msg_type == 0 else "metadata.json"
                out_path = session / out_name

                print(f"  Receiving {file_type}: {file_size:,} bytes -> {out_name}")
                if msg_type == 0 and file_size == 0:
                    print("  ⚠️  Video payload is 0 bytes. This is not a valid MP4; skipping calibration for this session.")

                # Stream to disk
                with open(out_path, "wb") as f:
                    remaining = file_size
                    while remaining > 0:
                        chunk = self._recv(sock, min(1024 * 1024, remaining))
                        if not chunk:
                            raise IOError("Connection lost")
                        f.write(chunk)
                        remaining -= len(chunk)

                print(f"  ✓ Saved: {out_path}")

                if msg_type == 0:
                    video_path = out_path
                    got_video = file_size > 0
                else:
                    meta_path = out_path
                    got_meta = file_size > 0

                if got_video and got_meta:
                    print("  ✅ Session complete")
                    break

            # Safety: stop IMU if still recording.
            # For session-aware connections (session_id set), skip the safety stop on
            # mere disconnects — the app is expected to reconnect with the same UUID and
            # the IMU should keep recording uninterrupted.  Only stop once the session
            # is fully delivered (got_video and got_meta) or for legacy connections.
            if session and self.imu_recorder and self.imu_recorder.is_recording:
                if session_id is None or (got_video and got_meta):
                    self.imu_recorder.stop_recording()

            if session and got_video and got_meta:
                self._unregister_session(session_id)
                proceed = self.calibration.auto_calibrate
                if not proceed and self.calibration.prompt:
                    proceed = _yes_no_prompt(f"\nRun calibration for {session.name}?")
                if proceed:
                    try:
                        self._run_calibration(session)
                    except subprocess.CalledProcessError as e:
                        print(f"\n❌ Calibration failed (exit {e.returncode})")
                    except Exception as e:
                        print(f"\n❌ Calibration error: {e}")

                if self.cluster.enabled:
                    proceed_upload = self.cluster.auto_upload
                    if not proceed_upload:
                        proceed_upload = _yes_no_prompt(
                            f"\nUpload {session.name} to cluster ({self.cluster.host})?",
                            default=True,
                        )
                    if proceed_upload:
                        _upload_to_cluster(session, self.cluster)
            elif session:
                # Helpful debug summary for partial / invalid sessions
                v_size = None
                m_size = None
                try:
                    if video_path and video_path.exists():
                        v_size = video_path.stat().st_size
                    if meta_path and meta_path.exists():
                        m_size = meta_path.stat().st_size
                except Exception:
                    pass

                print("\n⚠️ Session incomplete (skipping calibration):")
                print(f"   Session: {session}")
                print(f"   Video received: {got_video} (bytes={v_size})")
                print(f"   Metadata received: {got_meta} (bytes={m_size})")

        except Exception as e:
            print(f"  Error: {e}")
        finally:
            sock.close()
            print("  Connection closed")

    def _recv(self, sock: socket.socket, size: int) -> Optional[bytes]:
        data = b""
        while len(data) < size:
            chunk = sock.recv(size - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def _create_session(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session = self.output_path / f"recording_{ts}"
        session.mkdir(parents=True, exist_ok=True)
        print(f"  Created session: {session}")
        return session

    def _get_or_create_session(self, session_id: Optional[str]) -> Path:
        """Return the session directory for *session_id*, creating one if needed.

        If *session_id* is None (legacy protocol), falls back to ``_create_session``
        so that old clients still work.
        """
        if session_id is None:
            return self._create_session()

        with self._sessions_lock:
            existing = self._sessions.get(session_id)
            if existing is not None:
                return existing
            session = self._create_session()
            self._sessions[session_id] = session
            print(f"  Registered session {session_id} -> {session.name}")
            return session

    def _unregister_session(self, session_id: Optional[str]) -> None:
        if session_id is None:
            return
        with self._sessions_lock:
            self._sessions.pop(session_id, None)


def main() -> int:
    parser = argparse.ArgumentParser(description="ROSHI receiver with calibration")
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument("--output-dir", type=str, default=str(_PROJECT_ROOT / "received_recordings"))
    parser.add_argument("--imu-port", type=str, default="auto")
    parser.add_argument("--num-imus", type=int, default=9)

    # Calibration control
    parser.add_argument("--auto-calibrate", action="store_true", help="Run calibration automatically")
    parser.add_argument("--no-prompt", action="store_true", help="Never prompt")
    parser.add_argument("--session", type=str, default=None, help="Run calibration on existing session directory (skip receiver)")
    parser.add_argument(
        "--save-body-vis",
        action="store_true",
        help="Save SAM-3D-Body visualization renders to <session>/body_vis (default: disabled).",
    )

    # --local convenience flag
    parser.add_argument(
        "--local",
        action="store_true",
        help="Local-receiver mode: enable cluster upload + auto-upload, skip calibration.",
    )

    # Cluster upload (enabled by --local)
    parser.add_argument("--cluster-host", type=str, default=None,
                        help="SSH host for cluster (default from config_local.py)")
    parser.add_argument("--cluster-dir", type=str, default=None,
                        help="Remote directory on cluster (default from config_local.py)")
    parser.add_argument("--delete-after-upload", action="store_true",
                        help="Delete local session folder after successful upload to cluster")

    # Python executables (auto-detect from conda if not specified)
    parser.add_argument("--sam-python", type=str, default=None)
    parser.add_argument("--mhr-python", type=str, default=None)
    parser.add_argument("--calib-python", type=str, default=None)

    # Model paths
    parser.add_argument("--sam-checkpoint", type=str,
                        default=str(_PROJECT_ROOT / "sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt"))
    parser.add_argument("--sam-mhr-model", type=str,
                        default=str(_PROJECT_ROOT / "MHR/assets/mhr_model.pt"))
    parser.add_argument("--smplx-model", type=str,
                        default=str(_PROJECT_ROOT / "model/smplx/SMPLX_NEUTRAL.npz"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--calib-frame-start", type=int, default=None, help="Only use frames with frame_id >= this for IMU calibration.")
    parser.add_argument("--calib-frame-end", type=int, default=None, help="Only use frames with frame_id <= this for IMU calibration.")
    parser.add_argument("--calib-utc-start-ns", type=int, default=None, help="Only use frames.csv timestamps >= this for IMU calibration.")
    parser.add_argument("--calib-utc-end-ns", type=int, default=None, help="Only use frames.csv timestamps <= this for IMU calibration.")
    parser.add_argument("--calib-frames-csv", type=str, default=None, help="Optional path to frames.csv used for utc filtering (default: session_dir/frames.csv).")

    args = parser.parse_args()

    # Auto-detect Python executables
    sam_python = args.sam_python or _find_conda_python("sam_3d_body") or sys.executable
    mhr_python = args.mhr_python or _find_conda_python("mhr") or sys.executable
    calib_python = args.calib_python or _find_conda_python("roshi") or sys.executable

    print(f"SAM Python: {sam_python}")
    print(f"MHR Python: {mhr_python}")
    print(f"Calib Python: {calib_python}")
    _print_imu_joint_map()

    cfg = CalibrationConfig(
        auto_calibrate=args.auto_calibrate,
        prompt=not args.no_prompt and not args.auto_calibrate,
        sam_python=sam_python,
        mhr_python=mhr_python,
        calib_python=calib_python,
        sam_checkpoint=Path(args.sam_checkpoint),
        sam_mhr_model=Path(args.sam_mhr_model),
        smplx_model=Path(args.smplx_model),
        device=args.device,
        batch_size=args.batch_size,
        min_samples=args.min_samples,
        save_body_vis=args.save_body_vis,
        calib_frame_start=args.calib_frame_start,
        calib_frame_end=args.calib_frame_end,
        calib_utc_start_ns=args.calib_utc_start_ns,
        calib_utc_end_ns=args.calib_utc_end_ns,
        calib_frames_csv=Path(args.calib_frames_csv) if args.calib_frames_csv else None,
    )

    # Cluster upload config
    cluster_host = args.cluster_host or CLUSTER_HOST
    cluster_dir = args.cluster_dir or CLUSTER_RECEIVED_RECORDINGS_DIR

    cluster_cfg = ClusterConfig(
        enabled=args.local,
        host=cluster_host,
        remote_dir=cluster_dir,
        auto_upload=args.local,
        delete_local_after_upload=args.delete_after_upload,
    )

    # --local convenience: upload to cluster, skip calibration
    if args.local:
        cfg.auto_calibrate = False
        cfg.prompt = False

    # Validate cluster config if upload is enabled
    if cluster_cfg.enabled:
        if not cluster_cfg.host:
            print("Error: --cluster-host not specified and CLUSTER_HOST not set in config_local.py")
            return 1
        if not cluster_cfg.remote_dir:
            print("Error: --cluster-dir not specified and CLUSTER_RECEIVED_RECORDINGS_DIR not set in config_local.py")
            return 1

    # If --session is provided, run calibration directly on existing session
    if args.session:
        session_path = Path(args.session).resolve()
        if not session_path.exists():
            print(f"Error: Session directory not found: {session_path}")
            return 1
        print(f"\n=== Running calibration on existing session: {session_path} ===\n")
        # Create a temporary receiver just to call _run_calibration
        receiver = ROSHICalibrationReceiver(
            port=args.port,
            output_dir=str(session_path.parent),
            imu_port=None,
            num_imus=args.num_imus,
            calibration=cfg,
            cluster=cluster_cfg,
        )
        try:
            receiver._run_calibration(session_path)
            print("\n✅ Calibration pipeline complete!")
            return 0
        except Exception as e:
            print(f"\n❌ Calibration failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    imu_port = None if args.imu_port in (None, "auto") else args.imu_port

    receiver = ROSHICalibrationReceiver(
        port=args.port,
        output_dir=args.output_dir,
        imu_port=imu_port,
        num_imus=args.num_imus,
        calibration=cfg,
        cluster=cluster_cfg,
    )

    if cluster_cfg.enabled:
        print(f"\n📤 Cluster upload enabled:")
        print(f"   Host: {cluster_cfg.host}")
        print(f"   Remote dir: {cluster_cfg.remote_dir}")
        if cluster_cfg.auto_upload:
            print("   Mode: Auto-upload (no prompts)")
        if cluster_cfg.delete_local_after_upload:
            print("   Local files will be deleted after upload")
        print()

    receiver.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
