#!/usr/bin/env python3
"""
IMU-only pose reconstruction + viser visualization.

Workflow:
1) (Optional) Compute bone↤sensor offsets (B_R_S) from SMPLX + AprilTags:
     python imu_calibration.py <session_dir> ...
   This uses *all frames* by default.

2) Apply those offsets to recorded IMU quaternions to estimate bone rotations:
     W_R_B(t) ≈ W_R_S(t) @ (B_R_S)^T

3) Build a SMPLX pose with:
   - root (pelvis) driven by pelvis IMU
   - hips/knees driven by hip/knee IMUs
   - shoulders/elbows driven by shoulder/elbow IMUs
   - all other joints set to identity (torso rigid w/ pelvis)

Then visualize in viser.

IMPORTANT LIMITATIONS / ASSUMPTIONS:
- We assume the AprilTag frame used during calibration is rigidly aligned to the IMU sensor frame.
  If Tag axes != IMU PCB axes, you need an additional fixed R_tag_to_imu (see README.md).
- We assume the IMU quaternion->matrix conversion corresponds to world<-sensor (W_R_S).
  If your IMU outputs the inverse, pass --quat-is-world-from-sensor false.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import csv
import json
import os
import pickle
import re
import subprocess
import time
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


import viser
import viser.transforms as tf

from utils.imu_id_mapping import (
    OPTIMIZATION_IMU_ID_TO_ROSHI_JOINT,
    IMU_ID_TO_JOINT,
    JOINT_NAMES,
    JOINT_TO_TAG_ID,
    TAG_ID_TO_JOINT,
    SMPLX_JOINT_INDEX_MAP,
)
from utils.apriltag_utils import (
    rot_angle_deg,
    quaternion_to_matrix_wxyz,
    average_rotations_quaternion,
    sample_hold_1d,
    load_apriltag_rotations_by_time,
    estimate_world_alignment_from_tags,
)
from utils.sync_utils import (
    load_imu_csv,
    load_calibration,
    compute_time_intersection_ns,
    sample_streams_hold,
)
from utils.smpl_utils import (
    SmplxModel,
    load_smplx_model,
    precompute_shape,
    smplx_forward_kinematics,
    build_local_rots_from_imu,
)

# SMPL/SMPLX joint mapping for TTO local-rotation CSVs
TTO_JOINT_TO_SMPLX_INDEX = {
    "hips": 0,
    "leftUpLeg": 1,
    "rightUpLeg": 2,
    "spine": 3,
    "leftLeg": 4,
    "rightLeg": 5,
    "spine1": 6,
    "leftFoot": 7,
    "rightFoot": 8,
    "spine2": 9,
    "leftToeBase": 10,
    "rightToeBase": 11,
    "neck": 12,
    "leftShoulder": 13,
    "rightShoulder": 14,
    "head": 15,
    "leftArm": 16,
    "rightArm": 17,
    "leftForeArm": 18,
    "rightForeArm": 19,
    "leftHand": 20,
    "rightHand": 21,
}






def _str2bool(v: object) -> bool:
    """argparse-friendly bool parser that accepts true/false strings."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean (true/false), got: {v!r}")


@dataclass(frozen=True)
class SmplGroundTruth:
    """
    Ground-truth SMPLX sequence produced by SAM-3D-Body + MHR->SMPLX conversion.

    All arrays are in the SMPL output coordinate system (the one used in smpl_output/*.npz).
    """

    frame_id_to_index: Dict[int, int]
    vertices: np.ndarray  # (T,V,3) (can be memmap)
    joints: np.ndarray  # (T,127,3)
    joint_rotations_local: np.ndarray  # (T,55,3,3) local rotations

    def has_frame(self, frame_id: int) -> bool:
        return int(frame_id) in self.frame_id_to_index

    def index_of(self, frame_id: int) -> Optional[int]:
        return self.frame_id_to_index.get(int(frame_id))


def load_frames_csv(session_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load frames.csv -> (frame_ids, utc_timestamp_ns, color_paths), sorted by frame_id.
    """
    frames_csv = session_dir / "frames.csv"
    if not frames_csv.exists():
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), []

    ids = []
    ts = []
    paths: List[str] = []
    with open(frames_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                ids.append(int(row["frame_id"]))
                ts.append(int(row["utc_timestamp_ns"]))
                paths.append(str(row.get("color_path", "")))
            except Exception:
                continue
    if not ids:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), []

    order = np.argsort(np.array(ids, dtype=np.int64))
    ids_arr = np.array(ids, dtype=np.int64)[order]
    ts_arr = np.array(ts, dtype=np.int64)[order]
    paths_arr = [paths[i] for i in order.tolist()]
    return ids_arr, ts_arr, paths_arr


def load_smpl_ground_truth(session_dir: Path) -> Optional[SmplGroundTruth]:
    """
    Load SMPLX ground truth from a session folder.

    Uses:
      - smpl_output/smpl_parameters.npz (joints + joint_rotations + frame_names)
      - smpl_output/smpl_vertices.npy (vertices sequence; mmap)
    """
    smpl_dir = session_dir / "smpl_output"
    params_path = smpl_dir / "smpl_parameters.npz"
    verts_path = smpl_dir / "smpl_vertices.npy"
    if not params_path.exists() or not verts_path.exists():
        return None

    params = np.load(params_path, allow_pickle=True)
    joints = params["joints"].astype(np.float32)  # (T,127,3)
    joint_rots = params["joint_rotations"].astype(np.float32)  # (T,55,3,3)
    frame_names = params.get("frame_names")

    # frame_id -> index mapping
    frame_id_to_idx: Dict[int, int] = {}
    if frame_names is not None:
        for i, name in enumerate(frame_names):
            m = re.search(r"frame_(\d+)", str(name))
            if m:
                frame_id_to_idx[int(m.group(1))] = int(i)
    # Fallback: assume index == frame_id for [0..T-1]
    if not frame_id_to_idx:
        frame_id_to_idx = {int(i): int(i) for i in range(int(joints.shape[0]))}

    vertices = np.load(verts_path, mmap_mode="r")  # (T,V,3)

    return SmplGroundTruth(
        frame_id_to_index=frame_id_to_idx,
        vertices=vertices,
        joints=joints,
        joint_rotations_local=joint_rots,
    )


# --- Sync data loading functions (for --sync-dir option) ---

def load_sync_imu_info_csv(sync_dir: Path) -> Dict[int, Dict[int, np.ndarray]]:
    """Load imu_info.csv from sync folder -> {utc_ns: {imu_id: rot_matrix}}."""
    csv_path = sync_dir / "imu_info.csv"
    if not csv_path.exists():
        return {}
    out: Dict[int, Dict[int, np.ndarray]] = {}
    with open(csv_path, "r", newline="") as f:
        header = f.readline().strip().split(",")
        col_idx = {name: i for i, name in enumerate(header)}
        for line in f:
            parts = line.strip().split(",", 2)
            try:
                t_ns = int(parts[col_idx["utc_timestamp_ns"]])
                imu_id = int(parts[col_idx["imu_id"]])
                rot = np.array(ast.literal_eval(parts[col_idx["rot_matrix"]]), dtype=np.float64)
            except Exception:
                continue
            out.setdefault(t_ns, {})[imu_id] = rot
    return out


def load_sync_imu_info_pkl(sync_dir: Path) -> Dict[int, Dict[int, np.ndarray]]:
    """Load imu_info.pkl from sync folder -> {utc_ns: {imu_id: rot_matrix}}."""
    pkl_path = sync_dir / "imu_info.pkl"
    if not pkl_path.exists():
        return {}
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    out: Dict[int, Dict[int, np.ndarray]] = {}
    for t, by_imu in data.items():
        out[int(t)] = {int(k): np.array(v, dtype=np.float64) for k, v in by_imu.items()}
    return out


def get_nearest_sync_timestamp(sorted_t: List[int], t_ns: int) -> Optional[int]:
    """Find nearest timestamp in sorted list."""
    if not sorted_t:
        return None
    idx = bisect.bisect_left(sorted_t, t_ns)
    if idx == 0:
        return sorted_t[0]
    if idx >= len(sorted_t):
        return sorted_t[-1]
    left = sorted_t[idx - 1]
    right = sorted_t[idx]
    return left if abs(left - t_ns) <= abs(right - t_ns) else right


def find_tto_csv(session_dir: Path) -> Optional[Path]:
    tto_dir = session_dir / "tto"
    if not tto_dir.exists():
        return None
    matches = sorted(tto_dir.glob("*.csv"))
    return matches[0] if matches else None


def find_egoallo_csv(session_dir: Path) -> Optional[Path]:
    ego_dir = session_dir / "egoallo"
    if not ego_dir.exists():
        return None
    matches = sorted(ego_dir.glob("*.csv"))
    return matches[0] if matches else None


def load_tto_local_rotations(csv_path: Path, num_joints: int) -> Dict[int, np.ndarray]:
    """Load TTO joint local rotations -> {utc_ns: (J,3,3)}."""
    out: Dict[int, np.ndarray] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t_ns = int(row["utc_timestamp_ns"])
                joint_name = row["joint_name"]
                rot = np.array(ast.literal_eval(row["rot_matrix"]), dtype=np.float64)
            except Exception:
                continue
            jidx = TTO_JOINT_TO_SMPLX_INDEX.get(joint_name)
            if jidx is None:
                continue
            if t_ns not in out:
                out[t_ns] = np.tile(np.eye(3, dtype=np.float64), (num_joints, 1, 1))
            out[t_ns][jidx] = rot
    return out


def ensure_calibration(
    session_dir: Path,
    smplx_model_path: Path,
    output_json: Path,
    *,
    min_samples: int = 10,
    calib_duration_sec: Optional[float] = None,
    python_exe: str = "python",
) -> None:
    if output_json.exists():
        return
    imu_calib_py = session_dir.parent.parent / "02_imu_calibration.py"
    if not imu_calib_py.exists():
        # fall back to cwd resolution
        imu_calib_py = Path(__file__).resolve().parent / "02_imu_calibration.py"
    cmd = [
        python_exe,
        str(imu_calib_py),
        str(session_dir),
        "--smpl-model-path",
        str(smplx_model_path),
        "--min-samples",
        str(min_samples),
        "--output",
        str(output_json),
    ]
    if calib_duration_sec is not None:
        cmd.extend(["--calib-duration-sec", str(calib_duration_sec)])
    duration_msg = f"first {calib_duration_sec}s" if calib_duration_sec else "all frames"
    print(f"\n[calib] imu_calibration.json not found, computing it now ({duration_msg})...")
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="IMU-only pose reconstruction viewer (viser)")
    parser.add_argument("session_dir", type=Path, help="Session directory (received_recordings/recording_*/)")
    parser.add_argument("--smplx-model", type=Path, default=Path(__file__).resolve().parent / "model/smplx/SMPLX_NEUTRAL.npz")
    parser.add_argument("--imu-csv", type=Path, default=None, help="Path to imu_data.csv (default: session_dir/imu/imu_data.csv)")
    parser.add_argument("--imu-calibration", type=Path, default=None, help="Path to imu_calibration.json (default: session_dir/imu_calibration.json)")
    parser.add_argument("--compute-calibration", action="store_true", help="If imu_calibration.json is missing, compute it.")
    parser.add_argument("--min-samples", type=int, default=10, help="Min samples per joint for calibration (if computed).")
    parser.add_argument("--calib-duration-sec", type=float, default=None, help="Use first N seconds for calibration. E.g., --calib-duration-sec 46")
    parser.add_argument("--port", type=int, default=8082, help="Viser server port")
    parser.add_argument("--rate-hz", type=float, default=30.0, help="Playback sampling rate (Hz) from IMU data")
    parser.add_argument(
        "--up-direction",
        choices=["+y", "-y"],
        default="-y",
        help="Viser 'up' direction. Use '-y' for OpenCV-like camera coordinates (y down).",
    )
    parser.add_argument(
        "--imu-time-offset-ms",
        type=float,
        default=0.0,
        help="Time offset added to video frame timestamps when sampling IMUs (ms). Useful if phone/workstation clocks are offset.",
    )
    parser.add_argument(
        "--quat-is-world-from-sensor",
        nargs="?",
        const=True,
        default=True,
        type=_str2bool,
        help=(
            "Set to false if your IMU quaternion represents sensor<-world (the inverse). "
            "Default: true (expects world<-sensor). Example: --quat-is-world-from-sensor false"
        ),
    )
    parser.add_argument(
        "--timeline",
        choices=["auto", "imu", "video"],
        default="auto",
        help=(
            "Which timeline to play back. "
            "'imu' uses a fixed-rate timeline from IMU timestamps. "
            "'video' uses frames.csv timestamps (recommended). "
            "'auto' defaults to 'video'."
        ),
    )
    parser.add_argument("--gt-offset-x", type=float, default=-1.2, help="X offset applied to GT mesh/joints for side-by-side viewing.")
    parser.add_argument("--sync-offset-x", type=float, default=0.0, help="X offset applied to Sync mesh/joints (center).")
    parser.add_argument("--imu-offset-x", type=float, default=1.2, help="X offset applied to IMU mesh/joints for side-by-side viewing.")
    parser.add_argument(
        "--tto-csv",
        type=Path,
        default=None,
        help="Path to *.csv for extra mesh (defaults to session/tto/*.csv).",
    )
    parser.add_argument(
        "--egoallo-csv",
        type=Path,
        default=None,
        help="Path to EgoAllo *.csv (defaults to session/egoallo/*.csv).",
    )
    parser.add_argument("--show-sync", action="store_true", help="Enable sync mesh. Auto-detects <session_dir>/sync unless --sync-dir is given.")
    parser.add_argument("--sync-dir", type=Path, default=None, help="Path to sync folder (implies --show-sync).")
    parser.add_argument("--video-max-width", type=int, default=640, help="Max width (px) for displayed video frames (downscaled).")
    parser.add_argument("--share", action="store_true", help="Create a publicly-accessible URL via viser relay (no SSH tunnel needed).")

    args = parser.parse_args()
    session = args.session_dir.resolve()
    imu_csv = args.imu_csv or (session / "imu" / "imu_data.csv")
    calib_json = args.imu_calibration or (session / "imu_calibration.json")

    if not imu_csv.exists():
        print(f"Error: IMU CSV not found: {imu_csv}")
        return 1

    if args.compute_calibration:
        ensure_calibration(
            session_dir=session,
            smplx_model_path=args.smplx_model,
            output_json=calib_json,
            min_samples=args.min_samples,
            calib_duration_sec=args.calib_duration_sec,
            python_exe=os.environ.get("PYTHON", "python"),
        )

    if not calib_json.exists():
        print(f"Error: imu_calibration.json not found: {calib_json}")
        print("Run:\n  python imu_calibration.py <session_dir> --output <session_dir>/imu_calibration.json")
        return 1

    # Load data
    print(f"Loading IMU data: {imu_csv}")
    streams = load_imu_csv(imu_csv)
    print(f"  IMUs present: {sorted(streams.keys())}")
    print(f"Loading calibration: {calib_json}")
    calib, saved_calib_duration_sec = load_calibration(calib_json)
    print(f"  Calibrated joints: {sorted(calib.keys())}")
    # Use saved calibration duration from JSON if not explicitly set via CLI
    if args.calib_duration_sec is None and saved_calib_duration_sec is not None:
        args.calib_duration_sec = saved_calib_duration_sec
        print(f"  Using calibration window from JSON: first {saved_calib_duration_sec:.1f}s")

    # Warn early if calibration is incomplete for the joints we drive from IMUs.
    expected = set(JOINT_NAMES)
    missing = sorted(expected - set(calib.keys()))
    if missing:
        print("\n⚠️  Calibration missing for these joints:")
        print("   " + ", ".join(missing))
        print("   Those joints will fall back to raw IMU orientation and may look wrong/flipped.")
        print("   Fix: re-run calibration with a longer window (or all frames), e.g.:")
        print(f"     python imu_calibration.py {session} --calib-duration-sec 1000")

    # Default joint->IMU mapping
    joint_to_imu: Dict[str, int] = {joint: imu_id for imu_id, joint in IMU_ID_TO_JOINT.items()}

    imu_time_offset_ns = int(round(args.imu_time_offset_ms * 1e6))

    # Convert calibration offsets to bone<-imu offsets.
    # Fixed axis mapping (Tag↔IMU):
    #   x_sensor = -y_tag
    #   y_sensor = -x_tag
    #   z_sensor = -z_tag
    #
    # This means (Tag <- IMU) has columns equal to IMU axes expressed in Tag:
    #   T_R_imu = [x_imu^T, y_imu^T, z_imu^T]
    T_R_IMU = np.array(
        [
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    bone_from_imu_offset: Dict[str, np.ndarray] = {}
    for joint, B_R_tag in calib.items():
        bone_from_imu_offset[joint] = B_R_tag @ T_R_IMU
    # Needed for world alignment: S_R_T (sensor<-tag).
    sensor_from_tag = T_R_IMU.T
    print("\nUsing fixed Tag↔IMU mapping for all joints (bone<-tag -> bone<-imu).")

    # Align per-IMU "world" frames using AprilTag rotations.
    world_align: Dict[str, np.ndarray] = {}
    tag_rots_by_time = load_apriltag_rotations_by_time(session)
    if not tag_rots_by_time:
        print("\n⚠️  AprilTag detections missing; world alignment skipped.")
    else:
        world_align, world_stats = estimate_world_alignment_from_tags(
            streams=streams,
            tag_rots_by_time=tag_rots_by_time,
            joint_to_imu=joint_to_imu,
            imu_time_offset_ns=imu_time_offset_ns,
            sensor_from_tag=sensor_from_tag,
            quat_is_world_from_sensor=bool(args.quat_is_world_from_sensor),
        )
        if world_align:
            print("\nWorld alignment into pelvis world (using tags):")
            for jn in JOINT_NAMES:
                if jn in world_stats:
                    n, std = world_stats[jn]
                    print(f"  {jn:14s}: samples={n:4d}  std={std:5.2f}°")
        else:
            print("\n⚠️  World alignment could not be estimated; skipping.")

    # Load SMPLX betas (optional, use first-frame betas if available)
    betas = np.zeros((10,), dtype=np.float32)
    smpl_params = session / "smpl_output" / "smpl_parameters.npz"
    if smpl_params.exists():
        try:
            d = np.load(smpl_params, allow_pickle=True)
            if "betas" in d and d["betas"].ndim == 2 and d["betas"].shape[1] >= 10:
                betas = d["betas"][0, :10].astype(np.float32)
        except Exception:
            pass

    model = load_smplx_model(args.smplx_model, betas_dim=10)
    cached_v_shaped, cached_j_tpose = precompute_shape(model, betas)

    # Load SMPL ground truth (required)
    gt = load_smpl_ground_truth(session)
    if gt is None:
        raise FileNotFoundError(
            f"SMPL ground truth not found under {session}/smpl_output/*.\n"
            "Run the calibration pipeline first to generate SMPL outputs."
        )
    print(f"Loaded SMPL ground truth: vertices {gt.vertices.shape}, joints {gt.joints.shape}")

    # Load sync data (optional - only when --show-sync or --sync-dir is given)
    sync_dir = args.sync_dir
    want_sync = args.show_sync or sync_dir is not None
    if want_sync and sync_dir is None:
        # Auto-detect <session>/sync
        default_sync = session / "sync"
        if default_sync.exists():
            sync_dir = default_sync

    sync_imu_dict: Dict[int, Dict[int, np.ndarray]] = {}
    sync_t_sorted: List[int] = []
    if want_sync and sync_dir is not None and sync_dir.exists():
        sync_imu_dict = load_sync_imu_info_csv(sync_dir)
        if not sync_imu_dict:
            sync_imu_dict = load_sync_imu_info_pkl(sync_dir)
        if sync_imu_dict:
            sync_t_sorted = sorted(sync_imu_dict.keys())
            print(f"Loaded sync data from {sync_dir}: {len(sync_imu_dict)} timestamps")
        else:
            print(f"Warning: sync dir exists but no imu_info.csv/pkl found: {sync_dir}")
    elif want_sync and (sync_dir is None or not sync_dir.exists()):
        print(f"Warning: --show-sync requested but sync dir not found (tried {sync_dir or session / 'sync'})")

    show_sync = len(sync_imu_dict) > 0

    # Load TTO local-rotation CSV (optional - extra mesh)
    tto_csv = args.tto_csv or find_tto_csv(session)
    tto_local_dict: Dict[int, np.ndarray] = {}
    tto_t_sorted: List[int] = []
    if tto_csv is not None and tto_csv.exists():
        tto_local_dict = load_tto_local_rotations(tto_csv, model.num_joints)
        if tto_local_dict:
            tto_t_sorted = sorted(tto_local_dict.keys())
            print(f"Loaded TTO local rotations from {tto_csv}: {len(tto_local_dict)} timestamps")
        else:
            print(f"Warning: TTO CSV found but empty/invalid: {tto_csv}")
    elif tto_csv is not None:
        print(f"Warning: --tto-csv specified but does not exist: {tto_csv}")

    show_tto = len(tto_local_dict) > 0

    # Load EgoAllo original local-rotation CSV (optional - extra mesh)
    ego_csv = args.egoallo_csv or find_egoallo_csv(session)
    ego_local_dict: Dict[int, np.ndarray] = {}
    ego_t_sorted: List[int] = []
    if ego_csv is not None and ego_csv.exists():
        ego_local_dict = load_tto_local_rotations(ego_csv, model.num_joints)
        if ego_local_dict:
            ego_t_sorted = sorted(ego_local_dict.keys())
            print(f"Loaded EgoAllo original rotations from {ego_csv}: {len(ego_local_dict)} timestamps")
        else:
            print(f"Warning: EgoAllo CSV found but empty/invalid: {ego_csv}")
    elif ego_csv is not None:
        print(f"Warning: --egoallo-csv specified but does not exist: {ego_csv}")

    show_ego = len(ego_local_dict) > 0

    # Decide timeline
    timeline_mode = args.timeline
    if timeline_mode == "auto":
        timeline_mode = "video"

    expected_imus = sorted(IMU_ID_TO_JOINT.keys())

    # Load video frame metadata (for timeline + GUI video display)
    all_frame_ids, all_frame_times, all_color_paths = load_frames_csv(session)
    frame_id_to_color_path: Dict[int, Path] = {
        int(fid): (session / rel) for fid, rel in zip(all_frame_ids.tolist(), all_color_paths) if rel
    }
    show_video = True
    if all_frame_ids.size == 0:
        print("frames.csv not found/empty; disabling GUI video display.")
        show_video = False

    if timeline_mode == "video":
        frame_ids = all_frame_ids
        frame_times = all_frame_times
        frame_paths = all_color_paths
        if frame_ids.size == 0:
            raise ValueError("Requested video timeline, but frames.csv is missing or empty.")

        mask = np.array([gt.has_frame(int(fid)) for fid in frame_ids], dtype=bool)
        frame_ids = frame_ids[mask]
        frame_times = frame_times[mask]
        frame_paths = [p for p, m in zip(frame_paths, mask.tolist()) if m]
        if frame_ids.size == 0:
            raise ValueError("No overlap between frames.csv and SMPL ground truth frame_ids.")

        # Ensure IMU coverage for these timestamps (after applying optional time offset).
        t0, t1 = compute_time_intersection_ns(streams, expected_imus)
        times_for_imu = frame_times.astype(np.int64) + imu_time_offset_ns
        mask_imu = (times_for_imu >= t0) & (times_for_imu <= t1)
        frame_ids = frame_ids[mask_imu]
        frame_times = frame_times[mask_imu]
        frame_paths = [p for p, m in zip(frame_paths, mask_imu.tolist()) if m]

        timeline_frame_ids = frame_ids
        timeline = frame_times
        timeline_color_paths: List[Optional[Path]] = [(session / p) if p else None for p in frame_paths]
        print(f"Timeline (video): {timeline.size} frames from frames.csv")
    else:
        t0, t1 = compute_time_intersection_ns(streams, expected_imus)
        step_ns = int(round(1_000_000_000 / max(1e-3, args.rate_hz)))
        timeline = np.arange(t0, t1, step_ns, dtype=np.int64)
        timeline_frame_ids = np.arange(timeline.size, dtype=np.int64)
        # For IMU timeline, pick nearest video frame (if available) for display.
        timeline_color_paths = []
        if all_frame_times.size > 0:
            idxs = np.searchsorted(all_frame_times, timeline, side="left")
            for t, idx in zip(timeline.tolist(), idxs.tolist()):
                cand = []
                if 0 <= idx < all_frame_times.size:
                    cand.append(idx)
                if 0 <= idx - 1 < all_frame_times.size:
                    cand.append(idx - 1)
                if not cand:
                    timeline_color_paths.append(None)
                    continue
                best = min(cand, key=lambda j: abs(int(all_frame_times[j]) - int(t)))
                fid = int(all_frame_ids[best])
                timeline_color_paths.append(frame_id_to_color_path.get(fid))
        else:
            timeline_color_paths = [None] * int(timeline.size)
        print(f"Timeline (imu): {timeline.size} frames @ {args.rate_hz:.1f} Hz  (t0={t0}, t1={t1})")

    # Sample IMUs on either the IMU timeline, or the video timeline (+ offset).
    timeline_for_imu = timeline.astype(np.int64)
    if timeline_mode == "video" and imu_time_offset_ns != 0:
        timeline_for_imu = timeline_for_imu + imu_time_offset_ns
    sampled = sample_streams_hold(streams, expected_imus, timeline_for_imu)

    # Preload GT vertices into RAM (avoids per-frame memmap random I/O)
    gt_verts_preloaded: Optional[np.ndarray] = None
    gt_preload_map: Dict[int, int] = {}
    if gt is not None:
        print("Preloading GT vertices into RAM...", end=" ", flush=True)
        gt_indices = [gt.index_of(int(fid)) for fid in timeline_frame_ids]
        valid_gt_indices = [i for i in gt_indices if i is not None]
        if valid_gt_indices:
            gt_verts_preloaded = np.stack(
                [np.array(gt.vertices[i]) for i in valid_gt_indices],
                axis=0,
            ).astype(np.float32)  # (T_valid, V, 3)
            # Build mapping: timeline frame index -> preloaded array index
            preload_idx = 0
            for ti, gi in enumerate(gt_indices):
                if gi is not None:
                    gt_preload_map[ti] = preload_idx
                    preload_idx += 1
            mb = gt_verts_preloaded.nbytes / (1024 * 1024)
            print(f"{gt_verts_preloaded.shape[0]} frames, {mb:.0f} MB")
        else:
            print("no valid GT frames found")

    # Compute calibration end timestamp for UI indicator
    timeline_start_ns = int(timeline[0]) if timeline.size > 0 else 0
    calib_end_ns: Optional[int] = None
    if args.calib_duration_sec is not None:
        calib_end_ns = int(timeline_start_ns + args.calib_duration_sec * 1e9)
        print(f"Calibration segment: first {args.calib_duration_sec}s (until t_ns={calib_end_ns})")

    # Pre-create viser scene
    server = viser.ViserServer(port=args.port, share=args.share)

    if args.share:
        # share=True prints a public URL automatically; no tunnel needed.
        print("\nShare mode enabled — use the public URL printed above.")
    else:
        # Viser listens on args.port for HTTP/WS.
        viser_port = args.port
        print(f"\n{'='*60}")
        print(f"  Viser server started on port {viser_port}")
        print(f"{'='*60}")
        print(f"\n  Local access:  http://localhost:{viser_port}")

        ssh_connection = os.environ.get("SSH_CONNECTION", "")
        if ssh_connection:
            parts = ssh_connection.split()
            if len(parts) >= 3:
                server_ip = parts[2]
                user = os.environ.get("USER", "user")
                print(f"\n  SSH detected! Run on your LOCAL machine:")
                print(f"    ssh -NL {viser_port}:localhost:{viser_port} {user}@{server_ip}")
                print(f"  Then open:  http://127.0.0.1:{viser_port}")
                print(f"\n  Tip: use --share next time to skip the tunnel entirely.")
        else:
            print(f"\n  Remote? Either:")
            print(f"    1) Re-run with --share  (easiest, creates public URL)")
            print(f"    2) ssh -NL {viser_port}:localhost:{viser_port} <user>@<remote-ip>")
        print(f"{'='*60}\n")
    server.scene.set_up_direction(args.up_direction)
    # In OpenCV-like camera coordinates, +y points down, so put the ground at +y.
    grid_y = 1.2 if args.up_direction == "-y" else -1.2
    server.scene.add_grid("/grid", position=(0.0, grid_y, 0.0), plane="xz")

    # Display offsets: left-to-right: GT, IMU, EgoAllo, TTO  (sync further right if present)
    gt_display_x = float(args.gt_offset_x)                          # -1.2
    imu_display_x = 0.0                                              #  0.0
    ego_display_x = float(abs(args.imu_offset_x))                   #  1.2
    tto_display_x = float(abs(args.imu_offset_x) * 2.0)             #  2.4
    sync_display_x = float(abs(args.imu_offset_x) * 3.0)            #  3.6

    # Initialize with rest pose mesh
    rest_local = np.tile(np.eye(3, dtype=np.float32), (model.num_joints, 1, 1))
    joints_w, _, verts_w = smplx_forward_kinematics(
        model, rest_local, betas, compute_vertices=True,
        v_shaped=cached_v_shaped, j_tpose=cached_j_tpose,
    )
    pelvis_pos = joints_w[0].copy()
    verts_w = verts_w - pelvis_pos[None, :]
    verts_w = verts_w + np.array([imu_display_x, 0.0, 0.0], dtype=np.float32)[None, :]
    imu_mesh_handle = server.scene.add_mesh_simple(
        "/imu_mesh",
        vertices=verts_w.astype(np.float32),
        faces=model.faces.astype(np.uint32),
        color=(0.35, 0.75, 0.95),
        wireframe=False,
        opacity=0.55,
    )
    imu_label = server.scene.add_label(
        name="/imu_mesh_label",
        text="IMU mesh",
        position=(float(imu_display_x), 1.2, 0.0),
        visible=True,
    )

    gt_mesh_handle = None
    fid0 = int(timeline_frame_ids[0])
    gt_idx0 = gt.index_of(fid0)
    if gt_idx0 is not None:
        v0 = np.array(gt.vertices[gt_idx0], dtype=np.float32)
        p0 = np.array(gt.joints[gt_idx0, 0], dtype=np.float32)
        v0 = v0 - p0[None, :]
        v0 = v0 + np.array([gt_display_x, 0.0, 0.0], dtype=np.float32)[None, :]
        gt_mesh_handle = server.scene.add_mesh_simple(
            "/gt_mesh",
            vertices=v0,
            faces=model.faces.astype(np.uint32),
            color=(0.95, 0.55, 0.20),
            wireframe=False,
            opacity=0.45,
        )
        gt_label = server.scene.add_label(
            name="/gt_mesh_label",
            text="GT mesh",
            position=(float(gt_display_x), 1.2, 0.0),
            visible=True,
        )

    # Sync mesh (green, center) - from pre-exported sync data
    sync_mesh_handle = None
    sync_joint_frames = {}
    sync_joints_handle = None
    if show_sync:
        # Initialize with rest pose
        v_sync = verts_w.copy() - np.array([imu_display_x, 0.0, 0.0], dtype=np.float32)[None, :]
        v_sync = v_sync + np.array([sync_display_x, 0.0, 0.0], dtype=np.float32)[None, :]
        sync_mesh_handle = server.scene.add_mesh_simple(
            "/sync_mesh",
            vertices=v_sync.astype(np.float32),
            faces=model.faces.astype(np.uint32),
            color=(0.35, 0.95, 0.45),  # Green
            wireframe=False,
            opacity=0.55,
        )
        sync_label = server.scene.add_label(
            name="/sync_mesh_label",
            text="Sync mesh",
            position=(float(sync_display_x), 1.2, 0.0),
            visible=True,
        )
        for name, idx in SMPLX_JOINT_INDEX_MAP.items():
            sync_joint_frames[name] = server.scene.add_frame(
                f"/sync_joints/{name}",
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
                axes_length=0.06,
                axes_radius=0.003,
            )

    # TTO mesh (purple, other side of GT) - from joint_local_rot CSV
    tto_mesh_handle = None
    tto_joint_frames = {}
    tto_joints_handle = None
    if show_tto:
        v_tto = verts_w.copy() - np.array([imu_display_x, 0.0, 0.0], dtype=np.float32)[None, :]
        v_tto = v_tto + np.array([tto_display_x, 0.0, 0.0], dtype=np.float32)[None, :]
        tto_mesh_handle = server.scene.add_mesh_simple(
            "/tto_mesh",
            vertices=v_tto.astype(np.float32),
            faces=model.faces.astype(np.uint32),
            color=(0.6, 0.4, 0.95),  # Purple
            wireframe=False,
            opacity=0.55,
        )
        tto_label = server.scene.add_label(
            name="/tto_mesh_label",
            text="TTO mesh",
            position=(float(tto_display_x), 1.2, 0.0),
            visible=True,
        )
        tto_joints_handle = server.scene.add_point_cloud(
            "/tto_joints_all",
            points=np.zeros((model.num_joints, 3), dtype=np.float32),
            colors=np.tile(np.array([[0.7, 0.5, 0.95]], dtype=np.float32), (model.num_joints, 1)),
            point_size=0.01,
        )
        for name, idx in SMPLX_JOINT_INDEX_MAP.items():
            tto_joint_frames[name] = server.scene.add_frame(
                f"/tto_joints/{name}",
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
                axes_length=0.06,
                axes_radius=0.003,
            )

    # EgoAllo original mesh (magenta) - other side of TTO
    ego_mesh_handle = None
    ego_joint_frames = {}
    ego_joints_handle = None
    if show_ego:
        v_ego = verts_w.copy() - np.array([imu_display_x, 0.0, 0.0], dtype=np.float32)[None, :]
        v_ego = v_ego + np.array([ego_display_x, 0.0, 0.0], dtype=np.float32)[None, :]
        ego_mesh_handle = server.scene.add_mesh_simple(
            "/egoallo_mesh",
            vertices=v_ego.astype(np.float32),
            faces=model.faces.astype(np.uint32),
            color=(0.9, 0.4, 0.85),  # Magenta
            wireframe=False,
            opacity=0.55,
        )
        ego_label = server.scene.add_label(
            name="/egoallo_mesh_label",
            text="EgoAllo mesh",
            position=(float(ego_display_x), 1.2, 0.0),
            visible=True,
        )
        ego_joints_handle = server.scene.add_point_cloud(
            "/egoallo_joints_all",
            points=np.zeros((model.num_joints, 3), dtype=np.float32),
            colors=np.tile(np.array([[0.9, 0.5, 0.9]], dtype=np.float32), (model.num_joints, 1)),
            point_size=0.01,
        )
        for name, idx in SMPLX_JOINT_INDEX_MAP.items():
            ego_joint_frames[name] = server.scene.add_frame(
                f"/egoallo_joints/{name}",
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
                axes_length=0.06,
                axes_radius=0.003,
            )
        sync_joints_handle = server.scene.add_point_cloud(
            "/sync_joints_all",
            points=np.zeros((model.num_joints, 3), dtype=np.float32),
            colors=np.tile(np.array([[0.2, 0.8, 0.3]], dtype=np.float32), (model.num_joints, 1)),
            point_size=0.01,
        )
        print("Sync visualization enabled (green mesh, center)")

    # Frames for key joints
    joint_frames = {}
    for name, idx in SMPLX_JOINT_INDEX_MAP.items():
        joint_frames[name] = server.scene.add_frame(
            f"/imu_joints/{name}",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            axes_length=0.06,
            axes_radius=0.003,
        )

    # Joint point cloud (all 55 joints)
    joints_handle = server.scene.add_point_cloud(
        "/imu_joints_all",
        points=np.zeros((model.num_joints, 3), dtype=np.float32),
        colors=np.tile(np.array([[0.2, 0.95, 0.2]], dtype=np.float32), (model.num_joints, 1)),
        point_size=0.01,
    )

    # Ground-truth visualization
    gt_joint_frames = {}
    for name, idx in SMPLX_JOINT_INDEX_MAP.items():
        gt_joint_frames[name] = server.scene.add_frame(
            f"/gt_joints/{name}",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            axes_length=0.06,
            axes_radius=0.003,
        )
    gt_joints_handle = server.scene.add_point_cloud(
        "/gt_joints_all",
        points=np.zeros((model.num_joints, 3), dtype=np.float32),
        colors=np.tile(np.array([[0.95, 0.25, 0.15]], dtype=np.float32), (model.num_joints, 1)),
        point_size=0.01,
    )

    # GUI
    frame_slider = server.gui.add_slider("Frame", min=0, max=max(0, timeline.size - 1), step=1, initial_value=0)
    play_button = server.gui.add_button("Play/Pause")
    # Calibration mode indicator
    calib_label = "Calib: first {:.0f}s".format(args.calib_duration_sec) if args.calib_duration_sec else "Calib: all frames"
    mode_text = server.gui.add_text(calib_label, "")
    legend_text = server.gui.add_text(
        "Meshes",
        "IMU mesh: blue | Sync mesh: green | GT mesh: orange | TTO mesh: purple | EgoAllo mesh: magenta",
    )
    info_text = server.gui.add_text("Info", "")
    sync_toggle = server.gui.add_checkbox("Show Sync Mesh", initial_value=show_sync) if show_sync else None

    # Video panel (optional): show corresponding RGB frame in the GUI.
    video_handle = None

    @lru_cache(maxsize=32)
    def _load_video_frame(path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        if args.video_max_width and img.width > args.video_max_width:
            scale = float(args.video_max_width) / float(img.width)
            new_size = (int(round(img.width * scale)), int(round(img.height * scale)))
            img = img.resize(new_size, resample=Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)

    if show_video:
        init_path = timeline_color_paths[0] if timeline_color_paths else None
        if init_path is not None and Path(init_path).exists():
            init_img = _load_video_frame(str(init_path))
        else:
            init_img = np.zeros((240, 320, 3), dtype=np.uint8)
        video_handle = server.gui.add_image(init_img, label="Video")

    state = {"frame": 0, "playing": False, "last_frame_idx": -1}

    def _update(frame_idx: int) -> None:
        frame_idx = int(np.clip(frame_idx, 0, timeline.size - 1))
        if frame_idx == state["last_frame_idx"]:
            return
        state["last_frame_idx"] = frame_idx
        state["frame"] = frame_idx
        frame_id = int(timeline_frame_ids[frame_idx])
        t_ns = int(timeline[frame_idx])

        # Update video panel
        if video_handle is not None and timeline_color_paths:
            p = timeline_color_paths[frame_idx]
            if p is not None and Path(p).exists():
                try:
                    video_handle.image = _load_video_frame(str(p))
                except Exception:
                    pass

        # Global bone rotations from IMUs
        bone_global: Dict[str, np.ndarray] = {}
        for joint_name in JOINT_NAMES:
            imu_id = joint_to_imu.get(joint_name)
            if imu_id is None:
                continue
            q = sampled[imu_id][frame_idx]  # (wxyz)
            R = quaternion_to_matrix_wxyz(q)
            if not args.quat_is_world_from_sensor:
                R = R.T
            W_R_S = R
            if world_align:
                W_R_S = world_align.get(joint_name, np.eye(3, dtype=np.float64)) @ W_R_S

            B_R_S = bone_from_imu_offset.get(joint_name)
            if B_R_S is None:
                # No calibration: fall back to identity offset
                bone_global[joint_name] = W_R_S
            else:
                bone_global[joint_name] = W_R_S @ B_R_S.T

        # Align IMU reconstruction into GT coordinate system (using pelvis @ frame 0)
        gt_idx_for_frame: Optional[int] = None
        gt_idx0 = gt.index_of(int(timeline_frame_ids[0]))
        if gt_idx0 is not None:
            gt_pelvis_R0 = np.array(gt.joint_rotations_local[gt_idx0, 0], dtype=np.float64)
        else:
            gt_pelvis_R0 = None
        gt_idx_for_frame = gt.index_of(frame_id)

        if gt_pelvis_R0 is not None and "pelvis" in bone_global:
            if "imu_to_gt_align" not in state:
                state["imu_to_gt_align"] = gt_pelvis_R0 @ bone_global["pelvis"].T.copy()
            R_imu_to_gt = state["imu_to_gt_align"]
            for k in list(bone_global.keys()):
                bone_global[k] = R_imu_to_gt @ bone_global[k]

        local = build_local_rots_from_imu(bone_global)
        joints_w, T_w, verts_w = smplx_forward_kinematics(
            model, local, betas, compute_vertices=True,
            v_shaped=cached_v_shaped, j_tpose=cached_j_tpose,
        )

        # Center at pelvis for viewing
        pelvis_pos = joints_w[0].copy()
        joints_centered = joints_w - pelvis_pos[None, :]
        joints_centered = joints_centered + np.array([imu_display_x, 0.0, 0.0], dtype=np.float32)[None, :]

        # Pre-compute IMU mesh vertices outside atomic block
        imu_vv = None
        if imu_mesh_handle is not None and verts_w is not None:
            imu_vv = (verts_w - pelvis_pos[None, :]).astype(np.float32)
            imu_vv = imu_vv + np.array([imu_display_x, 0.0, 0.0], dtype=np.float32)[None, :]

        # --- Sync data pre-computation ---
        sync_visible = show_sync and sync_mesh_handle is not None and sync_toggle is not None and sync_toggle.value
        sync_results = None  # (sync_joints_centered, sync_T_w, vv_sync) or None
        if sync_visible:
            nearest_sync_t = get_nearest_sync_timestamp(sync_t_sorted, t_ns)
            if nearest_sync_t is not None:
                by_imu = sync_imu_dict.get(nearest_sync_t, {})
                sync_bone_global: Dict[str, np.ndarray] = {}
                for imu_id, rot in by_imu.items():
                    joint_name = OPTIMIZATION_IMU_ID_TO_ROSHI_JOINT.get(imu_id)
                    if joint_name:
                        sync_bone_global[joint_name] = np.array(rot, dtype=np.float64)

                # Apply GT alignment to sync data (same as IMU)
                if gt_pelvis_R0 is not None and "pelvis" in sync_bone_global:
                    if "sync_to_gt_align" not in state:
                        state["sync_to_gt_align"] = gt_pelvis_R0 @ sync_bone_global["pelvis"].T.copy()
                    R_sync_to_gt = state["sync_to_gt_align"]
                    for k in list(sync_bone_global.keys()):
                        sync_bone_global[k] = R_sync_to_gt @ sync_bone_global[k]

                sync_local = build_local_rots_from_imu(sync_bone_global)
                sync_joints_w, sync_T_w, sync_verts_w = smplx_forward_kinematics(
                    model, sync_local, betas, compute_vertices=True,
                    v_shaped=cached_v_shaped, j_tpose=cached_j_tpose,
                )

                sync_pelvis_pos = sync_joints_w[0].copy()
                sync_joints_centered = sync_joints_w - sync_pelvis_pos[None, :]
                sync_joints_centered = sync_joints_centered + np.array(
                    [sync_display_x, 0.0, 0.0], dtype=np.float32
                )[None, :]

                vv_sync = None
                if sync_verts_w is not None:
                    vv_sync = (sync_verts_w - sync_pelvis_pos[None, :]).astype(np.float32)
                    vv_sync = vv_sync + np.array([sync_display_x, 0.0, 0.0], dtype=np.float32)[None, :]

                sync_results = (sync_joints_centered, sync_T_w, vv_sync)

        # --- TTO mesh pre-computation ---
        tto_visible = show_tto and tto_mesh_handle is not None
        tto_results = None  # (tto_joints_centered, tto_T_w, tto_vv) or None
        if tto_visible:
            nearest_tto_t = get_nearest_sync_timestamp(tto_t_sorted, t_ns)
            if nearest_tto_t is not None:
                tto_local = tto_local_dict.get(nearest_tto_t)
                if tto_local is not None:
                    tto_joints_w, tto_T_w, tto_verts_w = smplx_forward_kinematics(
                        model, tto_local, betas, compute_vertices=True,
                        v_shaped=cached_v_shaped, j_tpose=cached_j_tpose,
                    )
                    # Align TTO into GT coordinate system using pelvis at first frame.
                    if gt_pelvis_R0 is not None:
                        tto_root_R = tto_T_w[0, :3, :3].astype(np.float64)
                        if "tto_to_gt_align" not in state:
                            state["tto_to_gt_align"] = gt_pelvis_R0 @ tto_root_R.T.copy()
                        R_tto_to_gt = state["tto_to_gt_align"]
                        tto_joints_w = (R_tto_to_gt @ tto_joints_w.T).T
                        tto_T_w[:, :3, :3] = R_tto_to_gt @ tto_T_w[:, :3, :3]
                        tto_T_w[:, :3, 3] = (R_tto_to_gt @ tto_T_w[:, :3, 3].T).T
                        if tto_verts_w is not None:
                            tto_verts_w = (R_tto_to_gt @ tto_verts_w.T).T

                    tto_pelvis_pos = tto_joints_w[0].copy()
                    tto_joints_centered = tto_joints_w - tto_pelvis_pos[None, :]
                    tto_joints_centered = tto_joints_centered + np.array(
                        [tto_display_x, 0.0, 0.0], dtype=np.float32
                    )[None, :]

                    tto_vv = None
                    if tto_verts_w is not None:
                        tto_vv = (tto_verts_w - tto_pelvis_pos[None, :]).astype(np.float32)
                        tto_vv = tto_vv + np.array([tto_display_x, 0.0, 0.0], dtype=np.float32)[None, :]

                    tto_results = (tto_joints_centered, tto_T_w, tto_vv)

        # --- EgoAllo original mesh pre-computation ---
        ego_visible = show_ego and ego_mesh_handle is not None
        ego_results = None  # (ego_joints_centered, ego_T_w, ego_vv) or None
        if ego_visible:
            nearest_ego_t = get_nearest_sync_timestamp(ego_t_sorted, t_ns)
            if nearest_ego_t is not None:
                ego_local = ego_local_dict.get(nearest_ego_t)
                if ego_local is not None:
                    ego_joints_w, ego_T_w, ego_verts_w = smplx_forward_kinematics(
                        model, ego_local, betas, compute_vertices=True,
                        v_shaped=cached_v_shaped, j_tpose=cached_j_tpose,
                    )
                    if gt_pelvis_R0 is not None:
                        ego_root_R = ego_T_w[0, :3, :3].astype(np.float64)
                        if "ego_to_gt_align" not in state:
                            state["ego_to_gt_align"] = gt_pelvis_R0 @ ego_root_R.T.copy()
                        R_ego_to_gt = state["ego_to_gt_align"]
                        ego_joints_w = (R_ego_to_gt @ ego_joints_w.T).T
                        ego_T_w[:, :3, :3] = R_ego_to_gt @ ego_T_w[:, :3, :3]
                        ego_T_w[:, :3, 3] = (R_ego_to_gt @ ego_T_w[:, :3, 3].T).T
                        if ego_verts_w is not None:
                            ego_verts_w = (R_ego_to_gt @ ego_verts_w.T).T

                    ego_pelvis_pos = ego_joints_w[0].copy()
                    ego_joints_centered = ego_joints_w - ego_pelvis_pos[None, :]
                    ego_joints_centered = ego_joints_centered + np.array(
                        [ego_display_x, 0.0, 0.0], dtype=np.float32
                    )[None, :]

                    ego_vv = None
                    if ego_verts_w is not None:
                        ego_vv = (ego_verts_w - ego_pelvis_pos[None, :]).astype(np.float32)
                        ego_vv = ego_vv + np.array([ego_display_x, 0.0, 0.0], dtype=np.float32)[None, :]

                    ego_results = (ego_joints_centered, ego_T_w, ego_vv)

        # --- Ground truth pre-computation ---
        gt_err_str = ""
        gt_results = None  # (v_gt, j_gt, gt_global) or None
        if gt_idx_for_frame is None:
            gt_idx_for_frame = gt.index_of(frame_id)
        if gt_idx_for_frame is not None:
            # Use preloaded vertices if available, else fall back to memmap
            preload_i = gt_preload_map.get(frame_idx) if gt_verts_preloaded is not None else None
            if preload_i is not None:
                v_gt = gt_verts_preloaded[preload_i]
            else:
                v_gt = np.array(gt.vertices[gt_idx_for_frame], dtype=np.float32)
            j_gt_raw = np.array(gt.joints[gt_idx_for_frame, : model.num_joints], dtype=np.float32)
            pelvis_gt = np.array(gt.joints[gt_idx_for_frame, 0], dtype=np.float32)

            # Base GT (centered at pelvis, no side-by-side offset) for metrics.
            j_gt_base = j_gt_raw - pelvis_gt[None, :]

            # Visualization offsets.
            v_gt = v_gt - pelvis_gt[None, :]
            v_gt = v_gt + np.array([gt_display_x, 0.0, 0.0], dtype=np.float32)[None, :]

            j_gt = j_gt_base + np.array([gt_display_x, 0.0, 0.0], dtype=np.float32)[None, :]

            # GT key joint frames (orientations from FK of local rotations)
            gt_local = np.array(gt.joint_rotations_local[gt_idx_for_frame], dtype=np.float64)
            gt_global = np.zeros_like(gt_local)
            gt_global[0] = gt_local[0]
            for ji in range(1, gt_global.shape[0]):
                p = int(model.parents[ji])
                gt_global[ji] = gt_global[p] @ gt_local[ji] if p >= 0 else gt_local[ji]

            gt_results = (v_gt, j_gt, gt_global)

            # Per-frame left-side rotation errors
            if "left-hip" in bone_global:
                def _err(jn: str, jidx: int) -> float:
                    if jn not in bone_global:
                        return float("nan")
                    return rot_angle_deg(gt_global[jidx].T @ bone_global[jn])

                e_lhip = _err("left-hip", 1)
                e_lknee = _err("left-knee", 4)
                e_lsho = _err("left-shoulder", 16)
                e_lelb = _err("left-elbow", 18)

                # Local (parent-relative) errors
                e_lhip_loc = float("nan")
                e_lknee_loc = float("nan")
                e_lelb_loc = float("nan")
                if "pelvis" in bone_global:
                    imu_lhip_loc = bone_global["pelvis"].T @ bone_global.get("left-hip", np.eye(3))
                    e_lhip_loc = rot_angle_deg(gt_local[1].T @ imu_lhip_loc)
                if "left-hip" in bone_global:
                    imu_lknee_loc = bone_global["left-hip"].T @ bone_global.get("left-knee", np.eye(3))
                    e_lknee_loc = rot_angle_deg(gt_local[4].T @ imu_lknee_loc)
                if "left-shoulder" in bone_global:
                    imu_lelb_loc = bone_global["left-shoulder"].T @ bone_global.get("left-elbow", np.eye(3))
                    e_lelb_loc = rot_angle_deg(gt_local[18].T @ imu_lelb_loc)

                # Position errors (cm)
                j_imu_base = (joints_w - pelvis_pos[None, :]).astype(np.float64)
                j_gt_base64 = j_gt_base.astype(np.float64)
                p_lhip = 100.0 * float(np.linalg.norm(j_imu_base[1] - j_gt_base64[1]))
                p_lknee = 100.0 * float(np.linalg.norm(j_imu_base[4] - j_gt_base64[4]))

                gt_err_str = (
                    f" | L global° hip={e_lhip:.0f} knee={e_lknee:.0f} sh={e_lsho:.0f} el={e_lelb:.0f}"
                    f" | L local° hip={e_lhip_loc:.0f} knee={e_lknee_loc:.0f} el={e_lelb_loc:.0f}"
                    f" | L pos cm hip={p_lhip:.1f} knee={p_lknee:.1f}"
                )

        # Calibration/mode text
        elapsed_sec = (t_ns - timeline_start_ns) / 1e9
        if calib_end_ns is not None:
            if t_ns <= calib_end_ns:
                mode_val = f"⏱ {elapsed_sec:.1f}s  ✅ CALIBRATION"
            else:
                mode_val = f"⏱ {elapsed_sec:.1f}s  🔵 IMU-ONLY"
        else:
            mode_val = f"⏱ {elapsed_sec:.1f}s"
        info_val = f"idx={frame_idx}/{timeline.size-1} frame_id={frame_id}{gt_err_str}"

        # --- Batch all viser scene updates ---
        with server.atomic():
            joints_handle.points = joints_centered.astype(np.float32)

            for name, idx in SMPLX_JOINT_INDEX_MAP.items():
                Rg = T_w[idx, :3, :3].astype(np.float64)
                pos = joints_centered[idx].astype(np.float64)
                joint_frames[name].wxyz = tf.SO3.from_matrix(Rg).wxyz
                joint_frames[name].position = pos

            if imu_vv is not None:
                imu_mesh_handle.vertices = imu_vv

            if sync_results is not None:
                s_jc, s_Tw, s_vv = sync_results
                if sync_joints_handle is not None:
                    sync_joints_handle.points = s_jc.astype(np.float32)
                for name, idx in SMPLX_JOINT_INDEX_MAP.items():
                    if name in sync_joint_frames:
                        Rg = s_Tw[idx, :3, :3].astype(np.float64)
                        pos = s_jc[idx].astype(np.float64)
                        sync_joint_frames[name].wxyz = tf.SO3.from_matrix(Rg).wxyz
                        sync_joint_frames[name].position = pos
                if s_vv is not None:
                    sync_mesh_handle.vertices = s_vv

            if tto_results is not None:
                t_jc, t_Tw, t_vv = tto_results
                if tto_joints_handle is not None:
                    tto_joints_handle.points = t_jc.astype(np.float32)
                for name, idx in SMPLX_JOINT_INDEX_MAP.items():
                    if name in tto_joint_frames:
                        Rg = t_Tw[idx, :3, :3].astype(np.float64)
                        pos = t_jc[idx].astype(np.float64)
                        tto_joint_frames[name].wxyz = tf.SO3.from_matrix(Rg).wxyz
                        tto_joint_frames[name].position = pos
                if tto_mesh_handle is not None and t_vv is not None:
                    tto_mesh_handle.vertices = t_vv

            if ego_results is not None:
                e_jc, e_Tw, e_vv = ego_results
                if ego_joints_handle is not None:
                    ego_joints_handle.points = e_jc.astype(np.float32)
                for name, idx in SMPLX_JOINT_INDEX_MAP.items():
                    if name in ego_joint_frames:
                        Rg = e_Tw[idx, :3, :3].astype(np.float64)
                        pos = e_jc[idx].astype(np.float64)
                        ego_joint_frames[name].wxyz = tf.SO3.from_matrix(Rg).wxyz
                        ego_joint_frames[name].position = pos
                if ego_mesh_handle is not None and e_vv is not None:
                    ego_mesh_handle.vertices = e_vv

            if gt_results is not None:
                v_gt, j_gt, gt_global = gt_results
                if gt_mesh_handle is not None:
                    gt_mesh_handle.vertices = v_gt
                if gt_joints_handle is not None:
                    gt_joints_handle.points = j_gt
                for name, idx in SMPLX_JOINT_INDEX_MAP.items():
                    gt_joint_frames[name].wxyz = tf.SO3.from_matrix(gt_global[idx]).wxyz
                    gt_joint_frames[name].position = j_gt[idx].astype(np.float64)

            mode_text.value = mode_val
            info_text.value = info_val

    _update(0)

    @frame_slider.on_update
    def _on_slider(_):
        _update(int(frame_slider.value))

    @play_button.on_click
    def _on_play(_):
        state["playing"] = not state["playing"]

    # Main loop (fixed 30 Hz playback)
    last_time = time.time()
    try:
        while True:
            if state["playing"]:
                now = time.time()
                if now - last_time >= 1.0 / 30.0:
                    nxt = (state["frame"] + 1) % timeline.size
                    frame_slider.value = nxt
                    _update(nxt)
                    last_time = now
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nExiting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

