#!/usr/bin/env python3
"""
Synchronization pipeline: UTC-mapped RGB + calibrated IMU -> sync/

All sync code lives in RoSHI_Calibration; 
Sync is driven by:
  - imu/imu_data.csv            (raw IMU)
  - mps_<sample_name>_vrs/slam/closed_loop_trajectory.csv 
  - [optional] frames.csv + color/ (optinal choice for third-person RGB)
  - [optional] <sample_name>.vrs (ego RGB).

UTC is the common reference for all the data:
  - VRS-RGB: capture_timestamp_ns (tracking) --{trajectory CSV}--> utc_timestamp_ns.
  - IMU: imu_data.csv has utc_timestamp_ns; calibrated with imu_calibration.json.
  - [optional] Third-person RGB: frames.csv + color/ (optional choice for third-person RGB)

Output under recording_dir/sync/:
  - frames.csv, color/
  - imu_info.csv (utc_ns, imu_id, rot_matrix; 9 rows per timestamp; imu_id in optimization order via imu_id_mapping.py)
  - imu_info.pkl (dict: utc_ns -> {imu_id: rot_matrix}, imu_id in optimization order)
  - vrs_frames.csv, vrs_color/ (when VRS + trajectory exist)
"""

"""
TODO:
Make RGB stream as optional and mark as a arg input in sync_pipeline.py
"""

"""
TODO:
the load_imu_streams_from_csv function in sync_utils.py:
- maybe it is wrong to load qx, qy, qz, qw from the imu_data.csv file? 
  Maybe other way to read raw IMU data is better
  Refer to the visualizer that Luyang wrote. 
- It is sorted by utc_timestamp_ns regarless of the imu_id. 
  Maybe sorting with the order with IMU_id would be better?
  Establish the logic of sorting with IMU_id. 
"""
"""
TODO:
Create a rotation utils.
"""

import argparse
import csv
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np

from utils.apriltag_utils import (
    load_apriltag_rotations_by_time,
    estimate_world_alignment_from_tags,
)
from utils.imu_id_mapping import JOINT_NAMES, ROSHI_TO_OPTIMIZATION_IMU_ID, ROSHI_IMU_ID_TO_JOINT
from utils.sync_utils import (
    load_frames_csv,
    load_imu_streams_from_csv,
    load_tracking_to_utc_dict,
    get_nearest_utc_with_error,
    get_sorted_keys_for_dict,
)

RGB_DEVICE_TRACKING_ERROR_THRESHOLD_US = 200  


T_R_IMU = np.array(
    [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float64,
)


def quaternion_to_matrix_wxyz(q: np.ndarray) -> np.ndarray:
    """
    Quaternion (w,x,y,z) -> rotation matrix.
    Uses a standard right-handed active rotation convention.
    """
    q = np.asarray(q, dtype=np.float64).reshape(4)
    w, x, y, z = q
    n = np.linalg.norm(q)
    if n <= 0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = (q / n).tolist()
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_calibration(calib_json: Path) -> Dict[str, np.ndarray]:
    """
    Load imu_calibration.json (B_R_S for each joint).
    """
    data = json.loads(calib_json.read_text())
    return {name: np.array(j["B_R_S"], dtype=np.float64) for name, j in data.get("joints", {}).items()}


def build_calibrated_imu_dict(
    streams: Dict[int, Dict[str, np.ndarray]],
    calib: Dict[str, np.ndarray],
    world_align: Optional[Dict[str, np.ndarray]] = None,
    quat_is_world_from_sensor: bool = True,
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Build calibrated IMU rotation dict with optional world alignment.

    Args:
        streams: Raw IMU data {imu_id: {"t_ns": ..., "quat_wxyz": ...}}
        calib: Calibration offsets {joint_name: B_R_tag}
        world_align: World alignment {joint_name: Wp_R_Wi} to align each IMU's
                     world frame to the pelvis world frame
        quat_is_world_from_sensor: Whether quaternion represents world<-sensor

    Returns:
        {utc_ns: {imu_id: W_R_B}} where W_R_B is the calibrated bone rotation
    """
    bone_from_imu = {j: np.array(B_R_tag, dtype=np.float64) @ T_R_IMU for j, B_R_tag in calib.items()}
    imu_dict: Dict[int, Dict[int, np.ndarray]] = {}
    for imu_id, data in streams.items():
        joint = ROSHI_IMU_ID_TO_JOINT.get(imu_id)
        B_R_imu = bone_from_imu.get(joint) if joint else None
        if B_R_imu is None:
            # Match imu_pose_viewer fallback: use raw IMU orientation if calibration missing.
            B_R_imu = np.eye(3, dtype=np.float64)

        # Get world alignment for this joint (if available)
        W_align = None
        if world_align and joint:
            W_align = world_align.get(joint)

        for i in range(len(data["t_ns"])):
            utc_ns = int(data["t_ns"][i])
            W_R_S = quaternion_to_matrix_wxyz(data["quat_wxyz"][i])
            if not quat_is_world_from_sensor:
                W_R_S = W_R_S.T

            # Apply world alignment (align this IMU's world to pelvis world)
            if W_align is not None:
                W_R_S = W_align @ W_R_S

            W_R_B = W_R_S @ B_R_imu.T
            imu_dict.setdefault(utc_ns, {})[imu_id] = W_R_B
    return imu_dict


EXPECTED_IMU_IDS = sorted(ROSHI_IMU_ID_TO_JOINT.keys())
MAX_GROUP_GAP_NS = int(50e6)  # 50 ms: readings within this of anchor count as same group


def _per_imu_streams_from_dict(
    imu_dict: Dict[int, Dict[int, np.ndarray]],
    expected_imus: List[int],
) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    """Convert imu_dict[utc_ns][imu_id]=R into per_imu[imu_id] = [(t_ns, R), ...] sorted by t_ns."""
    per_imu: Dict[int, List[Tuple[int, np.ndarray]]] = {i: [] for i in expected_imus}
    for utc_ns, by_imu in imu_dict.items():
        for imu_id, R in by_imu.items():
            if imu_id in per_imu:
                per_imu[imu_id].append((utc_ns, R))
    for imu_id in per_imu:
        per_imu[imu_id].sort(key=lambda x: x[0])
    return per_imu


def _find_nearest_within(
    stream: List[Tuple[int, np.ndarray]],
    t_ref_ns: int,
    max_gap_ns: int,
) -> Optional[Tuple[int, np.ndarray]]:
    """Return (t_ns, R) in stream with min |t_ns - t_ref_ns| <= max_gap_ns, or None."""
    best: Optional[Tuple[int, np.ndarray]] = None
    best_diff = max_gap_ns + 1
    for (t_ns, R) in stream:
        diff = abs(t_ns - t_ref_ns)
        if diff <= max_gap_ns and diff < best_diff:
            best_diff = diff
            best = (t_ns, R)
    return best


def group_imu_readings_to_complete_sets(
    imu_dict: Dict[int, Dict[int, np.ndarray]],
    expected_imus: List[int],
    max_gap_ns: int = MAX_GROUP_GAP_NS,
) -> List[Tuple[int, Dict[int, np.ndarray]]]:
    """
    Group IMU readings so each group has exactly one reading per IMU 1-9 with timestamps
    close to each other. Label each group with the first (anchor) timestamp (IMU 1).
    Returns [(label_t_ns, {imu_id: R}), ...].
    """
    if 1 not in expected_imus:
        return []
    per_imu = _per_imu_streams_from_dict(imu_dict, expected_imus)
    if not per_imu[1]:
        return []
    groups: List[Tuple[int, Dict[int, np.ndarray]]] = []
    for (t1, R1) in per_imu[1]:
        group: Dict[int, np.ndarray] = {1: R1}
        label_t = t1
        ok = True
        for imu_id in expected_imus:
            if imu_id == 1:
                continue
            hit = _find_nearest_within(per_imu[imu_id], t1, max_gap_ns)
            if hit is None:
                ok = False
                break
            group[imu_id] = hit[1]
        if ok:
            groups.append((label_t, group))
    return groups


def _find_trajectory_csv(recording_dir: Path) -> Optional[Path]:
    """Find MPS trajectory: mps_new_cali_three_vrs, then mps_dsdf, then any mps_*/slam/closed_loop_trajectory.csv."""
    rec = Path(recording_dir).resolve()
    candidates = [
        rec / "mps_new_cali_three_vrs" / "slam" / "closed_loop_trajectory.csv",
        rec / "mps_dsdf" / "slam" / "closed_loop_trajectory.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    for d in rec.iterdir():
        if d.is_dir() and "mps" in d.name.lower():
            p = d / "slam" / "closed_loop_trajectory.csv"
            if p.exists():
                return p
    return None


def _find_vrs_path(recording_dir: Path) -> Optional[Path]:
    vrs_list = list(Path(recording_dir).resolve().glob("*.vrs"))
    return vrs_list[0] if vrs_list else None


def _extract_vrs_rgb_to_sync(
    vrs_path: Path,
    trajectory_csv: Path,
    sync_dir: Path,
    error_threshold_us: float = RGB_DEVICE_TRACKING_ERROR_THRESHOLD_US,
) -> int:
    try:
        from projectaria_tools.core import data_provider
        from PIL import Image
    except ImportError:
        print("  VRS: projectaria_tools or Pillow not installed; skip. pip install projectaria-tools Pillow")
        return 0

    track_to_utc = load_tracking_to_utc_dict(trajectory_csv)
    sorted_tracking_us = get_sorted_keys_for_dict(track_to_utc)
    if not sorted_tracking_us:
        print("  VRS: trajectory has no rows; skip.")
        return 0

    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    stream_id = provider.get_stream_id_from_label("camera-rgb")
    total = provider.get_num_data(stream_id)
    if total == 0:
        print("  VRS: no RGB stream; skip.")
        return 0

    vrs_color_dir = sync_dir / "vrs_color"
    vrs_color_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Tuple[int, int, int, str]] = []

    for i in range(total):
        img_data = provider.get_image_data_by_index(stream_id, i)
        image_np = np.asarray(img_data[0].to_numpy_array())
        tracking_ns = int(img_data[1].capture_timestamp_ns)
        utc_ns, err_us = get_nearest_utc_with_error(track_to_utc, sorted_tracking_us, tracking_ns)
        if utc_ns is None or err_us > error_threshold_us:
            continue
        image_np = np.fliplr(np.rot90(image_np, 1))
        fname = f"vrs_frame_{i:06d}_{utc_ns}.jpg"
        out_path = vrs_color_dir / fname
        try:
            Image.fromarray(image_np).save(out_path, quality=95)
        except Exception:
            continue
        rows.append((i, utc_ns, tracking_ns, f"vrs_color/{fname}"))

    if not rows:
        print("  VRS: no frames within threshold; skip.")
        return 0

    with open(sync_dir / "vrs_frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        # utc_timestamp_ns: mapped UTC timestamp for RGB frame (from VRS tracking timestamp via trajectory CSV)
        # tracking_timestamp_ns: capture timestamp for RGB frame in VRS
        # color_path: path to the RGB frame in vrs_color/
        w.writerow(["frame_id", "utc_timestamp_ns", "tracking_timestamp_ns", "color_path"])
        w.writerows(rows)
    print(f"  vrs_frames: {len(rows)} rows, vrs_color: {len(rows)} images")
    return len(rows)


def run_sync(
    recording_dir: Path,
    sync_dir: Optional[Path] = None,
    copy_images: bool = False,
    vrs_path: Optional[Path] = None,
    trajectory_csv: Optional[Path] = None,
) -> Path:
    recording_dir = Path(recording_dir).resolve()
    sync_dir = Path(sync_dir or recording_dir / "sync").resolve()

    frames_csv = recording_dir / "frames.csv" # optional choice for third-person RGB
    imu_csv = recording_dir / "imu" / "imu_data.csv"
    calib_json = recording_dir / "imu_calibration.json"

    if not frames_csv.exists():
        raise FileNotFoundError(f"Missing {frames_csv}")
    if not imu_csv.exists():
        raise FileNotFoundError(f"Missing {imu_csv}")
    if not calib_json.exists():
        raise FileNotFoundError(f"Missing {calib_json}")

    sync_dir.mkdir(parents=True, exist_ok=True)
    color_out = sync_dir / "color"
    color_out.mkdir(parents=True, exist_ok=True)

    frame_ids, utc_ns_list, color_paths = load_frames_csv(frames_csv)
    streams = load_imu_streams_from_csv(imu_csv)
    calib = load_calibration(calib_json)

    # Compute world alignment using AprilTag detections
    # This aligns each IMU's internal world frame to the pelvis IMU world frame
    joint_to_imu: Dict[str, int] = {joint: imu_id for imu_id, joint in ROSHI_IMU_ID_TO_JOINT.items()}
    sensor_from_tag = T_R_IMU.T  # S_R_T

    tag_rots_by_time = load_apriltag_rotations_by_time(recording_dir)
    world_align: Optional[Dict[str, np.ndarray]] = None

    if tag_rots_by_time:
        world_align, world_stats = estimate_world_alignment_from_tags(
            streams=streams,
            tag_rots_by_time=tag_rots_by_time,
            joint_to_imu=joint_to_imu,
            imu_time_offset_ns=0,
            sensor_from_tag=sensor_from_tag,
            quat_is_world_from_sensor=True,
            min_samples=60,
        )
        if world_align:
            print("World alignment computed (aligning all IMU worlds to pelvis world):")
            for jn in JOINT_NAMES:
                if jn in world_stats:
                    n, std = world_stats[jn]
                    print(f"  {jn:14s}: samples={n:4d}  std={std:5.2f}°")
        else:
            print("Warning: Could not compute world alignment from AprilTags")
    else:
        print("Warning: No AprilTag detections found; world alignment skipped")
        print("  (each IMU will use its own world frame, results may be inconsistent)")

    imu_dict_raw = build_calibrated_imu_dict(streams, calib, world_align)

    imu_groups = group_imu_readings_to_complete_sets(
        imu_dict_raw, EXPECTED_IMU_IDS, MAX_GROUP_GAP_NS
    )

    # Align to canonical frame: make pelvis identity at frame 0
    # This ensures the person starts in a standard upright orientation
    if imu_groups:
        first_group = imu_groups[0][1]  # {roshi_imu_id: R}
        pelvis_R0 = first_group.get(1)  # RoSHI imu_id 1 = pelvis
        if pelvis_R0 is not None:
            # R_canonical = pelvis_R0.T @ R, so pelvis at t=0 becomes identity
            canonical_align = pelvis_R0.T
            print(f"Applying canonical alignment (pelvis -> identity at frame 0)")
            for i, (label_t, group) in enumerate(imu_groups):
                aligned_group = {}
                for imu_id, R in group.items():
                    aligned_group[imu_id] = canonical_align @ R
                imu_groups[i] = (label_t, aligned_group)

    with open(sync_dir / "frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "utc_timestamp_ns", "color_path"])
        for fid, utc_ns, rel in zip(frame_ids, utc_ns_list, color_paths):
            base = Path(rel).name
            w.writerow([fid, utc_ns, f"color/{base}"])

    for rel in color_paths:
        src = recording_dir / rel
        dst = color_out / Path(rel).name
        if not src.exists() or dst.exists():
            continue
        if copy_images:
            shutil.copy2(src, dst)
        else:
            try:
                dst.symlink_to(src)
            except OSError:
                shutil.copy2(src, dst)

    imu_info_pkl: Dict[int, Dict[int, np.ndarray]] = {}
    with open(sync_dir / "imu_info.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utc_timestamp_ns", "imu_id", "rot_matrix"])
        for label_t_ns, group in imu_groups:
            imu_info_pkl[label_t_ns] = {}
            for roshi_id in sorted(group.keys()):
                maid_id = ROSHI_TO_OPTIMIZATION_IMU_ID[roshi_id]
                rot = group[roshi_id]
                w.writerow([label_t_ns, maid_id, str(rot.tolist())])
                imu_info_pkl[label_t_ns][maid_id] = rot

    with open(sync_dir / "imu_info.pkl", "wb") as f:
        pickle.dump(imu_info_pkl, f)

    print(f"Sync written to {sync_dir}")
    print(f"  frames: {len(frame_ids)} rows, color: {len(list(color_out.iterdir()))} files")
    print(f"  imu_info: {len(imu_groups)} groups (9 rows each), {len(imu_groups) * 9} rows total")

    vrs_path = Path(vrs_path).resolve() if vrs_path else _find_vrs_path(recording_dir)
    trajectory_csv = Path(trajectory_csv).resolve() if trajectory_csv else _find_trajectory_csv(recording_dir)
    if vrs_path and trajectory_csv and vrs_path.exists() and trajectory_csv.exists():
        print("Extracting VRS RGB (tracking -> UTC via trajectory)...")
        _extract_vrs_rgb_to_sync(vrs_path, trajectory_csv, sync_dir)
    elif vrs_path and not trajectory_csv:
        print("  VRS found but no trajectory CSV (mps_*/slam/closed_loop_trajectory.csv); skip VRS.")
    elif trajectory_csv and not vrs_path:
        print("  Trajectory found but no .vrs; skip VRS.")

    return sync_dir


def main():
    parser = argparse.ArgumentParser(description="Build sync folder: UTC-mapped RGB + calibrated IMU")
    parser.add_argument("recording_dir", type=Path, nargs="?", default=Path("received_recordings/recording_20260127_170016"), help="Recording directory")
    parser.add_argument("--sync-dir", type=Path, default=None, help="Output sync dir (default: <recording_dir>/sync)")
    parser.add_argument("--copy-images", action="store_true", help="Copy third-person images instead of symlink")
    parser.add_argument("--vrs-path", type=Path, default=None, help="Path to .vrs (default: first *.vrs in recording_dir)")
    parser.add_argument("--trajectory-csv", type=Path, default=None, help="MPS trajectory CSV (default: mps_new_cali_three_vrs/slam/closed_loop_trajectory.csv or mps_dsdf/...)")
    args = parser.parse_args()
    run_sync(args.recording_dir, args.sync_dir, args.copy_images, args.vrs_path, args.trajectory_csv)


if __name__ == "__main__":
    main()
