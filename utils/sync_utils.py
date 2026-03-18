"""
Utilities for UTC-based synchronization of VRS_RGB frames and IMU data.
UTC is the common reference for VRS_RGB, Third-person RGB, and IMU.

- VRS_RGB: VRS capture_timestamp_ns (tracking) -> trajectory CSV -> utc_timestamp_ns.
- IMU: imu_data.csv (utc_timestamp_ns per sample).
- [Optional] Third-person RGB: frames.csv (utc_timestamp_ns per frame).
"""

import bisect
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# -----------------------------------------------------------------------------
# VRS RGB strem:
# Tracking (VRS) -> UTC mapping (from MPS trajectory: <sample_name>_vrs/slam/closed_loop_trajectory.csv)
# -----------------------------------------------------------------------------

def load_tracking_to_utc_dict(csv_path: Path) -> Dict[int, int]:
    """
    Load trajectory CSV: tracking_timestamp_us -> utc_timestamp_ns. (from MPS trajectory: <sample_name>_vrs/slam/closed_loop_trajectory.csv)

    Expected columns: tracking_timestamp_us, utc_timestamp_ns
    """
    mapping: Dict[int, int] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                tracking_timestamp_us = int(row["tracking_timestamp_us"])
                utc_ns = int(row["utc_timestamp_ns"])
                mapping[tracking_timestamp_us] = utc_ns
            except (KeyError, ValueError):
                continue
    return mapping

def get_sorted_keys_for_dict(mapped_dict: Dict) -> List:
    """Return sorted keys of a dict (e.g. UTC timestamps)."""
    return sorted(mapped_dict.keys())
    
def get_nearest_utc_with_error(
    tracking_to_utc: Dict[int, int],
    sorted_tracking_timestamp_us: List[int],
    tracking_timestamp_ns: int,
) -> Tuple[Optional[int], float]:
    """
    Map a VRS tracking timestamp (ns) to the nearest UTC (ns) using the <sample_name>_vrs/slam/closed_loop_trajectory.csv.

    Returns:
        (utc_timestamp_ns, error_us) or (None, error_us) if trajectory is empty.
    """
    if not sorted_tracking_timestamp_us:
        return None, float("inf")
    tracking_timestamp_us = int(tracking_timestamp_ns // 1000)
    idx = bisect.bisect_left(sorted_tracking_timestamp_us, tracking_timestamp_us)
    if idx == 0:
        nearest_timestamp_us = sorted_tracking_timestamp_us[0]
    elif idx >= len(sorted_tracking_timestamp_us):
        nearest_timestamp_us = sorted_tracking_timestamp_us[-1]
    else:
        left = sorted_tracking_timestamp_us[idx - 1]
        right = sorted_tracking_timestamp_us[idx]
        nearest_timestamp_us = left if abs(left - tracking_timestamp_us) <= abs(right - tracking_timestamp_us) else right
    utc_ns = tracking_to_utc[nearest_timestamp_us]
    error_us = abs(nearest_timestamp_us - tracking_timestamp_us)
    return utc_ns, error_us

def load_frames_csv(frames_csv_path: Path) -> Tuple[List[int], List[int], List[str]]:
    """
    Load frames.csv (frame_id, utc_timestamp_ns, color_path).

    Returns:
        frame_ids: list of frame indices
        utc_timestamps_ns: list of UTC timestamps (nanoseconds)
        color_paths: list of paths relative to the recording root (or absolute)
    """
    frame_ids: List[int] = []
    utc_ns: List[int] = []
    paths: List[str] = []
    with open(frames_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_ids.append(int(row["frame_id"]))
                utc_ns.append(int(row["utc_timestamp_ns"]))
                paths.append(row["color_path"].strip())
            except (KeyError, ValueError):
                continue
    return frame_ids, utc_ns, paths

def load_imu_streams_from_csv(imu_csv_path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Sorting logic: 
    IMU information is sorted by utc_timestamp_ns regarless of the imu_id. 
    """
    streams: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    with open(imu_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                imu_id = int(row["imu_id"])
                t_ns = int(row["utc_timestamp_ns"])
                qx, qy, qz, qw = float(row["quatI"]), float(row["quatJ"]), float(row["quatK"]), float(row["quatW"])
            except (KeyError, ValueError):
                continue
            streams.setdefault(imu_id, []).append((t_ns, np.array([qw, qx, qy, qz], dtype=np.float64)))
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for imu_id, items in streams.items():
        items.sort(key=lambda x: x[0])
        t = np.array([x[0] for x in items], dtype=np.int64)
        q = np.stack([x[1] for x in items], axis=0).astype(np.float64)
        out[imu_id] = {"t_ns": t, "quat_wxyz": q}
    return out

def load_imu_csv(path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load imu_data.csv written by IMUDataRecorder.

    Returns:
        {imu_id: {"t_ns": (N,), "quat_wxyz": (N,4)}}
    """
    streams: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                imu_id = int(row["imu_id"])
                t_ns = int(row["utc_timestamp_ns"])
                qx = float(row["quatI"])
                qy = float(row["quatJ"])
                qz = float(row["quatK"])
                qw = float(row["quatW"])
            except Exception:
                continue
            streams.setdefault(imu_id, []).append((t_ns, np.array([qw, qx, qy, qz], dtype=np.float64)))

    out: Dict[int, Dict[str, np.ndarray]] = {}
    for imu_id, items in streams.items():
        items.sort(key=lambda x: x[0])
        t = np.array([x[0] for x in items], dtype=np.int64)
        q = np.stack([x[1] for x in items], axis=0).astype(np.float64)
        out[imu_id] = {"t_ns": t, "quat_wxyz": q}
    return out


def load_calibration(calib_json: Path) -> Tuple[Dict[str, np.ndarray], Optional[float]]:
    """Load imu_calibration.json -> (B_R_S per joint, optional calib_duration_sec)."""
    import json
    data = json.loads(calib_json.read_text())
    out: Dict[str, np.ndarray] = {}
    for joint_name, joint_data in data.get("joints", {}).items():
        out[joint_name] = np.array(joint_data["B_R_S"], dtype=np.float64)
    calib_duration_sec = data.get("calib_duration_sec")
    return out, calib_duration_sec


def compute_time_intersection_ns(streams: Dict[int, Dict[str, np.ndarray]], imu_ids: List[int]) -> Tuple[int, int]:
    """Return (t_min, t_max) intersection across all listed IMU streams."""
    mins = []
    maxs = []
    for imu_id in imu_ids:
        s = streams.get(imu_id)
        if not s or s["t_ns"].size == 0:
            continue
        mins.append(int(s["t_ns"][0]))
        maxs.append(int(s["t_ns"][-1]))
    if not mins or not maxs:
        raise ValueError("No IMU streams available to build a timeline.")
    return max(mins), min(maxs)


def sample_streams_hold(
    streams: Dict[int, Dict[str, np.ndarray]],
    imu_ids: List[int],
    timeline_ns: np.ndarray,
) -> Dict[int, np.ndarray]:
    """
    Zero-order-hold sampling: for each imu_id and each time t, use latest sample <= t.

    Returns:
      {imu_id: quat_wxyz_timeline (T,4)}
    """
    sampled: Dict[int, np.ndarray] = {}
    for imu_id in imu_ids:
        s = streams.get(imu_id)
        if not s or s["t_ns"].size == 0:
            sampled[imu_id] = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (timeline_ns.size, 1))
            continue
        t = s["t_ns"]
        q = s["quat_wxyz"]
        out_q = np.zeros((timeline_ns.size, 4), dtype=np.float64)
        idx = 0
        for i, tt in enumerate(timeline_ns):
            while idx + 1 < t.size and t[idx + 1] <= tt:
                idx += 1
            out_q[i] = q[idx]
        sampled[imu_id] = out_q
    return sampled


def find_closest_imu_timestamp(
    reference_utc_ns: int,
    imu_dict: Dict[int, Dict],
    max_error_ns: Optional[int] = 20_000_000,
) -> Tuple[Optional[int], Optional[Dict], float]:
    """
    Find the IMU timestamp closest to reference_utc_ns in imu_dict.

    imu_dict is keyed by utc_timestamp_ns; values are dicts of imu_id -> rot_matrix.

    Returns:
        (closest_utc_ns, imu_data_dict, error_ns)
        or (None, None, error_ns) if no key or error > max_error_ns.
    """
    if not imu_dict:
        return None, None, float("inf")
    closest_ts = min(imu_dict.keys(), key=lambda k: abs(k - reference_utc_ns))
    error = abs(closest_ts - reference_utc_ns)
    if max_error_ns is not None and error > max_error_ns:
        return None, None, error
    return closest_ts, imu_dict[closest_ts], error
