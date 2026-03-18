"""
AprilTag rotation loading and world-alignment estimation.

Shared by 03_imu_pose_viewer.py, 04_sync_pipeline.py, visualize_utc_mapped.py.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.imu_id_mapping import JOINT_NAMES, JOINT_TO_TAG_ID


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def rot_angle_deg(R: np.ndarray) -> float:
    """Geodesic angle of a rotation matrix (degrees)."""
    tr = float(np.trace(R))
    c = (tr - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _matrix_to_quaternion_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (w,x,y,z)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = float(np.trace(R))
    if t > 0.0:
        S = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    if n <= 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


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


def average_rotations_quaternion(rotations: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """
    Average rotation matrices using quaternion mean + hemisphere alignment.
    Returns (avg_R, std_dev_deg).
    """
    if not rotations:
        return np.eye(3, dtype=np.float64), 0.0
    if len(rotations) == 1:
        return np.array(rotations[0], dtype=np.float64), 0.0

    qs = []
    ref = None
    for R in rotations:
        q = _matrix_to_quaternion_wxyz(R)
        if ref is None:
            ref = q
        else:
            if float(np.dot(q, ref)) < 0.0:
                q = -q
        qs.append(q)

    q_mean = np.mean(np.stack(qs, axis=0), axis=0)
    q_mean = q_mean / np.linalg.norm(q_mean)
    avg_R = quaternion_to_matrix_wxyz(q_mean)

    errs = [rot_angle_deg(avg_R.T @ np.asarray(R, dtype=np.float64)) for R in rotations]
    return avg_R, float(np.std(errs))


def sample_hold_1d(t_src: np.ndarray, x_src: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """ZOH sample x_src at t_query (assumes t_src sorted)."""
    out = np.zeros((t_query.size,) + x_src.shape[1:], dtype=x_src.dtype)
    idx = 0
    for i, tq in enumerate(t_query):
        while idx + 1 < t_src.size and t_src[idx + 1] <= tq:
            idx += 1
        out[i] = x_src[idx]
    return out


# ---------------------------------------------------------------------------
# AprilTag loading & world alignment
# ---------------------------------------------------------------------------

def load_apriltag_rotations_by_time(session_dir: Path) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Load AprilTag rotations from `color_apriltag/detection_summary.json`.

    Returns:
      tag_id -> { utc_timestamp_ns -> C_R_T }
    where C_R_T is the rotation of the tag frame in the camera frame.
    """
    frames_csv = session_dir / "frames.csv"
    det_sum = session_dir / "color_apriltag" / "detection_summary.json"
    if not frames_csv.exists() or not det_sum.exists():
        return {}

    frame_ts: Dict[int, int] = {}
    with open(frames_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                frame_ts[int(row["frame_id"])] = int(row["utc_timestamp_ns"])
            except Exception:
                continue

    data = json.loads(det_sum.read_text())

    def _frame_id_from_filename(name: str) -> Optional[int]:
        m = re.search(r"frame_(\d+)", name)
        return int(m.group(1)) if m else None

    out: Dict[int, Dict[int, np.ndarray]] = {}
    for img in data.get("images", []) or []:
        fid = _frame_id_from_filename(str(img.get("filename", "")))
        if fid is None or fid not in frame_ts:
            continue
        t_ns = int(frame_ts[fid])
        for det in img.get("detections", []) or []:
            tid = det.get("tag_id")
            R = det.get("rotation_matrix")
            if tid is None or R is None:
                continue
            if det.get("is_mirrored", False):
                continue
            Rm = np.array(R, dtype=np.float64)
            if Rm.shape != (3, 3) or not np.all(np.isfinite(Rm)):
                continue
            out.setdefault(int(tid), {})[t_ns] = Rm

    return out


def estimate_world_alignment_from_tags(
    *,
    streams: Dict[int, Dict[str, np.ndarray]],
    tag_rots_by_time: Dict[int, Dict[int, np.ndarray]],
    joint_to_imu: Dict[str, int],
    imu_time_offset_ns: int,
    sensor_from_tag: np.ndarray,
    quat_is_world_from_sensor: bool = True,
    min_samples: int = 60,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[int, float]]]:
    """
    Estimate a constant alignment for each joint IMU: W_pelvis <- W_joint.

    This is Option A: use AprilTag poses to align each IMU's internal world frame into the pelvis IMU world.

    IMPORTANT: We do *not* require tags to be co-visible in the same frame.
    We estimate each C_R_Wi independently:
        C_R_Wi(t) = C_R_Ti(t) @ (Wi_R_Ti(t))^T
    and average over time to get avg_C_R_Wi, then:
        Wp_R_Wi = (avg_C_R_Wp)^T @ avg_C_R_Wi
    """
    def _estimate_C_R_W(joint: str) -> Tuple[Optional[np.ndarray], int, float]:
        tag_id = JOINT_TO_TAG_ID.get(joint)
        imu_id = joint_to_imu.get(joint)
        if tag_id is None or imu_id is None:
            return None, 0, 0.0
        t_to_R = tag_rots_by_time.get(tag_id)
        if not t_to_R:
            return None, 0, 0.0

        times = np.array(sorted(t_to_R.keys()), dtype=np.int64)
        imu_t = streams[imu_id]["t_ns"].astype(np.int64)
        if imu_t.size == 0:
            return None, 0, 0.0
        t_query = times + int(imu_time_offset_ns)
        mask = (t_query >= int(imu_t[0])) & (t_query <= int(imu_t[-1]))
        times = times[mask]
        t_query = t_query[mask]
        if times.size < min_samples:
            return None, int(times.size), 0.0

        q = sample_hold_1d(imu_t, streams[imu_id]["quat_wxyz"], t_query)
        Wi_R_S = np.stack([quaternion_to_matrix_wxyz(qq) for qq in q], axis=0)
        if not quat_is_world_from_sensor:
            Wi_R_S = np.transpose(Wi_R_S, (0, 2, 1))
        Wi_R_T = np.einsum("nij,jk->nik", Wi_R_S, sensor_from_tag)

        C_R_T = np.stack([t_to_R[int(t)] for t in times.tolist()], axis=0)
        C_R_W = np.einsum("nij,njk->nik", C_R_T, np.transpose(Wi_R_T, (0, 2, 1)))
        avg_C_R_W, std_deg = average_rotations_quaternion([C_R_W[i] for i in range(C_R_W.shape[0])])
        return avg_C_R_W, int(times.size), std_deg

    C_R_Wp, n_p, std_p = _estimate_C_R_W("pelvis")
    if C_R_Wp is None:
        return {}, {}

    align: Dict[str, np.ndarray] = {"pelvis": np.eye(3, dtype=np.float64)}
    stats: Dict[str, Tuple[int, float]] = {"pelvis": (n_p, std_p)}

    for joint in JOINT_NAMES:
        if joint == "pelvis":
            continue
        C_R_Wi, n_i, std_i = _estimate_C_R_W(joint)
        if C_R_Wi is None:
            continue
        align[joint] = C_R_Wp.T @ C_R_Wi
        stats[joint] = (n_i, std_i)

    return align, stats
