"""Shared evaluation utilities for OptiTrack GT, Aria trajectory,
coordinate alignment, and timestamp matching.

Used by visualize_egoallo.py, visualize_imu_only.py, visualize_sam3d.py.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

_ts = get_typestore(Stores.ROS1_NOETIC)

# Ensure repo root is on sys.path for utils.sync_utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

ARIA_DIR = Path("aria_recordings")
RECORDINGS_DIR = Path("received_recordings")
BAG_DIR = Path("evaluation/optitrack_gt_data/ros_bag")
FITS_DIR = Path("evaluation/optitrack_gt_data/smplx_fits")
CALIB_DIR = Path("camera_calibration")
OUT_DIR = Path("evaluation/optitrack_gt_data/gifs")

DATASET_TO_ARIA = {f"dataset{i}": f"dataset{i}" for i in range(1, 12)}

BODY_BONES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11), (9, 12),
    (12, 13), (12, 14), (12, 15),
    (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
]

NUM_BODY_JOINTS = 22


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AriaTrajectory:
    utc_ns: np.ndarray          # (N,) int64
    position: np.ndarray        # (N, 3) float64
    orientation: np.ndarray     # (N, 4) float64  quaternion (x,y,z,w)


@dataclass
class OptiTrackBag:
    timestamps: list
    body: list
    camera: list
    box: list
    object_label: str = "box"


# ---------------------------------------------------------------------------
# Aria trajectory loader
# ---------------------------------------------------------------------------

def load_aria_trajectory(csv_path: Path) -> AriaTrajectory:
    utc, pos, ori = [], [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                utc.append(int(row["utc_timestamp_ns"]))
                pos.append([float(row[f"t{a}_world_device"]) for a in "xyz"])
                ori.append([float(row[f"q{a}_world_device"]) for a in "xyzw"])
            except Exception:
                continue
    return AriaTrajectory(
        utc_ns=np.array(utc, dtype=np.int64),
        position=np.array(pos, dtype=np.float64),
        orientation=np.array(ori, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# OptiTrack ROS bag loader
# ---------------------------------------------------------------------------

def _detect_yup(body_frames: list[np.ndarray]) -> bool:
    """Return True if the bag uses Y-up convention instead of Z-up."""
    sample = body_frames[0]
    y_span = sample[:, 1].max() - sample[:, 1].min()
    z_span = sample[:, 2].max() - sample[:, 2].min()
    return y_span > z_span * 2


def _yup_pos_to_zup(pos: np.ndarray) -> np.ndarray:
    """Convert positions from (X-right, Y-up, Z-back) → (X-right, Y-forward, Z-up)."""
    out = np.empty_like(pos)
    out[..., 0] = pos[..., 0]
    out[..., 1] = -pos[..., 2]
    out[..., 2] = pos[..., 1]
    return out


def _yup_quat_to_zup(quat_xyzw: np.ndarray) -> np.ndarray:
    """Rotate a quaternion from Y-up frame to Z-up frame."""
    from scipy.spatial.transform import Rotation
    r_conv = Rotation.from_matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    r_orig = Rotation.from_quat(quat_xyzw)
    return (r_conv * r_orig).as_quat()


def load_optitrack_bag(bag_path: Path) -> OptiTrackBag:
    body_msgs, cam_msgs, box_msgs = {}, {}, {}
    object_label = "box"
    with Reader(bag_path) as reader:
        for conn, stamp, raw in reader.messages():
            msg = _ts.deserialize_ros1(raw, conn.msgtype)
            if "fullbody" in conn.topic:
                body_msgs[stamp] = np.array(
                    [[p.position.x, p.position.y, p.position.z] for p in msg.poses]
                )
            elif "camera_hand" in conn.topic:
                p = msg.pose
                cam_msgs[stamp] = (
                    np.array([p.position.x, p.position.y, p.position.z]),
                    np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]),
                )
            else:
                # Any other PoseStamped topic is a tracked object
                # (box, tennis_racket, tennis_ball, etc.)
                parts = conn.topic.strip("/").split("/")
                object_label = parts[-2] if len(parts) >= 2 else parts[-1]
                p = msg.pose
                box_msgs[stamp] = (
                    np.array([p.position.x, p.position.y, p.position.z]),
                    np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]),
                )

    sorted_ts = sorted(body_msgs.keys())
    out = OptiTrackBag(timestamps=[], body=[], camera=[], box=[],
                       object_label=object_label.replace("_", " "))
    for t in sorted_ts:
        out.timestamps.append(t)
        out.body.append(body_msgs[t])
        if cam_msgs:
            nearest = min(cam_msgs, key=lambda ct: abs(ct - t))
            out.camera.append(cam_msgs[nearest])
        if box_msgs:
            nearest = min(box_msgs, key=lambda bt: abs(bt - t))
            out.box.append(box_msgs[nearest])

    # Auto-detect Y-up bags and convert to Z-up
    if out.body and _detect_yup(out.body):
        out.body = [_yup_pos_to_zup(b) for b in out.body]
        out.camera = [
            (_yup_pos_to_zup(pos), _yup_quat_to_zup(quat))
            for pos, quat in out.camera
        ]
        out.box = [
            (_yup_pos_to_zup(pos), _yup_quat_to_zup(quat))
            for pos, quat in out.box
        ]

    return out


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def resolve_calibration_json(path: Path) -> Path:
    p = path.resolve()
    if p.is_file():
        return p
    if p.is_dir():
        direct = p / "calibration_result.json"
        if direct.exists():
            return direct
        candidates = sorted(p.glob("**/calibration_result.json"))
        if candidates:
            return max(candidates, key=lambda x: x.stat().st_mtime)
    raise FileNotFoundError(f"No calibration_result.json found under {path}")


def load_cam_to_hand(calib_json: Path) -> np.ndarray:
    data = json.loads(calib_json.read_text())
    if "T_cam2hand" not in data:
        raise KeyError(f"T_cam2hand missing in {calib_json}")
    T = np.array(data["T_cam2hand"], dtype=np.float32)
    if T.shape != (4, 4):
        raise ValueError(f"T_cam2hand has invalid shape {T.shape}")
    return T


# ---------------------------------------------------------------------------
# OptiTrack fitted SMPL-X ground truth
# ---------------------------------------------------------------------------

def load_optitrack_smplx_fit(
    dataset_name: str,
    fits_dir: Path = FITS_DIR,
) -> Optional[dict]:
    """Load fitted SMPL-X joints/vertices in OptiTrack Z-up coords.

    Returns dict with 'joints_zup', 'vertices_zup', 'faces',
    'timestamps', 'subsample', or None.
    """
    bag_name = re.sub(r"dataset(\d)", r"dataset_\1", dataset_name)
    fit_path = fits_dir / f"{bag_name}.npz"
    if not fit_path.exists():
        return None

    try:
        import smplx as smplx_pkg
        import torch
    except ImportError:
        print("  Warning: smplx/torch not available, skipping GT overlay")
        return None

    fit = np.load(fit_path)
    device = torch.device("cpu")
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "MHR" / "model"

    body_model = smplx_pkg.create(
        str(model_dir), model_type="smplx", gender="neutral", ext="npz",
        num_betas=10, use_pca=False, flat_hand_mean=True, batch_size=1,
    ).to(device)
    body_model.eval()

    global_orient = torch.tensor(fit["global_orient"], dtype=torch.float32)
    transl = torch.tensor(fit["transl"], dtype=torch.float32)
    body_pose = torch.tensor(fit["body_pose"], dtype=torch.float32)
    betas = torch.tensor(fit["betas"], dtype=torch.float32)

    N = global_orient.shape[0]
    all_joints, all_verts = [], []
    zero_jaw = torch.zeros(1, 3)
    zero_eye = torch.zeros(1, 3)
    zero_hand = torch.zeros(1, 45)
    zero_expr = torch.zeros(1, 10)

    batch_size = 256
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            B = end - start
            out = body_model(
                betas=betas.unsqueeze(0).expand(B, -1),
                global_orient=global_orient[start:end],
                body_pose=body_pose[start:end],
                transl=transl[start:end],
                jaw_pose=zero_jaw.expand(B, -1),
                leye_pose=zero_eye.expand(B, -1),
                reye_pose=zero_eye.expand(B, -1),
                left_hand_pose=zero_hand.expand(B, -1),
                right_hand_pose=zero_hand.expand(B, -1),
                expression=zero_expr.expand(B, -1),
            )
            all_joints.append(out.joints[:, :22, :].cpu().numpy())
            all_verts.append(out.vertices.cpu().numpy())

    joints_yup = np.concatenate(all_joints, axis=0)
    verts_yup = np.concatenate(all_verts, axis=0)

    return {
        "joints_zup": _yup_to_zup(joints_yup),
        "vertices_zup": _yup_to_zup(verts_yup),
        "faces": body_model.faces.astype(np.int32),
        "timestamps": fit["timestamps"],
        "subsample": int(fit["subsample"]),
    }


def _yup_to_zup(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr)
    out[..., 0] = arr[..., 0]
    out[..., 1] = -arr[..., 2]
    out[..., 2] = arr[..., 1]
    return out


# ---------------------------------------------------------------------------
# Calibration duration
# ---------------------------------------------------------------------------

def load_calib_duration_sec(dataset_name: str) -> Optional[float]:
    """Load ``calib_duration_sec`` from the dataset's ``imu_calibration.json``.

    Returns *None* when the file or field is missing.
    """
    calib_path = RECORDINGS_DIR / dataset_name / "imu_calibration.json"
    if not calib_path.exists():
        return None
    data = json.loads(calib_path.read_text())
    return data.get("calib_duration_sec")


# ---------------------------------------------------------------------------
# Timestamp matching
# ---------------------------------------------------------------------------

def nearest_indices(source_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """For each query timestamp, return the index of the nearest source timestamp."""
    idx = np.searchsorted(source_ts, query_ts, side="left")
    idx = np.clip(idx, 0, source_ts.shape[0] - 1)
    prev = np.clip(idx - 1, 0, source_ts.shape[0] - 1)
    choose_prev = np.abs(query_ts - source_ts[prev]) <= np.abs(source_ts[idx] - query_ts)
    return np.where(choose_prev, prev, idx).astype(np.int64)


def tracking_ns_to_utc_ns(
    tracking_timestamps_ns: np.ndarray,
    trajectory_csv: Path,
) -> np.ndarray:
    """Convert Aria tracking timestamps to UTC using the MPS trajectory CSV.

    Uses the same nearest-neighbor mapping as 04_sync_pipeline.py.
    """
    from utils.sync_utils import (
        load_tracking_to_utc_dict,
        get_sorted_keys_for_dict,
        get_nearest_utc_with_error,
    )
    track_to_utc = load_tracking_to_utc_dict(trajectory_csv)
    sorted_tracking_us = get_sorted_keys_for_dict(track_to_utc)
    if not sorted_tracking_us:
        return tracking_timestamps_ns

    utc_list = []
    for t_ns in tracking_timestamps_ns:
        utc_ns, _ = get_nearest_utc_with_error(
            track_to_utc, sorted_tracking_us, int(t_ns),
        )
        utc_list.append(utc_ns if utc_ns is not None else int(t_ns))
    return np.array(utc_list, dtype=np.int64)


# ---------------------------------------------------------------------------
# Coordinate alignment: Aria world -> OptiTrack world
# ---------------------------------------------------------------------------

def estimate_aria_to_optitrack(
    aria: AriaTrajectory,
    optitrack: OptiTrackBag,
    head_to_pelvis_z: float = 0.65,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate yaw rotation + translation from Aria world to OptiTrack world.

    Both frames are Z-up (gravity-aligned), so only yaw + 3D offset needed.
    Uses the OptiTrack pelvis (joint 0) as reference.

    Returns (R_3x3, t_3) such that  p_opti = R @ p_aria + t.
    """
    opti_ts = np.array(optitrack.timestamps, dtype=np.int64)
    opti_pelvis = np.array([b[0] for b in optitrack.body], dtype=np.float64)

    aria_ts = aria.utc_ns
    idx = np.searchsorted(aria_ts, opti_ts, side="left")
    idx = np.clip(idx, 0, len(aria_ts) - 1)
    aria_matched = aria.position[idx].copy()
    aria_matched[:, 2] -= head_to_pelvis_z

    n = min(len(opti_pelvis), len(aria_matched))
    A = aria_matched[:n]
    B = opti_pelvis[:n]

    cA = A.mean(axis=0)
    cB = B.mean(axis=0)

    H = (A - cA)[:, :2].T @ (B - cB)[:, :2]
    U, _, Vt = np.linalg.svd(H)
    R2 = Vt.T @ U.T
    if np.linalg.det(R2) < 0:
        Vt[-1] *= -1
        R2 = Vt.T @ U.T

    R = np.eye(3, dtype=np.float64)
    R[:2, :2] = R2
    t = cB - R @ cA
    return R, t
