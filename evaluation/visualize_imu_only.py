"""Visualize IMU-only SMPL body localized via Aria SLAM trajectory.

The IMU-only pipeline provides body joint rotations but no global
translation.  We obtain localization from Project Aria's closed-loop
SLAM trajectory and apply a naive glasses-to-pelvis vertical offset.

The resulting body is aligned to the OptiTrack world frame (Z-up) via
Procrustes on the Aria+OptiTrack pelvis trajectories, so that OptiTrack
camera and object positions can be overlaid for reference.

Usage:
    python evaluation/visualize_imu_only.py --datasets dataset1
    python evaluation/visualize_imu_only.py --datasets dataset1 dataset5 --debug-frame 0
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import multiprocessing as mp
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.eval_utils import (
    ARIA_DIR,
    BAG_DIR,
    CALIB_DIR,
    DATASET_TO_ARIA,
    BODY_BONES,
    NUM_BODY_JOINTS,
    RECORDINGS_DIR,
    AriaTrajectory,
    OptiTrackBag,
    estimate_aria_to_optitrack,
    load_aria_trajectory,
    load_calib_duration_sec,
    load_cam_to_hand,
    load_optitrack_bag,
    load_optitrack_smplx_fit,
    nearest_indices,
    resolve_calibration_json,
)

OUT_DIR = Path("evaluation/imu_naive")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_SMPLX_MODEL = Path("model/smplx/SMPLX_NEUTRAL.npz")

# ── IMU constants ─────────────────────────────────────────────────────
IMU_ID_TO_JOINT = {
    1: "pelvis",
    2: "left-shoulder",
    3: "right-shoulder",
    4: "left-elbow",
    5: "right-elbow",
    6: "left-hip",
    7: "right-hip",
    8: "left-knee",
    9: "right-knee",
}
JOINT_NAMES = list(IMU_ID_TO_JOINT.values())
JOINT_TO_TAG_ID = {
    "pelvis": 0, "left-shoulder": 1, "right-shoulder": 2,
    "left-elbow": 3, "right-elbow": 4, "left-hip": 5,
    "right-hip": 6, "left-knee": 7, "right-knee": 8,
}
TAG_ID_TO_JOINT = {v: k for k, v in JOINT_TO_TAG_ID.items()}

T_R_IMU = np.array(
    [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float64,
)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║                     Data structures                              ║
# ╚═══════════════════════════════════════════════════════════════════╝

@dataclass
class SmplxModel:
    faces: np.ndarray
    v_template: np.ndarray
    shapedirs: np.ndarray
    posedirs: np.ndarray
    J_regressor: np.ndarray
    weights: np.ndarray
    parents: np.ndarray

    @property
    def num_joints(self) -> int:
        return int(self.weights.shape[1])


# ╔═══════════════════════════════════════════════════════════════════╗
# ║                 Quaternion / rotation helpers                    ║
# ╚═══════════════════════════════════════════════════════════════════╝

def _qwxyz_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).ravel()
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _mat_to_qwxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = float(np.trace(R))
    if t > 0:
        s = np.sqrt(t + 1) * 2
        return np.array([0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s])
    if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = np.sqrt(1+R[0,0]-R[1,1]-R[2,2]) * 2
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    if R[1,1] > R[2,2]:
        s = np.sqrt(1+R[1,1]-R[0,0]-R[2,2]) * 2
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    s = np.sqrt(1+R[2,2]-R[0,0]-R[1,1]) * 2
    return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


def _average_rotations(rots: List[np.ndarray]) -> np.ndarray:
    """Average rotation matrices via quaternion mean."""
    if len(rots) == 1:
        return np.array(rots[0], dtype=np.float64)
    qs = []
    ref = None
    for R in rots:
        q = _mat_to_qwxyz(R)
        if ref is None:
            ref = q
        elif np.dot(q, ref) < 0:
            q = -q
        qs.append(q)
    qm = np.mean(qs, axis=0)
    qm /= np.linalg.norm(qm)
    return _qwxyz_to_mat(qm)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║                       Data loaders                               ║
# ╚═══════════════════════════════════════════════════════════════════╝

def load_smplx_model(path: Path, betas_dim: int = 10) -> SmplxModel:
    m = np.load(path, allow_pickle=True)
    parents = m["kintree_table"][0].astype(np.int64)
    parents[parents > 1_000_000_000] = -1
    sd = m["shapedirs"]
    if sd.dtype != np.float32:
        sd = sd.astype(np.float32)
    if sd.shape[2] > betas_dim:
        sd = sd[:, :, :betas_dim]
    return SmplxModel(
        faces=m["f"].astype(np.int32),
        v_template=m["v_template"].astype(np.float32),
        shapedirs=sd,
        posedirs=m["posedirs"].astype(np.float32),
        J_regressor=m["J_regressor"].astype(np.float32),
        weights=m["weights"].astype(np.float32),
        parents=parents,
    )


def precompute_shape(model: SmplxModel, betas: np.ndarray):
    betas = betas.astype(np.float32).ravel()
    v_shaped = model.v_template + np.einsum("vxb,b->vx", model.shapedirs, betas)
    j_tpose = model.J_regressor @ v_shaped
    return v_shaped, j_tpose


def smplx_fk(
    model: SmplxModel,
    local_rots: np.ndarray,
    v_shaped: np.ndarray,
    j_tpose: np.ndarray,
    compute_verts: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Minimal SMPLX forward kinematics (numpy LBS)."""
    J = model.num_joints
    lr = local_rots.astype(np.float32)

    T_parent = np.zeros((J, 4, 4), dtype=np.float32)
    T_parent[:] = np.eye(4, dtype=np.float32)
    T_parent[:, :3, :3] = lr
    T_parent[0, :3, 3] = j_tpose[0]
    for i in range(1, J):
        p = int(model.parents[i])
        T_parent[i, :3, 3] = j_tpose[i] - (j_tpose[p] if p >= 0 else 0)

    T_world = np.zeros_like(T_parent)
    for i in range(J):
        p = int(model.parents[i])
        T_world[i] = T_parent[i] if p < 0 else T_world[p] @ T_parent[i]

    joints = T_world[:, :3, 3].copy()

    if not compute_verts:
        return joints, None

    pose_delta = (lr[1:] - np.eye(3, dtype=np.float32)).reshape(-1)
    v_posed = v_shaped + np.tensordot(model.posedirs, pose_delta, axes=([2], [0]))

    v_delta = np.ones((v_posed.shape[0], J, 4), dtype=np.float32)
    v_delta[:, :, :3] = v_posed[:, None, :] - j_tpose[None, :, :]
    verts = np.einsum("jxy,vj,vjy->vx", T_world[:, :3, :], model.weights, v_delta)
    return joints, verts


# ── IMU data ──────────────────────────────────────────────────────────

def load_imu_csv(path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    streams: Dict[int, list] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                iid = int(row["imu_id"])
                t = int(row["utc_timestamp_ns"])
                qx, qy, qz, qw = (float(row[k]) for k in ("quatI", "quatJ", "quatK", "quatW"))
            except Exception:
                continue
            streams.setdefault(iid, []).append((t, np.array([qw, qx, qy, qz], dtype=np.float64)))
    out = {}
    for iid, items in streams.items():
        items.sort(key=lambda x: x[0])
        out[iid] = {
            "t_ns": np.array([x[0] for x in items], dtype=np.int64),
            "quat_wxyz": np.stack([x[1] for x in items]),
        }
    return out


def load_imu_calibration(path: Path) -> Dict[str, np.ndarray]:
    data = json.loads(path.read_text())
    return {name: np.array(j["B_R_S"], dtype=np.float64)
            for name, j in data.get("joints", {}).items()}


def _sample_hold(t_src: np.ndarray, x_src: np.ndarray, t_q: np.ndarray) -> np.ndarray:
    """Zero-order hold sample."""
    out = np.zeros((t_q.size,) + x_src.shape[1:], dtype=x_src.dtype)
    idx = 0
    for i, tq in enumerate(t_q):
        while idx + 1 < t_src.size and t_src[idx + 1] <= tq:
            idx += 1
        out[i] = x_src[idx]
    return out


# ── AprilTag world-alignment ─────────────────────────────────────────

def load_apriltag_poses(
    session_dir: Path,
) -> Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """tag_id -> {utc_ns -> (C_R_T, C_t_T)}."""
    frames_csv = session_dir / "frames.csv"
    det_sum = session_dir / "color_apriltag" / "detection_summary.json"
    if not frames_csv.exists() or not det_sum.exists():
        return {}
    frame_ts: Dict[int, int] = {}
    with open(frames_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                frame_ts[int(row["frame_id"])] = int(row["utc_timestamp_ns"])
            except Exception:
                continue
    data = json.loads(det_sum.read_text())
    out: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
    for img in data.get("images", []) or []:
        m = re.search(r"frame_(\d+)", str(img.get("filename", "")))
        if not m:
            continue
        fid = int(m.group(1))
        if fid not in frame_ts:
            continue
        t_ns = frame_ts[fid]
        for det in img.get("detections", []) or []:
            tid = det.get("tag_id")
            R = det.get("rotation_matrix")
            t = det.get("translation")
            if tid is None or R is None or t is None or det.get("is_mirrored", False):
                continue
            Rm = np.array(R, dtype=np.float64)
            tm = np.array(t, dtype=np.float64).reshape(-1)
            if (
                Rm.shape == (3, 3)
                and tm.shape == (3,)
                and np.all(np.isfinite(Rm))
                and np.all(np.isfinite(tm))
            ):
                out.setdefault(int(tid), {})[t_ns] = (Rm, tm)
    return out


def extract_apriltag_rotations(
    tag_poses: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
) -> Dict[int, Dict[int, np.ndarray]]:
    """Keep only rotation matrices from `load_apriltag_poses` output."""
    return {
        tid: {t_ns: pose[0] for t_ns, pose in per_tag.items()}
        for tid, per_tag in tag_poses.items()
    }


def load_apriltag_rotations(session_dir: Path) -> Dict[int, Dict[int, np.ndarray]]:
    """Backward-compatible wrapper returning only AprilTag rotations."""
    return extract_apriltag_rotations(load_apriltag_poses(session_dir))


def estimate_world_alignment(
    streams: Dict[int, Dict[str, np.ndarray]],
    tag_rots: Dict[int, Dict[int, np.ndarray]],
    min_samples: int = 60,
) -> Dict[str, np.ndarray]:
    """Estimate Wp_R_Wi (pelvis-world from each IMU's world) via AprilTags."""
    joint_to_imu = {j: iid for iid, j in IMU_ID_TO_JOINT.items()}
    sensor_from_tag = T_R_IMU.T

    def _C_R_W(joint: str):
        tag_id = JOINT_TO_TAG_ID.get(joint)
        imu_id = joint_to_imu.get(joint)
        if tag_id is None or imu_id is None:
            return None
        t2r = tag_rots.get(tag_id)
        if not t2r:
            return None
        times = np.array(sorted(t2r.keys()), dtype=np.int64)
        imu_t = streams[imu_id]["t_ns"].astype(np.int64)
        if imu_t.size == 0:
            return None
        mask = (times >= int(imu_t[0])) & (times <= int(imu_t[-1]))
        times = times[mask]
        if times.size < min_samples:
            return None
        q = _sample_hold(imu_t, streams[imu_id]["quat_wxyz"], times)
        Wi_R_S = np.stack([_qwxyz_to_mat(qq) for qq in q])
        Wi_R_T = np.einsum("nij,jk->nik", Wi_R_S, sensor_from_tag)
        C_R_T = np.stack([t2r[int(t)] for t in times])
        C_R_W = np.einsum("nij,njk->nik", C_R_T, np.transpose(Wi_R_T, (0, 2, 1)))
        return _average_rotations([C_R_W[i] for i in range(C_R_W.shape[0])])

    C_R_Wp = _C_R_W("pelvis")
    if C_R_Wp is None:
        return {}
    align: Dict[str, np.ndarray] = {"pelvis": np.eye(3, dtype=np.float64)}
    for joint in JOINT_NAMES:
        if joint == "pelvis":
            continue
        C_R_Wi = _C_R_W(joint)
        if C_R_Wi is not None:
            align[joint] = C_R_Wp.T @ C_R_Wi
    return align


# ── Build calibrated IMU bone rotations per timestamp ─────────────────

def build_calibrated_bone_rotations(
    streams: Dict[int, Dict[str, np.ndarray]],
    calib: Dict[str, np.ndarray],
    world_align: Dict[str, np.ndarray],
    timeline_ns: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Sample IMU streams at *timeline_ns* and return {joint: (T, 3, 3) W_R_B}."""
    joint_to_imu = {j: iid for iid, j in IMU_ID_TO_JOINT.items()}
    bone_from_imu = {j: np.array(B_R_tag) @ T_R_IMU for j, B_R_tag in calib.items()}

    out: Dict[str, np.ndarray] = {}
    for joint in JOINT_NAMES:
        imu_id = joint_to_imu.get(joint)
        if imu_id is None or imu_id not in streams:
            out[joint] = np.tile(np.eye(3, dtype=np.float64), (len(timeline_ns), 1, 1))
            continue

        q_sampled = _sample_hold(streams[imu_id]["t_ns"], streams[imu_id]["quat_wxyz"], timeline_ns)
        W_R_S_all = np.stack([_qwxyz_to_mat(q) for q in q_sampled])

        Wa = world_align.get(joint)
        B_R_imu = bone_from_imu.get(joint, np.eye(3, dtype=np.float64))

        rots = np.empty_like(W_R_S_all)
        for i in range(len(timeline_ns)):
            W_R_S = W_R_S_all[i]
            if Wa is not None:
                W_R_S = Wa @ W_R_S
            rots[i] = W_R_S @ B_R_imu.T
        out[joint] = rots
    return out


def build_local_rots_from_global(global_rots: Dict[str, np.ndarray]) -> np.ndarray:
    """Build (55, 3, 3) SMPLX local rotations from sparse global bone rotations."""
    local = np.tile(np.eye(3, dtype=np.float64), (55, 1, 1))
    G = lambda name: global_rots.get(name, np.eye(3, dtype=np.float64))
    pelvis = G("pelvis")
    local[0] = pelvis
    local[1] = pelvis.T @ G("left-hip")
    local[2] = pelvis.T @ G("right-hip")
    local[4] = G("left-hip").T @ G("left-knee")
    local[5] = G("right-hip").T @ G("right-knee")
    local[16] = pelvis.T @ G("left-shoulder")
    local[17] = pelvis.T @ G("right-shoulder")
    local[18] = G("left-shoulder").T @ G("left-elbow")
    local[19] = G("right-shoulder").T @ G("right-elbow")
    return local.astype(np.float32)


# ── Aria SLAM trajectory ─────────────────────────────────────────────

def find_aria_trajectory(dataset_name: str) -> Optional[Path]:
    aria_name = DATASET_TO_ARIA.get(dataset_name)
    if aria_name is None:
        return None
    traj = ARIA_DIR / f"mps_{aria_name}_vrs" / "slam" / "closed_loop_trajectory.csv"
    return traj if traj.exists() else None


def _select_calibration_timestamps(
    all_frame_timestamps_ns: np.ndarray,
    calib_dur_sec: Optional[float],
    min_fallback_frames: int = 30,
) -> np.ndarray:
    """Pick timestamps for calibration-only alignment estimation."""
    if all_frame_timestamps_ns.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if calib_dur_sec is not None and calib_dur_sec > 0:
        t0 = int(all_frame_timestamps_ns[0])
        t1 = int(t0 + calib_dur_sec * 1e9)
        mask = (all_frame_timestamps_ns >= t0) & (all_frame_timestamps_ns <= t1)
        calib_ts = all_frame_timestamps_ns[mask]
        if calib_ts.size > 0:
            return calib_ts.astype(np.int64)
    n = min(len(all_frame_timestamps_ns), max(1, min_fallback_frames))
    return all_frame_timestamps_ns[:n].astype(np.int64)


def _rotz(yaw_rad: float) -> np.ndarray:
    c = float(np.cos(yaw_rad))
    s = float(np.sin(yaw_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _xy_kabsch_yaw_alignment(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Estimate yaw-only 3x3 rotation that best maps A to B (least squares on XY)."""
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
    return R


def estimate_world_camera_pose_calib_mean(
    optitrack: OptiTrackBag,
    calib_timestamps_ns: np.ndarray,
    calibration_path: Path,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Estimate static world camera pose from calibration-period OptiTrack samples."""
    if not optitrack.camera:
        raise ValueError("OptiTrack camera track unavailable")
    if calib_timestamps_ns.size == 0:
        raise ValueError("No calibration timestamps available")

    opti_ts = np.array(optitrack.timestamps, dtype=np.int64)
    sel_idx = np.unique(nearest_indices(opti_ts, calib_timestamps_ns))

    cam_pos_all = np.array([c[0] for c in optitrack.camera], dtype=np.float64)
    cam_quat_all = np.array([c[1] for c in optitrack.camera], dtype=np.float64)
    hand_pos = cam_pos_all[sel_idx]
    hand_quat = cam_quat_all[sel_idx]
    mean_hand_pos = hand_pos.mean(axis=0)
    mean_hand_rot = Rotation.from_quat(hand_quat).mean().as_matrix()

    T_world_hand = np.eye(4, dtype=np.float64)
    T_world_hand[:3, :3] = mean_hand_rot
    T_world_hand[:3, 3] = mean_hand_pos
    pose_src = f"hand_mean({len(sel_idx)} frames)"
    try:
        calib_json = resolve_calibration_json(calibration_path)
        T_cam2hand = load_cam_to_hand(calib_json).astype(np.float64)
        T_world_cam = T_world_hand @ T_cam2hand
        pose_src = f"{pose_src}+T_cam2hand"
    except (FileNotFoundError, KeyError, ValueError):
        T_world_cam = T_world_hand
        pose_src = f"{pose_src}+raw_hand"
    return T_world_cam[:3, :3], T_world_cam[:3, 3], pose_src


def estimate_aria_to_optitrack_tag_calib(
    aria: AriaTrajectory,
    optitrack: OptiTrackBag,
    calib_timestamps_ns: np.ndarray,
    tag_poses: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    calibration_path: Path,
    head_to_pelvis_z: float = 0.65,
    pelvis_tag_id: int = 0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Estimate Aria->OptiTrack from calibration-period pelvis-tag positions."""
    if calib_timestamps_ns.size == 0:
        raise ValueError("No calibration timestamps available")
    per_tag = tag_poses.get(int(pelvis_tag_id))
    if not per_tag:
        raise ValueError(f"No detections for pelvis tag_id={pelvis_tag_id}")

    t0 = int(calib_timestamps_ns[0])
    t1 = int(calib_timestamps_ns[-1])
    tag_ts = np.array(sorted(t for t in per_tag.keys() if t0 <= int(t) <= t1), dtype=np.int64)
    if tag_ts.size < 5:
        raise ValueError(
            f"Too few pelvis-tag detections in calibration window ({tag_ts.size})"
        )

    W_R_C, W_t_C, cam_pose_src = estimate_world_camera_pose_calib_mean(
        optitrack=optitrack,
        calib_timestamps_ns=calib_timestamps_ns,
        calibration_path=calibration_path,
    )

    C_t_T = np.stack([per_tag[int(t)][1] for t in tag_ts], axis=0)
    B = (W_R_C @ C_t_T.T).T + W_t_C[None, :]

    aria_idx = nearest_indices(aria.utc_ns, tag_ts)
    A = aria.position[aria_idx].copy()
    A[:, 2] -= head_to_pelvis_z

    R = _xy_kabsch_yaw_alignment(A, B)
    t = B.mean(axis=0) - R @ A.mean(axis=0)
    yaw_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

    info = {
        "yaw_deg": yaw_deg,
        "n_samples": int(tag_ts.size),
        "cam_pose_source": cam_pose_src,
        "tag_id": int(pelvis_tag_id),
    }
    return R, t.astype(np.float64), info


def estimate_aria_to_optitrack_camera_facing_init(
    aria: AriaTrajectory,
    optitrack: OptiTrackBag,
    calib_timestamps_ns: np.ndarray,
    head_to_pelvis_z: float = 0.65,
    best_facing_n: int = 20,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Estimate Aria->OptiTrack alignment from calibration-start camera-facing prior.

    Yaw is estimated from orientation-only assumption:
      "during calibration, Aria glasses face the (static) camera."

    Instead of averaging yaw over all calibration frames, we score each
    frame by how well the Aria forward direction aligns with the expected
    camera-facing direction (cosine similarity in XY plane).  Only the
    *best_facing_n* frames with the highest facing score are used for the
    yaw estimate, making it robust to frames where the subject is turning
    or not directly facing the camera.

    Translation is estimated from calibration timestamps only (no future frames).
    """
    if calib_timestamps_ns.size == 0:
        raise ValueError("No calibration timestamps available")
    if not optitrack.camera:
        raise ValueError("OptiTrack camera track unavailable")

    aria_idx = nearest_indices(aria.utc_ns, calib_timestamps_ns)
    opti_ts = np.array(optitrack.timestamps, dtype=np.int64)
    opti_idx = nearest_indices(opti_ts, calib_timestamps_ns)

    aria_rot = Rotation.from_quat(aria.orientation[aria_idx]).as_matrix().astype(np.float64)
    cam_quat = np.array([c[1] for c in optitrack.camera], dtype=np.float64)
    cam_rot_all = Rotation.from_quat(cam_quat).as_matrix().astype(np.float64)
    cam_rot = cam_rot_all[opti_idx]

    # Aria device +Z is treated as forward; camera +Z forward => face camera target is -Z.
    aria_fwd = np.einsum("nij,j->ni", aria_rot, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    target_fwd = -cam_rot[:, :, 2]
    aria_xy = aria_fwd[:, :2]
    target_xy = target_fwd[:, :2]
    aria_norm = np.linalg.norm(aria_xy, axis=1)
    target_norm = np.linalg.norm(target_xy, axis=1)
    valid = (aria_norm > 1e-8) & (target_norm > 1e-8)
    if int(valid.sum()) < 5:
        raise ValueError("Too few valid calibration samples for camera-facing yaw")

    # Compute per-frame facing score (cosine similarity in XY)
    facing_score = np.full(len(aria_xy), -2.0)
    facing_score[valid] = (
        np.sum(
            (aria_xy[valid] / aria_norm[valid, None])
            * (target_xy[valid] / target_norm[valid, None]),
            axis=1,
        )
    )

    # Select the best-facing frames for yaw estimation
    n_use = min(best_facing_n, int(valid.sum()))
    # Indices into the full array, sorted by facing score descending
    best_idx = np.argsort(facing_score)[-n_use:]
    best_mask = np.zeros(len(aria_xy), dtype=bool)
    best_mask[best_idx] = True
    best_valid = best_mask & valid

    dyaw = (
        np.arctan2(target_xy[best_valid, 1], target_xy[best_valid, 0])
        - np.arctan2(aria_xy[best_valid, 1], aria_xy[best_valid, 0])
    )
    yaw_rad = float(np.arctan2(np.mean(np.sin(dyaw)), np.mean(np.cos(dyaw))))
    R_align = _rotz(yaw_rad)

    # Also compute the all-frames yaw for diagnostic comparison
    dyaw_all = (
        np.arctan2(target_xy[valid, 1], target_xy[valid, 0])
        - np.arctan2(aria_xy[valid, 1], aria_xy[valid, 0])
    )
    yaw_all_rad = float(np.arctan2(np.mean(np.sin(dyaw_all)), np.mean(np.cos(dyaw_all))))

    trans_info = "camera_mean"
    if optitrack.body:
        opti_ts_min = int(calib_timestamps_ns[0])
        opti_ts_max = int(calib_timestamps_ns[-1])
        mask = (opti_ts >= opti_ts_min) & (opti_ts <= opti_ts_max)
        if not np.any(mask):
            # Fall back to nearest OptiTrack body frame to calibration start.
            nearest = int(np.argmin(np.abs(opti_ts - calib_timestamps_ns[0])))
            mask = np.zeros_like(opti_ts, dtype=bool)
            mask[nearest] = True

        opti_idx_body = np.where(mask)[0]
        opti_ts_used = opti_ts[opti_idx_body]
        opti_pelvis = np.array([optitrack.body[int(i)][0] for i in opti_idx_body], dtype=np.float64)

        aria_idx_body = nearest_indices(aria.utc_ns, opti_ts_used)
        aria_pelvis = aria.position[aria_idx_body].copy()
        aria_pelvis[:, 2] -= head_to_pelvis_z

        t_align = np.mean(opti_pelvis - (R_align @ aria_pelvis.T).T, axis=0)
        trans_info = f"pelvis_calib_mean({len(opti_idx_body)} frames)"
    else:
        cam_pos_all = np.array([c[0] for c in optitrack.camera], dtype=np.float64)
        cam_pos = cam_pos_all[opti_idx]
        aria_glasses = aria.position[aria_idx]
        t_align = np.mean(cam_pos - (R_align @ aria_glasses.T).T, axis=0)

    info = {
        "yaw_deg": float(np.degrees(yaw_rad)),
        "yaw_all_frames_deg": float(np.degrees(yaw_all_rad)),
        "n_calib_samples": int(valid.sum()),
        "n_best_facing": n_use,
        "best_facing_score_mean": float(np.mean(facing_score[best_valid])),
        "translation_source": trans_info,
    }
    return R_align, t_align.astype(np.float64), info




def transform_imu_world_to_optitrack(
    joints_imu: np.ndarray,
    imu_pelvis_global: np.ndarray,
    pelvis_pos_opti: np.ndarray,
) -> np.ndarray:
    """Move FK output from IMU world into OptiTrack world.

    The FK output has joints in the IMU pelvis world frame.
    We subtract the FK pelvis position, rotate by the alignment, and translate.
    """
    fk_pelvis = joints_imu[0].copy()
    centered = joints_imu - fk_pelvis[None, :]
    return centered + pelvis_pos_opti[None, :]


# ╔═══════════════════════════════════════════════════════════════════╗
# ║                       Rendering                                  ║
# ╚═══════════════════════════════════════════════════════════════════╝

_render_ctx: dict = {}


def _init_render_worker(ctx: dict):
    global _render_ctx
    _render_ctx = ctx


def _draw_frame(ax, frame_idx: int, ctx: dict):
    ax.cla()

    joints = ctx["joints_opti"][frame_idx]
    verts = ctx["verts_opti"][frame_idx] if ctx["verts_opti"] is not None else None
    faces = ctx["faces"]

    # Mesh
    if verts is not None:
        face_step = max(1, len(faces) // 3000)
        tris = verts[faces[::face_step]]
        ax.add_collection3d(Poly3DCollection(
            tris, alpha=ctx["mesh_alpha"], facecolor="salmon", edgecolor="none",
        ))

    # Skeleton
    ax.scatter(
        joints[:NUM_BODY_JOINTS, 0],
        joints[:NUM_BODY_JOINTS, 1],
        joints[:NUM_BODY_JOINTS, 2],
        c="crimson", s=25, alpha=0.9, marker="o",
        label="IMU-only", zorder=4,
    )
    for j1, j2 in BODY_BONES:
        if j1 < joints.shape[0] and j2 < joints.shape[0]:
            ax.plot(
                [joints[j1, 0], joints[j2, 0]],
                [joints[j1, 1], joints[j2, 1]],
                [joints[j1, 2], joints[j2, 2]],
                c="darkred", linewidth=1.2, alpha=0.7,
            )

    # OptiTrack fitted SMPL-X ground truth
    if ctx["show_gt"] and ctx["opti_fit"] is not None:
        opti_fit = ctx["opti_fit"]
        gt_idx = ctx["gt_match_idx"][frame_idx]
        gt_joints = opti_fit["joints_zup"][gt_idx].copy()
        gt_verts = opti_fit["vertices_zup"][gt_idx].copy()
        gt_faces = opti_fit["faces"]

        # Apply the same face-camera yaw correction as the prediction
        if ctx["face_cam_Rz"] is not None:
            Rz = ctx["face_cam_Rz"][frame_idx]
            pivot = gt_joints[0].copy()
            gt_joints = (Rz @ (gt_joints - pivot).T).T + pivot
            gt_verts = (Rz @ (gt_verts - pivot).T).T + pivot

        gt_face_step = max(1, len(gt_faces) // 2000)
        gt_tris = gt_verts[gt_faces[::gt_face_step]]
        ax.add_collection3d(Poly3DCollection(
            gt_tris, alpha=0.12, facecolor="deepskyblue", edgecolor="none",
        ))

        ax.scatter(
            gt_joints[:, 0], gt_joints[:, 1], gt_joints[:, 2],
            c="royalblue", s=18, alpha=0.85, marker="o",
            label="OptiTrack GT", zorder=3,
        )
        for j1, j2 in BODY_BONES:
            if j1 < gt_joints.shape[0] and j2 < gt_joints.shape[0]:
                ax.plot(
                    [gt_joints[j1, 0], gt_joints[j2, 0]],
                    [gt_joints[j1, 1], gt_joints[j2, 1]],
                    [gt_joints[j1, 2], gt_joints[j2, 2]],
                    c="steelblue", linewidth=1.0, alpha=0.6,
                )

    # Camera
    if ctx["cam_pos"] is not None:
        cam_pos = ctx["cam_pos"]
        cam_rot = ctx["cam_rot"]
        ax.scatter(*cam_pos, c="red", s=60, marker="^", label="Camera", zorder=5)
        ax_len = 0.15
        cam_axes = cam_rot * ax_len
        for i, c in enumerate(["red", "green", "blue"]):
            ax.quiver(
                cam_pos[0], cam_pos[1], cam_pos[2],
                cam_axes[0, i], cam_axes[1, i], cam_axes[2, i],
                color=c, linewidth=1.5, arrow_length_ratio=0.15,
            )
        # Frustum wireframe
        fd, fhw, fhh = 0.3, 0.12, 0.20
        corners_local = np.array([
            [-fhw, -fhh, fd], [fhw, -fhh, fd],
            [fhw, fhh, fd], [-fhw, fhh, fd],
        ], dtype=np.float32)
        corners_w = (cam_rot @ corners_local.T).T + cam_pos
        for cw in corners_w:
            ax.plot([cam_pos[0], cw[0]], [cam_pos[1], cw[1]], [cam_pos[2], cw[2]],
                    c="gray", linewidth=0.8, alpha=0.6)
        for i in range(4):
            j = (i + 1) % 4
            ax.plot([corners_w[i, 0], corners_w[j, 0]],
                    [corners_w[i, 1], corners_w[j, 1]],
                    [corners_w[i, 2], corners_w[j, 2]],
                    c="gray", linewidth=0.8, alpha=0.6)

    # Box / object
    if ctx["has_box"]:
        box_pos = ctx["box_positions"][frame_idx]
        ax.scatter(*box_pos, c="orange", s=120, marker="s",
                   label=f"Object ({ctx['object_label']})", zorder=5)

    # Aria glasses marker with orientation axes
    if ctx["glasses_opti"] is not None:
        gp = ctx["glasses_opti"][frame_idx]
        ax.scatter(*gp, c="limegreen", s=60, marker="D",
                   label="Aria glasses", zorder=5)

        # Draw glasses orientation axes (XYZ = RGB)
        if ctx["glasses_rot_opti"] is not None:
            g_rot = ctx["glasses_rot_opti"][frame_idx]
            g_ax_len = 0.12
            g_axes = g_rot * g_ax_len
            for i, c in enumerate(["red", "green", "blue"]):
                ax.quiver(
                    gp[0], gp[1], gp[2],
                    g_axes[0, i], g_axes[1, i], g_axes[2, i],
                    color=c, linewidth=1.2, arrow_length_ratio=0.2, alpha=0.8,
                )

        # Draw line from glasses to derived pelvis position
        pp = ctx["pelvis_pos_opti"][frame_idx]
        ax.plot(
            [gp[0], pp[0]], [gp[1], pp[1]], [gp[2], pp[2]],
            c="limegreen", linewidth=1.0, linestyle="--", alpha=0.6,
        )
        ax.scatter(*pp, c="limegreen", s=25, marker="x", zorder=5)

    # Ground grid
    center = ctx["center"]
    span = ctx["span"]
    gn = 10
    for gx in np.linspace(center[0] - span, center[0] + span, gn + 1):
        ax.plot([gx, gx], [center[1] - span, center[1] + span], [0, 0],
                c="lightgray", linewidth=0.4, alpha=0.5)
    for gy in np.linspace(center[1] - span, center[1] + span, gn + 1):
        ax.plot([center[0] - span, center[0] + span], [gy, gy], [0, 0],
                c="lightgray", linewidth=0.4, alpha=0.5)

    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    t0 = ctx["timestamps_ns"][0]
    time_s = (ctx["timestamps_ns"][frame_idx] - t0) / 1e9
    ax.set_title(
        f"{ctx['dataset_name']} IMU-only | t={time_s:.1f}s | "
        f"frame {frame_idx}/{ctx['n_frames']}"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.view_init(elev=25, azim=-60 + frame_idx * 0.3)


def _render_frame_to_png(args: tuple):
    frame_idx, png_path = args
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    _draw_frame(ax, frame_idx, _render_ctx)
    fig.savefig(png_path, dpi=100)
    plt.close(fig)


def _find_h264_encoder() -> str:
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True,
        ).stdout
    except FileNotFoundError:
        return "mpeg4"
    for enc in ("libx264", "h264_nvenc", "libopenh264"):
        if enc in out:
            return enc
    return "mpeg4"


# ╔═══════════════════════════════════════════════════════════════════╗
# ║                     Main pipeline                                ║
# ╚═══════════════════════════════════════════════════════════════════╝

def process_dataset(
    dataset_name: str,
    *,
    smplx_path: Path = DEFAULT_SMPLX_MODEL,
    calibration_path: Path = CALIB_DIR,
    mesh_alpha: float = 0.20,
    fps: int = 30,
    head_to_pelvis_z: float = 0.65,
    compute_verts: bool = True,
    show_gt: bool = True,
    debug_frame: Optional[int] = None,
    include_calibration: bool = False,
    face_camera: bool = False,
    align_mode: str = "gt_procrustes",
    best_facing_n: int = 20,
):
    dataset_dir = RECORDINGS_DIR / dataset_name
    bag_name = re.sub(r"dataset(\d)", r"dataset_\1", dataset_name) + ".bag"
    bag_path = BAG_DIR / bag_name
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}")
    print(f"{'='*60}")

    # ── Check prerequisites ──────────────────────────────────────────
    imu_csv = dataset_dir / "imu" / "imu_data.csv"
    calib_json = dataset_dir / "imu_calibration.json"
    if not imu_csv.exists():
        print(f"  SKIP: IMU data not found: {imu_csv}")
        return
    if not calib_json.exists():
        print(f"  SKIP: IMU calibration not found: {calib_json}")
        return

    aria_traj_path = find_aria_trajectory(dataset_name)
    if aria_traj_path is None:
        print(f"  SKIP: Aria trajectory not found for {dataset_name}")
        return

    if not bag_path.exists():
        print(f"  WARNING: OptiTrack bag not found: {bag_path} (camera/box skipped)")

    # ── Load SMPLX model ─────────────────────────────────────────────
    print("  Loading SMPLX model ...")
    model = load_smplx_model(smplx_path)
    betas = np.zeros(10, dtype=np.float32)
    smpl_params = dataset_dir / "smpl_output" / "smpl_parameters.npz"
    if smpl_params.exists():
        try:
            d = np.load(smpl_params, allow_pickle=True)
            if "betas" in d and d["betas"].ndim == 2:
                betas = d["betas"][0, :10].astype(np.float32)
        except Exception:
            pass
    v_shaped, j_tpose = precompute_shape(model, betas)

    # ── Load IMU data ────────────────────────────────────────────────
    print(f"  Loading IMU data: {imu_csv}")
    streams = load_imu_csv(imu_csv)
    print(f"    IMUs present: {sorted(streams.keys())}")

    print(f"  Loading calibration: {calib_json}")
    calib = load_imu_calibration(calib_json)
    print(f"    Calibrated joints: {sorted(calib.keys())}")

    # World alignment via AprilTags (restrict to calibration period only)
    tag_poses = load_apriltag_poses(dataset_dir)
    calib_dur = load_calib_duration_sec(dataset_name)
    if calib_dur is not None and calib_dur > 0 and streams:
        first_imu_ns = min(
            int(s["t_ns"][0]) for s in streams.values() if s["t_ns"].size > 0
        )
        calib_end_ns = int(first_imu_ns + calib_dur * 1e9)
        tag_poses = {
            tid: {t: p for t, p in per_tag.items() if int(t) <= calib_end_ns}
            for tid, per_tag in tag_poses.items()
        }
        print(f"    Filtered AprilTag detections to calibration window "
              f"(<= {calib_dur:.1f}s from first IMU sample)")
    tag_rots = extract_apriltag_rotations(tag_poses)
    world_align: Dict[str, np.ndarray] = {}
    if tag_rots:
        world_align = estimate_world_alignment(streams, tag_rots)
        if world_align:
            print(f"    World alignment: {len(world_align)} joints aligned")
        else:
            print("    WARNING: world alignment failed")
    else:
        print("    WARNING: no AprilTag detections in calibration window, "
              "skipping world alignment")

    # ── Load Aria trajectory ─────────────────────────────────────────
    print(f"  Loading Aria trajectory: {aria_traj_path}")
    aria = load_aria_trajectory(aria_traj_path)
    print(f"    Aria: {len(aria.utc_ns)} poses, "
          f"t=[{aria.utc_ns[0]}, {aria.utc_ns[-1]}]")

    # ── Build timeline from frames.csv timestamps ────────────────────
    frames_csv = dataset_dir / "frames.csv"
    frame_ts = []
    if frames_csv.exists():
        with open(frames_csv, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    frame_ts.append(int(row["utc_timestamp_ns"]))
                except Exception:
                    continue
    if not frame_ts:
        print("  SKIP: no frame timestamps found")
        return
    all_timeline_ns = np.array(frame_ts, dtype=np.int64)
    timeline_ns = all_timeline_ns.copy()

    if not include_calibration and calib_dur is not None and len(timeline_ns) > 0:
        calib_end_ns = int(timeline_ns[0] + calib_dur * 1e9)
        trim_idx = int(np.searchsorted(timeline_ns, calib_end_ns))
        timeline_ns = timeline_ns[trim_idx:]
        print(f"  Trimmed {trim_idx} calibration frames ({calib_dur:.1f}s)")

    n_frames = len(timeline_ns)
    print(f"  Timeline: {n_frames} frames")

    # ── Compute IMU-only body pose ───────────────────────────────────
    print("  Computing calibrated bone rotations ...")
    bone_rots = build_calibrated_bone_rotations(streams, calib, world_align, timeline_ns)

    # Align to canonical frame: pelvis identity at t=0
    pelvis_R0 = bone_rots["pelvis"][0].copy()
    canonical = pelvis_R0.T
    for joint in JOINT_NAMES:
        for i in range(n_frames):
            bone_rots[joint][i] = canonical @ bone_rots[joint][i]

    # When debugging a single frame, only compute FK for that frame
    # (vertex LBS is expensive at ~27 fps with mesh).
    if debug_frame is not None:
        fk_indices = [max(0, min(debug_frame, n_frames - 1))]
    else:
        fk_indices = list(range(n_frames))

    print(f"  Running SMPLX forward kinematics ({len(fk_indices)} frames) ...")
    all_joints = np.zeros((n_frames, model.num_joints, 3), dtype=np.float32)
    all_verts = np.zeros((n_frames, model.v_template.shape[0], 3), dtype=np.float32) if compute_verts else None
    for i in tqdm(fk_indices, desc="    FK", unit="frame"):
        global_rots_i = {j: bone_rots[j][i] for j in JOINT_NAMES}
        local_rots = build_local_rots_from_global(global_rots_i)
        joints_i, verts_i = smplx_fk(model, local_rots, v_shaped, j_tpose, compute_verts)
        all_joints[i] = joints_i
        if compute_verts and verts_i is not None:
            all_verts[i] = verts_i

    # ── Get Aria glasses position + orientation at each frame ─────────
    print("  Sampling Aria trajectory at frame timestamps ...")
    aria_idx = np.searchsorted(aria.utc_ns, timeline_ns, side="left")
    aria_idx = np.clip(aria_idx, 0, len(aria.utc_ns) - 1)
    glasses_pos = aria.position[aria_idx].copy()
    glasses_quat = aria.orientation[aria_idx].copy()  # (N, 4) xyzw
    glasses_rot_aria = Rotation.from_quat(glasses_quat).as_matrix().astype(np.float32)

    # Strict simple model: pelvis is a fixed vertical offset from glasses in
    # Aria world (Z-up), no pose-aware FK offset.
    pelvis_offset_aria = np.array([0.0, 0.0, -float(head_to_pelvis_z)], dtype=np.float64)
    pelvis_pos_aria = glasses_pos + pelvis_offset_aria[None, :]
    print(f"    Using fixed glasses->pelvis offset: [0.000, 0.000, {-head_to_pelvis_z:.3f}] m")

    # ── Load OptiTrack bag & align coordinate frames ─────────────────
    cam_pos_opti = None
    cam_rot_opti = None
    box_positions = None
    has_box = False
    object_label = "box"
    glasses_opti = None
    glasses_rot_opti = None
    pelvis_pos_opti_final = None

    if bag_path.exists():
        print(f"  Loading OptiTrack bag: {bag_path}")
        opti_bag = load_optitrack_bag(bag_path)
        print(f"    OptiTrack: {len(opti_bag.body)} frames, "
              f"camera={len(opti_bag.camera)>0}, box={len(opti_bag.box)>0}")

        # Align Aria to OptiTrack
        calib_ts_for_align = _select_calibration_timestamps(all_timeline_ns, calib_dur)
        print(f"  Estimating Aria → OptiTrack alignment ({align_mode}) ...")
        mode = align_mode
        aligned = False
        if mode == "tag_calib":
            try:
                R_align, t_align, info = estimate_aria_to_optitrack_tag_calib(
                    aria=aria,
                    optitrack=opti_bag,
                    tag_poses=tag_poses,
                    calibration_path=calibration_path,
                    calib_timestamps_ns=calib_ts_for_align,
                    head_to_pelvis_z=head_to_pelvis_z,
                )
                print(
                    "    Yaw offset: "
                    f"{info['yaw_deg']:.1f}°, "
                    f"samples: {info['n_samples']} (tag {info['tag_id']}), "
                    f"camera={info['cam_pose_source']} "
                    f"[{t_align[0]:.3f}, {t_align[1]:.3f}, {t_align[2]:.3f}]"
                )
                aligned = True
            except ValueError as e:
                print(f"    WARNING: {e}; falling back to camera_facing_init")
                mode = "camera_facing_init"

        if mode == "camera_facing_init":
            try:
                R_align, t_align, info = estimate_aria_to_optitrack_camera_facing_init(
                    aria=aria,
                    optitrack=opti_bag,
                    calib_timestamps_ns=calib_ts_for_align,
                    head_to_pelvis_z=head_to_pelvis_z,
                    best_facing_n=best_facing_n,
                )
                yaw_all = info.get("yaw_all_frames_deg")
                yaw_all_str = f" (all-frames: {yaw_all:.1f}°)" if yaw_all is not None else ""
                best_n = info.get("n_best_facing", "?")
                best_score = info.get("best_facing_score_mean")
                score_str = f", facing_score={best_score:.3f}" if best_score is not None else ""
                print(
                    "    Yaw offset: "
                    f"{info['yaw_deg']:.1f}°{yaw_all_str}, "
                    f"best {best_n}/{info['n_calib_samples']} frames{score_str}, "
                    f"translation={info['translation_source']} "
                    f"[{t_align[0]:.3f}, {t_align[1]:.3f}, {t_align[2]:.3f}]"
                )
                aligned = True
            except ValueError as e:
                print(f"    WARNING: {e}; falling back to gt_procrustes")
                mode = "gt_procrustes"

        if mode == "gt_procrustes":
            R_align, t_align = estimate_aria_to_optitrack(aria, opti_bag, head_to_pelvis_z)
            yaw_deg = float(np.degrees(np.arctan2(R_align[1, 0], R_align[0, 0])))
            print(f"    Yaw offset: {yaw_deg:.1f}°, Translation: [{t_align[0]:.3f}, {t_align[1]:.3f}, {t_align[2]:.3f}]")
            aligned = True
        elif mode not in {"tag_calib", "camera_facing_init", "gt_procrustes"}:
            raise ValueError(f"Unknown align_mode: {align_mode}")
        if not aligned:
            raise ValueError(f"Failed to estimate alignment with mode: {align_mode}")

        glasses_opti_arr = (R_align @ glasses_pos.T).T + t_align

        # Transform FK output (SMPLX Y-up) → OptiTrack Z-up world.
        R_yup_to_zup = np.array(
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64,
        )
        R_full = R_align @ R_yup_to_zup

        # Transfer Aria coordinates to OptiTrack via p_opti = R_align @ p_aria + t_align.
        # For pelvis we use the strict simple model:
        #   p_pelvis_aria = p_glasses_aria + [0, 0, -head_to_pelvis_z]
        pelvis_pos_opti = (R_align @ pelvis_pos_aria.T).T + t_align

        joints_opti = np.empty_like(all_joints)
        verts_opti = np.empty_like(all_verts) if all_verts is not None else None
        for i in range(n_frames):
            fk_pelvis = all_joints[i, 0].copy()
            centered_j = all_joints[i] - fk_pelvis[None, :]
            joints_opti[i] = (R_full @ centered_j.T).T + pelvis_pos_opti[i]
            if verts_opti is not None:
                centered_v = all_verts[i] - fk_pelvis[None, :]
                verts_opti[i] = (R_full @ centered_v.T).T + pelvis_pos_opti[i]

        # Camera (static tripod — average hand poses, then apply T_cam2hand)
        if opti_bag.camera:
            cam_positions = np.array([c[0] for c in opti_bag.camera])
            cam_quats = np.array([c[1] for c in opti_bag.camera])
            mean_hand_pos = cam_positions.mean(axis=0).astype(np.float32)
            mean_hand_rot = Rotation.from_quat(cam_quats).mean().as_matrix().astype(np.float32)
            T_world_hand = np.eye(4, dtype=np.float32)
            T_world_hand[:3, :3] = mean_hand_rot
            T_world_hand[:3, 3] = mean_hand_pos

            try:
                calib_json = resolve_calibration_json(calibration_path)
                T_cam2hand = load_cam_to_hand(calib_json)
                T_world_cam = T_world_hand @ T_cam2hand
                cam_pos_opti = T_world_cam[:3, 3].astype(np.float32)
                cam_rot_opti = T_world_cam[:3, :3].astype(np.float32)
                print(f"    Camera (with T_cam2hand): [{cam_pos_opti[0]:.3f}, {cam_pos_opti[1]:.3f}, {cam_pos_opti[2]:.3f}]")
            except (FileNotFoundError, KeyError) as e:
                print(f"    WARNING: camera calibration not found ({e}), using raw hand pose")
                cam_pos_opti = mean_hand_pos
                cam_rot_opti = mean_hand_rot

        # Box
        if opti_bag.box:
            has_box = True
            object_label = opti_bag.object_label
            opti_ts_arr = np.array(opti_bag.timestamps, dtype=np.int64)
            box_idx = np.searchsorted(opti_ts_arr, timeline_ns, side="left")
            box_idx = np.clip(box_idx, 0, len(opti_bag.box) - 1)
            box_positions = np.array([opti_bag.box[int(bi)][0] for bi in box_idx])

        glasses_opti = glasses_opti_arr
        # Rotate glasses orientations into OptiTrack frame
        glasses_rot_opti = np.einsum("ij,njk->nik", R_align.astype(np.float32), glasses_rot_aria)
        pelvis_pos_opti_final = pelvis_pos_opti
        pelvis_rot_opti_final = np.einsum(
            "ij,njk->nik", R_full, bone_rots["pelvis"],
        )
    else:
        # No bag: show in Aria world frame (Z-up) directly
        R_yup_to_zup = np.array(
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64,
        )
        pelvis_pos_zup = pelvis_pos_aria
        joints_opti = np.empty_like(all_joints)
        verts_opti = np.empty_like(all_verts) if all_verts is not None else None
        for i in range(n_frames):
            fk_pelvis = all_joints[i, 0].copy()
            centered_j = all_joints[i] - fk_pelvis[None, :]
            joints_opti[i] = (R_yup_to_zup @ centered_j.T).T + pelvis_pos_zup[i]
            if verts_opti is not None:
                centered_v = all_verts[i] - fk_pelvis[None, :]
                verts_opti[i] = (R_yup_to_zup @ centered_v.T).T + pelvis_pos_zup[i]
        glasses_opti = glasses_pos
        glasses_rot_opti = glasses_rot_aria
        pelvis_pos_opti_final = pelvis_pos_zup
        pelvis_rot_opti_final = np.einsum(
            "ij,njk->nik", R_yup_to_zup, bone_rots["pelvis"],
        )

    # ── Load OptiTrack fitted SMPL-X ground truth ──────────────────────
    opti_fit = None
    gt_match_idx = None
    if show_gt:
        print("  Loading OptiTrack fitted SMPL-X ...")
        opti_fit = load_optitrack_smplx_fit(dataset_name)
        if opti_fit is not None:
            gt_match_idx = nearest_indices(opti_fit["timestamps"], timeline_ns)
            print(f"    OptiTrack GT: {opti_fit['joints_zup'].shape[0]} fit frames, "
                  f"{opti_fit['vertices_zup'].shape[1]} vertices")
        else:
            print("    Warning: no fitted SMPL-X found, skipping GT overlay")

    # ── Save body data ────────────────────────────────────────────────
    body_data_path = OUT_DIR / f"{dataset_name}_imu_only.npz"
    print(f"  Saving body data: {body_data_path}")
    save_dict = {
        "joints_opti": joints_opti,
        "timestamps_ns": timeline_ns,
        "pelvis_pos_opti": pelvis_pos_opti_final,
        "pelvis_rot_opti": pelvis_rot_opti_final,
    }
    if verts_opti is not None:
        save_dict["vertices_opti"] = verts_opti
        save_dict["faces"] = model.faces
    np.savez_compressed(body_data_path, **save_dict)

    # ── Face-camera yaw correction ──────────────────────────────────
    # Rotate the body around the pelvis each frame so the face always
    # points toward the matplotlib camera.  Per-frame Rz stored for
    # applying the same correction to the GT overlay at render time.
    face_cam_Rz = None
    if face_camera:
        face_cam_Rz = np.empty((n_frames, 3, 3), dtype=np.float64)
        for i in range(n_frames):
            l_sh = joints_opti[i, 16]
            r_sh = joints_opti[i, 17]
            lr = r_sh - l_sh
            fwd = np.array([-lr[1], lr[0], 0.0])
            body_azim = np.degrees(np.arctan2(fwd[1], fwd[0]))

            cam_azim = -60.0 + i * 0.3
            delta = np.radians((cam_azim + 180.0) - body_azim)
            c, s = np.cos(delta), np.sin(delta)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            face_cam_Rz[i] = Rz

            pivot = joints_opti[i, 0].copy()
            joints_opti[i] = (Rz @ (joints_opti[i] - pivot).T).T + pivot
            if verts_opti is not None:
                verts_opti[i] = (Rz @ (verts_opti[i] - pivot).T).T + pivot
        print("  Applied face-camera yaw correction")

    # ── Compute axis limits ──────────────────────────────────────────
    all_pts = [joints_opti[:, :NUM_BODY_JOINTS, :].reshape(-1, 3)]
    if cam_pos_opti is not None:
        all_pts.append(cam_pos_opti[None, :])
    if glasses_opti is not None:
        all_pts.append(glasses_opti)
    if box_positions is not None:
        all_pts.append(box_positions)
    if opti_fit is not None:
        all_pts.append(opti_fit["joints_zup"].reshape(-1, 3))
    all_pts_cat = np.concatenate(all_pts, axis=0)
    mins = np.percentile(all_pts_cat, 1, axis=0)
    maxs = np.percentile(all_pts_cat, 99, axis=0)
    center = (mins + maxs) / 2
    span = (maxs - mins).max() / 2 * 1.2

    # ── Build rendering context ──────────────────────────────────────
    ctx = {
        "joints_opti": joints_opti,
        "verts_opti": verts_opti,
        "faces": model.faces,
        "mesh_alpha": mesh_alpha,
        "cam_pos": cam_pos_opti,
        "cam_rot": cam_rot_opti,
        "has_box": has_box,
        "box_positions": box_positions,
        "object_label": object_label,
        "glasses_opti": glasses_opti,
        "glasses_rot_opti": glasses_rot_opti,
        "pelvis_pos_opti": pelvis_pos_opti_final,
        "show_gt": show_gt and opti_fit is not None,
        "opti_fit": opti_fit,
        "gt_match_idx": gt_match_idx,
        "face_cam_Rz": face_cam_Rz,
        "timestamps_ns": timeline_ns,
        "center": center,
        "span": span,
        "dataset_name": dataset_name,
        "n_frames": n_frames,
    }

    # ── Debug single frame ───────────────────────────────────────────
    if debug_frame is not None:
        fi = max(0, min(debug_frame, n_frames - 1))
        out_path = OUT_DIR / f"{dataset_name}_imu_only_debug.png"
        global _render_ctx
        _render_ctx = ctx
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        _draw_frame(ax, fi, ctx)
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"  Debug frame {fi} saved: {out_path}")
        return

    # ── Parallel render to MP4 ───────────────────────────────────────
    out_path = OUT_DIR / f"{dataset_name}_imu_only.mp4"
    with tempfile.TemporaryDirectory() as tmpdir:
        args_list = [
            (i, str(Path(tmpdir) / f"{i:06d}.png"))
            for i in range(n_frames)
        ]
        n_workers = max(1, min(mp.cpu_count() // 2, n_frames))
        print(f"  Rendering with {n_workers} workers ...")
        with mp.Pool(n_workers, initializer=_init_render_worker,
                     initargs=(ctx,)) as pool:
            for _ in tqdm(
                pool.imap_unordered(_render_frame_to_png, args_list),
                total=len(args_list),
                desc=f"  Rendering {dataset_name}",
                unit="frame",
            ):
                pass

        encoder = _find_h264_encoder()
        ffmpeg_threads = max(1, mp.cpu_count() // 2)
        print(f"  Encoding to mp4 @ {fps} fps (encoder: {encoder}) ...")
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-start_number", "0",
            "-i", str(Path(tmpdir) / "%06d.png"),
            "-c:v", encoder,
            "-threads", str(ffmpeg_threads),
            "-pix_fmt", "yuv420p",
            "-b:v", "8000k",
            str(out_path),
        ]
        proc = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  ffmpeg stderr:\n{proc.stderr}")
            proc.check_returncode()

    print(f"  Saved video: {out_path}")
    print(f"  Saved body data: {body_data_path}")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║                            CLI                                   ║
# ╚═══════════════════════════════════════════════════════════════════╝

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize IMU-only SMPL body with Aria SLAM localization."
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=[f"dataset{i}" for i in range(1, 12)],
        help="Dataset names (e.g. dataset1 dataset5). Default: all 8.",
    )
    parser.add_argument(
        "--smplx-model", type=Path, default=DEFAULT_SMPLX_MODEL,
        help="Path to SMPLX_NEUTRAL.npz.",
    )
    parser.add_argument(
        "--calibration", type=Path, default=CALIB_DIR,
        help="Path to calibration_result.json or parent directory.",
    )
    parser.add_argument(
        "--mesh-alpha", type=float, default=0.80,
        help="Mesh face transparency (default: 0.80).",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Output video FPS (default: 30).",
    )
    parser.add_argument(
        "--head-to-pelvis-z", type=float, default=0.65,
        help="Naive vertical offset from Aria glasses to pelvis in meters (default: 0.65).",
    )
    parser.add_argument(
        "--no-mesh", action="store_true",
        help="Disable mesh rendering (skeleton only, much faster).",
    )
    parser.add_argument(
        "--no-gt", action="store_true",
        help="Hide OptiTrack fitted SMPL-X ground truth overlay.",
    )
    parser.add_argument(
        "--debug-frame", type=int, default=None,
        help="Render a single frame as PNG for quick inspection.",
    )
    parser.add_argument(
        "--include-calibration", action="store_true",
        help="Include the initial calibration period (excluded by default).",
    )
    parser.add_argument(
        "--face-camera", action="store_true",
        help="Rotate the body each frame so the face always points toward the camera.",
    )
    parser.add_argument(
        "--align-mode",
        choices=["tag_calib", "camera_facing_init", "gt_procrustes"],
        default="gt_procrustes",
        help=(
            "Aria->OptiTrack alignment mode: "
            "gt_procrustes (default; full-sequence pelvis fit), "
            "tag_calib (calibration-only pelvis tag + camera geometry), "
            "or camera_facing_init (calibration facing-camera prior)."
        ),
    )
    parser.add_argument(
        "--best-facing-n", type=int, default=20,
        help=(
            "For camera_facing_init: number of best-facing frames to use for "
            "yaw estimation (default: 20). Selects frames where Aria forward "
            "direction best matches camera-facing direction."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for ds in args.datasets:
        process_dataset(
            ds,
            smplx_path=args.smplx_model,
            calibration_path=args.calibration,
            mesh_alpha=args.mesh_alpha,
            fps=args.fps,
            head_to_pelvis_z=args.head_to_pelvis_z,
            compute_verts=not args.no_mesh,
            show_gt=not args.no_gt,
            debug_frame=args.debug_frame,
            include_calibration=args.include_calibration,
            face_camera=args.face_camera,
            align_mode=args.align_mode,
            best_facing_n=args.best_facing_n,
        )
    print("\nAll done!")
