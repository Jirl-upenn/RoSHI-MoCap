"""Fit SMPL-X body parameters to OptiTrack skeleton data from ROS bags.

Reads OptiTrack motion capture data (51 joints), maps 21 body joints to SMPL-X,
and optimizes global_orient, transl, body pose in 3 Adam stages.

Usage:
    python evaluation/optitrack_gt_data/fit_smplx.py
    python evaluation/optitrack_gt_data/fit_smplx.py --datasets dataset_1
    python evaluation/optitrack_gt_data/fit_smplx.py --device cpu
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import smplx
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

ts = get_typestore(Stores.ROS1_NOETIC)

# OptiTrack Z-up -> SMPL-X Y-up
#   x' = x
#   y' = z
#   z' = -y
R_OPTI_TO_SMPLX = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]  # RoSHI project root
BAG_DIR = Path("evaluation/optitrack_gt_data/ros_bag")
OUT_DIR = Path("evaluation/optitrack_gt_data/smplx_fits")
SMPLX_MODEL_DIR = ROOT / "MHR" / "model"

BAG_NAMES = [
    "dataset_1.bag",
    "dataset_2.bag",
    "dataset_3.bag",
    "dataset_4.bag",
    "dataset_5.bag",
    "dataset_6.bag",
    "dataset_7.bag",
    "dataset_8.bag",
]

# ── OptiTrack → SMPL-X body joint mapping ────────────────────────────────────
# OptiTrack skeleton has 51 joints. We select 21 body joints and map them
# to the corresponding SMPL-X body joint indices (0-21, 22 body joints total).
# SMPL-X spine2 (idx 6) has no direct OptiTrack match.
# NOTE:
#   For these datasets, the exported OptiTrack left/right limb index assignment
#   is mirrored relative to the expected semantic labels. We therefore map the
#   5..8 and 43..46 chains to SMPL-X LEFT joints and 24..27 and 47..50 chains
#   to SMPL-X RIGHT joints.
OPTITRACK_TO_SMPLX = {
    0: 0,    # Hips → pelvis
    1: 3,    # Spine → spine1
    2: 9,    # Chest → spine3
    3: 12,   # Neck → neck
    4: 15,   # Head → head
    5: 13,   # (mirrored stream) Shoulder chain A → left_collar
    6: 16,   # (mirrored stream) Arm chain A → left_shoulder
    7: 18,   # (mirrored stream) ForeArm chain A → left_elbow
    8: 20,   # (mirrored stream) Hand chain A → left_wrist
    24: 14,  # (mirrored stream) Shoulder chain B → right_collar
    25: 17,  # (mirrored stream) Arm chain B → right_shoulder
    26: 19,  # (mirrored stream) ForeArm chain B → right_elbow
    27: 21,  # (mirrored stream) Hand chain B → right_wrist
    43: 1,   # (mirrored stream) Leg chain A → left_hip
    44: 4,   # (mirrored stream) Leg chain A → left_knee
    45: 7,   # (mirrored stream) Leg chain A → left_ankle
    46: 10,  # (mirrored stream) Leg chain A → left_foot
    47: 2,   # (mirrored stream) Leg chain B → right_hip
    48: 5,   # (mirrored stream) Leg chain B → right_knee
    49: 8,   # (mirrored stream) Leg chain B → right_ankle
    50: 11,  # (mirrored stream) Leg chain B → right_foot
}

OPTITRACK_INDICES = sorted(OPTITRACK_TO_SMPLX.keys())
SMPLX_INDICES = [OPTITRACK_TO_SMPLX[i] for i in OPTITRACK_INDICES]

# SMPL-X bone connectivity for the 22 body joints (parent → child)
SMPLX_BODY_BONES = [
    (0, 1), (0, 2), (0, 3),       # pelvis → hips, spine1
    (1, 4), (2, 5), (3, 6),       # hips → knees, spine1 → spine2
    (4, 7), (5, 8), (6, 9),       # knees → ankles, spine2 → spine3
    (7, 10), (8, 11), (9, 12),    # ankles → feet, spine3 → neck
    (12, 13), (12, 14), (12, 15), # neck → collars, head
    (13, 16), (14, 17),           # collars → shoulders
    (16, 18), (17, 19),           # shoulders → elbows
    (18, 20), (19, 21),           # elbows → wrists
]

# Number of SMPL-X body joints (including pelvis)
NUM_SMPLX_BODY_JOINTS = 22


def read_body_from_bag(bag_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read body skeleton positions from a ROS bag.

    Returns:
        positions: (N, 51, 3) array of all joint positions
        timestamps: (N,) array of timestamps in nanoseconds
        root_quats_xyzw: (N, 4) pelvis orientation quaternions in OptiTrack frame
    """
    body_msgs: dict[int, np.ndarray] = {}
    root_quat_msgs: dict[int, np.ndarray] = {}

    with Reader(bag_path) as reader:
        for conn, timestamp, rawdata in reader.messages():
            if "fullbody" not in conn.topic:
                continue
            msg = ts.deserialize_ros1(rawdata, conn.msgtype)
            poses = np.array(
                [[p.position.x, p.position.y, p.position.z] for p in msg.poses],
                dtype=np.float32,
            )
            body_msgs[timestamp] = poses
            root_q = msg.poses[0].orientation
            root_quat_msgs[timestamp] = np.array(
                [root_q.x, root_q.y, root_q.z, root_q.w],
                dtype=np.float32,
            )

    sorted_ts = sorted(body_msgs.keys())
    positions = np.stack([body_msgs[t] for t in sorted_ts], axis=0)
    timestamps = np.array(sorted_ts, dtype=np.int64)
    root_quats_xyzw = np.stack([root_quat_msgs[t] for t in sorted_ts], axis=0)
    return positions, timestamps, root_quats_xyzw


def extract_body_joints(positions: np.ndarray) -> np.ndarray:
    """Extract the 21 mapped body joints from 51 OptiTrack joints.

    Args:
        positions: (N, 51, 3)
    Returns:
        body_joints: (N, 21, 3) in OptiTrack order (sorted by OptiTrack index)
    """
    return positions[:, OPTITRACK_INDICES, :]


def optitrack_to_smplx_coords(joints: np.ndarray) -> np.ndarray:
    """Convert from OptiTrack Z-up to SMPL-X Y-up coordinate system.

    OptiTrack: X-right, Y-forward, Z-up
    SMPL-X:    X-right, Y-up, Z-back

    Transform: x' = x, y' = z, z' = -y
    """
    out = np.empty_like(joints)
    out[..., 0] = joints[..., 0]   # x → x
    out[..., 1] = joints[..., 2]   # z → y (up)
    out[..., 2] = -joints[..., 1]  # -y → z (back)
    return out


def _ensure_quat_continuity(quats: np.ndarray) -> np.ndarray:
    """Flip quaternion signs to maintain temporal continuity.

    Quaternions q and -q represent the same rotation.  When w is near 0
    (rotation angle near pi), scipy's canonical form (w >= 0) can flip the
    sign between consecutive frames, creating artificial discontinuities.
    """
    result = quats.copy()
    for i in range(1, len(result)):
        if np.dot(result[i], result[i - 1]) < 0:
            result[i] = -result[i]
    return result


def _quat_to_rotvec(quats: np.ndarray) -> np.ndarray:
    """Convert quaternions (xyzw) to rotation vectors, allowing angle > pi.

    scipy's ``as_rotvec()`` always clips the angle to [0, pi] by
    canonicalizing w >= 0.  This preserves the quaternion's w sign so
    that angles in (pi, 2*pi) are representable, avoiding the axis-flip
    discontinuity at the pi boundary.  Use after ``_ensure_quat_continuity``.
    """
    xyz = quats[:, :3]
    w = quats[:, 3:]
    sin_half = np.linalg.norm(xyz, axis=1, keepdims=True)
    half_angle = np.arctan2(sin_half, w)
    angle = 2.0 * half_angle
    safe_sin = np.where(sin_half < 1e-8, 1.0, sin_half)
    scale = np.where(sin_half < 1e-8, 2.0, angle / safe_sin)
    return (xyz * scale).astype(np.float64)


def _make_rotvec_continuous(rotvecs: np.ndarray) -> np.ndarray:
    """Ensure temporal continuity of rotation vectors near the pi boundary.

    Converts to quaternions, enforces quaternion-hemisphere continuity,
    then converts back allowing angles beyond pi so that the rotation
    vector sequence is free of axis-flip discontinuities.
    """
    quats = Rotation.from_rotvec(rotvecs).as_quat()
    quats = _ensure_quat_continuity(quats)
    return _quat_to_rotvec(quats)


def optitrack_root_orient_to_smplx_rotvec(root_quats_xyzw: np.ndarray) -> np.ndarray:
    """Convert root orientation quaternions from OptiTrack frame to SMPL-X frame."""
    q = np.asarray(root_quats_xyzw, dtype=np.float64)
    R_opti = Rotation.from_quat(q).as_matrix()
    R_smplx = (
        R_OPTI_TO_SMPLX[None, :, :]
        @ R_opti
        @ R_OPTI_TO_SMPLX.T[None, :, :]
    )
    quats = Rotation.from_matrix(R_smplx).as_quat()
    quats = _ensure_quat_continuity(quats)
    rotvec = _quat_to_rotvec(quats)
    return rotvec.astype(np.float32)


def load_vposer_model(vposer_ckpt: Path, device: torch.device):
    """Load a VPoser model from a checkpoint directory/file."""
    if not vposer_ckpt.exists():
        raise FileNotFoundError(f"VPoser checkpoint not found: {vposer_ckpt}")

    try:
        from human_body_prior.tools.model_loader import load_model
        from human_body_prior.models.vposer_model import VPoser
    except Exception as e:
        raise ImportError(
            "VPoser requested but human_body_prior is not available. "
            "Install it first, e.g. pip install git+https://github.com/nghorbani/human_body_prior.git"
        ) from e

    vp, _ = load_model(
        str(vposer_ckpt),
        model_code=VPoser,
        remove_words_in_model_weights="vp_model.",
        disable_grad=True,
    )
    vp = vp.to(device)
    vp.eval()
    for p in vp.parameters():
        p.requires_grad_(False)
    return vp


def infer_vposer_latent_dim(vposer_model: Any, default: int = 32) -> int:
    """Infer VPoser latent dimensionality from model attributes."""
    for attr in ("latentD", "latent_dim", "num_neurons"):
        if hasattr(vposer_model, attr):
            try:
                val = int(getattr(vposer_model, attr))
                if val > 0:
                    return val
            except Exception:
                pass
    return default


def decode_vposer_body_pose(vposer_model: Any, pose_z: torch.Tensor) -> torch.Tensor:
    """Decode VPoser latent tensor to SMPL-X body pose axis-angle (B, 63)."""
    dec = vposer_model.decode(pose_z, output_type="aa")

    pose = dec
    if isinstance(dec, dict):
        for k in ("pose_body", "pose", "body_pose"):
            if k in dec:
                pose = dec[k]
                break
    elif hasattr(dec, "pose_body"):
        pose = dec.pose_body
    elif hasattr(dec, "pose"):
        pose = dec.pose

    if not isinstance(pose, torch.Tensor):
        raise TypeError(f"Unexpected VPoser decode output type: {type(pose)}")

    # Common VPoser outputs: (B, 21, 3) or (B, 63)
    if pose.ndim == 3:
        pose = pose.reshape(pose.shape[0], -1)
    if pose.ndim != 2:
        raise ValueError(f"Unexpected VPoser decoded pose shape: {tuple(pose.shape)}")
    if pose.shape[1] < 63:
        raise ValueError(f"VPoser decoded pose has <63 dims: {tuple(pose.shape)}")
    if pose.shape[1] > 63:
        pose = pose[:, :63]
    return pose


def fit_smplx(
    target_joints: np.ndarray,
    device: torch.device,
    global_orient_prior: np.ndarray | None = None,
    use_vposer: bool = False,
    vposer_ckpt: Path | None = None,
    lr_stage1: float = 0.05,
    lr_stage2: float = 0.01,
    lr_stage3: float = 0.002,
    iters_stage1: int = 50,
    iters_stage2: int = 100,
    iters_stage3: int = 80,
    w_joints: float = 1.0,
    w_pose_reg: float = 0.001,
    w_vposer: float = 0.001,
    w_orient_prior: float = 0.05,
    w_temporal: float = 0.15,
    w_smooth: float = 0.05,
    w_spine_chain: float = 0.5,
    smooth_sigma: float = 2.0,
    batch_size: int = 128,
) -> dict[str, np.ndarray]:
    """Fit SMPL-X body parameters to target joint positions.

    Args:
        target_joints: (N, 21, 3) target positions in SMPL-X coordinate frame,
                       ordered by OPTITRACK_INDICES.
        device: torch device
    Returns:
        dict with keys: global_orient, transl, body_pose, betas
    """
    N = target_joints.shape[0]
    smplx_idx_tensor = torch.tensor(SMPLX_INDICES, dtype=torch.long, device=device)
    orient_prior_t: torch.Tensor | None = None
    vposer_model = None
    vposer_latent_dim = 0
    if global_orient_prior is not None:
        if global_orient_prior.shape != (N, 3):
            raise ValueError(
                f"global_orient_prior must have shape ({N}, 3), got {global_orient_prior.shape}"
            )
        orient_prior_t = torch.tensor(
            global_orient_prior,
            dtype=torch.float32,
            device=device,
        )
    if use_vposer:
        if vposer_ckpt is None:
            raise ValueError("use_vposer=True but vposer_ckpt is not provided")
        vposer_model = load_vposer_model(vposer_ckpt, device)
        vposer_latent_dim = infer_vposer_latent_dim(vposer_model, default=32)
        print(f"  VPoser enabled: ckpt={vposer_ckpt} latent_dim={vposer_latent_dim}")

    # Load SMPL-X model
    body_model = smplx.create(
        str(SMPLX_MODEL_DIR),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        num_betas=10,
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
    ).to(device)
    body_model.eval()
    for p in body_model.parameters():
        p.requires_grad_(False)

    # Target tensor
    target = torch.tensor(target_joints, dtype=torch.float32, device=device)

    # Initialize parameters
    global_orient = torch.zeros(N, 3, device=device, requires_grad=True)
    transl = torch.zeros(N, 3, device=device, requires_grad=True)
    if use_vposer:
        body_pose = torch.zeros(N, 63, device=device, requires_grad=False)
        pose_latent = torch.zeros(
            N,
            vposer_latent_dim,
            device=device,
            requires_grad=True,
        )
    else:
        body_pose = torch.zeros(N, 63, device=device, requires_grad=True)
        pose_latent = None
    betas = torch.zeros(10, device=device)
    if orient_prior_t is not None:
        with torch.no_grad():
            global_orient.data[:] = orient_prior_t

    # Initialize transl from target pelvis (SMPLX joint 0 = OptiTrack joint 0 = first in our list)
    pelvis_idx_in_target = OPTITRACK_INDICES.index(0)
    with torch.no_grad():
        # Get T-pose pelvis offset
        tpose_out = body_model(
            betas=betas.unsqueeze(0),
            global_orient=torch.zeros(1, 3, device=device),
            body_pose=torch.zeros(1, 63, device=device),
            transl=torch.zeros(1, 3, device=device),
            jaw_pose=torch.zeros(1, 3, device=device),
            leye_pose=torch.zeros(1, 3, device=device),
            reye_pose=torch.zeros(1, 3, device=device),
            left_hand_pose=torch.zeros(1, 45, device=device),
            right_hand_pose=torch.zeros(1, 45, device=device),
            expression=torch.zeros(1, 10, device=device),
        )
        tpose_pelvis = tpose_out.joints[0, 0, :3]
        transl.data[:] = target[:, pelvis_idx_in_target, :] - tpose_pelvis.unsqueeze(0)

    # Pre-allocate zero tensors for unused SMPLX params
    zero_jaw = torch.zeros(1, 3, device=device)
    zero_eye = torch.zeros(1, 3, device=device)
    zero_hand = torch.zeros(1, 45, device=device)
    zero_expr = torch.zeros(1, 10, device=device)

    def compute_loss(
        batch_orient,
        batch_transl,
        batch_pose,
        batch_pose_latent,
        batch_target,
        batch_orient_prior,
        w_temp, w_sm,
    ):
        if use_vposer:
            if vposer_model is None or batch_pose_latent is None:
                raise RuntimeError("VPoser mode active but model/latent batch is unavailable")
            body_pose_eff = decode_vposer_body_pose(vposer_model, batch_pose_latent)
            loss_vposer = w_vposer * torch.mean(batch_pose_latent ** 2)
        else:
            if batch_pose is None:
                raise RuntimeError("Direct pose mode active but body pose batch is unavailable")
            body_pose_eff = batch_pose
            loss_vposer = torch.tensor(0.0, device=device)

        B = batch_orient.shape[0]
        output = body_model(
            betas=betas.unsqueeze(0).expand(B, -1),
            global_orient=batch_orient,
            body_pose=body_pose_eff,
            transl=batch_transl,
            jaw_pose=zero_jaw.expand(B, -1),
            leye_pose=zero_eye.expand(B, -1),
            reye_pose=zero_eye.expand(B, -1),
            left_hand_pose=zero_hand.expand(B, -1),
            right_hand_pose=zero_hand.expand(B, -1),
            expression=zero_expr.expand(B, -1),
        )
        pred_joints = output.joints[:, :NUM_SMPLX_BODY_JOINTS, :]
        pred_mapped = pred_joints[:, smplx_idx_tensor, :]

        # Joint position loss
        loss_joints = w_joints * torch.mean((pred_mapped - batch_target) ** 2)

        # Pose regularization
        loss_reg = w_pose_reg * torch.mean(body_pose_eff ** 2)

        # Root orientation prior from OptiTrack pelvis orientation.
        loss_orient = torch.tensor(0.0, device=device)
        if batch_orient_prior is not None:
            loss_orient = w_orient_prior * torch.mean((batch_orient - batch_orient_prior) ** 2)

        # Spine2 (SMPL-X joint 6) has no OptiTrack target.  Constrain it to
        # lie between spine1 and spine3 targets, and smooth the local rotations
        # across the spine chain to prevent a single joint from collapsing.
        loss_spine = torch.tensor(0.0, device=device)
        if w_spine_chain > 0:
            # Interpolated position target for spine2
            # batch_target[:, 1] = spine1 (OptiTrack 1 → SMPLX 3)
            # batch_target[:, 2] = spine3 (OptiTrack 2 → SMPLX 9)
            spine2_target = (batch_target[:, 1, :] + batch_target[:, 2, :]) / 2.0
            spine2_pred = pred_joints[:, 6, :]
            loss_spine = w_spine_chain * torch.mean((spine2_pred - spine2_target) ** 2)
            # Smooth consecutive spine local rotations (spine1, spine2, spine3)
            spine_rots = torch.stack([
                body_pose_eff[:, 6:9],    # spine1 (joint 3)
                body_pose_eff[:, 15:18],  # spine2 (joint 6)
                body_pose_eff[:, 24:27],  # spine3 (joint 9)
            ], dim=1)
            loss_spine = loss_spine + (w_spine_chain * 0.1) * torch.mean(
                (spine_rots[:, 1:] - spine_rots[:, :-1]) ** 2
            )

        # Temporal smoothness on parameters
        loss_temporal = torch.tensor(0.0, device=device)
        loss_smooth = torch.tensor(0.0, device=device)
        if B > 1:
            loss_temporal = w_temp * torch.mean(
                (body_pose_eff[1:] - body_pose_eff[:-1]) ** 2
            )
            loss_smooth = w_sm * torch.mean(
                (pred_mapped[1:] - pred_mapped[:-1]) ** 2
            )

        total = loss_joints + loss_reg + loss_vposer + loss_orient + loss_spine + loss_temporal + loss_smooth
        return total, loss_joints.item(), loss_reg.item(), loss_vposer.item(), loss_orient.item(), loss_spine.item()

    def run_stage(
        params: list[torch.Tensor],
        lr: float,
        n_iters: int,
        stage_name: str,
        w_temp: float,
        w_sm: float,
    ):
        optimizer = torch.optim.Adam(params, lr=lr)

        for it in range(n_iters):
            total_loss = 0.0
            total_jloss = 0.0
            total_rloss = 0.0
            total_vploss = 0.0
            total_oloss = 0.0
            total_sloss = 0.0
            n_batches = 0

            # Process in batches for memory efficiency
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                b_orient = global_orient[start:end]
                b_transl = transl[start:end]
                b_pose = body_pose[start:end] if pose_latent is None else None
                b_pose_latent = pose_latent[start:end] if pose_latent is not None else None
                b_target = target[start:end]
                b_orient_prior = (
                    orient_prior_t[start:end] if orient_prior_t is not None else None
                )

                optimizer.zero_grad()
                loss, jl, rl, vpl, ol, sl = compute_loss(
                    b_orient,
                    b_transl,
                    b_pose,
                    b_pose_latent,
                    b_target,
                    b_orient_prior,
                    w_temp,
                    w_sm,
                )
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_jloss += jl
                total_rloss += rl
                total_vploss += vpl
                total_oloss += ol
                total_sloss += sl
                n_batches += 1

            # Global temporal consistency across batch boundaries
            if w_temp > 0 and N > 1:
                optimizer.zero_grad()
                loss_global = w_temp * (
                    torch.mean((global_orient[1:] - global_orient[:-1]) ** 2)
                    + torch.mean((transl[1:] - transl[:-1]) ** 2)
                )
                if pose_latent is not None:
                    loss_global = loss_global + w_temp * torch.mean(
                        (pose_latent[1:] - pose_latent[:-1]) ** 2
                    )
                else:
                    loss_global = loss_global + w_temp * torch.mean(
                        (body_pose[1:] - body_pose[:-1]) ** 2
                    )
                loss_global.backward()
                optimizer.step()

            if (it + 1) % 10 == 0 or it == 0:
                print(
                    f"  [{stage_name}] iter {it+1:3d}/{n_iters} | "
                    f"loss={total_loss/n_batches:.6f} | "
                    f"joints={total_jloss/n_batches:.6f} | "
                    f"reg={total_rloss/n_batches:.6f} | "
                    f"vposer={total_vploss/n_batches:.6f} | "
                    f"orient={total_oloss/n_batches:.6f} | "
                    f"spine={total_sloss/n_batches:.6f}",
                    flush=True,
                )

    print(f"  Fitting {N} frames on {device}...", flush=True)

    # Stage 1: Global alignment (orient + translation only)
    print("  Stage 1: Global alignment", flush=True)
    run_stage([global_orient, transl], lr_stage1, iters_stage1, "S1", w_temp=0.0, w_sm=0.0)

    # Stage 2: Body pose
    print("  Stage 2: Body pose optimization", flush=True)
    run_stage(
        [global_orient, transl] + ([pose_latent] if pose_latent is not None else [body_pose]),
        lr_stage2, iters_stage2, "S2",
        w_temp=w_temporal, w_sm=w_smooth,
    )

    # Stage 3: Refinement with stronger temporal smoothing
    print("  Stage 3: Refinement", flush=True)
    run_stage(
        [global_orient, transl] + ([pose_latent] if pose_latent is not None else [body_pose]),
        lr_stage3, iters_stage3, "S3",
        w_temp=w_temporal * 3, w_sm=w_smooth * 3,
    )

    # Collect results
    with torch.no_grad():
        if use_vposer:
            if vposer_model is None or pose_latent is None:
                raise RuntimeError("VPoser mode active but final decode cannot run")
            out_pose = []
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                out_pose.append(
                    decode_vposer_body_pose(vposer_model, pose_latent[start:end]).cpu().numpy()
                )
            body_pose_np = np.concatenate(out_pose, axis=0)
            pose_latent_np = pose_latent.detach().cpu().numpy()
        else:
            body_pose_np = body_pose.detach().cpu().numpy()
            pose_latent_np = np.zeros((N, 0), dtype=np.float32)

        result = {
            "global_orient": global_orient.detach().cpu().numpy(),
            "transl": transl.detach().cpu().numpy(),
            "body_pose": body_pose_np,
            "betas": betas.detach().cpu().numpy(),
            "pose_latent": pose_latent_np,
        }

    if N > 1:
        print("  Enforcing global_orient continuity", flush=True)
        q_go = Rotation.from_rotvec(result["global_orient"]).as_quat()
        q_go = _ensure_quat_continuity(q_go)
        if smooth_sigma > 0:
            print(
                f"  Post-hoc Gaussian smoothing (sigma={smooth_sigma} frames)",
                flush=True,
            )
            q_go = gaussian_filter1d(q_go, sigma=smooth_sigma, axis=0)
            q_go /= np.linalg.norm(q_go, axis=1, keepdims=True)
            for key in ("transl", "body_pose"):
                result[key] = gaussian_filter1d(
                    result[key], sigma=smooth_sigma, axis=0,
                ).astype(np.float32)
        result["global_orient"] = _quat_to_rotvec(q_go).astype(np.float32)

    return result


def process_dataset(
    bag_name: str,
    device: torch.device,
    subsample: int = 3,
    use_orient_prior: bool = True,
    orient_prior_weight: float = 0.05,
    use_vposer: bool = False,
    vposer_ckpt: Path | None = None,
    vposer_weight: float = 0.001,
    smooth_sigma: float = 2.0,
):
    """Process a single dataset: load, fit, and save."""
    bag_path = BAG_DIR / bag_name
    dataset_name = bag_name.replace(".bag", "")
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}")
    print(f"{'='*60}")

    # Read data
    print("  Reading ROS bag...")
    positions, timestamps, root_quats = read_body_from_bag(bag_path)
    n_total = positions.shape[0]
    print(f"  Total frames: {n_total}, joints per frame: {positions.shape[1]}")

    # Subsample to ~30fps
    positions = positions[::subsample]
    timestamps = timestamps[::subsample]
    root_quats = root_quats[::subsample]
    n_sub = positions.shape[0]
    print(f"  After {subsample}x subsample: {n_sub} frames")

    # Extract body joints and convert coordinates
    body_joints = extract_body_joints(positions)
    body_joints = optitrack_to_smplx_coords(body_joints)
    print(f"  Body joints shape: {body_joints.shape}")

    global_orient_prior = None
    if use_orient_prior:
        global_orient_prior = optitrack_root_orient_to_smplx_rotvec(root_quats)
        print(
            "  Root orientation prior enabled: "
            f"shape={global_orient_prior.shape}, weight={orient_prior_weight}"
        )
    else:
        print("  Root orientation prior disabled.")

    # Fit
    t0 = time.time()
    result = fit_smplx(
        body_joints,
        device,
        global_orient_prior=global_orient_prior,
        use_vposer=use_vposer,
        vposer_ckpt=vposer_ckpt,
        w_vposer=vposer_weight,
        w_orient_prior=orient_prior_weight,
        smooth_sigma=smooth_sigma,
    )
    elapsed = time.time() - t0
    print(f"  Fitting took {elapsed:.1f}s")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{dataset_name}.npz"
    np.savez(
        out_path,
        **result,
        timestamps=timestamps,
        target_joints=body_joints,
        root_orient_prior=(
            global_orient_prior
            if global_orient_prior is not None
            else np.zeros((body_joints.shape[0], 3), dtype=np.float32)
        ),
        optitrack_indices=np.array(OPTITRACK_INDICES),
        smplx_indices=np.array(SMPLX_INDICES),
        subsample=subsample,
        use_orient_prior=np.array([1 if use_orient_prior else 0], dtype=np.int32),
        orient_prior_weight=np.array([orient_prior_weight], dtype=np.float32),
        use_vposer=np.array([1 if use_vposer else 0], dtype=np.int32),
        vposer_ckpt=np.array([str(vposer_ckpt) if vposer_ckpt is not None else ""]),
        vposer_weight=np.array([vposer_weight], dtype=np.float32),
    )
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fit SMPL-X to OptiTrack skeleton data")
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Dataset names (without .bag), e.g. dataset_1 dataset_2. Default: all.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available)",
    )
    parser.add_argument(
        "--subsample", type=int, default=3,
        help="Subsample factor (default: 3, ~100Hz → ~33Hz)",
    )
    parser.add_argument(
        "--no-orient-prior",
        action="store_true",
        help="Disable root orientation prior from OptiTrack pelvis orientation.",
    )
    parser.add_argument(
        "--orient-prior-weight",
        type=float,
        default=0.05,
        help="Weight for root orientation prior (default: 0.05).",
    )
    parser.add_argument(
        "--use-vposer",
        action="store_true",
        help="Enable VPoser latent-space body-pose optimization.",
    )
    parser.add_argument(
        "--vposer-ckpt",
        type=Path,
        default=None,
        help="Path to VPoser checkpoint directory/file (required with --use-vposer).",
    )
    parser.add_argument(
        "--vposer-weight",
        type=float,
        default=0.001,
        help="Weight for VPoser latent L2 prior (default: 0.001).",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=2.0,
        help="Post-hoc Gaussian smoothing sigma in frames (default: 2.0). 0 to disable.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    if args.use_vposer and args.vposer_ckpt is None:
        raise ValueError("--use-vposer requires --vposer-ckpt")

    if args.datasets:
        bags = [f"{d}.bag" for d in args.datasets]
    else:
        bags = BAG_NAMES

    for bag_name in bags:
        process_dataset(
            bag_name,
            device,
            args.subsample,
            use_orient_prior=not args.no_orient_prior,
            orient_prior_weight=float(args.orient_prior_weight),
            use_vposer=bool(args.use_vposer),
            vposer_ckpt=args.vposer_ckpt.resolve() if args.vposer_ckpt is not None else None,
            vposer_weight=float(args.vposer_weight),
            smooth_sigma=float(args.smooth_sigma),
        )

    print("\nAll done!")


if __name__ == "__main__":
    main()
