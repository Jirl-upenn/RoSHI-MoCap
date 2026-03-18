"""Visualize EgoAllo / IMU-TTO SMPL-H body in the OptiTrack world frame.

Loads inference outputs (.npz), runs SMPL-H forward kinematics to
obtain joints and mesh in the Aria world frame, then aligns to OptiTrack
world (Z-up) via Procrustes on the Aria+OptiTrack pelvis trajectories.

Supports two methods stored in received_recordings/<dataset>/egoallo_outputs/:
  - aria_wrist_only_*  → EgoAllo (baseline)
  - imu_tto_*          → IMU-TTO (ours)

Optionally overlays OptiTrack fitted SMPL-X ground truth for comparison.

Usage:
    python evaluation/visualize_egoallo.py --datasets dataset1
    python evaluation/visualize_egoallo.py --datasets dataset1 --methods egoallo imu_tto
    python evaluation/visualize_egoallo.py --no-gt --no-mesh
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

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
    RECORDINGS_DIR,
    BODY_BONES,
    NUM_BODY_JOINTS,
    load_calib_duration_sec,
    load_cam_to_hand,
    load_optitrack_bag,
    load_optitrack_smplx_fit,
    nearest_indices,
    resolve_calibration_json,
    tracking_ns_to_utc_ns,
)

DEFAULT_SMPLH_MODEL = Path("model/smplh/neutral/model.npz")

VIS_METHODS = {
    "egoallo": {
        "npz_prefix": "aria_wrist_only_",
        "out_dir": Path("evaluation/egoallo"),
        "label": "EgoAllo",
        "joint_color": "crimson",
        "bone_color": "darkred",
        "mesh_color": "salmon",
        "file_tag": "egoallo",
    },
    "imu_tto": {
        "npz_prefix": "imu_tto_",
        "out_dir": Path("evaluation/imu_tto"),
        "label": "IMU-TTO (Ours)",
        "joint_color": "limegreen",
        "bone_color": "darkgreen",
        "mesh_color": "lightgreen",
        "file_tag": "imu_tto",
    },
}

for _cfg in VIS_METHODS.values():
    _cfg["out_dir"].mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# EgoAllo output loading + SMPL-H forward kinematics
# ---------------------------------------------------------------------------

def load_egoallo_output(npz_path: Path) -> dict:
    """Load an EgoAllo output NPZ and return its contents."""
    data = np.load(npz_path)
    required = ["Ts_world_root", "body_quats", "betas", "timestamps_ns"]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {npz_path}")
    return dict(data)


def run_smplh_fk(
    npz_data: dict,
    smplh_model_path: Path,
    sample_index: int = 0,
    compute_verts: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Run SMPL-H FK on egoallo output, return joints + vertices in Aria world.

    Returns:
        joints_world: (T, 22, 3) body joint positions in Aria world frame
        verts_world: (T, V, 3) vertex positions or None
        faces: (F, 3) mesh face indices
        timestamps_ns: (T,) int64
    """
    import torch
    _egoallo_path = str(Path(__file__).resolve().parents[1] / "egoallo")
    if _egoallo_path not in sys.path:
        sys.path.insert(0, _egoallo_path)
    from egoallo import fncsmpl
    from egoallo.network import EgoDenoiseTraj
    from egoallo.transforms import SE3, SO3

    device = torch.device("cpu")
    body_model = fncsmpl.SmplhModel.load(smplh_model_path).to(device)

    betas_np = npz_data["betas"][sample_index]           # (T, 16)
    body_quats_np = npz_data["body_quats"][sample_index]  # (T, 21, 4)
    Ts_world_root_np = npz_data["Ts_world_root"][sample_index]  # (T, 7)
    timestamps_ns = npz_data["timestamps_ns"]

    left_hand_quats_np = npz_data.get("left_hand_quats")
    right_hand_quats_np = npz_data.get("right_hand_quats")
    if left_hand_quats_np is not None:
        left_hand_quats_np = left_hand_quats_np[sample_index]
    if right_hand_quats_np is not None:
        right_hand_quats_np = right_hand_quats_np[sample_index]

    T = betas_np.shape[0]
    batch_size = 256
    all_joints, all_verts = [], []

    with torch.no_grad():
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            b_betas = torch.from_numpy(betas_np[start:end]).to(device)
            b_body = torch.from_numpy(body_quats_np[start:end]).to(device)
            b_root = torch.from_numpy(Ts_world_root_np[start:end]).to(device)
            b_lh = torch.from_numpy(left_hand_quats_np[start:end]).to(device) if left_hand_quats_np is not None else None
            b_rh = torch.from_numpy(right_hand_quats_np[start:end]).to(device) if right_hand_quats_np is not None else None

            shaped = body_model.with_shape(b_betas)
            posed = shaped.with_pose_decomposed(
                T_world_root=b_root,
                body_quats=b_body,
                left_hand_quats=b_lh,
                right_hand_quats=b_rh,
            )

            root_pos = SE3(posed.T_world_root).translation().cpu().numpy()
            joint_pos = SE3(posed.Ts_world_joint[..., :21, :]).translation().cpu().numpy()
            batch_joints = np.concatenate([root_pos[:, None, :], joint_pos], axis=1)
            all_joints.append(batch_joints)

            if compute_verts:
                mesh = posed.lbs()
                all_verts.append(mesh.verts.cpu().numpy())

    joints_world = np.concatenate(all_joints, axis=0)
    verts_world = np.concatenate(all_verts, axis=0) if compute_verts else None

    faces = body_model.faces.detach().numpy().astype(np.int32)
    return joints_world, verts_world, faces, timestamps_ns




# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_render_ctx: dict = {}


def _init_render_worker(ctx: dict):
    global _render_ctx
    _render_ctx = ctx


def _draw_frame(ax, frame_idx: int, ctx: dict):
    ax.cla()
    joints = ctx["joints_opti"][frame_idx]
    verts = ctx["verts_opti"][frame_idx] if ctx["verts_opti"] is not None else None
    faces = ctx["faces"]

    style = ctx["style"]
    if verts is not None:
        face_step = max(1, len(faces) // 3000)
        tris = verts[faces[::face_step]]
        ax.add_collection3d(Poly3DCollection(
            tris, alpha=ctx["mesh_alpha"], facecolor=style["mesh_color"], edgecolor="none",
        ))

    ax.scatter(
        joints[:NUM_BODY_JOINTS, 0],
        joints[:NUM_BODY_JOINTS, 1],
        joints[:NUM_BODY_JOINTS, 2],
        c=style["joint_color"], s=25, alpha=0.9, marker="o",
        label=style["label"], zorder=4,
    )
    for j1, j2 in BODY_BONES:
        if j1 < joints.shape[0] and j2 < joints.shape[0]:
            ax.plot(
                [joints[j1, 0], joints[j2, 0]],
                [joints[j1, 1], joints[j2, 1]],
                [joints[j1, 2], joints[j2, 2]],
                c=style["bone_color"], linewidth=1.2, alpha=0.7,
            )

    if ctx["show_gt"] and ctx["opti_fit"] is not None:
        opti_fit = ctx["opti_fit"]
        gt_idx = ctx["gt_match_idx"][frame_idx]
        gt_joints = opti_fit["joints_zup"][gt_idx]
        gt_verts = opti_fit["vertices_zup"][gt_idx]
        gt_faces = opti_fit["faces"]
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

    if ctx["cam_pos"] is not None:
        cam_pos = ctx["cam_pos"]
        cam_rot = ctx["cam_rot"]
        ax.scatter(*cam_pos, c="red", s=60, marker="^", label="Camera", zorder=5)
        ax_len = 0.15
        cam_axes = cam_rot * ax_len
        for i, c in enumerate(["red", "green", "blue"]):
            ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                      cam_axes[0, i], cam_axes[1, i], cam_axes[2, i],
                      color=c, linewidth=1.5, arrow_length_ratio=0.15)
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

    if ctx["has_box"]:
        box_pos = ctx["box_positions"][frame_idx]
        ax.scatter(*box_pos, c="orange", s=120, marker="s",
                   label=f"Object ({ctx['object_label']})", zorder=5)

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
        f"{ctx['dataset_name']} {style['label']} | t={time_s:.1f}s | "
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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_dataset(
    dataset_name: str,
    *,
    method: str = "egoallo",
    smplh_model_path: Path = DEFAULT_SMPLH_MODEL,
    calibration_path: Path = CALIB_DIR,
    mesh_alpha: float = 0.20,
    fps: int = 30,
    compute_verts: bool = True,
    show_gt: bool = True,
    debug_frame: Optional[int] = None,
    include_calibration: bool = False,
):
    style = VIS_METHODS[method]
    out_dir = style["out_dir"]
    npz_prefix = style["npz_prefix"]
    file_tag = style["file_tag"]

    aria_name = DATASET_TO_ARIA.get(dataset_name)
    if aria_name is None:
        print(f"  SKIP: unknown dataset {dataset_name}")
        return

    egoallo_dir = RECORDINGS_DIR / dataset_name / "egoallo_outputs"
    if not egoallo_dir.exists():
        print(f"  SKIP: no egoallo_outputs at {egoallo_dir}")
        return

    npz_files = sorted(egoallo_dir.glob(f"{npz_prefix}*.npz"))
    if not npz_files:
        print(f"  SKIP: no {npz_prefix}*.npz files in {egoallo_dir}")
        return
    npz_path = max(npz_files, key=lambda f: f.stat().st_size)

    bag_name = re.sub(r"dataset(\d)", r"dataset_\1", dataset_name) + ".bag"
    bag_path = BAG_DIR / bag_name
    aria_traj_path = ARIA_DIR / f"mps_{aria_name}_vrs" / "slam" / "closed_loop_trajectory.csv"

    print(f"\n{'='*60}")
    print(f"Processing {dataset_name} [{style['label']}]")
    print(f"  Input: {npz_path}")
    print(f"{'='*60}")

    # -- Load egoallo output & run FK ---
    print("  Loading EgoAllo output & running SMPL-H FK ...")
    npz_data = load_egoallo_output(npz_path)
    joints_world, verts_world, faces, timestamps_ns = run_smplh_fk(
        npz_data, smplh_model_path, compute_verts=compute_verts,
    )
    n_frames = joints_world.shape[0]
    print(f"    {n_frames} frames, {joints_world.shape[1]} joints")

    # Egoallo joints are in Aria world frame (Z-up, gravity-aligned).
    # We need to align to OptiTrack Z-up via Procrustes.

    # -- Load Aria trajectory for alignment ---
    cam_pos_opti = None
    cam_rot_opti = None
    box_positions = None
    has_box = False
    object_label = "box"
    joints_opti = joints_world.copy()
    verts_opti = verts_world.copy() if verts_world is not None else None

    # Extract root rotation from Ts_world_root (wxyz_xyz SE3)
    Ts_root = npz_data["Ts_world_root"][0]  # (T, 7)
    root_quat_wxyz = Ts_root[:n_frames, :4]
    root_quat_xyzw = np.concatenate(
        [root_quat_wxyz[:, 1:], root_quat_wxyz[:, :1]], axis=1,
    )
    pelvis_rot_aria = Rotation.from_quat(root_quat_xyzw).as_matrix().astype(np.float64)

    if aria_traj_path.exists() and bag_path.exists():
        print(f"  Loading OptiTrack bag: {bag_path}")
        opti_bag = load_optitrack_bag(bag_path)

        # Align EgoAllo pelvis directly to OptiTrack pelvis (not via Aria head)
        print("  Estimating EgoAllo -> OptiTrack alignment ...")
        ego_ts_utc_align = tracking_ns_to_utc_ns(timestamps_ns, aria_traj_path)
        ego_pelvis_aria = joints_world[:, 0, :].astype(np.float64)
        opti_ts = np.array(opti_bag.timestamps, dtype=np.int64)
        opti_pelvis = np.array([b[0] for b in opti_bag.body], dtype=np.float64)
        # Match EgoAllo frames to OptiTrack timestamps
        ego_idx_for_opti = nearest_indices(ego_ts_utc_align, opti_ts)
        ego_for_align = ego_pelvis_aria[ego_idx_for_opti]
        # XY Procrustes (yaw only, both frames are Z-up gravity-aligned)
        cA = ego_for_align.mean(axis=0)
        cB = opti_pelvis.mean(axis=0)
        H = (ego_for_align - cA)[:, :2].T @ (opti_pelvis - cB)[:, :2]
        U, _, Vt = np.linalg.svd(H)
        R2 = Vt.T @ U.T
        if np.linalg.det(R2) < 0:
            Vt[-1] *= -1
            R2 = Vt.T @ U.T
        R_align = np.eye(3, dtype=np.float64)
        R_align[:2, :2] = R2
        t_align = cB - R_align @ cA
        yaw_deg = float(np.degrees(np.arctan2(R_align[1, 0], R_align[0, 0])))
        print(f"    Yaw offset: {yaw_deg:.1f}deg, Translation: "
              f"[{t_align[0]:.3f}, {t_align[1]:.3f}, {t_align[2]:.3f}]")

        for i in range(n_frames):
            joints_opti[i] = (R_align @ joints_world[i].T).T + t_align
            if verts_opti is not None and verts_world is not None:
                verts_opti[i] = (R_align @ verts_world[i].T).T + t_align

        pelvis_rot_opti = np.einsum("ij,njk->nik", R_align, pelvis_rot_aria)

        # Camera (static tripod)
        if opti_bag.camera:
            cam_positions = np.array([c[0] for c in opti_bag.camera])
            cam_quats = np.array([c[1] for c in opti_bag.camera])
            mean_hand_pos = cam_positions.mean(axis=0).astype(np.float32)
            mean_hand_rot = Rotation.from_quat(cam_quats).mean().as_matrix().astype(np.float32)
            T_world_hand = np.eye(4, dtype=np.float32)
            T_world_hand[:3, :3] = mean_hand_rot
            T_world_hand[:3, 3] = mean_hand_pos
            try:
                cj = resolve_calibration_json(calibration_path)
                T_cam2hand = load_cam_to_hand(cj)
                T_world_cam = T_world_hand @ T_cam2hand
                cam_pos_opti = T_world_cam[:3, 3].astype(np.float32)
                cam_rot_opti = T_world_cam[:3, :3].astype(np.float32)
            except (FileNotFoundError, KeyError):
                cam_pos_opti = mean_hand_pos
                cam_rot_opti = mean_hand_rot

        # Box – flag only; positions computed after calibration trimming
        if opti_bag.box:
            has_box = True
            object_label = opti_bag.object_label
            _opti_box_data = opti_bag.box
            _opti_box_ts = np.array(opti_bag.timestamps, dtype=np.int64)
    else:
        print("  WARNING: Aria trajectory or OptiTrack bag not found, "
              "showing in Aria world frame directly")
        pelvis_rot_opti = pelvis_rot_aria

    # -- Convert EgoAllo tracking timestamps to UTC for GT matching ---
    timestamps_utc_ns = timestamps_ns
    if aria_traj_path.exists():
        timestamps_utc_ns = tracking_ns_to_utc_ns(timestamps_ns, aria_traj_path)
        print(f"  Converted tracking -> UTC timestamps "
              f"(range: {(timestamps_utc_ns[-1] - timestamps_utc_ns[0])/1e9:.1f}s)")

    # -- Strip calibration period ---
    if not include_calibration:
        calib_dur = load_calib_duration_sec(dataset_name)
        if calib_dur is not None and len(timestamps_utc_ns) > 0:
            calib_end_ns = int(timestamps_utc_ns[0] + calib_dur * 1e9)
            trim_idx = int(np.searchsorted(timestamps_utc_ns, calib_end_ns))
            timestamps_utc_ns = timestamps_utc_ns[trim_idx:]
            timestamps_ns = timestamps_ns[trim_idx:]
            joints_world = joints_world[trim_idx:]
            joints_opti = joints_opti[trim_idx:]
            if verts_world is not None:
                verts_world = verts_world[trim_idx:]
            if verts_opti is not None:
                verts_opti = verts_opti[trim_idx:]
            pelvis_rot_opti = pelvis_rot_opti[trim_idx:]
            n_frames = joints_world.shape[0]
            print(f"  Trimmed {trim_idx} calibration frames ({calib_dur:.1f}s)")

    # -- Compute box positions using UTC timestamps (after calibration trim) ---
    if has_box:
        box_idx = np.searchsorted(_opti_box_ts, timestamps_utc_ns, side="left")
        box_idx = np.clip(box_idx, 0, len(_opti_box_data) - 1)
        box_positions = np.array([_opti_box_data[int(bi)][0] for bi in box_idx])

    # -- OptiTrack fitted SMPL-X ground truth ---
    opti_fit = None
    gt_match_idx = None
    if show_gt:
        print("  Loading OptiTrack fitted SMPL-X ...")
        opti_fit = load_optitrack_smplx_fit(dataset_name)
        if opti_fit is not None:
            gt_match_idx = nearest_indices(opti_fit["timestamps"], timestamps_utc_ns)
            print(f"    OptiTrack GT: {opti_fit['joints_zup'].shape[0]} fit frames")
        else:
            print("    Warning: no fitted SMPL-X found, skipping GT overlay")

    # -- Save body data ---
    body_data_path = out_dir / f"{dataset_name}_{file_tag}.npz"
    print(f"  Saving body data: {body_data_path}")
    save_dict = {
        "joints_opti": joints_opti,
        "timestamps_ns": timestamps_utc_ns,
        "pelvis_pos_opti": joints_opti[:, 0, :],
        "pelvis_rot_opti": pelvis_rot_opti,
    }
    if verts_opti is not None:
        save_dict["vertices_opti"] = verts_opti
        save_dict["faces"] = faces
    np.savez_compressed(body_data_path, **save_dict)

    # -- Axis limits ---
    all_pts = [joints_opti[:, :NUM_BODY_JOINTS, :].reshape(-1, 3)]
    if cam_pos_opti is not None:
        all_pts.append(cam_pos_opti[None, :])
    if box_positions is not None:
        all_pts.append(box_positions)
    if opti_fit is not None:
        all_pts.append(opti_fit["joints_zup"].reshape(-1, 3))
    all_pts_cat = np.concatenate(all_pts, axis=0)
    mins = np.percentile(all_pts_cat, 1, axis=0)
    maxs = np.percentile(all_pts_cat, 99, axis=0)
    center = (mins + maxs) / 2
    span = (maxs - mins).max() / 2 * 1.2

    ctx = {
        "joints_opti": joints_opti,
        "verts_opti": verts_opti,
        "faces": faces,
        "mesh_alpha": mesh_alpha,
        "cam_pos": cam_pos_opti,
        "cam_rot": cam_rot_opti,
        "has_box": has_box,
        "box_positions": box_positions,
        "object_label": object_label,
        "show_gt": show_gt and opti_fit is not None,
        "opti_fit": opti_fit,
        "gt_match_idx": gt_match_idx,
        "timestamps_ns": timestamps_ns,
        "center": center,
        "span": span,
        "dataset_name": dataset_name,
        "n_frames": n_frames,
        "style": style,
    }

    # Debug single frame
    if debug_frame is not None:
        fi = max(0, min(debug_frame, n_frames - 1))
        out_path = out_dir / f"{dataset_name}_{file_tag}_debug.png"
        global _render_ctx
        _render_ctx = ctx
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        _draw_frame(ax, fi, ctx)
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"  Debug frame {fi} saved: {out_path}")
        return

    # Parallel render to MP4
    out_path = out_dir / f"{dataset_name}_{file_tag}.mp4"
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize EgoAllo / IMU-TTO SMPL-H body in OptiTrack world frame."
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=[f"dataset{i}" for i in range(1, 12)],
        help="Dataset names (e.g. dataset1 dataset5). Default: all 11.",
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=list(VIS_METHODS.keys()),
        choices=list(VIS_METHODS.keys()),
        help=f"Methods to visualize. Default: all ({', '.join(VIS_METHODS.keys())}).",
    )
    parser.add_argument(
        "--smplh-model", type=Path, default=DEFAULT_SMPLH_MODEL,
        help="Path to SMPL-H neutral model.npz.",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for ds in args.datasets:
        for method in args.methods:
            process_dataset(
                ds,
                method=method,
                smplh_model_path=args.smplh_model,
                calibration_path=args.calibration,
                mesh_alpha=args.mesh_alpha,
                fps=args.fps,
                compute_verts=not args.no_mesh,
                show_gt=not args.no_gt,
                debug_frame=args.debug_frame,
                include_calibration=args.include_calibration,
            )
    print("\nAll done!")
