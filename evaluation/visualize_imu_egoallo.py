"""Visualize fused IMU + EgoAllo body: IMU joint rotations + EgoAllo root localization.

Uses EgoAllo's root position and orientation (from its egocentric body
estimation) combined with IMU-only body joint rotations.  This provides
better root localization than the naive glasses-to-pelvis offset used
in the IMU-only pipeline.

Saves body data (.npz) and video (.mp4) to evaluation/optitrack_gt_data/imu_egoallo/.

Usage:
    python evaluation/visualize_imu_egoallo.py --datasets dataset1
    python evaluation/visualize_imu_egoallo.py --datasets dataset1 dataset5 --debug-frame 0
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

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
    load_calib_duration_sec,
    load_cam_to_hand,
    load_optitrack_bag,
    load_optitrack_smplx_fit,
    nearest_indices,
    resolve_calibration_json,
    tracking_ns_to_utc_ns,
)

from evaluation.visualize_imu_only import (
    load_smplx_model,
    precompute_shape,
    smplx_fk,
    load_imu_csv,
    load_imu_calibration,
    load_apriltag_rotations,
    estimate_world_alignment,
    build_calibrated_bone_rotations,
    build_local_rots_from_global,
    JOINT_NAMES,
    DEFAULT_SMPLX_MODEL,
)

OUT_DIR = Path("evaluation/imu_egoallo")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║                   EgoAllo root loading                           ║
# ╚═══════════════════════════════════════════════════════════════════╝

def load_egoallo_root(
    dataset_name: str,
    aria_traj_path: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load EgoAllo root pose (position + orientation) in Aria world frame.

    Returns:
        (timestamps_utc_ns, root_positions, root_rotmats) or None
        root_positions: (T, 3) in Aria world (Z-up)
        root_rotmats:   (T, 3, 3) rotation from SMPL root frame to Aria world
    """
    aria_name = DATASET_TO_ARIA.get(dataset_name)
    if aria_name is None:
        return None

    egoallo_dir = RECORDINGS_DIR / dataset_name / "egoallo_outputs"
    if not egoallo_dir.exists():
        egoallo_dir = ARIA_DIR / aria_name / "egoallo_outputs"
    if not egoallo_dir.exists():
        return None

    npz_files = sorted(egoallo_dir.glob("aria_wrist_only_*.npz"))
    if not npz_files:
        return None

    npz_path = max(npz_files, key=lambda f: f.stat().st_size)
    print(f"    EgoAllo NPZ: {npz_path.name}")

    data = np.load(npz_path)
    if "Ts_world_root" not in data or "timestamps_ns" not in data:
        return None

    # SE3 format: wxyz_xyz (quaternion w,x,y,z + translation x,y,z)
    Ts_world_root = data["Ts_world_root"][0]   # (T, 7)
    timestamps_ns = data["timestamps_ns"]       # (T,)

    root_quat_wxyz = Ts_world_root[:, :4]
    root_pos = Ts_world_root[:, 4:]

    # scipy expects xyzw quaternion convention
    root_quat_xyzw = np.concatenate(
        [root_quat_wxyz[:, 1:], root_quat_wxyz[:, :1]], axis=1,
    )
    root_rotmats = Rotation.from_quat(root_quat_xyzw).as_matrix().astype(np.float64)

    timestamps_utc_ns = tracking_ns_to_utc_ns(timestamps_ns, aria_traj_path)

    return timestamps_utc_ns, root_pos.astype(np.float64), root_rotmats


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

    if verts is not None:
        face_step = max(1, len(faces) // 3000)
        tris = verts[faces[::face_step]]
        ax.add_collection3d(Poly3DCollection(
            tris, alpha=ctx["mesh_alpha"], facecolor="salmon", edgecolor="none",
        ))

    ax.scatter(
        joints[:NUM_BODY_JOINTS, 0],
        joints[:NUM_BODY_JOINTS, 1],
        joints[:NUM_BODY_JOINTS, 2],
        c="crimson", s=25, alpha=0.9, marker="o",
        label="IMU+EgoAllo", zorder=4,
    )
    for j1, j2 in BODY_BONES:
        if j1 < joints.shape[0] and j2 < joints.shape[0]:
            ax.plot(
                [joints[j1, 0], joints[j2, 0]],
                [joints[j1, 1], joints[j2, 1]],
                [joints[j1, 2], joints[j2, 2]],
                c="darkred", linewidth=1.2, alpha=0.7,
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
            ax.quiver(
                cam_pos[0], cam_pos[1], cam_pos[2],
                cam_axes[0, i], cam_axes[1, i], cam_axes[2, i],
                color=c, linewidth=1.5, arrow_length_ratio=0.15,
            )
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
        f"{ctx['dataset_name']} IMU+EgoAllo | t={time_s:.1f}s | "
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
    compute_verts: bool = True,
    show_gt: bool = True,
    debug_frame: Optional[int] = None,
    include_calibration: bool = False,
):
    dataset_dir = RECORDINGS_DIR / dataset_name
    bag_name = re.sub(r"dataset(\d)", r"dataset_\1", dataset_name) + ".bag"
    bag_path = BAG_DIR / bag_name
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name} (IMU + EgoAllo root fusion)")
    print(f"{'='*60}")

    # ── Check prerequisites ──────────────────────────────────────────
    imu_csv = dataset_dir / "imu" / "imu_data.csv"
    calib_json_path = dataset_dir / "imu_calibration.json"
    if not imu_csv.exists():
        print(f"  SKIP: IMU data not found: {imu_csv}")
        return
    if not calib_json_path.exists():
        print(f"  SKIP: IMU calibration not found: {calib_json_path}")
        return

    aria_name = DATASET_TO_ARIA.get(dataset_name)
    if aria_name is None:
        print(f"  SKIP: unknown dataset {dataset_name}")
        return

    aria_traj_path = (
        ARIA_DIR / f"mps_{aria_name}_vrs" / "slam" / "closed_loop_trajectory.csv"
    )
    if not aria_traj_path.exists():
        print(f"  SKIP: Aria trajectory not found: {aria_traj_path}")
        return

    # ── Load EgoAllo root poses ──────────────────────────────────────
    print("  Loading EgoAllo root localization ...")
    ego_root = load_egoallo_root(dataset_name, aria_traj_path)
    if ego_root is None:
        print(f"  SKIP: EgoAllo output not found for {dataset_name}")
        return
    ego_ts_utc, ego_root_pos, ego_root_rot = ego_root
    print(f"    EgoAllo: {len(ego_ts_utc)} frames, "
          f"t=[{ego_ts_utc[0]}, {ego_ts_utc[-1]}]")

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

    print(f"  Loading calibration: {calib_json_path}")
    calib = load_imu_calibration(calib_json_path)
    print(f"    Calibrated joints: {sorted(calib.keys())}")

    tag_rots = load_apriltag_rotations(dataset_dir)
    world_align: Dict[str, np.ndarray] = {}
    if tag_rots:
        world_align = estimate_world_alignment(streams, tag_rots)
        if world_align:
            print(f"    World alignment: {len(world_align)} joints aligned")
        else:
            print("    WARNING: world alignment failed")
    else:
        print("    WARNING: no AprilTag detections, skipping world alignment")

    # ── Build timeline from frames.csv ────────────────────────────────
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
    timeline_ns = np.array(frame_ts, dtype=np.int64)

    if not include_calibration:
        calib_dur = load_calib_duration_sec(dataset_name)
        if calib_dur is not None and len(timeline_ns) > 0:
            calib_end_ns = int(timeline_ns[0] + calib_dur * 1e9)
            trim_idx = int(np.searchsorted(timeline_ns, calib_end_ns))
            timeline_ns = timeline_ns[trim_idx:]
            print(f"  Trimmed {trim_idx} calibration frames ({calib_dur:.1f}s)")

    n_frames = len(timeline_ns)
    print(f"  Timeline: {n_frames} frames")

    # ── Resample EgoAllo root poses to frame timeline ────────────────
    print("  Resampling EgoAllo root to frame timestamps ...")
    ego_idx = nearest_indices(ego_ts_utc, timeline_ns)
    ego_root_pos_sync = ego_root_pos[ego_idx]     # (n_frames, 3), Aria Z-up
    ego_root_rot_sync = ego_root_rot[ego_idx]     # (n_frames, 3, 3), Aria Z-up

    # ── Compute IMU-only body pose ───────────────────────────────────
    print("  Computing calibrated bone rotations ...")
    bone_rots = build_calibrated_bone_rotations(
        streams, calib, world_align, timeline_ns,
    )

    pelvis_R0 = bone_rots["pelvis"][0].copy()
    canonical = pelvis_R0.T
    for joint in JOINT_NAMES:
        for i in range(n_frames):
            bone_rots[joint][i] = canonical @ bone_rots[joint][i]

    if debug_frame is not None:
        fk_indices = [max(0, min(debug_frame, n_frames - 1))]
    else:
        fk_indices = list(range(n_frames))

    # Z-up → Y-up conversion for injecting EgoAllo root into SMPLX FK
    R_zup_to_yup = np.array(
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64,
    )

    print(f"  Running SMPLX FK ({len(fk_indices)} frames) with EgoAllo root ...")
    all_joints = np.zeros((n_frames, model.num_joints, 3), dtype=np.float32)
    all_verts = (
        np.zeros((n_frames, model.v_template.shape[0], 3), dtype=np.float32)
        if compute_verts else None
    )
    for i in tqdm(fk_indices, desc="    FK", unit="frame"):
        global_rots_i = {j: bone_rots[j][i] for j in JOINT_NAMES}
        local_rots = build_local_rots_from_global(global_rots_i)

        # Replace root rotation with EgoAllo's orientation (Aria Z-up → SMPLX Y-up)
        local_rots[0] = (R_zup_to_yup @ ego_root_rot_sync[i]).astype(np.float32)

        joints_i, verts_i = smplx_fk(
            model, local_rots, v_shaped, j_tpose, compute_verts,
        )
        all_joints[i] = joints_i
        if compute_verts and verts_i is not None:
            all_verts[i] = verts_i

    # ── Align to OptiTrack ───────────────────────────────────────────
    cam_pos_opti = None
    cam_rot_opti = None
    box_positions = None
    has_box = False
    object_label = "box"
    pelvis_pos_opti = ego_root_pos_sync.copy()

    R_yup_to_zup = np.array(
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64,
    )

    pelvis_rot_opti = ego_root_rot_sync.copy()

    if bag_path.exists():
        print(f"  Loading OptiTrack bag: {bag_path}")
        opti_bag = load_optitrack_bag(bag_path)

        # Align EgoAllo pelvis directly to OptiTrack pelvis (not via Aria head)
        print("  Estimating EgoAllo → OptiTrack alignment ...")
        opti_ts = np.array(opti_bag.timestamps, dtype=np.int64)
        opti_pelvis = np.array([b[0] for b in opti_bag.body], dtype=np.float64)
        # Match EgoAllo frames to OptiTrack timestamps
        ego_idx_for_opti = nearest_indices(ego_ts_utc, opti_ts)
        ego_for_align = ego_root_pos[ego_idx_for_opti]
        opti_for_align = opti_pelvis
        # XY Procrustes (yaw only, both frames are Z-up gravity-aligned)
        cA = ego_for_align.mean(axis=0)
        cB = opti_for_align.mean(axis=0)
        H = (ego_for_align - cA)[:, :2].T @ (opti_for_align - cB)[:, :2]
        U, _, Vt = np.linalg.svd(H)
        R2 = Vt.T @ U.T
        if np.linalg.det(R2) < 0:
            Vt[-1] *= -1
            R2 = Vt.T @ U.T
        R_align = np.eye(3, dtype=np.float64)
        R_align[:2, :2] = R2
        t_align = cB - R_align @ cA
        yaw_deg = float(np.degrees(np.arctan2(R_align[1, 0], R_align[0, 0])))
        print(f"    Yaw offset: {yaw_deg:.1f}°, Translation: "
              f"[{t_align[0]:.3f}, {t_align[1]:.3f}, {t_align[2]:.3f}]")

        # EgoAllo root positions → OptiTrack world
        pelvis_pos_opti = (R_align @ ego_root_pos_sync.T).T + t_align
        pelvis_rot_opti = np.einsum("ij,njk->nik", R_align, ego_root_rot_sync)

        # FK output (Y-up) → OptiTrack (Z-up), centred on FK pelvis,
        # then translated to EgoAllo root in OptiTrack
        R_full = R_align @ R_yup_to_zup

        joints_opti = np.empty_like(all_joints)
        verts_opti = np.empty_like(all_verts) if all_verts is not None else None
        for i in range(n_frames):
            fk_pelvis = all_joints[i, 0].copy()
            centered_j = all_joints[i] - fk_pelvis[None, :]
            joints_opti[i] = (R_full @ centered_j.T).T + pelvis_pos_opti[i]
            if verts_opti is not None:
                centered_v = all_verts[i] - fk_pelvis[None, :]
                verts_opti[i] = (R_full @ centered_v.T).T + pelvis_pos_opti[i]

        # Camera (static tripod)
        if opti_bag.camera:
            cam_positions = np.array([c[0] for c in opti_bag.camera])
            cam_quats = np.array([c[1] for c in opti_bag.camera])
            mean_hand_pos = cam_positions.mean(axis=0).astype(np.float32)
            mean_hand_rot = (
                Rotation.from_quat(cam_quats).mean().as_matrix().astype(np.float32)
            )
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

        # Box
        if opti_bag.box:
            has_box = True
            object_label = opti_bag.object_label
            opti_ts_arr = np.array(opti_bag.timestamps, dtype=np.int64)
            box_idx = np.searchsorted(opti_ts_arr, timeline_ns, side="left")
            box_idx = np.clip(box_idx, 0, len(opti_bag.box) - 1)
            box_positions = np.array(
                [opti_bag.box[int(bi)][0] for bi in box_idx],
            )
    else:
        print("  WARNING: OptiTrack bag not found, showing in Aria world frame")
        joints_opti = np.empty_like(all_joints)
        verts_opti = np.empty_like(all_verts) if all_verts is not None else None
        for i in range(n_frames):
            fk_pelvis = all_joints[i, 0].copy()
            centered_j = all_joints[i] - fk_pelvis[None, :]
            joints_opti[i] = (R_yup_to_zup @ centered_j.T).T + ego_root_pos_sync[i]
            if verts_opti is not None:
                centered_v = all_verts[i] - fk_pelvis[None, :]
                verts_opti[i] = (
                    (R_yup_to_zup @ centered_v.T).T + ego_root_pos_sync[i]
                )

    # ── Load OptiTrack fitted SMPL-X ground truth ────────────────────
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

    # ── Save body data ───────────────────────────────────────────────
    body_data_path = OUT_DIR / f"{dataset_name}_imu_egoallo.npz"
    print(f"  Saving body data: {body_data_path}")
    save_dict = {
        "joints_opti": joints_opti,
        "timestamps_ns": timeline_ns,
        "pelvis_pos_opti": pelvis_pos_opti,
        "pelvis_rot_opti": pelvis_rot_opti,
    }
    if verts_opti is not None:
        save_dict["vertices_opti"] = verts_opti
        save_dict["faces"] = model.faces
    np.savez_compressed(body_data_path, **save_dict)

    # ── Compute axis limits ──────────────────────────────────────────
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
        "show_gt": show_gt and opti_fit is not None,
        "opti_fit": opti_fit,
        "gt_match_idx": gt_match_idx,
        "timestamps_ns": timeline_ns,
        "center": center,
        "span": span,
        "dataset_name": dataset_name,
        "n_frames": n_frames,
    }

    # ── Debug single frame ───────────────────────────────────────────
    if debug_frame is not None:
        fi = max(0, min(debug_frame, n_frames - 1))
        out_path = OUT_DIR / f"{dataset_name}_imu_egoallo_debug.png"
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
    out_path = OUT_DIR / f"{dataset_name}_imu_egoallo.mp4"
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
        description="Visualize IMU+EgoAllo fused body (IMU pose + EgoAllo root)."
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
        process_dataset(
            ds,
            smplx_path=args.smplx_model,
            calibration_path=args.calibration,
            mesh_alpha=args.mesh_alpha,
            fps=args.fps,
            compute_verts=not args.no_mesh,
            show_gt=not args.no_gt,
            debug_frame=args.debug_frame,
            include_calibration=args.include_calibration,
        )
    print("\nAll done!")
