"""Visualize SAM3D SMPL mesh/skeleton localized in the OptiTrack world frame.

Transforms per-frame SMPL output from camera coordinates into the OptiTrack
world frame using:
  - Camera hand-eye calibration (T_cam2hand)
  - OptiTrack camera_hand poses from ROS bags

Optionally overlays OptiTrack body markers for comparison.

Usage:
    python evaluation/visualize_sam3d.py --datasets dataset1
    python evaluation/visualize_sam3d.py --datasets dataset1 dataset2 --no-optitrack-overlay
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation
from tqdm import tqdm

_ts = get_typestore(Stores.ROS1_NOETIC)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.eval_utils import (
    BAG_DIR,
    BODY_BONES,
    NUM_BODY_JOINTS,
    RECORDINGS_DIR,
    load_calib_duration_sec,
    load_cam_to_hand,
    load_optitrack_bag,
    load_optitrack_smplx_fit,
    nearest_indices,
    resolve_calibration_json,
)

OUT_DIR = Path("evaluation/sam3d")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CameraPoseSequence:
    timestamps: np.ndarray      # (N,) int64 nanoseconds
    world_from_hand: np.ndarray  # (N, 4, 4) float32

@dataclass
class Sam3dData:
    vertices: np.ndarray    # (N, 10475, 3) float32
    joints: np.ndarray      # (N, 22, 3) float32  (body joints only)
    timestamps: np.ndarray  # (N,) int64 nanoseconds
    faces: np.ndarray       # (F, 3) int32  mesh faces
    frame_ids: np.ndarray   # (N,) int32  video frame indices from frame_names


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_sam3d_smpl(dataset_dir: Path) -> Sam3dData:
    """Load SMPL vertices, body joints, timestamps, and faces.

    Applies ``cam_t`` to place the body in metric camera coordinates
    (X-right, Y-down, Z-forward) at the correct depth.
    """
    smpl_dir = dataset_dir / "smpl_output"
    verts = np.load(smpl_dir / "smpl_vertices.npy").astype(np.float32)
    params = np.load(smpl_dir / "smpl_parameters.npz")
    all_joints = params["joints"].astype(np.float32)
    body_joints = all_joints[:, :NUM_BODY_JOINTS, :]
    cam_t = params["cam_t"].astype(np.float32)  # (N, 3)

    verts += cam_t[:, None, :]
    body_joints += cam_t[:, None, :]

    # Extract per-frame UTC timestamps and frame IDs from frame_names
    # embedded in the SMPL output (e.g. "frame_000042_1771532945060830976_color").
    # This is robust to the SMPL pipeline skipping video frames, which
    # would break positional alignment with frames.csv.
    frame_names = params["frame_names"]
    parsed = [re.search(r"frame_(\d+)_(\d+)_color", str(fn)) for fn in frame_names]
    frame_ids = np.array([int(m.group(1)) for m in parsed], dtype=np.int32)
    frame_ts = np.array([int(m.group(2)) for m in parsed], dtype=np.int64)

    n = min(verts.shape[0], frame_ts.shape[0])
    verts = verts[:n]
    body_joints = body_joints[:n]
    frame_ts = frame_ts[:n]
    frame_ids = frame_ids[:n]

    faces = _load_smplx_faces()
    return Sam3dData(vertices=verts, joints=body_joints,
                     timestamps=frame_ts, faces=faces, frame_ids=frame_ids)


def _load_smplx_faces() -> np.ndarray:
    """Load SMPL-X mesh faces from the model file or the smplx package."""
    model_path = Path("model/smplx/SMPLX_NEUTRAL.npz")
    if model_path.exists():
        m = np.load(model_path, allow_pickle=True)
        return m["f"].astype(np.int32)
    try:
        import smplx as smplx_pkg
        body_model = smplx_pkg.create(
            str(Path("MHR/model")), model_type="smplx",
            gender="neutral", ext="npz", batch_size=1,
        )
        return body_model.faces.astype(np.int32)
    except Exception:
        pass
    raise FileNotFoundError(
        "Cannot find SMPL-X faces. Provide model/smplx/SMPLX_NEUTRAL.npz "
        "or install the smplx package with MHR/model."
    )




def load_camera_hand_poses(bag_path: Path) -> CameraPoseSequence:
    """Read /camera_hand poses from a ROS bag as 4x4 transforms."""
    timestamps: list[int] = []
    poses: list[np.ndarray] = []
    with Reader(bag_path) as reader:
        for conn, stamp, rawdata in reader.messages():
            if "camera_hand" not in conn.topic:
                continue
            msg = _ts.deserialize_ros1(rawdata, conn.msgtype)
            p = msg.pose.position
            q = msg.pose.orientation
            Rm = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix().astype(np.float32)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = Rm
            T[:3, 3] = np.array([p.x, p.y, p.z], dtype=np.float32)
            timestamps.append(int(stamp))
            poses.append(T)
    if not timestamps:
        raise RuntimeError(f"No /camera_hand poses in {bag_path}")
    return CameraPoseSequence(
        timestamps=np.array(timestamps, dtype=np.int64),
        world_from_hand=np.stack(poses, axis=0),
    )




def transform_points(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (..., 3) points."""
    ones = np.ones((*pts.shape[:-1], 1), dtype=pts.dtype)
    homo = np.concatenate([pts, ones], axis=-1)
    return (T @ homo[..., None])[..., :3, 0]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _find_h264_encoder() -> str:
    """Return the best available H.264 encoder name."""
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


_render_ctx: dict = {}


def _init_render_worker(ctx: dict):
    global _render_ctx
    _render_ctx = ctx


def _draw_frame(ax, frame_idx: int, ctx: dict):
    """Draw a single visualization frame onto *ax*.

    ``frame_idx`` is a *timeline* index (one per video frame).
    SAM3D data may not exist for every timeline frame (skipped by SMPL
    pipeline); in that case only the GT overlay is rendered.
    """
    ax.cla()

    smpl_idx = ctx["timeline_to_smpl"][frame_idx]
    has_sam3d = smpl_idx >= 0

    # SAM3D SMPL mesh + skeleton (only when SMPL output exists)
    if has_sam3d:
        faces = ctx["faces"]
        verts = ctx["sam3d_verts_world"][smpl_idx] if ctx["sam3d_verts_world"] is not None else None
        joints = ctx["sam3d_joints_world"][smpl_idx]

        if verts is not None:
            face_step = max(1, len(faces) // 3000)
            triangles = verts[faces[::face_step]]
            mesh_coll = Poly3DCollection(
                triangles,
                alpha=ctx["mesh_alpha"],
                facecolor="salmon",
                edgecolor="none",
            )
            ax.add_collection3d(mesh_coll)

        ax.scatter(
            joints[:, 0], joints[:, 1], joints[:, 2],
            c="crimson", s=25, alpha=0.9, marker="o",
            label="SAM3D SMPL", zorder=4,
        )
        for j1, j2 in BODY_BONES:
            if j1 < joints.shape[0] and j2 < joints.shape[0]:
                ax.plot(
                    [joints[j1, 0], joints[j2, 0]],
                    [joints[j1, 1], joints[j2, 1]],
                    [joints[j1, 2], joints[j2, 2]],
                    c="darkred", linewidth=1.2, alpha=0.7,
                )

    # OptiTrack fitted SMPL-X overlay (always shown when available)
    if ctx["show_optitrack"] and ctx["opti_fit"] is not None:
        opti_fit = ctx["opti_fit"]
        opti_idx = ctx["opti_match_idx"][frame_idx]
        gt_joints = opti_fit["joints_zup"][opti_idx]
        gt_verts = opti_fit["vertices_zup"][opti_idx]
        gt_faces = opti_fit["faces"]

        gt_face_step = max(1, len(gt_faces) // 2000)
        gt_tris = gt_verts[gt_faces[::gt_face_step]]
        gt_mesh = Poly3DCollection(
            gt_tris, alpha=0.12,
            facecolor="deepskyblue", edgecolor="none",
        )
        ax.add_collection3d(gt_mesh)

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

    # Camera position, orientation axes & frustum (static tripod)
    cam_pos = ctx["cam_position"]
    cam_rot = ctx["cam_rotation"]
    ax.scatter(*cam_pos, c="red", s=60, marker="^", label="Camera", zorder=5)
    axis_len = 0.15
    cam_axes = cam_rot * axis_len
    for i, c in enumerate(["red", "green", "blue"]):
        ax.quiver(
            cam_pos[0], cam_pos[1], cam_pos[2],
            cam_axes[0, i], cam_axes[1, i], cam_axes[2, i],
            color=c, linewidth=1.5, arrow_length_ratio=0.15,
        )

    # Camera frustum wireframe
    frust_d = 0.3
    frust_hw, frust_hh = 0.12, 0.20
    corners_local = np.array([
        [-frust_hw, -frust_hh, frust_d],
        [ frust_hw, -frust_hh, frust_d],
        [ frust_hw,  frust_hh, frust_d],
        [-frust_hw,  frust_hh, frust_d],
    ], dtype=np.float32)
    corners_world = (cam_rot @ corners_local.T).T + cam_pos
    for cw in corners_world:
        ax.plot(
            [cam_pos[0], cw[0]], [cam_pos[1], cw[1]], [cam_pos[2], cw[2]],
            c="gray", linewidth=0.8, alpha=0.6,
        )
    for i in range(4):
        j = (i + 1) % 4
        ax.plot(
            [corners_world[i, 0], corners_world[j, 0]],
            [corners_world[i, 1], corners_world[j, 1]],
            [corners_world[i, 2], corners_world[j, 2]],
            c="gray", linewidth=0.8, alpha=0.6,
        )

    # Box / object
    if ctx["has_box"]:
        box_pos = ctx["box_positions"][frame_idx]
        ax.scatter(*box_pos, c="orange", s=120, marker="s",
                   label=f"Object ({ctx['object_label']})", zorder=5)

    # Ground plane (Z=0 grid, OptiTrack Z-up)
    center = ctx["center"]
    span = ctx["span"]
    grid_n = 10
    grid_range = np.linspace(center[0] - span, center[0] + span, grid_n + 1)
    grid_y = np.linspace(center[1] - span, center[1] + span, grid_n + 1)
    for gx in grid_range:
        ax.plot(
            [gx, gx], [center[1] - span, center[1] + span], [0, 0],
            c="lightgray", linewidth=0.4, alpha=0.5,
        )
    for gy in grid_y:
        ax.plot(
            [center[0] - span, center[0] + span], [gy, gy], [0, 0],
            c="lightgray", linewidth=0.4, alpha=0.5,
        )

    # Axis limits & labels
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    timeline_ts = ctx["timeline_ts"]
    t0 = timeline_ts[0]
    time_s = (timeline_ts[frame_idx] - t0) / 1e9
    sam3d_tag = "" if has_sam3d else " [no SMPL]"
    title = f"{ctx['dataset_name']} SAM3D | t={time_s:.1f}s | frame {frame_idx}/{ctx['n_timeline']}{sam3d_tag}"
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.view_init(elev=25, azim=-60 + frame_idx * 0.3)


def _render_frame_to_png(args: tuple):
    """Render one frame to a PNG file (runs in a worker process)."""
    frame_idx, png_path = args
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    _draw_frame(ax, frame_idx, _render_ctx)
    fig.savefig(png_path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_dataset(
    dataset_name: str,
    *,
    calibration_path: Path,
    show_optitrack: bool = True,
    mesh_alpha: float = 0.20,
    flip_yz: bool = False,
    fps: int = 30,
    compute_verts: bool = True,
    debug_frame: int | None = None,
    include_calibration: bool = False,
):
    """Transform SAM3D SMPL into OptiTrack world frame and render to MP4."""
    dataset_dir = RECORDINGS_DIR / dataset_name
    bag_name = re.sub(r"dataset(\d)", r"dataset_\1", dataset_name) + ".bag"
    bag_path = BAG_DIR / bag_name
    print(f"\nProcessing {dataset_name}")
    print(f"  SMPL source: {dataset_dir / 'smpl_output'}")
    print(f"  OptiTrack bag: {bag_path}")

    # Load data
    print("  Loading SAM3D SMPL data ...")
    sam3d = load_sam3d_smpl(dataset_dir)
    n_smpl = sam3d.vertices.shape[0]
    print(f"  SAM3D: {n_smpl} SMPL frames, {sam3d.vertices.shape[1]} vertices")

    # Build full video timeline from frames.csv so the output covers every
    # video frame.  GT plays continuously; SAM3D disappears on skipped frames.
    frames_csv = dataset_dir / "frames.csv"
    timeline_ids: list[int] = []
    timeline_ts_list: list[int] = []
    with open(frames_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("utc_timestamp_ns"):
                timeline_ids.append(int(row["frame_id"]))
                timeline_ts_list.append(int(row["utc_timestamp_ns"]))
    timeline_ts = np.array(timeline_ts_list, dtype=np.int64)

    # Optionally strip the initial calibration period
    if not include_calibration:
        calib_dur = load_calib_duration_sec(dataset_name)
        if calib_dur is not None and len(timeline_ts) > 0:
            calib_end_ns = int(timeline_ts[0] + calib_dur * 1e9)
            trim_idx = int(np.searchsorted(timeline_ts, calib_end_ns))
            timeline_ts = timeline_ts[trim_idx:]
            timeline_ids = timeline_ids[trim_idx:]
            print(f"  Trimmed {trim_idx} calibration frames ({calib_dur:.1f}s)")

    n_timeline = len(timeline_ts)

    smpl_id_to_idx = {int(fid): idx for idx, fid in enumerate(sam3d.frame_ids)}
    timeline_to_smpl = np.full(n_timeline, -1, dtype=np.int64)
    for ti, fid in enumerate(timeline_ids):
        if fid in smpl_id_to_idx:
            timeline_to_smpl[ti] = smpl_id_to_idx[fid]

    n_present = int((timeline_to_smpl >= 0).sum())
    print(f"  Timeline: {n_timeline} video frames, {n_present} with SMPL, "
          f"{n_timeline - n_present} skipped")

    print("  Loading camera calibration ...")
    calib_json = resolve_calibration_json(calibration_path)
    T_cam2hand = load_cam_to_hand(calib_json)

    print("  Loading OptiTrack camera poses ...")
    cam_seq = load_camera_hand_poses(bag_path)
    print(f"  Camera poses: {cam_seq.world_from_hand.shape[0]} frames")

    R_flip = np.eye(4, dtype=np.float32)
    if flip_yz:
        R_flip[1, 1] = -1.0
        R_flip[2, 2] = -1.0
        print("  Applying manual Y/Z flip")

    # Camera is on a tripod (static). Average all poses for robustness.
    mean_pos = cam_seq.world_from_hand[:, :3, 3].mean(axis=0)
    rots = Rotation.from_matrix(cam_seq.world_from_hand[:, :3, :3])
    mean_rot = rots.mean().as_matrix().astype(np.float32)
    T_world_hand = np.eye(4, dtype=np.float32)
    T_world_hand[:3, :3] = mean_rot
    T_world_hand[:3, 3] = mean_pos
    T_world_cam = T_world_hand @ T_cam2hand @ R_flip
    T_world_cam_viz = T_world_hand @ T_cam2hand
    cam_position = T_world_cam_viz[:3, 3].astype(np.float32)
    cam_rotation = T_world_cam_viz[:3, :3].astype(np.float32)
    print(f"  Static camera position: [{cam_position[0]:.3f}, {cam_position[1]:.3f}, {cam_position[2]:.3f}]")

    # Transform SMPL frames to OptiTrack world frame
    print("  Transforming to OptiTrack world frame ...")
    verts_world = None
    if compute_verts:
        verts_world = np.empty_like(sam3d.vertices)
    joints_world = np.empty_like(sam3d.joints)
    for i in range(n_smpl):
        if verts_world is not None:
            verts_world[i] = transform_points(sam3d.vertices[i], T_world_cam)
        joints_world[i] = transform_points(sam3d.joints[i], T_world_cam)

    # Load fitted SMPL-X for GT overlay, matched to the full timeline
    opti_fit = None
    opti_match_idx = None
    if show_optitrack:
        print("  Loading OptiTrack fitted SMPL-X ...")
        opti_fit = load_optitrack_smplx_fit(dataset_name)
        if opti_fit is not None:
            opti_match_idx = nearest_indices(
                opti_fit["timestamps"], timeline_ts,
            )
            print(f"  OptiTrack GT: {opti_fit['joints_zup'].shape[0]} fit frames, "
                  f"{opti_fit['vertices_zup'].shape[1]} vertices")
        else:
            print("  Warning: no fitted SMPL-X found, skipping GT overlay")

    # Box object from OptiTrack bag
    box_positions = None
    has_box = False
    object_label = "box"
    if bag_path.exists():
        opti_bag = load_optitrack_bag(bag_path)
        if opti_bag.box:
            has_box = True
            object_label = opti_bag.object_label
            opti_ts_arr = np.array(opti_bag.timestamps, dtype=np.int64)
            box_idx = np.searchsorted(opti_ts_arr, timeline_ts, side="left")
            box_idx = np.clip(box_idx, 0, len(opti_bag.box) - 1)
            box_positions = np.array([opti_bag.box[int(bi)][0] for bi in box_idx])
            print(f"  Object ({object_label}): {len(opti_bag.box)} frames")

    # Pelvis rotation: global_orient (axis-angle in camera frame) → OptiTrack
    smpl_params = np.load(dataset_dir / "smpl_output" / "smpl_parameters.npz")
    global_orient_aa = smpl_params["global_orient"][:n_smpl].astype(np.float32)
    pelvis_rot_cam = Rotation.from_rotvec(global_orient_aa).as_matrix().astype(np.float64)
    R_world_cam = T_world_cam[:3, :3].astype(np.float64)
    pelvis_rot_opti = np.einsum("ij,njk->nik", R_world_cam, pelvis_rot_cam)

    # Save body data: only SMPL frames within the (trimmed) timeline window
    save_mask = (sam3d.timestamps >= timeline_ts[0]) & (sam3d.timestamps <= timeline_ts[-1])
    save_joints = joints_world[save_mask]
    save_ts = sam3d.timestamps[save_mask]
    save_pelvis_rot = pelvis_rot_opti[save_mask]

    body_data_path = OUT_DIR / f"{dataset_name}_sam3d.npz"
    print(f"  Saving body data: {body_data_path} ({save_joints.shape[0]} frames)")
    save_dict = {
        "joints_opti": save_joints,
        "timestamps_ns": save_ts,
        "video_timestamps_ns": timeline_ts,
        "pelvis_pos_opti": save_joints[:, 0, :],
        "pelvis_rot_opti": save_pelvis_rot,
    }
    if verts_world is not None:
        save_dict["vertices_opti"] = verts_world[save_mask]
        save_dict["faces"] = sam3d.faces
    np.savez_compressed(body_data_path, **save_dict)

    # Compute axis limits from world-space data
    pts_for_limits = verts_world.reshape(-1, 3) if verts_world is not None else joints_world.reshape(-1, 3)
    all_pts = [pts_for_limits]
    all_pts.append(cam_position[None, :])
    if opti_fit is not None:
        all_pts.append(opti_fit["joints_zup"].reshape(-1, 3))
    if box_positions is not None:
        all_pts.append(box_positions)
    all_pts = np.concatenate(all_pts, axis=0)
    mins = np.percentile(all_pts, 1, axis=0)
    maxs = np.percentile(all_pts, 99, axis=0)
    center = (mins + maxs) / 2
    span = (maxs - mins).max() / 2 * 1.2

    ctx = {
        "sam3d_verts_world": verts_world,
        "sam3d_joints_world": joints_world,
        "faces": sam3d.faces,
        "mesh_alpha": mesh_alpha,
        "cam_position": cam_position,
        "cam_rotation": cam_rotation,
        "show_optitrack": show_optitrack and opti_fit is not None,
        "opti_fit": opti_fit,
        "opti_match_idx": opti_match_idx,
        "has_box": has_box,
        "box_positions": box_positions,
        "object_label": object_label,
        "timeline_ts": timeline_ts,
        "timeline_to_smpl": timeline_to_smpl,
        "center": center,
        "span": span,
        "dataset_name": dataset_name,
        "n_timeline": n_timeline,
    }

    # Debug: single frame
    if debug_frame is not None:
        fi = max(0, min(debug_frame, n_timeline - 1))
        out_path = OUT_DIR / f"{dataset_name}_sam3d_debug.png"
        global _render_ctx
        _render_ctx = ctx
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        _draw_frame(ax, fi, ctx)
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"  Debug frame {fi} saved: {out_path}")
        return

    # Parallel rendering over the full timeline
    out_path = OUT_DIR / f"{dataset_name}_sam3d.mp4"
    with tempfile.TemporaryDirectory() as tmpdir:
        args_list = [
            (i, str(Path(tmpdir) / f"{i:06d}.png"))
            for i in range(n_timeline)
        ]
        n_workers = max(1, min(mp.cpu_count() // 2, n_timeline))
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
        description="Visualize SAM3D SMPL in OptiTrack world frame."
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=[f"dataset{i}" for i in range(1, 12)],
        help="Dataset names (e.g. dataset1 dataset2). Default: all 11.",
    )
    parser.add_argument(
        "--calibration", type=Path,
        default=Path("camera_calibration"),
        help="Path to calibration_result.json or parent directory.",
    )
    parser.add_argument(
        "--no-optitrack-overlay",
        action="store_true",
        help="Hide OptiTrack body markers overlay.",
    )
    parser.add_argument(
        "--mesh-alpha", type=float, default=0.80,
        help="Mesh face transparency (default: 0.80).",
    )
    parser.add_argument(
        "--no-mesh", action="store_true",
        help="Disable mesh rendering (skeleton only, much faster).",
    )
    parser.add_argument(
        "--flip-yz", action="store_true", default=None,
        help="Force SMPL Y-up -> OpenCV flip (auto-detected if omitted).",
    )
    parser.add_argument(
        "--no-flip-yz", action="store_true",
        help="Force no coordinate flip.",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Output video FPS (default: 30, matching video frame rate).",
    )
    parser.add_argument(
        "--debug-frame", type=int, default=None,
        help="Render a single frame and save as PNG for quick debugging.",
    )
    parser.add_argument(
        "--include-calibration", action="store_true",
        help="Include the initial calibration period (excluded by default).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    flip = None
    if args.flip_yz:
        flip = True
    elif args.no_flip_yz:
        flip = False

    for ds in args.datasets:
        process_dataset(
            ds,
            calibration_path=args.calibration,
            show_optitrack=not args.no_optitrack_overlay,
            mesh_alpha=args.mesh_alpha,
            flip_yz=flip,
            fps=args.fps,
            compute_verts=not args.no_mesh,
            debug_frame=args.debug_frame,
            include_calibration=args.include_calibration,
        )
    print("\nAll done!")
