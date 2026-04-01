"""Multi-method pose visualization with third-person RGB.

Displays side-by-side comparisons of different pose estimation methods:
  - Ground truth (SMPLX, orange)
  - IMU-only reconstruction (blue)
  - TTO predictions (purple)
  - EgoAllo predictions (magenta)

Usage:
  python 05_visualize.py <session_dir>
  python 05_visualize.py <session_dir> --no-imu --no-gt
"""

from __future__ import annotations

import bisect
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure src/ is importable.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import viser
import viser.transforms as vtf
from PIL import Image

from utils.smpl_utils import (
    SmplxModel,
    build_local_rots_from_imu,
    load_smplx_model,
    precompute_shape,
    smplx_forward_kinematics,
)
from utils.sync_utils import (
    compute_time_intersection_ns,
    load_calibration,
    load_imu_csv,
    sample_streams_hold,
)
from utils.apriltag_utils import (
    load_apriltag_rotations_by_time,
    estimate_world_alignment_from_tags,
    quaternion_to_matrix_wxyz,
    rot_angle_deg,
)
from utils.imu_id_mapping import IMU_ID_TO_JOINT, SMPLX_JOINT_INDEX_MAP

# Lazy imports — only used when respective methods are enabled.
_viewer = None

def _get_viewer():
    global _viewer
    if _viewer is None:
        _UTILS_DIR = _SRC_DIR / "utils"
        if str(_UTILS_DIR) not in sys.path:
            sys.path.insert(0, str(_UTILS_DIR))
        from importlib import import_module
        _viewer = import_module("imu_pose_viewer")
    return _viewer


# ── Colours & layout ────────────────────────────────────────────────────────

METHODS = {
    "gt":      {"color": (0.95, 0.55, 0.20), "x": -1.5, "label": "Ground Truth"},
    "imu":     {"color": (0.35, 0.75, 0.95), "x":  0.0, "label": "IMU-only"},
    "tto":     {"color": (0.60, 0.40, 0.95), "x":  1.5, "label": "TTO"},
    "egoallo": {"color": (0.90, 0.40, 0.85), "x":  3.0, "label": "EgoAllo"},
}


# ── CLI arguments ───────────────────────────────────────────────────────────

@dataclass
class Args:
    session_dir: Path
    """Session directory containing IMU data, calibration, and outputs."""

    smplx_model: Path = Path("model/smplx/SMPLX_NEUTRAL.npz")
    """Path to SMPLX neutral model."""

    egoallo_csv: Optional[Path] = None
    """EgoAllo prediction CSV. Auto-discovered from <session>/egoallo/ if None."""

    tto_csv: Optional[Path] = None
    """TTO prediction CSV. Auto-discovered from <session>/tto/ if None."""

    show_imu: bool = True
    show_gt: bool = True
    show_egoallo: bool = True
    show_tto: bool = True
    show_rgb: bool = True

    rate_hz: float = 30.0
    """Playback frame rate (also IMU sampling rate)."""

    port: int = 8080
    video_max_width: int = 480


# ── Data loaders ────────────────────────────────────────────────────────────

def _load_gt(session_dir: Path) -> Optional[object]:
    """Load SMPLX ground truth."""
    viewer = _get_viewer()
    return viewer.load_smpl_ground_truth(session_dir)


def _discover_csv(session_dir: Path, subdir: str) -> Optional[Path]:
    d = session_dir / subdir
    if not d.is_dir():
        return None
    csvs = sorted(d.glob("*.csv"))
    return csvs[0] if csvs else None


def _load_tto_or_egoallo(csv_path: Path, num_joints: int) -> Tuple[Dict, List]:
    """Load TTO/EgoAllo CSV → {utc_ns: (J,3,3)} and sorted timestamps."""
    viewer = _get_viewer()
    d = viewer.load_tto_local_rotations(csv_path, num_joints)
    t_sorted = sorted(d.keys())
    return d, t_sorted


def _get_nearest(sorted_t: List[int], t_ns: int) -> Optional[int]:
    idx = bisect.bisect_left(sorted_t, t_ns)
    if len(sorted_t) == 0:
        return None
    if idx == 0:
        return sorted_t[0]
    if idx >= len(sorted_t):
        return sorted_t[-1]
    left, right = sorted_t[idx - 1], sorted_t[idx]
    return left if abs(left - t_ns) <= abs(right - t_ns) else right


# ── Core ────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    session = args.session_dir.resolve()
    print(f"Session: {session}")

    # ── Load SMPLX model ────────────────────────────────────────────────────
    model = load_smplx_model(args.smplx_model)
    betas = np.zeros(model.shapedirs.shape[2], dtype=np.float32)
    v_shaped, j_tpose = precompute_shape(model, betas)
    print(f"Loaded SMPLX model ({model.num_joints} joints)")

    # ── Load IMU data + calibration ─────────────────────────────────────────
    imu_csv = session / "imu" / "imu_data.csv"
    calib_json = session / "imu_calibration.json"
    streams, calib, world_align = None, None, None
    expected_imus = sorted(IMU_ID_TO_JOINT.keys())

    if args.show_imu and imu_csv.exists() and calib_json.exists():
        streams = load_imu_csv(imu_csv)
        calib, _ = load_calibration(calib_json)
        # World alignment from AprilTags (optional)
        apriltag_dir = session / "apriltag"
        if apriltag_dir.is_dir():
            tag_rots = load_apriltag_rotations_by_time(apriltag_dir)
            if tag_rots:
                world_align = estimate_world_alignment_from_tags(tag_rots, calib)
        print(f"Loaded IMU streams ({len(streams)} sensors) + calibration")
    elif args.show_imu:
        print("IMU data or calibration not found — skipping IMU-only")
        args.show_imu = False

    # ── Load ground truth ───────────────────────────────────────────────────
    gt = None
    if args.show_gt:
        gt = _load_gt(session)
        if gt is None:
            print("Ground truth not found — skipping")
            args.show_gt = False
        else:
            print(f"Loaded ground truth ({len(gt.frame_id_to_index)} frames)")

    # ── Load TTO predictions ────────────────────────────────────────────────
    tto_dict, tto_t = None, []
    if args.show_tto:
        csv = args.tto_csv or _discover_csv(session, "tto")
        if csv and csv.exists():
            tto_dict, tto_t = _load_tto_or_egoallo(csv, model.num_joints)
            print(f"Loaded TTO ({len(tto_t)} frames) from {csv.name}")
        else:
            print("TTO CSV not found — skipping")
            args.show_tto = False

    # ── Load EgoAllo predictions ────────────────────────────────────────────
    ego_dict, ego_t = None, []
    if args.show_egoallo:
        csv = args.egoallo_csv or _discover_csv(session, "egoallo")
        if csv and csv.exists():
            ego_dict, ego_t = _load_tto_or_egoallo(csv, model.num_joints)
            print(f"Loaded EgoAllo ({len(ego_t)} frames) from {csv.name}")
        else:
            print("EgoAllo CSV not found — skipping")
            args.show_egoallo = False

    # ── Build timeline ──────────────────────────────────────────────────────
    # Load third-person RGB frames
    rgb_frame_ids, rgb_utc_ns, rgb_paths = np.array([]), np.array([]), []
    frames_csv = session / "frames.csv"
    if frames_csv.exists():
        viewer = _get_viewer()
        rgb_frame_ids, rgb_utc_ns, rgb_paths = viewer.load_frames_csv(session)

    # Timeline from IMU data
    if streams:
        t0, t1 = compute_time_intersection_ns(streams, expected_imus)
        step_ns = int(1e9 / args.rate_hz)
        timeline = np.arange(t0, t1, step_ns, dtype=np.int64)
        sampled = sample_streams_hold(streams, expected_imus, timeline)
    elif len(rgb_utc_ns) > 0:
        timeline = rgb_utc_ns.astype(np.int64)
        sampled = None
    else:
        print("No data source for timeline!")
        return

    # Map timeline → frame IDs (for GT lookup)
    timeline_frame_ids = np.full(len(timeline), -1, dtype=np.int64)
    if len(rgb_utc_ns) > 0:
        for i, t_ns in enumerate(timeline):
            idx = np.argmin(np.abs(rgb_utc_ns - t_ns))
            if abs(rgb_utc_ns[idx] - t_ns) < 50_000_000:  # 50ms
                timeline_frame_ids[i] = rgb_frame_ids[idx]

    # Map timeline → RGB paths
    timeline_rgb: List[Optional[str]] = [None] * len(timeline)
    if args.show_rgb and len(rgb_utc_ns) > 0:
        for i, t_ns in enumerate(timeline):
            idx = np.argmin(np.abs(rgb_utc_ns - t_ns))
            if abs(rgb_utc_ns[idx] - t_ns) < 50_000_000:
                timeline_rgb[i] = rgb_paths[idx]

    print(f"Timeline: {len(timeline)} frames")

    # ── IMU calibration offsets ─────────────────────────────────────────────
    B_R_imu = {}
    if calib:
        # Fixed tag→IMU rotation (90° axes swap typical for RoSHI hardware)
        T_R_IMU = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)
        for jname, B_R_tag in calib.items():
            B_R_imu[jname] = B_R_tag @ T_R_IMU

    # ── First-frame alignment reference ─────────────────────────────────────
    # Compute IMU pelvis at frame 0 for alignment
    imu_pelvis_G0 = None
    if sampled and args.show_imu:
        pelvis_imu_id = next(
            iid for iid, jn in IMU_ID_TO_JOINT.items() if jn == "pelvis"
        )
        q0 = sampled[pelvis_imu_id][0]
        W_R_S0 = quaternion_to_matrix_wxyz(q0)
        if world_align and "pelvis" in world_align:
            W_R_S0 = world_align["pelvis"] @ W_R_S0
        imu_pelvis_G0 = W_R_S0 @ B_R_imu.get("pelvis", np.eye(3)).T

    # GT pelvis at first valid frame for alignment
    gt_pelvis_G0 = None
    if gt is not None:
        for fid in timeline_frame_ids:
            gidx = gt.index_of(int(fid)) if fid >= 0 else None
            if gidx is not None:
                gt_pelvis_G0 = gt.joint_rotations_local[gidx, 0]
                break

    # Alignment: rotate IMU/TTO/EgoAllo to match GT pelvis at frame 0
    align_R = np.eye(3)
    if gt_pelvis_G0 is not None and imu_pelvis_G0 is not None:
        align_R = gt_pelvis_G0 @ imu_pelvis_G0.T

    # ── Viser setup ─────────────────────────────────────────────────────────
    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(dark_mode=True)
    server.scene.set_up_direction("y")
    server.scene.add_grid("/grid", position=(0, -0.01, 0), width=6, height=6)

    # Create meshes for each enabled method
    identity_verts = v_shaped.copy()
    mesh_handles = {}
    active_methods = []
    for name, cfg in METHODS.items():
        enabled = getattr(args, f"show_{name}", False)
        if not enabled:
            continue
        active_methods.append(name)
        mesh_handles[name] = server.scene.add_mesh_simple(
            f"/{name}_mesh",
            vertices=identity_verts,
            faces=model.faces,
            color=cfg["color"],
            opacity=0.6,
        )

    # Legend
    legend_parts = [f"{METHODS[m]['label']}" for m in active_methods]
    server.gui.add_markdown("**Methods:** " + " | ".join(legend_parts))

    # RGB image display
    rgb_handle = None
    if args.show_rgb and any(p is not None for p in timeline_rgb):
        init_img = np.zeros((240, 320, 3), dtype=np.uint8)
        rgb_handle = server.gui.add_image(init_img, label="Third-person RGB")

    @lru_cache(maxsize=64)
    def _load_rgb(path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if w > args.video_max_width:
            ratio = args.video_max_width / w
            img = img.resize((args.video_max_width, int(h * ratio)))
        return np.asarray(img, dtype=np.uint8)

    # Playback controls
    frame_slider = server.gui.add_slider(
        "Frame", min=0, max=max(0, len(timeline) - 1), step=1, initial_value=0
    )
    play_button = server.gui.add_button("Play / Pause")
    info_text = server.gui.add_text("Info", "")

    state = {"playing": False, "last_frame": -1}

    # ── Per-frame update ────────────────────────────────────────────────────

    def _compute_imu_pose(fi: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Compute IMU-only vertices and joints for frame fi."""
        if sampled is None:
            return None
        bone_global: Dict[str, np.ndarray] = {}
        for imu_id, jname in IMU_ID_TO_JOINT.items():
            q = sampled[imu_id][fi]
            W_R_S = quaternion_to_matrix_wxyz(q)
            if world_align and jname in world_align:
                W_R_S = world_align[jname] @ W_R_S
            bone_global[jname] = W_R_S @ B_R_imu.get(jname, np.eye(3)).T
        # Apply first-frame alignment
        for jname in bone_global:
            bone_global[jname] = align_R @ bone_global[jname]
        local = build_local_rots_from_imu(bone_global)
        joints, T_w, verts = smplx_forward_kinematics(
            model, local, betas, compute_vertices=True,
            v_shaped=v_shaped, j_tpose=j_tpose,
        )
        return verts, joints

    def _compute_csv_pose(
        local_dict: Dict, t_sorted: List[int], t_ns: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Compute pose from TTO/EgoAllo CSV predictions."""
        nearest = _get_nearest(t_sorted, t_ns)
        if nearest is None or abs(nearest - t_ns) > 100_000_000:  # 100ms
            return None
        local = local_dict.get(nearest)
        if local is None:
            return None
        # Apply alignment
        local = local.copy()
        local[0] = align_R @ local[0]
        joints, T_w, verts = smplx_forward_kinematics(
            model, local, betas, compute_vertices=True,
            v_shaped=v_shaped, j_tpose=j_tpose,
        )
        return verts, joints

    def _update(fi: int) -> None:
        if fi == state["last_frame"]:
            return
        state["last_frame"] = fi

        t_ns = int(timeline[fi])
        frame_id = int(timeline_frame_ids[fi])
        results: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}

        # Compute poses for each method
        if args.show_imu:
            results["imu"] = _compute_imu_pose(fi)

        if args.show_gt and gt is not None and frame_id >= 0:
            gidx = gt.index_of(frame_id)
            if gidx is not None:
                gt_local = gt.joint_rotations_local[gidx]
                j, T, v = smplx_forward_kinematics(
                    model, gt_local, betas, compute_vertices=True,
                    v_shaped=v_shaped, j_tpose=j_tpose,
                )
                results["gt"] = (v, j)
            else:
                results["gt"] = None
        elif args.show_gt:
            results["gt"] = None

        if args.show_tto and tto_dict:
            results["tto"] = _compute_csv_pose(tto_dict, tto_t, t_ns)

        if args.show_egoallo and ego_dict:
            results["egoallo"] = _compute_csv_pose(ego_dict, ego_t, t_ns)

        # Update meshes with offset
        with server.atomic():
            for name in active_methods:
                r = results.get(name)
                if r is None:
                    continue
                verts, joints = r
                pelvis = joints[0].copy()
                verts_c = verts - pelvis[None, :]
                verts_c[:, 0] += METHODS[name]["x"]
                mesh_handles[name].vertices = verts_c

            # Update RGB
            if rgb_handle is not None and timeline_rgb[fi] is not None:
                rgb_handle.image = _load_rgb(timeline_rgb[fi])

            info_text.value = f"Frame {fi}/{len(timeline)-1}  t={t_ns}"

    # ── Event handlers ──────────────────────────────────────────────────────

    @frame_slider.on_update
    def _(_) -> None:
        _update(int(frame_slider.value))

    @play_button.on_click
    def _(_) -> None:
        state["playing"] = not state["playing"]

    # ── Main loop ───────────────────────────────────────────────────────────
    print(f"Viser server running at http://localhost:{args.port}")
    _update(0)
    last_time = time.time()
    while True:
        time.sleep(1.0 / 60.0)
        if state["playing"]:
            now = time.time()
            if now - last_time >= 1.0 / args.rate_hz:
                last_time = now
                nxt = (int(frame_slider.value) + 1) % len(timeline)
                frame_slider.value = nxt


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))
