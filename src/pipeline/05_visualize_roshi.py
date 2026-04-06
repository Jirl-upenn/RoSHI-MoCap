"""Visualize RoSHI inference results.

Usage:
  python 05_visualize_roshi.py --session-dir received_recordings/new_walking
  python 05_visualize_roshi.py --session-dir received_recordings/new_walking --npz path/to/specific.npz
"""

from __future__ import annotations

import bisect
import csv
import json
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import viser
from PIL import Image
from scipy.spatial.transform import Rotation

from utils.smpl_utils import (
    load_smplx_model,
    precompute_shape,
    smplx_forward_kinematics,
)


@dataclass
class Args:
    session_dir: Path
    """Session directory containing outputs."""

    smplx_model: Path = Path("model/smplx/SMPLX_NEUTRAL.npz")
    """Path to SMPLX neutral model."""

    npz: Optional[Path] = None
    """Path to specific RoSHI .npz file. Auto-discovered if None."""

    show_rgb: bool = True
    """Show third-person RGB."""

    show_pointcloud: bool = True
    """Show MPS semidense point cloud."""

    rate_hz: float = 30.0
    port: int = 8080
    video_max_width: int = 480


# ── Helpers ──────────────────────────────────────────────────────────────────


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(... , 4) wxyz → (... , 3, 3)."""
    if q.ndim == 1:
        return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    shape = q.shape[:-1]
    flat = q.reshape(-1, 4)
    mats = Rotation.from_quat(flat[:, [1, 2, 3, 0]]).as_matrix()
    return mats.reshape(*shape, 3, 3)


def discover_roshi_npz(session_dir: Path) -> Optional[Path]:
    d = session_dir / "egoallo_outputs"
    if not d.is_dir():
        return None
    npzs = sorted(d.glob("roshi_*.npz"))
    required = {"body_quats", "Ts_world_root", "timestamps_ns"}
    for p in reversed(npzs):  # newest first
        try:
            with np.load(p) as data:
                if required.issubset(data.files):
                    return p
        except Exception:
            continue
    return None


def load_roshi_npz(
    npz_path: Path, num_joints: int
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], List[int]]:
    """Load RoSHI .npz → local rotations + root translations.

    Returns:
        rots:  {utc_ns: (J, 3, 3)}
        trans: {utc_ns: (3,) root translation in camera coords}
        sorted_timestamps
    """
    data = np.load(npz_path)
    timestamps = data["timestamps_ns"]
    body_q = data["body_quats"][0]        # (T, 21, 4) wxyz
    lhand_q = data["left_hand_quats"][0]  # (T, 15, 4)
    rhand_q = data["right_hand_quats"][0] # (T, 15, 4)
    Ts_root = data["Ts_world_root"][0]    # (T, 7) wxyz_xyz

    # Z-up (Aria world) → camera coords (Y-down)
    R_zup_to_cam = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)

    rots = {}
    trans = {}
    for t_idx in range(len(timestamps)):
        local = np.tile(np.eye(3, dtype=np.float32), (num_joints, 1, 1))
        local[0] = (R_zup_to_cam @ quat_wxyz_to_rotmat(Ts_root[t_idx, :4])).astype(np.float32)
        local[1:22] = quat_wxyz_to_rotmat(body_q[t_idx])
        local[25:40] = quat_wxyz_to_rotmat(lhand_q[t_idx])
        local[40:55] = quat_wxyz_to_rotmat(rhand_q[t_idx])

        xyz_cam = R_zup_to_cam @ Ts_root[t_idx, 4:7].astype(np.float64)

        key = int(timestamps[t_idx])
        rots[key] = local
        trans[key] = xyz_cam.astype(np.float32)

    return rots, trans, sorted(rots.keys())


def get_nearest(sorted_t: List[int], t_ns: int) -> Optional[int]:
    if not sorted_t:
        return None
    idx = bisect.bisect_left(sorted_t, t_ns)
    if idx == 0:
        return sorted_t[0]
    if idx >= len(sorted_t):
        return sorted_t[-1]
    a, b = sorted_t[idx - 1], sorted_t[idx]
    return a if abs(a - t_ns) <= abs(b - t_ns) else b


# ── Main ─────────────────────────────────────────────────────────────────────


def main(args: Args) -> None:
    session = args.session_dir.resolve()
    print(f"Session: {session}")

    # SMPLX
    model = load_smplx_model(args.smplx_model)
    betas = np.zeros(model.shapedirs.shape[2], dtype=np.float32)
    v_shaped, j_tpose = precompute_shape(model, betas)
    print(f"SMPLX: {model.num_joints} joints")

    # RoSHI predictions
    npz_path = args.npz or discover_roshi_npz(session)
    if npz_path is None:
        print("ERROR: No RoSHI .npz found in egoallo_outputs/. Run 04_inference.py first.")
        return
    roshi_dict, roshi_trans, roshi_t = load_roshi_npz(npz_path, model.num_joints)
    print(f"RoSHI: {len(roshi_t)} frames from {npz_path.name}")

    # Third-person RGB
    rgb_utc_ns, rgb_paths = np.array([]), []
    frames_csv = session / "frames.csv"
    if frames_csv.exists():
        utcs, paths = [], []
        with open(frames_csv) as f:
            for row in csv.DictReader(f):
                utcs.append(int(row["utc_timestamp_ns"]))
                paths.append(row["color_path"].strip())
        rgb_utc_ns = np.array(utcs, dtype=np.int64)
        rgb_paths = paths

    # Timeline: use RoSHI timestamps directly
    timeline = np.array(roshi_t, dtype=np.int64)

    # Skip calibration segment by default
    calib_skip = 0
    meta_json = session / "metadata.json"
    if meta_json.exists():
        with open(meta_json) as f:
            meta = json.load(f)
        calib_sec = meta.get("calibrationSegment", {}).get("suggestedCalibDurationSec", 0)
        if calib_sec > 0:
            calib_end_ns = timeline[0] + int(calib_sec * 1e9)
            calib_skip = int(np.searchsorted(timeline, calib_end_ns))
            print(f"Calibration: {calib_sec:.1f}s ({calib_skip} frames) — skipped")

    # Map timeline → RGB paths
    timeline_rgb: List[Optional[str]] = [None] * len(timeline)
    if args.show_rgb and len(rgb_utc_ns) > 0:
        for i, t_ns in enumerate(timeline):
            idx = np.argmin(np.abs(rgb_utc_ns - t_ns))
            if abs(rgb_utc_ns[idx] - t_ns) < 50_000_000:
                timeline_rgb[i] = rgb_paths[idx]

    # MPS semidense point cloud
    R_zup_to_cam = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    pts_cam = None
    if args.show_pointcloud:
        cached = sorted(session.glob("**/slam/_cached_filtered_points.npz"))
        if cached:
            pts = np.load(cached[0])["points"]  # (N, 3) Z-up
        else:
            csv_gz = sorted(session.glob("**/slam/semidense_points.csv.gz"))
            if csv_gz:
                import pandas as pd
                df = pd.read_csv(csv_gz[0])
                if "inv_dist_std" in df.columns:
                    df = df[df["inv_dist_std"] > 0.001]
                pts = df[["px_world", "py_world", "pz_world"]].values
            else:
                pts = None
        if pts is not None:
            pts_cam = (R_zup_to_cam @ pts.T).T.astype(np.float32)
            print(f"Point cloud: {len(pts_cam)} points")
        else:
            print("No point cloud found")

    print(f"Timeline: {len(timeline)} frames (starting from frame {calib_skip})")

    # ── Viser ────────────────────────────────────────────────────────────────
    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(dark_mode=True)
    server.scene.set_up_direction("-y")
    server.scene.add_grid("/grid", position=(0, 1.2, 0), width=50, height=50, plane="xz")

    if pts_cam is not None:
        # Color by height (Y in camera coords, inverted since -Y is up)
        import matplotlib.cm as cm
        height = -pts_cam[:, 1]  # -Y = up in camera coords
        h_norm = (height - height.min()) / (height.max() - height.min() + 1e-6)
        colors = (cm.turbo(h_norm)[:, :3] * 255).astype(np.uint8)
        server.scene.add_point_cloud(
            "/pointcloud", points=pts_cam, colors=colors, point_size=0.015,
        )

    mesh_handle = server.scene.add_mesh_simple(
        "/roshi",
        vertices=v_shaped.copy(),
        faces=model.faces,
        color=(0.60, 0.40, 0.95),
        opacity=0.8,
    )
    label_handle = server.scene.add_label("/roshi_label", text="RoSHI", position=(0, -0.3, 0))

    rgb_handle = None
    if args.show_rgb and any(p is not None for p in timeline_rgb):
        rgb_handle = server.gui.add_image(
            np.zeros((240, 320, 3), dtype=np.uint8), label="Third-person RGB"
        )

    @lru_cache(maxsize=64)
    def _load_rgb(path: str) -> np.ndarray:
        p = Path(path)
        if not p.is_absolute():
            p = session / p
        img = Image.open(p).convert("RGB")
        w, h = img.size
        if w > args.video_max_width:
            r = args.video_max_width / w
            img = img.resize((args.video_max_width, int(h * r)))
        return np.asarray(img, dtype=np.uint8)

    frame_slider = server.gui.add_slider(
        "Frame", min=0, max=max(0, len(timeline) - 1), step=1, initial_value=calib_skip
    )
    play_btn = server.gui.add_button("Play / Pause")
    info_text = server.gui.add_text("Info", "")

    state = {"playing": False, "last_frame": -1}

    def _update(fi: int) -> None:
        if fi == state["last_frame"]:
            return
        state["last_frame"] = fi
        t_ns = int(timeline[fi])

        nearest = get_nearest(roshi_t, t_ns)
        if nearest is None or abs(nearest - t_ns) > 100_000_000:
            return

        local = roshi_dict[nearest]
        j, _, v = smplx_forward_kinematics(
            model, local, betas, compute_vertices=True,
            v_shaped=v_shaped, j_tpose=j_tpose,
        )
        root_t = roshi_trans[nearest]
        pelvis = j[0].copy()
        v_world = v - pelvis[None, :] + root_t[None, :]
        j_world = j - pelvis[None, :] + root_t[None, :]

        with server.atomic():
            mesh_handle.vertices = v_world
            label_handle.position = (
                float(j_world[0, 0]),
                float(v_world[:, 1].min()) - 0.15,
                float(j_world[0, 2]),
            )
            if rgb_handle is not None and timeline_rgb[fi] is not None:
                rgb_handle.image = _load_rgb(timeline_rgb[fi])
            info_text.value = f"Frame {fi}/{len(timeline)-1}"

    @frame_slider.on_update
    def _(_) -> None:
        _update(int(frame_slider.value))

    @play_btn.on_click
    def _(_) -> None:
        state["playing"] = not state["playing"]

    _update(calib_skip)
    print(f"\nOpen the URL above in your browser")
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
