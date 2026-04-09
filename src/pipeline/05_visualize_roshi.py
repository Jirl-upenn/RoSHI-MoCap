"""Visualize RoSHI inference results.

Usage:
  python 05_visualize_roshi.py --session-dir received_recordings/new_walking
  python 05_visualize_roshi.py --session-dir received_recordings/new_walking --record output.mp4
"""

from __future__ import annotations

import bisect
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
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

    record: Optional[Path] = None
    """Record to video file (e.g. output.mp4). Renders offline at 1x speed."""

    camera_json: Optional[Path] = None
    """Camera pose JSON saved from interactive viewer. Auto-discovered from session dir if not set."""

    record_width: int = 1920
    record_height: int = 1080

    test_frame: Optional[int] = None
    """Render a single frame (by index) to a PNG for quick testing."""

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
    """Load RoSHI .npz → local rotations + root translations."""
    data = np.load(npz_path)
    timestamps = data["timestamps_ns"]
    body_q = data["body_quats"][0]
    lhand_q = data["left_hand_quats"][0]
    rhand_q = data["right_hand_quats"][0]
    Ts_root = data["Ts_world_root"][0]

    R_zup_to_cam = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)

    rots, trans = {}, {}
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


def compute_pose(model, roshi_dict, roshi_trans, roshi_t, t_ns, betas, v_shaped, j_tpose):
    """Compute mesh vertices + joints for a given timestamp."""
    nearest = get_nearest(roshi_t, t_ns)
    if nearest is None or abs(nearest - t_ns) > 100_000_000:
        return None, None
    local = roshi_dict[nearest]
    j, _, v = smplx_forward_kinematics(
        model, local, betas, compute_vertices=True,
        v_shaped=v_shaped, j_tpose=j_tpose,
    )
    root_t = roshi_trans[nearest]
    pelvis = j[0].copy()
    v_world = v - pelvis[None, :] + root_t[None, :]
    j_world = j - pelvis[None, :] + root_t[None, :]
    return v_world, j_world


def _make_pc_colors(pts_cam: np.ndarray) -> np.ndarray:
    """Compute point cloud colors matching viser's blue-purple palette."""
    import matplotlib.cm as cm
    height = -pts_cam[:, 1]
    h_norm = (height - height.min()) / (height.max() - height.min() + 1e-6)
    colors_rgba = (cm.cool(h_norm) * 255).astype(np.uint8)
    return colors_rgba


def _make_pc_mesh(pts_cam: np.ndarray, radius: float = 0.015):
    """Create point cloud as instanced small spheres for visible rendering."""
    import pyrender
    import trimesh
    colors = _make_pc_colors(pts_cam)  # (N, 4) uint8 RGBA
    sphere = trimesh.creation.uv_sphere(radius=radius, count=[4, 4])
    meshes = []
    for pt, c in zip(pts_cam, colors):
        s = sphere.copy()
        s.apply_translation(pt)
        s.visual.vertex_colors = np.tile(c, (len(s.vertices), 1))
        meshes.append(s)
    combined = trimesh.util.concatenate(meshes)
    return pyrender.Mesh.from_trimesh(combined, smooth=False)


# ── Test frame ──────────────────────────────────────────────────────────────


def render_test_frame(args, model, betas, v_shaped, j_tpose,
                      roshi_dict, roshi_trans, roshi_t,
                      timeline, calib_skip, pts_cam, timeline_rgb, session):
    """Render a single frame to PNG for quick visual check."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    import trimesh

    W, H = args.record_width, args.record_height
    fi = args.test_frame
    if fi >= len(timeline):
        print(f"ERROR: frame {fi} out of range (max {len(timeline)-1})")
        return

    t_ns = int(timeline[fi])
    v_world, j_world = compute_pose(
        model, roshi_dict, roshi_trans, roshi_t, t_ns,
        betas, v_shaped, j_tpose,
    )
    if v_world is None:
        print(f"ERROR: no pose at frame {fi}")
        return

    body_color = np.array([0.55, 0.35, 0.90, 0.9])
    cam_up = np.array([0.0, -1.0, 0.0])

    scene = pyrender.Scene(bg_color=[0.12, 0.12, 0.14, 1.0],
                           ambient_light=[0.3, 0.3, 0.3])

    if pts_cam is not None:
        scene.add(_make_pc_mesh(pts_cam))

    body_tm = trimesh.Trimesh(vertices=v_world, faces=model.faces, process=False)
    body_tm.visual.vertex_colors = (body_color * 255).astype(np.uint8)
    scene.add(pyrender.Mesh.from_trimesh(body_tm, smooth=True))

    # Camera
    cam_fov = np.pi / 3.0
    saved_cam_pos, saved_look_at = None, None
    cam_json = args.camera_json
    if cam_json is None:
        auto = session / "camera_pose.json"
        if auto.exists():
            cam_json = auto
    if cam_json is not None and cam_json.exists():
        with open(cam_json) as f:
            cam_data = json.load(f)
        saved_cam_pos = np.array(cam_data["position"])
        saved_look_at = np.array(cam_data["look_at"])
        if "fov" in cam_data:
            cam_fov = cam_data["fov"]

    if saved_cam_pos is not None:
        cam_pos = saved_cam_pos
        look_at = saved_look_at
    else:
        target = j_world[0].copy()
        cam_pos = target + np.array([0.0, -1.5, -4.0])
        look_at = target

    camera = pyrender.PerspectiveCamera(yfov=cam_fov, aspectRatio=W / H)
    forward = look_at - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, cam_up.astype(np.float64))
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    cam_pose = np.eye(4)
    cam_pose[:3, 0] = right
    cam_pose[:3, 1] = up
    cam_pose[:3, 2] = -forward
    cam_pose[:3, 3] = cam_pos
    scene.add(camera, pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0), pose=cam_pose)

    renderer = pyrender.OffscreenRenderer(W, H)
    color, _ = renderer.render(scene)
    renderer.delete()

    # Composite RGB inset
    if args.show_rgb and timeline_rgb[fi] is not None:
        rgb_path = Path(timeline_rgb[fi])
        if not rgb_path.is_absolute():
            rgb_path = session / rgb_path
        if rgb_path.exists():
            rgb_img = Image.open(rgb_path).convert("RGB")
            inset_h = H // 4
            ratio = inset_h / rgb_img.height
            inset_w = int(rgb_img.width * ratio)
            rgb_img = rgb_img.resize((inset_w, inset_h))
            rgb_arr = np.asarray(rgb_img)
            y0 = H - inset_h - 10
            x0 = W - inset_w - 10
            color[y0:y0+inset_h, x0:x0+inset_w] = rgb_arr

    out_png = session / f"test_frame_{fi}.png"
    Image.fromarray(color).save(out_png)
    print(f"Saved test frame → {out_png} ({W}x{H})")


# ── Offline recording ────────────────────────────────────────────────────────


def record_video(args, model, betas, v_shaped, j_tpose,
                 roshi_dict, roshi_trans, roshi_t,
                 timeline, calib_skip, pts_cam, timeline_rgb, session):
    """Render frames with pyrender and encode to video with ffmpeg."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    import trimesh

    W, H = args.record_width, args.record_height
    out_path = args.record

    # Body mesh color
    body_color = np.array([0.55, 0.35, 0.90, 0.9])

    # Setup pyrender scene (static elements)
    pc_mesh = None
    if pts_cam is not None:
        pc_cloud = _make_pc_mesh(pts_cam)


    cam_fov = np.pi / 3.0
    cam_up = np.array([0.0, -1.0, 0.0])
    saved_cam_pos = None
    saved_look_at = None
    cam_json = args.camera_json
    if cam_json is None:
        auto = session / "camera_pose.json"
        if auto.exists():
            cam_json = auto
    if cam_json is not None and cam_json.exists():
        with open(cam_json) as f:
            cam_data = json.load(f)
        saved_cam_pos = np.array(cam_data["position"])
        saved_look_at = np.array(cam_data["look_at"])
        if "fov" in cam_data:
            cam_fov = cam_data["fov"]
        print(f"Using saved camera from {cam_json} (fov={np.degrees(cam_fov):.1f}°, fixed)")

    camera = pyrender.PerspectiveCamera(yfov=cam_fov, aspectRatio=W / H)

    # Light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)

    renderer = pyrender.OffscreenRenderer(W, H)

    frames_to_render = range(calib_skip, len(timeline))
    n_frames = len(timeline) - calib_skip
    print(f"Recording {n_frames} frames at {args.rate_hz}fps → {out_path}")

    # ffmpeg pipe
    ffmpeg_cmd = [
        "/usr/bin/ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24",
        "-r", str(args.rate_hz),
        "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", "12",
        "-b:v", "8M", "-minrate", "5M", "-maxrate", "15M", "-bufsize", "10M",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    for count, fi in enumerate(frames_to_render):
        t_ns = int(timeline[fi])
        v_world, j_world = compute_pose(
            model, roshi_dict, roshi_trans, roshi_t, t_ns,
            betas, v_shaped, j_tpose,
        )

        # Build scene each frame
        scene = pyrender.Scene(bg_color=[0.12, 0.12, 0.14, 1.0],
                               ambient_light=[0.3, 0.3, 0.3])

        # Point cloud
        if pts_cam is not None:
            scene.add(pc_cloud)

        # Body mesh
        if v_world is not None:
            body_tm = trimesh.Trimesh(vertices=v_world, faces=model.faces, process=False)
            body_tm.visual.vertex_colors = (body_color * 255).astype(np.uint8)
            scene.add(pyrender.Mesh.from_trimesh(body_tm, smooth=True))

        if saved_cam_pos is not None:
            cam_pos = saved_cam_pos
            look_at = saved_look_at
        else:
            target = j_world[0].copy() if j_world is not None else np.zeros(3, dtype=np.float32)
            cam_pos = target + np.array([0.0, -1.5, -4.0])
            look_at = target
        forward = look_at - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        right = np.cross(forward, cam_up.astype(np.float64))
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)
        cam_pose = np.eye(4)
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = up
        cam_pose[:3, 2] = -forward
        cam_pose[:3, 3] = cam_pos
        scene.add(camera, pose=cam_pose)
        scene.add(light, pose=cam_pose)

        # Render
        color, _ = renderer.render(scene)

        # Composite RGB inset
        if args.show_rgb and timeline_rgb[fi] is not None:
            rgb_path = Path(timeline_rgb[fi])
            if not rgb_path.is_absolute():
                rgb_path = session / rgb_path
            if rgb_path.exists():
                rgb_img = Image.open(rgb_path).convert("RGB")
                # Resize to fit in corner
                inset_h = H // 4
                ratio = inset_h / rgb_img.height
                inset_w = int(rgb_img.width * ratio)
                rgb_img = rgb_img.resize((inset_w, inset_h))
                rgb_arr = np.asarray(rgb_img)
                # Bottom-right corner with margin
                y0 = H - inset_h - 10
                x0 = W - inset_w - 10
                color[y0:y0+inset_h, x0:x0+inset_w] = rgb_arr

        ffmpeg_proc.stdin.write(color.tobytes())

        if (count + 1) % 100 == 0 or count == 0:
            print(f"  {count+1}/{n_frames} frames rendered")

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    if ffmpeg_proc.returncode != 0:
        print(f"ffmpeg error (rc={ffmpeg_proc.returncode}):")
        print(ffmpeg_proc.stderr.read().decode())
    renderer.delete()
    print(f"Saved → {out_path}")


# ── Interactive viser ────────────────────────────────────────────────────────


def run_interactive(args, model, betas, v_shaped, j_tpose,
                    roshi_dict, roshi_trans, roshi_t,
                    timeline, calib_skip, pts_cam, timeline_rgb, session):
    """Run interactive viser viewer."""
    import viser

    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(dark_mode=True)
    server.scene.set_up_direction("-y")

    if pts_cam is not None:
        import matplotlib.cm as cm
        height = -pts_cam[:, 1]
        h_norm = (height - height.min()) / (height.max() - height.min() + 1e-6)
        colors = (cm.cool(h_norm)[:, :3] * 255).astype(np.uint8)
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
    record_btn = server.gui.add_button("Start Recording")
    save_cam_btn = server.gui.add_button("Save Camera")
    info_text = server.gui.add_text("Info", "")

    state = {"playing": False, "last_frame": -1, "last_j_world": None}
    # Fixed camera for recording: loaded from camera_pose.json or captured via Save Camera
    saved_cam = {"position": None, "look_at": None, "fov": None}
    cam_json_path = session / "camera_pose.json"
    if cam_json_path.exists():
        with open(cam_json_path) as f:
            _cd = json.load(f)
        saved_cam["position"] = np.array(_cd["position"], dtype=np.float64)
        saved_cam["look_at"] = np.array(_cd["look_at"], dtype=np.float64)
        saved_cam["fov"] = _cd.get("fov", np.pi / 3.0)
        print(f"Loaded saved camera from {cam_json_path}")

    rec = {
        "active": False,
        "renderer": None,
        "ffmpeg": None,
        "out_path": None,
        "count": 0,
        "cam_pos": None,
        "cam_look_at": None,
        "cam_fov": None,
    }

    def _render_rec_frame(fi: int, v_world: np.ndarray, j_world: np.ndarray) -> None:
        """Render one frame from the fixed camera into the ffmpeg pipe."""
        import pyrender
        import trimesh

        W, H = args.record_width, args.record_height
        body_color = np.array([0.55, 0.35, 0.90, 0.9])
        cam_up = np.array([0.0, -1.0, 0.0])

        scene = pyrender.Scene(bg_color=[0.12, 0.12, 0.14, 1.0],
                               ambient_light=[0.3, 0.3, 0.3])

        # Point cloud
        if pts_cam is not None:
            scene.add(_make_pc_mesh(pts_cam))

        # Body mesh
        body_tm = trimesh.Trimesh(vertices=v_world, faces=model.faces, process=False)
        body_tm.visual.vertex_colors = (body_color * 255).astype(np.uint8)
        scene.add(pyrender.Mesh.from_trimesh(body_tm, smooth=True))

        # Camera from fixed position
        cam_pos = rec["cam_pos"]
        look_at = rec["cam_look_at"]
        cam_fov = rec["cam_fov"]

        camera = pyrender.PerspectiveCamera(yfov=cam_fov, aspectRatio=W / H)
        forward = look_at - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        right = np.cross(forward, cam_up.astype(np.float64))
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)
        cam_pose = np.eye(4)
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = up
        cam_pose[:3, 2] = -forward
        cam_pose[:3, 3] = cam_pos
        scene.add(camera, pose=cam_pose)

        # Lights
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
        scene.add(light, pose=cam_pose)

        color, _ = rec["renderer"].render(scene)

        # Composite RGB inset
        if args.show_rgb and timeline_rgb[fi] is not None:
            rgb_path = Path(timeline_rgb[fi])
            if not rgb_path.is_absolute():
                rgb_path = session / rgb_path
            if rgb_path.exists():
                rgb_img = Image.open(rgb_path).convert("RGB")
                inset_h = H // 4
                ratio = inset_h / rgb_img.height
                inset_w = int(rgb_img.width * ratio)
                rgb_img = rgb_img.resize((inset_w, inset_h))
                rgb_arr = np.asarray(rgb_img)
                y0 = H - inset_h - 10
                x0 = W - inset_w - 10
                color[y0:y0+inset_h, x0:x0+inset_w] = rgb_arr

        rec["ffmpeg"].stdin.write(color.tobytes())
        rec["count"] += 1
        if rec["count"] % 100 == 0:
            print(f"  Recorded {rec['count']} frames")

    def _update(fi: int) -> None:
        if fi == state["last_frame"]:
            return
        state["last_frame"] = fi
        t_ns = int(timeline[fi])
        v_world, j_world = compute_pose(
            model, roshi_dict, roshi_trans, roshi_t, t_ns,
            betas, v_shaped, j_tpose,
        )
        if v_world is None:
            return
        state["last_j_world"] = j_world
        with server.atomic():
            mesh_handle.vertices = v_world
            label_handle.position = (
                float(j_world[0, 0]),
                float(v_world[:, 1].min()) - 0.15,
                float(j_world[0, 2]),
            )
            if rgb_handle is not None and timeline_rgb[fi] is not None:
                rgb_handle.image = _load_rgb(timeline_rgb[fi])
            status = f"Frame {fi}/{len(timeline)-1}"
            if rec["active"]:
                status += f"  [REC {rec['count']}]"
            info_text.value = status

        if rec["active"] and rec["renderer"] is not None and v_world is not None:
            _render_rec_frame(fi, v_world, j_world)

    @frame_slider.on_update
    def _(_) -> None:
        _update(int(frame_slider.value))

    @play_btn.on_click
    def _(_) -> None:
        state["playing"] = not state["playing"]

    @record_btn.on_click
    def _(_) -> None:
        if not rec["active"]:
            # Determine fixed camera: prefer saved_cam, else capture from current client
            if saved_cam["position"] is not None:
                rec["cam_pos"] = saved_cam["position"].copy()
                rec["cam_look_at"] = saved_cam["look_at"].copy()
                rec["cam_fov"] = saved_cam["fov"]
                print("Recording with saved camera pose from camera_pose.json")
            else:
                clients = server.get_clients()
                if not clients:
                    info_text.value = "No saved camera & no client connected! Save camera first."
                    return
                client = next(iter(clients.values()))
                rec["cam_pos"] = np.array(client.camera.position, dtype=np.float64).copy()
                rec["cam_look_at"] = np.array(client.camera.look_at, dtype=np.float64).copy()
                rec["cam_fov"] = float(client.camera.fov)
                print("Recording with current client camera (no saved camera found)")

            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
            import pyrender

            W, H = args.record_width, args.record_height
            out_path = session / "roshi_output.mp4"
            rec["renderer"] = pyrender.OffscreenRenderer(W, H)
            rec["out_path"] = out_path
            rec["count"] = 0
            ffmpeg_cmd = [
                "/usr/bin/ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{W}x{H}", "-pix_fmt", "rgb24",
                "-r", str(args.rate_hz),
                "-i", "-",
                "-c:v", "libx264", "-preset", "medium", "-crf", "12",
                "-b:v", "8M", "-minrate", "5M", "-maxrate", "15M", "-bufsize", "10M",
                "-pix_fmt", "yuv420p",
                str(out_path),
            ]
            rec["ffmpeg"] = subprocess.Popen(
                ffmpeg_cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            rec["active"] = True
            record_btn.name = "Stop Recording"
            info_text.value = "Recording started (fixed camera) — press Play to capture frames"
            print(f"Recording started → {out_path} ({W}x{H})")
        else:
            rec["active"] = False
            if rec["ffmpeg"]:
                rec["ffmpeg"].stdin.close()
                rec["ffmpeg"].wait()
                rec["ffmpeg"] = None
            if rec["renderer"]:
                rec["renderer"].delete()
                rec["renderer"] = None
            record_btn.name = "Start Recording"
            duration = rec["count"] / args.rate_hz
            info_text.value = (
                f"Saved {rec['count']} frames ({duration:.1f}s @ {args.rate_hz:.0f}fps) "
                f"→ {rec['out_path'].name}"
            )
            print(f"Recording saved → {rec['out_path']} "
                  f"({rec['count']} frames, {duration:.1f}s @ {args.rate_hz:.0f}fps)")

    @save_cam_btn.on_click
    def _(_) -> None:
        clients = server.get_clients()
        if not clients:
            info_text.value = "No browser client connected!"
            return
        client = next(iter(clients.values()))
        cam_pos = np.array(client.camera.position, dtype=np.float64)
        cam_look_at = np.array(client.camera.look_at, dtype=np.float64)
        cam_fov = float(client.camera.fov)

        fi = int(frame_slider.value)
        cam_data = {
            "position": cam_pos.tolist(),
            "look_at": cam_look_at.tolist(),
            "fov": cam_fov,
            "frame": fi,
        }
        out_json = session / "camera_pose.json"
        with open(out_json, "w") as f:
            json.dump(cam_data, f, indent=2)
        # Update in-memory saved camera so recording uses it immediately
        saved_cam["position"] = cam_pos.copy()
        saved_cam["look_at"] = cam_look_at.copy()
        saved_cam["fov"] = cam_fov
        info_text.value = f"Camera saved → {out_json.name} (frame={fi}, fov={np.degrees(cam_fov):.1f}°)"
        print(f"Camera pose saved → {out_json} (frame={fi})")

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


# ── Main ─────────────────────────────────────────────────────────────────────


def main(args: Args) -> None:
    session = args.session_dir.resolve()
    print(f"Session: {session}")

    model = load_smplx_model(args.smplx_model)
    betas = np.zeros(model.shapedirs.shape[2], dtype=np.float32)
    v_shaped, j_tpose = precompute_shape(model, betas)
    print(f"SMPLX: {model.num_joints} joints")

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

    timeline = np.array(roshi_t, dtype=np.int64)

    # Skip calibration
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

    # Point cloud
    R_zup_to_cam = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    pts_cam = None
    if args.show_pointcloud:
        cached = sorted(session.glob("**/slam/_cached_filtered_points.npz"))
        if cached:
            pts = np.load(cached[0])["points"]
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

    print(f"Timeline: {len(timeline)} frames (starting from frame {calib_skip})")

    common = (model, betas, v_shaped, j_tpose,
              roshi_dict, roshi_trans, roshi_t,
              timeline, calib_skip, pts_cam, timeline_rgb, session)

    if args.test_frame is not None:
        render_test_frame(args, *common)
    elif args.record:
        record_video(args, *common)
    else:
        run_interactive(args, *common)


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))
