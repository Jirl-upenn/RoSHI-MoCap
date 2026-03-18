from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import trimesh
import tyro
import viser

from egoallo import fncsmpl
from egoallo.data.aria_mps import load_point_cloud_and_find_ground
from egoallo.inference_utils import InferenceTrajectoryPaths
from egoallo.network import EgoDenoiseTraj
from egoallo.transforms import SE3, SO3

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

_LOCAL_PKGS = Path("/mnt/aloque_scratch/mwenjing/.local_pkgs")
if _LOCAL_PKGS.exists() and str(_LOCAL_PKGS) not in sys.path:
    sys.path.insert(0, str(_LOCAL_PKGS))

BAG_DIR = Path("evaluation/optitrack_gt_data/ros_bag")

try:
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    _ts = get_typestore(Stores.ROS1_NOETIC)
    _HAS_ROSBAGS = True
except ImportError:
    _HAS_ROSBAGS = False


# ── Inlined helpers (avoid hard dependency on eval_utils) ────────────

def _nearest_indices(source_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(source_ts, query_ts, side="left")
    idx = np.clip(idx, 0, source_ts.shape[0] - 1)
    prev = np.clip(idx - 1, 0, source_ts.shape[0] - 1)
    choose_prev = np.abs(query_ts - source_ts[prev]) <= np.abs(source_ts[idx] - query_ts)
    return np.where(choose_prev, prev, idx).astype(np.int64)


def _detect_yup(body_frames: list) -> bool:
    sample = body_frames[0]
    y_span = sample[:, 1].max() - sample[:, 1].min()
    z_span = sample[:, 2].max() - sample[:, 2].min()
    return y_span > z_span * 2


def _yup_pos_to_zup(pos: np.ndarray) -> np.ndarray:
    out = np.empty_like(pos)
    out[..., 0] = pos[..., 0]
    out[..., 1] = -pos[..., 2]
    out[..., 2] = pos[..., 1]
    return out


def _load_optitrack_bag(bag_path: Path):
    """Read OptiTrack ROS1 bag → (timestamps, body_list, box_list)."""
    body_msgs, box_msgs = {}, {}
    with Reader(bag_path) as reader:
        for conn, stamp, raw in reader.messages():
            msg = _ts.deserialize_ros1(raw, conn.msgtype)
            if "fullbody" in conn.topic:
                body_msgs[stamp] = np.array(
                    [[p.position.x, p.position.y, p.position.z] for p in msg.poses]
                )
            elif "camera_hand" not in conn.topic:
                p = msg.pose
                box_msgs[stamp] = (
                    np.array([p.position.x, p.position.y, p.position.z]),
                    np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]),
                )
    sorted_ts = sorted(body_msgs.keys())
    timestamps, body_list, box_list = [], [], []
    for t in sorted_ts:
        timestamps.append(t)
        body_list.append(body_msgs[t])
        if box_msgs:
            nearest = min(box_msgs, key=lambda bt: abs(bt - t))
            box_list.append(box_msgs[nearest])
    if body_list and _detect_yup(body_list):
        body_list = [_yup_pos_to_zup(b) for b in body_list]
        box_list = [(_yup_pos_to_zup(pos), quat) for pos, quat in box_list]
    return timestamps, body_list, box_list


def _tracking_ns_to_utc_ns(
    tracking_timestamps_ns: np.ndarray,
    trajectory_csv: Path,
) -> np.ndarray:
    """Convert Aria tracking timestamps to UTC via closed_loop_trajectory.csv."""
    import bisect
    import csv as csv_mod
    mapping: dict[int, int] = {}
    with open(trajectory_csv, newline="") as f:
        for row in csv_mod.DictReader(f):
            try:
                mapping[int(row["tracking_timestamp_us"])] = int(row["utc_timestamp_ns"])
            except (KeyError, ValueError):
                continue
    if not mapping:
        return tracking_timestamps_ns
    sorted_keys = sorted(mapping.keys())
    utc_list = []
    for t_ns in tracking_timestamps_ns:
        t_us = int(t_ns) // 1000
        idx = bisect.bisect_left(sorted_keys, t_us)
        if idx == 0:
            nearest_us = sorted_keys[0]
        elif idx >= len(sorted_keys):
            nearest_us = sorted_keys[-1]
        else:
            left, right = sorted_keys[idx - 1], sorted_keys[idx]
            nearest_us = left if abs(left - t_us) <= abs(right - t_us) else right
        utc_list.append(mapping[nearest_us])
    return np.array(utc_list, dtype=np.int64)


# ── Object geometry helpers (from visualize_viser_slam.py) ───────────

def _sphere_geometry(radius: float = 0.06, n_lat: int = 12, n_lon: int = 16):
    verts = []
    for i in range(n_lat + 1):
        theta = np.pi * i / n_lat
        for j in range(n_lon):
            phi = 2.0 * np.pi * j / n_lon
            verts.append([
                radius * np.sin(theta) * np.cos(phi),
                radius * np.sin(theta) * np.sin(phi),
                radius * np.cos(theta),
            ])
    verts = np.array(verts, dtype=np.float32)
    faces = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            c = (i + 1) * n_lon + (j + 1) % n_lon
            d = (i + 1) * n_lon + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    return verts, np.array(faces, dtype=np.uint32)


def _cube_geometry(half_extent: float = 0.08):
    """Centered cube (legacy). Prefer _cube_geometry_floor_to_top for object viz."""
    h = half_extent
    verts = np.array([
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
        [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ], dtype=np.uint32)
    return verts, faces


def _cube_geometry_floor_to_top(
    half_extent_xy: float = 0.4,
    height: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Cube from z=0 to z=height (detected position is top face). Larger horizontal size.

    Local: x,y in [-half_extent_xy, half_extent_xy], z in [0, height].
    Place at (bp[0], bp[1], floor_z) so bottom is on floor and top at bp[2].
    """
    h = half_extent_xy
    verts = np.array([
        [-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0],
        [-h, -h, height], [h, -h, height], [h, h, height], [-h, h, height],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ], dtype=np.uint32)
    return verts, faces


def _stick_between_points(p0: np.ndarray, p1: np.ndarray, radius: float = 0.02, n_seg: int = 8):
    axis = np.array(p1, dtype=np.float64) - np.array(p0, dtype=np.float64)
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        v, f = _sphere_geometry(radius=radius * 2)
        return v.astype(np.float32) + np.array(p0, dtype=np.float32), f
    axis = axis / length
    center = (np.array(p0, dtype=np.float64) + np.array(p1, dtype=np.float64)) / 2
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u = u / (np.linalg.norm(u) + 1e-9)
    v_dir = np.cross(axis, u)
    v_dir = v_dir / (np.linalg.norm(v_dir) + 1e-9)
    R = np.column_stack([u, v_dir, axis])
    verts = []
    for i in range(n_seg):
        theta = 2.0 * np.pi * i / n_seg
        for z in (-length / 2, length / 2):
            pt = np.array([radius * np.cos(theta), radius * np.sin(theta), z])
            pt = R @ pt + center
            verts.append(pt)
    verts.append(center - axis * (length / 2))
    verts.append(center + axis * (length / 2))
    verts = np.array(verts, dtype=np.float32)
    faces = []
    n = n_seg
    for i in range(n):
        a, b = 2 * i, 2 * ((i + 1) % n)
        faces.append([a, b, b + 1])
        faces.append([a, b + 1, a + 1])
        faces.append([2 * n, b, a])
        faces.append([2 * n + 1, a + 1, b + 1])
    return verts, np.array(faces, dtype=np.uint32)


def _ellipse_geometry_oriented(
    center: np.ndarray,
    stick_direction: np.ndarray,
    semi_major: float = 0.18,
    semi_minor: float = 0.12,
    thickness: float = 0.02,
    n_seg: int = 16,
):
    axis = np.array(stick_direction, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    short_ax = np.cross(axis, ref)
    short_ax = short_ax / (np.linalg.norm(short_ax) + 1e-9)
    thick_ax = np.cross(axis, short_ax)
    thick_ax = thick_ax / (np.linalg.norm(thick_ax) + 1e-9)
    verts = []
    for i in range(n_seg):
        theta = 2.0 * np.pi * i / n_seg
        x = semi_major * np.cos(theta)
        y = semi_minor * np.sin(theta)
        verts.append(center + axis * x + short_ax * y + thick_ax * (thickness / 2))
        verts.append(center + axis * x + short_ax * y - thick_ax * (thickness / 2))
    verts.append(center + thick_ax * (thickness / 2))
    verts.append(center - thick_ax * (thickness / 2))
    verts = np.array(verts, dtype=np.float32)
    faces = []
    n = n_seg
    for i in range(n):
        a, b = 2 * i, 2 * ((i + 1) % n)
        faces.append([a, b, b + 1])
        faces.append([a, b + 1, a + 1])
        faces.append([2 * n, 2 * i, 2 * ((i + 1) % n)])
        faces.append([2 * n + 1, 2 * ((i + 1) % n) + 1, 2 * i + 1])
    return verts, np.array(faces, dtype=np.uint32)


SMPLH_RIGHT_WRIST_JOINT_IDX = 20

OBJ_COLOR = (0.95, 0.75, 0.10)
OBJ_OPACITY = 0.85
# Cube horizontal half-size (x/y). Height is from floor to detected top face.
CUBE_HALF_EXTENT_XY = 0.15


def _xy_procrustes(A: np.ndarray, B: np.ndarray):
    """XY Procrustes: find yaw-only R (3x3) + t (3,) so that B ≈ R @ A + t.

    Both A and B are (N, 3) in Z-up frames. Only the XY rotation (yaw)
    is estimated; Z is left unchanged.
    Returns (R_3x3, t_3).
    """
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


def load_box_positions_in_aria_frame(
    dataset_name: str,
    timestamps_ns: np.ndarray,
    pelvis_positions_aria: np.ndarray,
    trajectory_csv: Optional[Path],
) -> Optional[np.ndarray]:
    """Load object positions from OptiTrack bag and transform to Aria frame.

    The OptiTrack bag has object + body data in OptiTrack world frame.
    We align the OptiTrack pelvis trajectory to the EgoAllo pelvis trajectory
    (in Aria SLAM frame) via XY Procrustes, then apply the *inverse*
    transform to bring the object positions into Aria frame.
    """
    if not _HAS_ROSBAGS:
        print("  rosbags not installed, skipping object loading")
        return None

    bag_name = re.sub(r"dataset(\d+)", r"dataset_\1", dataset_name) + ".bag"
    bag_path = _REPO_ROOT / BAG_DIR / bag_name
    if not bag_path.exists():
        print(f"  Bag file not found: {bag_path}")
        return None

    print(f"  Loading object data from {bag_path}")
    opti_timestamps, opti_body, opti_box = _load_optitrack_bag(bag_path)
    if not opti_box:
        print(f"  No object tracked in bag for {dataset_name}")
        return None

    opti_ts = np.array(opti_timestamps, dtype=np.int64)
    opti_pelvis = np.array([b[0] for b in opti_body], dtype=np.float64)
    box_pos_all = np.array([b[0] for b in opti_box], dtype=np.float32)

    # Convert EgoAllo tracking timestamps to UTC for time-alignment
    if trajectory_csv is not None and trajectory_csv.exists():
        ego_ts_utc = _tracking_ns_to_utc_ns(timestamps_ns, trajectory_csv)
        print(f"  Converted {len(ego_ts_utc)} tracking timestamps to UTC")
    else:
        ego_ts_utc = timestamps_ns
        print("  No trajectory CSV; using raw timestamps for alignment")

    # Match EgoAllo frames to OptiTrack timestamps
    ego_idx_for_opti = _nearest_indices(ego_ts_utc, opti_ts)
    ego_pelvis_matched = pelvis_positions_aria[ego_idx_for_opti].astype(np.float64)

    # XY Procrustes: p_opti = R_align @ p_aria + t_align
    R_align, t_align = _xy_procrustes(ego_pelvis_matched, opti_pelvis)
    yaw_deg = float(np.degrees(np.arctan2(R_align[1, 0], R_align[0, 0])))
    print(f"  Alignment: yaw={yaw_deg:.1f}deg, "
          f"t=[{t_align[0]:.3f}, {t_align[1]:.3f}, {t_align[2]:.3f}]")

    # Inverse: p_aria = R_align^T @ (p_opti - t_align)
    R_inv = R_align.T
    t_inv = -R_inv @ t_align

    # Align box positions to EgoAllo timestamps and transform to Aria frame
    box_idx = _nearest_indices(opti_ts, ego_ts_utc)
    box_at_ego_times = box_pos_all[box_idx].astype(np.float64)
    box_aria = (R_inv @ box_at_ego_times.T).T + t_inv

    return box_aria.astype(np.float32)


def main(
    search_root_dir: Path,
    smplh_npz_path: Path = Path("../model/smplh/neutral/model.npz"),
) -> None:
    """Visualization script for outputs from EgoAllo.

    Arguments:
        search_root_dir: Root directory where inputs/outputs are stored. All
            NPZ files in this directory will be assumed to be outputs from EgoAllo.
        smplh_npz_path: Path to the SMPLH model NPZ file.
    """
    device = torch.device("cuda")

    body_model = fncsmpl.SmplhModel.load(smplh_npz_path).to(device)

    server = viser.ViserServer()
    server.gui.configure_theme(dark_mode=True)
    server.scene.set_up_direction("+z")

    # Default camera: elevated view, ground horizontal (no sliding)
    _DEFAULT_CAM_POSITION = (-1.2, -1.5, 1.0)
    _DEFAULT_CAM_WXYZ = (0.72, -0.69, 0.0, 0.0)

    @server.on_client_connect
    def _on_connect(client: viser.ClientHandle) -> None:
        client.camera.position = _DEFAULT_CAM_POSITION
        client.camera.wxyz = _DEFAULT_CAM_WXYZ
        client.camera.up_direction = (0.0, 0.0, 1.0)

    def get_file_list():
        return ["None"] + sorted(
            str(p.relative_to(search_root_dir))
            for p in search_root_dir.glob("**/egoallo_outputs/*.npz")
        )

    options = get_file_list()
    file_dropdown = server.gui.add_dropdown("File", options=options)

    refresh_file_list = server.gui.add_button("Refresh File List")

    @refresh_file_list.on_click
    def _(_) -> None:
        file_dropdown.options = get_file_list()

    trajectory_folder = server.gui.add_folder("Trajectory")

    current_file = "None"
    loop_cb = lambda: None

    while True:
        loop_cb()
        if current_file != file_dropdown.value:
            current_file = file_dropdown.value

            # Clear the scene.
            server.scene.reset()

            if current_file != "None":
                trajectory_folder.remove()
                trajectory_folder = server.gui.add_folder("Trajectory")

                with trajectory_folder:
                    npz_path = Path(search_root_dir / current_file).resolve()
                    loop_cb = load_and_visualize(
                        server,
                        npz_path,
                        body_model,
                        device=device,
                    )
                    args = npz_path.parent / (npz_path.stem + "_args.yaml")
                    if args.exists():
                        with server.gui.add_folder("Args"):
                            server.gui.add_markdown(
                                "```\n" + args.read_text() + "\n```"
                            )


NUM_MESHES = 1

MESH_COLOR = (140, 184, 235)
MESH_OPACITY = 0.72


def load_and_visualize(
    server: viser.ViserServer,
    npz_path: Path,
    body_model: fncsmpl.SmplhModel,
    device: torch.device,
) -> Callable[[], int]:
    outputs = np.load(npz_path)
    expected_keys = [
        "Ts_world_cpf",
        "Ts_world_root",
        "body_quats",
        "left_hand_quats",
        "right_hand_quats",
        "betas",
        "frame_nums",
        "timestamps_ns",
    ]
    assert all(key in outputs for key in expected_keys), (
        f"Missing keys in NPZ file. Expected: {expected_keys}, Found: {list(outputs.keys())}"
    )
    (num_samples, timesteps, _, _) = outputs["body_quats"].shape

    traj_dir = npz_path.resolve().parent.parent
    paths = InferenceTrajectoryPaths.find(traj_dir)

    # Get point cloud + floor.
    points_data, floor_z = load_point_cloud_and_find_ground(
        paths.points_path, "filtered"
    )

    traj = EgoDenoiseTraj(
        betas=torch.from_numpy(outputs["betas"]).to(device),
        body_rotmats=SO3(
            torch.from_numpy(outputs["body_quats"]),
        )
        .as_matrix()
        .to(device),
        contacts=torch.zeros((num_samples, timesteps, 21), device=device)
        if "contacts" not in outputs
        else torch.from_numpy(outputs["contacts"]).to(device),
        hand_rotmats=SO3(
            torch.from_numpy(
                np.concatenate(
                    [
                        outputs["left_hand_quats"],
                        outputs["right_hand_quats"],
                    ],
                    axis=-2,
                )
            ).to(device)
        ).as_matrix(),
    )
    Ts_world_cpf = torch.from_numpy(outputs["Ts_world_cpf"]).to(device)

    # --- Build FK outputs (same logic as vis_helpers) ---
    betas = traj.betas
    body_quats = SO3.from_matrix(traj.body_rotmats).wxyz
    if traj.hand_rotmats is not None:
        hand_quats = SO3.from_matrix(traj.hand_rotmats).wxyz
        left_hand_quats = hand_quats[..., :15, :]
        right_hand_quats = hand_quats[..., 15:30, :]
    else:
        left_hand_quats = None
        right_hand_quats = None

    shaped = body_model.with_shape(torch.mean(betas, dim=1, keepdim=True))
    fk_outputs = shaped.with_pose_decomposed(
        T_world_root=SE3.identity(
            device=device, dtype=body_quats.dtype
        ).parameters(),
        body_quats=body_quats,
        left_hand_quats=left_hand_quats,
        right_hand_quats=right_hand_quats,
    )
    from egoallo import fncsmpl_extensions
    T_world_root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(
        fk_outputs, Ts_world_cpf[None, ...],
    )
    fk_outputs = fk_outputs.with_new_T_world_root(T_world_root)

    # --- Load object positions from OptiTrack bag (transformed to Aria frame) ---
    dataset_name = traj_dir.name
    timestamps_ns = outputs["timestamps_ns"]
    sample_idx = 0
    pelvis_positions_aria = fk_outputs.T_world_root[sample_idx, :, 4:7].numpy(force=True)
    traj_csv = paths.closed_loop_trajectory_csv
    if isinstance(traj_csv, tuple):
        traj_csv = traj_csv[0]
    box_positions = load_box_positions_in_aria_frame(
        dataset_name,
        timestamps_ns,
        pelvis_positions_aria,
        traj_csv,
    )
    if box_positions is not None:
        print(f"Loaded {len(box_positions)} object positions for {dataset_name}")
    else:
        print(f"No object data for {dataset_name}")

    # --- Scene setup ---
    server.scene.add_grid(
        "/ground",
        width=1000,
        height=5,
        plane="xy",
        cell_color=(210, 215, 220),
        cell_thickness=0.8,
        section_color=(180, 185, 190),
        section_thickness=1.2,
        position=(0.0, 0.0, floor_z),
    )

    # World coordinate frame (Viser: X=red, Y=green, Z=blue)
    server.scene.add_frame(
        "/world_frame",
        show_axes=True,
        axes_length=0.6,
        axes_radius=0.02,
        position=(0.0, 0.0, floor_z),
        wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    if points_data is not None:
        server.scene.add_point_cloud(
            "/aria_points",
            points=points_data,
            colors=np.cos(points_data + np.arange(3)) / 3.0 + 0.7,
            point_size=0.005,
            point_shape="sparkle",
        )

    glasses_mesh = trimesh.load("./data/glasses.stl")
    assert isinstance(glasses_mesh, trimesh.Trimesh)
    glasses_mesh.visual.face_colors = [10, 20, 20, 255]  # type: ignore

    # --- Create NUM_MESHES skinned meshes + per-mesh glasses ---
    body_handles = []
    glasses_handles = []
    frame_sliders = []

    default_frames = [min(max(1405, 0), timesteps - 1)]

    for mi in range(NUM_MESHES):
        body_handle = server.scene.add_mesh_skinned(
            f"/mesh_{mi}",
            vertices=shaped.verts_zero[sample_idx, 0, :, :].numpy(force=True),
            faces=body_model.faces.numpy(force=True),
            bone_wxyzs=viser.transforms.SO3.identity(
                batch_axes=(body_model.get_num_joints() + 1,)
            ).wxyz,
            bone_positions=np.concatenate(
                [
                    np.zeros((1, 3)),
                    shaped.joints_zero[sample_idx, :, :, :]
                    .numpy(force=True)
                    .squeeze(axis=0),
                ],
                axis=0,
            ),
            color=MESH_COLOR,
            skin_weights=body_model.weights.numpy(force=True),
        )
        body_handle.opacity = MESH_OPACITY
        body_handles.append(body_handle)

        cpf_frame = server.scene.add_frame(
            f"/cpf_{mi}",
            show_axes=False,
        )
        server.scene.add_mesh_trimesh(
            f"/cpf_{mi}/glasses", glasses_mesh, scale=0.001 * 1.05
        )
        glasses_handles.append(cpf_frame)

    Ts_world_cpf_np = Ts_world_cpf.numpy(force=True)

    # --- IMU sensor cubes (9 sensors on body links) ---
    # SMPL-H joint IDs: 0=pelvis(root), 1=L_hip, 2=R_hip, 3=spine1,
    #   4=L_knee, 5=R_knee, 7=L_ankle, 8=R_ankle,
    #   16=L_shoulder, 17=R_shoulder, 18=L_elbow, 19=R_elbow,
    #   20=L_wrist, 21=R_wrist (note: 0 is root, stored separately)
    # Each entry: (label, smpl_joint_a, smpl_joint_b, skin_offset)
    IMU_PLACEMENTS = [
        ("pelvis",       0,  0,  0.0),   # centered at pelvis (was 0.14, deviated)
        ("L_shoulder",  16, 18,  0.06),  # upper arm
        ("R_shoulder",  17, 19,  0.06),  # upper arm
        ("L_elbow",     18, 20,  0.05),  # forearm
        ("R_elbow",     19, 21,  0.05),  # forearm
        ("L_thigh",      1,  4,  0.09),  # hip → knee
        ("R_thigh",      2,  5,  0.09),  # hip → knee
        ("L_shin",       4,  7,  0.06),  # knee → ankle
        ("R_shin",       5,  8,  0.06),  # knee → ankle
    ]
    IMU_HALF_W = 0.02       # width (across bone)
    IMU_HALF_H = 0.025      # height (along bone)
    IMU_HALF_T = 0.007      # thickness (radial, thin against skin)
    IMU_COLOR = (0.15, 0.15, 0.15)
    IMU_OPACITY = 0.95

    # Flat rectangular box: 8 corners, same face topology as _cube_geometry
    _imu_local_corners = np.array([
        [-IMU_HALF_W, -IMU_HALF_H, -IMU_HALF_T],
        [ IMU_HALF_W, -IMU_HALF_H, -IMU_HALF_T],
        [ IMU_HALF_W,  IMU_HALF_H, -IMU_HALF_T],
        [-IMU_HALF_W,  IMU_HALF_H, -IMU_HALF_T],
        [-IMU_HALF_W, -IMU_HALF_H,  IMU_HALF_T],
        [ IMU_HALF_W, -IMU_HALF_H,  IMU_HALF_T],
        [ IMU_HALF_W,  IMU_HALF_H,  IMU_HALF_T],
        [-IMU_HALF_W,  IMU_HALF_H,  IMU_HALF_T],
    ], dtype=np.float32)
    _imu_faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ], dtype=np.uint32)

    imu_handles: list[list] = []
    for mi in range(NUM_MESHES):
        mesh_imu_handles = []
        for ii, (label, _, _, _) in enumerate(IMU_PLACEMENTS):
            h = server.scene.add_mesh_simple(
                f"/mesh_{mi}/imu_{ii}_{label}",
                vertices=_imu_local_corners.copy(),
                faces=_imu_faces,
                color=IMU_COLOR,
                wireframe=False,
                opacity=IMU_OPACITY,
            )
            mesh_imu_handles.append(h)
        imu_handles.append(mesh_imu_handles)

    # IMU axis frames (X,Y,Z per sensor)
    imu_axis_frames: list[list] = []
    for mi in range(NUM_MESHES):
        mesh_axis_frames = []
        for ii, (label, _, _, _) in enumerate(IMU_PLACEMENTS):
            f = server.scene.add_frame(
                f"/mesh_{mi}/imu_axes_{ii}_{label}",
                show_axes=True,
                axes_length=0.08,
                axes_radius=0.006,
                position=(0.0, 0.0, 0.0),
                wxyz=(1.0, 0.0, 0.0, 0.0),
            )
            mesh_axis_frames.append(f)
        imu_axis_frames.append(mesh_axis_frames)

    def _get_smpl_joint_pos(frame: int, smpl_joint_id: int) -> np.ndarray:
        if smpl_joint_id == 0:
            return fk_outputs.T_world_root[sample_idx, frame, 4:7].numpy(force=True)
        return fk_outputs.Ts_world_joint[
            sample_idx, frame, smpl_joint_id - 1, 4:7
        ].numpy(force=True)

    def _oriented_imu_verts(
        position: np.ndarray,
        outward: np.ndarray,
        bone_dir: np.ndarray,
        R_extra: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build oriented flat-box vertices: thin axis = outward, tall axis = bone.
        R_extra: optional extra rotation (3x3) in IMU local frame (Euler XYZ applied before base orient).
        Returns (verts, R) where R is the 3x3 rotation matrix (world from local)."""
        t_ax = outward / (np.linalg.norm(outward) + 1e-9)
        h_ax = bone_dir / (np.linalg.norm(bone_dir) + 1e-9)
        h_ax = h_ax - np.dot(h_ax, t_ax) * t_ax
        n = np.linalg.norm(h_ax)
        if n < 1e-6:
            h_ax = np.array([0.0, 0.0, 1.0])
        else:
            h_ax = h_ax / n
        w_ax = np.cross(h_ax, t_ax)
        w_ax = w_ax / (np.linalg.norm(w_ax) + 1e-9)
        R = np.column_stack([w_ax, h_ax, t_ax]).astype(np.float64)
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
        if R_extra is not None:
            R = R @ R_extra.astype(np.float64)
            if np.linalg.det(R) < 0:
                R[:, 0] *= -1
        verts = (_imu_local_corners.astype(np.float64) @ R.T).astype(np.float32) + position[None, :].astype(np.float32)
        return verts, R

    def _update_imus(mesh_idx: int, frame: int) -> None:
        root_pos = _get_smpl_joint_pos(frame, 0)
        spine_pos = _get_smpl_joint_pos(frame, 3)
        head_pos = _get_smpl_joint_pos(frame, 15)

        for ii, (label, ja, jb, offset) in enumerate(IMU_PLACEMENTS):
            pa = _get_smpl_joint_pos(frame, ja)
            pb = _get_smpl_joint_pos(frame, jb)
            mid = (pa + pb) / 2.0

            if label == "pelvis":
                # Centered at pelvis (offset 0). Use pelvis_imu_offset for fine-tuning.
                spine_dir = head_pos - root_pos
                spine_dir[2] = 0.0
                n = np.linalg.norm(spine_dir)
                if n < 1e-6:
                    spine_dir = np.array([1.0, 0.0, 0.0])
                else:
                    spine_dir = spine_dir / n
                outward = spine_dir
                bone_dir = np.array([0.0, 0.0, 1.0])
            else:
                bone_dir = pb - pa
                bone_len = np.linalg.norm(bone_dir)
                if bone_len < 1e-6:
                    bone_dir = np.array([0.0, 0.0, 1.0])
                else:
                    bone_dir = bone_dir / bone_len
                outward = mid - root_pos
                outward = outward - np.dot(outward, bone_dir) * bone_dir
                n = np.linalg.norm(outward)
                if n < 1e-6:
                    outward = np.array([0.0, 1.0, 0.0])
                else:
                    outward = outward / n

            skin_pos = mid + outward * offset + imu_offsets[ii]
            rx, ry, rz = imu_rotations[ii]
            R_extra = (
                Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
                if (rx != 0 or ry != 0 or rz != 0)
                else None
            )
            v, R = _oriented_imu_verts(skin_pos, outward, bone_dir, R_extra)
            imu_handles[mesh_idx][ii].vertices = v
            # Update IMU axis frame (X,Y,Z)
            q_xyzw = Rotation.from_matrix(R).as_quat()
            wxyz = (float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))
            imu_axis_frames[mesh_idx][ii].position = tuple(skin_pos.astype(float).tolist())
            imu_axis_frames[mesh_idx][ii].wxyz = wxyz

    # --- Object state ---
    obj_geometry: dict = {}
    obj_geometry["verts"], obj_geometry["faces"] = _sphere_geometry()
    obj_handles: list = []
    ellipse_handles: list = []

    def _get_wrist_pos(frame: int) -> np.ndarray:
        return fk_outputs.Ts_world_joint[
            sample_idx, frame, SMPLH_RIGHT_WRIST_JOINT_IDX, 4:7
        ].numpy(force=True)

    def _create_object_mesh(mesh_idx: int, frame: int) -> None:
        if box_positions is None:
            return
        bp = box_positions[frame]
        obj_shape = object_dropdown.value

        if obj_shape == "Stick":
            p0 = _get_wrist_pos(frame)
            p1 = bp
            v_stick, f_stick = _stick_between_points(p0, p1)
            bh = server.scene.add_mesh_simple(
                f"/obj_{mesh_idx}",
                vertices=v_stick, faces=f_stick,
                color=OBJ_COLOR, wireframe=False, opacity=OBJ_OPACITY,
            )
            stick_dir = p1 - p0
            stick_dir = stick_dir / (np.linalg.norm(stick_dir) + 1e-9)
            v_ell, f_ell = _ellipse_geometry_oriented(center=p1, stick_direction=stick_dir)
            eh = server.scene.add_mesh_simple(
                f"/ellipse_{mesh_idx}",
                vertices=v_ell, faces=f_ell,
                color=OBJ_COLOR, wireframe=False, opacity=OBJ_OPACITY,
            )
            obj_handles.append(bh)
            ellipse_handles.append(eh)
        else:
            if obj_shape == "Cube":
                ref_bp = box_positions[frame_sliders[0].value]
                height = max(ref_bp[2] - floor_z, 0.01)
                v_template, f_template = _cube_geometry_floor_to_top(CUBE_HALF_EXTENT_XY, height)
                # Same shape for all; place so top face is at detected position bp
                v_obj = v_template + np.array([bp[0], bp[1], bp[2] - height], dtype=np.float32)
            else:
                v_template = obj_geometry["verts"]
                f_template = obj_geometry["faces"]
                v_obj = v_template.copy() + bp[None, :]
            bh = server.scene.add_mesh_simple(
                f"/obj_{mesh_idx}",
                vertices=v_obj, faces=f_template,
                color=OBJ_COLOR, wireframe=False, opacity=OBJ_OPACITY,
            )
            obj_handles.append(bh)

    def _recreate_all_objects() -> None:
        for h in obj_handles:
            h.remove()
        obj_handles.clear()
        for h in ellipse_handles:
            h.remove()
        ellipse_handles.clear()
        if box_positions is None:
            return
        val = object_dropdown.value
        if val == "Cube":
            _, obj_geometry["faces"] = _cube_geometry_floor_to_top(CUBE_HALF_EXTENT_XY, 0.2)
            obj_geometry["verts"] = None  # computed per frame from floor to top
        elif val != "Stick":
            obj_geometry["verts"], obj_geometry["faces"] = _sphere_geometry()
        for mi in range(NUM_MESHES):
            _create_object_mesh(mi, frame_sliders[mi].value)

    def _update_object(mesh_idx: int, frame: int) -> None:
        if box_positions is None:
            return
        bp = box_positions[frame]
        obj_shape = object_dropdown.value

        si = mesh_idx
        if obj_shape == "Stick":
            p0 = _get_wrist_pos(frame)
            p1 = bp
            if si < len(obj_handles):
                v_stick, _ = _stick_between_points(p0, p1)
                obj_handles[si].vertices = v_stick.astype(np.float32)
            if si < len(ellipse_handles):
                stick_dir = p1 - p0
                stick_dir = stick_dir / (np.linalg.norm(stick_dir) + 1e-9)
                v_ell, _ = _ellipse_geometry_oriented(center=p1, stick_direction=stick_dir)
                ellipse_handles[si].vertices = v_ell.astype(np.float32)
        else:
            if si < len(obj_handles):
                if obj_shape == "Cube":
                    ref_bp = box_positions[frame_sliders[0].value]
                    height = max(ref_bp[2] - floor_z, 0.01)
                    v_template, _ = _cube_geometry_floor_to_top(CUBE_HALF_EXTENT_XY, height)
                    # Same shape for all; place so top face is at detected position bp
                    v_obj = v_template + np.array([bp[0], bp[1], bp[2] - height], dtype=np.float32)
                else:
                    v_template = obj_geometry["verts"]
                    v_obj = v_template.copy() + bp[None, :]
                obj_handles[si].vertices = v_obj.astype(np.float32)

    def update_mesh(mesh_idx: int, frame: int) -> None:
        handle = body_handles[mesh_idx]
        for b, bone_handle in enumerate(handle.bones):
            if b == 0:
                bone_transform = fk_outputs.T_world_root[sample_idx, frame].numpy(force=True)
            else:
                bone_transform = fk_outputs.Ts_world_joint[sample_idx, frame, b - 1].numpy(force=True)
            bone_handle.wxyz = bone_transform[:4]
            bone_handle.position = bone_transform[4:7]
        glasses_handles[mesh_idx].wxyz = Ts_world_cpf_np[frame, :4]
        glasses_handles[mesh_idx].position = Ts_world_cpf_np[frame, 4:7]
        _update_imus(mesh_idx, frame)
        _update_object(mesh_idx, frame)

    # --- GUI: per-mesh frame sliders ---
    with server.gui.add_folder("Mesh Frames"):
        for mi in range(NUM_MESHES):
            slider = server.gui.add_slider(
                f"Mesh {mi + 1} frame",
                min=0,
                max=timesteps - 1,
                step=1,
                initial_value=default_frames[mi],
            )
            frame_sliders.append(slider)

            mesh_idx = mi
            @slider.on_update
            def _(_, _mi=mesh_idx) -> None:
                update_mesh(_mi, frame_sliders[_mi].value)

    # --- GUI: object shape ---
    with server.gui.add_folder("Object"):
        object_dropdown = server.gui.add_dropdown(
            "Shape",
            options=["Ball", "Cube", "Stick"],
            initial_value="Ball",
        )

    @object_dropdown.on_update
    def _(_) -> None:
        _recreate_all_objects()

    # --- IMU adjustments: position (x,y,z) + rotation (rx,ry,rz deg) for all 9 sensors ---
    NUM_IMUS = 9
    imu_offsets = np.zeros((NUM_IMUS, 3), dtype=np.float32)
    imu_rotations = np.zeros((NUM_IMUS, 3), dtype=np.float32)
    imu_sliders: list[list] = []

    def _sync_imu_adjustments() -> None:
        for ii in range(NUM_IMUS):
            for j in range(3):
                imu_offsets[ii, j] = float(imu_sliders[ii][j].value)
            for j in range(3):
                imu_rotations[ii, j] = float(imu_sliders[ii][3 + j].value)

    def _on_imu_adjustment_change(_) -> None:
        _sync_imu_adjustments()
        for mi in range(NUM_MESHES):
            update_mesh(mi, frame_sliders[mi].value)

    with server.gui.add_folder("IMU adjustments"):
        for ii, (label, _, _, _) in enumerate(IMU_PLACEMENTS):
            with server.gui.add_folder(label):
                sliders = [
                    server.gui.add_slider(
                        "Offset X (m)", min=-0.15, max=0.15, step=0.005, initial_value=0.0
                    ),
                    server.gui.add_slider(
                        "Offset Y (m)", min=-0.15, max=0.15, step=0.005, initial_value=0.0
                    ),
                    server.gui.add_slider(
                        "Offset Z (m)", min=-0.15, max=0.15, step=0.005, initial_value=0.0
                    ),
                    server.gui.add_slider(
                        "Rot X (°)", min=-45, max=45, step=1.0, initial_value=0.0
                    ),
                    server.gui.add_slider(
                        "Rot Y (°)", min=-45, max=45, step=1.0, initial_value=0.0
                    ),
                    server.gui.add_slider(
                        "Rot Z (°)", min=-45, max=45, step=1.0, initial_value=0.0
                    ),
                ]
                for s in sliders:
                    s.on_update(_on_imu_adjustment_change)
                imu_sliders.append(sliders)

    # --- GUI: shared controls ---
    gui_show_imus = server.gui.add_checkbox("Show IMU sensors", initial_value=True)
    gui_show_imu_axes = server.gui.add_checkbox("Show IMU axes (X,Y,Z)", initial_value=True)

    @gui_show_imus.on_update
    def _(_) -> None:
        for mesh_imus in imu_handles:
            for h in mesh_imus:
                h.visible = gui_show_imus.value

    @gui_show_imu_axes.on_update
    def _(_) -> None:
        for mesh_axes in imu_axis_frames:
            for f in mesh_axes:
                f.visible = gui_show_imu_axes.value

    gui_show_body = server.gui.add_checkbox("Show bodies", initial_value=True)
    gui_wireframe = server.gui.add_checkbox("Wireframe", initial_value=False)
    gui_smpl_opacity = server.gui.add_slider(
        "SMPL Opacity", initial_value=MESH_OPACITY, min=0.0, max=1.0, step=0.01
    )

    @gui_show_body.on_update
    def _(_) -> None:
        for h in body_handles:
            h.visible = gui_show_body.value

    @gui_wireframe.on_update
    def _(_) -> None:
        for h in body_handles:
            h.wireframe = gui_wireframe.value

    @gui_smpl_opacity.on_update
    def _(_) -> None:
        for h in body_handles:
            h.opacity = gui_smpl_opacity.value

    # Initial pose and objects for all meshes.
    for mi in range(NUM_MESHES):
        update_mesh(mi, default_frames[mi])
    if box_positions is not None:
        _recreate_all_objects()

    def loop_cb() -> None:
        pass

    return loop_cb


if __name__ == "__main__":
    tyro.cli(main)
