"""IMU Transformer inference: an alternative to 04_inference.py (EgoAllo diffusion).

A single forward pass regresses SMPL-H body poses from the Aria CPF trajectory
and the nine calibrated RoSHI bone orientations at EVERY frame, followed by the
same JAX post-guidance optimizer used by 04_inference.py. Output files have the
same format as 04_inference.py, so 05_visualize_roshi.py and 06_evaluate.py work.

Two input modes
---------------
1) RoSHI recording (after 03_sync.py), same layout as 04_inference.py:
       python src/pipeline/04b_transformer_inference.py --traj-root <recording_dir>
   Uses the .vrs, mps_*/slam/ (closed_loop_trajectory.csv, semidense points for
   the floor) and sync/imu_info.pkl.

2) Any other data, as one .npz (see `load_input_npz` for the exact schema):
       python src/pipeline/04b_transformer_inference.py --input-npz seq.npz
   `--save-inputs` in mode (1) writes exactly this file, as a worked example.

How the model input is built (mode 1)
-------------------------------------
  a. Aria closed-loop poses -> CPF poses at 30 Hz (InferenceInputTransforms).
  b. Floor height from the MPS semidense point cloud.
  c. Each CPF tracking timestamp -> UTC with the VRS time-sync (as 04_inference.py).
  d. imu_info.pkl holds calibrated bone rotations R_world_bone in a common IMU
     world frame, keyed by UTC, in RoSHI IMU-id order (1..9). Complete 9-IMU
     groups are reordered to the transformer's sensor order and SLERP-
     interpolated to every CPF UTC time. Frames outside IMU coverage or across
     gaps > --max-imu-gap-ms are never extrapolated: the longest fully
     covered contiguous interval is used, and the trim is printed.
  e. Frame 0 of the interval is context only; outputs cover frames 1..N-1.
"""
from __future__ import annotations

import dataclasses
import json
import pickle
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from egoallo import fncsmpl, fncsmpl_extensions
from egoallo.guidance_optimizer_jax import JaxGuidanceParams, do_guidance_optimization
from egoallo.imu_detection_structs import CorrespondedImuReadings, IMUPoseWrtCanonical
from egoallo.imu_transformer import (
    IMU_ID_FOR_TRANSFORMER_SENSOR,
    TRANSFORMER_BONE_NAMES,
    TRANSFORMER_SMPL_JOINT_INDICES,
    load_transformer,
    predict_sequence,
)
from egoallo.transforms import SO3

# Post-guidance weights used for the reported IMU-Transformer results. They differ
# from the 04_inference.py defaults in the IMU weights (10 vs 5) and joint limit (0 vs 50).
TRANSFORMER_GUIDANCE_OVERRIDES = dict(
    imu_local_quat_weight=10.0,
    imu_pelvis_relative_rotation_weight=10.0,
    joint_limit_weight=0.0,
)
# Inverse of IMU_ID_FOR_TRANSFORMER_SENSOR: guidance slot (IMU id - 1) -> transformer sensor.
TRANSFORMER_SENSOR_FOR_GUIDANCE_SLOT = tuple(
    IMU_ID_FOR_TRANSFORMER_SENSOR.index(i) for i in range(1, 10))


@dataclasses.dataclass
class Args:
    traj_root: Path | None = None
    """RoSHI recording directory (mode 1)."""
    input_npz: Path | None = None
    """Generic input file (mode 2); see load_input_npz."""
    checkpoint: Path = Path("model/transformer/plain_gt_noise_162k.pt")
    smplh_npz_path: Path = Path("model/smplh/neutral/model.npz")
    guidance_mode: Literal["egoallo", "roshi", "none"] = "egoallo"
    """egoallo: foot skating/smoothness/prior terms. roshi: also fit the IMU
    orientations. none: raw network output."""
    start_index: int = 0
    """Mode 1: index within the 30 Hz trajectory to start at."""
    traj_length: int = -1
    """Mode 1: number of output frames; -1 for the full sequence."""
    window: int = 128
    """Transformer context length (frames); the model was trained on 128."""
    overlap: int = 32
    """Overlap between consecutive windows, blended."""
    max_imu_gap_ms: float = 100.0
    """Do not interpolate IMU rotations across larger gaps."""
    floor_z: float | None = None
    """Override the floor height (world z, metres)."""
    save_inputs: bool = False
    """Also save the model input as <name>_inputs.npz (mode-2 format)."""
    output_dir: Path | None = None
    """Default: <traj_root>/egoallo_outputs, or the input npz's directory."""
    name: str | None = None
    """Output filename stem (without extension)."""
    overwrite_output: bool = False
    device: str = "cuda"


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
def load_input_npz(path: Path) -> dict:
    """Mode-2 schema (N >= 2 frames at ~30 Hz):

    Ts_world_cpf      (N,7)      Aria CPF poses [qw qx qy qz tx ty tz], metres, z up.
    bone_quats_world  (N,9,4)    Calibrated bone orientations R_world_bone as unit wxyz
                                 quaternions, all sensors in ONE common world frame
                                 (need not be the Aria world), transformer sensor order:
                                 pelvis, L_hip, R_hip, L_knee, R_knee, L_shoulder,
                                 R_shoulder, L_elbow, R_elbow.
      or bone_rotmats_world (N,9,3,3), same content as rotation matrices.
    floor_z           ()         World z of the floor (or pass --floor-z).
    timestamps_ns     (N,)       Optional, copied to the output.
    """
    with np.load(path) as d:
        data = {k: d[k] for k in d.files}
    cpf = data["Ts_world_cpf"].astype(np.float32)
    if "bone_quats_world" in data:
        quats = torch.from_numpy(data["bone_quats_world"].astype(np.float32))
    else:
        quats = SO3.from_matrix(torch.from_numpy(data["bone_rotmats_world"].astype(np.float32))).wxyz
    n = len(cpf)
    if cpf.shape != (n, 7) or quats.shape != (n, 9, 4):
        raise ValueError("Expected Ts_world_cpf (N,7) and bones (N,9,4) or (N,9,3,3)")
    ts = data.get("timestamps_ns")
    return dict(Ts_world_cpf=torch.from_numpy(cpf), bone_quats_world=quats,
                floor_z=float(data["floor_z"]) if "floor_z" in data else None,
                timestamps_ns=None if ts is None else ts.astype(np.int64),
                frame_nums=np.arange(n))


def tracking_to_utc_ns(vrs_path: Path, closed_loop_csv: Path, tracking_sec) -> np.ndarray:
    """Map Aria tracking time to UTC. Uses the VRS time-sync like 04_inference.py;
    projectaria-tools versions without TimeSyncMode.UTC fall back to the
    utc_timestamp_ns column of closed_loop_trajectory.csv (same rows as the poses)."""
    tracking_us = np.array([int(round(t * 1e6)) for t in tracking_sec], dtype=np.int64)
    from projectaria_tools.core.sensor_data import TimeSyncMode
    if hasattr(TimeSyncMode, "UTC"):
        from projectaria_tools.core import data_provider
        provider = data_provider.create_vrs_data_provider(str(vrs_path))
        return np.array([provider.convert_from_device_time_to_synctime_ns(int(t) * 1000, TimeSyncMode.UTC)
                         for t in tracking_us], dtype=np.int64)
    import pandas as pd
    print("[IMU] projectaria-tools has no TimeSyncMode.UTC; using closed_loop_trajectory.csv utc_timestamp_ns")
    df = pd.read_csv(closed_loop_csv, usecols=["tracking_timestamp_us", "utc_timestamp_ns"])
    lookup = pd.Series(df.utc_timestamp_ns.to_numpy(np.int64), index=df.tracking_timestamp_us.to_numpy(np.int64))
    missing = ~np.isin(tracking_us, lookup.index)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} pose timestamps missing from {closed_loop_csv}")
    return lookup.loc[tracking_us].to_numpy(np.int64)


def dense_bone_quats(imu_pkl: Path, utc_ns: np.ndarray, max_gap_ms: float):
    """SLERP calibrated bone rotations to `utc_ns`. Returns (quats (M,9,4), keep slice)."""
    from scipy.spatial.transform import Rotation, Slerp
    with imu_pkl.open("rb") as f:
        imu = pickle.load(f)
    keys = np.array(sorted(t for t, row in imu.items()
                           if all(row.get(i) is not None for i in range(1, 10))), dtype=np.int64)
    if len(keys) < 2:
        raise ValueError(f"{imu_pkl}: fewer than two complete 9-IMU groups")
    right = np.clip(np.searchsorted(keys, utc_ns), 1, len(keys) - 1)
    gaps = keys[right] - keys[right - 1]
    valid = (utc_ns >= keys[0]) & (utc_ns <= keys[-1]) & ((gaps <= max_gap_ms * 1e6) | np.isin(utc_ns, keys))
    edges = np.diff(np.r_[0, valid.astype(np.int8), 0])
    runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    if not runs or max(b - a for a, b in runs) < 2:
        raise ValueError("IMU data does not cover at least two consecutive CPF frames")
    start, stop = max(runs, key=lambda r: r[1] - r[0])
    if (start, stop) != (0, len(utc_ns)):
        print(f"[IMU] Using frames [{start}, {stop}) of {len(utc_ns)}: the longest interval "
              f"with IMU coverage and gaps <= {max_gap_ms:g} ms ({len(runs)} covered run(s)).")
    mats = np.array([[imu[int(t)][i] for i in IMU_ID_FOR_TRANSFORMER_SENSOR] for t in keys], dtype=np.float64)
    if not np.allclose(mats.swapaxes(-1, -2) @ mats, np.eye(3), atol=1e-3):
        raise ValueError("imu_info.pkl contains non-rotation matrices")
    src_t = (keys - keys[0]) / 1e9
    query = (utc_ns[start:stop] - keys[0]) / 1e9
    quats = np.stack([Slerp(src_t, Rotation.from_matrix(mats[:, k]))(query).as_quat()[:, [3, 0, 1, 2]]
                      for k in range(9)], axis=1)
    return torch.from_numpy(quats.astype(np.float32)), slice(int(start), int(stop))


def load_recording(args: Args) -> dict:
    from egoallo.data.aria_mps import load_point_cloud_and_find_ground
    from egoallo.inference_utils import InferenceInputTransforms, InferenceTrajectoryPaths

    paths = InferenceTrajectoryPaths.find(args.traj_root)
    if paths.imu_readings_pkl is None:
        raise FileNotFoundError("No imu_info.pkl found; run 03_sync.py first")
    _, floor_z = load_point_cloud_and_find_ground(paths.points_path)
    transforms = InferenceInputTransforms.load(paths.vrs_file, paths.slam_root_dir, fps=30)
    total = len(transforms.pose_timesteps)
    stop = total if args.traj_length <= 0 else min(total, args.start_index + args.traj_length + 1)
    sl = slice(args.start_index, stop)
    tracking_sec = np.array(transforms.pose_timesteps[sl])
    utc = tracking_to_utc_ns(paths.vrs_file, paths.slam_root_dir / "closed_loop_trajectory.csv", tracking_sec)
    quats, keep = dense_bone_quats(paths.imu_readings_pkl, utc, args.max_imu_gap_ms)
    frames = np.arange(total)[sl][keep]
    return dict(Ts_world_cpf=transforms.Ts_world_cpf[sl][keep], bone_quats_world=quats,
                floor_z=float(floor_z), frame_nums=frames,
                timestamps_ns=(tracking_sec[keep] * 1e9).round().astype(np.int64),
                utc_ns=utc[keep])


# ----------------------------------------------------------------------------
# Guidance
# ----------------------------------------------------------------------------
def post_guidance(traj, Ts_world_cpf, bone_quats_world, body, mode):
    if mode == "none":
        return traj
    imu = None
    if mode == "roshi":
        # Dense calibrated orientations, reordered to IMU_CONFIG order (frames 1..N-1).
        mats = SO3(bone_quats_world[1:]).as_matrix()[:, list(TRANSFORMER_SENSOR_FOR_GUIDANCE_SLOT)]
        imu = CorrespondedImuReadings(readings=IMUPoseWrtCanonical(
            confidence=torch.ones(len(mats), device=mats.device),
            indices=torch.arange(len(mats), device=mats.device), imu_readings=mats.contiguous()))
    params = dataclasses.replace(JaxGuidanceParams.defaults(mode, "post"), **TRANSFORMER_GUIDANCE_OVERRIDES)
    traj, _ = do_guidance_optimization(
        Ts_world_cpf=Ts_world_cpf[1:], traj=traj, body_model=body, guidance_mode=mode, phase="post",
        aria_all_hand_detections=None, imu_readings=imu, verbose=False, guidance_params=params)
    return traj


def main(args: Args) -> None:
    if (args.traj_root is None) == (args.input_npz is None):
        raise ValueError("Pass exactly one of --traj-root or --input-npz")
    device = torch.device(args.device)
    data = load_recording(args) if args.traj_root is not None else load_input_npz(args.input_npz)
    floor_z = args.floor_z if args.floor_z is not None else data["floor_z"]
    if floor_z is None:
        raise ValueError("No floor height: add floor_z to the input file or pass --floor-z")
    cpf = data["Ts_world_cpf"].to(device)
    bones = data["bone_quats_world"].to(device)
    head_height = float(cpf[:, 6].median()) - floor_z
    print(f"{len(cpf)} frames (first is context), floor_z={floor_z:.3f}, "
          f"median CPF height above floor={head_height:.2f} m")
    if not 0.5 < head_height < 2.5:
        raise ValueError(f"Implausible floor: CPF is {head_height:.2f} m above floor_z={floor_z:.3f}. "
                         "The point-cloud floor estimate likely failed; pass --floor-z.")

    out_dir = args.output_dir or (args.traj_root / "egoallo_outputs" if args.traj_root is not None
                                  else args.input_npz.parent)
    f0, f1 = int(data["frame_nums"][1]), int(data["frame_nums"][-1]) + 1
    name = (args.name or "").strip() or f"transformer_{args.guidance_mode}_{time.strftime('%Y%m%d-%H%M%S')}_{f0}-{f1}"
    out_path = out_dir / f"{name}.npz"
    if out_path.exists() and not args.overwrite_output:
        raise FileExistsError(f"{out_path} exists; pass --overwrite-output or a new --name")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_inputs:
        extra = {} if data["timestamps_ns"] is None else dict(timestamps_ns=data["timestamps_ns"])
        np.savez(out_dir / f"{name}_inputs.npz", Ts_world_cpf=cpf.numpy(force=True),
                 bone_quats_world=bones.numpy(force=True), floor_z=np.float64(floor_z), **extra)

    model = load_transformer(args.checkpoint, device=device)
    body = fncsmpl.SmplhModel.load(args.smplh_npz_path).to(device)
    traj = predict_sequence(model, cpf, bones, floor_z=floor_z, window=args.window, overlap=args.overlap)
    traj = post_guidance(traj, cpf, bones, body, args.guidance_mode)

    posed = traj.apply_to_body(body)
    Ts_world_root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(posed, cpf[1:])
    outputs = dict(
        Ts_world_cpf=cpf[1:].numpy(force=True), Ts_world_root=Ts_world_root.numpy(force=True),
        body_quats=posed.local_quats[..., :21, :].numpy(force=True),
        left_hand_quats=posed.local_quats[..., 21:36, :].numpy(force=True),
        right_hand_quats=posed.local_quats[..., 36:51, :].numpy(force=True),
        contacts=traj.contacts.numpy(force=True), betas=traj.betas.numpy(force=True),
        frame_nums=data["frame_nums"][1:])
    if data["timestamps_ns"] is not None:
        outputs["timestamps_ns"] = data["timestamps_ns"][1:]
    np.savez(out_path, **outputs)
    meta = {k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(args).items()}
    meta.update(floor_z_used=floor_z, model_config=dataclasses.asdict(model.config),
                sensor_order=list(TRANSFORMER_BONE_NAMES), smpl_joint_indices=list(TRANSFORMER_SMPL_JOINT_INDICES),
                guidance_overrides=TRANSFORMER_GUIDANCE_OVERRIDES if args.guidance_mode != "none" else None)
    out_path.with_name(f"{name}_args.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
