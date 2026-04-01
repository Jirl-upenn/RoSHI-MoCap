"""Convert EgoAllo NPZ output to a joint-rotation CSV.

This script takes:
  1. An NPZ file produced by 3_aria_inference.py
  2. The closed_loop_trajectory.csv for timestamp mapping (inside traj_root)

It extracts the LOCAL (child-to-parent) rotation matrices for ALL 22 SMPL
body joints (root + 21 body joints), maps tracking timestamps to UTC, and
writes a CSV with columns:
    utc_timestamp_ns, joint_id, joint_name, rot_matrix

Usage:
    PYTHONPATH="src" python npz_to_imu_csv.py \
        --npz-path  <path_to_output.npz> \
        --traj-root <path_to_trajectory_root> \
        --output-csv <output.csv>

TODO: change the imu mapping in optimization code. 
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from egoallo.transforms import SO3


# -----------------------------------------------------------------------
# SMPL-H joint ID -> joint name (22 body joints: root + 21 body)
# This is the standard SMPL ordering from specs.py.
# Joint 0 (hips/pelvis) is the root; joints 1-21 are stored in body_quats.
# -----------------------------------------------------------------------
SMPL_JOINT_NAMES: dict[int, str] = {
    0: "hips",
    1: "leftUpLeg",
    2: "rightUpLeg",
    3: "spine",
    4: "leftLeg",
    5: "rightLeg",
    6: "spine1",
    7: "leftFoot",
    8: "rightFoot",
    9: "spine2",
    10: "leftToeBase",
    11: "rightToeBase",
    12: "neck",
    13: "leftShoulder",
    14: "rightShoulder",
    15: "head",
    16: "leftArm",
    17: "rightArm",
    18: "leftForeArm",
    19: "rightForeArm",
    20: "leftHand",
    21: "rightHand",
}


@dataclasses.dataclass
class Args:
    npz_path: Path
    """Path to the EgoAllo NPZ output file."""

    traj_root: Path
    """Trajectory root directory (contains the VRS, MPS data, etc.)."""

    output_csv: Path | None = None
    """Output CSV path. Defaults to <npz_stem>_joint_local_rot.csv next to the NPZ."""

    sample_index: int = 0
    """Which sample to use (the NPZ may contain multiple samples in axis 0)."""


def main(args: Args) -> None:
    # ------------------------------------------------------------------
    # 1. Load the NPZ
    # ------------------------------------------------------------------
    data = np.load(args.npz_path)
    body_quats = torch.from_numpy(
        data["body_quats"][args.sample_index]
    )  # (T, 21, 4) wxyz local quaternions
    Ts_world_root = torch.from_numpy(
        data["Ts_world_root"][args.sample_index]
    )  # (T, 7) wxyz_xyz
    timestamps_ns = data["timestamps_ns"]  # (T,) tracking ns
    T = body_quats.shape[0]

    print(f"Loaded NPZ: {T} frames, body_quats {body_quats.shape}")

    # ------------------------------------------------------------------
    # 2. Convert local quaternions to local rotation matrices
    # ------------------------------------------------------------------
    # body_quats are already LOCAL (child-to-parent) quaternions.
    # No FK needed — just convert quat -> 3x3 rotation matrix.

    # Root (joint 0): local rotation = world orientation from Ts_world_root
    root_quat_wxyz = Ts_world_root[:, :4]  # (T, 4)
    root_rot = SO3(root_quat_wxyz).as_matrix().detach().numpy()  # (T, 3, 3)

    # Body joints 1-21: local quaternions from body_quats[:, 0:21, :]
    body_rot = SO3(body_quats).as_matrix().detach().numpy()  # (T, 21, 3, 3)

    print(f"Converted: root_rot {root_rot.shape}, body_rot {body_rot.shape}")

    # ------------------------------------------------------------------
    # 3. Map tracking timestamps -> UTC timestamps
    # ------------------------------------------------------------------
    tracking_us = (timestamps_ns / 1000).astype(np.int64)

    slam_csvs = list(args.traj_root.glob("**/closed_loop_trajectory.csv"))
    assert len(slam_csvs) == 1, (
        f"Expected 1 closed_loop_trajectory.csv, found {len(slam_csvs)}"
    )
    traj_csv_path = slam_csvs[0]
    print(f"Loading timestamp mapping from {traj_csv_path}")

    traj_df = pd.read_csv(
        traj_csv_path,
        usecols=["tracking_timestamp_us", "utc_timestamp_ns"],
    )
    tracking_to_utc = dict(
        zip(
            traj_df["tracking_timestamp_us"].values,
            traj_df["utc_timestamp_ns"].values,
        )
    )

    utc_timestamps = np.array(
        [tracking_to_utc[int(t)] for t in tracking_us], dtype=np.int64
    )
    print(f"Mapped {len(utc_timestamps)} timestamps to UTC")

    # ------------------------------------------------------------------
    # 4. Write CSV: utc_timestamp_ns, joint_id, joint_name, rot_matrix
    # ------------------------------------------------------------------
    if args.output_csv is None:
        output_csv = args.npz_path.parent / (
            args.npz_path.stem + "_joint_local_rot.csv"
        )
    else:
        output_csv = args.output_csv

    num_joints = len(SMPL_JOINT_NAMES)  # 22
    rows: list[dict] = []
    for t_idx in range(T):
        utc_ns = int(utc_timestamps[t_idx])
        for joint_id in range(num_joints):
            joint_name = SMPL_JOINT_NAMES[joint_id]
            if joint_id == 0:
                rot = root_rot[t_idx]  # (3, 3)
            else:
                rot = body_rot[t_idx, joint_id - 1]  # (3, 3)
            rot_str = repr(rot.tolist())
            rows.append(
                {
                    "utc_timestamp_ns": utc_ns,
                    "joint_id": joint_id,
                    "joint_name": joint_name,
                    "rot_matrix": rot_str,
                }
            )

    out_df = pd.DataFrame(
        rows,
        columns=["utc_timestamp_ns", "joint_id", "joint_name", "rot_matrix"],
    )
    out_df.to_csv(output_csv, index=False, quoting=1)
    print(
        f"Wrote {len(out_df)} rows "
        f"({T} timestamps x {num_joints} joints) to {output_csv}"
    )


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
