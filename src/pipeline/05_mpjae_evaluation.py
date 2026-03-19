#!/usr/bin/env python3
"""
Joint angle evaluation (MPJAE): SAM-3D ground truth vs IMU-only / EgoAllo / TTO.

Metric (degrees):
  - Local rotation error (MPJAE):  geodesic angle between GT and predicted
    parent-relative rotations per joint.

Usage:
  python 05_mpjae_evaluation.py <session_dir>
  python 05_mpjae_evaluation.py <session_dir> --egoallo-csv path/to/egoallo.csv
  python 05_mpjae_evaluation.py <session_dir> --tto-csv path/to/tto.csv --csv results.csv
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
# Also add pipeline/ so relative module import works
_PIPELINE_DIR = Path(__file__).resolve().parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

import numpy as np

from importlib import import_module

_viewer = import_module("03_imu_pose_viewer")

load_imu_csv = _viewer.load_imu_csv
load_calibration = _viewer.load_calibration
load_smplx_model = _viewer.load_smplx_model
precompute_shape = _viewer.precompute_shape
load_frames_csv = _viewer.load_frames_csv
load_smpl_ground_truth = _viewer.load_smpl_ground_truth
load_apriltag_rotations_by_time = _viewer.load_apriltag_rotations_by_time
estimate_world_alignment_from_tags = _viewer.estimate_world_alignment_from_tags
quaternion_to_matrix_wxyz = _viewer.quaternion_to_matrix_wxyz
sample_streams_hold = _viewer.sample_streams_hold
compute_time_intersection_ns = _viewer.compute_time_intersection_ns
ensure_calibration = _viewer.ensure_calibration
rot_angle_deg = _viewer.rot_angle_deg
_str2bool = _viewer._str2bool
load_tto_local_rotations = _viewer.load_tto_local_rotations
find_egoallo_csv = _viewer.find_egoallo_csv
find_tto_csv = _viewer.find_tto_csv
get_nearest_sync_timestamp = _viewer.get_nearest_sync_timestamp

from utils.imu_id_mapping import (
    IMU_ID_TO_JOINT,
    JOINT_NAMES,
    SMPLX_JOINT_INDEX_MAP,
)

# Joints to report: instrumented (have IMU) + propagated children.
REPORT_JOINTS = {
    "pelvis": 0,
    "left-hip": 1,
    "right-hip": 2,
    "spine1": 3,
    "left-knee": 4,
    "right-knee": 5,
    "spine2": 6,
    "left-ankle": 7,
    "right-ankle": 8,
    "spine3": 9,
    "left-shoulder": 16,
    "right-shoulder": 17,
    "left-elbow": 18,
    "right-elbow": 19,
    "left-wrist": 20,
    "right-wrist": 21,
}

# All 22 SMPLX body joints (for EgoAllo / TTO evaluation over all joints).
ALL_SMPLX_BODY_JOINTS = {
    "pelvis": 0,
    "left-hip": 1,
    "right-hip": 2,
    "spine1": 3,
    "left-knee": 4,
    "right-knee": 5,
    "spine2": 6,
    "left-ankle": 7,
    "right-ankle": 8,
    "spine3": 9,
    "left-toe": 10,
    "right-toe": 11,
    "neck": 12,
    "left-collar": 13,
    "right-collar": 14,
    "head": 15,
    "left-shoulder": 16,
    "right-shoulder": 17,
    "left-elbow": 18,
    "right-elbow": 19,
    "left-wrist": 20,
    "right-wrist": 21,
}

# Parent map for computing IMU local rotations
# (joint_name -> parent_joint_name used by build_local_rots_from_imu)
IMU_LOCAL_PARENT = {
    "pelvis": None,
    "left-hip": "pelvis",
    "right-hip": "pelvis",
    "left-knee": "left-hip",
    "right-knee": "right-hip",
    "left-shoulder": "pelvis",  # torso chain is rigid with pelvis
    "right-shoulder": "pelvis",
    "left-elbow": "left-shoulder",
    "right-elbow": "right-shoulder",
}


def compute_errors_imu(
    *,
    model: _viewer.SmplxModel,
    gt: _viewer.SmplGroundTruth,
    sampled: Dict[int, np.ndarray],
    timeline_frame_ids: np.ndarray,
    joint_to_imu: Dict[str, int],
    bone_from_imu_offset: Dict[str, np.ndarray],
    world_align: Dict[str, np.ndarray],
    quat_is_world_from_sensor: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-frame local rotation errors for IMU-only reconstruction.
    Uses first-frame pelvis alignment.

    Returns:
        local_rot_errors:  (T, J) in degrees
        valid_mask:        (T,) bool
    """
    T = len(timeline_frame_ids)
    n_gt_joints = gt.joint_rotations_local.shape[1]
    local_rot_errors = np.full((T, n_gt_joints), np.nan, dtype=np.float64)
    valid_mask = np.zeros(T, dtype=bool)

    first_frame_align: Optional[np.ndarray] = None

    for fi in range(T):
        frame_id = int(timeline_frame_ids[fi])
        gt_idx = gt.index_of(frame_id)
        if gt_idx is None:
            continue
        valid_mask[fi] = True

        # --- IMU bone globals ---
        bone_global: Dict[str, np.ndarray] = {}
        for joint_name in JOINT_NAMES:
            imu_id = joint_to_imu.get(joint_name)
            if imu_id is None:
                continue
            q = sampled[imu_id][fi]
            R = quaternion_to_matrix_wxyz(q)
            if not quat_is_world_from_sensor:
                R = R.T
            W_R_S = R
            if world_align:
                W_R_S = world_align.get(joint_name, np.eye(3)) @ W_R_S
            B_R_S = bone_from_imu_offset.get(joint_name)
            if B_R_S is None:
                bone_global[joint_name] = W_R_S
            else:
                bone_global[joint_name] = W_R_S @ B_R_S.T

        # --- GT local rotations ---
        gt_pelvis_R = np.array(gt.joint_rotations_local[gt_idx, 0], dtype=np.float64)
        gt_local = np.array(gt.joint_rotations_local[gt_idx], dtype=np.float64)

        # --- First-frame pelvis alignment ---
        if first_frame_align is None and "pelvis" in bone_global:
            first_frame_align = gt_pelvis_R @ bone_global["pelvis"].T

        if first_frame_align is not None:
            for k in bone_global:
                bone_global[k] = first_frame_align @ bone_global[k]

        # --- Local rotation errors (parent-relative) ---
        for jname, jidx in SMPLX_JOINT_INDEX_MAP.items():
            if jname not in bone_global:
                continue
            parent_name = IMU_LOCAL_PARENT.get(jname)
            if parent_name is None:
                imu_local_R = bone_global[jname]
            elif parent_name in bone_global:
                imu_local_R = bone_global[parent_name].T @ bone_global[jname]
            else:
                continue
            local_rot_errors[fi, jidx] = rot_angle_deg(
                gt_local[jidx].T @ imu_local_R
            )

    return local_rot_errors, valid_mask


def compute_errors_local_rot(
    *,
    gt: _viewer.SmplGroundTruth,
    local_rot_dict: Dict[int, np.ndarray],  # {utc_ns: (J,3,3)}
    t_sorted: List[int],
    timeline_frame_ids: np.ndarray,
    timeline: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-frame local rotation errors for a method that provides SMPLX
    local rotations (e.g. EgoAllo, TTO).

    Returns:
        local_rot_errors:  (T, J) in degrees
        valid_mask:        (T,) bool
    """
    T = timeline.size
    n_gt_joints = gt.joint_rotations_local.shape[1]
    local_rot_errors = np.full((T, n_gt_joints), np.nan, dtype=np.float64)
    valid_mask = np.zeros(T, dtype=bool)

    # First-frame pelvis alignment: pelvis local == global, so we need
    # to align the predicted root orientation to GT at frame 0.
    first_frame_align: Optional[np.ndarray] = None

    for fi in range(T):
        frame_id = int(timeline_frame_ids[fi])
        t_ns = int(timeline[fi])

        gt_idx = gt.index_of(frame_id)
        if gt_idx is None:
            continue

        # Find nearest method timestamp (skip if too far: >100ms)
        nearest_t = get_nearest_sync_timestamp(t_sorted, t_ns)
        if nearest_t is None:
            continue
        if abs(nearest_t - t_ns) > 100_000_000:  # 100ms in ns
            continue
        pred_local = local_rot_dict.get(nearest_t)
        if pred_local is None:
            continue

        valid_mask[fi] = True

        gt_local = np.array(gt.joint_rotations_local[gt_idx], dtype=np.float64)
        pred_local_f64 = pred_local.astype(np.float64)

        # Compute first-frame pelvis alignment
        if first_frame_align is None:
            first_frame_align = gt_local[0] @ pred_local_f64[0].T

        # --- Local rotation errors (all joints with predictions) ---
        for jidx in range(min(pred_local_f64.shape[0], gt_local.shape[0])):
            pred_R = pred_local_f64[jidx]
            if jidx == 0:
                pred_R = first_frame_align @ pred_R
            local_rot_errors[fi, jidx] = rot_angle_deg(
                gt_local[jidx].T @ pred_R
            )

    return local_rot_errors, valid_mask


def print_report(
    mode: str,
    local_rot_errors: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, float]:
    """Print summary table. Returns {metric_name: value} dict."""
    n_valid = int(valid_mask.sum())
    le = local_rot_errors[valid_mask]

    print(f"\n{'='*60}")
    print(f"  MPJAE Report  --  mode: {mode}  ({n_valid} frames)")
    print(f"{'='*60}")

    metrics: Dict[str, float] = {}

    # Use ALL_SMPLX_BODY_JOINTS for EgoAllo/TTO, REPORT_JOINTS for IMU
    is_full_body = mode in ("EgoAllo", "TTO")
    joint_set = ALL_SMPLX_BODY_JOINTS if is_full_body else REPORT_JOINTS

    hdr = f"  {'Joint':<18s} {'Local (°)':>10s} {'± std':>8s}"
    print(f"\n{hdr}")
    print(f"  {'-'*18} {'-'*10} {'-'*8}")

    instr_local_vals: List[float] = []
    all_local_vals: List[float] = []

    for jname, jidx in sorted(joint_set.items(), key=lambda x: x[1]):
        l = le[:, jidx]
        l_valid = l[~np.isnan(l)]

        l_mean = float(np.mean(l_valid)) if l_valid.size > 0 else float("nan")
        l_std = float(np.std(l_valid)) if l_valid.size > 0 else float("nan")

        is_instr = jname in SMPLX_JOINT_INDEX_MAP
        marker = " *" if is_instr else ""

        l_str = f"{l_mean:>10.1f}" if not np.isnan(l_mean) else f"{'--':>10s}"
        ls_str = f"{l_std:>8.1f}" if not np.isnan(l_std) else f"{'':>8s}"

        print(f"  {jname + marker:<18s} {l_str} {ls_str}")

        metrics[f"{mode}/local_deg/{jname}"] = l_mean

        if is_instr and not np.isnan(l_mean):
            instr_local_vals.append(l_mean)
        if not np.isnan(l_mean):
            all_local_vals.append(l_mean)

    # Aggregates
    mean_local = float(np.mean(instr_local_vals)) if instr_local_vals else float("nan")
    mean_all = float(np.mean(all_local_vals)) if all_local_vals else float("nan")

    print(f"\n  Instrumented joints (*) MPJAE: {mean_local:.1f}°")
    metrics[f"{mode}/local_deg/instrumented_mean"] = mean_local

    if is_full_body:
        print(f"  All joints ({len(all_local_vals)}) MPJAE:     {mean_all:.1f}°")
        metrics[f"{mode}/local_deg/all_joints_mean"] = mean_all

    return metrics


def write_csv(
    path: Path,
    mode: str,
    local_rot_errors: np.ndarray,
    valid_mask: np.ndarray,
    timeline_frame_ids: np.ndarray,
) -> None:
    """Write per-frame local rotation errors (degrees) to CSV."""
    is_full_body = mode in ("EgoAllo", "TTO")
    joint_set = ALL_SMPLX_BODY_JOINTS if is_full_body else REPORT_JOINTS
    joint_names_ordered = sorted(joint_set.items(), key=lambda x: x[1])
    header = ["frame_id", "mode"]
    for jn, _ in joint_names_ordered:
        header.append(f"{jn}_local_deg")

    rows = []
    for fi in range(local_rot_errors.shape[0]):
        if not valid_mask[fi]:
            continue
        row: List = [int(timeline_frame_ids[fi]), mode]
        for jn, jidx in joint_names_ordered:
            lv = local_rot_errors[fi, jidx]
            row.append(f"{lv:.2f}" if not np.isnan(lv) else "")
        rows.append(row)

    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv_mod.writer(f)
        if write_header:
            w.writerow(header)
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Joint angle error: SAM-3D GT vs IMU-only (degrees)"
    )
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--smplx-model", type=Path,
                        default=Path(__file__).resolve().parent / "model/smplx/SMPLX_NEUTRAL.npz")
    parser.add_argument("--imu-csv", type=Path, default=None)
    parser.add_argument("--imu-calibration", type=Path, default=None)
    parser.add_argument("--compute-calibration", action="store_true")
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--calib-duration-sec", type=float, default=None)
    parser.add_argument("--include-calibration", action="store_true",
                        help="Include calibration frames in evaluation. "
                             "By default, frames during calibration are skipped "
                             "(using metadata.json allRequiredTagsSeenFrameIndex).")
    parser.add_argument("--skip-seconds", type=float, default=None,
                        help="Override the calibration duration (seconds from recording start to skip). "
                             "Implies --after-calibration.")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Write per-frame errors (degrees) to this CSV.")
    parser.add_argument("--imu-time-offset-ms", type=float, default=0.0)
    parser.add_argument("--quat-is-world-from-sensor", nargs="?", const=True, default=True,
                        type=_str2bool)
    parser.add_argument("--egoallo-csv", type=Path, default=None,
                        help="Path to EgoAllo *.csv (defaults to session/egoallo/*.csv).")
    parser.add_argument("--tto-csv", type=Path, default=None,
                        help="Path to TTO *.csv (defaults to session/tto/*.csv).")

    args = parser.parse_args()
    session = args.session_dir.resolve()
    imu_csv = args.imu_csv or (session / "imu" / "imu_data.csv")
    calib_json = args.imu_calibration or (session / "imu_calibration.json")

    if not imu_csv.exists():
        print(f"Error: IMU CSV not found: {imu_csv}")
        return 1

    if args.compute_calibration:
        ensure_calibration(
            session_dir=session,
            smplx_model_path=args.smplx_model,
            output_json=calib_json,
            min_samples=args.min_samples,
            calib_duration_sec=args.calib_duration_sec,
            python_exe=os.environ.get("PYTHON", "python"),
        )

    if not calib_json.exists():
        print(f"Error: imu_calibration.json not found: {calib_json}")
        return 1

    # Load data
    print(f"Loading IMU data: {imu_csv}")
    streams = load_imu_csv(imu_csv)
    print(f"  IMUs present: {sorted(streams.keys())}")
    print(f"Loading calibration: {calib_json}")
    calib = load_calibration(calib_json)
    print(f"  Calibrated joints: {sorted(calib.keys())}")

    joint_to_imu: Dict[str, int] = {j: i for i, j in IMU_ID_TO_JOINT.items()}
    imu_time_offset_ns = int(round(args.imu_time_offset_ms * 1e6))

    # Tag <-> IMU axis mapping (same as viewer)
    T_R_IMU = np.array([
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)
    bone_from_imu_offset: Dict[str, np.ndarray] = {}
    for joint, B_R_tag in calib.items():
        bone_from_imu_offset[joint] = B_R_tag @ T_R_IMU
    sensor_from_tag = T_R_IMU.T

    # World alignment
    world_align: Dict[str, np.ndarray] = {}
    tag_rots_by_time = load_apriltag_rotations_by_time(session)
    if tag_rots_by_time:
        world_align, world_stats = estimate_world_alignment_from_tags(
            streams=streams,
            tag_rots_by_time=tag_rots_by_time,
            joint_to_imu=joint_to_imu,
            imu_time_offset_ns=imu_time_offset_ns,
            sensor_from_tag=sensor_from_tag,
            quat_is_world_from_sensor=bool(args.quat_is_world_from_sensor),
        )
        if world_align:
            print("World alignment estimated OK")
        else:
            print("Warning: world alignment failed")
    else:
        print("Warning: no AprilTag detections, world alignment skipped")

    # SMPLX model + betas
    betas = np.zeros((10,), dtype=np.float32)
    smpl_params = session / "smpl_output" / "smpl_parameters.npz"
    if smpl_params.exists():
        try:
            d = np.load(smpl_params, allow_pickle=True)
            if "betas" in d and d["betas"].ndim == 2 and d["betas"].shape[1] >= 10:
                betas = d["betas"][0, :10].astype(np.float32)
        except Exception:
            pass

    model = load_smplx_model(args.smplx_model, betas_dim=10)
    v_shaped, j_tpose = precompute_shape(model, betas)

    # GT
    gt = load_smpl_ground_truth(session)
    if gt is None:
        print(f"Error: SMPL ground truth not found under {session}/smpl_output/")
        return 1
    print(f"Loaded GT: {gt.joints.shape[0]} frames")

    # Build video timeline
    frame_ids, frame_times, _ = load_frames_csv(session)
    if frame_ids.size == 0:
        print("Error: frames.csv missing or empty")
        return 1

    # Filter to frames with GT
    mask = np.array([gt.has_frame(int(fid)) for fid in frame_ids], dtype=bool)
    frame_ids = frame_ids[mask]
    frame_times = frame_times[mask]
    if frame_ids.size == 0:
        print("Error: no overlap between frames.csv and GT")
        return 1

    # Filter to IMU coverage
    expected_imus = sorted(IMU_ID_TO_JOINT.keys())
    t0, t1 = compute_time_intersection_ns(streams, expected_imus)
    times_for_imu = frame_times.astype(np.int64) + imu_time_offset_ns
    mask_imu = (times_for_imu >= t0) & (times_for_imu <= t1)
    frame_ids = frame_ids[mask_imu]
    frame_times = frame_times[mask_imu]

    # Skip calibration period (default) unless --include-calibration
    skip_calib = not args.include_calibration or args.skip_seconds is not None
    if skip_calib:
        n_before = frame_ids.size
        if args.skip_seconds is not None:
            calib_end_ns = int(frame_times[0]) + int(args.skip_seconds * 1e9)
            mask_post = frame_times.astype(np.int64) > calib_end_ns
            frame_ids = frame_ids[mask_post]
            frame_times = frame_times[mask_post]
            print(f"Skipping calibration: {n_before - frame_ids.size} frames "
                  f"(first {args.skip_seconds:.1f}s), {frame_ids.size} remaining")
        else:
            meta_path = session / "metadata.json"
            if not meta_path.exists():
                print("Warning: metadata.json not found, cannot skip calibration frames")
            else:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                calib_seg = meta.get("calibrationSegment", {})
                frame_idx = calib_seg.get("allRequiredTagsSeenFrameIndex")
                elapsed_sec = calib_seg.get("allRequiredTagsSeenElapsedSec")
                if frame_idx is not None:
                    mask_post = frame_ids > int(frame_idx)
                    frame_ids = frame_ids[mask_post]
                    frame_times = frame_times[mask_post]
                    elapsed_str = f", ~{elapsed_sec:.1f}s" if elapsed_sec is not None else ""
                    print(f"Skipping calibration: {n_before - frame_ids.size} frames "
                          f"(frame_index > {frame_idx}{elapsed_str}), {frame_ids.size} remaining")
                elif elapsed_sec is not None:
                    calib_end_ns = int(frame_times[0]) + int(elapsed_sec * 1e9)
                    mask_post = frame_times.astype(np.int64) > calib_end_ns
                    frame_ids = frame_ids[mask_post]
                    frame_times = frame_times[mask_post]
                    print(f"Skipping calibration: {n_before - frame_ids.size} frames "
                          f"(first {elapsed_sec:.1f}s), {frame_ids.size} remaining")
                else:
                    print("Warning: metadata.json missing calibration info, "
                          "use --skip-seconds or --include-calibration")

    print(f"Evaluation frames: {frame_ids.size}")

    # Sample IMU
    timeline_for_imu = frame_times.astype(np.int64) + imu_time_offset_ns
    sampled = sample_streams_hold(streams, expected_imus, timeline_for_imu)

    # Clear CSV if writing
    if args.csv and args.csv.exists():
        args.csv.unlink()

    # Load EgoAllo / TTO local rotations (optional)
    ego_csv = args.egoallo_csv or find_egoallo_csv(session)
    ego_local_dict: Dict[int, np.ndarray] = {}
    ego_t_sorted: List[int] = []
    if ego_csv is not None and ego_csv.exists():
        ego_local_dict = load_tto_local_rotations(ego_csv, model.num_joints)
        if ego_local_dict:
            ego_t_sorted = sorted(ego_local_dict.keys())
            print(f"Loaded EgoAllo rotations from {ego_csv}: {len(ego_local_dict)} timestamps")

    tto_csv = args.tto_csv or find_tto_csv(session)
    tto_local_dict: Dict[int, np.ndarray] = {}
    tto_t_sorted: List[int] = []
    if tto_csv is not None and tto_csv.exists():
        tto_local_dict = load_tto_local_rotations(tto_csv, model.num_joints)
        if tto_local_dict:
            tto_t_sorted = sorted(tto_local_dict.keys())
            print(f"Loaded TTO rotations from {tto_csv}: {len(tto_local_dict)} timestamps")

    all_metrics: Dict[str, float] = {}

    # --- IMU-only evaluation ---
    le, vmask = compute_errors_imu(
        model=model,
        gt=gt,
        sampled=sampled,
        timeline_frame_ids=frame_ids,
        joint_to_imu=joint_to_imu,
        bone_from_imu_offset=bone_from_imu_offset,
        world_align=world_align,
        quat_is_world_from_sensor=bool(args.quat_is_world_from_sensor),
    )
    all_metrics.update(print_report("IMU", le, vmask))
    if args.csv:
        write_csv(args.csv, "IMU", le, vmask, frame_ids)

    # --- EgoAllo evaluation ---
    if ego_local_dict:
        le_e, vm_e = compute_errors_local_rot(
            gt=gt,
            local_rot_dict=ego_local_dict,
            t_sorted=ego_t_sorted,
            timeline_frame_ids=frame_ids,
            timeline=frame_times,
        )
        all_metrics.update(print_report("EgoAllo", le_e, vm_e))
        if args.csv:
            write_csv(args.csv, "EgoAllo", le_e, vm_e, frame_ids)

    # --- TTO evaluation ---
    if tto_local_dict:
        le_t, vm_t = compute_errors_local_rot(
            gt=gt,
            local_rot_dict=tto_local_dict,
            t_sorted=tto_t_sorted,
            timeline_frame_ids=frame_ids,
            timeline=frame_times,
        )
        all_metrics.update(print_report("TTO", le_t, vm_t))
        if args.csv:
            write_csv(args.csv, "TTO", le_t, vm_t, frame_ids)

    if args.csv:
        print(f"\nPer-frame errors written to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
