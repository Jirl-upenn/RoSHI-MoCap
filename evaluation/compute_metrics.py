"""Compute MPJPE, MPJAR, and JAE for all baselines against OptiTrack GT.

Loads per-method NPZ files (joints in OptiTrack Z-up frame) and the
fitted SMPL-X ground truth, aligns them by timestamp, and reports:

  - MPJPE:  Mean Per-Joint Position Error (cm), absolute world positions
  - MPJAR:  Mean Per-Joint Angular error (degrees),
            bone direction vectors in world space (affected by root orientation)
  - JAE:    Joint Angle Error (degrees), root-invariant bend angle error

Usage:
    python evaluation/compute_metrics.py
    python evaluation/compute_metrics.py --datasets dataset1 dataset5
    python evaluation/compute_metrics.py --methods imu_naive sam3d
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.eval_utils import (
    BODY_BONES,
    NUM_BODY_JOINTS,
    RECORDINGS_DIR,
    load_calib_duration_sec,
    load_optitrack_smplx_fit,
    nearest_indices,
)

# ── Method definitions ────────────────────────────────────────────────

METHOD_CONFIG = {
    "imu_naive": {
        "dir": Path("evaluation/imu_naive"),
        "pattern": "{dataset}_imu_only.npz",
        "label": "IMU-only (naive)",
    },
    "imu_egoallo": {
        "dir": Path("evaluation/imu_egoallo"),
        "pattern": "{dataset}_imu_egoallo.npz",
        "label": "IMU + EgoAllo root",
    },
    "egoallo": {
        "dir": Path("evaluation/egoallo"),
        "pattern": "{dataset}_egoallo.npz",
        "label": "EgoAllo",
    },
    "imu_tto": {
        "dir": Path("evaluation/imu_tto"),
        "pattern": "{dataset}_imu_tto.npz",
        "label": "IMU-TTO (Ours)",
    },
    "sam3d": {
        "dir": Path("evaluation/sam3d"),
        "pattern": "{dataset}_sam3d.npz",
        "label": "SAM3D",
    },
}


# ── Metric helpers ────────────────────────────────────────────────────

def compute_mpjpe(pred_joints: np.ndarray, gt_joints: np.ndarray) -> float:
    """MPJPE in cm. Both inputs: (N, J, 3) in meters."""
    err = np.linalg.norm(pred_joints - gt_joints, axis=-1)  # (N, J)
    return float(np.mean(err) * 100.0)


def compute_mpjar(
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    bones: List[Tuple[int, int]],
) -> float:
    """Mean Per-Joint Angular error in degrees on bone direction vectors."""
    angles = []
    for j1, j2 in bones:
        if j1 >= pred_joints.shape[1] or j2 >= pred_joints.shape[1]:
            continue
        pred_dir = pred_joints[:, j2, :] - pred_joints[:, j1, :]
        gt_dir = gt_joints[:, j2, :] - gt_joints[:, j1, :]

        pred_norm = np.linalg.norm(pred_dir, axis=-1, keepdims=True)
        gt_norm = np.linalg.norm(gt_dir, axis=-1, keepdims=True)
        pred_norm = np.clip(pred_norm, 1e-8, None)
        gt_norm = np.clip(gt_norm, 1e-8, None)

        pred_unit = pred_dir / pred_norm
        gt_unit = gt_dir / gt_norm

        cos = np.clip(np.sum(pred_unit * gt_unit, axis=-1), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos)))

    if not angles:
        return 0.0
    return float(np.mean(np.concatenate(angles)))


def _vec_angle(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Angle in degrees between vectors v1, v2 of shape (N, 3)."""
    n1 = np.clip(np.linalg.norm(v1, axis=-1, keepdims=True), 1e-8, None)
    n2 = np.clip(np.linalg.norm(v2, axis=-1, keepdims=True), 1e-8, None)
    cos = np.clip(np.sum((v1 / n1) * (v2 / n2), axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def compute_jae(
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    bones: List[Tuple[int, int]],
) -> float:
    """Joint Angle Error (root-invariant) in degrees.

    For each joint with both a parent bone and at least one child bone,
    compute the bend angle (between incoming and outgoing bone vectors)
    and report the mean absolute difference between predicted and GT.
    """
    children: Dict[int, List[int]] = {}
    parent: Dict[int, int] = {}
    for j1, j2 in bones:
        children.setdefault(j1, []).append(j2)
        parent[j2] = j1

    errors = []
    n_joints = pred_joints.shape[1]
    for joint in range(n_joints):
        if joint not in parent or joint not in children:
            continue
        p = parent[joint]
        if p >= n_joints:
            continue
        for c in children[joint]:
            if c >= n_joints:
                continue
            pred_in = pred_joints[:, joint, :] - pred_joints[:, p, :]
            pred_out = pred_joints[:, c, :] - pred_joints[:, joint, :]
            gt_in = gt_joints[:, joint, :] - gt_joints[:, p, :]
            gt_out = gt_joints[:, c, :] - gt_joints[:, joint, :]

            pred_angle = _vec_angle(pred_in, pred_out)
            gt_angle = _vec_angle(gt_in, gt_out)
            errors.append(np.abs(pred_angle - gt_angle))

    if not errors:
        return 0.0
    return float(np.mean(np.concatenate(errors)))



# ── Loading helpers ───────────────────────────────────────────────────

def load_method_joints(
    method: str,
    dataset_name: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load (joints_22x3, timestamps_ns) for a method+dataset, or None."""
    cfg = METHOD_CONFIG[method]
    npz_path = cfg["dir"] / cfg["pattern"].format(dataset=dataset_name)
    if not npz_path.exists():
        return None
    d = np.load(npz_path)
    joints = d["joints_opti"][:, :NUM_BODY_JOINTS, :]  # (N, 22, 3)
    ts = d["timestamps_ns"]
    n = min(joints.shape[0], ts.shape[0])
    return joints[:n], ts[:n]


# ── Per-dataset evaluation ────────────────────────────────────────────

def evaluate_dataset(
    dataset_name: str,
    methods: List[str],
    include_calibration: bool = False,
    compute_mpjar_flag: bool = False,
) -> Tuple[Dict[str, Dict[str, float]], Optional[Dict[str, float]]]:
    """Evaluate all methods on one dataset. Returns (results, dataset_info).
    dataset_info: {gt_frames, dur_s, n_eval_frames} for printing, or None if skipped.
    """
    gt = load_optitrack_smplx_fit(dataset_name)
    if gt is None:
        return {}, None

    gt_joints = gt["joints_zup"]  # (N_gt, 22, 3)
    gt_ts = gt["timestamps"]       # (N_gt,)

    # Load third-person camera timeline as the common evaluation window
    frames_csv = RECORDINGS_DIR / dataset_name / "frames.csv"
    timeline_ts: Optional[np.ndarray] = None
    if frames_csv.exists():
        ts_list = []
        with open(frames_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("utc_timestamp_ns"):
                    ts_list.append(int(row["utc_timestamp_ns"]))
        if ts_list:
            timeline_ts = np.array(ts_list, dtype=np.int64)

    # Determine evaluation time window: [eval_start, eval_end]
    calib_dur = None if include_calibration else load_calib_duration_sec(dataset_name)
    if timeline_ts is not None and len(timeline_ts) > 0:
        eval_start = timeline_ts[0]
        if calib_dur is not None:
            eval_start = int(timeline_ts[0] + calib_dur * 1e9)
        eval_end = int(timeline_ts[-1])
    elif calib_dur is not None and len(gt_ts) > 0:
        eval_start = int(gt_ts[0] + calib_dur * 1e9)
        eval_end = int(gt_ts[-1])
    else:
        eval_start = int(gt_ts[0]) if len(gt_ts) > 0 else 0
        eval_end = int(gt_ts[-1]) if len(gt_ts) > 0 else 0

    # Trim GT to the evaluation window
    gt_mask = (gt_ts >= eval_start) & (gt_ts <= eval_end)
    n_gt_orig = len(gt_ts)
    gt_joints = gt_joints[gt_mask]
    gt_ts = gt_ts[gt_mask]
    n_trimmed = n_gt_orig - len(gt_ts)

    # Number of third-person camera frames in the eval window (recall denominator)
    n_eval_frames = 0
    if timeline_ts is not None:
        n_eval_frames = int(((timeline_ts >= eval_start) & (timeline_ts <= eval_end)).sum())

    dur_s = (eval_end - eval_start) / 1e9
    dataset_info: Optional[Dict[str, float]] = {
        "gt_frames": float(gt_joints.shape[0]),
        "dur_s": dur_s,
        "n_eval_frames": float(n_eval_frames),
    }

    results: Dict[str, Dict[str, float]] = {}

    # First pass: load and evaluate all methods on their full data
    method_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for method in methods:
        data = load_method_joints(method, dataset_name)
        if data is None:
            continue

        pred_joints, pred_ts = data
        pred_mask = (pred_ts >= eval_start) & (pred_ts <= eval_end)
        pred_joints = pred_joints[pred_mask]
        pred_ts = pred_ts[pred_mask]

        if len(pred_ts) == 0:
            continue

        match_idx = nearest_indices(gt_ts, pred_ts)
        method_data[method] = (pred_joints, pred_ts, match_idx)

        gt_matched = gt_joints[match_idx]
        mpjpe = compute_mpjpe(pred_joints, gt_matched)
        mpjar = compute_mpjar(pred_joints, gt_matched, BODY_BONES) if compute_mpjar_flag else None
        jae = compute_jae(pred_joints, gt_matched, BODY_BONES)
        n_pred = pred_joints.shape[0]
        recall = n_pred / n_eval_frames if n_eval_frames > 0 else 1.0

        results[method] = {
            "MPJPE": mpjpe,
            "MPJAR": mpjar,
            "JAE": jae,
            "n_frames": n_pred,
            "recall": recall,
        }

    # Second pass: re-evaluate all methods on the SAM3D-covered GT subset
    if "sam3d" in method_data and len(method_data) > 1:
        sam3d_gt_set = set(method_data["sam3d"][2].tolist())
        for method in methods:
            if method not in method_data:
                continue
            pred_joints, pred_ts, match_idx = method_data[method]
            keep = np.array([i for i, gi in enumerate(match_idx) if gi in sam3d_gt_set])
            if len(keep) == 0:
                continue
            pj = pred_joints[keep]
            gm = gt_joints[match_idx[keep]]
            results[method]["MPJPE_matched"] = compute_mpjpe(pj, gm)
            if compute_mpjar_flag:
                results[method]["MPJAR_matched"] = compute_mpjar(pj, gm, BODY_BONES)
            results[method]["JAE_matched"] = compute_jae(pj, gm, BODY_BONES)

    return results, dataset_info


# ── Aggregate table ───────────────────────────────────────────────────

def print_summary(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    methods: List[str],
    dataset_info: Optional[Dict[str, Dict[str, float]]] = None,
):
    """Print PER-DATASET BREAKDOWN (with GT line per dataset) then PER-SESSION AVERAGES."""
    has_mpjar = any(
        r.get("MPJAR") is not None
        for ds_r in all_results.values()
        for r in ds_r.values()
    )

    # Per-dataset breakdown: same format as before (header + GT line + method lines)
    print(f"\n{'='*60}")
    print("  PER-DATASET BREAKDOWN")
    print(f"{'='*60}")

    for ds in sorted(all_results.keys()):
        ds_results = all_results[ds]
        if not ds_results:
            continue
        print(f"\n{'='*60}")
        print(f"  {ds}")
        print(f"{'='*60}")
        info = (dataset_info or {}).get(ds)
        if info is not None:
            print(f"  GT: {int(info['gt_frames'])} frames, eval window {info['dur_s']:.1f}s, "
                  f"camera frames: {int(info['n_eval_frames'])}")
        for method in methods:
            if method not in ds_results:
                continue
            r = ds_results[method]
            label = METHOD_CONFIG[method]["label"]
            jae_m, jae_n = r.get("JAE_matched"), r["JAE"]
            jae_str = f"{jae_m:5.1f}°/ {jae_n:5.1f}°" if jae_m is not None else f"{jae_n:5.1f}°"
            mpjpe_m, mpjpe_n = r.get("MPJPE_matched"), r["MPJPE"]
            mpjpe_str = f"{mpjpe_m:5.1f}/ {mpjpe_n:5.1f} cm" if mpjpe_m is not None else f"{mpjpe_n:5.1f} cm"
            parts = [f"  {label:25s}  MPJPE= {mpjpe_str}  "]
            if r.get("MPJAR") is not None:
                mpjar_m, mpjar_n = r.get("MPJAR_matched"), r["MPJAR"]
                mpjar_str = f"{mpjar_m:5.1f}°/ {mpjar_n:5.1f}°" if mpjar_m is not None else f"{mpjar_n:5.1f}°"
                parts.append(f"MPJAR= {mpjar_str}  ")
            parts.append(f"JAE= {jae_str}")
            if method == "sam3d":
                parts.append(f"  recall={r['recall']:5.1%}")
            print("".join(parts))

    SESSIONS = {
        "Session 1 (dataset1-4)": [f"dataset{i}" for i in range(1, 5)],
        "Session 2 (dataset5-8)": [f"dataset{i}" for i in range(5, 9)],
    }
    print(f"\n{'='*80}")
    print("  PER-SESSION AVERAGES")
    print(f"{'='*80}")

    shdr_parts = [f"{'Method':25s}", f"{'MPJPE (cm)':>14s}"]
    if has_mpjar:
        shdr_parts.append(f"{'MPJAR (°)':>14s}")
    shdr_parts += [f"{'JAE (°)':>14s}", f"{'Recall':>8s}"]
    sess_header = "  ".join(shdr_parts)
    for session_name, session_ds in SESSIONS.items():
        print(f"\n  {session_name}:")
        print(f"  {sess_header}")
        print(f"  {'-' * len(sess_header)}")
        for method in methods:
            mpjpe_v, mpjpe_m_v, mpjar_v, mpjar_m_v, jae_v, jae_m_v, recall_v = [], [], [], [], [], [], []
            for ds in session_ds:
                if ds not in all_results or method not in all_results[ds]:
                    continue
                r = all_results[ds][method]
                mpjpe_v.append(r["MPJPE"])
                jae_v.append(r["JAE"])
                recall_v.append(r["recall"])
                if r.get("MPJPE_matched") is not None:
                    mpjpe_m_v.append(r["MPJPE_matched"])
                if r.get("JAE_matched") is not None:
                    jae_m_v.append(r["JAE_matched"])
                if r.get("MPJAR") is not None:
                    mpjar_v.append(r["MPJAR"])
                    if r.get("MPJAR_matched") is not None:
                        mpjar_m_v.append(r["MPJAR_matched"])
            if not mpjpe_v:
                continue
            label = METHOD_CONFIG[method]["label"]
            mpjpe_str = f"{np.mean(mpjpe_m_v):5.1f}/ {np.mean(mpjpe_v):5.1f}" if mpjpe_m_v else f"{np.mean(mpjpe_v):5.1f}"
            jae_str = f"{np.mean(jae_m_v):5.1f}/ {np.mean(jae_v):5.1f}" if jae_m_v else f"{np.mean(jae_v):5.1f}"
            recall_str = f"{np.mean(recall_v):6.1%}" if method == "sam3d" else "   —"
            parts = [f"{label:25s}", f"{mpjpe_str:>14s}"]
            if has_mpjar:
                mpjar_str = f"{np.mean(mpjar_m_v):5.1f}/ {np.mean(mpjar_v):5.1f}" if mpjar_m_v else (f"{np.mean(mpjar_v):5.1f}" if mpjar_v else "N/A")
                parts.append(f"{mpjar_str:>14s}")
            parts += [f"{jae_str:>14s}", f"{recall_str:>8s}"]
            print("  " + "  ".join(parts))


# ── Export ────────────────────────────────────────────────────────────

def export_csv(
    path: Path,
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    dataset_info: Optional[Dict[str, Dict[str, float]]],
    methods: List[str],
    has_mpjar: bool,
) -> None:
    """Write one row per (dataset, method) with all metrics."""
    with open(path, "w", newline="") as f:
        base_cols = [
            "dataset", "method", "gt_frames", "eval_sec", "camera_frames",
            "mpjpe_matched", "mpjpe_full", "jae_matched", "jae_full", "recall",
        ]
        if has_mpjar:
            base_cols.extend(["mpjar_matched", "mpjar_full"])
        w = csv.DictWriter(f, fieldnames=base_cols, extrasaction="ignore")
        w.writeheader()
        info = dataset_info or {}
        for ds in sorted(all_results.keys()):
            ds_results = all_results[ds]
            di = info.get(ds) or {}
            for method in methods:
                if method not in ds_results:
                    continue
                r = ds_results[method]
                row = {
                    "dataset": ds,
                    "method": METHOD_CONFIG[method]["label"],
                    "gt_frames": int(di.get("gt_frames", 0)),
                    "eval_sec": round(di.get("dur_s", 0), 2),
                    "camera_frames": int(di.get("n_eval_frames", 0)),
                    "mpjpe_matched": round(r.get("MPJPE_matched") or r["MPJPE"], 2),
                    "mpjpe_full": round(r["MPJPE"], 2),
                    "jae_matched": round(r.get("JAE_matched") or r["JAE"], 2),
                    "jae_full": round(r["JAE"], 2),
                    "recall": round(r["recall"], 4),
                }
                if has_mpjar:
                    row["mpjar_matched"] = round(r.get("MPJAR_matched") or r.get("MPJAR") or 0, 2)
                    row["mpjar_full"] = round(r.get("MPJAR") or 0, 2)
                w.writerow(row)


def export_markdown(
    path: Path,
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    dataset_info: Optional[Dict[str, Dict[str, float]]],
    methods: List[str],
    has_mpjar: bool,
) -> None:
    """Write per-dataset tables and session averages in Markdown."""
    lines: List[str] = []
    lines.append("# Metrics (MPJPE cm, JAE °; matched / full)\n")
    info = dataset_info or {}
    SESSIONS = {
        "Session 1 (dataset1-4)": [f"dataset{i}" for i in range(1, 5)],
        "Session 2 (dataset5-8)": [f"dataset{i}" for i in range(5, 9)],
    }

    # Per-dataset
    lines.append("## Per-dataset breakdown\n")
    for ds in sorted(all_results.keys()):
        ds_results = all_results[ds]
        if not ds_results:
            continue
        di = info.get(ds) or {}
        lines.append(f"### {ds}\n")
        lines.append(f"- GT: {int(di.get('gt_frames', 0))} frames, "
                    f"eval window {di.get('dur_s', 0):.1f}s, "
                    f"camera frames: {int(di.get('n_eval_frames', 0))}\n")
        header = ["Method", "MPJPE (cm)", "JAE (°)"]
        if has_mpjar:
            header.append("MPJAR (°)")
        header.append("Recall")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + " --- |" * len(header))
        for method in methods:
            if method not in ds_results:
                continue
            r = ds_results[method]
            label = METHOD_CONFIG[method]["label"]
            mpjpe_m, mpjpe_n = r.get("MPJPE_matched"), r["MPJPE"]
            mpjpe_str = f"{mpjpe_m:.1f}/ {mpjpe_n:.1f}" if mpjpe_m is not None else f"{mpjpe_n:.1f}"
            jae_m, jae_n = r.get("JAE_matched"), r["JAE"]
            jae_str = f"{jae_m:.1f}/ {jae_n:.1f}" if jae_m is not None else f"{jae_n:.1f}"
            recall_str = f"{r['recall']:.1%}" if method == "sam3d" else "—"
            row = [label, mpjpe_str, jae_str]
            if has_mpjar:
                mpjar_m, mpjar_n = r.get("MPJAR_matched"), r.get("MPJAR")
                mpjar_str = f"{mpjar_m:.1f}/ {mpjar_n:.1f}" if mpjar_m is not None else f"{mpjar_n:.1f}"
                row.append(mpjar_str)
            row.append(recall_str)
            lines.append("| " + " | ".join(str(x) for x in row) + " |")
        lines.append("")

    # Session averages
    lines.append("## Per-session averages\n")
    for session_name, session_ds in SESSIONS.items():
        lines.append(f"### {session_name}\n")
        header = ["Method", "MPJPE (cm)", "JAE (°)"]
        if has_mpjar:
            header.append("MPJAR (°)")
        header.append("Recall")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + " --- |" * len(header))
        for method in methods:
            mpjpe_v, mpjpe_m_v, mpjar_v, mpjar_m_v, jae_v, jae_m_v, recall_v = [], [], [], [], [], [], []
            for ds in session_ds:
                if ds not in all_results or method not in all_results[ds]:
                    continue
                r = all_results[ds][method]
                mpjpe_v.append(r["MPJPE"])
                jae_v.append(r["JAE"])
                recall_v.append(r["recall"])
                if r.get("MPJPE_matched") is not None:
                    mpjpe_m_v.append(r["MPJPE_matched"])
                if r.get("JAE_matched") is not None:
                    jae_m_v.append(r["JAE_matched"])
                if r.get("MPJAR") is not None:
                    mpjar_v.append(r["MPJAR"])
                    if r.get("MPJAR_matched") is not None:
                        mpjar_m_v.append(r["MPJAR_matched"])
            if not mpjpe_v:
                continue
            label = METHOD_CONFIG[method]["label"]
            mpjpe_str = f"{np.mean(mpjpe_m_v):.1f}/ {np.mean(mpjpe_v):.1f}" if mpjpe_m_v else f"{np.mean(mpjpe_v):.1f}"
            jae_str = f"{np.mean(jae_m_v):.1f}/ {np.mean(jae_v):.1f}" if jae_m_v else f"{np.mean(jae_v):.1f}"
            recall_str = f"{np.mean(recall_v):.1%}" if method == "sam3d" else "—"
            row = [label, mpjpe_str, jae_str]
            if has_mpjar:
                mpjar_str = f"{np.mean(mpjar_m_v):.1f}/ {np.mean(mpjar_v):.1f}" if mpjar_m_v else (f"{np.mean(mpjar_v):.1f}" if mpjar_v else "N/A")
                row.append(mpjar_str)
            row.append(recall_str)
            lines.append("| " + " | ".join(str(x) for x in row) + " |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute MPJPE / MPJAR / JAE for all baselines vs OptiTrack GT."
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=[f"dataset{i}" for i in range(1, 9)],
        help="Dataset names. Default: all 8.",
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=list(METHOD_CONFIG.keys()),
        choices=list(METHOD_CONFIG.keys()),
        help=f"Methods to evaluate. Default: all ({', '.join(METHOD_CONFIG.keys())}).",
    )
    parser.add_argument(
        "--include-calibration", action="store_true",
        help="Include the initial calibration period (excluded by default).",
    )
    parser.add_argument(
        "--mpjar", action="store_true",
        help="Include MPJAR (bone direction angular error). Default: off.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Export results to file. Format from extension: .csv or .md",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    dataset_info: Dict[str, Dict[str, float]] = {}
    for ds in args.datasets:
        results, info = evaluate_dataset(
            ds, args.methods, args.include_calibration,
            compute_mpjar_flag=args.mpjar,
        )
        all_results[ds] = results
        if info is not None:
            dataset_info[ds] = info

    print_summary(all_results, args.methods, dataset_info)

    if args.output is not None:
        has_mpjar = any(
            r.get("MPJAR") is not None
            for ds_r in all_results.values()
            for r in ds_r.values()
        )
        suf = args.output.suffix.lower()
        if suf == ".csv":
            export_csv(args.output, all_results, dataset_info, args.methods, has_mpjar)
        else:
            if suf != ".md":
                args.output = args.output.with_suffix(".md")
            export_markdown(args.output, all_results, dataset_info, args.methods, has_mpjar)
        print(f"Wrote {args.output}")

    print()
