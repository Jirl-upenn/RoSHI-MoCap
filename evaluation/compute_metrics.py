"""Compute MPJPE, MPJAR, and JAE for all baselines against OptiTrack GT.

Reads the released evaluation bundle::

    evaluation/data/<activity>/optitrack_gt.npz    OptiTrack SMPL-X fit
    evaluation/data/<activity>/<method>.npz        predicted joints

Every file stores ``joints_opti`` (T, 22, 3) in the OptiTrack Z-up world frame
plus ``timestamps_ns``. Predictions are matched to the nearest ground-truth
frame in time and scored with:

  - MPJPE:  Mean Per-Joint Position Error (cm), absolute world positions
  - MPJAR:  Mean Per-Joint Angular error (degrees),
            bone direction vectors in world space (affected by root orientation)
  - JAE:    Joint Angle Error (degrees), root-invariant bend angle error

SAM3D recall is the fraction of third-person camera frames it reconstructed;
the frame count is stored as ``n_camera_frames`` in the ground-truth file.

Usage:
    python evaluation/compute_metrics.py
    python evaluation/compute_metrics.py --activities 10_tennis 09_sliding
    python evaluation/compute_metrics.py --methods imu_only sam3d -o metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.eval_utils import BODY_BONES, NUM_BODY_JOINTS, nearest_indices

DATA_DIR = Path("evaluation/data")
GT_FILENAME = "optitrack_gt.npz"

# Ordered so that tables read baselines first, ours last.
METHOD_LABELS = {
    "imu_only": "IMU-only (naive)",
    "imu_egoallo": "IMU + EgoAllo root",
    "egoallo": "EgoAllo",
    "roshi": "RoSHI (Ours)",
    "sam3d": "SAM3D",
}

# Recall counts how many camera frames a method reconstructed, which is only
# meaningful for the image-based baseline; the IMU-driven methods always emit a
# pose and run slightly faster than the camera.
RECALL_METHODS = {"sam3d"}

# Predictions further than this from any GT frame are reported as unmatched.
MATCH_WARN_MS = 50.0


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

        pred_norm = np.clip(np.linalg.norm(pred_dir, axis=-1, keepdims=True), 1e-8, None)
        gt_norm = np.clip(np.linalg.norm(gt_dir, axis=-1, keepdims=True), 1e-8, None)

        cos = np.clip(np.sum((pred_dir / pred_norm) * (gt_dir / gt_norm), axis=-1), -1.0, 1.0)
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
            pred_angle = _vec_angle(
                pred_joints[:, joint, :] - pred_joints[:, p, :],
                pred_joints[:, c, :] - pred_joints[:, joint, :],
            )
            gt_angle = _vec_angle(
                gt_joints[:, joint, :] - gt_joints[:, p, :],
                gt_joints[:, c, :] - gt_joints[:, joint, :],
            )
            errors.append(np.abs(pred_angle - gt_angle))

    if not errors:
        return 0.0
    return float(np.mean(np.concatenate(errors)))


# ── Loading helpers ───────────────────────────────────────────────────

def discover_activities(data_dir: Path) -> List[str]:
    if not data_dir.is_dir():
        return []
    return sorted(
        p.name for p in data_dir.iterdir()
        if p.is_dir() and (p / GT_FILENAME).exists()
    )


def load_joints(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load (joints (N, 22, 3), timestamps_ns) sorted by time."""
    d = np.load(npz_path)
    joints = d["joints_opti"][:, :NUM_BODY_JOINTS, :]
    ts = d["timestamps_ns"].astype(np.int64)
    n = min(joints.shape[0], ts.shape[0])
    joints, ts = joints[:n], ts[:n]
    order = np.argsort(ts, kind="stable")
    return joints[order], ts[order]


def load_camera_frames(gt_path: Path) -> Optional[int]:
    """Number of third-person camera frames in the activity, if recorded."""
    d = np.load(gt_path)
    return int(d["n_camera_frames"]) if "n_camera_frames" in d.files else None


# ── Per-activity evaluation ───────────────────────────────────────────

def evaluate_activity(
    data_dir: Path,
    activity: str,
    methods: List[str],
    compute_mpjar_flag: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Evaluate every available method on one activity."""
    gt_path = data_dir / activity / GT_FILENAME
    gt_joints, gt_ts = load_joints(gt_path)
    n_camera = load_camera_frames(gt_path)

    results: Dict[str, Dict[str, float]] = {}
    for method in methods:
        npz_path = data_dir / activity / f"{method}.npz"
        if not npz_path.exists():
            continue
        pred_joints, pred_ts = load_joints(npz_path)

        idx = nearest_indices(gt_ts, pred_ts)
        gt_matched = gt_joints[idx]
        matched = np.abs(pred_ts - gt_ts[idx]) <= MATCH_WARN_MS * 1e6

        results[method] = {
            "MPJPE": compute_mpjpe(pred_joints, gt_matched),
            "MPJAR": compute_mpjar(pred_joints, gt_matched, BODY_BONES) if compute_mpjar_flag else None,
            "JAE": compute_jae(pred_joints, gt_matched, BODY_BONES),
            "n_frames": int(len(pred_ts)),
            "matched": float(matched.mean()),
            "recall": (len(pred_ts) / n_camera)
                      if (n_camera and method in RECALL_METHODS) else None,
        }
    return results


# ── Reporting ─────────────────────────────────────────────────────────

def _aggregate(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    method: str,
    key: str,
) -> Optional[float]:
    vals = [
        res[method][key] for res in all_results.values()
        if method in res and res[method].get(key) is not None
    ]
    return float(np.mean(vals)) if vals else None


def _row(label: str, r: Dict[str, Optional[float]], has_mpjar: bool) -> str:
    cells = [f"{label:20s}", f"{r['MPJPE']:11.1f}"]
    if has_mpjar:
        cells.append(f"{r['MPJAR']:12.1f}" if r.get("MPJAR") is not None else f"{'—':>12s}")
    cells.append(f"{r['JAE']:10.1f}")
    cells.append(f"{r['matched']:8.0%}")
    cells.append(f"{r['recall']:8.1%}" if r.get("recall") is not None else f"{'—':>8s}")
    return "  ".join(cells)


def _header(has_mpjar: bool) -> str:
    cells = [f"{'Method':20s}", f"{'MPJPE (cm)':>11s}"]
    if has_mpjar:
        cells.append(f"{'MPJAR (deg)':>12s}")
    cells += [f"{'JAE (deg)':>10s}", f"{'Matched':>8s}", f"{'Recall':>8s}"]
    return "  ".join(cells)


def print_summary(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    methods: List[str],
    has_mpjar: bool,
):
    header = _header(has_mpjar)
    rule = "=" * len(header)

    print(f"\n{rule}\n  PER-ACTIVITY BREAKDOWN\n{rule}")
    for activity in sorted(all_results):
        results = all_results[activity]
        if not results:
            continue
        frames = max(r["n_frames"] for r in results.values())
        print(f"\n  {activity}  ({frames} frames)")
        print(f"  {header}")
        print(f"  {'-' * len(header)}")
        for method in methods:
            if method in results:
                print("  " + _row(METHOD_LABELS[method], results[method], has_mpjar))

    print(f"\n{rule}\n  AVERAGE OVER ALL ACTIVITIES\n{rule}")
    print(f"  {header}")
    print(f"  {'-' * len(header)}")
    for method in methods:
        if _aggregate(all_results, method, "MPJPE") is None:
            continue
        agg = {k: _aggregate(all_results, method, k)
               for k in ("MPJPE", "MPJAR", "JAE", "matched", "recall")}
        print("  " + _row(METHOD_LABELS[method], agg, has_mpjar))


# ── Export ────────────────────────────────────────────────────────────

def export_csv(
    path: Path,
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    methods: List[str],
    has_mpjar: bool,
) -> None:
    """Write one row per (activity, method), then the overall average."""
    cols = ["activity", "method", "frames", "mpjpe_cm", "jae_deg", "matched", "recall"]
    if has_mpjar:
        cols.insert(5, "mpjar_deg")

    def row_for(activity, method, r):
        row = {
            "activity": activity,
            "method": METHOD_LABELS[method],
            "frames": r["n_frames"],
            "mpjpe_cm": round(r["MPJPE"], 2),
            "jae_deg": round(r["JAE"], 2),
            "matched": round(r["matched"], 4),
            "recall": round(r["recall"], 4) if r.get("recall") is not None else "",
        }
        if has_mpjar and r.get("MPJAR") is not None:
            row["mpjar_deg"] = round(r["MPJAR"], 2)
        return row

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for activity in sorted(all_results):
            for method in methods:
                r = all_results[activity].get(method)
                if r is not None:
                    w.writerow(row_for(activity, method, r))
        for method in methods:
            if _aggregate(all_results, method, "MPJPE") is None:
                continue
            agg = {k: _aggregate(all_results, method, k)
                   for k in ("MPJPE", "MPJAR", "JAE", "matched", "recall")}
            agg["n_frames"] = sum(
                res[method]["n_frames"] for res in all_results.values() if method in res
            )
            w.writerow(row_for("ALL", method, agg))


def export_markdown(
    path: Path,
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    methods: List[str],
    has_mpjar: bool,
) -> None:
    """Write per-activity tables and the overall average in Markdown."""
    header = ["Method", "MPJPE (cm)", "JAE (deg)"]
    if has_mpjar:
        header.insert(2, "MPJAR (deg)")
    header.append("Recall")

    def md_row(method, r):
        cells = [METHOD_LABELS[method], f"{r['MPJPE']:.1f}"]
        if has_mpjar:
            cells.append(f"{r['MPJAR']:.1f}" if r.get("MPJAR") is not None else "—")
        cells.append(f"{r['JAE']:.1f}")
        cells.append(f"{r['recall']:.1%}" if r.get("recall") is not None else "—")
        return "| " + " | ".join(cells) + " |"

    lines = ["# Metrics by Activity (MPJPE cm, JAE deg)\n"]
    for activity in sorted(all_results):
        if not all_results[activity]:
            continue
        lines.append(f"### {activity}\n")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + " --- |" * len(header))
        for method in methods:
            r = all_results[activity].get(method)
            if r is not None:
                lines.append(md_row(method, r))
        lines.append("")

    lines.append("## Overall Average\n")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + " --- |" * len(header))
    for method in methods:
        if _aggregate(all_results, method, "MPJPE") is None:
            continue
        agg = {k: _aggregate(all_results, method, k)
               for k in ("MPJPE", "MPJAR", "JAE", "recall")}
        lines.append(md_row(method, agg))
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute MPJPE / MPJAR / JAE for all baselines vs OptiTrack GT."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR,
        help=f"Evaluation bundle directory. Default: {DATA_DIR}",
    )
    parser.add_argument(
        "--activities", nargs="+", default=None,
        help="Activity directory names. Default: every activity in --data-dir.",
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=list(METHOD_LABELS.keys()),
        choices=list(METHOD_LABELS.keys()),
        help=f"Methods to evaluate. Default: all ({', '.join(METHOD_LABELS)}).",
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


def main():
    args = parse_args()

    available = discover_activities(args.data_dir)
    if not available:
        raise SystemExit(
            f"No activities found under {args.data_dir}.\n"
            "Download the evaluation bundle first — see "
            "https://roshi-mocap.github.io/documentation/pipeline/evaluation.html"
        )

    activities = args.activities or available
    unknown = sorted(set(activities) - set(available))
    if unknown:
        raise SystemExit(
            f"Unknown activities: {', '.join(unknown)}\n"
            f"Available: {', '.join(available)}"
        )

    all_results = {
        activity: evaluate_activity(
            args.data_dir, activity, args.methods, compute_mpjar_flag=args.mpjar,
        )
        for activity in activities
    }

    has_mpjar = any(
        r.get("MPJAR") is not None
        for res in all_results.values() for r in res.values()
    )
    print_summary(all_results, args.methods, has_mpjar)

    if args.output is not None:
        out = args.output
        if out.suffix.lower() == ".csv":
            export_csv(out, all_results, args.methods, has_mpjar)
        else:
            if out.suffix.lower() != ".md":
                out = out.with_suffix(".md")
            export_markdown(out, all_results, args.methods, has_mpjar)
        print(f"\nWrote {out}")

    print()


if __name__ == "__main__":
    main()
