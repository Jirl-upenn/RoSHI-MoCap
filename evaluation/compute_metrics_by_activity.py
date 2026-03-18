"""Recompute MPJPE / JAE per activity by splitting each dataset at the
transition point defined in sequence_splits.json.

Most recordings contain two activities split at a transition point.
Adjacent datasets share an activity (e.g. dataset1-second-half and
dataset2-first-half are both "stretch_boxing_bow_wave").  Some datasets
contain a single activity for the entire sequence (transition_sec = -1,
no activity_2).  Segments with the same activity name across datasets are
merged automatically (e.g. "sliding" from dataset8 + dataset11).

Usage:
    python evaluation/compute_metrics_by_activity.py
    python evaluation/compute_metrics_by_activity.py -o evaluation/metrics_by_activity.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.compute_metrics import (
    METHOD_CONFIG,
    compute_jae,
    compute_mpjpe,
    load_method_joints,
)
from evaluation.eval_utils import (
    BODY_BONES,
    RECORDINGS_DIR,
    load_calib_duration_sec,
    load_optitrack_smplx_fit,
    nearest_indices,
)

SPLITS_JSON = Path("evaluation/sequence_splits.json")
DATASETS = [f"dataset{i}" for i in range(1, 12)]
METHODS = list(METHOD_CONFIG.keys())


# ── helpers ──────────────────────────────────────────────────────────

def _load_timeline_ts(dataset_name: str) -> np.ndarray:
    frames_csv = RECORDINGS_DIR / dataset_name / "frames.csv"
    ts_list: List[int] = []
    with open(frames_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("utc_timestamp_ns"):
                ts_list.append(int(row["utc_timestamp_ns"]))
    return np.array(ts_list, dtype=np.int64)


def _transition_ns(dataset_name: str, transition_sec: float, timeline_ts: np.ndarray) -> int:
    return int(timeline_ts[0] + transition_sec * 1e9)


def _get_eval_window(
    dataset_name: str, transition_sec: float,
) -> Tuple[int, int, int]:
    """Return (eval_start_ns, eval_end_ns, transition_ns)."""
    timeline_ts = _load_timeline_ts(dataset_name)
    calib_dur = load_calib_duration_sec(dataset_name)
    eval_start = int(timeline_ts[0] + calib_dur * 1e9)
    eval_end = int(timeline_ts[-1])
    if transition_sec < 0:
        trans_ns = eval_end + 1
    else:
        trans_ns = _transition_ns(dataset_name, transition_sec, timeline_ts)
    return eval_start, eval_end, trans_ns


# ── per-dataset split evaluation ─────────────────────────────────────

def evaluate_dataset_split(
    dataset_name: str,
    transition_sec: float,
    methods: List[str],
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, float]]:
    """Evaluate each method on the halves of *dataset_name*.

    For two-activity datasets (transition_sec > 0), returns results keyed
    by "first" and "second".  For single-activity datasets
    (transition_sec < 0), the whole eval window is returned under "first".
    """
    gt = load_optitrack_smplx_fit(dataset_name)
    if gt is None:
        return {}, {}

    gt_joints = gt["joints_zup"]
    gt_ts = gt["timestamps"]
    timeline_ts = _load_timeline_ts(dataset_name)
    calib_dur = load_calib_duration_sec(dataset_name)

    eval_start = int(timeline_ts[0] + calib_dur * 1e9)
    eval_end = int(timeline_ts[-1])

    single_activity = transition_sec < 0
    if single_activity:
        trans_ns = eval_end + 1
    else:
        trans_ns = _transition_ns(dataset_name, transition_sec, timeline_ts)

    # trim GT to eval window
    gt_mask = (gt_ts >= eval_start) & (gt_ts <= eval_end)
    gt_joints = gt_joints[gt_mask]
    gt_ts = gt_ts[gt_mask]

    # camera frame counts per half
    cam_eval = timeline_ts[(timeline_ts >= eval_start) & (timeline_ts <= eval_end)]
    n_cam_first = int(np.sum(cam_eval < trans_ns))
    n_cam_second = int(np.sum(cam_eval >= trans_ns))
    dur_first = (trans_ns - eval_start) / 1e9
    dur_second = (eval_end - trans_ns) / 1e9

    split_info = {
        "n_cam_first": n_cam_first,
        "n_cam_second": n_cam_second,
        "dur_first": dur_first,
        "dur_second": dur_second,
    }

    half_results: Dict[str, Dict[str, Dict[str, float]]] = {"first": {}, "second": {}}

    halves_to_eval = [("first", lambda ts: ts < trans_ns, n_cam_first)]
    if not single_activity:
        halves_to_eval.append(("second", lambda ts: ts >= trans_ns, n_cam_second))

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
        gt_matched = gt_joints[match_idx]

        for half_name, mask_fn, n_cam in halves_to_eval:
            mask = mask_fn(pred_ts)
            pj = pred_joints[mask]
            gm = gt_matched[mask]
            if len(pj) == 0:
                continue
            mpjpe = compute_mpjpe(pj, gm)
            jae = compute_jae(pj, gm, BODY_BONES)
            recall = len(pj) / n_cam if n_cam > 0 else 1.0
            half_results[half_name][method] = {
                "MPJPE": mpjpe,
                "JAE": jae,
                "n_frames": len(pj),
                "recall": recall,
            }

    return half_results, split_info


# ── merge activity segments ──────────────────────────────────────────

def merge_method_metrics(segments: List[Dict[str, float]]) -> Dict[str, float]:
    """Weighted-average MPJPE/JAE across segments by frame count."""
    total_frames = sum(s["n_frames"] for s in segments)
    if total_frames == 0:
        return {"MPJPE": 0, "JAE": 0, "n_frames": 0, "recall": 0}
    mpjpe = sum(s["MPJPE"] * s["n_frames"] for s in segments) / total_frames
    jae = sum(s["JAE"] * s["n_frames"] for s in segments) / total_frames
    total_cam = sum(s.get("n_cam", s["n_frames"]) for s in segments)
    recall = total_frames / total_cam if total_cam > 0 else 1.0
    return {"MPJPE": mpjpe, "JAE": jae, "n_frames": total_frames, "recall": recall}


# ── export cropped / concatenated NPZ per activity ───────────────────

def export_activity_npz(
    activities: OrderedDict[str, List[Tuple[str, str]]],
    splits: Dict,
    methods: List[str],
    export_dir: Path,
) -> None:
    """Crop each dataset to its activity half (excluding calibration),
    concatenate across datasets that share the same activity, and save
    one NPZ per (activity, source).

    Output layout::

        <export_dir>/<activity>/gt.npz          – joints_zup, timestamps_ns
        <export_dir>/<activity>/<method>.npz    – joints_opti, timestamps_ns
    """
    # Cache expensive GT loads so datasets appearing in two activities
    # (first + second) don't re-run the SMPL-X forward pass.
    _gt_cache: Dict[str, Optional[dict]] = {}
    _window_cache: Dict[str, Tuple[int, int, int]] = {}

    def _ensure_cached(ds: str) -> Optional[dict]:
        if ds not in _gt_cache:
            gt = load_optitrack_smplx_fit(ds)
            if gt is None:
                _gt_cache[ds] = None
            else:
                ev_start, ev_end, t_ns = _get_eval_window(ds, splits[ds]["transition_sec"])
                gt_ts = gt["timestamps"]
                mask = (gt_ts >= ev_start) & (gt_ts <= ev_end)
                _gt_cache[ds] = {
                    "joints": gt["joints_zup"][mask],
                    "ts": gt_ts[mask],
                }
                _window_cache[ds] = (ev_start, ev_end, t_ns)
        return _gt_cache[ds]

    def _half_mask(ts: np.ndarray, half: str, trans_ns: int) -> np.ndarray:
        return ts < trans_ns if half == "first" else ts >= trans_ns

    for activity, segments in activities.items():
        act_dir = export_dir / activity
        act_dir.mkdir(parents=True, exist_ok=True)

        # ---- GT ----
        gt_j_parts: List[np.ndarray] = []
        gt_t_parts: List[np.ndarray] = []
        for ds, half in segments:
            cached = _ensure_cached(ds)
            if cached is None:
                continue
            ev_start, ev_end, trans_ns = _window_cache[ds]
            m = _half_mask(cached["ts"], half, trans_ns)
            if m.any():
                gt_j_parts.append(cached["joints"][m])
                gt_t_parts.append(cached["ts"][m])

        if gt_j_parts:
            np.savez_compressed(
                act_dir / "gt.npz",
                joints_zup=np.concatenate(gt_j_parts, axis=0),
                timestamps_ns=np.concatenate(gt_t_parts, axis=0),
            )

        # ---- each method ----
        for method in methods:
            m_j_parts: List[np.ndarray] = []
            m_t_parts: List[np.ndarray] = []
            for ds, half in segments:
                if _ensure_cached(ds) is None:
                    continue
                ev_start, ev_end, trans_ns = _window_cache[ds]
                mdata = load_method_joints(method, ds)
                if mdata is None:
                    continue
                pred_j, pred_ts = mdata
                eval_mask = (pred_ts >= ev_start) & (pred_ts <= ev_end)
                pred_j = pred_j[eval_mask]
                pred_ts = pred_ts[eval_mask]
                m = _half_mask(pred_ts, half, trans_ns)
                if m.any():
                    m_j_parts.append(pred_j[m])
                    m_t_parts.append(pred_ts[m])

            if m_j_parts:
                np.savez_compressed(
                    act_dir / f"{method}.npz",
                    joints_opti=np.concatenate(m_j_parts, axis=0),
                    timestamps_ns=np.concatenate(m_t_parts, axis=0),
                )

        n_exported = sum(1 for m in methods if (act_dir / f"{m}.npz").exists())
        gt_tag = "GT + " if gt_j_parts else ""
        print(f"  {activity}: {gt_tag}{n_exported} method(s)")


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=Path, default=Path("evaluation/metrics_by_activity.md"))
    parser.add_argument(
        "--export-dir", type=Path, default=Path("evaluation/by_activity"),
        help="Directory to save per-activity NPZ files (GT + each method).",
    )
    parser.add_argument(
        "--no-export", action="store_true",
        help="Skip NPZ export, only compute metrics.",
    )
    args = parser.parse_args()

    with open(SPLITS_JSON) as f:
        splits = json.load(f)["datasets"]

    # Collect per-half results for every dataset
    all_halves: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    all_info: Dict[str, Dict[str, float]] = {}
    for ds in DATASETS:
        ds_entry = splits.get(ds)
        if ds_entry is None:
            continue
        trans_sec = ds_entry["transition_sec"]
        half_results, info = evaluate_dataset_split(ds, trans_sec, METHODS)
        if not half_results:
            continue
        all_halves[ds] = half_results
        all_info[ds] = info

    # Build ordered activity list by merging adjacent halves.
    # Single-activity datasets (no activity_2) contribute their whole
    # eval window under "first".  Activities with the same name across
    # datasets are merged (e.g. "sliding" from dataset8 + dataset11).
    Activity = str
    activities: OrderedDict[Activity, List[Tuple[str, str]]] = OrderedDict()

    for ds in DATASETS:
        ds_entry = splits.get(ds)
        if ds_entry is None or ds not in all_halves:
            continue
        act1 = ds_entry["activity_1"]
        act2 = ds_entry.get("activity_2")
        activities.setdefault(act1, []).append((ds, "first"))
        if act2:
            activities.setdefault(act2, []).append((ds, "second"))

    # Compute per-activity metrics
    act_results: OrderedDict[str, Dict[str, Dict[str, float]]] = OrderedDict()
    act_info: OrderedDict[str, Dict[str, float]] = OrderedDict()

    for activity, segments in activities.items():
        act_results[activity] = {}
        total_cam = 0
        total_dur = 0.0
        for ds, half in segments:
            info = all_info.get(ds, {})
            total_cam += info.get(f"n_cam_{half}", 0)
            total_dur += info.get(f"dur_{half}", 0)
        act_info[activity] = {"n_cam": total_cam, "dur_s": total_dur}

        for method in METHODS:
            segs = []
            n_cam_total = 0
            for ds, half in segments:
                half_data = all_halves.get(ds, {}).get(half, {})
                if method in half_data:
                    seg = half_data[method].copy()
                    info = all_info.get(ds, {})
                    seg["n_cam"] = info.get(f"n_cam_{half}", seg["n_frames"])
                    segs.append(seg)
                    n_cam_total += seg["n_cam"]
            if segs:
                merged = merge_method_metrics(segs)
                merged["recall"] = merged["n_frames"] / n_cam_total if n_cam_total > 0 else 1.0
                act_results[activity][method] = merged

    # Export per-activity NPZ files
    if not args.no_export:
        print("Exporting per-activity NPZ files …")
        export_activity_npz(activities, splits, METHODS, args.export_dir)
        print(f"Saved to {args.export_dir}/\n")

    # Print and export
    lines: List[str] = []
    lines.append("# Metrics by Activity (MPJPE cm, JAE deg)\n")

    for activity, method_results in act_results.items():
        info = act_info[activity]
        lines.append(f"### {activity}\n")
        lines.append(f"- Eval: {info['dur_s']:.1f}s, camera frames: {int(info['n_cam'])}\n")
        lines.append("| Method | MPJPE (cm) | JAE (deg) | Recall |")
        lines.append("| --- | --- | --- | --- |")
        for method in METHODS:
            if method not in method_results:
                continue
            r = method_results[method]
            label = METHOD_CONFIG[method]["label"]
            recall_str = f"{r['recall']:.1%}" if method == "sam3d" else "—"
            lines.append(f"| {label} | {r['MPJPE']:.1f} | {r['JAE']:.1f} | {recall_str} |")
        lines.append("")

    # Overall average
    lines.append("## Overall Average\n")
    lines.append("| Method | MPJPE (cm) | JAE (deg) | Recall |")
    lines.append("| --- | --- | --- | --- |")
    for method in METHODS:
        mpjpe_vals, jae_vals, recall_vals = [], [], []
        for activity in act_results:
            if method in act_results[activity]:
                r = act_results[activity][method]
                mpjpe_vals.append(r["MPJPE"])
                jae_vals.append(r["JAE"])
                recall_vals.append(r["recall"])
        if mpjpe_vals:
            label = METHOD_CONFIG[method]["label"]
            recall_str = f"{np.mean(recall_vals):.1%}" if method == "sam3d" else "—"
            lines.append(f"| {label} | {np.mean(mpjpe_vals):.1f} | {np.mean(jae_vals):.1f} | {recall_str} |")
    lines.append("")

    # Dataset summary table (grouped by recording session)
    activity_list = list(act_results.keys())
    DATASET_GROUPS = [
        ("Dataset 1", activity_list[:4]),
        ("Dataset 2", activity_list[4:8]),
        ("Dataset 3", activity_list[8:]),
    ]
    lines.append("## Dataset Summary\n")
    lines.append("| | Dataset 1 | | Dataset 2 | | Dataset 3 | |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    lines.append("| | MPJPE (cm) | JAE (deg) | MPJPE (cm) | JAE (deg) | MPJPE (cm) | JAE (deg) |")

    for method in METHODS:
        label = METHOD_CONFIG[method]["label"]
        cells = [label]
        for _, group_activities in DATASET_GROUPS:
            mpjpe_vals = []
            jae_vals = []
            for activity in group_activities:
                if method in act_results.get(activity, {}):
                    mpjpe_vals.append(act_results[activity][method]["MPJPE"])
                    jae_vals.append(act_results[activity][method]["JAE"])
            if mpjpe_vals:
                cells.append(f"{np.mean(mpjpe_vals):.1f}")
                cells.append(f"{np.mean(jae_vals):.1f}")
            else:
                cells.append("—")
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    md_text = "\n".join(lines)
    print(md_text)

    args.output.write_text(md_text, encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
