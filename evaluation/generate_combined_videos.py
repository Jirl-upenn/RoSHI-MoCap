"""Generate side-by-side 3x2 comparison videos for each activity.

Layout:
  ┌───────────────┬───────────────┬───────────────┐
  │  RGB Video    │  IMU-only     │ IMU + EgoAllo │
  ├───────────────┼───────────────┼───────────────┤
  │  EgoAllo      │ IMU-TTO (Ours)│    SAM3D      │
  └───────────────┴───────────────┴───────────────┘

All panels are synchronized by UTC timestamp and exclude calibration.
Activities spanning multiple datasets are concatenated.

Usage:
    python evaluation/generate_combined_videos.py
    python evaluation/generate_combined_videos.py --activities tennis sliding
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SPLITS_JSON = Path("evaluation/sequence_splits.json")
OUTPUT_DIR = Path("evaluation/by_activity/comparison")
RECORDINGS_DIR = Path("received_recordings")

PANELS = [
    {
        "key": "rgb",
        "label": "RGB Video",
    },
    {
        "key": "imu_naive",
        "dir": "evaluation/imu_naive",
        "vid": "{ds}_imu_only.mp4",
        "npz": "{ds}_imu_only.npz",
        "label": "IMU-only (naive)",
    },
    {
        "key": "imu_egoallo",
        "dir": "evaluation/imu_egoallo",
        "vid": "{ds}_imu_egoallo.mp4",
        "npz": "{ds}_imu_egoallo.npz",
        "label": "IMU + EgoAllo",
    },
    {
        "key": "egoallo",
        "dir": "evaluation/egoallo",
        "vid": "{ds}_egoallo.mp4",
        "npz": "{ds}_egoallo.npz",
        "label": "EgoAllo",
    },
    {
        "key": "imu_tto",
        "dir": "evaluation/imu_tto",
        "vid": "{ds}_imu_tto.mp4",
        "npz": "{ds}_imu_tto.npz",
        "label": "IMU-TTO (Ours)",
    },
    {
        "key": "sam3d",
        "dir": "evaluation/sam3d",
        "vid": "{ds}_sam3d.mp4",
        "npz": "{ds}_sam3d.npz",
        "label": "SAM3D",
    },
]

DATASETS = [f"dataset{i}" for i in range(1, 12)]
FPS = 30
CELL_W, CELL_H = 480, 360


# ── ffmpeg helpers ───────────────────────────────────────────────────

def _probe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip())


def _trim(src: Path, dst: Path, start_s: float, end_s: float) -> bool:
    dur = end_s - start_s
    if dur < 0.03:
        return False
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-i", str(src),
         "-t", f"{dur:.3f}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-b:v", "4000k",
         "-an", str(dst)],
        capture_output=True, check=True,
    )
    return True


def _black(dst: Path, duration: float):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", f"color=c=black:s={CELL_W}x{CELL_H}:d={duration:.3f}:r={FPS}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(dst)],
        capture_output=True, check=True,
    )


def _concat(parts: List[Path], dst: Path):
    if len(parts) == 1:
        shutil.copy2(parts[0], dst)
        return
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in parts:
            f.write(f"file '{p.resolve()}'\n")
        list_file = f.name
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", list_file, "-c", "copy", str(dst)],
        capture_output=True, check=True,
    )
    Path(list_file).unlink()


def _compose_grid(
    panel_vids: List[Path],
    labels: List[str],
    dst: Path,
    duration: float,
    title: str,
) -> bool:
    """Build a labelled 3x2 grid from 6 input videos."""
    cmd = ["ffmpeg", "-y"]
    for p in panel_vids:
        cmd += ["-i", str(p)]

    filters = []
    for i, label in enumerate(labels):
        safe = label.replace(":", "\\:").replace("'", "\\'")
        filters.append(
            f"[{i}:v]scale={CELL_W}:{CELL_H}:force_original_aspect_ratio=decrease,"
            f"pad={CELL_W}:{CELL_H}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1,"
            f"drawtext=text='{safe}':fontsize=20:fontcolor=white:"
            f"borderw=2:bordercolor=black:x=10:y=10"
            f"[v{i}]"
        )

    layout = (
        f"0_0|{CELL_W}_0|{CELL_W * 2}_0|"
        f"0_{CELL_H}|{CELL_W}_{CELL_H}|{CELL_W * 2}_{CELL_H}"
    )
    safe_title = title.replace(":", "\\:").replace("'", "\\'")
    filters.append(
        f"[v0][v1][v2][v3][v4][v5]xstack=inputs=6:layout={layout},"
        f"drawtext=text='{safe_title}':fontsize=24:fontcolor=white:"
        f"borderw=2:bordercolor=black:x=(w-text_w)/2:y={CELL_H - 30}"
        f"[out]"
    )

    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[out]",
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-b:v", "8000k",
        "-an", str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    ffmpeg error: {r.stderr[-400:]}")
        return False
    return True


# ── timestamp helpers ────────────────────────────────────────────────

def _load_cam_ts(ds: str) -> np.ndarray:
    frames_csv = RECORDINGS_DIR / ds / "frames.csv"
    ts: List[int] = []
    with open(frames_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("utc_timestamp_ns"):
                ts.append(int(row["utc_timestamp_ns"]))
    return np.array(ts, dtype=np.int64)


def _frame_range(ts: np.ndarray, win_start: int, win_end: int) -> Tuple[float, float]:
    """Map a UTC window to video-time [start_s, end_s] (frame-index / FPS)."""
    i0 = max(0, int(np.searchsorted(ts, win_start, side="left")))
    i1 = min(len(ts), int(np.searchsorted(ts, win_end, side="right")))
    return i0 / FPS, i1 / FPS


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activities", nargs="+", default=None,
                        help="Only generate for these activities (default: all).")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    with open(SPLITS_JSON) as f:
        splits = json.load(f)["datasets"]

    # ── precompute per-dataset eval windows ──
    ds_windows: Dict[str, Dict[str, int]] = {}
    ds_cam_ts: Dict[str, np.ndarray] = {}

    for ds in DATASETS:
        frames_csv = RECORDINGS_DIR / ds / "frames.csv"
        if not frames_csv.exists():
            continue
        cam_ts = _load_cam_ts(ds)
        ds_cam_ts[ds] = cam_ts

        calib_json = RECORDINGS_DIR / ds / "imu_calibration.json"
        with open(calib_json) as f:
            calib_dur = json.load(f)["calib_duration_sec"]

        eval_start = int(cam_ts[0] + calib_dur * 1e9)
        eval_end = int(cam_ts[-1])

        ds_entry = splits.get(ds, {})
        trans_sec = ds_entry.get("transition_sec", -1)
        trans_ns = eval_end + 1 if trans_sec < 0 else int(cam_ts[0] + trans_sec * 1e9)

        ds_windows[ds] = {
            "eval_start": eval_start,
            "eval_end": eval_end,
            "transition_ns": trans_ns,
        }

    # ── build activity -> segments ──
    activities: OrderedDict[str, List[Tuple[str, str]]] = OrderedDict()
    for ds in DATASETS:
        ds_entry = splits.get(ds)
        if ds_entry is None or ds not in ds_windows:
            continue
        activities.setdefault(ds_entry["activity_1"], []).append((ds, "first"))
        act2 = ds_entry.get("activity_2")
        if act2:
            activities.setdefault(act2, []).append((ds, "second"))

    if args.activities:
        activities = OrderedDict(
            (a, s) for a, s in activities.items() if a in args.activities
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── generate each activity ──
    for idx, (activity, segments) in enumerate(activities.items(), start=1):
        safe_name = activity.replace("/", "-")
        out_path = args.output_dir / f"{idx:02d}_{safe_name}.mp4"
        print(f"\n[{idx:02d}] {activity}")

        segment_grids: List[Path] = []

        for ds, half in segments:
            w = ds_windows.get(ds)
            if w is None:
                continue

            if half == "first":
                win_start, win_end = w["eval_start"], w["transition_ns"]
            else:
                win_start, win_end = w["transition_ns"], w["eval_end"]

            ref_dur = (win_end - win_start) / 1e9
            if ref_dur < 0.1:
                continue

            panel_paths: List[Path] = []
            panel_labels: List[str] = []
            missing: List[str] = []

            for panel in PANELS:
                key = panel["key"]
                tmp = args.output_dir / f"_tmp_{ds}_{half}_{key}.mp4"

                if key == "rgb":
                    rgb_path = RECORDINGS_DIR / ds / "video.mp4"
                    cam_ts = ds_cam_ts.get(ds)
                    if rgb_path.exists() and cam_ts is not None:
                        t0, t1 = _frame_range(cam_ts, win_start, win_end)
                        if not _trim(rgb_path, tmp, t0, t1):
                            _black(tmp, ref_dur)
                    else:
                        _black(tmp, ref_dur)
                        missing.append(key)
                else:
                    vid = Path(panel["dir"]) / panel["vid"].format(ds=ds)
                    npz = Path(panel["dir"]) / panel["npz"].format(ds=ds)
                    if vid.exists() and npz.exists():
                        npz_data = np.load(npz)
                        method_ts = npz_data["video_timestamps_ns"] if "video_timestamps_ns" in npz_data else npz_data["timestamps_ns"]
                        t0, t1 = _frame_range(method_ts, win_start, win_end)
                        if not _trim(vid, tmp, t0, t1):
                            _black(tmp, ref_dur)
                    else:
                        _black(tmp, ref_dur)
                        if not vid.exists():
                            missing.append(f"{key} (no .mp4)")

                panel_paths.append(tmp)
                panel_labels.append(panel["label"])

            if missing:
                print(f"    {ds}({half}): missing {', '.join(missing)} -> black panel")

            durations = [_probe_duration(p) for p in panel_paths]
            shortest = min(durations)

            grid = args.output_dir / f"_grid_{ds}_{half}.mp4"
            title = f"{activity}  [{ds} {half}]"
            if _compose_grid(panel_paths, panel_labels, grid, shortest, title):
                segment_grids.append(grid)

            for p in panel_paths:
                p.unlink(missing_ok=True)

        if not segment_grids:
            print("    No segments produced")
            continue

        _concat(segment_grids, out_path)
        for p in segment_grids:
            p.unlink(missing_ok=True)

        dur = _probe_duration(out_path)
        src_desc = " + ".join(f"{ds}({half})" for ds, half in segments)
        print(f"    -> {out_path} ({dur:.1f}s) [{src_desc}]")

    print(f"\nDone! Comparison videos in {args.output_dir}/")


if __name__ == "__main__":
    main()
