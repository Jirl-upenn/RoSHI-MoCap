"""Generate per-activity visualization videos by trimming and concatenating
existing per-dataset MP4s at the transition points from sequence_splits.json.

All methods are trimmed to the camera timeline window so they stay
synchronized in the combined 2x2 grid.

Usage:
    python evaluation/generate_activity_videos.py
    python evaluation/generate_activity_videos.py --methods sam3d imu_naive
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SPLITS_JSON = Path("evaluation/sequence_splits.json")
OUTPUT_DIR = Path("evaluation/by_activity")
RECORDINGS_DIR = Path("received_recordings")

METHOD_VIDEO = {
    "imu_naive": ("evaluation/imu_naive", "{dataset}_imu_only.mp4", "{dataset}_imu_only.npz"),
    "imu_egoallo": ("evaluation/imu_egoallo", "{dataset}_imu_egoallo.mp4", "{dataset}_imu_egoallo.npz"),
    "egoallo": ("evaluation/egoallo", "{dataset}_egoallo.mp4", "{dataset}_egoallo.npz"),
    "sam3d": ("evaluation/sam3d", "{dataset}_sam3d.mp4", "{dataset}_sam3d.npz"),
}

DATASETS = [f"dataset{i}" for i in range(1, 9)]
FPS = 30


def get_video_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def trim_video(src: Path, dst: Path, start: float, end: float | None = None):
    """Frame-accurate trim via re-encoding."""
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(src)]
    if end is not None:
        cmd += ["-t", f"{end - start:.3f}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-b:v", "8000k", str(dst)]
    subprocess.run(cmd, capture_output=True, check=True)


def concat_videos(parts: List[Path], dst: Path):
    if len(parts) == 1:
        import shutil
        shutil.copy2(parts[0], dst)
        return

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in parts:
            f.write(f"file '{p.resolve()}'\n")
        list_file = f.name

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c", "copy", str(dst),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    Path(list_file).unlink()


def load_npz_timestamps(npz_path: Path) -> np.ndarray:
    """Load timestamps that correspond 1:1 to video frames.

    Prefers ``video_timestamps_ns`` (full camera timeline) over
    ``timestamps_ns`` (which may only cover detected frames, e.g. SAM3D).
    """
    data = np.load(npz_path)
    if "video_timestamps_ns" in data:
        return data["video_timestamps_ns"]
    return data["timestamps_ns"]


def get_trim_times(
    method_ts: np.ndarray,
    window_start_ns: int,
    window_end_ns: int,
) -> Tuple[float, float]:
    """Find video-time [start, end] in a method's video that corresponds
    to the real-time window [window_start_ns, window_end_ns].

    Video frame i corresponds to method_ts[i], played at FPS."""
    idx_start = int(np.searchsorted(method_ts, window_start_ns, side="left"))
    idx_end = int(np.searchsorted(method_ts, window_end_ns, side="right"))
    # clamp
    idx_start = max(0, idx_start)
    idx_end = min(len(method_ts), idx_end)
    return idx_start / FPS, idx_end / FPS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods", nargs="+", default=list(METHOD_VIDEO.keys()),
        choices=list(METHOD_VIDEO.keys()),
    )
    args = parser.parse_args()

    with open(SPLITS_JSON) as f:
        splits = json.load(f)["datasets"]

    # Precompute per-dataset: camera eval_start, eval_end, transition_ns
    ds_windows: Dict[str, Dict[str, int]] = {}
    for ds in DATASETS:
        frames_csv = RECORDINGS_DIR / ds / "frames.csv"
        with open(frames_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        ts_cam = [int(r["utc_timestamp_ns"]) for r in rows if r.get("utc_timestamp_ns")]
        first_ts = ts_cam[0]
        last_ts = ts_cam[-1]

        calib_json = RECORDINGS_DIR / ds / "imu_calibration.json"
        with open(calib_json) as f:
            calib_dur = json.load(f)["calib_duration_sec"]

        eval_start = int(first_ts + calib_dur * 1e9)
        trans_ns = int(first_ts + splits[ds]["transition_sec"] * 1e9)
        eval_end = last_ts

        ds_windows[ds] = {
            "eval_start": eval_start,
            "eval_end": eval_end,
            "transition_ns": trans_ns,
        }

    # Build activity -> list of (dataset, "first"|"second") segments
    activities: OrderedDict[str, List[Tuple[str, str]]] = OrderedDict()
    for ds in DATASETS:
        act1 = splits[ds]["activity_1"]
        act2 = splits[ds]["activity_2"]
        activities.setdefault(act1, []).append((ds, "first"))
        activities.setdefault(act2, []).append((ds, "second"))

    # Drop slide
    activities.pop("slide", None)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        method_dir, vid_pattern, npz_pattern = METHOD_VIDEO[method]
        method_out = OUTPUT_DIR / method
        method_out.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {method} ===")

        for idx, (activity, segments) in enumerate(activities.items(), start=1):
            safe_name = activity.replace("/", "-")
            out_path = method_out / f"{idx:02d}_{safe_name}_{method}.mp4"

            tmp_parts: List[Path] = []
            ok = True

            for ds, half in segments:
                src = Path(method_dir) / vid_pattern.format(dataset=ds)
                npz = Path(method_dir) / npz_pattern.format(dataset=ds)
                if not src.exists() or not npz.exists():
                    print(f"  SKIP {activity}: missing {src} or {npz}")
                    ok = False
                    break

                method_ts = load_npz_timestamps(npz)
                w = ds_windows[ds]

                # Define the real-time window for this half
                if half == "first":
                    win_start = w["eval_start"]
                    win_end = w["transition_ns"]
                else:
                    win_start = w["transition_ns"]
                    win_end = w["eval_end"]

                # Find the corresponding video-time trim points
                trim_start, trim_end = get_trim_times(method_ts, win_start, win_end)

                # Skip empty segments (e.g. SAM3D with low recall)
                if trim_end - trim_start < 0.05:
                    continue

                tmp_path = method_out / f"_tmp_{ds}_{half}.mp4"
                trim_video(src, tmp_path, start=trim_start, end=trim_end)
                tmp_parts.append(tmp_path)

            if not ok or len(tmp_parts) == 0:
                for p in tmp_parts:
                    p.unlink(missing_ok=True)
                if ok and len(tmp_parts) == 0:
                    print(f"  SKIP {idx:02d} {activity}: no frames for {method}")
                continue

            concat_videos(tmp_parts, out_path)

            for p in tmp_parts:
                p.unlink(missing_ok=True)

            dur = get_video_duration(out_path)
            src_desc = " + ".join(f"{ds}({half})" for ds, half in segments)
            print(f"  {idx:02d} {activity}: {dur:.1f}s  [{src_desc}] -> {out_path}")

    print(f"\nDone! Videos in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
