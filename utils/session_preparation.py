"""
Session preparation: extract frames, camera.json, frames.csv, AprilTag summary.

Moved from receiver_calibrate.py to keep the receiver module focused on networking.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from utils.imu_id_mapping import REQUIRED_TAG_IDS


@dataclass(frozen=True)
class FrameMeta:
    frame_index: int
    utc_timestamp_ns: int


def _parse_intrinsics(meta: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    intr = meta.get("cameraIntrinsics") or {}
    if not isinstance(intr, dict):
        raise ValueError("metadata.json missing 'cameraIntrinsics' dict")

    try:
        fx = float(intr["m11"])
        fy = float(intr["m22"])
        cx = float(intr["m13"])
        cy = float(intr["m23"])
    except Exception as exc:
        raise ValueError("metadata.json cameraIntrinsics does not have expected keys m11/m22/m13/m23") from exc

    resolution = meta.get("resolution") or {}
    width = resolution.get("width")
    height = resolution.get("height")

    intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    if width is not None:
        intrinsics["width"] = int(width)
    if height is not None:
        intrinsics["height"] = int(height)

    return intrinsics, {"cameraIntrinsics": intr, "resolution": resolution, "fps": meta.get("fps")}


def _parse_frames(meta: Dict[str, Any]) -> List[FrameMeta]:
    frames = meta.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("metadata.json missing non-empty 'frames' list")

    out: List[FrameMeta] = []
    for entry in frames:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("frameIndex")
        ts_s = entry.get("timestampSeconds")
        if idx is None or ts_s is None:
            continue
        utc_ns = int(round(float(ts_s) * 1_000_000_000))
        out.append(FrameMeta(frame_index=int(idx), utc_timestamp_ns=utc_ns))

    if not out:
        raise ValueError("metadata.json frames did not contain frameIndex + timestampSeconds entries")

    out.sort(key=lambda x: x.frame_index)
    return out


def _compute_calibration_segment_from_metadata(
    meta: Dict[str, Any],
    *,
    required_tag_ids: List[int],
) -> Dict[str, Any]:
    """
    Compute a suggested calibration window end time from metadata.json.

    We define "calibration complete" as: the first moment when every required AprilTag ID has been
    observed at least once in the per-frame detection stream.

    Returns a JSON-serializable payload with both absolute times (timestampSeconds) and elapsed
    seconds from the start of the recording.
    """
    required = {int(t) for t in required_tag_ids}
    frames = meta.get("frames")
    if not isinstance(frames, list) or not frames:
        return {
            "method": "all_required_tags_first_seen",
            "required_tag_ids": sorted(required),
            "all_required_tags_seen": False,
            "missing_tag_ids": sorted(required),
            "reason": "metadata.json frames missing/empty",
        }

    # Recording start time (seconds since epoch)
    start_ts_s: Optional[float] = None
    for fr in frames:
        if not isinstance(fr, dict):
            continue
        ts_s = fr.get("timestampSeconds")
        if ts_s is None:
            continue
        try:
            ts_f = float(ts_s)
        except Exception:
            continue
        start_ts_s = ts_f if start_ts_s is None else min(start_ts_s, ts_f)

    if start_ts_s is None:
        return {
            "method": "all_required_tags_first_seen",
            "required_tag_ids": sorted(required),
            "all_required_tags_seen": False,
            "missing_tag_ids": sorted(required),
            "reason": "metadata.json frames missing timestampSeconds",
        }

    # First-seen time per required tag
    first_seen_ts_s: Dict[int, float] = {}
    first_seen_frame_idx: Dict[int, int] = {}

    for fr in frames:
        if not isinstance(fr, dict):
            continue
        ts_s = fr.get("timestampSeconds")
        if ts_s is None:
            continue
        try:
            ts_f = float(ts_s)
        except Exception:
            continue

        frame_idx = fr.get("frameIndex")
        frame_idx_i: Optional[int] = None
        if frame_idx is not None:
            try:
                frame_idx_i = int(frame_idx)
            except Exception:
                frame_idx_i = None

        dets = fr.get("detections")
        if not isinstance(dets, list):
            continue

        for det in dets:
            if not isinstance(det, dict):
                continue
            tag_id = det.get("id")
            if tag_id is None:
                continue
            try:
                tag_i = int(tag_id)
            except Exception:
                continue
            if tag_i not in required:
                continue

            prev = first_seen_ts_s.get(tag_i)
            if prev is None or ts_f < prev:
                first_seen_ts_s[tag_i] = ts_f
                if frame_idx_i is not None:
                    first_seen_frame_idx[tag_i] = frame_idx_i

    missing = sorted(required.difference(first_seen_ts_s.keys()))
    all_seen = len(missing) == 0

    # JSON can't have int keys, so stringify tag IDs.
    first_seen_ts_s_json = {str(k): v for k, v in sorted(first_seen_ts_s.items())}
    first_seen_elapsed_json = {
        str(k): (v - start_ts_s) for k, v in sorted(first_seen_ts_s.items())
    }

    payload: Dict[str, Any] = {
        "method": "all_required_tags_first_seen",
        "required_tag_ids": sorted(required),
        "recording_start_timestampSeconds": start_ts_s,
        "recording_start_utc_timestamp_ns": int(round(start_ts_s * 1e9)),
        "all_required_tags_seen": all_seen,
        "missing_tag_ids": missing,
        "first_seen_timestampSeconds_by_tag_id": first_seen_ts_s_json,
        "first_seen_elapsed_sec_by_tag_id": first_seen_elapsed_json,
    }

    if all_seen:
        last_tag_id, last_ts_s = max(first_seen_ts_s.items(), key=lambda kv: kv[1])
        payload.update(
            {
                "all_required_tags_seen_timestampSeconds": last_ts_s,
                "all_required_tags_seen_utc_timestamp_ns": int(round(last_ts_s * 1e9)),
                "all_required_tags_seen_elapsed_sec": float(last_ts_s - start_ts_s),
                "suggested_calib_duration_sec": float(last_ts_s - start_ts_s),
                "last_required_tag_id": int(last_tag_id),
                "last_required_tag_frameIndex": first_seen_frame_idx.get(int(last_tag_id)),
            }
        )

    return payload


def prepare_session(session_dir: Path) -> None:
    """
    Prepare a received iPhone recording folder for the calibration pipeline.
    Extracts frames, creates meta/camera.json, frames.csv, and color_apriltag/detection_summary.json.
    Skips if already prepared.
    """
    session_dir = session_dir.resolve()
    color_dir = session_dir / "color"
    already_extracted = color_dir.exists() and any(color_dir.glob("*.jpg"))

    video_path = session_dir / "video.mp4"
    meta_path = session_dir / "metadata.json"

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    meta = json.loads(meta_path.read_text())
    fps = meta.get("fps")
    frames = _parse_frames(meta)
    intrinsics, raw_summary = _parse_intrinsics(meta)
    calib_segment = _compute_calibration_segment_from_metadata(meta, required_tag_ids=REQUIRED_TAG_IDS)

    # Validate video matches metadata length
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if video_frame_count and video_frame_count != len(frames):
        raise ValueError(
            f"Video frame_count ({video_frame_count}) != metadata frames ({len(frames)}). "
            "Refusing to proceed to avoid misalignment."
        )

    # Output dirs
    meta_dir = session_dir / "meta"
    frames_csv = session_dir / "frames.csv"
    color_apriltag_dir = session_dir / "color_apriltag"
    camera_json_path = meta_dir / "camera.json"
    calib_json_path = meta_dir / "calibration_segment.json"
    det_summary_path = color_apriltag_dir / "detection_summary.json"

    color_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    color_apriltag_dir.mkdir(parents=True, exist_ok=True)

    # Write meta/camera.json
    camera_payload = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "color_intrinsics": intrinsics,
        "source": "ROSHI iPhone metadata.json",
        "source_summary": raw_summary,
        "video_fps": video_fps,
        "metadata_fps": fps,
        "calibration_segment": calib_segment,
    }
    # Preserve any extra keys already present in camera.json, but refresh the known keys above.
    if camera_json_path.exists():
        try:
            existing = json.loads(camera_json_path.read_text())
            if isinstance(existing, dict):
                existing.update(camera_payload)
                camera_payload = existing
        except Exception:
            pass
    camera_json_path.write_text(json.dumps(camera_payload, indent=2))
    calib_json_path.write_text(json.dumps(calib_segment, indent=2))

    # Extract frames (if needed) + always (re)write frames.csv and detection_summary.json so
    # downstream steps can rely on the derived files even if the session was prepared earlier.
    cap = None
    if not already_extracted:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    images_summary: List[Dict[str, Any]] = []

    with open(frames_csv, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["frame_id", "utc_timestamp_ns", "color_path"])
        writer.writeheader()

        for i, fmeta in enumerate(frames):
            # Extract the RGB frame if we haven't already.
            base = f"frame_{fmeta.frame_index:06d}_{fmeta.utc_timestamp_ns}"
            out_path = color_dir / f"{base}_color.jpg"
            if cap is not None:
                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.release()
                    raise RuntimeError(f"Failed to read frame {i} from {video_path}")
                if not cv2.imwrite(str(out_path), frame, encode_params):
                    cap.release()
                    raise RuntimeError(f"Failed to write image: {out_path}")

            rel = out_path.relative_to(session_dir)
            writer.writerow(
                {
                    "frame_id": fmeta.frame_index,
                    "utc_timestamp_ns": fmeta.utc_timestamp_ns,
                    "color_path": str(rel),
                }
            )

            # Extract AprilTag detections from metadata
            meta_frame = (
                meta.get("frames", [])[i]
                if isinstance(meta.get("frames"), list) and i < len(meta.get("frames", []))
                else {}
            )
            raw_dets = meta_frame.get("detections", []) if isinstance(meta_frame, dict) else []
            dets_out: List[Dict[str, Any]] = []
            for det in raw_dets if isinstance(raw_dets, list) else []:
                if not isinstance(det, dict):
                    continue
                tag_id = det.get("id")
                rot = det.get("rotation") or {}
                pos = det.get("position") or {}
                if tag_id is None or not isinstance(rot, dict) or not isinstance(pos, dict):
                    continue
                try:
                    R = [
                        [float(rot["m11"]), float(rot["m12"]), float(rot["m13"])],
                        [float(rot["m21"]), float(rot["m22"]), float(rot["m23"])],
                        [float(rot["m31"]), float(rot["m32"]), float(rot["m33"])],
                    ]
                    t = [float(pos["x"]), float(pos["y"]), float(pos["z"])]
                except Exception:
                    continue

                center = det.get("center") or {}
                corners = det.get("corners") or []
                center_xy = (
                    [float(center["x"]), float(center["y"])]
                    if isinstance(center, dict) and "x" in center and "y" in center
                    else None
                )
                corners_xy = [
                    [float(c["x"]), float(c["y"])]
                    for c in corners
                    if isinstance(c, dict) and "x" in c and "y" in c
                ]

                dets_out.append(
                    {
                        "tag_id": int(tag_id),
                        "center": center_xy,
                        "corners": corners_xy,
                        "rotation_matrix": R,
                        "translation": t,
                        "distance": float(det.get("distance"))
                        if det.get("distance") is not None
                        else None,
                        "camera_frame": "opencv_like",
                        "is_mirrored": False,
                    }
                )

            images_summary.append(
                {"filename": out_path.name, "num_detections": len(dets_out), "detections": dets_out}
            )

    if cap is not None:
        cap.release()

    # Write AprilTag detection summary
    summary_payload = {
        "input_directory": str(color_dir),
        "output_directory": str(color_apriltag_dir),
        "total_images": len(frames),
        "images": images_summary,
        "camera_params": [intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]],
        "source": "ROSHI iPhone metadata.json",
    }
    det_summary_path.write_text(json.dumps(summary_payload, indent=2))

    msg = "✅ Session prepared" if not already_extracted else "✅ Session updated"
    print(f"{msg}: {len(frames)} frames -> {color_dir}")
