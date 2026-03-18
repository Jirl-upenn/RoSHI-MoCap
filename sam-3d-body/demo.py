# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import json
import os
from glob import glob
from pathlib import Path

try:
    import pyrootutils
except ImportError:  # pragma: no cover
    pyrootutils = None

if pyrootutils is not None:
    root = pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git", "pyproject.toml", ".sl"],
        pythonpath=True,
        dotenv=True,
    )
else:
    # Fallback: make sure the package is importable when running from source without pyrootutils.
    import sys

    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import cv2
import numpy as np
import torch
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import visualize_sample, visualize_sample_together
from tqdm import tqdm


def _infer_camera_json_path(image_folder: str, explicit_path: str) -> str:
    if explicit_path:
        return explicit_path

    folder = Path(image_folder).resolve()
    candidates = [
        folder / "meta" / "camera.json",
        folder.parent / "meta" / "camera.json",
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return ""


def _load_camera_intrinsics(camera_json_path: str, device: torch.device):
    with open(camera_json_path, "r") as f:
        data = json.load(f)
    intr = data.get("color_intrinsics", data)
    required = ["fx", "fy", "cx", "cy"]
    if not all(k in intr for k in required):
        raise ValueError(f"Camera intrinsics file missing keys {required}: {camera_json_path}")
    cam_tensor = torch.tensor(
        [
            [
                [float(intr["fx"]), 0.0, float(intr["cx"])],
                [0.0, float(intr["fy"]), float(intr["cy"])],
                [0.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )
    return cam_tensor


def _select_subject(outputs):
    """Keep only the primary subject from multiple detections.

    Selection strategy (designed for single-subject calibration recordings):
      1. Largest bounding-box area  (subject dominates the frame)
      2. Tie-break by closest to camera  (smallest pred_cam_t z-depth)
    """
    if len(outputs) <= 1:
        return outputs

    def _bbox_area(det):
        b = det["bbox"]
        return max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))

    def _depth(det):
        cam_t = det.get("pred_cam_t")
        if cam_t is not None:
            return float(cam_t[2])
        return float("inf")

    best = max(outputs, key=lambda d: (_bbox_area(d), -_depth(d)))
    dropped = len(outputs) - 1
    print(f"  subject_only: kept 1/{len(outputs)} detections (dropped {dropped} background)")
    return [best]


def main(args):
    if args.output_folder == "":
        output_folder = os.path.join("./output", os.path.basename(args.image_folder))
    else:
        output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    # Use command-line args or environment variables
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    # Initialize sam-3d-body model and other optional modules
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=mhr_path
    )

    human_detector, human_segmentor, fov_estimator = None, None, None
    if args.detector_name:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=detector_path
        )
    if len(segmentor_path):
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=args.segmentor_name, device=device, path=segmentor_path
        )
    if args.fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.bmp",
        "*.tiff",
        "*.webp",
    ]
    images_list = sorted(
        [
            image
            for ext in image_extensions
            for image in glob(os.path.join(args.image_folder, ext))
        ]
    )

    camera_json_path = _infer_camera_json_path(args.image_folder, args.camera_json)
    cam_int_tensor = None
    if len(camera_json_path):
        try:
            cam_int_tensor = _load_camera_intrinsics(camera_json_path, device)
            print(f"Using camera intrinsics from {camera_json_path}")
        except (OSError, ValueError) as exc:
            print(f"Warning: failed to load camera intrinsics ({exc}). Falling back to default FOV.")

    data_output_folder = args.data_folder
    if data_output_folder:
        os.makedirs(data_output_folder, exist_ok=True)

    for image_path in tqdm(images_list):
        outputs = estimator.process_one_image(
            image_path,
            bbox_thr=args.bbox_thresh,
            use_mask=args.use_mask,
            cam_int=cam_int_tensor,
        )

        if args.subject_only:
            outputs = _select_subject(outputs)

        img = cv2.imread(image_path)
        base = os.path.basename(image_path)[:-4]

        if not outputs:
            cv2.imwrite(f"{output_folder}/{base}.jpg", img)
            if data_output_folder:
                np.savez_compressed(
                    os.path.join(data_output_folder, f"{base}.npz"),
                    data=np.array([], dtype=object),
                )
            continue

        rend_img = visualize_sample_together(img, outputs, estimator.faces)
        cv2.imwrite(
            f"{output_folder}/{base}.jpg",
            rend_img.astype(np.uint8),
        )
        if data_output_folder:
            npz_path = os.path.join(data_output_folder, f"{base}.npz")
            np.savez_compressed(npz_path, data=np.array(outputs, dtype=object))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Demo - Single Image Human Mesh Recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python demo.py --image_folder ./images --checkpoint_path ./checkpoints/model.ckpt

                Environment Variables:
                SAM3D_MHR_PATH: Path to MHR asset
                SAM3D_DETECTOR_PATH: Path to human detection model folder
                SAM3D_SEGMENTOR_PATH: Path to human segmentation model folder
                SAM3D_FOV_PATH: Path to fov estimation model folder
                """,
    )
    parser.add_argument(
        "--image_folder",
        required=True,
        type=str,
        help="Path to folder containing input images",
    )
    parser.add_argument(
        "--output_folder",
        default="",
        type=str,
        help="Path to output folder (default: ./output/<image_folder_name>)",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to SAM 3D Body model checkpoint",
    )
    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Human detection model for demo (Default `vitdet`, add your favorite detector if needed).",
    )
    parser.add_argument(
        "--segmentor_name",
        default="sam2",
        type=str,
        help="Human segmentation model for demo (Default `sam2`, add your favorite segmentor if needed).",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed).",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--segmentor_path",
        default="",
        type=str,
        help="Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH)",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)",
    )
    parser.add_argument(
        "--mhr_path",
        default="",
        type=str,
        help="Path to MoHR/assets folder (or set SAM3D_mhr_path)",
    )
    parser.add_argument(
        "--camera_json",
        default="",
        type=str,
        help="Path to camera.json (default: auto-detect <image_folder>/meta/camera.json or its parent).",
    )
    parser.add_argument(
        "--bbox_thresh",
        default=0.8,
        type=float,
        help="Bounding box detection threshold",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=False,
        help="Use mask-conditioned prediction (segmentation mask is automatically generated from bbox)",
    )
    parser.add_argument(
        "--data_folder",
        default="",
        type=str,
        help="Optional folder to save raw SAM3D outputs (.npz). Load with numpy.load(..., allow_pickle=True).",
    )
    parser.add_argument(
        "--subject_only",
        action="store_true",
        default=False,
        help="Keep only the primary subject per frame (largest bbox, closest to camera). "
             "Filters out background people entering the scene.",
    )
    args = parser.parse_args()

    main(args)
