#!/usr/bin/env python3
"""
Convert MHR mesh vertices from body_data npz files to SMPL parameters.

Usage:
    pixi run python convert_mhr_to_smpl.py \
        --input /path/to/body_data \
        --output /path/to/output \
        --smplx /path/to/SMPLX_NEUTRAL.npz
"""

# IMPORTANT: pymomentum must be imported BEFORE numpy to avoid segfault
# due to shared library conflicts (OpenBLAS/MKL)
import pymomentum.geometry  # noqa: F401 - must be first

import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from conversion import Conversion
from mhr.mhr import MHR


def load_mhr_data_from_npz(npz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MHR vertices, camera translation, joints and rotations from a body_data npz file.
    
    Args:
        npz_path: Path to the npz file
        
    Returns:
        Tuple of (vertices, cam_t, joints, joint_rots):
            - vertices: MHR vertices array of shape (18439, 3)
            - cam_t: Camera translation of shape (3,)
            - joints: Joint coordinates of shape (127, 3) or None
            - joint_rots: Joint rotations of shape (127, 3, 3) or None
    """
    data = np.load(npz_path, allow_pickle=True)
    detections = data['data']
    
    # Handle multiple detections - take the first one (primary/largest detection)
    if detections.size == 0:
        return None, None, None, None
    elif detections.size == 1:
        inner = detections.item()
    else:
        # Multiple people detected - take first (usually the main subject)
        inner = detections[0]
    
    vertices = inner['pred_vertices']
    cam_t = inner.get('pred_cam_t', np.zeros(3, dtype=np.float32))
    joints = inner.get('pred_joint_coords', None)
    joint_rots = inner.get('pred_global_rots', None)
    return vertices, cam_t, joints, joint_rots


def main():
    parser = argparse.ArgumentParser(
        description="Convert MHR mesh vertices to SMPL parameters"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to input body_data directory containing npz files"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Path to output directory for SMPL results"
    )
    parser.add_argument(
        "--smplx",
        type=str,
        required=True,
        help="Path to SMPLX model file (e.g., SMPLX_NEUTRAL.npz)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--single-identity",
        action="store_true",
        help="Use single identity across all frames"
    )
    parser.add_argument(
        "--save-mesh",
        action="store_true",
        help="Also save SMPL meshes as PLY files"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Find all npz files
    npz_pattern = os.path.join(args.input, "frame_*_color.npz")
    npz_files = sorted(glob(npz_pattern))
    
    if not npz_files:
        print(f"No npz files found in {args.input}")
        return
    
    print(f"Found {len(npz_files)} npz files")
    
    # Initialize models
    print(f"Loading models on {args.device}...")
    device = torch.device(args.device)
    
    mhr_model = MHR.from_files(lod=1, device=device)
    smplx_model = smplx.SMPLX(
        model_path=args.smplx,
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
    ).to(device)
    
    # Create converter
    converter = Conversion(
        mhr_model=mhr_model,
        smpl_model=smplx_model,
        method="pytorch",
        batch_size=args.batch_size,
    )
    
    # Load all MHR vertices, camera translations, and joint data
    print("Loading MHR vertices from npz files...")
    all_vertices = []
    all_cam_t = []
    all_mhr_joints = []
    all_mhr_joint_rots = []
    frame_names = []
    
    skipped = 0
    for npz_path in tqdm(npz_files, desc="Loading"):
        vertices, cam_t, joints, joint_rots = load_mhr_data_from_npz(npz_path)
        if vertices is None:
            skipped += 1
            continue
        all_vertices.append(vertices)
        all_cam_t.append(cam_t)
        all_mhr_joints.append(joints)
        all_mhr_joint_rots.append(joint_rots)
        frame_names.append(Path(npz_path).stem)
    if skipped:
        print(f"Skipped {skipped} frames with no detections")
    
    all_cam_t = np.stack(all_cam_t, axis=0)  # [N, 3]
    # Stack MHR joints and rotations if available
    has_mhr_joints = all_mhr_joints[0] is not None
    has_mhr_joint_rots = all_mhr_joint_rots[0] is not None
    if has_mhr_joints:
        all_mhr_joints = np.stack(all_mhr_joints, axis=0)
    if has_mhr_joint_rots:
        all_mhr_joint_rots = np.stack(all_mhr_joint_rots, axis=0)
    
    # Stack vertices: [N, 18439, 3]
    all_vertices = np.stack(all_vertices, axis=0)
    print(f"Loaded vertices shape: {all_vertices.shape}")
    
    # IMPORTANT: Sam3D outputs MHR vertices in METERS, but the conversion tool
    # expects CENTIMETERS (see conversion.py line 462-464, 505-507).
    # The tool internally converts: input_cm * 0.01 -> output_m
    # So we need to scale our input from meters to centimeters first.
    all_vertices_cm = all_vertices * 100.0
    print(f"Scaled vertices from meters to centimeters for conversion")
    
    # Convert to tensor
    mhr_vertices = torch.from_numpy(all_vertices_cm).float().to(device)
    
    # Perform conversion
    print("Converting MHR to SMPL...")
    result = converter.convert_mhr2smpl(
        mhr_vertices=mhr_vertices,
        single_identity=args.single_identity,
        return_smpl_meshes=args.save_mesh,
        return_smpl_parameters=True,
        return_smpl_vertices=True,
        return_fitting_errors=True,
        batch_size=args.batch_size,
    )
    
    # Save results
    print("Saving results...")
    
    # Save parameters as a single npz file (will add joints and full_pose later)
    params_to_save = {}
    for key, value in result.result_parameters.items():
        if isinstance(value, torch.Tensor):
            params_to_save[key] = value.detach().cpu().numpy()
        else:
            params_to_save[key] = value
    params_to_save['frame_names'] = np.array(frame_names)
    
    # Save vertices (keep in same units as input after internal conversion)
    if result.result_vertices is not None:
        vertices_output_path = os.path.join(args.output, "smpl_vertices.npy")
        np.save(vertices_output_path, result.result_vertices)
        print(f"Saved vertices to {vertices_output_path}")
    
    # Save fitting errors
    if result.result_errors is not None:
        errors_output_path = os.path.join(args.output, "fitting_errors.npy")
        np.save(errors_output_path, result.result_errors)
        print(f"Saved fitting errors to {errors_output_path}")
        print(f"Mean fitting error: {np.mean(result.result_errors):.6f}")
    
    # Save individual meshes if requested
    if args.save_mesh and result.result_meshes:
        mesh_dir = os.path.join(args.output, "meshes")
        os.makedirs(mesh_dir, exist_ok=True)
        for i, (mesh, frame_name) in enumerate(zip(result.result_meshes, frame_names)):
            mesh_path = os.path.join(mesh_dir, f"{frame_name}_smpl.ply")
            mesh.export(mesh_path)
        print(f"Saved {len(result.result_meshes)} meshes to {mesh_dir}")
    
    # Use model.forward() to compute joints and full_pose for each frame
    print("Computing joints and full_pose using SMPL forward pass...")
    all_joints = []
    all_full_pose = []
    
    # Process in batches
    for batch_start in tqdm(range(0, len(frame_names), args.batch_size), desc="Forward pass"):
        batch_end = min(batch_start + args.batch_size, len(frame_names))
        batch_indices = range(batch_start, batch_end)
        
        # Prepare batch inputs
        batch_global_orient = torch.stack([
            torch.from_numpy(result.result_parameters['global_orient'][i] if isinstance(result.result_parameters['global_orient'], np.ndarray) 
                           else result.result_parameters['global_orient'][i].detach().cpu().numpy()).float()
            for i in batch_indices
        ]).to(device)
        
        batch_body_pose = torch.stack([
            torch.from_numpy(result.result_parameters['body_pose'][i] if isinstance(result.result_parameters['body_pose'], np.ndarray)
                           else result.result_parameters['body_pose'][i].detach().cpu().numpy()).float()
            for i in batch_indices
        ]).to(device)
        
        batch_betas = torch.stack([
            torch.from_numpy(result.result_parameters['betas'][i] if isinstance(result.result_parameters['betas'], np.ndarray)
                           else result.result_parameters['betas'][i].detach().cpu().numpy()).float()
            for i in batch_indices
        ]).to(device)
        
        # Also batch hand poses and expression if available
        batch_size = len(batch_indices)
        
        def get_batch_param(key, default_shape):
            if key in result.result_parameters:
                param = result.result_parameters[key]
                if isinstance(param, torch.Tensor):
                    param = param.detach().cpu().numpy()
                return torch.stack([
                    torch.from_numpy(param[i]).float() for i in batch_indices
                ]).to(device)
            else:
                return torch.zeros(batch_size, *default_shape).float().to(device)
        
        batch_left_hand_pose = get_batch_param('left_hand_pose', (45,))
        batch_right_hand_pose = get_batch_param('right_hand_pose', (45,))
        batch_expression = get_batch_param('expression', (10,))
        batch_jaw_pose = get_batch_param('jaw_pose', (3,))
        batch_leye_pose = get_batch_param('leye_pose', (3,))
        batch_reye_pose = get_batch_param('reye_pose', (3,))
        batch_transl = get_batch_param('transl', (3,))  # Important: include translation!
        
        # Forward pass
        with torch.no_grad():
            output = smplx_model(
                global_orient=batch_global_orient,
                body_pose=batch_body_pose,
                betas=batch_betas,
                left_hand_pose=batch_left_hand_pose,
                right_hand_pose=batch_right_hand_pose,
                expression=batch_expression,
                jaw_pose=batch_jaw_pose,
                leye_pose=batch_leye_pose,
                reye_pose=batch_reye_pose,
                transl=batch_transl,  # Include transl to match result.result_vertices
                return_full_pose=True,
            )
        
        # Collect results
        all_joints.append(output.joints.detach().cpu().numpy())
        all_full_pose.append(output.full_pose.detach().cpu().numpy())
    
    # Concatenate all batches
    all_joints = np.concatenate(all_joints, axis=0)  # (N, 127, 3) - includes transl, matches vertices
    all_full_pose = np.concatenate(all_full_pose, axis=0)  # (N, 165) - local rotations (axis-angle)
    
    print(f"Computed joints shape: {all_joints.shape} (127 = 55 skeleton + 72 landmarks)")
    print(f"Computed full_pose shape: {all_full_pose.shape}")
    
    # Convert full_pose (axis-angle) to rotation matrices
    # full_pose: (N, 165) -> (N, 55, 3) -> (N, 55, 3, 3)
    print("Converting full_pose to rotation matrices...")
    num_frames = all_full_pose.shape[0]
    all_full_pose_reshaped = all_full_pose.reshape(num_frames, 55, 3)  # (N, 55, 3)
    all_joint_rotations = np.zeros((num_frames, 55, 3, 3), dtype=np.float32)
    for i in range(num_frames):
        all_joint_rotations[i] = Rotation.from_rotvec(all_full_pose_reshaped[i]).as_matrix()
    print(f"Computed joint_rotations shape: {all_joint_rotations.shape}")
    
    # Now save the main parameters file with joints, full_pose, and joint_rotations
    params_to_save['joints'] = all_joints  # (N, 127, 3) joints from forward (with transl applied)
    params_to_save['full_pose'] = all_full_pose  # (N, 165) local axis-angle rotations
    params_to_save['joint_rotations'] = all_joint_rotations  # (N, 55, 3, 3) rotation matrices
    params_to_save['cam_t'] = all_cam_t  # (N, 3) camera translation
    
    params_output_path = os.path.join(args.output, "smpl_parameters.npz")
    np.savez(params_output_path, **params_to_save)
    print(f"Saved parameters to {params_output_path}")
    
    # Also save per-frame parameters for convenience
    per_frame_dir = os.path.join(args.output, "per_frame")
    os.makedirs(per_frame_dir, exist_ok=True)
    
    for i, frame_name in enumerate(tqdm(frame_names, desc="Saving per-frame")):
        frame_params = {}
        for key, value in result.result_parameters.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            if isinstance(value, np.ndarray) and len(value.shape) > 0:
                if value.shape[0] == len(frame_names):
                    frame_params[key] = value[i]
                else:
                    frame_params[key] = value  # Shared parameter
            else:
                frame_params[key] = value
        
        if result.result_vertices is not None:
            frame_params['vertices'] = result.result_vertices[i]
        
        # Add camera translation from original MHR data (already in meters)
        frame_params['cam_t'] = all_cam_t[i]
        
        # Add joints (from forward with transl applied), full_pose, and joint_rotations
        frame_params['joints'] = all_joints[i]  # (127, 3) joints matching vertices coordinate system
        frame_params['full_pose'] = all_full_pose[i]  # (165,) local axis-angle rotations
        frame_params['joint_rotations'] = all_joint_rotations[i]  # (55, 3, 3) rotation matrices
        
        frame_output_path = os.path.join(per_frame_dir, f"{frame_name}_smpl.npz")
        np.savez(frame_output_path, **frame_params)
    
    print(f"Saved per-frame parameters to {per_frame_dir}")
    
    # Print summary of all outputs
    print("\n" + "="*60)
    print("SMPL CONVERSION OUTPUT SUMMARY")
    print("="*60)
    print(f"\nOutput directory: {args.output}")
    print(f"Total frames: {len(frame_names)}")
    
    print("\n--- Main Files ---")
    print(f"  smpl_parameters.npz  : All frames combined")
    print(f"  smpl_vertices.npy    : ({len(frame_names)}, 10475, 3) - vertices")
    print(f"  fitting_errors.npy   : ({len(frame_names)},) - fitting errors")
    if args.save_mesh:
        print(f"  meshes/              : {len(frame_names)} PLY files")
    print(f"  per_frame/           : {len(frame_names)} NPZ files")
    
    print("\n--- Per-frame NPZ Contents ---")
    print("  Field              Shape           Coord System   Description")
    print("  " + "-"*70)
    print("  vertices           (10475, 3)      Pelvis         Mesh vertices (meters)")
    print("  joints             (127, 3)        Pelvis         All joints (55 skeleton + 72 landmarks)")
    print("  joint_rotations    (55, 3, 3)      Local          Rotation matrices (55 skeleton only)")
    print("  full_pose          (165,)          Local          Axis-angle (55×3)")
    print("  global_orient      (3,)            World→Pelvis   Root rotation (axis-angle)")
    print("  body_pose          (63,)           Local          21 body joints (axis-angle)")
    print("  left_hand_pose     (45,)           Local          15 hand joints (axis-angle)")
    print("  right_hand_pose    (45,)           Local          15 hand joints (axis-angle)")
    print("  betas              (10,)           -              Shape parameters")
    print("  expression         (10,)           -              Expression parameters")
    print("  cam_t              (3,)            Camera         Camera translation (meters)")
    print("  transl             (3,)            -              Translation parameter")
    
    print("\n--- Coordinate Systems ---")
    print("  Pelvis: Origin at pelvis center")
    print("  Camera: Origin at camera optical center")
    print("  Local:  Rotations relative to parent joint")
    print("  To camera space: joints_cam = joints + cam_t")
    
    print("\n" + "="*60)
    print("Done!")


if __name__ == "__main__":
    main()

