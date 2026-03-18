#!/usr/bin/env python3
"""
IMU Calibration Script for SMPL Body Model

Computes bone-to-sensor rotation (B_R_S) using multi-frame least squares.
Uses natural motion data - no T-pose required.

Math:
    C_R_S^(i) = C_R_B^(i) @ B_R_S
    => B_R_S = (C_R_B^(i))^T @ C_R_S^(i)
    
    Average in SO(3) using log-space averaging.
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation
import smplx

from utils.imu_id_mapping import SMPLX_JOINT_INDEX_MAP, TAG_TO_JOINT_MAP, JOINT_TO_TAG_MAP


# ==================== Data Classes ====================

@dataclass
class CalibrationSample:
    """Single frame calibration data for one joint."""
    frame_idx: int
    joint_name: str
    C_R_B: np.ndarray  # Bone rotation in camera frame (3x3)
    C_R_S: np.ndarray  # Sensor/Tag rotation in camera frame (3x3)
    B_R_S: np.ndarray  # Computed bone-to-sensor rotation (3x3)


@dataclass
class CalibrationResult:
    """Calibration result for one joint."""
    joint_name: str
    tag_id: int
    B_R_S: np.ndarray  # Average bone-to-sensor rotation (3x3)
    num_samples: int
    std_dev_deg: float  # Standard deviation in degrees
    extra_info: dict = field(default_factory=dict)  # Method-specific info (e.g. RANSAC inliers)


# ==================== SO(3) Operations ====================

def so3_log(R: np.ndarray) -> np.ndarray:
    """Map rotation matrix to so(3) (axis-angle vector)."""
    return Rotation.from_matrix(R).as_rotvec()


def so3_exp(r: np.ndarray) -> np.ndarray:
    """Map so(3) vector to rotation matrix."""
    return Rotation.from_rotvec(r).as_matrix()


def geodesic_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """Compute geodesic distance between two rotations on SO(3)."""
    return np.linalg.norm(so3_log(R1.T @ R2))


def compute_rotation_stats(rotations: List[np.ndarray], mean_R: np.ndarray) -> Tuple[float, float, List[float]]:
    """
    Compute statistics of rotations around a mean.
    
    Returns:
        (std_dev_deg, median_error_deg, list_of_angular_errors_rad)
    """
    angular_errors = [geodesic_distance(mean_R, R) for R in rotations]
    std_dev_deg = np.degrees(np.std(angular_errors))
    median_deg = np.degrees(np.median(angular_errors))
    return std_dev_deg, median_deg, angular_errors


def average_rotations_quaternion(rotations: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """
    Average rotation matrices using quaternion averaging.
    
    This method is more robust than log-space averaging for rotations
    near ±180° where axis-angle representation becomes unstable.
    
    Uses the method of ensuring all quaternions are in the same hemisphere
    before averaging.
    
    Args:
        rotations: List of 3x3 rotation matrices
    
    Returns:
        (average_rotation, std_dev_in_degrees)
    """
    if not rotations:
        return np.eye(3), 0.0
    
    if len(rotations) == 1:
        return rotations[0], 0.0
    
    # Convert to quaternions (xyzw format from scipy)
    quats = np.array([Rotation.from_matrix(R).as_quat() for R in rotations])
    
    # Ensure all quaternions are in the same hemisphere as the first one
    # q and -q represent the same rotation, so we flip signs to make dot products positive
    reference = quats[0]
    for i in range(1, len(quats)):
        if np.dot(quats[i], reference) < 0:
            quats[i] = -quats[i]
    
    # Average quaternions (simple mean, then normalize)
    avg_quat = np.mean(quats, axis=0)
    avg_quat = avg_quat / np.linalg.norm(avg_quat)
    
    # Convert back to rotation matrix
    avg_R = Rotation.from_quat(avg_quat).as_matrix()
    
    # Compute standard deviation (angular distance from mean)
    std_dev_deg, _, _ = compute_rotation_stats(rotations, avg_R)
    
    return avg_R, std_dev_deg


def average_rotations_log_space(rotations: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """
    Average rotation matrices using log-space averaging.
    
    WARNING: This method fails for rotations near ±180°!
    Use average_rotations_quaternion instead.
    
    Args:
        rotations: List of 3x3 rotation matrices
    
    Returns:
        (average_rotation, std_dev_in_degrees)
    """
    # Delegate to quaternion method which is more robust
    return average_rotations_quaternion(rotations)


# ==================== Optimization Methods for SO(3) ====================

def karcher_mean(
    rotations: List[np.ndarray],
    max_iter: int = 100,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, float]:
    """
    Compute the Karcher (Fréchet) mean on SO(3) using iterative refinement.
    
    The Karcher mean minimizes the sum of squared geodesic distances:
        R* = argmin_R Σ_i d(R, R_i)²
    
    where d(R1, R2) = ||log(R1^T @ R2)||_F is the geodesic distance on SO(3).
    
    Algorithm:
        1. Initialize with quaternion average
        2. Iterate: R_new = R_old @ exp(mean(log(R_old^T @ R_i)))
        3. Until convergence
    
    Args:
        rotations: List of 3x3 rotation matrices
        max_iter: Maximum iterations
        tol: Convergence tolerance (radians)
    
    Returns:
        (karcher_mean, std_dev_in_degrees)
    """
    if not rotations:
        return np.eye(3), 0.0
    
    if len(rotations) == 1:
        return rotations[0], 0.0
    
    # Initialize with quaternion average
    mean_R, _ = average_rotations_quaternion(rotations)
    
    for iteration in range(max_iter):
        # Compute mean of tangent vectors at current estimate
        tangent_sum = np.zeros(3)
        for R in rotations:
            tangent_sum += so3_log(mean_R.T @ R)
        tangent_mean = tangent_sum / len(rotations)
        
        # Update: move along the geodesic
        delta = np.linalg.norm(tangent_mean)
        if delta < tol:
            break
        
        mean_R = mean_R @ so3_exp(tangent_mean)
    
    std_dev_deg, _, _ = compute_rotation_stats(rotations, mean_R)
    return mean_R, std_dev_deg


def huber_weight(residual: float, delta: float = 0.1) -> float:
    """Huber weight function for robust estimation."""
    if abs(residual) <= delta:
        return 1.0
    return delta / abs(residual)


def cauchy_weight(residual: float, c: float = 0.1) -> float:
    """Cauchy (Lorentzian) weight function for robust estimation."""
    return 1.0 / (1.0 + (residual / c) ** 2)


def robust_karcher_mean(
    rotations: List[np.ndarray],
    loss_type: str = "huber",
    max_iter: int = 100,
    tol: float = 1e-8,
    c: float = 0.1,  # Robust loss parameter (radians, ~5.7 degrees)
) -> Tuple[np.ndarray, float]:
    """
    Compute robust Karcher mean using M-estimators.
    
    Uses iteratively reweighted least squares (IRLS) with robust loss functions
    to downweight outliers.
    
    Loss functions:
        - "huber": L2 for small errors, L1 for large errors
        - "cauchy": Heavy-tailed Cauchy/Lorentzian loss
        - "l1": Pure L1 (median-like behavior)
    
    Args:
        rotations: List of 3x3 rotation matrices
        loss_type: "huber", "cauchy", or "l1"
        max_iter: Maximum iterations
        tol: Convergence tolerance (radians)
        c: Scale parameter for robust loss (radians)
    
    Returns:
        (robust_mean, std_dev_in_degrees)
    """
    if not rotations:
        return np.eye(3), 0.0
    
    if len(rotations) == 1:
        return rotations[0], 0.0
    
    # Initialize with quaternion average
    mean_R, _ = average_rotations_quaternion(rotations)
    
    for iteration in range(max_iter):
        # Compute weighted mean of tangent vectors
        tangent_sum = np.zeros(3)
        weight_sum = 0.0
        
        for R in rotations:
            tangent = so3_log(mean_R.T @ R)
            residual = np.linalg.norm(tangent)
            
            # Compute weight based on loss type
            if loss_type == "huber":
                w = huber_weight(residual, c)
            elif loss_type == "cauchy":
                w = cauchy_weight(residual, c)
            elif loss_type == "l1":
                w = 1.0 / max(residual, 1e-8)
            else:
                w = 1.0  # L2 (no reweighting)
            
            tangent_sum += w * tangent
            weight_sum += w
        
        tangent_mean = tangent_sum / weight_sum
        
        # Update: move along the geodesic
        delta = np.linalg.norm(tangent_mean)
        if delta < tol:
            break
        
        mean_R = mean_R @ so3_exp(tangent_mean)
    
    std_dev_deg, _, _ = compute_rotation_stats(rotations, mean_R)
    return mean_R, std_dev_deg


def ransac_rotation_mean(
    rotations: List[np.ndarray],
    inlier_threshold_deg: float = 5.0,
    max_iter: int = 100,
    min_inlier_ratio: float = 0.5,
) -> Tuple[np.ndarray, float, int]:
    """
    RANSAC-based rotation estimation.
    
    Finds the rotation that has the most inliers (samples within threshold).
    
    Args:
        rotations: List of 3x3 rotation matrices
        inlier_threshold_deg: Inlier threshold in degrees
        max_iter: Maximum RANSAC iterations
        min_inlier_ratio: Minimum ratio of inliers required
    
    Returns:
        (best_rotation, std_dev_in_degrees, num_inliers)
    """
    if not rotations:
        return np.eye(3), 0.0, 0
    
    if len(rotations) == 1:
        return rotations[0], 0.0, 1
    
    n = len(rotations)
    threshold_rad = np.radians(inlier_threshold_deg)
    
    best_R = None
    best_inliers = []
    
    # For small sample sizes, try all samples as hypotheses
    n_iter = min(max_iter, n)
    
    for i in range(n_iter):
        # Use sample i as hypothesis
        hypothesis = rotations[i % n]
        
        # Count inliers
        inliers = []
        for j, R in enumerate(rotations):
            dist = geodesic_distance(hypothesis, R)
            if dist < threshold_rad:
                inliers.append(j)
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_R = hypothesis
    
    # Refine with inliers only using Karcher mean
    if len(best_inliers) >= max(2, int(n * min_inlier_ratio)):
        inlier_rotations = [rotations[i] for i in best_inliers]
        best_R, std_dev_deg = karcher_mean(inlier_rotations)
    else:
        # Fall back to all samples
        best_R, std_dev_deg = karcher_mean(rotations)
        best_inliers = list(range(n))
    
    return best_R, std_dev_deg, len(best_inliers)


def optimize_rotation_scipy(
    rotations: List[np.ndarray],
    loss_type: str = "soft_l1",
) -> Tuple[np.ndarray, float]:
    """
    Optimize rotation using scipy.optimize.least_squares.
    
    Uses scipy's robust loss functions for optimization on SO(3).
    
    Loss types: 'linear', 'soft_l1', 'huber', 'cauchy', 'arctan'
    
    Args:
        rotations: List of 3x3 rotation matrices
        loss_type: Scipy loss function name
    
    Returns:
        (optimized_rotation, std_dev_in_degrees)
    """
    from scipy.optimize import least_squares
    
    if not rotations:
        return np.eye(3), 0.0
    
    if len(rotations) == 1:
        return rotations[0], 0.0
    
    # Initialize with quaternion average
    init_R, _ = average_rotations_quaternion(rotations)
    init_rotvec = so3_log(init_R)
    
    def residuals(rotvec):
        """Compute residuals (angular distances) for all rotations."""
        R = so3_exp(rotvec)
        return [geodesic_distance(R, Ri) for Ri in rotations]
    
    # Optimize using scipy
    result = least_squares(
        residuals,
        init_rotvec,
        loss=loss_type,
        f_scale=0.1,  # ~5.7 degrees
        method='lm' if loss_type == 'linear' else 'trf',
    )
    
    opt_R = so3_exp(result.x)
    std_dev_deg, _, _ = compute_rotation_stats(rotations, opt_R)
    
    return opt_R, std_dev_deg


# Available optimization methods
OPTIMIZATION_METHODS = {
    "quat_avg": "Simple quaternion averaging (baseline)",
    "karcher": "Karcher/Fréchet mean (geodesic L2)",
    "huber": "Robust Karcher with Huber loss",
    "cauchy": "Robust Karcher with Cauchy loss",
    "l1": "Robust Karcher with L1 loss (median-like)",
    "ransac": "RANSAC with Karcher refinement",
    "scipy_soft_l1": "Scipy optimize with soft L1 loss",
    "scipy_huber": "Scipy optimize with Huber loss",
    "scipy_cauchy": "Scipy optimize with Cauchy loss",
}


def optimize_rotation(
    rotations: List[np.ndarray],
    method: str = "ransac",
    **kwargs,
) -> Tuple[np.ndarray, float, dict]:
    """
    Unified interface for rotation optimization methods.
    
    Args:
        rotations: List of 3x3 rotation matrices
        method: Optimization method (see OPTIMIZATION_METHODS)
        **kwargs: Method-specific parameters
    
    Returns:
        (optimized_rotation, std_dev_in_degrees, extra_info)
    """
    extra_info = {"method": method}
    
    if method == "quat_avg":
        R, std = average_rotations_quaternion(rotations)
    
    elif method == "karcher":
        R, std = karcher_mean(
            rotations,
            max_iter=kwargs.get("max_iter", 100),
            tol=kwargs.get("tol", 1e-8),
        )
    
    elif method in ["huber", "cauchy", "l1"]:
        R, std = robust_karcher_mean(
            rotations,
            loss_type=method,
            max_iter=kwargs.get("max_iter", 100),
            c=kwargs.get("c", 0.1),  # ~5.7 degrees
        )
    
    elif method == "ransac":
        R, std, n_inliers = ransac_rotation_mean(
            rotations,
            inlier_threshold_deg=kwargs.get("inlier_threshold_deg", 5.0),
            max_iter=kwargs.get("max_iter", 100),
        )
        extra_info["num_inliers"] = n_inliers
        extra_info["inlier_ratio"] = n_inliers / len(rotations) if rotations else 0
    
    elif method.startswith("scipy_"):
        loss = method.replace("scipy_", "")
        R, std = optimize_rotation_scipy(rotations, loss_type=loss)
    
    else:
        print(f"Warning: Unknown method '{method}', falling back to karcher")
        R, std = karcher_mean(rotations)
    
    return R, std, extra_info


# ==================== Data Loading ====================

def load_smpl_model(smpl_model_path: str, model_type: str = "smplx"):
    """Load SMPL/SMPLX model."""
    if smplx is None:
        raise ImportError("smplx package required. Install with: pip install smplx")
    if model_type == "smplx":
        return smplx.SMPLX(smpl_model_path, gender="neutral", use_pca=False, flat_hand_mean=True)
    return smplx.SMPL(model_path=smpl_model_path)


def local_to_global_rotations(local_rots: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """
    Convert local rotation matrices to global rotations via forward kinematics.
    
    Args:
        local_rots: (N, 3, 3) local rotation matrices
        parents: (N,) parent joint indices (-1 for root)
    
    Returns:
        (N, 3, 3) global rotation matrices in camera frame
    """
    num_joints = local_rots.shape[0]
    global_rots = np.zeros_like(local_rots)
    global_rots[0] = local_rots[0]  # Root is already global
    
    for i in range(1, num_joints):
        parent = parents[i]
        if parent >= 0:
            global_rots[i] = global_rots[parent] @ local_rots[i]
        else:
            global_rots[i] = local_rots[i]
    
    return global_rots


def load_smpl_joint_rotations(
    npz_path: str,
    smpl_model,
) -> Optional[np.ndarray]:
    """
    Load SMPL joint rotations from npz file.
    Returns global rotations in camera frame (N, 3, 3).
    """
    data = np.load(npz_path, allow_pickle=True)
    
    # Load local joint rotations
    local_rots = None
    for key in ["joint_rotations", "joint_rots"]:
        if key in data and data[key].shape[-2:] == (3, 3):
            local_rots = data[key].astype(np.float32)
            break
    
    if local_rots is None:
        return None
    
    # Convert to global using forward kinematics
    parents = smpl_model.parents.cpu().numpy()
    num_skel_joints = min(local_rots.shape[0], len(parents))
    global_rots = local_to_global_rotations(local_rots[:num_skel_joints], parents[:num_skel_joints])
    
    return global_rots


def load_apriltag_data(summary_path: str) -> Dict[int, List[dict]]:
    """Load AprilTag summary and return frame_number -> detections mapping."""
    with open(summary_path) as f:
        data = json.load(f)
    
    frame_map: Dict[int, List[dict]] = {}
    for entry in data.get("images", []):
        match = re.search(r"frame_(\d+)", entry.get("filename", ""))
        if match:
            frame_map[int(match.group(1))] = entry.get("detections", [])
    return frame_map


def extract_frame_number(path: str) -> Optional[int]:
    """Extract frame number from filename like frame_000107_0.npz."""
    match = re.search(r"frame_(\d+)", Path(path).stem)
    return int(match.group(1)) if match else None


def load_frame_timestamps(frames_csv: Path) -> Dict[int, int]:
    """
    Load frames.csv into a mapping: frame_id -> utc_timestamp_ns.
    Expected CSV columns: frame_id, utc_timestamp_ns, color_path
    """
    mapping: Dict[int, int] = {}
    with open(frames_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_id = int(row["frame_id"])
                ts = int(row["utc_timestamp_ns"])
            except Exception:
                continue
            mapping[frame_id] = ts
    return mapping


def filter_npz_files(
    npz_files: List[Path],
    *,
    utc_end_ns: Optional[int] = None,
    frame_timestamps: Optional[Dict[int, int]] = None,
) -> Tuple[List[Path], int]:
    """
    Filter SMPL per-frame .npz files by timestamp.

    Returns:
        (filtered_files, num_skipped_missing_timestamp)
    """
    if utc_end_ns is not None and frame_timestamps is None:
        raise ValueError("utc_end_ns specified but no frames.csv timestamp map provided")

    kept: List[Path] = []
    skipped_missing_ts = 0

    for p in npz_files:
        frame_num = extract_frame_number(str(p))
        if frame_num is None:
            continue

        if utc_end_ns is not None:
            ts = frame_timestamps.get(frame_num) if frame_timestamps is not None else None
            if ts is None:
                skipped_missing_ts += 1
                continue
            if ts > utc_end_ns:
                continue

        kept.append(p)

    return kept, skipped_missing_ts


# ==================== Calibration Logic ====================

def compute_bone_to_sensor_rotation(
    C_R_B: np.ndarray,
    C_R_S: np.ndarray,
) -> np.ndarray:
    """
    Compute bone-to-sensor rotation.
    
    B_R_S = (C_R_B)^T @ C_R_S
    
    This represents the fixed rotation from sensor frame to bone frame.
    """
    return C_R_B.T @ C_R_S


def collect_calibration_samples(
    npz_files: List[Path],
    apriltag_data: Dict[int, List[dict]],
    smpl_model,
) -> Dict[str, List[CalibrationSample]]:
    """
    Collect calibration samples from all frames.
    
    Returns: Dict mapping joint_name -> list of CalibrationSample
    """
    samples: Dict[str, List[CalibrationSample]] = {name: [] for name in TAG_TO_JOINT_MAP.values()}
    
    for frame_idx, npz_path in enumerate(npz_files):
        # Load SMPL joint rotations
        joint_rots = load_smpl_joint_rotations(str(npz_path), smpl_model)
        if joint_rots is None:
            continue
        
        # Get AprilTag detections for this frame
        frame_num = extract_frame_number(str(npz_path))
        if frame_num is None:
            continue
        
        tags = apriltag_data.get(frame_num, [])
        
        # Process each detected tag
        for tag in tags:
            if "rotation_matrix" not in tag:
                continue
            
            tag_id = tag["tag_id"]
            joint_name = TAG_TO_JOINT_MAP.get(tag_id)
            if joint_name is None:
                continue
            
            # Get joint index for this joint
            joint_idx = SMPLX_JOINT_INDEX_MAP.get(joint_name)
            if joint_idx is None or joint_idx >= joint_rots.shape[0]:
                continue
            
            # C_R_B: Bone rotation in camera frame
            C_R_B = joint_rots[joint_idx]
            
            # C_R_S: Sensor/Tag rotation in camera frame
            C_R_S = np.array(tag["rotation_matrix"])
            
            # Validate rotation matrices
            if C_R_S.shape != (3, 3) or np.any(~np.isfinite(C_R_S)):
                continue
            
            # Skip mirrored detections (ambiguous)
            if tag.get("is_mirrored", False):
                continue
            
            # Compute B_R_S for this frame
            B_R_S = compute_bone_to_sensor_rotation(C_R_B, C_R_S)
            
            sample = CalibrationSample(
                frame_idx=frame_idx,
                joint_name=joint_name,
                C_R_B=C_R_B,
                C_R_S=C_R_S,
                B_R_S=B_R_S,
            )
            samples[joint_name].append(sample)
    
    return samples


def calibrate_all_joints(
    samples: Dict[str, List[CalibrationSample]],
    min_samples: int = 10,
    method: str = "ransac",
    **method_kwargs,
) -> Dict[str, CalibrationResult]:
    """
    Compute optimized bone-to-sensor rotation for each joint.
    
    Args:
        samples: Dict mapping joint_name -> list of CalibrationSample
        min_samples: Minimum number of samples required for calibration
        method: Optimization method (see OPTIMIZATION_METHODS)
        **method_kwargs: Additional parameters for the optimization method
    
    Returns:
        Dict mapping joint_name -> CalibrationResult
    """
    results: Dict[str, CalibrationResult] = {}
    
    print(f"  Using optimization method: {method}")
    if method in OPTIMIZATION_METHODS:
        print(f"    ({OPTIMIZATION_METHODS[method]})")
    
    for joint_name, joint_samples in samples.items():
        if len(joint_samples) < min_samples:
            print(f"  {joint_name}: Skipped (only {len(joint_samples)} samples, need {min_samples})")
            continue
        
        # Extract all B_R_S matrices
        B_R_S_list = [s.B_R_S for s in joint_samples]
        
        # Optimize rotation using selected method
        opt_B_R_S, std_dev_deg, extra_info = optimize_rotation(
            B_R_S_list,
            method=method,
            **method_kwargs,
        )
        
        tag_id = JOINT_TO_TAG_MAP[joint_name]
        
        results[joint_name] = CalibrationResult(
            joint_name=joint_name,
            tag_id=tag_id,
            B_R_S=opt_B_R_S,
            num_samples=len(joint_samples),
            std_dev_deg=std_dev_deg,
            extra_info=extra_info,
        )
        
        # Format extra info for display
        extra_str = ""
        if "num_inliers" in extra_info:
            inlier_pct = 100 * extra_info['num_inliers'] / len(joint_samples)
            extra_str = f", inliers={extra_info['num_inliers']}/{len(joint_samples)} ({inlier_pct:.0f}%)"
        
        print(f"  {joint_name}: {len(joint_samples)} samples, std_dev={std_dev_deg:.2f}°{extra_str}")
    
    return results


def save_calibration(
    results: Dict[str, CalibrationResult],
    output_path: str,
    method: str = "ransac",
    calib_duration_sec: Optional[float] = None,
):
    """Save calibration results to JSON file."""
    output = {
        "description": "IMU bone-to-sensor calibration (B_R_S)",
        "coordinate_frame": "camera_frame",
        "optimization_method": method,
        "joints": {}
    }
    if calib_duration_sec is not None:
        output["calib_duration_sec"] = calib_duration_sec
    
    for joint_name, result in results.items():
        # Convert rotation to axis-angle for readability
        rotvec = so3_log(result.B_R_S)
        angle_deg = np.degrees(np.linalg.norm(rotvec))
        axis = rotvec / (np.linalg.norm(rotvec) + 1e-8)
        
        output["joints"][joint_name] = {
            "tag_id": result.tag_id,
            "B_R_S": result.B_R_S.tolist(),
            "axis_angle": {
                "axis": axis.tolist(),
                "angle_deg": float(angle_deg),
            },
            "num_samples": result.num_samples,
            "std_dev_deg": float(result.std_dev_deg),
        }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nCalibration saved to: {output_path}")


# ==================== Main ====================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="IMU Calibration using SMPL body model and AprilTags"
    )
    parser.add_argument(
        "session_dir",
        nargs="?",
        default="fusion_recordings/fusion_session_20251124_231205",
        help="Session directory containing smpl_output/per_frame and color_apriltag"
    )
    parser.add_argument(
        "--smpl-model-path",
        default=str(Path(__file__).resolve().parent / "MHR" / "model" / "smplx" / "SMPLX_NEUTRAL.npz"),
        help="Path to SMPLX model file"
    )
    parser.add_argument(
        "--apriltag-summary",
        help="Path to AprilTag detection_summary.json (auto-detected if not specified)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output calibration JSON file (default: session_dir/imu_calibration.json)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum samples required per joint (default: 10)"
    )
    parser.add_argument(
        "--calib-duration-sec",
        type=float,
        default=None,
        help=(
            "Use first N seconds from start of recording for calibration. "
            "If not provided, uses metadata suggestedCalibDurationSec when all tags reached the app target; "
            "otherwise uses all frames. E.g., --calib-duration-sec 46"
        ),
    )
    parser.add_argument(
        "--frames-csv",
        type=str,
        default=None,
        help="Optional path to frames.csv for utc filtering (default: session_dir/frames.csv).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="karcher",
        choices=list(OPTIMIZATION_METHODS.keys()),
        help=f"Optimization method for rotation estimation. Choices: {list(OPTIMIZATION_METHODS.keys())}. Default: karcher",
    )
    parser.add_argument(
        "--robust-scale",
        type=float,
        default=0.1,
        help="Scale parameter (radians) for robust loss functions (huber/cauchy). ~0.1 = 5.7°. Default: 0.1",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=5.0,
        help="Inlier threshold (degrees) for RANSAC method. Default: 5.0",
    )
    parser.add_argument(
        "--compare-methods",
        action="store_true",
        help="Run all optimization methods and compare results.",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    session_path = Path(args.session_dir)
    body_path = session_path / "smpl_output" / "per_frame"
    calib_duration_from_metadata = False

    # Auto-default calib window from iOS metadata.json when app reports all tags reached target.
    if args.calib_duration_sec is None:
        meta_path = session_path / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                calib_seg = meta.get("calibrationSegment") if isinstance(meta, dict) else None
                if isinstance(calib_seg, dict) and calib_seg.get("method") == "per_tag_min_detections":
                    suggested = calib_seg.get("suggestedCalibDurationSec")
                    if suggested is not None:
                        suggested_f = float(suggested)
                        if suggested_f > 0:
                            args.calib_duration_sec = suggested_f
                            calib_duration_from_metadata = True
                            min_det = calib_seg.get("minDetectionsPerTag")
                            msg = f"Auto calib window: using first {suggested_f:.3f}s (all tags reached target"
                            if min_det is not None:
                                msg += f", min {min_det} detections per tag"
                            msg += ")"
                            print(msg)
            except Exception as exc:
                print(f"Warning: failed to read metadata.json calibrationSegment: {exc}")
    
    if not body_path.exists():
        print(f"Error: SMPL data directory not found: {body_path}")
        return 1
    
    # Find npz files (exclude macOS AppleDouble/resource-fork artifacts ._*)
    npz_files = sorted(p for p in body_path.glob("*.npz") if not p.name.startswith("._"))
    if not npz_files:
        print(f"No .npz files found in {body_path}")
        return 1
    npz_files_all = list(npz_files)

    # Optional filtering by calibration duration
    if args.calib_duration_sec is not None:
        frames_csv_path = Path(args.frames_csv) if args.frames_csv else (session_path / "frames.csv")
        if not frames_csv_path.exists():
            # Keep backward compatibility: if the value came from metadata.json, fall back to all frames.
            print(f"Warning: --calib-duration-sec requires frames.csv, not found: {frames_csv_path}")
            print("  Falling back to using all frames (no calibration window filtering).")
            args.calib_duration_sec = None
        else:
            frame_ts_map = load_frame_timestamps(frames_csv_path)
            if not frame_ts_map:
                print("Error: frames.csv is empty or has no valid timestamps")
                return 1
            min_ts = min(frame_ts_map.values())
            utc_end_ns = int(min_ts + args.calib_duration_sec * 1e9)
            print(f"Using first {args.calib_duration_sec} seconds for calibration")
            print(f"  Start timestamp: {min_ts} ns")
            print(f"  End timestamp:   {utc_end_ns} ns")

            filtered_npz, skipped_missing_ts = filter_npz_files(
                npz_files,
                utc_end_ns=utc_end_ns,
                frame_timestamps=frame_ts_map,
            )
            print(f"Found {len(npz_files)} SMPL frames (before filtering)")
            print(f"Using  {len(filtered_npz)} SMPL frames (after filtering)")
            if skipped_missing_ts:
                print(f"  Skipped {skipped_missing_ts} frame(s) missing timestamp in frames.csv")
            npz_files = filtered_npz
    else:
        print(f"Found {len(npz_files)} SMPL frames")
    if not npz_files:
        print("Error: No SMPL frames remain after filtering")
        return 1
    
    # Load SMPL model
    print("Loading SMPLX model...")
    smpl_model = load_smpl_model(args.smpl_model_path, "smplx")
    
    # Auto-detect apriltag summary
    apriltag_summary = args.apriltag_summary
    if apriltag_summary is None:
        possible_paths = [
            session_path / "color_apriltag" / "detection_summary.json",
            session_path / "apriltag_detections" / "detection_summary.json",
        ]
        for p in possible_paths:
            if p.exists():
                apriltag_summary = str(p)
                break
    
    if apriltag_summary is None:
        print("Error: AprilTag summary not found")
        return 1
    
    print(f"Loading AprilTag data from {apriltag_summary}")
    apriltag_data = load_apriltag_data(apriltag_summary)
    print(f"  Frames with detections: {len(apriltag_data)}")
    
    # Collect calibration samples
    print("\nCollecting calibration samples...")
    samples = collect_calibration_samples(npz_files, apriltag_data, smpl_model)

    # If the calibration window came from metadata and it doesn't provide enough samples for some joints,
    # fall back to using all frames. This avoids silently skipping joints (e.g., tags first appear at the
    # tail end of the suggested window).
    if calib_duration_from_metadata and npz_files is not npz_files_all:
        missing_joints = sorted([jn for jn, ss in samples.items() if len(ss) < args.min_samples])
        if missing_joints:
            print("\n⚠️  Metadata-suggested calibration window yielded too few samples for:")
            print("   " + ", ".join(missing_joints))
            print("   Falling back to using ALL frames for calibration.")
            npz_files = npz_files_all
            print("\nRe-collecting calibration samples on all frames...")
            samples = collect_calibration_samples(npz_files, apriltag_data, smpl_model)
    
    total_samples = sum(len(s) for s in samples.values())
    print(f"  Total samples: {total_samples}")
    
    # Prepare optimization kwargs
    method_kwargs = {
        "c": args.robust_scale,
        "inlier_threshold_deg": args.ransac_threshold,
    }
    
    # Comparison mode: run all methods
    if args.compare_methods:
        print("\n" + "=" * 70)
        print("Comparing all optimization methods")
        print("=" * 70)
        
        comparison_results = {}
        for method in OPTIMIZATION_METHODS.keys():
            print(f"\n--- Method: {method} ({OPTIMIZATION_METHODS[method]}) ---")
            results = calibrate_all_joints(
                samples,
                min_samples=args.min_samples,
                method=method,
                **method_kwargs,
            )
            comparison_results[method] = results
        
        # Print comparison table
        print("\n" + "=" * 70)
        print("Comparison Summary (std_dev in degrees)")
        print("=" * 70)
        
        # Get all joints that were calibrated
        all_joints = set()
        for results in comparison_results.values():
            all_joints.update(results.keys())
        all_joints = sorted(all_joints)
        
        # Header
        methods_short = list(OPTIMIZATION_METHODS.keys())
        header = f"{'Joint':<18} | " + " | ".join(f"{m:>10}" for m in methods_short)
        print(header)
        print("-" * len(header))
        
        # Data rows
        for joint in all_joints:
            row = f"{joint:<18} | "
            values = []
            for method in methods_short:
                if joint in comparison_results[method]:
                    std = comparison_results[method][joint].std_dev_deg
                    values.append(f"{std:>10.2f}")
                else:
                    values.append(f"{'N/A':>10}")
            row += " | ".join(values)
            print(row)
        
        # Print RANSAC inlier ratios separately
        if "ransac" in comparison_results:
            print("\n" + "-" * 70)
            print("RANSAC Inlier Analysis (⚠️ low inlier % = aggressive outlier rejection)")
            print("-" * 70)
            for joint in all_joints:
                if joint in comparison_results["ransac"]:
                    result = comparison_results["ransac"][joint]
                    extra = result.extra_info
                    if "num_inliers" in extra:
                        n_total = result.num_samples
                        n_inliers = extra["num_inliers"]
                        pct = 100 * n_inliers / n_total if n_total > 0 else 0
                        warning = " ⚠️" if pct < 50 else ""
                        print(f"  {joint:<18}: {n_inliers:>4}/{n_total:<4} inliers ({pct:>5.1f}%){warning}")
        
        # Use karcher results as default for saving
        results = comparison_results.get("karcher", {})
        if not results:
            results = next(iter(comparison_results.values()), {})
    else:
        # Single method calibration
        print("\nComputing bone-to-sensor rotations (B_R_S)...")
        results = calibrate_all_joints(
            samples,
            min_samples=args.min_samples,
            method=args.method,
            **method_kwargs,
        )
    
    if not results:
        print("Error: No joints calibrated (not enough samples)")
        return 1
    
    # Save results
    output_path = args.output or str(session_path / "imu_calibration.json")
    save_calibration(results, output_path, method=args.method,
                     calib_duration_sec=args.calib_duration_sec)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Calibration Summary (method: {args.method})")
    print("=" * 60)
    for joint_name, result in results.items():
        rotvec = so3_log(result.B_R_S)
        angle_deg = np.degrees(np.linalg.norm(rotvec))
        print(f"  {joint_name:20s}: angle={angle_deg:6.2f}°, std={result.std_dev_deg:5.2f}°, n={result.num_samples}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
