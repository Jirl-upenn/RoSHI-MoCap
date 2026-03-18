"""
Data structure definition that we use for IMU detections.

We'll read IMU data from pickle files, produce the dictionary defined by `SavedImuOutputs`, then
process this dictionary to match target timestamps.
"""

from __future__ import annotations

import pickle
import pandas as pd
from pathlib import Path
from typing import Protocol, TypedDict, cast

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor

from .tensor_dataclass import TensorDataclass
from .imu_utils import IMU_CONFIG, CLOSEST_TIMESTAMP_DIFF_IN_NS


class SingleImuOutputWrtCanonical(TypedDict):
    """IMU outputs with respect to the canonical frame. For use in pickle files."""

    rotation_matrix: np.ndarray  # 3x3 rotation matrix


class SavedImuOutputs(TypedDict):
    """Outputs from IMU readings. This is the structure to pickle.

    `imu_readings` uses utc_ns timestamps as keys, and each timestamp maps to
    a dictionary with IMU IDs (1-9) as keys, containing 3x3 rotation matrices.
    """

    imu_readings: dict[int, dict[int, SingleImuOutputWrtCanonical | None]]


class IMUPoseWrtCanonical(TensorDataclass):
    """IMU pose data in canonical coordinates."""
    
    confidence: Float[Tensor, "n_detections"]
    indices: Int[Tensor, "n_detections"]
    imu_readings: Float[Tensor, "n_detections 9 3 3"]  # 9 IMUs, 3x3 rotation matrices


class CorrespondedImuReadings(TensorDataclass):
    """Corresponded IMU readings that match target timestamps."""

    readings: IMUPoseWrtCanonical | None
    

    @staticmethod
    def load(
        imu_readings_pkl_path: Path,
        target_timestamps_sec: tuple[float, ...],
        closed_loop_trajectory_csv_path: Path,
    ) -> CorrespondedImuReadings:
        """Helper which takes as input:

        (1) A path to a pickle file containing IMU readings through time.
            Expected format: dict[utc_timestamp_ns][imu_id] = 3x3 rotation matrix

        (2) A set of target timestamps, sorted, in seconds (tracking_timestamp_us / 1e6).

        (3) Path to closed_loop_trajectory.csv for timestamp mapping.

        We then output a data structure that has IMU readings for each target timestamp.
        """

        # Load the closed loop trajectory CSV to get timestamp mapping
        # Handle case where path might be a tuple (from glob pattern)
        if isinstance(closed_loop_trajectory_csv_path, tuple):
            csv_path = closed_loop_trajectory_csv_path[0]
        else:
            csv_path = closed_loop_trajectory_csv_path
            
        trajectory_df = pd.read_csv(csv_path)
        
        # Create mapping from tracking_timestamp_us to utc_timestamp_ns
        # tracking_timestamp_us is in microseconds, utc_timestamp_ns is in nanoseconds
        tracking_to_utc_mapping = {}
        
        for _, row in trajectory_df.iterrows():
            utc_ns = row['utc_timestamp_ns']
            tracking_us = row['tracking_timestamp_us']
            tracking_to_utc_mapping[tracking_us] = utc_ns

        with open(imu_readings_pkl_path, "rb") as f:
            imu_out = cast(SavedImuOutputs, pickle.load(f))
        
        est_fps = len(imu_out) / (
            (max(imu_out.keys()) - min(imu_out.keys())) / 1e9
        )
        assert 10 < est_fps < 100, f"Estimated FPS is {est_fps}, which is not between 10 and 100"

        def match_imu_to_targets(
            imu_readings: dict[int, dict[int, SingleImuOutputWrtCanonical | None]],
            target_timestamps_sec: tuple[float, ...],
            tracking_to_utc_mapping: dict[int, int],
        ) -> tuple[list[dict[int, SingleImuOutputWrtCanonical | None] | None], list[int]]:
            """Match IMU readings to target timestamps using the CSV mapping."""
            
            if not imu_readings:
                return [None] * len(target_timestamps_sec), []

            # Get all available UTC timestamps from IMU readings
            available_utc_timestamps = sorted(imu_readings.keys())
            
            # For each target tracking timestamp, find the corresponding UTC timestamp
            out: list[dict[int, SingleImuOutputWrtCanonical | None] | None] = []
            indices: list[int] = []
            
            for i, target_tracking_sec in enumerate(target_timestamps_sec):
                # Convert target_tracking_sec to microseconds to match CSV format
                target_tracking_us = int(target_tracking_sec * 1e6)
                
                # Find which UTC timestamp maps to this tracking timestamp
                if target_tracking_us in tracking_to_utc_mapping:
                    matching_utc_ns = tracking_to_utc_mapping[target_tracking_us]
                else:
                    continue
                
                # Find the closest UTC timestamp in IMU readings
                if matching_utc_ns in imu_readings:
                    # Exact match found
                    out.append(imu_readings[matching_utc_ns])
                    indices.append(i)
                else:
                    # Find closest available UTC timestamp
                    closest_utc = min(available_utc_timestamps, 
                                    key=lambda x: abs(x - matching_utc_ns))
                    
                    # Check if the closest timestamp is within reasonable range
                    time_diff_ns = abs(closest_utc - matching_utc_ns)
                    if time_diff_ns <= CLOSEST_TIMESTAMP_DIFF_IN_NS:  # 10ms in nanoseconds
                        out.append(imu_readings[closest_utc])
                        indices.append(i)
            
            return out, indices

        # Match IMU readings to target timestamps using the CSV mapping
        imu_readings_matched, valid_indices = match_imu_to_targets(
            imu_out,
            target_timestamps_sec,
            tracking_to_utc_mapping
        )

        def form_imu_concat(
            imu_readings_matched: list[dict[int, SingleImuOutputWrtCanonical | None] | None],
            valid_indices: list[int],
        ) -> IMUPoseWrtCanonical | None:
            """TODO: 
            - 1. condifence selection
            - 2. frame transformation
            - 3. np -> tensor
            """

            assert imu_readings_matched, f"Missing IMU readings"
            assert len(imu_readings_matched) == len(valid_indices), f"IMU readings length: {len(imu_readings_matched)}, valid indices length: {len(valid_indices)}"
            assert None not in imu_readings_matched, f"There are None in the IMU readings"

            # Process IMU readings for each valid timestamp
            imu_rotations = []
            for reading in imu_readings_matched:
                # Create 9x3x3 tensor for all IMUs at this timestamp
                timestamp_rotations = torch.zeros(9, 3, 3, dtype=torch.float32)
                for imu_id in range(1, 10):  # IMU IDs 1-9
                    if imu_id in reading and reading[imu_id] is not None:
                        # Convert list to numpy array first, then to torch tensor
                        rot_matrix = torch.tensor(
                            reading[imu_id], dtype=torch.float32
                        )
                        assert IMU_CONFIG[imu_id][2] == imu_id - 1, f"IMU ID {imu_id} has tensor index {IMU_CONFIG[imu_id][2]} which is not equal to {imu_id - 1}"
                        timestamp_rotations[IMU_CONFIG[imu_id][2]] = rot_matrix  # Convert to 0-indexed
                
                imu_rotations.append(timestamp_rotations)

            # Stack all IMU readings into tensor format
            imu_readings_tensor = torch.stack(imu_rotations)  # [n_detections, 9, 3, 3]

            assert imu_readings_tensor.shape == (len(valid_indices), 9, 3, 3), f"IMU readings tensor shape: {imu_readings_tensor.shape}, expected shape: {(len(valid_indices), 9, 3, 3)}"

            return IMUPoseWrtCanonical(
                confidence=torch.ones(len(valid_indices), dtype=torch.float32),  # Default confidence of 1.0
                indices=torch.from_numpy(np.array(valid_indices, dtype=np.int64)),
                imu_readings=imu_readings_tensor,
            )

        return CorrespondedImuReadings(
            readings=form_imu_concat(imu_readings_matched, valid_indices)
        )