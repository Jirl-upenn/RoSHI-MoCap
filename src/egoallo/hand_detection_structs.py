"""Data structures for Aria hand tracking detections."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from jaxtyping import Float, Int
from projectaria_tools.core import mps
from projectaria_tools.core.mps.utils import get_nearest_wrist_and_palm_pose, get_nearest_hand_tracking_result
from torch import Tensor

from .tensor_dataclass import TensorDataclass
from .transforms import SE3, SO3


class AriaHandWristPoseWrtWorld(TensorDataclass):
    confidence: Float[Tensor, "n_detections"]
    wrist_position: Float[Tensor, "n_detections 3"]
    wrist_normal: Float[Tensor, "n_detections 3"]

    palm_position: Float[Tensor, "n_detections 3"]
    palm_normal: Float[Tensor, "n_detections 3"]

    indices: Int[Tensor, "n_detections"]


class CorrespondedAriaHandWristPoseDetections(TensorDataclass):
    detections_left_concat: AriaHandWristPoseWrtWorld | None
    detections_right_concat: AriaHandWristPoseWrtWorld | None

    @staticmethod
    def load(
        wrist_and_palm_poses_csv_path: Path,
        target_timestamps_sec: tuple[float, ...],
        Ts_world_device: Float[np.ndarray, "timesteps 7"],
    ) -> CorrespondedAriaHandWristPoseDetections:
        # API from runtime inspection of `projectaria_tools` outputs.
        class WristAndPalmNormals(Protocol):
            wrist_normal_device: np.ndarray
            palm_normal_device: np.ndarray

        class OneSide(Protocol):
            confidence: float
            wrist_position_device: np.ndarray
            palm_position_device: np.ndarray
            wrist_and_palm_normal_device: WristAndPalmNormals

        wp_poses = mps.hand_tracking.read_wrist_and_palm_poses(
            str(wrist_and_palm_poses_csv_path)
        )
        detections_left = list[OneSide]()
        detections_right = list[OneSide]()
        indices_left = list[int]()
        indices_right = list[int]()
        for i, time_sec in enumerate(target_timestamps_sec):
            wp_pose = get_nearest_wrist_and_palm_pose(wp_poses, int(time_sec * 1e9))
            if (
                wp_pose is None
                or abs(wp_pose.tracking_timestamp.total_seconds() - time_sec)
                >= 1.0 / 30.0
            ):
                continue

            if wp_pose.left_hand is not None and wp_pose.left_hand.confidence > 0.7:
                indices_left.append(i)
                detections_left.append(wp_pose.left_hand)

            if wp_pose.right_hand is not None and wp_pose.right_hand.confidence > 0.7:
                indices_right.append(i)
                detections_right.append(wp_pose.right_hand)

        def form_detections_concat(
            detections: list[OneSide], indices: list[int]
        ) -> AriaHandWristPoseWrtWorld | None:
            assert len(detections) == len(indices)
            if len(indices) == 0:
                return None

            Tslice_world_device = SE3(
                torch.from_numpy(Ts_world_device[np.array(indices), :]).to(
                    dtype=torch.float32
                )
            )
            Rslice_world_device = SO3(
                torch.from_numpy(Ts_world_device[np.array(indices), :4]).to(
                    dtype=torch.float32
                )
            )

            return AriaHandWristPoseWrtWorld(
                confidence=torch.from_numpy(
                    np.array([d.confidence for d in detections])
                ),
                wrist_position=Tslice_world_device
                @ torch.from_numpy(
                    np.array(
                        [d.wrist_position_device for d in detections], dtype=np.float32
                    )
                ),
                wrist_normal=Rslice_world_device
                @ torch.from_numpy(
                    np.array(
                        [
                            d.wrist_and_palm_normal_device.wrist_normal_device
                            for d in detections
                        ],
                        dtype=np.float32,
                    )
                ),
                palm_position=Tslice_world_device
                @ torch.from_numpy(
                    np.array(
                        [d.palm_position_device for d in detections], dtype=np.float32
                    )
                ),
                palm_normal=Rslice_world_device
                @ torch.from_numpy(
                    np.array(
                        [
                            d.wrist_and_palm_normal_device.palm_normal_device
                            for d in detections
                        ],
                        dtype=np.float32,
                    )
                ),
                indices=torch.from_numpy(np.array(indices, dtype=np.int64)),
            )

        return CorrespondedAriaHandWristPoseDetections(
            detections_left_concat=form_detections_concat(
                detections_left, indices_left
            ),
            detections_right_concat=form_detections_concat(
                detections_right, indices_right
            ),
        )


class AriaHandAllPoseWrtWorld(TensorDataclass):
    confidence: Float[Tensor, "n_detections"]

    landmarks_3d: Float[Tensor, "n_detections 21 3"]  # 21 landmarks for each hand

    wrist_position: Float[Tensor, "n_detections 3"]
    wrist_normal: Float[Tensor, "n_detections 3"]

    palm_position: Float[Tensor, "n_detections 3"]
    palm_normal: Float[Tensor, "n_detections 3"]

    indices: Int[Tensor, "n_detections"]


class CorrespondedAriaHandAllPoseWrtWorld(TensorDataclass):
    detections_left_concat: AriaHandAllPoseWrtWorld | None
    detections_right_concat: AriaHandAllPoseWrtWorld | None

    @staticmethod
    def load(
        hand_tracking_results_path: Path,
        target_timestamps_sec: tuple[float, ...],
        Ts_world_device: Float[np.ndarray, "timesteps 7"],
    ) -> CorrespondedAriaHandAllPoseWrtWorld:
        # API from runtime inspection of 'projectaria_tools' outputs.
        class WristAndPalmNormals(Protocol):
            wrist_normal_device: np.ndarray
            palm_normal_device: np.ndarray

        class OneSide(Protocol):
            confidence: float
            landmark_positions_device: np.ndarray  # This should contain 21 landmarks, x,y,z 
            wrist_position_device: np.ndarray # the wrist position x,y,z 
            wrist_and_palm_normal_device: WristAndPalmNormals
        
        hand_tracking_results = mps.hand_tracking.read_hand_tracking_results(
            str(hand_tracking_results_path)
        )
        detections_left = list[OneSide]()
        detections_right = list[OneSide]()
        indices_left = list[int]()
        indices_right = list[int]()
        for i, time_sec in enumerate(target_timestamps_sec):
            hand_pose = get_nearest_hand_tracking_result(hand_tracking_results, int(time_sec * 1e9))
            # time checking
            if (
                hand_pose is None
                or abs(hand_pose.tracking_timestamp.total_seconds() - time_sec)
                >= 1.0 / 30.0
            ):
                continue

            # confidence checking
            if hand_pose.left_hand is not None and hand_pose.left_hand.confidence > 0.7:
                indices_left.append(i)
                detections_left.append(hand_pose.left_hand)
            
            if hand_pose.right_hand is not None and hand_pose.right_hand.confidence > 0.7:
                indices_right.append(i)
                detections_right.append(hand_pose.right_hand)
        
        def form_detections_concat(
            detections: list[OneSide], indices: list[int]
        ) -> AriaHandAllPoseWrtWorld | None:
            assert len(detections) == len(indices)
            if len(indices) == 0:
                return None
            
            Tslice_world_device = SE3(
                torch.from_numpy(Ts_world_device[np.array(indices), :]).to(
                    dtype=torch.float32
                )
            ) # Tslice_world_device shape: [n_detections, 7] (for SE3)
            Rslice_world_device = SO3(
                torch.from_numpy(Ts_world_device[np.array(indices), :4]).to(
                    dtype=torch.float32
                )
            )

            # Extract all 21 landmarks and convert from list to numpy array
            landmarks_device = torch.from_numpy(
                np.array(
                    [d.landmark_positions_device for d in detections], dtype=np.float32
                )
            ) # landmarks_device shape: [n_detections, 21, 3]
            # Convert to torch tensor first
            landmarks_world_transformed = torch.zeros_like(landmarks_device)
            for j in range(landmarks_device.shape[1]):
                landmark_trans_per_joint = Tslice_world_device @ landmarks_device[:,j,:]
                landmarks_world_transformed[:,j,:] = landmark_trans_per_joint
            
            
            return AriaHandAllPoseWrtWorld(
                confidence=torch.from_numpy(
                    np.array([d.confidence for d in detections])
                ),

                landmarks_3d=landmarks_world_transformed, # All 21 landmarks in world coordinates

                wrist_position = Tslice_world_device
                @ torch.from_numpy(
                    np.array(
                        [d.get_wrist_position_device() for d in detections], dtype=np.float32
                    )
                ),

                # TODO: maybe you need a if condition to select the available device
                wrist_normal = Rslice_world_device
                @torch.from_numpy(
                    np.array(
                        [d.wrist_and_palm_normal_device.wrist_normal_device for d in detections], dtype=np.float32
                    )
                ),

                palm_position = Tslice_world_device
                @ torch.from_numpy(
                    np.array(
                        [d.get_palm_position_device() for d in detections], dtype=np.float32
                    )
                ),

                # TODO: maybe you need a if condition to select the available device
                palm_normal = Rslice_world_device
                @torch.from_numpy(
                    np.array(
                        [d.wrist_and_palm_normal_device.palm_normal_device for d in detections], dtype=np.float32
                    )
                ),
                
                indices=torch.from_numpy(np.array(indices, dtype=np.int64)),
                )

        return CorrespondedAriaHandAllPoseWrtWorld(
            detections_left_concat=form_detections_concat(
                detections_left, indices_left
            ),
            detections_right_concat=form_detections_concat(
                detections_right, indices_right
            ),
        )

