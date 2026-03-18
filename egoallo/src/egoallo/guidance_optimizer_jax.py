"""Optimize constraints using Levenberg-Marquardt."""

from __future__ import annotations

import os

from .hand_detection_structs import (
    CorrespondedAriaHandWristPoseDetections,
    CorrespondedHamerDetections,
    CorrespondedAriaHandAllPoseWrtWorld,
)
from .imu_detection_structs import (
    CorrespondedImuReadings
)

# Need to play nice with PyTorch!
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import dataclasses
import time
from functools import partial
from typing import Callable, Literal, Unpack, assert_never, cast

import jax
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import torch
from jax import numpy as jnp
from jaxtyping import Float, Int
from torch import Tensor

from . import fncsmpl, fncsmpl_jax, network
from .transforms._so3 import SO3
from .imu_utils import IMU_CONFIG, JOINT_ID_TO_NAME_AND_PARENT
from .config import PARENT_JOINT_ID, BODY_24_ID2NAME


def do_guidance_optimization(
    Ts_world_cpf: Float[Tensor, "time 7"],
    traj: network.EgoDenoiseTraj,
    body_model: fncsmpl.SmplhModel,
    guidance_mode: GuidanceMode,
    phase: Literal["inner", "post"],
    hamer_detections: None | CorrespondedHamerDetections, # check the input format
    aria_detections: None | CorrespondedAriaHandWristPoseDetections, # check the input format
    aria_all_hand_detections: None | CorrespondedAriaHandAllPoseWrtWorld,
    imu_readings: None | CorrespondedImuReadings, # check the input format
    verbose: bool,
) -> tuple[network.EgoDenoiseTraj, dict]:
    """Run an optimizer to apply foot contact constraints."""

    assert traj.hand_rotmats is not None
    guidance_params = JaxGuidanceParams.defaults(guidance_mode, phase)

    start_time = time.time()
    quats, debug_info = _optimize_vmapped(
        body=fncsmpl_jax.SmplhModel(
            faces=cast(jax.Array, body_model.faces.numpy(force=True)),
            J_regressor=cast(jax.Array, body_model.J_regressor.numpy(force=True)),
            parent_indices=cast(jax.Array, onp.array(body_model.parent_indices)),
            weights=cast(jax.Array, body_model.weights.numpy(force=True)),
            posedirs=cast(jax.Array, body_model.posedirs.numpy(force=True)),
            v_template=cast(jax.Array, body_model.v_template.numpy(force=True)),
            shapedirs=cast(jax.Array, body_model.shapedirs.numpy(force=True)),
        ),
        Ts_world_cpf=cast(jax.Array, Ts_world_cpf.numpy(force=True)),
        betas=cast(jax.Array, traj.betas.numpy(force=True)),
        body_rotmats=cast(jax.Array, traj.body_rotmats.numpy(force=True)),
        hand_rotmats=cast(jax.Array, traj.hand_rotmats.numpy(force=True)),
        contacts=cast(jax.Array, traj.contacts.numpy(force=True)),
        guidance_params=guidance_params,
        # The hand detections are a torch tensors in a TensorDataclass form. We
        # use dictionaries to convert to pytrees.
        hamer_detections=None
        if hamer_detections is None
        else hamer_detections.as_nested_dict(numpy=True),
        aria_detections=None
        if aria_detections is None
        else aria_detections.as_nested_dict(numpy=True),
        aria_all_hand_detections=None
        if aria_all_hand_detections is None
        else aria_all_hand_detections.as_nested_dict(numpy=True), # TODO whether the all hand output is a dict. 
        imu_readings=None
        if imu_readings is None
        else imu_readings.as_nested_dict(numpy=True),
        verbose=verbose,
    )
    rotmats = SO3(
        torch.from_numpy(onp.array(quats))
        .to(traj.body_rotmats.dtype)
        .to(traj.body_rotmats.device)
    ).as_matrix()

    print(f"Constraint optimization finished in {time.time() - start_time}sec")
    return dataclasses.replace(
        traj,
        body_rotmats=rotmats[:, :, :21, :],
        hand_rotmats=rotmats[:, :, 21:, :],
    ), debug_info


class _SmplhBodyPosesVar(
    jaxls.Var[jax.Array],
    default_factory=lambda: jnp.concatenate(
        [jnp.ones((21, 1)), jnp.zeros((21, 3))], axis=-1
    ),
    retract_fn=lambda val, delta: (
        jaxlie.SO3(val) @ jaxlie.SO3.exp(delta.reshape(21, 3))
    ).wxyz,
    tangent_dim=21 * 3,
):
    """Variable containing local joint poses for a SMPL-H human."""


class _SmplhSingleHandPosesVar(
    jaxls.Var[jax.Array],
    default_factory=lambda: jnp.concatenate(
        [jnp.ones((15, 1)), jnp.zeros((15, 3))], axis=-1
    ),
    retract_fn=lambda val, delta: (
        jaxlie.SO3(val) @ jaxlie.SO3.exp(delta.reshape(15, 3))
    ).wxyz,
    tangent_dim=15 * 3,
):
    """Variable containing local joint poses for one hand of a SMPL-H human."""


@jdc.jit
def _optimize_vmapped(
    Ts_world_cpf: jax.Array,
    body: fncsmpl_jax.SmplhModel,
    betas: jax.Array,
    body_rotmats: jax.Array,
    hand_rotmats: jax.Array,
    contacts: jax.Array,
    guidance_params: JaxGuidanceParams,
    hamer_detections: dict | None,
    aria_detections: dict | None,
    aria_all_hand_detections: dict | None, 
    imu_readings: dict | None,
    verbose: jdc.Static[bool],
) -> tuple[jax.Array, dict]:

    return jax.vmap(
        partial(
            _optimize,
            Ts_world_cpf=Ts_world_cpf,
            body=body,
            guidance_params=guidance_params,
            hamer_detections=hamer_detections,
            aria_detections=aria_detections,
            aria_all_hand_detections=aria_all_hand_detections,
            imu_readings=imu_readings,
            verbose=verbose,
        )
    )(
        betas=betas,
        body_rotmats=body_rotmats,
        hand_rotmats=hand_rotmats,
        contacts=contacts,
    )


# Modes for guidance.
GuidanceMode = Literal[
    # Foot skating only.
    "no_hands",
    # Only use Aria wrist pose.
    "aria_wrist_only",
    # Use Aria wrist pose + HaMeR 3D estimates.
    "aria_hamer",
    # Use only HaMeR 3D estimates.
    "hamer_wrist",
    # Use HaMeR 3D estimates + reprojection.
    "hamer_reproj2",
    # Only use Aria all hand.
    "aria_all_hand",
    # Only use IMU.
    "imu_tto",
    # Use IMU and Aria all hand.
    "imu_and_all_aria",
    # Use IMU and Aria wrist pose.
    "imu_and_aria_wrist",
    # IMU and Hamer 3D estimates + reprojection.
    "imu_hamer_reproj2",
    # Use All information:
    "imu_hamer_aria_wrist",
]

@jdc.pytree_dataclass
class JaxGuidanceParams:
    prior_quat_weight: float = 1.0
    prior_pos_weight: float = 5.0 #5.0 control the torso_joint
    body_quat_vel_smoothness_weight: float = 5.0
    body_quat_smoothness_weight: float = 1.0
    body_quat_delta_smoothness_weight: float = 10.0
    skate_weight: float = 30.0

    # Note: this should be quite high. If the hand quaternions aren't
    # constrained enough the reprojecction loss can get wild.
    hand_quats: jdc.Static[bool] = True
    hand_quat_weight = 5.0

    hand_quat_priors: jdc.Static[bool] = True
    hand_quat_prior_weight = 0.1
    hand_quat_smoothness_weight = 10.0

    hamer_reproj: jdc.Static[bool] = True
    hand_reproj_weight: float = 1.0

    hamer_wrist_pose: jdc.Static[bool] = True
    hamer_abspos_weight: float = 20.0
    hamer_ori_weight: float = 5.0

    aria_wrists: jdc.Static[bool] = True
    aria_wrist_pos_weight: float = 50.0
    aria_wrist_ori_weight: float = 10.0

    # TODO: Change the aria_all_hand:
    aria_all_landmarks: jdc.Static[bool] = True
    aria_all_landmarks_weight: float = 50.0

    # TODO: Change the aria_all_hand:
    aria_all_hand: jdc.Static[bool] = True
    aria_all_wrist_pose_weight: float = 25.0
    aria_all_wrist_ori_weight: float = 5.0

    # TODO Here: add the weight for the imu readings: 
    # TODO: At the very begginning, make it super large, and then gradually decrease it.
    imu_readings: jdc.Static[bool] = True
    imu_local_quat_weight: float = 10.0  #5.0 
    imu_pelvis_relative_rotation_weight: float = 10.0 #5.0 
    imu_body_prior_weight: float = 0.1 #0.1
    # imu_body_smoothness_weight: float = 10.0

    # Optimization parameters.
    lambda_initial: float = 0.1
    max_iters: jdc.Static[int] = 50

    @staticmethod
    def defaults(
        mode: GuidanceMode,
        phase: Literal["inner", "post"],
    ) -> JaxGuidanceParams:
        # TODO: Add IMU mode condition to enable IMU guidance. 
        if mode == "no_hands":
            return {
                "inner": JaxGuidanceParams(
                    hand_quats=False,
                    hand_quat_priors=False,
                    hamer_reproj=False,
                    hamer_wrist_pose=False,
                    aria_wrists=False,
                    aria_all_hand=False,
                    aria_all_landmarks=False,
                    imu_readings=False,
                    max_iters=5,
                ),
                "post": JaxGuidanceParams(
                    hand_quats=False,
                    hand_quat_priors=False,
                    hamer_reproj=False,
                    hamer_wrist_pose=False,
                    aria_wrists=False,
                    aria_all_hand=False,
                    aria_all_landmarks=False,
                    imu_readings=False,
                    max_iters=20,
                ),
            }[phase]
        elif mode == "aria_wrist_only":
            return {
                "inner": JaxGuidanceParams(
                    hand_quats=False,
                    hand_quat_priors=True,
                    hamer_reproj=False,
                    hamer_wrist_pose=False,
                    aria_wrists=True,
                    aria_all_hand=False,
                    aria_all_landmarks=False,
                    imu_readings=False,
                    max_iters=5,
                ),
                "post": JaxGuidanceParams(
                    hand_quats=False,
                    hand_quat_priors=True,
                    hamer_reproj=False,
                    hamer_wrist_pose=False,
                    aria_wrists=True,
                    aria_all_hand=False,
                    aria_all_landmarks=False,
                    imu_readings=False,
                    max_iters=20,
                ),
            }[phase]
        elif mode == "aria_hamer":
            return {
                "inner": JaxGuidanceParams(
                    hand_quats=True,
                    hand_quat_priors=True,
                    hamer_reproj=False,
                    hamer_wrist_pose=False,
                    aria_wrists=True,
                    aria_all_hand=False,
                    aria_all_landmarks=False,
                    imu_readings=False,
                    max_iters=5,
                ),
                "post": JaxGuidanceParams(
                    hand_quats=True,
                    hand_quat_priors=True,
                    hamer_reproj=False,
                    hamer_wrist_pose=False,
                    aria_wrists=True,
                    aria_all_hand=False,
                    aria_all_landmarks=False,
                    imu_readings=False,
                    max_iters=20,
                ),
            }[phase]
        elif mode == "hamer_wrist":
            return {
                "inner": JaxGuidanceParams(
                    hand_quats=True,
                    hand_quat_priors=True,
                    # NOTE: we turn off reprojection during the inner loop optimization.
                    hamer_reproj=False,
                    hamer_wrist_pose=True,
                    aria_wrists=False,
                    aria_all_hand=False,
                    aria_all_landmarks=False,
                    imu_readings=False,
                    max_iters=5,
                ),
                "post": JaxGuidanceParams(
                    hand_quats=True,
                    hand_quat_priors=True,
                    # Turn on reprojection.
                    hamer_reproj=False,
                    hamer_wrist_pose=True,
                    aria_wrists=False,
                    aria_all_hand=False,
                    aria_all_landmarks=False,
                    imu_readings=False,
                    max_iters=20,
                ),
            }[phase]
        elif mode == "hamer_reproj2":
            return {
                "inner": JaxGuidanceParams(
                    hand_quats=True,
                    hand_quat_priors=True,
                    # NOTE: we turn off reprojection during the inner loop optimization.
                    hamer_reproj=False,
                    hamer_wrist_pose=True,
                    aria_wrists=False,
                    aria_all_hand=False,
                    aria_all_landmarks=False,
                    imu_readings=False,
                    max_iters=5,
                ),
                "post": JaxGuidanceParams(
                    hand_quats=True,
                    hand_quat_priors=True,
                    # Turn on reprojection.
                    hamer_reproj=True,
                    hamer_wrist_pose=True,
                    aria_wrists=False,
                    aria_all_hand=False,
                    aria_all_landmarks=False,
                    imu_readings=False,
                    max_iters=20,
                ),
            }[phase]
        elif mode == "aria_all_hand":
            return {
                "inner": JaxGuidanceParams(
                    hand_quats=False,
                    hand_quat_priors=True,
                    hamer_reproj=False,
                    hamer_wrist_pose=False,
                    aria_wrists=False,
                    aria_all_hand=True,
                    aria_all_landmarks=False,
                    imu_readings=False,
                    max_iters=5,
                ),
                "post": JaxGuidanceParams(
                    hand_quats=False,
                    hand_quat_priors=True,
                    hamer_reproj=False,
                    hamer_wrist_pose=False,
                    aria_wrists=False,
                    aria_all_hand=True,
                    aria_all_landmarks=False,
                    imu_readings=False,
                    max_iters=20,
                ),
            }[phase]
        elif mode == "imu_tto":
            return {
                    "inner": JaxGuidanceParams(
                        hand_quats=False,
                        hand_quat_priors=False, # True is previous version
                        hamer_reproj=False,
                        hamer_wrist_pose=False,
                        aria_wrists=False,
                        aria_all_hand=False,
                        aria_all_landmarks=False,
                        imu_readings=True,
                        max_iters=5,
                    ),
                    "post": JaxGuidanceParams(
                        hand_quats=False,
                        hand_quat_priors=False, # True is previous version
                        hamer_reproj=False,
                        hamer_wrist_pose=False,
                        aria_wrists=False,
                        aria_all_hand=False,
                        aria_all_landmarks=False,
                        imu_readings=True,
                        max_iters=20,
                    ),
            }[phase]
        elif mode == "imu_and_all_aria":
            return {
                    "inner": JaxGuidanceParams(
                        hand_quats=True,
                        hand_quat_priors=True,
                        hamer_reproj=False,
                        hamer_wrist_pose=False,
                        aria_wrists=False,
                        aria_all_hand=True,
                        aria_all_landmarks=False,
                        imu_readings=True,
                        max_iters=5,
                    ),
                    "post": JaxGuidanceParams(
                        hand_quats=True,
                        hand_quat_priors=True,
                        hamer_reproj=False,
                        hamer_wrist_pose=False,
                        aria_wrists=False,
                        aria_all_hand=True,
                        aria_all_landmarks=False,
                        imu_readings=True,
                        max_iters=20,
                    ),
            }[phase]
        elif mode == "imu_and_aria_wrist":
            return {
                "inner": JaxGuidanceParams(
                    hand_quats=False,
                    hand_quat_priors=True,
                    hamer_reproj=False,
                    hamer_wrist_pose=False,
                    aria_wrists=True,
                    aria_all_hand=False,
                    aria_all_landmarks=False,
                    imu_readings=True,
                    max_iters=5,
                ),
                "post": JaxGuidanceParams(
                    hand_quats=False,
                    hand_quat_priors=True,
                    hamer_reproj=False,
                    hamer_wrist_pose=False,
                    aria_wrists=True,
                    aria_all_hand=False,
                    aria_all_landmarks=False,
                    imu_readings=True,
                    max_iters=20,
                ),
            }[phase]
        elif mode == "imu_hamer_reproj2":
            return {
                "inner": JaxGuidanceParams(
                    hand_quats=True,
                    hand_quat_priors=True,
                    # NOTE: we turn off reprojection during the inner loop optimization.
                    hamer_reproj=False,
                    hamer_wrist_pose=True,
                    aria_wrists=False,
                    aria_all_hand=False,
                    aria_all_landmarks=False,
                    imu_readings=True,
                    max_iters=5,
                ),
                "post": JaxGuidanceParams(
                    hand_quats=True,
                    hand_quat_priors=True,
                    # Turn on reprojection.
                    hamer_reproj=True,
                    hamer_wrist_pose=True,
                    aria_wrists=False,
                    aria_all_hand=False,
                    aria_all_landmarks=False,
                    imu_readings=True,
                    max_iters=20,
                ),
            }[phase]
        elif mode == "imu_hamer_aria_wrist":
            return {
                "inner": JaxGuidanceParams(
                    hand_quats=True,
                    hand_quat_priors=True,
                    hamer_reproj=False,
                    hamer_wrist_pose=False,
                    aria_wrists=True,
                    aria_all_hand=False,
                    aria_all_landmarks=False,
                    imu_readings=True,
                    max_iters=5,
                ),
                "post": JaxGuidanceParams(
                    hand_quats=True,
                    hand_quat_priors=True,
                    hamer_reproj=False,
                    hamer_wrist_pose=False,
                    aria_wrists=True,
                    aria_all_hand=False,
                    aria_all_landmarks=False,
                    imu_readings=True,
                    max_iters=20,
                ),
            }[phase]
        else:
            assert_never(mode)


def _optimize(
    Ts_world_cpf: jax.Array,
    body: fncsmpl_jax.SmplhModel,
    betas: jax.Array,
    body_rotmats: jax.Array,
    hand_rotmats: jax.Array,
    contacts: jax.Array,
    guidance_params: JaxGuidanceParams,
    hamer_detections: dict | None,
    aria_detections: dict | None,
    aria_all_hand_detections: dict | None,
    imu_readings: dict | None,
    verbose: bool,
) -> tuple[jax.Array, dict]:
    """Apply constraints using Levenberg-Marquardt optimizer. Returns updated
    body_rotmats and hand_rotmats matrices."""
    timesteps = body_rotmats.shape[0]
    assert Ts_world_cpf.shape == (timesteps, 7)
    assert body_rotmats.shape == (timesteps, 21, 3, 3)
    assert hand_rotmats.shape == (timesteps, 30, 3, 3)
    # assert imu_readings.shape == (timesteps, 9, 3, 3) # imu_reading[timestamp][imu_num 1-9] = 3*3 matrix
    assert contacts.shape == (timesteps, 21)
    assert betas.shape == (timesteps, 16)

    init_quats = jaxlie.SO3.from_matrix(
        # body_rotmats
        jnp.concatenate([body_rotmats, hand_rotmats], axis=1)
    ).wxyz
    assert init_quats.shape == (timesteps, 51, 4)

    # Assume body shape is time-invariant.
    shaped_body = body.with_shape(jnp.mean(betas, axis=0))
    T_head_cpf = shaped_body.get_T_head_cpf()
    T_cpf_head = jaxlie.SE3(T_head_cpf).inverse().parameters()
    assert T_cpf_head.shape == (7,)

    init_posed = shaped_body.with_pose(
        jaxlie.SE3.identity(batch_axes=(timesteps,)).wxyz_xyz, init_quats
    )
    T_world_head = jaxlie.SE3(Ts_world_cpf) @ jaxlie.SE3(T_cpf_head)
    T_root_head = jaxlie.SE3(init_posed.Ts_world_joint[:, 14])
    init_posed = init_posed.with_new_T_world_root(
        (T_world_head @ T_root_head.inverse()).wxyz_xyz
    )
    del T_world_head
    del T_root_head

    foot_joint_indices = jnp.array([6, 7, 9, 10])
    num_foot_joints = foot_joint_indices.shape[0]

    contacts = contacts[..., foot_joint_indices]
    pairwise_contacts = (contacts[:-1, :] + contacts[1:, :]) / 2.0
    assert pairwise_contacts.shape == (timesteps - 1, num_foot_joints)
    del contacts

    # We'll populate a list of factors (cost terms).
    factors = list[jaxls.Cost]()

    def cost_with_args[*CostArgs](
        *args: Unpack[tuple[*CostArgs]],
    ) -> Callable[
        [Callable[[jaxls.VarValues, *CostArgs], jax.Array]],
        Callable[[jaxls.VarValues, *CostArgs], jax.Array],
    ]:
        """Decorator for appending to the factor list."""

        def inner(
            cost_func: Callable[[jaxls.VarValues, *CostArgs], jax.Array],
        ) -> Callable[[jaxls.VarValues, *CostArgs], jax.Array]:
            factors.append(jaxls.Cost(cost_func, args))
            return cost_func

        return inner

    def do_forward_kinematics(
        vals: jaxls.VarValues,
        var: _SmplhBodyPosesVar,
        left_hand: _SmplhSingleHandPosesVar | None = None,
        right_hand: _SmplhSingleHandPosesVar | None = None,
        output_frame: Literal["world", "root"] = "world",
    ) -> fncsmpl_jax.SmplhShapedAndPosed:
        """Helper for computing forward kinematics from variables."""
        assert (left_hand is None) == (right_hand is None)
        if left_hand is None and right_hand is None:
            posed = shaped_body.with_pose(
                T_world_root=jaxlie.SE3.identity().wxyz_xyz,
                local_quats=vals[var],
            )
        elif left_hand is not None and right_hand is None:
            posed = shaped_body.with_pose(
                T_world_root=jaxlie.SE3.identity().wxyz_xyz,
                local_quats=jnp.concatenate([vals[var], vals[left_hand]], axis=-2),
            )
        elif left_hand is not None and right_hand is not None:
            posed = shaped_body.with_pose(
                T_world_root=jaxlie.SE3.identity().wxyz_xyz,
                local_quats=jnp.concatenate(
                    [vals[var], vals[left_hand], vals[right_hand]], axis=-2
                ),
            )
        else:
            assert False

        if output_frame == "world":
            T_world_root = (
                # T_world_cpf
                jaxlie.SE3(Ts_world_cpf[var.id, :])
                # T_cpf_head
                @ jaxlie.SE3(T_cpf_head)
                # T_head_root
                @ jaxlie.SE3(posed.Ts_world_joint[14]).inverse()
            )
            return posed.with_new_T_world_root(T_world_root.wxyz_xyz)
        elif output_frame == "root":
            return posed
        
    # IMU pose cost:
    if imu_readings is not None and guidance_params.imu_readings:

        readings = imu_readings["readings"]


        # IMU pelvis relative rotation cost -- compute rotation difference between two consecutive IMU readings available timestamps
        @(
            cost_with_args(
                _SmplhBodyPosesVar(readings["indices"][:-1]),  # All frames except last
                _SmplhBodyPosesVar(readings["indices"][1:]),   # All frames except first
                readings["imu_readings"][:-1],  # All IMU readings except last
                readings["imu_readings"][1:],   # All IMU readings except first
            )
        )
        def imu_pelvis_relative_rotation_cost(
            vals: jaxls.VarValues,
            current_frame_from_model: _SmplhBodyPosesVar,
            next_frame_from_model: _SmplhBodyPosesVar,
            current_imu_reading: jax.Array,
            next_imu_reading: jax.Array,
        ) -> jax.Array:
            """
            Computes the relative rotation difference between consecutive timestamps
            for pelvis-relative rotations for both frames from model and IMU readings. This captures temporal dynamics.
            """
            # SECTION ONE:  MODEL LOCAL QUATERNIONS
            # Get current and next frame local quaternions
            current_local_quats = vals[current_frame_from_model]  # (21, 4)
            next_local_quats = vals[next_frame_from_model]        # (21, 4)
            # Compute pelvis rotation in ego world frame
            current_posed = do_forward_kinematics(vals, current_frame_from_model,output_frame="world")
            next_posed = do_forward_kinematics(vals, next_frame_from_model,output_frame="world")
            quat_current_in_ego_world = current_posed.T_world_root[:4] # (4,) quaternion
            quat_next_in_ego_world = next_posed.T_world_root[:4]
            # Noramlize quaternions
            quat_current_in_ego_world = jaxlie.SO3(quat_current_in_ego_world).wxyz
            quat_next_in_ego_world = jaxlie.SO3(quat_next_in_ego_world).wxyz
            # Compute relative rotation difference
            relative_rotation_from_model =  jaxlie.SO3(quat_current_in_ego_world).inverse() @ jaxlie.SO3(quat_next_in_ego_world)

            # SECTION TWO:  IMU READINGS
            # Get current and next IMU reading rotation matrices on pelvis
            pelvis_imu_id = 1 # pelvis
            pelvis_imu_reading_index = IMU_CONFIG[pelvis_imu_id][2]
            assert IMU_CONFIG[pelvis_imu_id][1] == "pelvis", f"IMU ID {pelvis_imu_id} does not correspond to pelvis"
            assert pelvis_imu_reading_index == pelvis_imu_id - 1, f"IMU ID {pelvis_imu_id} has reading index {pelvis_imu_reading_index} which is not equal to {pelvis_imu_id - 1}"
            # Get pelvis ration in SMPL world frame
            R_current_in_smpl_world = current_imu_reading[pelvis_imu_reading_index]
            R_next_in_smpl_world = next_imu_reading[pelvis_imu_reading_index]
            # Noramlize rotation matrices
            R_current_in_smpl_world = jaxlie.SO3.from_matrix(R_current_in_smpl_world).as_matrix()
            R_next_in_smpl_world = jaxlie.SO3.from_matrix(R_next_in_smpl_world).as_matrix()
            # Compute relative rotation difference
            R_next_in_current_IMU = jaxlie.SO3.from_matrix(R_current_in_smpl_world).inverse() @ jaxlie.SO3.from_matrix(R_next_in_smpl_world)

            return (
                guidance_params.imu_pelvis_relative_rotation_weight
                * (relative_rotation_from_model.inverse() @ R_next_in_current_IMU)
                .log()
                .flatten()
            )
        

        # # IMU Left and Right Shoulder Relative Rotation Cost 
        # @(
        #     cost_with_args(
        #         _SmplhBodyPosesVar(readings["indices"][:-1]),
        #         _SmplhBodyPosesVar(readings["indices"][1:]),
        #         readings["imu_readings"][:-1],
        #         readings["imu_readings"][1:],
        #     )
        # )
        # def imu_shoulder_relative_rotation_cost(
        #     vals: jaxls.VarValues,
        #     current_frame_from_model: _SmplhBodyPosesVar,
        #     next_frame_from_model: _SmplhBodyPosesVar,
        #     current_imu_reading: jax.Array,
        #     next_imu_reading: jax.Array,
        # ) -> jax.Array:
        #     """
        #     Computes the relative rotation difference between consecutive timestamps
        #     for left and right shoulder rotations for both frames from model and IMU readings. This captures temporal dynamics.
        #     """
        #     # SECTION ONE:  MODEL LOCAL QUATERNIONS
        #     # Get current and next frame local quaternions
        #     current_local_quats = vals[current_frame_from_model]  # (21, 4)
        #     next_local_quats = vals[next_frame_from_model]        # (21, 4)
        #     # Compute  rotation in ego world frame
        #     current_posed = do_forward_kinematics(vals, current_frame_from_model,output_frame="world")
        #     next_posed = do_forward_kinematics(vals, next_frame_from_model,output_frame="world")
            
        #     # Deal with left 
        #     left_shoulder_joint_id = 16 
        #     left_shoulder_joint_index_in_kinematic_tree = left_shoulder_joint_id - 1
        #     assert BODY_24_ID2NAME[left_shoulder_joint_id] == "L_shoulder", f"Joint ID {left_shoulder_joint_id} does not correspond to L_shoulder"
        #     quat_current_left_in_ego_world = current_posed.Ts_world_joint[left_shoulder_joint_index_in_kinematic_tree, :4]
        #     quat_next_left_in_ego_world = next_posed.Ts_world_joint[left_shoulder_joint_index_in_kinematic_tree, :4]
        #     quat_current_left_in_ego_world = jaxlie.SO3(quat_current_left_in_ego_world).wxyz
        #     quat_next_left_in_ego_world = jaxlie.SO3(quat_next_left_in_ego_world).wxyz
        #     R_next_in_current_from_model_left = jaxlie.SO3(quat_current_left_in_ego_world).inverse() @ jaxlie.SO3(quat_next_left_in_ego_world)

        #     # Deal with right
        #     right_shoulder_joint_id = 17
        #     right_shoulder_joint_index_in_kinematic_tree = right_shoulder_joint_id - 1
        #     assert BODY_24_ID2NAME[right_shoulder_joint_id] == "R_shoulder", f"Joint ID {right_shoulder_joint_id} does not correspond to R_shoulder"
        #     quat_current_right_in_ego_world = current_posed.Ts_world_joint[right_shoulder_joint_index_in_kinematic_tree, :4]
        #     quat_next_right_in_ego_world = next_posed.Ts_world_joint[right_shoulder_joint_index_in_kinematic_tree, :4]
        #     quat_current_right_in_ego_world = jaxlie.SO3(quat_current_right_in_ego_world).wxyz
        #     quat_next_right_in_ego_world = jaxlie.SO3(quat_next_right_in_ego_world).wxyz
        #     R_next_in_current_from_model_right = jaxlie.SO3(quat_current_right_in_ego_world).inverse() @ jaxlie.SO3(quat_next_right_in_ego_world)
            
        #     # SECTION TWO:  IMU READINGS
        #     # Left
        #     left_shoulder_IMU_id = 4
        #     left_shoulder_IMU_reading_index = IMU_CONFIG[left_shoulder_IMU_id][2]
        #     assert IMU_CONFIG[left_shoulder_IMU_id][1] == "L_shoulder", f"IMU ID {left_shoulder_IMU_id} does not correspond to L_shoulder"
        #     assert left_shoulder_IMU_reading_index == left_shoulder_IMU_id - 1, f"IMU ID {left_shoulder_IMU_id} has reading index {left_shoulder_IMU_reading_index} which is not equal to {left_shoulder_IMU_id - 1}"
        #     # Get left shoulder rotation matrix in SMPL world frame
        #     R_current_left_in_smpl_world = current_imu_reading[left_shoulder_IMU_reading_index]
        #     R_next_left_in_smpl_world = next_imu_reading[left_shoulder_IMU_reading_index]
        #     R_current_left_in_smpl_world = jaxlie.SO3.from_matrix(R_current_left_in_smpl_world).as_matrix()
        #     R_next_left_in_smpl_world = jaxlie.SO3.from_matrix(R_next_left_in_smpl_world).as_matrix()
        #     R_next_in_current_IMU_left = jaxlie.SO3.from_matrix(R_current_left_in_smpl_world).inverse() @ jaxlie.SO3.from_matrix(R_next_left_in_smpl_world)

        #     # Right
        #     right_shoulder_IMU_id = 3
        #     right_shoulder_IMU_reading_index = IMU_CONFIG[right_shoulder_IMU_id][2]
        #     assert IMU_CONFIG[right_shoulder_IMU_id][1] == "R_shoulder", f"IMU ID {right_shoulder_IMU_id} does not correspond to R_shoulder"
        #     assert right_shoulder_IMU_reading_index == right_shoulder_IMU_id - 1, f"IMU ID {right_shoulder_IMU_id} has reading index {right_shoulder_IMU_reading_index} which is not equal to {right_shoulder_IMU_id - 1}"
        #     # Get right shoulder rotation matrix in SMPL world frame
        #     R_current_right_in_smpl_world = current_imu_reading[right_shoulder_IMU_reading_index]
        #     R_next_right_in_smpl_world = next_imu_reading[right_shoulder_IMU_reading_index]
        #     R_current_right_in_smpl_world = jaxlie.SO3.from_matrix(R_current_right_in_smpl_world).as_matrix()
        #     R_next_right_in_smpl_world = jaxlie.SO3.from_matrix(R_next_right_in_smpl_world).as_matrix()
        #     R_next_in_current_IMU_right = jaxlie.SO3.from_matrix(R_current_right_in_smpl_world).inverse() @ jaxlie.SO3.from_matrix(R_next_right_in_smpl_world)

        #     # SECTION THREE: Combine costs
        #     left_rotation_cost = (R_next_in_current_IMU_left.inverse() @ R_next_in_current_from_model_left).log().flatten()
        #     right_rotation_cost = (R_next_in_current_IMU_right.inverse() @ R_next_in_current_from_model_right).log().flatten()
            
        #     combined_cost = jnp.concatenate([
        #         left_rotation_cost,
        #         right_rotation_cost
        #     ])
        #     return (
        #         guidance_params.imu_shoulder_relative_rotation_weight
        #         * combined_cost
        #     )
            

        # IMU local quaternion optimization - optimizes relative rotations between parent-child joints
        @(
            cost_with_args(
                _SmplhBodyPosesVar(readings["indices"]),
                readings["imu_readings"],
                # readings['confidence'],
            )
        )
        def imu_local_quat_cost(
            vals: jaxls.VarValues,
            body_pose: _SmplhBodyPosesVar,  
            imu_reading_details: jax.Array,
            # confidence: jax.Array,
        ) -> jax.Array:
            """
            IMU local quaternion cost: optimizes the relative rotation between 
            parent and child joints in the kinematic chain.
            Local quaternions represent the rotation of a joint relative to its parent joint.
            """
            local_quats_from_model_all = vals[body_pose]  # (21, 4)
            assert local_quats_from_model_all.shape == (21, 4)
            assert imu_reading_details.shape == (9, 3, 3)

            # Step 1: Map: from joint name to IMU rotation reading in the SMPL world frames
            joint_name_to_imu_rot_in_smpl_world = {}  # key: joint name, value: rotation matrix
            for imu_index in range(imu_reading_details.shape[0]):
                imu_id = imu_index + 1
                assert imu_index == IMU_CONFIG[imu_id][2], f"IMU index {imu_index} does not match the true index {IMU_CONFIG[imu_id][2]}"

                joint_name = IMU_CONFIG[imu_id][1]
                # NORMALIZATION POINT 1: Normalize IMU rotation matrices to ensure SO(3) properties
                imu_rotation = imu_reading_details[imu_index]
                imu_rotation_normalized = jaxlie.SO3.from_matrix(imu_rotation).as_matrix()
                joint_name_to_imu_rot_in_smpl_world[joint_name] = imu_rotation_normalized

            # Step 2: Process each IMU joint
            imu_local_quats = []
            model_local_quats = [] 
            
            for imu_index in range(imu_reading_details.shape[0]):
                imu_id = imu_index + 1
                assert imu_id in IMU_CONFIG, f"IMU ID {imu_id} not found in IMU_CONFIG"
                assert imu_index == IMU_CONFIG[imu_id][2], f"IMU index {imu_index} does not match the true index {IMU_CONFIG[imu_id][2]}"
                
                joint_id = IMU_CONFIG[imu_id][0]
                child_joint_index_in_kinematic_tree = joint_id - 1
                joint_name = IMU_CONFIG[imu_id][1]

                # get the parent name & the parent rotation in the SMPL world frames & the local quaternion from diffusion model
                parent_name = JOINT_ID_TO_NAME_AND_PARENT[joint_id][1]["parent"]
                local_quats_from_model_all = jaxlie.SO3(local_quats_from_model_all).wxyz # noramlize
                parent_rot_in_smpl_world, local_quat_in_parent_frame_from_model = _get_parent_joint_rot_in_smpl_world_and_model_quat(
                    parent_name, joint_name_to_imu_rot_in_smpl_world, local_quats_from_model_all, 
                    child_joint_index_in_kinematic_tree
                )
                
                if parent_rot_in_smpl_world is None:
                    # skip the pelvis joint as it doesn't have a parent rotation
                    continue
                
                # Compute IMU local quaternion
                # NORMALIZATION POINT 2: Normalize parent rotation matrix before inverse operation
                parent_rot_normalized = jaxlie.SO3.from_matrix(parent_rot_in_smpl_world).as_matrix()
                # NORMALIZATION POINT 2.5: Also normalize the IMU reading before matrix multiplication
                imu_reading_normalized = jaxlie.SO3.from_matrix(imu_reading_details[imu_index]).as_matrix()
                R_parent_joint_local = jaxlie.SO3.from_matrix(parent_rot_normalized).inverse().as_matrix() @ imu_reading_normalized
                
                # NORMALIZATION POINT 3: Normalize the computed local rotation matrix
                R_parent_joint_local = jaxlie.SO3.from_matrix(R_parent_joint_local).as_matrix()
                quat_parent_joint_local = jaxlie.SO3.from_matrix(R_parent_joint_local).wxyz
                
                # Store results
                imu_local_quats.append(quat_parent_joint_local)
                model_local_quats.append(local_quat_in_parent_frame_from_model)
            
            # Step 3: Compute and return cost
            assert len(imu_local_quats) == len(model_local_quats)
            
            model_local_so3 = jaxlie.SO3(jnp.array(model_local_quats))
            imu_local_so3 = jaxlie.SO3(jnp.array(imu_local_quats))

            return (
                guidance_params.imu_local_quat_weight 
                * (model_local_so3.inverse() @ imu_local_so3)
                .log()
                .flatten()
            )


        def _get_parent_joint_rot_in_smpl_world_and_model_quat(
            parent_joint_name: str,
            joint_name_to_imu_rot_in_smpl_world: dict,
            local_quats_from_model_all: jax.Array,
            child_joint_index_in_kinematic_tree: int
        ) -> tuple[jax.Array | None, jax.Array]:
            """
            Helper function to get parent rotation and model local quaternion.
            parent_joint_name: normally the parent joint name from which you want to retrieve 
                                the rotation matrix in the SMPL world frame.
            joint_name_to_imu_rot_in_smpl_world: dict mapping each joint name to its 
                                                 corresponding IMU rotation in the SMPL world frame.
            local_quats: (21 x 4) array containing all local quaternion predictions from the model.
            child_joint_index_in_kinematic_tree: the index of the current joint in the kinematic tree.
            """
            
            if parent_joint_name in joint_name_to_imu_rot_in_smpl_world:
                # Your current joint can be observed by IMU  
                # get the parent rotation in the SMPL world frames
                # get the local quaternion in the parent frame from the diffusion model
                parent_rot_in_smpl_world = joint_name_to_imu_rot_in_smpl_world[parent_joint_name]
                local_quat_in_parent_frame_from_model = local_quats_from_model_all[child_joint_index_in_kinematic_tree, :]
                # NORMALIZATION POINT 1.5: Normalize quaternion from diffusion model
                local_quat_in_parent_frame_from_model = jaxlie.SO3(local_quat_in_parent_frame_from_model).wxyz
                return parent_rot_in_smpl_world, local_quat_in_parent_frame_from_model
                
            elif parent_joint_name is None:
                # Skip the pelvis joint as no parent and paret_joint_name is None
                return None, None
                
            else:
                # Parent joint is not observable by IMU
                # draw the long kinematic chain until you find a joint that is observable by IMU
                # normally pelvis
                
                # Compute composite model local quaternion through kinematic chain
                # Initialize with identity quaternion using JAX
                local_quat_in_parent_frame_from_model = jnp.array([1.0, 0.0, 0.0, 0.0])
                
                while True:
                    if parent_joint_name == "pelvis":
                        break
                        
                    # Get the local quaternion for current joint
                    current_local_quat_from_model_temp = local_quats_from_model_all[child_joint_index_in_kinematic_tree, :]
                    
                    # Use jaxlie.SO3 for quaternion multiplication (JAX-compatible)
                    so3_accumulated = jaxlie.SO3(local_quat_in_parent_frame_from_model)
                    so3_current = jaxlie.SO3(current_local_quat_from_model_temp)
                    
                    # Multiply quaternions using jaxlie.SO3
                    so3_result = so3_current @ so3_accumulated
                    local_quat_in_parent_frame_from_model = so3_result.wxyz
                    # NORMALIZATION POINT 4: Quaternion is automatically normalized by jaxlie.SO3

                    # Move up the kinematic chain
                    parent_joint_index_in_kinematic_tree = PARENT_JOINT_ID[child_joint_index_in_kinematic_tree] - 1
                    parent_id = PARENT_JOINT_ID[parent_joint_index_in_kinematic_tree] #update parent joint name 
                    parent_joint_name = BODY_24_ID2NAME[parent_id]
                    child_joint_index_in_kinematic_tree = parent_joint_index_in_kinematic_tree
                
                parent_rot_in_smpl_world = joint_name_to_imu_rot_in_smpl_world[parent_joint_name]
                return parent_rot_in_smpl_world, local_quat_in_parent_frame_from_model


        # IMU body local quaternion smoothness.
        @(
            cost_with_args(
                _SmplhBodyPosesVar(jnp.arange(timesteps - 1)),
                _SmplhBodyPosesVar(jnp.arange(1, timesteps)),
                # guidance_params.body_quat_smoothness_weights,
                # Adjust weight for different joints
            )
        )
        def body_quat_smoothness_first_order(
            vals: jaxls.VarValues,
            body_pose_t: _SmplhBodyPosesVar,      
            body_pose_t1: _SmplhBodyPosesVar,    
            # joint_weights: jax.Array,           
        ) -> jax.Array:
            Rt  = jaxlie.SO3(vals[body_pose_t])
            Rt1 = jaxlie.SO3(vals[body_pose_t1])
            rel = Rt.inverse() @ Rt1              
            so3 = rel.log()                      
            return guidance_params.body_quat_smoothness_weight * (so3).flatten()
        
        # Whole body prior loss. 
        @cost_with_args(
            _SmplhBodyPosesVar(jnp.arange(timesteps)),
            init_quats[:, :21, :].reshape((timesteps, 21, 4)),
        )
        def body_prior(
            vals: jaxls.VarValues,
            body_pose: _SmplhBodyPosesVar,
            init_body_quats: jax.Array,
        ) -> jax.Array:
            return (
                guidance_params.imu_body_prior_weight
                * (jaxlie.SO3(vals[body_pose]).inverse() @ jaxlie.SO3(init_body_quats))
                .log()
                .flatten()
            )

    
    # HaMeR pose cost.
    if hamer_detections is not None and guidance_params.hand_quat_priors:
        hamer_left = hamer_detections["detections_left_concat"]
        hamer_right = hamer_detections["detections_right_concat"]

        # HaMeR local quaternion smoothness.
        @(
            cost_with_args(
                _SmplhSingleHandPosesVar(jnp.arange(timesteps * 2 - 2)),
                _SmplhSingleHandPosesVar(jnp.arange(2, timesteps * 2)),
            )
        )
        def hand_smoothness(
            vals: jaxls.VarValues,
            hand_pose: _SmplhSingleHandPosesVar,
            hand_pose_next: _SmplhSingleHandPosesVar,
        ) -> jax.Array:
            return (
                guidance_params.hand_quat_smoothness_weight
                * (
                    jaxlie.SO3(vals[hand_pose]).inverse()
                    @ jaxlie.SO3(vals[hand_pose_next])
                )
                .log()
                .flatten()
            )

        # Hand prior loss.
        @cost_with_args(
            _SmplhSingleHandPosesVar(jnp.arange(timesteps * 2)),
            init_quats[:, 21:51, :].reshape((timesteps * 2, 15, 4)),
        )
        def hand_prior(
            vals: jaxls.VarValues,
            hand_pose: _SmplhSingleHandPosesVar,
            init_hand_quats: jax.Array,
        ) -> jax.Array:
            return (
                guidance_params.hand_quat_prior_weight
                * (jaxlie.SO3(vals[hand_pose]).inverse() @ jaxlie.SO3(init_hand_quats))
                .log()
                .flatten()
            )

    if hamer_detections is not None and guidance_params.hand_quats:
        hamer_left = hamer_detections["detections_left_concat"]
        hamer_right = hamer_detections["detections_right_concat"]

        # HaMeR local pose matching.
        @(
            cost_with_args(
                _SmplhSingleHandPosesVar(hamer_left["indices"] * 2),
                hamer_left["single_hand_quats"],
            )
            if hamer_left is not None
            else lambda x: x
        )
        @(
            cost_with_args(
                _SmplhSingleHandPosesVar(hamer_right["indices"] * 2 + 1),
                hamer_right["single_hand_quats"],
            )
            if hamer_right is not None
            else lambda x: x
        )
        def hamer_local_pose_cost(
            vals: jaxls.VarValues,
            hand_pose: _SmplhSingleHandPosesVar,
            estimated_hand_quats: jax.Array,
        ) -> jax.Array:
            hand_quats = vals[hand_pose]
            assert hand_quats.shape == estimated_hand_quats.shape
            return guidance_params.hand_quat_weight * (
                (jaxlie.SO3(hand_quats).inverse() @ jaxlie.SO3(estimated_hand_quats))
                .log()
                .flatten()
            )

    if aria_all_hand_detections is not None and guidance_params.aria_all_landmarks:
        aria_all_left = aria_all_hand_detections["detections_left_concat"]
        aria_all_right = aria_all_hand_detections["detections_right_concat"]

        mano_from_openpose_indices = jnp.array([5, 8, 9, 10, 11, 12, 13, 17, 18, 19, 14, 15, 16, 6, 7])
        # Aria_landmark 3D in world:
        # We'll get the indices inside the cost function based on left0_right1 parameter
        # what do you have? 
        # what did hamer have? 

        @(
            cost_with_args(
                _SmplhBodyPosesVar(aria_all_left["indices"]),
                _SmplhSingleHandPosesVar(aria_all_left["indices"] * 2),
                _SmplhSingleHandPosesVar(aria_all_left["indices"] * 2 + 1),
                jnp.full_like(aria_all_left["indices"], fill_value=0),
                aria_all_left["landmarks_3d"],
            )
            if aria_all_left is not None
            else lambda x: x
        )
        @(
            cost_with_args( 
                _SmplhBodyPosesVar(aria_all_right["indices"]),
                _SmplhSingleHandPosesVar(aria_all_right["indices"] * 2),
                _SmplhSingleHandPosesVar(aria_all_right["indices"] * 2 + 1),
                jnp.full_like(aria_all_right["indices"], fill_value=1),
                aria_all_right["landmarks_3d"],
            )
            if aria_all_right is not None
            else lambda x: x
        )
        def hand_3d_alignment_cost_world(
            vals: jaxls.VarValues,
            body_pose: _SmplhBodyPosesVar,
            left_hand_pose: _SmplhSingleHandPosesVar,
            right_hand_pose: _SmplhSingleHandPosesVar,
            left0_right1: jax.Array, # Set to 0 for left, 1 for right.
            landmarks_3d: jax.Array, # Aria landmarks in world coordinates (21, 3)
        ) -> jax.Array:

            # Forward kinematics for all joints in world frame
            posed = do_forward_kinematics(
                vals, 
                body_pose, 
                left_hand_pose, 
                right_hand_pose, 
                output_frame="world",
            )
            Ts_world_joint = posed.Ts_world_joint # get the joint in world frame (51, 7) - 21 body + 30 hand joints
            del posed

            assert left0_right1.shape == ()
            assert Ts_world_joint.shape == (51, 7)
            
            # Get hand joint positions from forward kinematics
            # Hand joints start at index 21 in the joint array
            # For left hand: indices 21-35 (15 joints), for right hand: indices 36-50 (15 joints)
            wrist_index = 19 + left0_right1
            hand_start_idx = 21 + left0_right1 * 15  # 21 for left, 36 for right
            joint_positions_wrt_world = Ts_world_joint[:, 4:7]
            # Get wrist joint
            wrist_joint = jax.lax.dynamic_slice_in_dim(
                joint_positions_wrt_world, 
                start_index=wrist_index,
                slice_size=1,
                axis=-2,
            )
            
            # Get all 15 hand joints
            all_hand_joints = jax.lax.dynamic_slice_in_dim(
                joint_positions_wrt_world, 
                start_index=hand_start_idx,
                slice_size=15,
                axis=-2,
            )
            
            # Remove thumb proximal joint (index 13 within the 15 hand joints)
            # Split into parts before and after thumb proximal
            joints_before_thumb = jax.lax.dynamic_slice_in_dim(
                all_hand_joints,
                start_index=0,
                slice_size=13,  # joints 0-12
                axis=-2,
            )
            joints_after_thumb = jax.lax.dynamic_slice_in_dim(
                all_hand_joints,
                start_index=14,  # skip index 13 (thumb proximal)
                slice_size=1,    # only 1 joint after thumb proximal
                axis=-2,
            )
            
            # Combine: wrist + 13 joints before thumb + 1 joint after thumb = 15 total
            mano_hand_joints_wrt_world = jnp.concatenate(
                [wrist_joint, joints_before_thumb, joints_after_thumb],
                axis=0,
            )
            assert mano_hand_joints_wrt_world.shape == (15, 3)
            assert landmarks_3d.shape == (21, 3)
            
            # Map Aria landmarks to MANO joints using the same mapping as HaMeR
            # Get the Aria landmark indices that correspond to MANO joint order
            aria_landmark_indices = mano_from_openpose_indices
            
            # Select the corresponding Aria landmarks
            selected_landmarks = landmarks_3d[aria_landmark_indices]  # (16, 3)
            
            # Since there are no tips to consider, we can directly use the MANO joint positions
            # mano_hand_joints_wrt_world already has 16 joints (wrist + 15 hand joints)
            # and selected_landmarks also has 16 landmarks, so they match directly
            
            # Compute relative distances for key hand relationships
            # Focus on finger lengths and inter-finger distances for better efficiency
            
            # # Keep the absolute one
            # # Compute alignment cost between Aria landmarks and MANO joint positions
            # alignment_error = mano_hand_joints_wrt_world - selected_landmarks


            # Method 1: Use wrist-relative positions (translation invariant)
            wrist_pos = mano_hand_joints_wrt_world[0]  # wrist is first joint
            aria_wrist_pos = selected_landmarks[0]
            
            # Compute wrist-relative positions for MANO joints
            mano_relative = mano_hand_joints_wrt_world - wrist_pos
            aria_relative = selected_landmarks - aria_wrist_pos
            
            # Method 2: Compute finger lengths (scale invariant)
            # Define finger joint sequences (assuming standard MANO order)
            finger_joints = [
                [0, 1, 2, 3],      # thumb
                [0, 4, 5, 6],      # index
                [0, 7, 8, 9],      # middle  
                [0, 10, 11, 12],   # ring
                [0, 13, 14, 15],   # pinky
            ]
            
            mano_finger_lengths = []
            aria_finger_lengths = []
            
            for finger in finger_joints:
                # Compute segment lengths for each finger
                for i in range(len(finger) - 1):
                    mano_len = jnp.linalg.norm(mano_hand_joints_wrt_world[finger[i+1]] - mano_hand_joints_wrt_world[finger[i]])
                    aria_len = jnp.linalg.norm(selected_landmarks[finger[i+1]] - selected_landmarks[finger[i]])
                    mano_finger_lengths.append(mano_len)
                    aria_finger_lengths.append(aria_len)
            
            mano_finger_lengths = jnp.array(mano_finger_lengths)
            aria_finger_lengths = jnp.array(aria_finger_lengths)
            
            # Combine relative position error and finger length error
            # Use a weighted combination to balance shape preservation
            relative_pos_error = (mano_relative - aria_relative).flatten()
            finger_length_error = mano_finger_lengths - aria_finger_lengths
            
            # Combine both errors (you can adjust weights as needed)
            relative_distance_error = jnp.concatenate([
                relative_pos_error, #* 0.7,  # 70% weight for relative positions
                # finger_length_error * 0.3   # 30% weight for finger lengths
            ])
            
            return (
                guidance_params.aria_all_landmarks_weight
                * relative_distance_error
                # * alignment_error.flatten() # This is for absolute error.
            )

    if hamer_detections is not None and (
        guidance_params.hamer_reproj and guidance_params.hamer_wrist_pose
    ):
        hamer_left = hamer_detections["detections_left_concat"]
        hamer_right = hamer_detections["detections_right_concat"]

        # HaMeR reprojection.
        mano_from_openpose_indices = _get_mano_from_openpose_indices(include_tips=False)

        @(
            cost_with_args(
                _SmplhBodyPosesVar(hamer_left["indices"]),
                _SmplhSingleHandPosesVar(hamer_left["indices"] * 2),
                _SmplhSingleHandPosesVar(hamer_left["indices"] * 2 + 1),
                jnp.full_like(hamer_left["indices"], fill_value=0),
                hamer_left["keypoints_3d"],
                hamer_left["mano_hand_global_orient"],
            )
            if hamer_left is not None
            else lambda x: x
        )
        @(
            cost_with_args(
                _SmplhBodyPosesVar(hamer_right["indices"]),
                _SmplhSingleHandPosesVar(hamer_right["indices"] * 2),
                _SmplhSingleHandPosesVar(hamer_right["indices"] * 2 + 1),
                jnp.full_like(hamer_right["indices"], fill_value=1),
                hamer_right["keypoints_3d"],
                hamer_right["mano_hand_global_orient"],
            )
            if hamer_right is not None
            else lambda x: x
        )
        def hamer_wrist_and_reproj(
            vals: jaxls.VarValues,
            body_pose: _SmplhBodyPosesVar,
            left_hand_pose: _SmplhSingleHandPosesVar,
            right_hand_pose: _SmplhSingleHandPosesVar,
            left0_right1: jax.Array,  # Set to 0 for left, 1 for right.
            keypoints3d_wrt_cam: jax.Array,  # These are in OpenPose order!!
            Rmat_cam_wrist: jax.Array,
        ) -> jax.Array:
            posed = do_forward_kinematics(
                # The right hand comes _after_ the left hand, we can exclude it.
                vals,
                body_pose,
                left_hand_pose,
                right_hand_pose,
                output_frame="root",
            )
            Ts_root_joint = posed.Ts_world_joint  # Sorry for the naming...
            del posed

            # 19 for left wrist, 20 for right wrist.
            wrist_index = 19 + left0_right1
            hand_start_index = 21 + 15 * left0_right1

            assert Ts_root_joint.shape == (51, 7)
            joint_positions_wrt_root = Ts_root_joint[:, 4:7]
            mano_joints_wrt_root = jnp.concatenate(
                [
                    jax.lax.dynamic_slice_in_dim(
                        joint_positions_wrt_root,
                        start_index=wrist_index,
                        slice_size=1,
                        axis=-2,
                    ),
                    jax.lax.dynamic_slice_in_dim(
                        joint_positions_wrt_root,
                        start_index=hand_start_index,
                        slice_size=15,
                        axis=-2,
                    ),
                ],
                axis=0,
            )
            assert mano_joints_wrt_root.shape == (16, 3)
            assert keypoints3d_wrt_cam.shape == (21, 3)  # In OpenPose.

            T_cam_root = (
                # T_cam_cpf (7,)
                jaxlie.SE3(hamer_detections["T_cpf_cam"]).inverse()
                # T_cpf_head (7,)
                @ jaxlie.SE3(T_cpf_head)
                # T_head_root (7,)
                @ jaxlie.SE3(Ts_root_joint[14, :]).inverse()
            )
            assert T_cam_root.parameters().shape == (7,)
            mano_joints_wrt_cam = T_cam_root @ mano_joints_wrt_root
            obs_joints_wrt_cam = keypoints3d_wrt_cam[mano_from_openpose_indices, :]

            mano_uv_wrt_cam = mano_joints_wrt_cam[:, :2] / mano_joints_wrt_cam[:, 2:3]
            obs_uv_wrt_cam = obs_joints_wrt_cam[:, :2] / obs_joints_wrt_cam[:, 2:3]

            T_cam_wrist = jaxlie.SE3.from_rotation_and_translation(
                T_cam_root.rotation() @ jaxlie.SO3(Ts_root_joint[wrist_index, :4]),
                mano_joints_wrt_cam[0, :],
            )
            obs_T_cam_wrist = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3.from_matrix(Rmat_cam_wrist),
                obs_joints_wrt_cam[0, :],
            )

            return jnp.concatenate(
                [
                    (T_cam_wrist.inverse() @ obs_T_cam_wrist).log()
                    * jnp.array(
                        [guidance_params.hamer_abspos_weight] * 3
                        + [guidance_params.hamer_ori_weight] * 3
                    ),
                    guidance_params.hand_reproj_weight
                    * (mano_uv_wrt_cam - obs_uv_wrt_cam).flatten(),
                ]
            )
    elif (
        hamer_detections is not None
        and not guidance_params.hamer_reproj
        and guidance_params.hamer_wrist_pose
    ):
        hamer_left = hamer_detections["detections_left_concat"]
        hamer_right = hamer_detections["detections_right_concat"]

        @(
            cost_with_args(
                _SmplhBodyPosesVar(hamer_left["indices"]),
                jnp.full_like(hamer_left["indices"], fill_value=0),
                hamer_left["keypoints_3d"],
                hamer_left["mano_hand_global_orient"],
            )
            if hamer_left is not None
            else lambda x: x
        )
        @(
            cost_with_args(
                _SmplhBodyPosesVar(hamer_right["indices"]),
                jnp.full_like(hamer_right["indices"], fill_value=1),
                hamer_right["keypoints_3d"],
                hamer_right["mano_hand_global_orient"],
            )
            if hamer_right is not None
            else lambda x: x
        )
        def hamer_wrist_only(
            vals: jaxls.VarValues,
            body_pose: _SmplhBodyPosesVar,
            left0_right1: jax.Array,  # Set to 0 for left, 1 for right.
            keypoints3d_wrt_cam: jax.Array,  # These are in OpenPose order!!
            Rmat_cam_wrist: jax.Array,
        ) -> jax.Array:
            posed = do_forward_kinematics(vals, body_pose, output_frame="root")
            Ts_root_joint = posed.Ts_world_joint  # Sorry for the naming...
            del posed

            # 19 for left wrist, 20 for right wrist.
            wrist_index = 19 + left0_right1

            assert Ts_root_joint.shape == (21, 7)
            wrist_position_wrt_root = Ts_root_joint[wrist_index, 4:7]

            T_cam_root = (
                # T_cam_cpf (7,)
                jaxlie.SE3(hamer_detections["T_cpf_cam"]).inverse()
                # T_cpf_head (7,)
                @ jaxlie.SE3(T_cpf_head)
                # T_head_root (7,)
                @ jaxlie.SE3(Ts_root_joint[14, :]).inverse()
            )
            assert T_cam_root.parameters().shape == (7,)
            wrist_position_wrt_cam = T_cam_root @ wrist_position_wrt_root

            # Assumes OpenPose root is same as Mano root!!
            wrist_pos_wrt_cam = keypoints3d_wrt_cam[0, :]

            T_cam_wrist = jaxlie.SE3.from_rotation_and_translation(
                T_cam_root.rotation() @ jaxlie.SO3(Ts_root_joint[wrist_index, :4]),
                wrist_position_wrt_cam,
            )
            obs_T_cam_wrist = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3.from_matrix(Rmat_cam_wrist),
                wrist_pos_wrt_cam,
            )
            return (T_cam_wrist.inverse() @ obs_T_cam_wrist).log() * jnp.array(
                [guidance_params.hamer_abspos_weight] * 3
                + [guidance_params.hamer_ori_weight] * 3
            )

    # Wrist pose cost.
    if aria_detections is not None and guidance_params.aria_wrists:
        aria_left = aria_detections["detections_left_concat"]
        aria_right = aria_detections["detections_right_concat"]

        @(
            cost_with_args(
                _SmplhBodyPosesVar(aria_left["indices"]),
                aria_left["confidence"],
                aria_left["wrist_position"],
                aria_left["palm_position"],
                aria_left["palm_normal"],
                jnp.full_like(aria_left["indices"], fill_value=0),
            )
            if aria_left is not None
            else lambda x: x
        )
        @(
            cost_with_args(
                _SmplhBodyPosesVar(aria_right["indices"]),
                aria_right["confidence"],
                aria_right["wrist_position"],
                aria_right["palm_position"],
                aria_right["palm_normal"],
                jnp.full_like(aria_right["indices"], fill_value=1),
            )
            if aria_right is not None
            else lambda x: x
        )
        def wrist_pose_cost(
            vals: jaxls.VarValues,
            pose: _SmplhBodyPosesVar,
            confidence: jax.Array,
            wrist_position: jax.Array,
            palm_position: jax.Array,
            palm_normal: jax.Array,
            left0_right1: jax.Array,  # Set to 0 for left, 1 for right.
        ) -> jax.Array:
            assert wrist_position.shape == (3,)
            assert left0_right1.shape == ()
            posed = do_forward_kinematics(vals, pose)

            T_world_wrist = posed.Ts_world_joint[19 + left0_right1]

            pos_cost = (
                # Left wrist is joint 19, right is joint 20.
                T_world_wrist[4:7] - wrist_position
            )

            # Estimate wrist orientation from forward + normal directions.
            palm_forward = palm_position - wrist_position
            palm_forward = palm_forward / jnp.linalg.norm(palm_forward)
            palm_normal = palm_normal / jnp.linalg.norm(palm_normal)
            palm_forward = (  # Flip palm forward if right hand.
                palm_forward * jnp.array([1, -1])[left0_right1]
            )
            palm_forward = (  # Gram-schmidt for forward direction.
                palm_forward - jnp.dot(palm_forward, palm_normal) * palm_normal
            )
            estimatedR_world_wrist = jaxlie.SO3.from_matrix(
                jnp.stack(
                    [
                        palm_forward,
                        -palm_normal,
                        jnp.cross(palm_normal, palm_forward),
                    ],
                    axis=1,
                )
            )
            R_world_wrist = jaxlie.SO3(T_world_wrist[:4])
            ori_cost = (estimatedR_world_wrist.inverse() @ R_world_wrist).log()

            return confidence * jnp.concatenate(
                [
                    guidance_params.aria_wrist_pos_weight * pos_cost,
                    guidance_params.aria_wrist_ori_weight * ori_cost,
                ]
            )
    
    # All hand pose cost.
    if aria_all_hand_detections is not None and guidance_params.aria_all_hand:
        aria_all_left = aria_all_hand_detections["detections_left_concat"]
        aria_all_right = aria_all_hand_detections["detections_right_concat"]
        
        # 
        @(
            cost_with_args(
                _SmplhBodyPosesVar(aria_all_left["indices"]),
                aria_all_left["confidence"],
                aria_all_left["landmarks_3d"],
                aria_all_left["wrist_position"],
                aria_all_left["wrist_normal"],
                aria_all_left["palm_position"],
                aria_all_left["palm_normal"],
                jnp.full_like(aria_all_left["indices"], fill_value=0),
            )
            if aria_all_left is not None
            else lambda x: x
        )
        @(
            cost_with_args(
                _SmplhBodyPosesVar(aria_all_right["indices"]),
                aria_all_right["confidence"],
                aria_all_right["landmarks_3d"],
                aria_all_right["wrist_position"],
                aria_all_right["wrist_normal"],
                aria_all_right["palm_position"],
                aria_all_right["palm_normal"],
                jnp.full_like(aria_all_right["indices"], fill_value=1),
            )
            if aria_all_right is not None
            else lambda x: x
        )
        def wrist_pose_cost_in_hand_all_tracking(
            vals: jaxls.VarValues,
            pose: _SmplhBodyPosesVar,
            confidence: jax.Array,
            landmarks_3d: jax.Array, # (21, 3)
            wrist_position: jax.Array, # (3,)
            wrist_normal: jax.Array, # (3,)
            palm_position: jax.Array, # (3,)
            palm_normal: jax.Array, # (3,)
            left0_right1: jax.Array, # ()
        ) -> jax.Array:
            assert wrist_position.shape == (3,) and palm_position.shape == (3,) and palm_normal.shape == (3,) and wrist_normal.shape == (3,)
            assert landmarks_3d.shape == (21, 3)
            assert left0_right1.shape == ()
            # Compute forward kinematics to get joint position from the model.
            posed = do_forward_kinematics(vals, pose)

            # Compute wrist position cost
            T_world_wrist = posed.Ts_world_joint[19 + left0_right1] # wrist from model
            pose_cost = (
                # Left wrist is joint 19, right is joint 20.
                T_world_wrist[4:7] - wrist_position
            )

            # Estimate wrist orientation from forward + normal directions.
            palm_forward = palm_position - wrist_position
            palm_forward = palm_forward / jnp.linalg.norm(palm_forward)
            palm_normal = palm_normal / jnp.linalg.norm(palm_normal)
            palm_forward = (  # Flip palm forward if right hand.
                palm_forward * jnp.array([1, -1])[left0_right1]
            )
            palm_forward = (  # Gram-schmidt for forward direction.
                palm_forward - jnp.dot(palm_forward, palm_normal) * palm_normal
            )
            estimatedR_world_wrist = jaxlie.SO3.from_matrix(
                jnp.stack(
                    [
                        palm_forward,
                        -palm_normal,
                        jnp.cross(palm_normal, palm_forward),
                    ],
                    axis=1,
                )
            )
            R_world_wrist = jaxlie.SO3(T_world_wrist[:4])
            ori_cost = (estimatedR_world_wrist.inverse() @ R_world_wrist).log()

            return confidence * jnp.concatenate(
                [
                    guidance_params.aria_wrist_pos_weight * pose_cost,
                    guidance_params.aria_wrist_ori_weight * ori_cost,
                ]
            )

    # Per-frame regularization cost.
    @cost_with_args(
        _SmplhBodyPosesVar(jnp.arange(timesteps)),
    )
    def reg_cost(
        vals: jaxls.VarValues,
        pose: _SmplhBodyPosesVar,
    ) -> jax.Array:
        posed = do_forward_kinematics(vals, pose)

        torso_indices = jnp.array([0, 1, 2, 5, 8])  # [0, 1, 2, 5, 8] #[2, 5, 8, 12, 13]
        return jnp.concatenate(
            [
                guidance_params.prior_quat_weight
                * (
                    jaxlie.SO3(vals[pose]).inverse()
                    @ jaxlie.SO3(init_quats[pose.id, :21, :])
                )
                .log()
                .flatten(),
                # Only include some torso joints.
                guidance_params.prior_pos_weight
                * (
                    posed.Ts_world_joint[torso_indices, 4:7]
                    - init_posed.Ts_world_joint[pose.id, torso_indices, 4:7]
                ).flatten(),
            ]
        )

    @cost_with_args(
        _SmplhBodyPosesVar(jnp.arange(timesteps - 1)),
        _SmplhBodyPosesVar(jnp.arange(1, timesteps)),
    )
    def delta_smoothness_cost(
        vals: jaxls.VarValues,
        current: _SmplhBodyPosesVar,
        next: _SmplhBodyPosesVar,
    ) -> jax.Array:
        curdelt = jaxlie.SO3(vals[current]).inverse() @ jaxlie.SO3(
            init_quats[current.id, :21, :]
        )
        nexdelt = jaxlie.SO3(vals[next]).inverse() @ jaxlie.SO3(
            init_quats[next.id, :21, :]
        )
        return jnp.concatenate(
            [
                guidance_params.body_quat_delta_smoothness_weight
                * (curdelt.inverse() @ nexdelt).log().flatten(),
                guidance_params.body_quat_smoothness_weight
                * (jaxlie.SO3(vals[current]).inverse() @ jaxlie.SO3(vals[next]))
                .log()
                .flatten(),
            ]
        )

    @cost_with_args(
        _SmplhBodyPosesVar(jnp.arange(timesteps - 2)),
        _SmplhBodyPosesVar(jnp.arange(1, timesteps - 1)),
        _SmplhBodyPosesVar(jnp.arange(2, timesteps)),
    )
    def vel_smoothness_cost(
        vals: jaxls.VarValues,
        t0: _SmplhBodyPosesVar,
        t1: _SmplhBodyPosesVar,
        t2: _SmplhBodyPosesVar,
    ) -> jax.Array:
        curdelt = jaxlie.SO3(vals[t0]).inverse() @ jaxlie.SO3(vals[t1])
        nexdelt = jaxlie.SO3(vals[t1]).inverse() @ jaxlie.SO3(vals[t2])
        return (
            guidance_params.body_quat_vel_smoothness_weight
            * (curdelt.inverse() @ nexdelt).log().flatten()
        )

    @cost_with_args(
        _SmplhBodyPosesVar(jnp.arange(timesteps - 1)),
        _SmplhBodyPosesVar(jnp.arange(1, timesteps)),
        pairwise_contacts,
    )
    def skating_cost(
        vals: jaxls.VarValues,
        current: _SmplhBodyPosesVar,
        next: _SmplhBodyPosesVar,
        foot_contacts: jax.Array,
    ) -> jax.Array:
        # Do forward kinematics.
        posed_current = do_forward_kinematics(vals, current)
        posed_next = do_forward_kinematics(vals, next)
        footpos_current = posed_current.Ts_world_joint[foot_joint_indices, 4:7]
        footpos_next = posed_next.Ts_world_joint[foot_joint_indices, 4:7]
        assert footpos_current.shape == footpos_next.shape == (num_foot_joints, 3)
        assert foot_contacts.shape == (num_foot_joints,)

        return (
            guidance_params.skate_weight
            * (foot_contacts[:, None] * (footpos_current - footpos_next)).flatten()
        )

    vars_body_pose = _SmplhBodyPosesVar(jnp.arange(timesteps))
    vars_hand_pose = _SmplhSingleHandPosesVar(jnp.arange(timesteps * 2))
    graph = jaxls.LeastSquaresProblem(
        costs=factors, variables=[vars_body_pose, vars_hand_pose]
    ).analyze()
    solutions = graph.solve(
        initial_vals=jaxls.VarValues.make(
            [
                vars_body_pose.with_value(init_quats[:, :21, :]),
                vars_hand_pose.with_value(
                    init_quats[:, 21:51, :].reshape((timesteps * 2, 15, 4))
                ),
            ]
        ),
        linear_solver="conjugate_gradient",
        trust_region=jaxls.TrustRegionConfig(
            lambda_initial=guidance_params.lambda_initial
        ),
        termination=jaxls.TerminationConfig(max_iterations=guidance_params.max_iters),
        verbose=verbose,
    )
    out_body_quats = solutions[_SmplhBodyPosesVar]
    assert out_body_quats.shape == (timesteps, 21, 4)
    out_hand_quats = solutions[_SmplhSingleHandPosesVar].reshape((timesteps, 30, 4))
    assert out_hand_quats.shape == (timesteps, 30, 4)
    return (
        jnp.concatenate([out_body_quats, out_hand_quats], axis=-2),
        {},  # Metadata dict that we use for debugging.
    )


# def _get_mano_from_openpose_indices(include_tips: bool) -> Int[onp.ndarray, "21"]:
#     # https://github.com/geopavlakos/hamer/blob/272d68f176e0ea8a506f761663dd3dca4a03ced0/hamer/models/mano_wrapper.py#L20
#     # fmt: off
#     mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
#     # fmt: on
#     openpose_from_mano_idx = {
#         mano_idx: openpose_idx for openpose_idx, mano_idx in enumerate(mano_to_openpose)
#     }
#     return onp.array(
#         [openpose_from_mano_idx[i] for i in range(21 if include_tips else 16)]
    # )




def _get_mano_from_openpose_indices(include_tips: bool) -> Int[onp.ndarray, "21"]:
    # https://github.com/geopavlakos/hamer/blob/272d68f176e0ea8a506f761663dd3dca4a03ced0/hamer/models/mano_wrapper.py#L20
    # fmt: off
    # mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
    mano_to_aria = [5, 8, 9, 10, 11, 12, 13, 17, 18, 19, 14, 15, 16, 6, 7, 0] #thumb_proximal + wrist
    # fmt: on
    aria_from_mano_idx = {
        mano_idx: openpose_idx for openpose_idx, mano_idx in enumerate(mano_to_aria)
    }
    # return np.array(
    return onp.array(
        [aria_from_mano_idx[i] for i in range(21 if include_tips else 16)]
    )
