"""
SMPL / SMPLX utilities.

Contains both the lightweight SmplHelper (class-based API) and the
SmplxModel dataclass + forward-kinematics functions used by the IMU pose viewer.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class SmplOutputs:
    """Output data from SMPL model computation."""
    vertices: np.ndarray
    faces: np.ndarray
    T_world_joint: np.ndarray
    T_parent_joint: np.ndarray


class SmplHelper:
    """Helper class for SMPL model operations (loads .npz)."""

    def __init__(self, model_path: Path) -> None:
        model_path = Path(model_path)
        assert model_path.suffix.lower() == ".npz", "Model should be an .npz file!"
        body_dict = dict(**np.load(model_path, allow_pickle=True))
        self.J_regressor = body_dict["J_regressor"]
        self.weights = body_dict["weights"]
        self.v_template = body_dict["v_template"]
        self.posedirs = body_dict["posedirs"]
        self.shapedirs = body_dict["shapedirs"]
        self.faces = body_dict["f"]
        self.num_joints: int = self.weights.shape[-1]
        self.num_betas: int = self.shapedirs.shape[-1]
        self.parent_idx: np.ndarray = body_dict["kintree_table"][0]

    def get_outputs(self, betas: np.ndarray, joint_rotmats: np.ndarray) -> SmplOutputs:
        v_tpose = self.v_template + np.einsum("vxb,b->vx", self.shapedirs, betas)
        j_tpose = np.einsum("jv,vx->jx", self.J_regressor, v_tpose)
        T_parent_joint = np.zeros((self.num_joints, 4, 4)) + np.eye(4)
        T_parent_joint[:, :3, :3] = joint_rotmats
        T_parent_joint[0, :3, 3] = j_tpose[0]
        T_parent_joint[1:, :3, 3] = j_tpose[1:] - j_tpose[self.parent_idx[1:]]
        T_world_joint = T_parent_joint.copy()
        for i in range(1, self.num_joints):
            T_world_joint[i] = T_world_joint[self.parent_idx[i]] @ T_parent_joint[i]
        pose_delta = (joint_rotmats[1:, ...] - np.eye(3)).flatten()
        v_blend = v_tpose + np.einsum("byn,n->by", self.posedirs, pose_delta)
        v_delta = np.ones((v_blend.shape[0], self.num_joints, 4))
        v_delta[:, :, :3] = v_blend[:, None, :] - j_tpose[None, :, :]
        v_posed = np.einsum("jxy,vj,vjy->vx", T_world_joint[:, :3, :], self.weights, v_delta)
        return SmplOutputs(v_posed, self.faces, T_world_joint, T_parent_joint)


# ---------------------------------------------------------------------------
# SmplxModel: dataclass-based API used by IMU pose viewer & sync pipeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SmplxModel:
    faces: np.ndarray  # (F,3)
    v_template: np.ndarray  # (V,3)
    shapedirs: np.ndarray  # (V,3,B)
    posedirs: np.ndarray  # (V,3,(J-1)*9)
    J_regressor: np.ndarray  # (J,V)
    weights: np.ndarray  # (V,J)
    parents: np.ndarray  # (J,)

    @property
    def num_joints(self) -> int:
        return int(self.weights.shape[1])


def load_smplx_model(model_path: Path, betas_dim: int = 10) -> SmplxModel:
    m = np.load(model_path, allow_pickle=True)

    parents = m["kintree_table"][0].astype(np.int64)
    parents[parents > 1_000_000_000] = -1

    shapedirs = m["shapedirs"]
    if shapedirs.dtype != np.float32:
        shapedirs = shapedirs.astype(np.float32)
    if shapedirs.shape[2] > betas_dim:
        shapedirs = shapedirs[:, :, :betas_dim]

    return SmplxModel(
        faces=m["f"].astype(np.int32),
        v_template=m["v_template"].astype(np.float32),
        shapedirs=shapedirs,
        posedirs=m["posedirs"].astype(np.float32),
        J_regressor=m["J_regressor"].astype(np.float32),
        weights=m["weights"].astype(np.float32),
        parents=parents,
    )


def precompute_shape(
    model: SmplxModel, betas: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute shape-dependent arrays (v_shaped, j_tpose) that only depend on betas."""
    betas = betas.astype(np.float32).reshape(-1)
    v_shaped = model.v_template + np.einsum("vxb,b->vx", model.shapedirs, betas)
    j_tpose = model.J_regressor @ v_shaped  # (J,3)
    return v_shaped, j_tpose


def smplx_forward_kinematics(
    model: SmplxModel,
    local_rots: np.ndarray,  # (J,3,3)
    betas: np.ndarray,  # (B,)
    *,
    compute_vertices: bool = True,
    v_shaped: Optional[np.ndarray] = None,
    j_tpose: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Minimal SMPLX LBS (numpy).

    Returns:
      joints_world: (J,3)
      T_world_joint: (J,4,4)
      vertices_world: (V,3) or None
    """
    J = model.num_joints
    local_rots = local_rots.astype(np.float32)

    if v_shaped is None or j_tpose is None:
        betas = betas.astype(np.float32).reshape(-1)
        v_shaped = model.v_template + np.einsum("vxb,b->vx", model.shapedirs, betas)
        j_tpose = model.J_regressor @ v_shaped  # (J,3)

    T_parent = np.zeros((J, 4, 4), dtype=np.float32)
    T_parent[:] = np.eye(4, dtype=np.float32)
    T_parent[:, :3, :3] = local_rots
    T_parent[0, :3, 3] = j_tpose[0]
    for i in range(1, J):
        p = int(model.parents[i])
        if p < 0:
            T_parent[i, :3, 3] = j_tpose[i]
        else:
            T_parent[i, :3, 3] = j_tpose[i] - j_tpose[p]

    T_world = np.zeros_like(T_parent)
    for i in range(J):
        p = int(model.parents[i])
        if p < 0:
            T_world[i] = T_parent[i]
        else:
            T_world[i] = T_world[p] @ T_parent[i]

    joints_world = T_world[:, :3, 3].copy()

    if not compute_vertices:
        return joints_world, T_world, None

    pose_delta = (local_rots[1:] - np.eye(3, dtype=np.float32)).reshape(-1)
    v_posed = v_shaped + np.tensordot(model.posedirs, pose_delta, axes=([2], [0]))

    v_delta = np.ones((v_posed.shape[0], J, 4), dtype=np.float32)
    v_delta[:, :, :3] = v_posed[:, None, :] - j_tpose[None, :, :]
    vertices = np.einsum("jxy,vj,vjy->vx", T_world[:, :3, :], model.weights, v_delta)

    return joints_world, T_world, vertices


def build_local_rots_from_imu(
    imu_global_rots: Dict[str, np.ndarray],
    *,
    pelvis_name: str = "pelvis",
) -> np.ndarray:
    """
    Build SMPLX local rotations (55 joints) from a sparse set of global joint rotations.

    Assumes the torso chain is rigid with pelvis (all intermediate joints local=I),
    so their global rotation equals pelvis global.
    """
    J = 55
    local = np.tile(np.eye(3, dtype=np.float64), (J, 1, 1))

    def G(name: str) -> np.ndarray:
        return imu_global_rots.get(name, np.eye(3, dtype=np.float64))

    pelvis_G = G(pelvis_name)
    local[0] = pelvis_G

    local[1] = pelvis_G.T @ G("left-hip")
    local[2] = pelvis_G.T @ G("right-hip")
    local[4] = G("left-hip").T @ G("left-knee")
    local[5] = G("right-hip").T @ G("right-knee")

    local[16] = pelvis_G.T @ G("left-shoulder")
    local[17] = pelvis_G.T @ G("right-shoulder")
    local[18] = G("left-shoulder").T @ G("left-elbow")
    local[19] = G("right-shoulder").T @ G("right-elbow")

    return local.astype(np.float32)
