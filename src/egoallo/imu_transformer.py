"""IMU Transformer: direct (non-diffusion) body-pose regression from Aria CPF
poses and nine calibrated RoSHI bone orientations.

Inference-only port of the model trained in the egoallo training repository.
Parameter names match the training code, so its checkpoints load unchanged.
Supported checkpoints: architecture="temporal", input_encoding="separate",
separate_imu_frame="pelvis", bone_orientation_frame="pelvis", predicted betas.

Per frame t the network sees
  * head features (22): T_cpf(t-1)^-1 T_cpf(t) as a 3x4 matrix, CPF height above
    the floor, and the CPF rotation with its world-yaw removed;
  * per sensor (18):   R_world_pelvis(t)^T R_world_bone(t)   (pelvis-relative)
                       R_world_bone(t-1)^T R_world_bone(t)   (temporal delta).
Both IMU terms are invariant to a constant rotation of the shared IMU world
frame, so the IMUs need a common world frame but not the Aria world frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .network import EgoDenoiseTraj
from .transforms import SE3, SO3

# Model input order of the nine sensors (SMPL joint index in brackets):
# pelvis(0) L_hip(1) R_hip(2) L_knee(4) R_knee(5) L_shoulder(16) R_shoulder(17)
# L_elbow(18) R_elbow(19).
TRANSFORMER_BONE_NAMES = ("pelvis", "L_hip", "R_hip", "L_knee", "R_knee",
                          "L_shoulder", "R_shoulder", "L_elbow", "R_elbow")
TRANSFORMER_SMPL_JOINT_INDICES = (0, 1, 2, 4, 5, 16, 17, 18, 19)
# imu_info.pkl / guidance use RoSHI IMU ids 1..9 (imu_utils.IMU_CONFIG order:
# pelvis, L_knee, R_shoulder, L_shoulder, L_hip, R_elbow, R_hip, R_knee, L_elbow).
# IMU_ID_FOR_TRANSFORMER_SENSOR[k] = RoSHI IMU id feeding transformer sensor k.
IMU_ID_FOR_TRANSFORMER_SENSOR = (1, 5, 7, 2, 8, 4, 3, 9, 6)


@dataclass(frozen=True)
class IMUTransformerConfig:
    architecture: Literal["temporal", "mixste"] = "temporal"
    d_latent: int = 256
    num_heads: int = 8
    num_layers: int = 6
    d_feedforward: int = 1024
    dropout: float = 0.1
    include_hands: bool = False
    predict_contacts: bool = True
    known_betas: bool = False
    input_encoding: Literal["conditioning", "separate"] = "separate"
    separate_imu_frame: Literal["pelvis", "world"] = "pelvis"
    nonlinear_encoders: bool = False
    d_head_encoding: int = 64
    d_imu_encoding: int = 16
    d_beta_encoding: int = 48
    bone_orientation_frame: Literal["pelvis", "pelvis_only", "cpf_heading"] = "pelvis"

    def __post_init__(self):
        unsupported = dict(architecture="temporal", input_encoding="separate",
                           separate_imu_frame="pelvis", bone_orientation_frame="pelvis",
                           known_betas=False)
        bad = {k: getattr(self, k) for k, v in unsupported.items() if getattr(self, k) != v}
        if bad:
            raise ValueError(f"Checkpoint configuration not supported by this port: {bad}")
        if self.d_latent % self.num_heads:
            raise ValueError("d_latent must be divisible by num_heads")


D_HEAD_FEATURES = 12 + 1 + 9


def head_features(T_cpf_tm1_cpf_t: Tensor, T_world_cpf: Tensor) -> Tensor:
    """(B,T,7),(B,T,7) -> (B,T,22). T_world_cpf z must be relative to the floor."""
    batch, time, _ = T_world_cpf.shape
    R_world_cpf = SE3(T_world_cpf).rotation().wxyz
    forward = SO3(R_world_cpf) @ R_world_cpf.new_tensor([0.0, 0.0, 1.0])
    R_canonical_world = SO3.from_z_radians(-torch.arctan2(forward[..., 1], forward[..., 0]))
    return torch.cat([
        SE3(T_cpf_tm1_cpf_t).as_matrix()[..., :3, :].reshape(batch, time, 12),
        T_world_cpf[..., 6:7],
        (R_canonical_world @ SO3(R_world_cpf)).as_matrix().reshape(batch, time, 9),
    ], dim=-1)


def rotation_6d_to_matrix(x: Tensor) -> Tensor:
    """Gram-Schmidt on two columns, with finite proper-rotation fallbacks."""
    a, b = x[..., :3], x[..., 3:]
    fallback = torch.zeros_like(a)
    fallback[..., 0] = 1
    a = torch.where(a.norm(dim=-1, keepdim=True) > 1e-6, a, fallback)
    u = F.normalize(a, dim=-1)
    v = b - (u * b).sum(-1, keepdim=True) * u
    axis = F.one_hot(u.abs().argmin(-1), 3).to(x.dtype)
    fallback = axis - (axis * u).sum(-1, keepdim=True) * u
    v = F.normalize(torch.where(v.norm(dim=-1, keepdim=True) > 1e-6, v, fallback), dim=-1)
    return torch.stack((u, v, torch.cross(u, v, dim=-1)), dim=-1)


class IMUTransformer(nn.Module):
    def __init__(self, config: IMUTransformerConfig):
        super().__init__()
        self.config = config
        c = config
        if c.nonlinear_encoders:
            self.head_encoder = nn.Sequential(nn.Linear(D_HEAD_FEATURES, c.d_head_encoding), nn.GELU(),
                                              nn.Linear(c.d_head_encoding, c.d_head_encoding))
            self.imu_encoder = nn.Sequential(nn.Linear(18, 32), nn.GELU(), nn.Linear(32, c.d_imu_encoding))
            self.input_projection = nn.Sequential(nn.Linear(c.d_head_encoding + 9 * c.d_imu_encoding, c.d_latent), nn.GELU())
        else:
            self.head_encoder = nn.Linear(D_HEAD_FEATURES, c.d_head_encoding)
            # One shared encoder applied independently to each of the nine IMUs.
            self.imu_encoder = nn.Linear(18, c.d_imu_encoding)
            self.input_projection = nn.Linear(c.d_head_encoding + 9 * c.d_imu_encoding, c.d_latent)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(c.d_latent, c.num_heads, c.d_feedforward, c.dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(c.num_layers)
        ])
        self.norm = nn.LayerNorm(c.d_latent)
        self.num_joints = 51 if c.include_hands else 21
        self.rotation_head = nn.Linear(c.d_latent, self.num_joints * 6)
        self.shape_head = nn.Linear(c.d_latent, 16)
        self.contact_head = nn.Linear(c.d_latent, 21) if c.predict_contacts else None

    def forward(self, T_world_cpf, T_cpf_tm1_cpf_t, bone_quats_world, bone_delta_quats) -> EgoDenoiseTraj:
        """All inputs (B,T,...): CPF (floor-relative z), CPF delta, bones (B,T,9,4) wxyz in a
        common world frame, bone deltas R(t-1)^T R(t). Returns rotations for frames T."""
        batch, time = T_world_cpf.shape[:2]
        if bone_quats_world.shape != (batch, time, 9, 4) or bone_delta_quats.shape != (batch, time, 9, 4):
            raise ValueError("Expected current and delta IMU quaternions (B,T,9,4).")
        current = SO3(bone_quats_world[..., :1, :]).inverse() @ SO3(bone_quats_world)
        imu_input = torch.cat((current.as_matrix().flatten(-2),
                               SO3(bone_delta_quats).as_matrix().flatten(-2)), dim=-1)
        x = torch.cat([self.imu_encoder(imu_input).flatten(-2),
                       self.head_encoder(head_features(T_cpf_tm1_cpf_t, T_world_cpf))], dim=-1)
        x = self.input_projection(x)
        d = self.config.d_latent
        position = torch.arange(time, device=x.device, dtype=x.dtype)[:, None]
        frequency = torch.exp(torch.arange(0, d, 2, device=x.device, dtype=x.dtype) * (-math.log(10000.0) / d))
        pe = torch.zeros(time, d, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(position * frequency)
        pe[:, 1::2] = torch.cos(position * frequency[: pe[:, 1::2].shape[-1]])
        x = x + pe
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        rotations = rotation_6d_to_matrix(self.rotation_head(x).reshape(batch, time, -1, 6))
        betas = self.shape_head(x.mean(1))[:, None].expand(batch, time, 16)
        contacts = (self.contact_head(x).sigmoid() if self.contact_head is not None
                    else x.new_zeros(batch, time, 21))
        return EgoDenoiseTraj(betas=betas, body_rotmats=rotations[..., :21, :, :], contacts=contacts,
                              hand_rotmats=rotations[..., 21:, :, :] if self.config.include_hands else None)


def load_transformer(checkpoint: Path, device="cpu") -> IMUTransformer:
    """Load a training-repo checkpoint ({'model', 'model_config', ...})."""
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    known = {f.name for f in fields(IMUTransformerConfig)}
    config = IMUTransformerConfig(**{k: v for k, v in state["model_config"].items() if k in known})
    model = IMUTransformer(config)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


@torch.inference_mode()
def predict_sequence(model: IMUTransformer, T_world_cpf: Tensor, bone_quats_world: Tensor, *,
                     floor_z: float = 0.0, window: int = 128, overlap: int = 32) -> EgoDenoiseTraj:
    """T_world_cpf (N,7), bone_quats_world (N,9,4). Frame 0 is context only: the
    returned trajectory (batch 1) covers frames 1..N-1.

    Long sequences run in windows; overlaps are blended with triangular weights in
    6D-rotation space and projected back to SO(3). Deltas are computed before
    windowing, so window boundaries keep their true previous observation.
    """
    if window < 1 or not 0 <= overlap < window:
        raise ValueError("Require window>0 and 0<=overlap<window.")
    n = T_world_cpf.shape[0]
    if n < 2 or T_world_cpf.shape != (n, 7) or bone_quats_world.shape != (n, 9, 4):
        raise ValueError("Expected CPF (N,7) and bones (N,9,4), N>=2.")
    for q in (T_world_cpf[:, :4], bone_quats_world):
        if not torch.isfinite(q).all() or not torch.allclose(q.norm(dim=-1), torch.ones_like(q[..., 0]), atol=1e-3):
            raise ValueError("Input orientations must be finite unit wxyz quaternions.")
    cpf = T_world_cpf.clone()
    cpf[:, 6] -= floor_z
    head_delta = (SE3(cpf[:-1]).inverse() @ SE3(cpf[1:])).parameters()
    bone_delta = (SO3(bone_quats_world[:-1]).inverse() @ SO3(bone_quats_world[1:])).wxyz
    count = n - 1
    rotations = cpf.new_zeros(count, model.num_joints, 6)
    betas = cpf.new_zeros(count, 16)
    contacts = cpf.new_zeros(count, 21)
    weight_sum = cpf.new_zeros(count, 1)
    for start in range(0, count, window - overlap):
        end = min(start + window, count)
        result = model(cpf[None, start + 1:end + 1], head_delta[None, start:end],
                       bone_quats_world[None, start + 1:end + 1], bone_delta[None, start:end])
        mats = result.body_rotmats if result.hand_rotmats is None else torch.cat((result.body_rotmats, result.hand_rotmats), -3)
        six = torch.cat((mats[0, ..., :, 0], mats[0, ..., :, 1]), -1)
        index = torch.arange(end - start, device=cpf.device, dtype=cpf.dtype)
        weight = torch.minimum(index + 1, end - start - index).clamp_max(overlap + 1)[:, None]
        rotations[start:end] += six * weight[..., None]
        betas[start:end] += result.betas[0] * weight
        contacts[start:end] += result.contacts[0] * weight
        weight_sum[start:end] += weight
        if end == count:
            break
    mats = rotation_6d_to_matrix(rotations / weight_sum[..., None])
    shape = (betas / weight_sum).mean(0).expand(count, 16)
    hands = mats[:, 21:] if model.config.include_hands else \
        torch.eye(3, device=cpf.device, dtype=cpf.dtype).expand(count, 30, 3, 3)
    return EgoDenoiseTraj(betas=shape[None], body_rotmats=mats[None, :, :21],
                          contacts=(contacts / weight_sum)[None], hand_rotmats=hands[None])
