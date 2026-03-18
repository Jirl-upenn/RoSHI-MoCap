"""
IMU ID mapping & shared body-model constants for RoSHI_Calibration.

This module is the single source of truth for:
  - RoSHI imu_id <-> joint name mapping
  - RoSHI <-> optimization imu_id mapping
  - AprilTag ID <-> joint name mapping
  - SMPLX joint indices for the 9 instrumented joints

RoSHI_Calibration (receiver / imu_pose_viewer):
  imu_id 1=pelvis, 2=left-shoulder, 3=right-shoulder, 4=left-elbow, 5=right-elbow,
  6=left-hip, 7=right-hip, 8=left-knee, 9=right-knee

Optimization (visualize_utc_mapped imu_to_joint_mapping):
  imu_id 1=pelvis, 2=left_knee, 3=right_shoulder, 4=left_shoulder, 5=left_hip,
  6=right_elbow, 7=right_hip, 8=right_knee, 9=left_elbow

Body-part alignment (same physical sensor -> same joint):
  pelvis       RoSHI 1 -> optimization 1
  left-shoulder RoSHI 2 -> optimization 4  (left_shoulder)
  right-shoulder RoSHI 3 -> optimization 3
  left-elbow   RoSHI 4 -> optimization 9   (left_elbow)
  right-elbow  RoSHI 5 -> optimization 6   (right_elbow)
  left-hip     RoSHI 6 -> optimization 5
  right-hip    RoSHI 7 -> optimization 7
  left-knee    RoSHI 8 -> optimization 2   (left_knee)
  right-knee   RoSHI 9 -> optimization 8
"""

# RoSHI_Calibration imu_id -> optimization imu_id (for writing imu_info.csv in optimization format)
ROSHI_TO_OPTIMIZATION_IMU_ID = {
    1: 1,   # pelvis
    2: 4,   # left-shoulder -> optimization left_shoulder
    3: 3,   # right-shoulder
    4: 9,   # left-elbow -> optimization left_elbow
    5: 6,   # right-elbow -> optimization right_elbow
    6: 5,   # left-hip -> optimization left_hip
    7: 7,   # right-hip
    8: 2,   # left-knee -> optimization left_knee
    9: 8,   # right-knee -> optimization right_knee
}

# optimization imu_id -> RoSHI_Calibration imu_id (for reading optimization-format CSV back into RoSHI)
OPTIMIZATION_TO_ROSHI_IMU_ID = {v: k for k, v in ROSHI_TO_OPTIMIZATION_IMU_ID.items()}

# optimization imu_id -> SMPL body joint name (for RoSHI visualize_utc_mapped when reading optimization-format imu_info.csv)
OPTIMIZATION_IMU_ID_TO_ROSHI_JOINT = {
    1: "pelvis",
    2: "left-knee",       # optimization 2 = left_knee
    3: "right-shoulder",
    4: "left-shoulder",  # optimization 4 = left_shoulder
    5: "left-hip",
    6: "right-elbow",    # optimization 6 = right_elbow
    7: "right-hip",
    8: "right-knee",     # optimization 8 = right_knee
    9: "left-elbow",     # optimization 9 = left_elbow
}

# RoSHI_Calibration imu_id -> SMPL body joint name
ROSHI_IMU_ID_TO_JOINT = {
    1: "pelvis",
    2: "left-shoulder",
    3: "right-shoulder",
    4: "left-elbow",
    5: "right-elbow",
    6: "left-hip",
    7: "right-hip",
    8: "left-knee",
    9: "right-knee",
}

# Convenience alias used by receiver_calibrate.py, imu_pose_viewer.py, etc.
IMU_ID_TO_JOINT = ROSHI_IMU_ID_TO_JOINT

# Ordered list of joint names (matches dict insertion order for imu_id 1..9)
JOINT_NAMES = list(IMU_ID_TO_JOINT.values())

# ---------------------------------------------------------------------------
# AprilTag ID <-> joint name
# ---------------------------------------------------------------------------
JOINT_TO_TAG_ID = {
    "pelvis": 0,
    "left-shoulder": 1,
    "right-shoulder": 2,
    "left-elbow": 3,
    "right-elbow": 4,
    "left-hip": 5,
    "right-hip": 6,
    "left-knee": 7,
    "right-knee": 8,
}
TAG_ID_TO_JOINT = {v: k for k, v in JOINT_TO_TAG_ID.items()}

# Aliases used by imu_calibration.py
TAG_TO_JOINT_MAP = TAG_ID_TO_JOINT
JOINT_TO_TAG_MAP = JOINT_TO_TAG_ID

# Sorted list of all required AprilTag IDs (for calibration pipeline)
REQUIRED_TAG_IDS = sorted(JOINT_TO_TAG_ID.values())

# ---------------------------------------------------------------------------
# SMPLX joint indices for the 9 instrumented joints
# ---------------------------------------------------------------------------
SMPLX_JOINT_INDEX_MAP = {
    "pelvis": 0,
    "left-hip": 1,
    "right-hip": 2,
    "left-knee": 4,
    "right-knee": 5,
    "left-shoulder": 16,
    "right-shoulder": 17,
    "left-elbow": 18,
    "right-elbow": 19,
}