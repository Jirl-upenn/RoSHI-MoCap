"""
IMU utilities 
"""

# IMU_ID: [joint_id, joint_name, tensor_index]
# IMU-to-joint mapping.
IMU_CONFIG = {
    1: [0, "pelvis", 0],
    2: [4, "L_knee", 1], 
    3: [17, "R_shoulder", 2],
    4: [16, "L_shoulder", 3],
    5: [1, "L_hip", 4],
    6: [19, "R_elbow", 5],
    7: [2, "R_hip", 6],
    8: [5, "R_knee", 7],
    9: [18, "L_elbow", 8]
}
# How to read the config:
# joint_id = IMU_CONFIG[imu_id][0]
# joint_name = IMU_CONFIG[imu_id][1]
# tensor_index = IMU_CONFIG[imu_id][2]


# closest timestamp: 10ms
CLOSEST_TIMESTAMP_DIFF_IN_NS: int = 10_000_000  # 10ms in nanoseconds

JOINT_ID_TO_NAME_AND_PARENT = {
    0: ["pelvis", {"parent":None, "parent_id":None}],
    1: ["L_hip", {"parent":"pelvis", "parent_id":0}],
    2: ["R_hip", {"parent":"pelvis", "parent_id":0}],
    4: ["L_knee", {"parent":"L_hip", "parent_id":1}],
    5: ["R_knee", {"parent":"R_hip", "parent_id":2}],
    16: ["L_shoulder", {"parent":"L_collar", "parent_id":13}],
    17: ["R_shoulder", {"parent":"R_collar", "parent_id":14}],
    18: ["L_elbow", {"parent":"L_shoulder", "parent_id":16}],
    19: ["R_elbow", {"parent":"R_shoulder", "parent_id":17}],
}