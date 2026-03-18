DATA_FOLEDER = "/mnt/aloque_scratch/mwenjing/egoallo_optimization/data/run_test"
BODY_24_ID2NAME = {
    0:"pelvis",
    1:"L_hip",          2:"R_hip",
    3:"spine1",
    4:"L_knee",         5:"R_knee",
    6:"spine2",
    7:"L_ankle",        8:"R_ankle",
    9:"spine3",
    10:"L_foot",        11:"R_foot",
    12:"neck",
    13:"L_collar",      14:"R_collar",
    15:"head",
    16:"L_shoulder",    17:"R_shoulder",
    18:"L_elbow",       19:"R_elbow",
    20:"L_wrist",       21:"R_wrist",
}

L_HAND_15_ID2NAME = {
    22:"L_index_proximal_joint",
    23:"L_index_intermediate_joint",
    24:"L_index_distal_joint",
    25:"L_middle_proximal_joint",
    26:"L_middle_intermediate_joint",
    27:"L_middle_distal_joint",
    28:"L_pinky_proximal_joint",
    29:"L_pinky_intermediate_joint",
    30:"L_pinky_distal_joint",
    31:"L_ring_proximal_joint",
    32:"L_ring_intermediate_joint",
    33:"L_ring_distal_joint",
    34:"L_thumb_proximal_joint",
    35:"L_thumb_intermediate_joint",
    36:"L_thumb_distal_joint",
}

R_HAND_15_ID2NAME = {
    37:"R_index_proximal_joint",
    38:"R_index_intermediate_joint",
    39:"R_index_distal_joint",
    40:"R_middle_proximal_joint",
    41:"R_middle_intermediate_joint",
    42:"R_middle_distal_joint",
    43:"R_pinky_proximal_joint",
    44:"R_pinky_intermediate_joint",
    45:"R_pinky_distal_joint",
    46:"R_ring_proximal_joint",
    47:"R_ring_intermediate_joint",
    48:"R_ring_distal_joint",
    49:"R_thumb_proximal_joint",
    50:"R_thumb_intermediate_joint",
    51:"R_thumb_distal_joint",
}

PARENT_JOINT_ID = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, \
        9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 22, \
        23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, \
        34, 35, 21, 37, 38, 21, 40, 41, 21, 43, 44, 21, \
        46, 47, 21, 49, 50]

PARENT_INDICES = [-1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, \
        8, 8, 8, 11, 12, 13, 15, 16, 17, 18, 19, 21, \
        22, 19, 24, 25, 19, 27, 28, 19, 30, 31, 19, \
        33, 34, 20, 36, 37, 20, 39, 40, 20, 42, 43, \
        20, 45, 46, 20, 48, 49]