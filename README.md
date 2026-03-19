# RoSHI: Robust Sparse-sensor Human-body IMU Pose Estimation

A complete pipeline for IMU-based human body pose estimation using 9 body-worn IMU sensors with AprilTag markers. RoSHI calibrates sensor-to-bone rotation offsets, reconstructs full-body pose from IMU data alone, and evaluates against OptiTrack motion capture ground truth.

## Overview

We wear **9 IMUs** (pelvis, shoulders, elbows, hips, knees), each with an **AprilTag** rigidly attached. The system:

1. **Calibrates** bone-to-sensor rotation offsets (${}^{B}R_{S}$) using single-camera AprilTag detection and SMPL-X body estimation
2. **Reconstructs** full-body pose from IMU quaternions alone, with world-frame alignment via AprilTag observations
3. **Synchronizes** third-person RGB, egocentric Aria video, and calibrated IMU into a unified UTC timeline
4. **Evaluates** against OptiTrack ground truth using MPJPE, JAE, and per-activity metrics

## Repository Structure

```
RoSHI/
├── src/
│   ├── pipeline/                   # Core pipeline scripts
│   │   ├── 01_receiver.py          #   Receive recordings from iOS app + record IMU data
│   │   ├── 02_imu_calibration.py   #   Compute bone-to-sensor rotation offsets
│   │   ├── 03_imu_pose_viewer.py   #   IMU-only pose reconstruction and visualization
│   │   ├── 04_sync_pipeline.py     #   UTC-aligned RGB + calibrated IMU synchronization
│   │   └── 05_mpjae_evaluation.py  #   Mean Per-Joint Angle Error evaluation
│   └── utils/                      # Core utility modules
│       ├── imu_id_mapping.py       #   IMU/tag/joint/SMPLX ID mappings (single source of truth)
│       ├── session_preparation.py  #   Frame extraction, camera.json, AprilTag summary
│       ├── sync_utils.py           #   Sync pipeline helpers (CSV loaders, trajectory mapping)
│       ├── apriltag_utils.py       #   AprilTag detection and rotation utilities
│       └── smpl_utils.py           #   SMPL-X model loading and forward kinematics
│
├── evaluation/                 # Evaluation scripts and results (see evaluation/README.md)
│   ├── compute_metrics.py      #   MPJPE, JAE, Recall computation
│   ├── compute_metrics_by_activity.py
│   ├── fit_smplx.py            #   Fit SMPL-X to OptiTrack ground truth
│   ├── eval_utils.py           #   Shared evaluation utilities
│   ├── sequence_splits.json    #   Activity annotations per dataset
│   └── visualize_*.py          #   Per-method visualization scripts
│
├── hardware/                   # Hardware interface examples
│   └── imu_reader.py           #   ESP32 serial IMU reader/recorder/visualizer
│
├── sam-3d-body/                # SAM-3D-Body (Meta AI) — modified (see MODIFICATIONS.md)
├── MHR/                        # Momentum Human Rig (Meta AI) — modified (see MODIFICATIONS.md)
├── egoallo/                    # EgoAllo (Meta AI) — modified (see MODIFICATIONS.md)
│
├── model/                      # SMPL/SMPL-H/SMPL-X body models (not included, see below)
├── received_recordings/        # Processed recording sessions (not included)
├── aria_recordings/            # Aria egocentric VRS recordings (not included)
└── outputdata/                 # Optimization outputs (not included)
```

## Pipeline

### Step 1: Data Collection (`01_receiver.py`)

Receives video + metadata from the [RoSHI iOS App](https://github.com/Jirl-upenn/RoSHI-App) and records IMU data from the ESP32 serial receiver.

```bash
python src/pipeline/01_receiver.py --output-dir received_recordings
```

The iOS app captures video with camera intrinsics and AprilTag detections embedded in `metadata.json`.

### Step 2: Calibration (`02_imu_calibration.py`)

Computes the bone-to-sensor rotation offset ${}^{B}R_{S}$ for each of the 9 joints. First, run pose estimation on the recorded frames:

```bash
# SAM-3D-Body inference (with subject filtering for single-person recordings)
python sam-3d-body/demo.py \
  --image_folder <session>/color \
  --checkpoint_path sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt \
  --output_folder <session>/body_vis \
  --data_folder <session>/body_data \
  --mhr_path MHR/assets/mhr_model.pt \
  --camera_json <session>/meta/camera.json \
  --subject_only

# Convert MHR output to SMPL-X
python MHR/tools/mhr_smpl_conversion/convert_mhr_to_smpl.py \
  --input <session>/body_data \
  --output <session>/smpl_output \
  --smplx model/smplx/SMPLX_NEUTRAL.npz

# Compute bone-to-sensor offsets
python src/pipeline/02_imu_calibration.py <session> \
  --smpl-model-path model/smplx/SMPLX_NEUTRAL.npz \
  --output <session>/imu_calibration.json
```

The calibration uses geodesic optimization on SO(3) (Karcher mean by default). For noisy data with outliers, use `--method huber` or `--method ransac`.

### Step 3: IMU-Only Pose Viewer (`03_imu_pose_viewer.py`)

Applies calibrated offsets to recorded IMU quaternions and visualizes the reconstructed pose with optional ground truth overlay:

```bash
python src/pipeline/03_imu_pose_viewer.py <session> --port 8082
```

### Step 4: Synchronization (`04_sync_pipeline.py`)

Builds a sync folder with UTC-aligned third-person RGB and calibrated IMU rotations:

```bash
python src/pipeline/04_sync_pipeline.py <session>
```

Output goes to `<session>/sync/`. If an Aria `.vrs` file and MPS trajectory are present, first-person VRS frames are also extracted and aligned.

### Step 5: Evaluation (`05_mpjae_evaluation.py`)

See [evaluation/README.md](evaluation/README.md) for detailed evaluation instructions. Quick start:

```bash
# Fit SMPL-X to OptiTrack ground truth
python evaluation/fit_smplx.py

# Compute metrics across all methods
python evaluation/compute_metrics.py

# Per-activity breakdown
python evaluation/compute_metrics_by_activity.py
```

## Session Data Layout

After processing, each recording session contains:

```
<session>/
├── video.mp4                         # Raw recording from iOS app
├── metadata.json                     # Camera intrinsics + AprilTag detections
├── color/                            # Extracted video frames
├── frames.csv                        # frame_id, utc_timestamp_ns, color_path
├── meta/
│   ├── camera.json                   # Camera intrinsics for SAM-3D-Body
│   └── calibration_segment.json      # Calibration window (from AprilTag analysis)
├── color_apriltag/
│   └── detection_summary.json        # Per-frame tag rotations
├── imu/
│   └── imu_data.csv                  # Raw IMU packets (timestamp, quaternion, accel, gyro)
├── body_data/                        # SAM-3D-Body / MHR output
├── smpl_output/per_frame/            # SMPL-X rotations per frame
├── imu_calibration.json              # Calibrated bone-to-sensor offsets
└── sync/                             # Synchronized outputs
    ├── frames.csv                    # UTC-aligned third-person frames
    ├── color/                        # Symlinks to session color/
    ├── imu_info.csv                  # Calibrated IMU rotations (utc_ns, imu_id, rot_matrix)
    ├── imu_info.pkl                  # Same as pickle dict
    ├── vrs_frames.csv                # Aria first-person frames (if available)
    └── vrs_color/                    # Extracted VRS RGB (if available)
```

## IMU / AprilTag Mapping

| IMU ID | Body Part       | AprilTag ID | SMPLX Joint |
|--------|-----------------|-------------|-------------|
| 1      | pelvis          | 0           | 0           |
| 2      | left-shoulder   | 1           | 16          |
| 3      | right-shoulder  | 2           | 17          |
| 4      | left-elbow      | 3           | 18          |
| 5      | right-elbow     | 4           | 19          |
| 6      | left-hip        | 5           | 1           |
| 7      | right-hip       | 6           | 2           |
| 8      | left-knee       | 7           | 4           |
| 9      | right-knee      | 8           | 5           |

## Calibration Math

### Coordinate Frames

| Frame | Description |
|-------|-------------|
| $C$ | Camera frame (OpenCV: $x$ right, $y$ down, $z$ forward) |
| $B$ | Bone frame (SMPL-X joint) |
| $T$ | AprilTag frame (rigidly attached to IMU) |
| $S$ | IMU sensor frame (fused quaternion output) |
| $W_i$ | Per-IMU world frame (gravity aligned, heading arbitrary) |
| $W_p$ | Shared world frame (pelvis IMU as reference) |

### Offset Calibration

The AprilTag is rigidly attached to the bone segment. At any time $t$:

$${}^{C}R_{T}(t) = {}^{C}R_{B}(t) \cdot {}^{B}R_{T}$$

Rearranging to solve for the constant offset:

$${}^{B}R_{T}(t) = \left({}^{C}R_{B}(t)\right)^{\top} \cdot {}^{C}R_{T}(t)$$

The final calibrated offset is the geodesic mean over all frame estimates:

$$\widehat{{}^{B}R_{T}} = \arg\min_{R \in SO(3)} \sum_{t=1}^{N} \rho\left(d_g\left(R, {}^{B}R_{T}(t)\right)\right)$$

where $d_g(R_1, R_2) = \|\log(R_1^\top R_2)\|$ is the geodesic distance on SO(3).

### Supported Optimization Methods

| Method | Loss | Use Case |
|--------|------|----------|
| `karcher` (default) | Geodesic L2 | Clean data, Gaussian noise |
| `huber` | L2 near origin, L1 for outliers | Moderate outliers |
| `cauchy` | Cauchy/Lorentzian | Heavy outliers |
| `l1` | Geodesic L1 | Median-like robustness |
| `ransac` | RANSAC + Karcher | Severely corrupted data |

### IMU-Only Pose Reconstruction

Given calibrated offset ${}^{B}R_{S}$ and live IMU reading ${}^{W_i}R_{S}(t)$:

$${}^{W_i}R_{B}(t) = {}^{W_i}R_{S}(t) \cdot \left({}^{B}R_{S}\right)^{\top}$$

World frame alignment uses AprilTag observations to estimate ${}^{W_p}R_{W_i}$ (per-IMU world to pelvis world), resolving heading/yaw mismatches across sensors.

### Tag-to-IMU Axis Mapping

$${}^{T}R_{S} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 0 & 0 \\ 0 & 0 & -1 \end{bmatrix}$$

## External Dependencies (Modified)

These are third-party codebases included with RoSHI-specific modifications. See each directory's `MODIFICATIONS.md` for details.

| Directory | Original | Our Modifications |
|-----------|----------|-------------------|
| `sam-3d-body/` | [Meta AI SAM-3D-Body](https://github.com/facebookresearch/sam-3d-body) | Subject filtering for single-person recordings, empty detection handling, camera intrinsics passthrough |
| `MHR/` | [Meta AI MHR](https://github.com/facebookresearch/MomentumHumanRig) | Import order fix (pymomentum/numpy segfault), multi-detection handling |
| `egoallo/` | [Meta AI EgoAllo](https://egoallo.github.io/) | IMU-guided pose estimation: added IMU data structures, whole-body guidance optimizer (JAX), IMU-aware sampling |

## Related Repositories

- [RoSHI-App](https://github.com/Jirl-upenn/RoSHI-App) — iOS companion app for video recording with AprilTag detection

## Model Files

SMPL body models are licensed and must be downloaded separately. Place in `model/`:

```
model/
├── smplh/
│   └── neutral/model.npz          # From https://mano.is.tue.mpg.de/
└── smplx/
    └── SMPLX_NEUTRAL.npz          # From https://smpl-x.is.tue.mpg.de/
```

## Dependencies

Core:
- Python 3.10+
- numpy, scipy, torch, smplx
- opencv-python, viser, pillow
- pyserial (for IMU hardware interface)

EgoAllo additionally requires:
- Python 3.12+, JAX with CUDA, jaxls
- See `egoallo/README_RoSHI_egoallo.md` for full setup

SAM-3D-Body and MHR have their own dependencies; see their respective `README.md` files.
