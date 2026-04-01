# RoSHI: Robust Sparse-sensor Human-body IMU Pose Estimation

[Project Page](https://roshi-mocap.github.io/) | [Documentation](https://roshi-mocap.github.io/documentation/) | [RoSHI-App](https://github.com/Jirl-upenn/RoSHI-App)

A complete pipeline for IMU-based human body pose estimation using 9 body-worn
IMU sensors with rigid AprilTags. RoSHI calibrates sensor-to-bone offsets,
reconstructs full-body pose, synchronizes third-person and egocentric data, and
supports evaluation against OptiTrack ground truth.

## Installation

```bash
git clone git@github.com:Jirl-upenn/RoSHI-MoCap.git
cd RoSHI-MoCap
pip install -e .
pip install git+https://github.com/brentyi/jaxls.git
```

### Download the SMPL-H model

Download the "Extended SMPL+H model" (16 shape parameters) from the
[MANO project page](https://mano.is.tue.mpg.de/) and place it at:

```
model/smplh/neutral/model.npz
```

### Download the EgoAllo checkpoint

```bash
bash scripts/download_checkpoint_and_data.sh
```

Or download manually from
[Google Drive](https://drive.google.com/file/d/14bDkWixFgo3U6dgyrCRmLoXSsXkrDA2w/view?usp=drive_link).

## Core Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `src/pipeline/01_receiver.py` | Receive calibration data from iOS app |
| 2 | `src/pipeline/02_calibrate.py` | Calibrate bone-to-sensor rotation offsets |
| 3 | `src/pipeline/03_sync.py` | Synchronize RGB + calibrated IMU data |
| 4 | `src/pipeline/04_inference.py` | Run EgoAllo diffusion-based pose estimation |
| 5 | `src/pipeline/05_visualize.py` | Multi-method visualization (GT, IMU, TTO, EgoAllo) |
| 6 | `src/pipeline/06_evaluate.py` | Evaluate against OptiTrack ground truth |

## Quick Start

```bash
# Calibration & data preparation
python src/pipeline/01_receiver.py --output-dir received_recordings
python src/pipeline/02_calibrate.py <session> --smpl-model-path model/smplx/SMPLX_NEUTRAL.npz --output <session>/imu_calibration.json
python src/pipeline/03_sync.py <session>

# EgoAllo inference (requires GPU + JAX with CUDA)
python src/pipeline/04_inference.py \
  --traj-root /path/to/received_recordings/dataset1 \
  --guidance-mode imu_aria_hand

# Visualize all methods side-by-side (GT, IMU-only, TTO, EgoAllo + third-person RGB)
python src/pipeline/05_visualize.py <session>

# Only compare specific methods
python src/pipeline/05_visualize.py <session> --no-imu --no-tto          # EgoAllo vs GT
python src/pipeline/05_visualize.py <session> --no-egoallo --no-tto      # IMU-only vs GT
python src/pipeline/05_visualize.py <session> --egoallo-csv path/to/pred.csv  # custom CSV

# Evaluate against OptiTrack ground truth
python src/pipeline/06_evaluate.py <session>
```

Available guidance modes: `imu_only`, `imu_aria_hand`, `aria_hand`.

## Project Structure

```
RoSHI-MoCap/
├── src/
│   ├── egoallo/           # Modified EgoAllo: diffusion model + IMU guidance
│   ├── pipeline/          # End-to-end pipeline scripts (01–06)
│   └── utils/             # Shared utilities (incl. imu_pose_viewer for debugging)
├── evaluation/            # Evaluation scripts and ground truth
├── scripts/               # Download scripts
├── model/                 # SMPL-H model files (not tracked)
├── pyproject.toml         # Package configuration
└── requirements_roshi.txt # Pip requirements
```

Environment setup, calibration math, and session layout are documented on the
[project documentation site](https://roshi-mocap.github.io/documentation/).

## Related Repositories

- [RoSHI-App](https://github.com/Jirl-upenn/RoSHI-App) — iOS companion app for video recording with AprilTag detection
