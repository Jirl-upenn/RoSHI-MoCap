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
| 5 | `src/pipeline/05_visualize.py` | Multi-method visualization (SAM-3D, IMU FK, RoSHI, EgoAllo) |
| 6 | `src/pipeline/06_evaluate.py` | Evaluate against OptiTrack ground truth |

## Quick Start

```bash
# Calibration & data preparation
python src/pipeline/01_receiver.py --output-dir received_recordings
python src/pipeline/02_calibrate.py <session> --smpl-model-path model/smplx/SMPLX_NEUTRAL.npz --output <session>/imu_calibration.json
python src/pipeline/03_sync.py <session>

# EgoAllo inference (requires GPU + JAX with CUDA)
python src/pipeline/04_inference.py \
  --traj-root /path/to/received_recordings/dataset1

# Visualize all methods side-by-side (SAM-3D, IMU FK, RoSHI, EgoAllo + third-person RGB)
python src/pipeline/05_visualize.py <session>

# Only compare specific methods
python src/pipeline/05_visualize.py <session> --no-imu --no-roshi        # EgoAllo vs SAM-3D
python src/pipeline/05_visualize.py <session> --no-egoallo --no-roshi    # IMU FK vs SAM-3D
python src/pipeline/05_visualize.py <session> --roshi-csv path/to/pred.csv    # custom CSV
```

Available guidance modes: `egoallo`, `egoallo_ariawrist`, `roshi` (default), `roshi_ariahand`.

## Project Structure

```
RoSHI-MoCap/
├── src/
│   ├── egoallo/           # Modified EgoAllo: diffusion model + IMU guidance
│   ├── pipeline/          # End-to-end pipeline scripts (01–06)
│   └── utils/             # Shared utilities (incl. imu_pose_viewer for debugging)
├── evaluation/            # Evaluation scripts and ground truth
├── scripts/               # Download scripts
├── model/                 # Model files: SMPL-H, SMPL-X, EgoAllo checkpoint (not tracked)
├── pyproject.toml         # Package configuration
└── requirements_roshi.txt # Pip requirements
```

Environment setup, calibration math, and session layout are documented on the
[project documentation site](https://roshi-mocap.github.io/documentation/).

## Related Repositories

- [RoSHI-App](https://github.com/Jirl-upenn/RoSHI-App) — iOS companion app for video recording with AprilTag detection
