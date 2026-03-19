# RoSHI: Robust Sparse-sensor Human-body IMU Pose Estimation

[Project Page](https://roshi-mocap.github.io/) | [Documentation](https://roshi-mocap.github.io/documentation/) | [RoSHI-App](https://github.com/Jirl-upenn/RoSHI-App)

A complete pipeline for IMU-based human body pose estimation using 9 body-worn
IMU sensors with rigid AprilTags. RoSHI calibrates sensor-to-bone offsets,
reconstructs full-body pose, synchronizes third-person and egocentric data, and
supports evaluation against OptiTrack ground truth.

## Core Pipeline

- **Collect** a calibration session from the iOS app and the local receiver
- **Calibrate** bone-to-sensor rotation offsets from AprilTag detections and
  SMPL-X body estimates
- **Reconstruct** IMU-only pose and build UTC-aligned synchronized outputs
- **Evaluate** against OptiTrack using the scripts under ``evaluation/``

## Quick Start

```bash
git clone git@github.com:Jirl-upenn/RoSHI-MoCap.git
cd RoSHI-MoCap

python src/pipeline/01_receiver.py --output-dir received_recordings
python src/pipeline/02_imu_calibration.py <session> --smpl-model-path model/smplx/SMPLX_NEUTRAL.npz --output <session>/imu_calibration.json
python src/pipeline/03_imu_pose_viewer.py <session> --port 8082
python src/pipeline/04_sync_pipeline.py <session>
```

Environment setup, model downloads, calibration math, and session layout are
documented on the project documentation site.

## Documentation

- [Getting Started](https://roshi-mocap.github.io/documentation/getting_started/index.html)
- [Installation](https://roshi-mocap.github.io/documentation/getting_started/installation.html)
- [Hardware Components](https://roshi-mocap.github.io/documentation/hardware/components.html)
- [Calibration Math](https://roshi-mocap.github.io/documentation/calibration/math.html)
- [Session Data Layout](https://roshi-mocap.github.io/documentation/pipeline/session_layout.html)
- [Pose Estimation](https://roshi-mocap.github.io/documentation/pipeline/pose_estimation.html)

## Related Repositories

- [RoSHI-App](https://github.com/Jirl-upenn/RoSHI-App) — iOS companion app for video recording with AprilTag detection
