# RoSHI: A Versatile Robot-oriented Suit for Human Data In-the-Wild

[Project Page](https://roshi-mocap.github.io/) | [Documentation](https://roshi-mocap.github.io/documentation/) | [RoSHI-App](https://github.com/Jirl-upenn/RoSHI-App)

## Quick Start

```bash
pip install -e .
pip install git+https://github.com/brentyi/jaxls.git
```

For full installation (conda environments, model downloads, hardware setup),
see the [Installation Guide](https://roshi-mocap.github.io/documentation/pipeline/installation.html).

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_receiver.py` | Receive data from iOS app + run calibration |
| 2 | `02_calibrate.py` | Calibrate bone-to-sensor rotation offsets |
| 3 | `03_sync.py` | Synchronize RGB + calibrated IMU data |
| 4 | `04_inference.py` | EgoAllo diffusion-based pose estimation |
| 5 | `05_visualize.py` | Multi-method visualization |
| 6 | `06_evaluate.py` | Evaluate against OptiTrack ground truth |

All scripts are under `src/pipeline/`. See the
[Recording](https://roshi-mocap.github.io/documentation/pipeline/recording.html) and
[Postprocessing](https://roshi-mocap.github.io/documentation/pipeline/postprocessing.html)
guides for detailed usage.

## Project Structure

```
RoSHI-MoCap/
├── src/
│   ├── egoallo/       # EgoAllo diffusion model + IMU guidance optimizer
│   ├── pipeline/      # Pipeline scripts (01–06)
│   └── utils/         # Shared utilities
├── hardware/          # IMU hardware driver
├── model/             # SMPL-H, SMPL-X, EgoAllo checkpoint (not tracked)
└── pyproject.toml
```

## Related Repositories

- [RoSHI-App](https://github.com/Jirl-upenn/RoSHI-App) — iOS companion app for video recording with AprilTag detection
