# RoSHI: A Versatile Robot-oriented Suit for Human Data In-the-Wild

[Project Page](https://roshi-mocap.github.io/) | [Documentation](https://roshi-mocap.github.io/documentation/) | [RoSHI-App](https://github.com/Jirl-upenn/RoSHI-App) | [RoSHI-Hardware](https://github.com/Jirl-upenn/RoSHI-Hardware)

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
| 4b | `04b_transformer_inference.py` | IMU Transformer pose regression (alternative to step 4) |
| 5a | `05_visualize.py` | Visualize IMU FK and SAM results (no localization) |
| 5b | `05_visualize_roshi.py` | Visualize RoSHI pipeline results |
| 6 | `06_evaluate.py` | Evaluate against OptiTrack ground truth |

All scripts are under `src/pipeline/`. See the
[Recording](https://roshi-mocap.github.io/documentation/pipeline/recording.html) and
[Postprocessing](https://roshi-mocap.github.io/documentation/pipeline/postprocessing.html)
guides for detailed usage.

## IMU Transformer inference (step 4b)

`04b_transformer_inference.py` replaces the diffusion sampler with a transformer
that regresses SMPL-H body poses in a single forward pass. As input it takes the
Aria CPF trajectory and the nine calibrated bone orientations from every frame.
The same JAX post-guidance as step 4 runs afterwards. Output files use the
step-4 format.

Setup: place the checkpoint at `model/transformer/plain_gt_noise_162k.pt` (it is
distributed separately, like the EgoAllo checkpoint) next to `model/smplh/`.

```bash
# On a recording processed by 01-03 (.vrs, mps_*/slam/, sync/imu_info.pkl):
python src/pipeline/04b_transformer_inference.py --traj-root <recording_dir>
# --guidance-mode egoallo (default) | roshi | none; --save-inputs also writes
# the exact model input as <name>_inputs.npz

# On any other data, given as one .npz file:
python src/pipeline/04b_transformer_inference.py --input-npz seq.npz
```

`seq.npz` contains these arrays (30 Hz, N frames; frame 0 is context only):

| Key | Shape | Content |
|-----|-------|---------|
| `Ts_world_cpf` | (N,7) | Aria CPF poses `[qw qx qy qz tx ty tz]`, metres, z up |
| `bone_quats_world` | (N,9,4) | Calibrated bone orientations, wxyz, one common world frame. Alternatively `bone_rotmats_world` (N,9,3,3) |
| `floor_z` | () | Floor height in world z (or `--floor-z`) |
| `timestamps_ns` | (N,) | Optional |

Sensor order: pelvis, L_hip, R_hip, L_knee, R_knee, L_shoulder, R_shoulder,
L_elbow, R_elbow. Pass bone orientations, not raw sensor orientations: apply the
step-2 bone-to-sensor calibration first. The IMU world frame need not match the
Aria world, because the model sees only pelvis-relative orientations and
frame-to-frame rotations. The model expects 30 Hz input. Post-guidance uses IMU
weights of 10 and disables the joint-limit term, matching the reported transformer
results.

## Project Structure

```
RoSHI-MoCap/
├── src/
│   ├── egoallo/       # EgoAllo diffusion model, IMU Transformer, IMU guidance optimizer
│   ├── pipeline/      # Pipeline scripts (01–06)
│   └── utils/         # Shared utilities
├── hardware/          # IMU hardware driver
├── model/             # SMPL-H, SMPL-X, EgoAllo + transformer checkpoints (not tracked)
└── pyproject.toml
```

## Related Repositories

- [RoSHI-App](https://github.com/Jirl-upenn/RoSHI-App) — iOS companion app for video recording with AprilTag detection
- [RoSHI-Hardware](https://github.com/Jirl-upenn/RoSHI-Hardware) — Hardware design files, 3D-printable enclosures, BOM, and ESP32 firmware for wireless IMU trackers
