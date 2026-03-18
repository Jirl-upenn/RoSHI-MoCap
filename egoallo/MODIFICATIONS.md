# RoSHI Modifications to EgoAllo

**Original**: [EgoAllo](https://egoallo.github.io/) (Meta AI Research)

## Summary

EgoAllo is extended with IMU-guided body pose estimation for the RoSHI pipeline. The core change is adding IMU sensor data as an additional guidance signal during the diffusion-based pose sampling process, enabling more accurate body pose estimation when IMU data is available alongside Aria egocentric video.

## Key Modifications

### IMU Data Integration
- **`src/egoallo/imu_detection_structs.py`** (new): Data structures for loading and aligning IMU sensor readings with Aria trajectory timestamps.
- **`src/egoallo/imu_utils.py`** (new): IMU quaternion processing and coordinate frame alignment utilities.
- **`src/egoallo/joint_config.py`** (new): RoSHI joint ID mapping for the 9 instrumented body joints.

### Guidance Optimizers (JAX)
- **`src/egoallo/guidance_optimizer_jax.py`**: Extended with IMU rotation constraints in the optimization objective.
- **`src/egoallo/guidance_optimizer_jax_whole_body.py`** (new): Whole-body guidance optimizer that jointly optimizes over all IMU-instrumented joints.
- **`src/egoallo/guidance_optimizer_jax_manipulation.py`** (new): Manipulation-specific optimizer variant for object interaction scenarios.
- **`src/egoallo/guidance_optimizer_jax_ori.py`** (new): Original optimizer preserved as reference.

### Inference & Sampling Pipeline
- **`src/egoallo/inference_utils.py`**: Modified to accept and forward IMU data through the inference pipeline.
- **`src/egoallo/sampling.py`**: Extended diffusion sampling to incorporate IMU guidance at each denoising step.
- **`src/egoallo/hand_detection_structs.py`**: Extended hand landmark data structures.

### Data & Configuration
- **`src/egoallo/config.py`** (new): Centralized configuration for IMU-EgoAllo experiments.
- **`src/egoallo/data/`** (new): Data loading modules for AMASS, Aria MPS trajectories, and EgoPose datasets.

### Visualization & Output
- **`src/egoallo/vis_helpers.py`**: Extended with IMU overlay, SMPLX mesh rendering, and Aria glasses visualization.
- **`vis_output.py`** (new): Standalone visualization for comparing EgoAllo outputs with ground truth and IMU-only baselines.
- **`npz_to_imu_csv.py`** (new): Converts EgoAllo NPZ outputs to calibrated IMU CSV format for downstream evaluation.

## Setup

See `README_RoSHI_egoallo.md` for environment setup, data preparation, and inference instructions.
