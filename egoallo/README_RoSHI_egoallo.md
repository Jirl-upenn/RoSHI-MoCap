# RoSHI EgoAllo

This is the RoSHI project's fork of [EgoAllo](https://egoallo.github.io/), extended with IMU-guided body pose estimation.

---

## Download Environment

EgoAllo requires **Python 3.12** or newer.

### 1. Create conda environment

```bash
conda create -n egoallo_opt python=3.12
conda activate egoallo_opt
```

### 2. Install the package

```bash
cd egoallo
pip install -e .
```

### 3. Install JAX with CUDA support (for guidance optimization)

```bash
pip install "jax[cuda12]==0.6.1"
pip install git+https://github.com/brentyi/jaxls.git
```

### 4. Install additional dependencies

```bash
pip install tyro safetensors pyyaml tqdm pandas
```

### 5. Download the SMPL-H model

Download the "Extended SMPL+H model" (16 shape parameters) from the [MANO project webpage](https://mano.is.tue.mpg.de/) and place it at:

```
data/smplh/neutral/model.npz
```

### 6. Download the model checkpoint

```bash
bash download_checkpoint_and_data.sh
```

Or download manually: [checkpoint](https://drive.google.com/file/d/14bDkWixFgo3U6dgyrCRmLoXSsXkrDA2w/view?usp=drive_link).

The checkpoint should be at `./egoallo_checkpoint_april13/checkpoints_3000000/`.

---

## Prepare Data

Each dataset lives under `received_recordings/` in the RoSHI_Calibration repo. To run EgoAllo inference on a dataset, you need to place the Aria VRS file, its JSON sidecar, and the MPS outputs inside the dataset folder.

### Expected folder structure

```
received_recordings/
└── dataset1/                       # Try to keep the folder name and the Aria VRS recording the same
    ├── Dataset1_1.vrs              # Aria VRS recording
    ├── Dataset1_1.vrs.json         # VRS metadata sidecar
    ├── mps_Dataset1_1_vrs/         # MPS outputs (name: mps_<vrs_name>)
    │   ├── slam/
    │   │   ├── closed_loop_trajectory.csv
    │   │   ├── semidense_points.csv.gz
    │   │   ├── open_loop_trajectory.csv
    │   │   └── online_calibration.jsonl
    │   ├── hand_tracking/
    │   │   └── hand_tracking_results.csv
    │   └── eye_gaze/
    │       └── general_eye_gaze.csv
    ├── imu/                        # IMU sensor data
    ├── body_data/                  # Body tracking data
    └── ...
```

### Steps

1. **Copy the VRS file and its JSON sidecar** into the dataset folder:

   ```bash
   cp /path/to/recording.vrs received_recordings/dataset1/
   cp /path/to/recording.vrs.json received_recordings/dataset1/
   ```

2. **Run MPS** on the VRS file using [Project Aria Machine Perception Services](https://facebookresearch.github.io/projectaria_tools/docs/data_utilities/core_code_snippets/mps), then place the output folder (containing `slam/`, `hand_tracking/`, `eye_gaze/`) into the dataset folder. The MPS folder should be named `mps_<vrs_filename>/` (e.g., `mps_Dataset1_1_vrs/`).

3. **Verify** that the required files exist:

   ```bash
   # These are required for EgoAllo inference:
   ls received_recordings/dataset1/*.vrs
   ls received_recordings/dataset1/mps_*/slam/closed_loop_trajectory.csv
   ls received_recordings/dataset1/mps_*/slam/semidense_points.csv.gz
   ```

---

## Run Inference

Run inference from the `egoallo/` directory with the conda environment activated:

```bash
conda activate egoallo_opt
cd egoallo
```

### Basic usage

```bash
python 3_aria_inference.py \
  --traj-root /path/to/received_recordings/dataset1
```

### Full example with common options

```bash
python 3_aria_inference.py \
  --traj-root '/mnt/aloque_scratch/mwenjing/research/RoSHI_Calibration/received_recordings/Dataset1_1' \
  --guidance-mode aria_hamer \
  --start-index 0 \
  --traj-length 128
```

### Command-line options

| Option | Default | Description |
|---|---|---|
| `--traj-root PATH` | *(required)* | Path to the dataset folder containing the VRS and MPS data |
| `--checkpoint-dir PATH` | `egoallo_checkpoint_april13/checkpoints_3000000` | Path to model checkpoint |
| `--smplh-npz-path PATH` | `../model/smplh/neutral/model.npz` | Path to the SMPL-H model file |
| `--start-index INT` | `0` | Frame index to start inference at (within the downsampled trajectory) |
| `--traj-length INT` | `128` | Number of timesteps to estimate body motion for |
| `--num-samples INT` | `1` | Number of samples to take |
| `--guidance-mode` | `aria_hamer` | Guidance mode: `no_hands`, `aria_wrist_only`, `aria_hamer`, `hamer_wrist`, `hamer_reproj2` |
| `--guidance-inner / --no-guidance-inner` | `True` | Apply guidance optimizer between denoising steps |
| `--guidance-post / --no-guidance-post` | `True` | Apply guidance optimizer after diffusion sampling |
| `--save-traj / --no-save-traj` | `True` | Save output trajectory to `traj_root/egoallo_outputs/` |
| `--overwrite-output / --no-overwrite-output` | `False` | Overwrite existing output file when using the same output name |
| `--visualize-traj / --no-visualize-traj` | `False` | Launch viser visualization after sampling |
| `--glasses-x-angle-offset FLOAT` | `0.0` | Rotate CPF poses by an X angle offset |

### Output

Results are saved to `<traj_root>/egoallo_outputs/` as NPZ files containing:

- `Ts_world_cpf` — CPF poses in world frame
- `Ts_world_root` — Root joint pose in world frame
- `body_quats` — Local body joint quaternions (21 joints)
- `left_hand_quats` / `right_hand_quats` — Local hand joint quaternions
- `contacts` — Contact values per joint
- `betas` — Body shape parameters
- `frame_nums` — Frame indices
- `timestamps_ns` — Tracking timestamps in nanoseconds
