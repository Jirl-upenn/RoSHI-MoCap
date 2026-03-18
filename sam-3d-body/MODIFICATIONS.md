# RoSHI Modifications to SAM-3D-Body

**Original**: [SAM-3D-Body](https://github.com/facebookresearch/sam-3d-body) (Meta AI Research)

## Modifications

1. **Subject filtering (`--subject_only`)**: Added `_select_subject()` to keep only the primary subject per frame when multiple people are detected. Selection criteria: largest bounding-box area, tie-broken by closest camera depth. This is essential for single-subject calibration recordings where background people may enter the scene.
2. **Empty detection handling**: Gracefully handles frames where no human is detected — writes an empty `.npz` and a plain image instead of crashing. This prevents the pipeline from failing on occasional missed frames.
3. **Camera intrinsics passthrough**: Uses per-session `camera.json` intrinsics (from the iOS app) via the `--camera_json` argument, enabling accurate 3D reconstruction from the specific recording device.

