# RoSHI Modifications to MHR (Momentum Human Rig)

**Original**: [MHR](https://github.com/facebookresearch/MomentumHumanRig) (Meta AI Research)

## Modifications:

### `tools/mhr_smpl_conversion/convert_mhr_to_smpl.py`

1. **Empty detection handling**: Added null checks for frames where SAM-3D-Body detected no person (`detections.size == 0`). Returns `None` and skips the frame instead of crashing.
2. **Multiple detection handling**: When SAM-3D-Body returns multiple people, takes the first detection (primary/largest subject) rather than failing on unexpected array shapes.

