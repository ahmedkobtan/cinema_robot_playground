# Follow Anything (FAn) Setup Guide

This guide explains how to set up the Follow Anything (FAn) model for use in the Cinema Robot project.

## Overview

Follow Anything (FAn) is a unified open-vocabulary detection and tracking system based on:
- **SAM 2 (Segment Anything Model 2)**: For segmentation (newer, preferred)
- **SAM (Segment Anything Model)**: For segmentation (fallback)
- **Open-CLIP**: Modern CLIP implementation for text/image encoding
- **DINOv2**: For feature extraction
- **Bot-SORT**: For tracking

All components are installed via PyPI packages - no manual repository setup required!

Repository: https://github.com/alaamaalouf/FollowAnything

## Installation

All dependencies are automatically installed via Poetry:

```bash
poetry install
```

This installs:
- `open-clip-torch` - Modern CLIP implementation
- `sam2` - Segment Anything Model 2 (newer, preferred)
- `segment-anything-py` - Segment Anything Model (fallback)
- `transformers` - For DINOv2
- `boxmot` - For Bot-SORT tracking

### Optional: Download Model Checkpoints

For full functionality with SAM 2 or SAM, you can download checkpoints:

**SAM 2 Checkpoint:**
```bash
# Set environment variable to checkpoint path
export SAM2_CHECKPOINT=/path/to/sam2_checkpoint.pth
```

**SAM Checkpoint (fallback):**
```bash
# Set environment variable to checkpoint path
export SAM_CHECKPOINT=/path/to/sam_checkpoint.pth
```

Note: The model works without checkpoints using Open-CLIP + DINOv2 for detection, but segmentation (SAM/SAM2) requires checkpoints.

## Installation Options (Legacy - Not Required)

### Option 1: Full FAn Repository Setup (Legacy - Not Recommended)

1. **Clone the FollowAnything repository:**
   ```bash
   cd ~
   git clone https://github.com/alaamaalouf/FollowAnything.git
   cd FollowAnything
   ```

2. **Set up Segment-and-Track-Anything submodule:**
   ```bash
   git submodule update --init --recursive
   cd Segment-and-Track-Anything
   ```

3. **Install Segment-and-Track-Anything dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download model weights:**
   ```bash
   # Create checkpoint directory
   mkdir -p ckpt

   # Download SAM model (default: SAM-VIT-B)
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O ckpt/sam_vit_b_01ec64.pth

   # Download AOT model (R50-DeAOT-L)
   wget https://github.com/z-x-yang/Segment-and-Track-Anything/releases/download/v1.0/R50_DeAOTL_PRE_YTB_DAV.pth -O ckpt/R50_DeAOTL_PRE_YTB_DAV.pth
   ```

5. **Install additional dependencies:**
   ```bash
   # Install segment-anything
   pip install git+https://github.com/facebookresearch/segment-anything.git

   # Install CLIP
   pip install git+https://github.com/openai/CLIP.git

   # Install DINOv2 (via transformers - already installed)
   # transformers package already includes DINOv2 support
   ```

6. **Configure the Cinema Robot to use FAn:**
   ```python
   from ahmedkobtan_cinema_robot_playground.src.models.follow_anything_model import FollowAnythingModel

   # Point to your FAn repository
   fan_model = FollowAnythingModel(
       fan_repo_path="~/FollowAnything",
       use_sam=True,
       tracker_type="aot"
   )
   ```

### Option 2: Minimal Setup (Uses Fallback)

If you don't want to set up the full FAn repository, the system will automatically use a fallback implementation that combines:
- **Grounding DINO**: For open-vocabulary detection
- **Bot-SORT**: For tracking

This fallback provides similar functionality and is easier to set up, but may have slightly different performance characteristics.

The fallback is used automatically if:
- The FAn repository is not found
- Required model weights are missing
- Dependencies are not installed

## Dependencies

### Required (via Poetry)
- `torch` and `torchvision`: PyTorch
- `transformers`: For DINOv2
- `opencv-python`: For image processing

### Optional (for full FAn)
- `segment-anything`: SAM model
  ```bash
  pip install git+https://github.com/facebookresearch/segment-anything.git
  ```
- `clip-by-openai`: CLIP model
  ```bash
  pip install git+https://github.com/openai/CLIP.git
  ```

## Usage

### Basic Usage

```python
from ahmedkobtan_cinema_robot_playground.src.models.follow_anything_model import FollowAnythingModel

# Initialize (auto-detects FAn repo or uses fallback)
fan = FollowAnythingModel()

# Detect object from text
image = cv2.imread("frame.jpg")
boxes = fan.detect(image, text_prompt="red cup")

# Track object
initial_bbox = (100, 100, 200, 200)  # x, y, width, height
state = fan.update(image, initial_bbox=initial_bbox)

# Continue tracking
for frame in video_frames:
    state = fan.update(frame)
    if state and not state.lost:
        print(f"Tracking: {state.bbox}")
```

### With Custom FAn Repository Path

```python
fan = FollowAnythingModel(
    fan_repo_path="/path/to/FollowAnything",
    use_sam=True,  # Use SAM for better segmentation
    tracker_type="aot"  # or "siammask"
)
```

## Troubleshooting

### "FollowAnything repository not found"
- Ensure the repository is cloned and accessible
- Check that `follow_anything.py` exists in the repository root
- Verify the path is correct

### "SAM checkpoint not found"
- Download the SAM model weights (see Installation Option 1)
- Ensure the checkpoint is in `FollowAnything/Segment-and-Track-Anything/ckpt/`

### "CLIP not available"
- Install CLIP: `pip install git+https://github.com/openai/CLIP.git`
- The system will fall back to DINO features if CLIP is unavailable

### "AOT tracker not available"
- Ensure Segment-and-Track-Anything submodule is initialized
- Download AOT model weights
- The system will use Bot-SORT as fallback if AOT is unavailable

## Performance Notes

- **With full FAn setup**: Best accuracy, ~20-30 FPS on RTX 2080 Ti
- **With fallback (Grounding DINO + Bot-SORT)**: Good accuracy, ~25-35 FPS on RTX 2080 Ti
- **Without SAM**: Faster but less accurate segmentation
- **VRAM usage**: ~6-8GB for full FAn, ~4-5GB for fallback

## References

- Follow Anything Paper: https://arxiv.org/abs/2308.05737
- Follow Anything Repository: https://github.com/alaamaalouf/FollowAnything
- Segment Anything Model: https://github.com/facebookresearch/segment-anything
- Segment-and-Track-Anything: https://github.com/z-x-yang/Segment-and-Track-Anything
