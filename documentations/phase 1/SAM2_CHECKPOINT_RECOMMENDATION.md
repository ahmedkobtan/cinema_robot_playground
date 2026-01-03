# SAM 2 Checkpoint Recommendation

## Recommended Checkpoint: `sam2.1_hiera_base_plus.pt`

For your RTX 2080 Ti (11GB VRAM) and cinema robot application, I recommend:

**`sam2.1_hiera_base_plus.pt`**

### Why This Checkpoint?

| Metric | Value | Rationale |
|--------|-------|-----------|
| **Accuracy (J&F)** | 78.2 | Excellent accuracy for cinema robot |
| **Speed (FPS)** | 64.1 | Well above 30 FPS target (2x headroom) |
| **Model Size** | 80.8M params | Fits comfortably in 11GB VRAM with CLIP/DINO |
| **VRAM Usage** | ~2-3GB | Leaves ~8GB for CLIP, DINOv2, and other models |

### Comparison with Other Checkpoints

| Checkpoint | Size (M) | FPS | J&F | VRAM | Recommendation |
|------------|----------|-----|-----|------|----------------|
| `sam2.1_hiera_tiny` | 38.9 | 91.2 | 76.5 | ~1GB | Too small, lower accuracy |
| `sam2.1_hiera_small` | 46 | 84.8 | 76.6 | ~1.5GB | Good speed, slightly lower accuracy |
| **`sam2.1_hiera_base_plus`** | **80.8** | **64.1** | **78.2** | **~2-3GB** | **✅ RECOMMENDED** |
| `sam2.1_hiera_large` | 224.4 | 39.5 | 79.5 | ~5-6GB | Best accuracy but tight on VRAM |

## Download Instructions

### Option 1: Direct Download (Recommended)

```bash
# Create checkpoints directory
mkdir -p ~/checkpoints
cd ~/checkpoints

# Download SAM 2.1 checkpoint (September 2024 release)
# Try this URL first:
wget https://dl.fbaipublicfiles.com/segment_anything_2/092924/sam2.1_hiera_base_plus.pt

# If that fails, try the July 2024 release:
# wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt

# Verify download (should be ~300-400 MB)
ls -lh sam2.1_hiera_base_plus.pt
```

### Option 2: Using SAM 2 Repository Script

```bash
# Clone SAM 2 repository
git clone https://github.com/facebookresearch/sam2.git
cd sam2/checkpoints

# Run download script
./download_ckpts.sh

# The checkpoint will be in: sam2/checkpoints/sam2.1_hiera_base_plus.pt
```

## Configuration

After downloading, set the environment variable:

```bash
export SAM2_CHECKPOINT=~/checkpoints/sam2.1_hiera_base_plus.pt
```

Or specify in your code:

```python
import os
os.environ["SAM2_CHECKPOINT"] = "/path/to/sam2.1_hiera_base_plus.pt"
```

## Testing

Once downloaded, test the checkpoint:

```bash
# Test on CPU (laptop)
poetry run python scripts/test_sam2_checkpoint.py ~/checkpoints/sam2.1_hiera_base_plus.pt --device cpu

# Test on GPU (PC with RTX 2080 Ti)
poetry run python scripts/test_sam2_checkpoint.py ~/checkpoints/sam2.1_hiera_base_plus.pt --device cuda
```

## Expected Performance on RTX 2080 Ti

With `sam2.1_hiera_base_plus.pt`:

- **Inference Speed**: ~60-65 FPS (well above 30 FPS target)
- **VRAM Usage**: ~2-3GB for SAM 2
- **Total VRAM**: ~6-7GB (SAM 2 + CLIP + DINOv2 + tracking)
- **Remaining VRAM**: ~4-5GB headroom for other operations

## Alternative Checkpoints

If you want to experiment:

- **For maximum speed**: `sam2.1_hiera_small.pt` (84.8 FPS, 76.6 J&F)
- **For maximum accuracy**: `sam2.1_hiera_large.pt` (39.5 FPS, 79.5 J&F) - may be tight on VRAM

## References

- [SAM 2 PyPI](https://pypi.org/project/sam2/)
- [SAM 2 GitHub](https://github.com/facebookresearch/sam2)
- [Model Performance Table](https://pypi.org/project/sam2/#model-description)
