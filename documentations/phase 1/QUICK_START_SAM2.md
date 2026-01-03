# Quick Start: Download and Test SAM 2 Checkpoint

## Step 1: Download Checkpoint

**Recommended**: `sam2.1_hiera_base_plus.pt` (best balance for RTX 2080 Ti)

```bash
# Create directory
mkdir -p ~/checkpoints
cd ~/checkpoints

# Download (try SAM 2.1 first, fallback to SAM 2.0)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092924/sam2.1_hiera_base_plus.pt || \
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt

# Verify
ls -lh sam2.1_hiera_base_plus.pt sam2_hiera_base_plus.pt 2>/dev/null
```

## Step 2: Set Environment Variable

```bash
# For SAM 2.1
export SAM2_CHECKPOINT=~/checkpoints/sam2.1_hiera_base_plus.pt

# OR for SAM 2.0
export SAM2_CHECKPOINT=~/checkpoints/sam2_hiera_base_plus.pt
```

## Step 3: Test on CPU (Laptop)

```bash
cd /path/to/cinema_robot_playground
poetry run python scripts/test_sam2_checkpoint.py ~/checkpoints/sam2.1_hiera_base_plus.pt --device cpu
```

Expected output:
- ✓ Checkpoint file exists
- ✓ SAM 2 package imported
- ✓ Model built successfully
- ✓ Predictor created successfully
- ✓ Mask generation works

## Step 4: Test Follow Anything Integration

```bash
# Test FAn with SAM 2
poetry run python scripts/test_fan_end_to_end.py --device cpu
```

## Step 5: Push to PC and Test on GPU

```bash
# On PC with RTX 2080 Ti
export SAM2_CHECKPOINT=~/checkpoints/sam2.1_hiera_base_plus.pt
poetry run python scripts/test_sam2_checkpoint.py ~/checkpoints/sam2.1_hiera_base_plus.pt --device cuda
```

## Troubleshooting

### Config File Not Found

The code auto-detects the config file based on checkpoint name. If it fails:

1. Check SAM 2 installation:
   ```bash
   poetry run python -c "import sam2; print(sam2.__file__)"
   ```

2. Check config directory:
   ```bash
   poetry run python -c "import sam2; import os; print(os.path.join(os.path.dirname(sam2.__file__), 'configs'))"
   ```

3. Manually specify config in code if needed (see `follow_anything_model.py`)

### CUDA Errors on CPU

If you see CUDA errors when testing on CPU, that's expected. The test will still verify:
- Checkpoint file is valid
- Model structure loads correctly
- Code integration works

Full functionality requires GPU.
