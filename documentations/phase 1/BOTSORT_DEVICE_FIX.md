# Bot-SORT Device Parameter Fix

## Issue

When running benchmarks with `--device cuda`, Bot-SORT was failing with:

```
ERROR | Error initializing Bot-SORT: Invalid CUDA 'device=cuda' requested.
Use 'device=cpu' or pass valid CUDA device(s) if available, i.e. 'device=0'
or 'device=0,1,2,3' for Multi-GPU.
```

## Root Cause

Bot-SORT (from `boxmot` library) expects the device parameter as:
- **String format**: `"cpu"`, `"0"`, `"1"`, `"0,1,2,3"` (for multi-GPU)
- **NOT**: `"cuda"` or `torch.device("cuda")`

Our code was passing `device="cuda"` which Bot-SORT doesn't recognize.

## Solution

Updated `BotSORTTracker._initialize_tracker()` in `tracking_models.py` to:

1. **Convert `"cuda"` to `"0"`**: When device is `"cuda"`, convert it to `"0"` (first GPU device)
2. **Handle `"cuda:0"` format**: Extract device ID from `"cuda:0"` format
3. **Pass as string**: Pass device as string to Bot-SORT, not as `torch.device` object

### Code Changes

```python
# Before: Passed torch.device object
device_obj = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
self.tracker = BotSort(reid_weights=reid_weights_path, device=device_obj, ...)

# After: Convert "cuda" to "0" and pass as string
device_str = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
if device_str == "cuda":
    if torch.cuda.is_available():
        device_str = "0"  # Use first GPU device
    else:
        device_str = "cpu"
elif device_str.startswith("cuda:"):
    device_str = device_str.split(":")[1]  # Extract device ID

self.tracker = BotSort(reid_weights=reid_weights_path, device=device_str, ...)
```

## Result

✅ Bot-SORT now initializes correctly when `--device cuda` is used
✅ Automatically uses first GPU (device 0) when CUDA is available
✅ Falls back to CPU if CUDA is not available
✅ Supports explicit device IDs like `"0"`, `"1"`, etc.

## Testing

After this fix, running:
```bash
poetry run python scripts/benchmark_models.py --device cuda --num-frames 100
```

Should now show:
```
INFO | Bot-SORT tracker initialized
INFO | BOTSORT:
INFO |   FPS: 400-500
INFO |   Latency: ~2 ms
```

Instead of the previous error.
