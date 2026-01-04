# Tracking Methods Explanation

## Three Methods Available

The live tracking script (`test_object_tracking_live.py`) supports **3 methods**:

### 1. `fan` - Follow Anything with SAM 2
- **Components**: Follow Anything model + SAM 2 + Open-CLIP + DINOv2 + Bot-SORT
- **How it works**:
  1. SAM 2 generates segmentation masks
  2. Open-CLIP encodes text prompt ("desk lamp")
  3. Open-CLIP encodes each masked region
  4. Finds mask with highest similarity to text
  5. Converts mask to bounding box
  6. Bot-SORT tracks the bounding box
- **Best for**: High-quality detection and tracking
- **Requires**: SAM 2 checkpoint, Open-CLIP, DINOv2

### 2. `fan_fallback` - Follow Anything Fallback
- **Components**: Grounding DINO + Bot-SORT
- **How it works**:
  1. Grounding DINO detects objects from text prompt
  2. Returns bounding boxes with confidence scores
  3. Bot-SORT tracks the best bounding box
- **Best for**: When SAM 2 is not available or for simpler detection
- **Same as**: `grounding_dino` method (both use Grounding DINO + Bot-SORT)

### 3. `grounding_dino` - Grounding DINO + Bot-SORT Direct
- **Components**: Grounding DINO + Bot-SORT
- **How it works**: Same as `fan_fallback`
- **Best for**: Direct Grounding DINO usage without FAn wrapper
- **Note**: This is essentially the same as `fan_fallback`

## Benchmark Script Methods

The benchmark script (`benchmark_models.py`) tests:

1. **Follow Anything (FAn)**:
   - Uses SAM 2 if available
   - Falls back to Grounding DINO + Bot-SORT if SAM 2 not available
   - This is the same as `fan` method in live tracking

2. **Grounding DINO**:
   - Standalone Grounding DINO detection
   - Same as used in `fan_fallback` and `grounding_dino` methods

3. **Bot-SORT**:
   - Standalone Bot-SORT tracking
   - Used by all methods for frame-to-frame tracking

## Method Comparison

| Method | Detection | Tracking | Quality | Speed |
|--------|-----------|----------|---------|-------|
| `fan` | SAM 2 + Open-CLIP | Bot-SORT | Highest | Medium |
| `fan_fallback` | Grounding DINO | Bot-SORT | Good | Fast |
| `grounding_dino` | Grounding DINO | Bot-SORT | Good | Fast |

**Note**: `fan_fallback` and `grounding_dino` are **functionally identical** - both use Grounding DINO for detection and Bot-SORT for tracking.

## Why Three Methods?

1. **`fan`**: Best quality when SAM 2 is available
2. **`fan_fallback`**: Fallback when SAM 2 components aren't available
3. **`grounding_dino`**: Direct access to Grounding DINO (same as fallback, but explicit)

## Detection Issues Fixed

### Problem 1: Full-Frame Detection
**Issue**: FAn was detecting the entire frame (`x=0, y=0, w=1920, h=1080`) with low confidence (0.22-0.28)

**Cause**:
- When SAM 2 masks don't match well, FAn falls back to full-image similarity
- Confidence threshold was too low (0.1), allowing false positives

**Fix**:
- Reject full-frame bounding boxes (detect if bbox covers >95% of image)
- Raise confidence threshold to 0.3 for FAn
- Log warnings when full-frame detections are rejected

### Problem 2: Grounding DINO Not Detecting
**Issue**: `fan_fallback` wasn't detecting "desk lamp" at all

**Cause**:
- Grounding DINO threshold was 0.3, which might be too conservative
- Confidence scores might be just below 0.3

**Fix**:
- Lower Grounding DINO threshold to 0.25
- Try multiple detections (sorted by confidence)
- Reject full-frame detections
- Use confidence threshold of 0.25 in test script

### Problem 3: Tracking Loss
**Issue**: Detecting object but losing tracking immediately

**Cause**:
- Full-frame bounding boxes can't be tracked (entire image)
- Bot-SORT can't track a box that's the entire frame
- Invalid bounding boxes cause tracking to fail

**Fix**:
- Reject invalid bounding boxes before tracking
- Validate bounding box size and position
- Only initialize tracking with valid detections

## Usage Recommendations

### For Testing:
```bash
# Test all three methods
poetry run python scripts/test_object_tracking_live.py \
  --test-all \
  --device cuda \
  --max-frames 100
```

### For Best Quality:
```bash
# Use fan method (requires SAM 2)
poetry run python scripts/test_object_tracking_live.py \
  --prompt "desk lamp" \
  --method fan \
  --device cuda
```

### For Fast Detection:
```bash
# Use fan_fallback or grounding_dino (same thing)
poetry run python scripts/test_object_tracking_live.py \
  --prompt "desk lamp" \
  --method fan_fallback \
  --device cuda
```

## Expected Behavior After Fixes

1. **FAn method**:
   - Should detect specific objects (not full frame)
   - Confidence > 0.3
   - Tracking should maintain across frames

2. **fan_fallback/grounding_dino**:
   - Should detect objects with Grounding DINO
   - Confidence > 0.25
   - Multiple detections tried (sorted by confidence)
   - Full-frame detections rejected

3. **All methods**:
   - Invalid bounding boxes rejected
   - Better error messages
   - Tracking should be more stable
