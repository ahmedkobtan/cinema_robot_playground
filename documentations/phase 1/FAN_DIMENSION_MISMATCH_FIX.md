# FAn Dimension Mismatch Fix

## Critical Bug Identified

The FAn implementation had a **critical dimension mismatch** that prevented all detections from working:

### Error
```
WARNING | Error encoding image with DINOv2: mat1 and mat2 shapes cannot be multiplied (1x512 and 768x1)
```

### Root Cause

- **CLIP text features**: 512 dimensions (ViT-B-32)
- **DINOv2 mask features**: 768 dimensions (dinov2-base)

The code was trying to compute similarity between CLIP text features (512-dim) and DINOv2 mask features (768-dim), which is mathematically impossible.

### Original FAn Methodology (Corrected Understanding)

After thorough research of the original FAn implementation:

**For Text Queries**:
- Use **CLIP** for text encoding (512-dim)
- Use **CLIP** for masked region encoding (512-dim)
- Compare CLIP text features with CLIP mask features (same dimension)

**For Image/Click Queries** (not implemented):
- Use **DINO** for query image encoding (768-dim)
- Use **DINO** for masked region encoding (768-dim)
- Compare DINO query features with DINO mask features (same dimension)

### The Fix

Changed the detection logic to use **CLIP for masked regions** when we have a text query:

```python
# Before (WRONG):
# Text query → CLIP (512-dim)
# Masked regions → DINOv2 (768-dim)
# ❌ Cannot compute similarity (dimension mismatch)

# After (CORRECT):
# Text query → CLIP (512-dim)
# Masked regions → CLIP (512-dim)
# ✅ Can compute similarity (same dimension)
```

### Changes Made

1. **Detection (`detect()` method)**:
   - Changed from DINOv2 to CLIP for masked region features
   - Both text and mask features are now 512-dim (compatible)

2. **Re-detection (`update()` method)**:
   - Changed from DINOv2 to CLIP for storing tracked object features
   - Stored features are now CLIP (512-dim), matching detection features

3. **Automatic Re-detection**:
   - Uses CLIP features consistently (text, masks, stored features)
   - All features are 512-dim, so similarity computation works

### Bot-SORT Tracking Fix

Additionally fixed Bot-SORT tracking loss:

**Issue**: Bot-SORT requires `min_hits=3` to confirm a track. We were only providing one detection, then trying to track without detections.

**Fix**: Provide detections for the first 3 frames to establish the track, then switch to periodic re-detection.

```python
# First 3 frames: Provide detections to establish track (min_hits=3)
# After that: Re-detect every 10 frames, track in between
```

## Expected Improvements

1. **FAn Detection**: Should now work (dimension mismatch fixed)
2. **FAn Tracking**: Should be more stable (proper track establishment)
3. **FAn Fallback**: Should also benefit from better track establishment

## Testing

After this fix, FAn should:
- Successfully detect objects (no dimension mismatch errors)
- Establish tracks properly (3 consecutive detections)
- Maintain tracking between re-detections

## References

- Original FAn: https://github.com/alaamaalouf/FollowAnything
- CLIP dimensions: ViT-B-32 produces 512-dim features
- DINOv2 dimensions: dinov2-base produces 768-dim features
- Bot-SORT: Requires min_hits=3 to confirm tracks
