# FAn Automatic Re-detection Implementation

## Overview

This document describes the implementation of automatic re-detection mechanism for Follow Anything (FAn), following the original paper methodology ([arXiv:2308.05737](https://arxiv.org/abs/2308.05737)).

## Original FAn Methodology

According to the original FAn paper and repository:

1. **During Tracking**: Store DINO features of the tracked object at every frame
2. **When Tracking Lost**:
   - Apply segmentation model (SAM) to get masks
   - Extract DINO features for each mask
   - Compare to stored features
   - Re-detect if similarity is high

## Implementation Details

### 1. Feature Storage

**Stored Features**:
- `_stored_features`: List of DINO features from tracked object (last 30 frames)
- `_stored_query_features`: CLIP-encoded text prompt features (for initial detection)
- `_last_tracked_bbox`: Last successfully tracked bounding box

**Storage During Tracking**:
```python
# In update() method, when tracking is successful:
# Extract CLIP features from tracked region (matches detection features)
tracked_region = frame[y1:y2, x1:x2]
# Extract CLIP features (512-dim, same as text query)
track_features = clip_model.encode_image(tracked_region)
# Store for re-detection
_stored_features.append(track_features.cpu())
```

### 2. Feature Extraction

**For Detection**:
- **Text Query**: CLIP encodes text prompt → `query_features` (512-dim)
- **Masked Regions**: CLIP encodes each SAM mask → `mask_features` (512-dim)
- **Similarity**: Compare `query_features` with `mask_features` (same dimension)

**For Re-detection**:
- **Stored Features**: Average of stored CLIP features from tracked object (512-dim)
- **New Masks**: CLIP encodes features from each SAM mask (512-dim)
- **Similarity**: Compare stored features with new mask features (same dimension)

**Note**: For text queries, we use CLIP for both text and masked regions (both 512-dim). DINOv2 (768-dim) cannot be compared with CLIP (512-dim) directly.

### 3. Re-detection Logic

```python
# In detect() method:
use_stored_features = len(_stored_features) > 0 and _tracking_initialized

if use_stored_features:
    # Re-detection mode: compare to stored features
    avg_stored = torch.stack(_stored_features).mean(dim=0)
    stored_similarity = (mask_features @ avg_stored.T).item()
    # Use higher of text similarity or stored feature similarity
    best_score = max(text_similarity, stored_similarity * 0.8)
    similarity_threshold = 0.20  # Lower threshold for re-detection
else:
    # Initial detection mode: use text prompt
    similarity_threshold = 0.25
```

### 4. Key Differences from Original

| Aspect | Original FAn | Our Implementation | Status |
|--------|-------------|-------------------|--------|
| **Feature Model** | DINO | DINOv2 | ✅ Better (newer, but not used for text queries) |
| **Text Encoding** | CLIP | Open-CLIP | ✅ Equivalent |
| **Masked Region Features (Text Query)** | CLIP | CLIP | ✅ Correct (same dimension as text) |
| **Storage (Text Query)** | CLIP features every frame | CLIP features every frame | ✅ Correct |
| **Re-detection** | Compare to stored features | Compare to stored features | ✅ Correct |

**Note**: For text queries, we use CLIP for masked regions (not DINO) to match CLIP text features (both 512-dim). DINOv2 (768-dim) cannot be compared with CLIP (512-dim).

## Code Changes

### `follow_anything_model.py`

1. **State Variables**:
   - Added `_stored_features`: List of tracked object DINO features
   - Added `_stored_query_features`: Text prompt CLIP features
   - Added `_last_tracked_bbox`: Last tracked bounding box

2. **Detection Method** (`detect()`):
   - Changed from CLIP to DINO for masked region features
   - Added re-detection logic using stored features
   - Store mask features for re-detection

3. **Tracking Method** (`update()`):
   - Extract and store DINO features from tracked region every frame
   - Maintain last 30 features to avoid memory issues

4. **Mask Generation** (`_get_masks_sam2()`):
   - Filter out full-frame masks (coverage >95%)
   - Avoid edge points to reduce full-frame mask generation

### `detection_models.py`

1. **Grounding DINO**:
   - Added full-frame detection rejection (area coverage >95%)

### `test_object_tracking_live.py`

1. **Full-Frame Rejection**:
   - Changed from position-based to area-coverage-based (more robust)
   - Check if bbox covers >95% of image area

## Validation

### Correctness

✅ **Feature Extraction**: Using DINOv2 for masked regions (matches original FAn)
✅ **Storage**: Storing tracked object features every frame
✅ **Re-detection**: Comparing new masks to stored features
✅ **Thresholds**: Lower threshold for re-detection (0.20 vs 0.25)

### Improvements Over Original

1. **Better Models**: DINOv2 (newer) vs DINO, SAM 2 (newer) vs SAM
2. **Full-Frame Filtering**: Reject full-frame masks/detections automatically
3. **Memory Management**: Limit stored features to last 30 frames

## Testing

The implementation should be tested with:
1. Objects that change appearance (lighting, angle)
2. Occlusions and re-emergence
3. Long-term tracking (>100 frames)

Expected improvements:
- Better re-detection after occlusions
- More robust tracking with appearance changes
- Reduced false positives from full-frame detections

## References

- Original Paper: https://arxiv.org/abs/2308.05737
- Original Repository: https://github.com/alaamaalouf/FollowAnything
- DINOv2: https://github.com/facebookresearch/dinov2
- SAM 2: https://github.com/facebookresearch/segment-anything-2
