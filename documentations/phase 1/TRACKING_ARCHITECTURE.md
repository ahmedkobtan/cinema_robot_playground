# Tracking Architecture: AOT vs Bot-SORT

## Original FAn Tracking (AOT)

**Original FAn uses AOT (Associating Objects with Transformers)** - a mask-based tracker:

```python
# Frame 0: Initialize
tracker.add_reference(frame, pred_mask)

# Frame N: Track continuously
pred_mask = tracker.track(frame, update_memory=True)

# Only re-detect when tracking fails
if mean_point is None and cfg['redetect_by'] != 'tracker':
    return "FAILED"  # Trigger re-detection
```

**Key characteristics:**
- Tracks segmentation masks directly (not bounding boxes)
- Tracks continuously without periodic re-detections
- Only re-detects when tracking fails (`mean_point is None`)
- Simpler logic: just track every frame

## Current Implementation (Bot-SORT)

**We use Bot-SORT** - a tracking-by-detection algorithm:

```python
# Frame 0: Initialize with detection
tracker.update(frame, initial_bbox=bbox)

# Frame N: Continue tracking
tracker.update(frame, None)  # Uses Kalman filter prediction

# Periodic re-detection needed (every 5-10 frames)
if frame_count % 10 == 0:
    tracker.update(frame, initial_bbox=new_detection)
```

**Key characteristics:**
- Tracks bounding boxes (not masks)
- Requires periodic re-detections for accuracy
- Can work with prediction for a few frames (max_age=30)
- More complex logic: establishment phase + periodic re-detection

## Why We Use Bot-SORT Instead of AOT

1. **Availability**: AOT requires Segment-and-Track-Anything repository setup
2. **Dependencies**: AOT has complex dependencies (DeAOT model, specific SAM setup)
3. **Bot-SORT is simpler**: Available via PyPI (`boxmot`), easier to integrate
4. **Compatibility**: Works with our bounding box-based detection pipeline

## Solution: SmartTracker Wrapper

We've created a `SmartTracker` wrapper that:
- Handles re-detection logic internally
- Works with any underlying tracker (Bot-SORT, or future AOT)
- Eliminates complex logic from test scripts
- Provides unified interface similar to AOT's simplicity

**Usage:**
```python
# Initialize
tracker = SmartTracker(
    base_tracker=BotSORTTracker(device=device),
    detection_callback=detect_function,  # Optional
    min_hits_to_confirm=5,
    redetect_interval=10,
)

# Simple usage - tracker handles re-detection internally
state = tracker.update(frame, initial_bbox)  # First frame
state = tracker.update(frame, None)  # Subsequent frames
```

## Future: AOT Integration

If we want to match original FAn exactly, we could:
1. Integrate AOT from Segment-and-Track-Anything
2. Create AOTTracker wrapper similar to BotSORTTracker
3. Use SmartTracker with AOT as base_tracker
4. Get continuous tracking like original FAn

But Bot-SORT with SmartTracker is a good compromise for now.
