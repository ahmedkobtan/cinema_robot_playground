---
name: Cinema Bot Architecture Research & Plan
overview: Comprehensive research-based architecture plan for the AI-powered cinema robot, addressing all questions from the brainstorming document with state-of-the-art model recommendations and simplified first-iteration approach.
todos:
  - id: research-models
    content: Research and document best real-time video detection/segmentation/grounding models for RTX 2080 Ti
    status: pending
  - id: design-architecture
    content: Design three-layer hierarchical architecture (Director/Cinematographer/Pilot) with clear interfaces
    status: pending
    dependencies:
      - research-models
  - id: solve-handoff-problem
    content: Design solution for VLM-to-tracker handoff that works with any object (not just pre-trained classes)
    status: pending
    dependencies:
      - research-models
  - id: define-shot-types
    content: Research and document common cinematic shot types with robot motion specifications
    status: pending
  - id: rl-strategy
    content: "Define RL strategy: when to use, what to train, offline vs online, integration points"
    status: pending
    dependencies:
      - design-architecture
  - id: simplify-first-iteration
    content: "Define minimal viable first iteration: what to include, what to skip, core features only"
    status: pending
    dependencies:
      - design-architecture
      - solve-handoff-problem
  - id: latency-optimization
    content: Define latency optimization strategy for real-time control loop
    status: pending
    dependencies:
      - design-architecture
  - id: edge-migration-path
    content: Document future migration path to Jetson/Orange Pi for edge deployment
    status: pending
    dependencies:
      - design-architecture
---

# Cinema Bot Architecture: Research-Based Implementation Plan

## Executive Summary

This plan addresses all questions raised in the brainstorming document through comprehensive research on state-of-the-art models, architectures, and best practices for real-time video processing, object tracking, and robotic control. The architecture is designed for an RTX 2080 Ti (11GB VRAM) with a focus on simplicity for the first iteration while maintaining a path to advanced features.

## 1. Model Selection: Real-Time Video Detection, Segmentation & Grounding

### Research Findings

**Key Question**: Are PaliGemma/SAM3/DINO the best models, or are there better real-time alternatives?

**Answer**: Based on 2024-2025 research, the optimal stack differs from initial suggestions:

#### For Object Grounding (VLM Layer):

- **Grounding DINO** (Recommended for First Iteration)
  - Zero-shot detection: 52.5% AP on COCO
  - Combines DINO detection with language understanding
  - Runs efficiently on RTX 2080 Ti
  - Better than PaliGemma for this specific use case (grounding vs. general VLM tasks)

- **Grounded SAM 2** (Alternative/Upgrade Path)
  - Integrates Florence-2 with SAM 2
  - Zero-shot segmentation based on text prompts
  - More recent (July 2024) than original SAM
  - Better for objects not in pre-trained lists

- **PaliGemma**: Good for general VLM tasks but overkill for simple grounding. Consider for future iterations if you need scene understanding beyond object detection.

#### For Real-Time Tracking (After Grounding):

- **SAM 3 Performance Reality Check**:
  - Research shows SAM 3 can take several minutes for full video propagation
  - **NOT suitable for real-time** (1-2 FPS at best)
  - Better alternatives exist

- **Recommended Tracking Stack**:

  1. **FastSAM or MobileSAM** (For Segmentation Masks)

     - Provides pixel-level masks (better than bounding boxes)
     - Runs at 30+ FPS on RTX 2080 Ti
     - Can be prompted with initial bounding box from Grounding DINO

  1. **ByteTrack** (For Multi-Object Tracking)

     - State-of-the-art: 80.3 MOTA, 77.3 IDF1 on MOT17
     - Runs at 30 FPS on V100 (will be faster on RTX 2080 Ti)
     - Handles low-score detections better than other trackers

  1. **Bot-SORT** (Alternative Lightweight Tracker)

     - Good for single-object tracking
     - Very fast, minimal computational overhead

#### For Real-Time Detection (If Not Using VLM):

- **RTMDet** (Best Overall Choice)
  - 52.8% AP on COCO at 300+ FPS on RTX 3090
  - Will achieve 200+ FPS on RTX 2080 Ti
  - Multiple variants (Tiny: 1020+ FPS, Small: 819 FPS)
  - Better than YOLOv11 for this use case

- **YOLOv12** (Alternative)
  - Latest YOLO variant with "Area Attention"
  - 1.64ms latency on T4 GPU
  - Good if you prefer YOLO ecosystem

### Handoff Problem Solution

**Your Concern**: "How does handoff to YOLO work if object isn't in YOLO's classes?"

**Answer**: Don't use YOLO for tracking. Use this workflow instead:

1. **Grounding DINO** finds object from text prompt → outputs bounding box
2. **FastSAM/MobileSAM** receives bounding box as prompt → generates segmentation mask
3. **ByteTrack or Bot-SORT** tracks the mask frame-to-frame (doesn't care about object class)
4. Result: 30+ FPS tracking of ANY object, regardless of whether it's in a pre-trained list

This solves the "open-vocabulary" problem completely.

## 2. VLM Direct Control: Speed Analysis

**Your Question**: "If VLM provides inputs directly to motor, would it be too slow? Can it be a realtime VLM agent?"

**Answer**:

- **Direct VLM control is too slow** (5-15 FPS)
- **Solution**: Hierarchical architecture (see Section 3)
- VLM runs **once** at start (or every 5 seconds if tracking lost)
- Fast tracker handles real-time updates (30+ FPS)
- This is the standard approach in robotics

## 3. Agentic AI Architecture

### Three-Layer Hierarchical Architecture

Based on research and your requirements, this architecture balances capability with simplicity:

```
┌─────────────────────────────────────────────────────────┐
│ Layer 1: Director (VLM Agent) - "What to Film"          │
│ - Input: Voice/Text command                              │
│ - Model: Grounding DINO                                  │
│ - Frequency: Once at start, or every 5s if lost          │
│ - Output: Initial bounding box coordinates                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 2: Cinematographer (Motion Planner) - "How to Film"│
│ - Input: Object coordinates + Shot type command         │
│ - Logic: Trajectory calculation (orbit, dolly, etc.)     │
│ - Frequency: 30 FPS (real-time)                          │
│ - Output: High-level JSON commands                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 3: Pilot (Controller) - "Execute Smoothly"        │
│ - Input: JSON commands from Layer 2                      │
│ - Method: PID controller (first iteration)                │
│ - Future: RL agent (trained offline)                      │
│ - Output: Serial commands to Arduino                     │
└────────────────────┬────────────────────────────────────┘
```

### Decision-Making Approach

**Your Concern**: "Not aligned on decision-making approach"

**Answer for First Iteration**:

- **Layer 2 (Cinematographer)**: Rule-based trajectory planner
  - Simple math: calculate circular path for "orbit"
  - Calculate forward/backward for "dolly"
  - No AI needed here initially

- **Layer 3 (Pilot)**: PID controller
  - Classic control theory
  - Smooths out jerky motor movements
  - Can be replaced with RL later

**Future Iteration**:

- RL policy can replace Layer 3 (Pilot)
- Input to RL: Object position, desired shot type, current robot state
- Output: Motor commands
- Trained offline in Isaac Sim, then deployed

## 4. Reinforcement Learning Strategy

### Your Questions:

1. "Can RL be used to keep object in frame and follow cinematic shot style?"
2. "Should RL be trained offline?"

### Research-Based Answers:

**Yes, RL can do both, but prioritize for first iteration:**

#### First Iteration (Simplified):

- **Skip RL entirely**
- Use PID controller for smooth motion
- Focus on getting basic tracking working
- RL adds complexity without immediate benefit

#### Future Iteration (RL Integration):

- **Two RL Use Cases**:

  1. **Motion Smoothness** (Layer 3 - Pilot)

     - Learn motor control to overcome friction
     - Train in Isaac Sim with your robot model
     - Input: Desired velocity, current state
     - Output: Motor PWM values

  1. **Shot Planning** (Layer 2 - Cinematographer)

     - Learn optimal camera angles and movements
     - Input: Object position, shot type, scene context
     - Output: Trajectory parameters
     - More complex, save for later

- **Training Strategy**:
  - Train **offline** in simulation (Isaac Sim)
  - Transfer to real robot (Sim2Real)
  - Fine-tune with real-world data if needed

## 5. Cinematic Shot Types

**Your Question**: "I don't know what cinematic shots exist"

### Common Professional Shots:

| Shot Type | Robot Motion | Servo Motion | Use Case |

|-----------|--------------|--------------|----------|

| **Dolly In/Out** | Drive forward/backward | Locked forward | Dramatic focus, reveal |

| **Orbit** | Circle around object | Pan to keep centered | Dynamic hero shot |

| **Dolly Zoom** | Drive back + digital zoom in | Locked forward | Vertigo effect |

| **Reveal** | Drive sideways (arc) | Pan from side to center | Reveal object from behind |

| **Follow** | Match object speed | Pan to keep centered | Standard tracking |

| **Lead** | Drive ahead of object | Pan backward | Anticipatory shot |

**First Iteration**: Implement "Follow" and "Orbit" only. Add others incrementally.

## 6. System Behavior with Moving Objects

**Your Question**: "How will system behave if object keeps moving?"

**Answer**:

- **Fast tracker** (ByteTrack/Bot-SORT) updates at 30 FPS
- **Cinematographer** calculates error every frame
- **Pilot** adjusts motor speed in real-time
- If object moves faster than robot can follow:
  - Increase motor speed (up to max)
  - If still can't keep up, that's a hardware limitation (acceptable)
- If object is lost (occlusion):
  - VLM (Grounding DINO) re-runs to re-acquire
  - Happens automatically every 5 seconds or on loss detection

## 7. First Iteration: Simplified Architecture

### Core Principle: "Get it working, then make it smart"

**Minimal Viable Stack**:

1. **Vision Pipeline**:

   - Phone → Wi-Fi stream (720p @ 30fps)
   - OpenCV captures stream
   - Grounding DINO finds object (once)
   - FastSAM tracks object (30 FPS)

2. **Control Pipeline**:

   - Calculate error (object center vs screen center)
   - PID controller generates motor commands
   - Send to Arduino via Serial

3. **Shot Types**:

   - "Follow" (center object)
   - "Orbit" (circle around object)

**What to Skip in First Iteration**:

- ❌ RL (use PID instead)
- ❌ Voice commands (use text input)
- ❌ Complex shot types (just follow + orbit)
- ❌ LangChain/AutoGPT agents (simple Python logic)
- ❌ Multiple object tracking (single object only)

**What to Include**:

- ✅ Grounding DINO for object finding
- ✅ FastSAM for tracking
- ✅ PID controller for smooth motion
- ✅ Basic shot planning (follow/orbit)
- ✅ Serial communication to Arduino

## 8. Hardware-Software Integration

### Serial Protocol (Arduino Communication)

**Fixed Arduino Firmware** (upload once):

```
Commands:
- M:F:speed    (Move Forward)
- M:B:speed    (Move Backward)
- M:L:speed    (Turn Left)
- M:R:speed    (Turn Right)
- S:angle      (Set Servo angle, 0-180)
- STOP         (Emergency stop)
```

**Python Agent** (PC side):

- Calculates desired motion
- Converts to serial commands
- Sends at 30 FPS
- Handles acknowledgments

## 9. Latency Optimization

**Strategy**:

- 720p @ 30fps stream (not 4K)
- Process every frame (no frame skipping)
- Use GPU acceleration (CUDA for all models)
- Minimize serial communication overhead
- Target: <100ms total latency (camera → motor)

## 10. Future Migration Path (Jetson/Orange Pi)

**Architecture is designed for edge deployment**:

- All models can run on Jetson Orin (40 TOPS)
- Quantize models for smaller devices
- Same three-layer architecture
- Just change hardware, not software structure

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

- [ ] Set up video streaming (phone → PC)
- [ ] Implement Grounding DINO integration
- [ ] Implement FastSAM tracking
- [ ] Basic Arduino firmware
- [ ] Simple "center object" control (servo only)

### Phase 2: Basic Tracking (Week 3-4)

- [ ] Add wheel control
- [ ] Implement PID controller
- [ ] "Follow" shot type
- [ ] Test with moving objects

### Phase 3: Cinematic Shots (Week 5-6)

- [ ] Implement "Orbit" shot
- [ ] Add depth control (Z-axis)
- [ ] Refine motion smoothness

### Phase 4: Advanced Features (Future)

- [ ] Voice command integration (Whisper)
- [ ] Additional shot types
- [ ] RL training in Isaac Sim
- [ ] RL deployment (replace PID)

## Key Takeaways

1. **Model Stack**: Grounding DINO → FastSAM → ByteTrack (not PaliGemma → YOLO → SAM3)
2. **Architecture**: Three-layer hierarchy (Director → Cinematographer → Pilot)
3. **First Iteration**: Skip RL, use PID. Focus on getting basic tracking working.
4. **Handoff Problem**: Solved by using promptable trackers (FastSAM) instead of class-based (YOLO)
5. **Simplicity**: Start with text input, "Follow" and "Orbit" shots only
6. **Future-Proof**: Architecture designed for edge deployment later

This plan addresses all questions from your brainstorming document with research-backed recommendations while keeping the first iteration simple and achievable.
