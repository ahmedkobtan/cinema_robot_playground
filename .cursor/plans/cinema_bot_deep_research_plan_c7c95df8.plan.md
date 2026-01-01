---
name: Cinema Bot Deep Research Plan
overview: Comprehensive research plan addressing all technical questions from the cinema bot architecture brainstorming, including model selection, architecture decisions, RL approaches, and first iteration simplification strategies.
todos:
  - id: research-models
    content: Research and benchmark RTMDet, Grounding DINO, Mobile-Seed, and Bot-SORT on RTX 2080 Ti hardware to verify real-time performance claims
    status: pending
  - id: prototype-tracking
    content: "Implement prototype tracking pipeline: Grounding DINO → Bot-SORT handoff for open-vocabulary object tracking"
    status: pending
    dependencies:
      - research-models
  - id: design-architecture
    content: Design detailed software architecture with Director/Cinematographer/Pilot layers and define serial communication protocol
    status: pending
  - id: implement-visual-servoing
    content: Implement PID-based visual servoing controller for servo (X-axis) and wheels (Z-axis) with movement hierarchy
    status: pending
    dependencies:
      - prototype-tracking
  - id: test-mvp
    content: "Build and test MVP: object detection → tracking → visual servoing → Arduino control on actual hardware"
    status: pending
    dependencies:
      - implement-visual-servoing
      - design-architecture
---

# Cinema Bot Architecture: Deep Research & Technical Analysis

## Executive Summary

This document provides in-depth research and analysis for all questions raised in the cinema bot architecture brainstorming, aligned with the cinema bot vision and available hardware (RTX 2080 Ti, Arduino Uno, 4WD DC motors, FS90 servo, smartphone camera).

## 1. Real-Time Video Detection, Segmentation & Grounding Models

### 1.1 Current State-of-the-Art (2024-2025)

**Key Finding**: The models mentioned (PaliGemma, SAM3, Grounding DINO) are valid but not necessarily optimal for real-time robotics on RTX 2080 Ti.

#### Recommended Models for First Iteration:

**A. Object Detection: RTMDet (OpenMMLab)**

- **Performance**: 52.8% AP on COCO at 300+ FPS (NVIDIA 3090)
- **RTX 2080 Ti Estimate**: ~200-250 FPS (scaled)
- **Why**: Best speed/accuracy balance for real-time robotics
- **Variants**: RTMDet-Tiny (fastest), RTMDet-Small (balanced), RTMDet-Large (most accurate)
- **Recommendation**: Start with RTMDet-Small

**B. Segmentation: Mobile-Seed**

- **Performance**: 23.9 FPS on RTX 2080 Ti (1024x2048 resolution)
- **Why**: Specifically designed for mobile robots, lightweight
- **Capability**: Simultaneous semantic segmentation + boundary detection
- **Alternative**: SAM 2 (if speed acceptable after optimization)

**C. Open-Vocabulary Grounding: Grounding DINO**

- **Performance**: 52.5% AP zero-shot on COCO
- **Why**: Solves the "YOLO handoff problem" - can detect any object from text
- **Latency**: ~100-200ms per frame (acceptable for initial detection, not real-time tracking)
- **Strategy**: Use for initial object grounding, then handoff to tracker

### 1.2 The "YOLO Handoff Problem" - SOLVED

**Problem**: YOLO only tracks pre-trained classes. What if object isn't in COCO?

**Solution Architecture**:

1. **Initial Detection (VLM)**: Grounding DINO finds object from text prompt → outputs bounding box
2. **Handoff to Tracker**: Pass bounding box to **promptable tracker** (not YOLO)
3. **Real-Time Tracking**: Use **SAM 2 Video** or **Bot-SORT** or **ByteTrack** for frame-to-frame tracking
4. **Re-grounding**: If tracker loses object, re-run Grounding DINO (every 5-10 seconds)

**Key Insight**: Don't use YOLO for tracking. Use a **class-agnostic tracker** that tracks appearance/motion, not object class.

### 1.3 Model Performance on RTX 2080 Ti

| Model | Task | FPS (Est.) | VRAM | Use Case |

|-------|------|------------|------|----------|

| RTMDet-Small | Detection | 200-250 | ~2GB | Real-time detection |

| Grounding DINO | Grounding | 5-10 | ~4GB | Initial object finding |

| SAM 2 | Segmentation | 20-30 | ~6GB | High-quality masks |

| Mobile-Seed | Segmentation | 24 | ~3GB | Real-time segmentation |

| SAM 3 | Video Seg | 1-5 | ~8GB | Too slow for real-time |

**Recommendation**: RTMDet + Mobile-Seed + Grounding DINO combination fits comfortably in 11GB VRAM.

## 2. Vision-Language Models (VLMs) for Object Grounding

### 2.1 VLM Selection

**PaliGemma Analysis**:

- **Pros**: Strong vision-language understanding
- **Cons**: Heavy, may be slow on RTX 2080 Ti
- **Verdict**: Good for initial detection, not real-time control

**Better Alternatives**:

- **Grounding DINO**: Specifically designed for open-vocabulary detection
- **Grounded SAM 2**: Combines grounding + segmentation
- **OpenVLA**: For end-to-end vision-language-action (if going full VLA route)

### 2.2 Direct VLM Control - Feasibility

**Question**: Can VLM directly control motors in real-time?

**Answer**: **No, too slow**. VLMs run at 5-15 FPS, but visual servoing needs 30+ FPS.

**Correct Architecture**:

- **VLM (5-15 FPS)**: Finds object once, outputs bounding box
- **Tracker (30+ FPS)**: Tracks that bounding box frame-to-frame
- **Controller (30+ FPS)**: Generates motor commands from tracker output

**Exception**: If using **Vision-Language-Action (VLA)** models like RT-2 or OpenVLA, they can output actions directly, but still need fast inference for real-time.

## 3. Reinforcement Learning for Cinematic Shots

### 3.1 RL Application Strategy

**Two RL Use Cases Identified**:

**A. Motor Smoothness (Original Idea)**

- **Goal**: Learn to overcome jerky DC motor behavior
- **Method**: Train in Isaac Sim to learn power ramping
- **Reward**: Smooth camera movement (low acceleration, no jitter)
- **Complexity**: Medium - requires accurate physics simulation

**B. Cinematic Shot Planning (Your Idea)**

- **Goal**: Keep object in frame + execute predefined shot types
- **Method**: RL policy takes object position + shot type → motor commands
- **Reward**: Object stays centered, shot type executed correctly
- **Complexity**: High - requires defining shot types and rewards

### 3.2 RL for First Iteration - RECOMMENDATION

**Start WITHOUT RL**:

1. Use **classical control** (PID) for visual servoing
2. Use **predefined shot trajectories** (hard-coded paths)
3. Add RL in iteration 2 after baseline works

**Why**: RL adds complexity. First iteration should prove the concept works with simpler methods.

**When to Add RL**:

- After basic tracking works
- To learn motor smoothness (easier than shot planning)
- To adapt to different surfaces (carpet vs hardwood)

### 3.3 NVIDIA Isaac Sim for Training

**Feasibility**: Yes, Isaac Sim runs on RTX 2080 Ti

**Use Case**: Train motor smoothness policy, not shot planning (too complex initially)

**Timeline**: Post-MVP feature

## 4. Agentic AI Architecture

### 4.1 Director Agent Architecture

**Three-Layer Hierarchy** (from brainstorming):

**Layer 1: Director (VLM Agent)**

- **Input**: Voice/text command ("Orbit the cat")
- **Task**: Ground object, identify shot type
- **Frequency**: Once at start, or when tracker loses object
- **Models**: Grounding DINO + LLM for command parsing

**Layer 2: Cinematographer (Motion Planner)**

- **Input**: Object coordinates + shot type
- **Task**: Calculate trajectory (e.g., circular path for orbit)
- **Output**: High-level commands (JSON: `{"action": "orbit", "radius": 1.0}`)
- **Implementation**: Rule-based initially, RL later

**Layer 3: Pilot (Controller)**

- **Input**: High-level commands from Layer 2
- **Task**: Generate motor PWM signals
- **Implementation**: PID controller + motor smoothness (RL later)

### 4.2 Agent Frameworks

**LangChain/AutoGPT**: Overkill for first iteration

**Recommendation**: Custom Python agent with modular design

- Use LangChain only if you want LLM integration for command parsing
- For first iteration: Simple state machine is sufficient

## 5. Visual Servoing Architecture

### 5.1 Movement Hierarchy (Confirmed)

**Tier 1: Servo (Fine Control)**

- **Range**: ±90° from center
- **Speed**: Fast, smooth
- **Use**: Small object movements (< 20% of frame width)
- **Priority**: Always try servo first

**Tier 2: Wheels (Coarse Control)**

- **Trigger**: When servo at limit OR object moving fast
- **Use**: Large movements, rotation
- **Priority**: Secondary to servo

**Tier 3: Depth (Z-Axis)**

- **Method**: Bounding box area analysis
- **Logic**:
  - Box getting bigger → object closer → drive backward
  - Box getting smaller → object farther → drive forward
- **Goal**: Maintain desired shot size (close-up, medium, wide)

### 5.2 Control Algorithm

**Recommended**: **PID Controller** for each axis

- **X-axis**: Servo pan (left/right)
- **Y-axis**: (Future: tilt if adding second servo)
- **Z-axis**: Forward/backward movement
- **Theta**: Rotation (if needed)

**Error Calculation**:

- **X-error**: `object_center_x - frame_center_x`
- **Z-error**: `current_box_area - desired_box_area`

## 6. Cinematic Shot Types

### 6.1 Professional Shot Types for Cinema Bot

**Basic Shots (First Iteration)**:

1. **Follow/Track**: Keep object centered (default)
2. **Dolly In/Out**: Drive straight toward/away from object
3. **Orbit**: Circle around object while keeping it centered
4. **Pan**: Servo-only horizontal movement

**Advanced Shots (Future)**:

5. **Dolly Zoom**: Drive back + digital zoom in (Vertigo effect)
6. **Reveal**: Drive sideways while panning to reveal object
7. **Lead**: Keep object at edge of frame (rule of thirds)

### 6.2 Shot Implementation

**For First Iteration**: Hard-code trajectories

- **Orbit**: Calculate circular path, adjust servo to keep object centered
- **Dolly**: Drive forward/backward at constant speed
- **Follow**: Standard visual servoing

**For Future**: RL can learn optimal trajectories for each shot type

## 7. First Iteration Architecture (Simplified)

### 7.1 Core Components (MVP)

**Must Have**:

1. **Object Detection**: RTMDet-Small (real-time)
2. **Object Grounding**: Grounding DINO (initial detection)
3. **Object Tracking**: Bot-SORT or ByteTrack (frame-to-frame)
4. **Visual Servoing**: PID controller
5. **Serial Communication**: PC → Arduino (existing `commander.py` pattern)
6. **Camera Stream**: Phone → PC via Wi-Fi (IP Webcam/DroidCam)

**Nice to Have (Iteration 2)**:

- Voice commands (Whisper)
- RL for motor smoothness
- Multiple shot types
- Depth control

**Skip for Now**:

- Full RL shot planning
- Complex agent frameworks
- Multiple VLMs
- SAM 3 (too slow)

### 7.2 Software Architecture

```
┌─────────────────────────────────────────┐
│         PC (RTX 2080 Ti)                │
│                                         │
│  ┌─────────────────────────────────┐  │
│  │   Camera Stream (OpenCV)         │  │
│  │   Phone → Wi-Fi → PC             │  │
│  └──────────────┬──────────────────┘  │
│                 │                      │
│  ┌──────────────▼──────────────────┐  │
│  │   Object Detection Pipeline     │  │
│  │   - Grounding DINO (once)       │  │
│  │   - RTMDet (real-time)          │  │
│  │   - Bot-SORT (tracking)         │  │
│  └──────────────┬──────────────────┘  │
│                 │                      │
│  ┌──────────────▼──────────────────┐  │
│  │   Visual Servoing Controller    │  │
│  │   - PID for X-axis (servo)      │  │
│  │   - PID for Z-axis (wheels)    │  │
│  │   - Error calculation           │  │
│  └──────────────┬──────────────────┘  │
│                 │                      │
│  ┌──────────────▼──────────────────┐  │
│  │   Command Generator              │  │
│  │   - Serial protocol              │  │
│  │   - M:L:speed, M:R:speed         │  │
│  │   - S:angle (servo)              │  │
│  └──────────────┬──────────────────┘  │
└─────────────────┼──────────────────────┘
                  │ USB Serial
                  ▼
┌─────────────────────────────────────────┐
│         Arduino Uno                      │
│  - Motor control (L298N)                │
│  - Servo control                        │
│  - Command listener                     │
└─────────────────────────────────────────┘
```

### 7.3 Data Flow

1. **User Input**: Text command ("Track the red cup")
2. **Initial Grounding**: Grounding DINO finds "red cup" → bounding box
3. **Tracking Loop** (30 FPS):

   - RTMDet detects objects
   - Bot-SORT tracks the "red cup" bounding box
   - Calculate error (object center vs frame center)
   - PID controller generates motor commands
   - Send commands to Arduino via serial

4. **Re-grounding**: If tracker confidence < threshold, re-run Grounding DINO

## 8. Hardware-Specific Considerations

### 8.1 RTX 2080 Ti Constraints

**VRAM**: 11GB - sufficient for:

- RTMDet: ~2GB
- Grounding DINO: ~4GB
- Bot-SORT: <1GB
- **Total**: ~7GB (comfortable margin)

**Compute**: Can handle 30 FPS processing with RTMDet + tracking

### 8.2 Latency Budget

**Target**: <100ms total latency (camera → processing → motor command)

**Breakdown**:

- Network (phone → PC): 20-50ms
- Detection (RTMDet): 5-10ms
- Tracking (Bot-SORT): 1-2ms
- Control (PID): <1ms
- Serial (PC → Arduino): 1-5ms
- **Total**: ~30-70ms (acceptable)

### 8.3 Arduino Communication Protocol

**Recommended Format**: Simple string protocol

- `M:F:200` - Move Forward at speed 200
- `M:B:150` - Move Backward at speed 150
- `M:L:180` - Turn Left at speed 180
- `M:R:180` - Turn Right at speed 180
- `S:1500` - Set Servo to 1500 microseconds (90°)
- `STOP` - Emergency stop

**Arduino Firmware**: Fixed listener (no code generation needed)

## 9. Research Gaps & Future Work

### 9.1 Questions Requiring Further Research

1. **SAM 3 Performance**: Need actual benchmarks on RTX 2080 Ti
2. **VLA Models**: Evaluate RT-2 vs OpenVLA for this use case
3. **RL Training Time**: How long to train motor smoothness policy in Isaac Sim?
4. **Edge Deployment**: Model distillation strategies for Jetson/Orange Pi

### 9.2 Recommended Next Steps

1. **Benchmark Models**: Test RTMDet, Grounding DINO, Mobile-Seed on actual hardware
2. **Prototype Tracking**: Implement Bot-SORT with RTMDet
3. **Build MVP**: Object detection → tracking → basic visual servoing
4. **Iterate**: Add shot types, then RL, then voice commands

## 10. Key Recommendations Summary

### For First Iteration:

✅ **Use**: RTMDet-Small + Grounding DINO + Bot-SORT + PID controller

✅ **Skip**: RL, complex agents, SAM 3, multiple VLMs

✅ **Architecture**: Simple 3-layer (Director → Cinematographer → Pilot)

✅ **Shot Types**: Follow, Dolly, Orbit (hard-coded)

### For Future Iterations:

- Add RL for motor smoothness
- Add more shot types
- Add voice commands (Whisper)
- Consider VLA models for end-to-end control
- Explore model distillation for edge deployment

### Critical Success Factors:

1. **Real-time performance**: Must maintain 30 FPS
2. **Robust tracking**: Handle occlusion, fast movement
3. **Smooth control**: Minimize jitter in footage
4. **Modular design**: Easy to add features later
