---
name: Cinema Bot AI Architecture Research & Planning
overview: Comprehensive research and architecture plan for the Cinema Bot AI system, addressing all questions raised in the brainstorming document with evidence-based recommendations for model selection, architecture design, and implementation strategy.
todos:
  - id: research-validation
    content: Validate model performance benchmarks on RTX 2080 Ti hardware (Grounding DINO, SAM 2, BoT-SORT, RTMDet)
    status: pending
  - id: video-pipeline
    content: "Implement video streaming pipeline: phone → Wi-Fi → OpenCV (720p @ 30fps)"
    status: pending
  - id: arduino-firmware
    content: Develop Arduino firmware for motor/servo control with serial command interface
    status: pending
  - id: object-detection
    content: Integrate Grounding DINO for initial object detection from text commands
    status: pending
    dependencies:
      - video-pipeline
  - id: segmentation-tracking
    content: Integrate SAM 2 for segmentation and BoT-SORT for real-time tracking (30 FPS)
    status: pending
    dependencies:
      - object-detection
  - id: visual-servoing
    content: "Implement visual servoing control: PID controller for servo panning based on tracked object position"
    status: pending
    dependencies:
      - segmentation-tracking
      - arduino-firmware
  - id: basic-shots
    content: "Implement basic cinematic shots: orbit, dolly in/out, tracking shot with rule-based trajectory planning"
    status: pending
    dependencies:
      - visual-servoing
  - id: motor-control
    content: Add wheel control for coarse tracking when servo reaches limits, implement depth control based on bounding box size
    status: pending
    dependencies:
      - visual-servoing
---

# Cinema Bot AI Architecture: Research & Implementation Plan

## Executive Summary

This plan addresses all technical questions raised in the brainstorming document, providing research-backed recommendations for building an AI-powered cinema robot. The architecture is designed for the first iteration (simple, functional) while remaining extensible for future enhancements.

## Hardware Specifications (Corrected)

- **GPU**: RTX 2080 Ti (11GB VRAM) - corrected from RTX 2070 Ti
- **Controller**: Arduino Uno
- **Drivetrain**: 4WD DC motors via L298N driver
- **Camera Pan**: 180° Servo (FS90)
- **Camera**: Smartphone streaming via Wi-Fi
- **Power**: 6-AA battery pack

## 1. Model Selection: Real-Time Video Processing

### 1.1 Object Detection & Grounding

**Research Findings:**

**Primary Recommendation: RTMDet + Grounding DINO**

- **RTMDet**: Achieves 52.8% AP on COCO at 300+ FPS on RTX 3090 (scaled to ~200-250 FPS on RTX 2080 Ti)
- **Grounding DINO**: Zero-shot open-vocabulary detection, 52.5% AP on COCO
- **Why this combination**: RTMDet provides speed for tracking, Grounding DINO provides initial object grounding from natural language

**Alternative Consideration: YOLOv10-X**

- 58.1% mAP at 245 FPS on RTX 4090 (likely ~180-200 FPS on RTX 2080 Ti)
- More accurate but slightly slower than RTMDet
- Well-established ecosystem

**Verdict**: Start with **RTMDet** for first iteration (simpler, faster). Consider YOLOv10-X if accuracy becomes limiting.

### 1.2 Video Segmentation & Tracking

**Research Findings:**

**Primary Recommendation: SAM 2 (not SAM 3)**

- **SAM 2**: Processes images in 47ms, handles video natively with cross-frame attention
- **SAM 3**: More recent but heavier; may struggle with real-time on RTX 2080 Ti
- **Why SAM 2**: Proven real-time performance, native video support, better documented

**Alternative: Grounded SAM 2**

- Combines SAM 2 with Florence-2 for text-based segmentation
- Zero-shot segmentation based on textual prompts
- More flexible but potentially slower

**For Tracking: BoT-SORT or ByteTrack**

- **BoT-SORT**: State-of-the-art multi-object tracking, handles occlusions well
- **ByteTrack**: Fast, efficient, good for real-time applications
- **Why these over YOLO tracking**: YOLO is class-limited; these work with any detected object

**Verdict**: Use **SAM 2** for segmentation, **BoT-SORT** for tracking. This solves the "YOLO class limitation" problem.

### 1.3 Vision-Language Models (VLMs)

**Research Findings:**

**Primary Recommendation: Grounding DINO (not PaliGemma/LLaVA for initial detection)**

- **Grounding DINO**: Specifically designed for visual grounding, faster than general VLMs
- **PaliGemma**: Good for general VLM tasks but slower (5-15 FPS typical)
- **LLaVA-1.5**: More general-purpose, slower than Grounding DINO for detection tasks
- **Moondream2**: Lighter but less accurate than alternatives

**For High-Level Planning: Consider VLA Models**

- **OpenVLA**: Open-source Vision-Language-Action model
- **RT-2**: Google's VLA model (proprietary)
- **Why VLA**: Can translate "orbit the cat" directly into action sequences

**Verdict**: Use **Grounding DINO** for initial object detection from text. Consider **OpenVLA** for shot planning in future iterations.

## 2. Architecture Design: Solving the Key Questions

### 2.1 The "Open-Vocabulary Handoff" Problem

**Question**: "How does handoff to YOLO work if object isn't in YOLO's classes?"

**Solution**: **Don't use YOLO for tracking. Use SAM 2 + BoT-SORT.**

**Workflow**:

1. **Grounding DINO** (Layer 1): Takes text "Find the vintage blue typewriter" → outputs bounding box [x, y, w, h]
2. **SAM 2** (Layer 2): Takes bounding box → generates pixel-level mask
3. **BoT-SORT** (Layer 3): Tracks the mask across frames at 30+ FPS
4. **Result**: Any object can be tracked, regardless of class

**Why this works**: SAM 2 doesn't care about object class - it tracks pixels. BoT-SORT tracks based on appearance and motion, not semantic class.

### 2.2 VLM Direct Control: Too Slow?

**Question**: "Can VLM directly control motors, or is it too slow?"

**Answer**: **Too slow for direct control, but perfect for initial detection.**

**Architecture**:

- **VLM (Grounding DINO)**: Runs once at start (or every 5-10 seconds if object lost)
- **Fast Tracker (BoT-SORT)**: Runs at 30 FPS for continuous tracking
- **Control Loop**: Uses tracker output, not VLM output

**Performance**:

- Grounding DINO: ~100-200ms per frame (5-10 FPS)
- BoT-SORT: ~33ms per frame (30 FPS)
- **Verdict**: VLM finds object once, tracker maintains it

### 2.3 Handling Moving Objects

**Question**: "How does system behave if object keeps moving?"

**Solution**: **Multi-Layer Tracking with Predictive Control**

**Architecture**:

1. **BoT-SORT** tracks object at 30 FPS
2. **Kalman Filter** predicts object position (handles occlusions)
3. **PID Controller** calculates motor commands based on predicted position
4. **Visual Servoing**: Servo responds immediately, wheels respond if servo reaches limit

**If Object Moves Fast**:

- Tracker maintains lock (BoT-SORT handles fast motion)
- PID controller increases motor speed proportionally
- System adapts in real-time

**If Object Lost**:

- Re-run Grounding DINO to re-acquire
- Resume tracking with BoT-SORT

### 2.4 RL for Cinematic Shots vs Motor Smoothness

**Question**: "Should RL be for shot planning or motor control?"

**Answer**: **Both, but prioritize motor smoothness for first iteration.**

**Phase 1 (First Iteration): RL for Motor Smoothness**

- **Problem**: DC motors are jerky
- **Solution**: Train RL policy to learn power ramp-up curves
- **Training**: NVIDIA Isaac Sim simulation
- **Output**: Smooth motor commands that compensate for friction

**Phase 2 (Future): RL for Shot Planning**

- **Problem**: Optimal camera movements for cinematic shots
- **Solution**: Train RL agent to learn shot compositions
- **Input**: Object position, shot type ("orbit", "dolly", etc.)
- **Output**: Trajectory planning for wheels + servo

**Verdict**: Start with motor smoothness RL (immediate value). Add shot planning RL later.

## 3. Cinematic Shots: Types & Implementation

### 3.1 Common Cinematic Shots

**Research-Based Shot Types**:

1. **Dolly In/Out**

   - Motion: Drive straight toward/away from subject
   - Servo: Locked forward
   - Effect: Dramatic focus, subject size changes

2. **Orbit/Circle**

   - Motion: Circular path around subject
   - Servo: Continuously pans to keep subject centered
   - Effect: Dynamic "hero shot", parallax

3. **Dolly Zoom (Vertigo Effect)**

   - Motion: Drive backward
   - Digital: Zoom in simultaneously
   - Effect: Subject stays same size, background compresses

4. **Reveal**

   - Motion: Drive sideways (strafing)
   - Servo: Pan from 0° to target
   - Effect: Reveals object from behind obstacle

5. **Tracking Shot**

   - Motion: Follow subject's movement
   - Servo: Fine adjustments to keep centered
   - Effect: Smooth following, subject stays framed

### 3.2 Implementation Strategy

**First Iteration: Rule-Based Shot Execution**

- Hard-code shot trajectories (no RL yet)
- Input: Shot type + object position
- Output: Pre-calculated path for wheels + servo
- **Why**: Simple, predictable, debuggable

**Future Iteration: RL-Based Shot Planning**

- Train RL agent to learn optimal trajectories
- Reward: Framing quality, smoothness, cinematic appeal
- **Why**: Can learn complex, adaptive shots

## 4. System Architecture

### 4.1 Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Layer 1: Director (VLM Agent) - "What"                  │
│ - Grounding DINO: Initial object detection              │
│ - Frequency: Once at start, or when object lost        │
│ - Output: Bounding box [x, y, w, h]                     │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼──────────────────────────────────┐
│ Layer 2: Cinematographer (Motion Planner) - "How"      │
│ - SAM 2: Generate mask from bounding box                │
│ - BoT-SORT: Track mask at 30 FPS                        │
│ - Shot Planner: Calculate trajectory for shot type     │
│ - Output: High-level commands {"action": "orbit", ...}  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼──────────────────────────────────┐
│ Layer 3: Pilot (Controller) - "Reflex"                 │
│ - PID Controller: Convert trajectory to motor commands │
│ - RL Policy (optional): Smooth motor control           │
│ - Visual Servoing: Servo first, wheels if needed       │
│ - Output: Serial commands to Arduino                   │
└────────────────────┬───────────────────────────────────┘
                     │
┌────────────────────▼──────────────────────────────────┐
│ Hardware: Arduino + Motors + Servo                     │
└────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow

1. **Video Stream**: Phone → Wi-Fi → PC (OpenCV, 720p @ 30fps)
2. **Object Detection**: Grounding DINO processes frame → bounding box
3. **Segmentation**: SAM 2 generates mask from bounding box
4. **Tracking**: BoT-SORT tracks mask across frames
5. **Planning**: Shot planner calculates trajectory
6. **Control**: PID controller generates motor commands
7. **Execution**: Serial commands sent to Arduino

### 4.3 Communication Protocol

**Arduino Firmware** (Fixed, uploaded once):

- Listens for serial commands
- Format: `M:F:200` (Move Forward speed 200) or `S:1500` (Servo angle 1500μs)
- Responds with "ok" after execution

**Python Agent** (PC):

- Calculates commands based on tracking
- Sends formatted strings via serial
- Handles acknowledgments

## 5. First Iteration: Simplified Architecture

### 5.1 Core Components (MVP)

1. **Video Pipeline**

   - Phone streaming (IP Webcam/DroidCam)
   - OpenCV frame capture
   - 720p @ 30fps

2. **Object Detection**

   - Grounding DINO for initial detection
   - Text input: "Find the [object]"

3. **Tracking**

   - SAM 2 for segmentation
   - BoT-SORT for tracking
   - 30 FPS tracking loop

4. **Control**

   - Simple PID controller
   - Visual servoing (servo first, wheels if needed)
   - Basic shot types (orbit, dolly, track)

5. **Hardware Interface**

   - Arduino firmware for motor/servo control
   - Serial communication (115200 baud)

### 5.2 What to Defer

- **RL for motor smoothness**: Add after basic tracking works
- **RL for shot planning**: Add after basic shots work
- **VLA models**: Add after basic VLM integration works
- **Advanced shot types**: Add after basic shots work

### 5.3 Success Criteria

- Robot can find object from text command
- Robot can track object at 30 FPS
- Robot can execute basic shots (orbit, dolly, track)
- Footage is stable (no excessive jitter)

## 6. Edge-Ready Architecture (Future)

### 6.1 Design for Portability

**Current (Prototype)**:

- Heavy models on PC (RTX 2080 Ti)
- Wi-Fi communication
- USB serial to Arduino

**Future (Edge Deployment)**:

- Distill models to smaller versions
- Deploy on Jetson Nano/Orange Pi
- Same architecture, different hardware

**Key**: Keep layers modular and hardware-agnostic

### 6.2 Model Distillation Strategy

1. **Teacher-Student Distillation**:

   - Train small model to mimic large model
   - Reduce model size by 10-100x
   - Maintain 80-90% of accuracy

2. **Quantization**:

   - Convert FP32 → INT8
   - 4x size reduction
   - Minimal accuracy loss

3. **Pruning**:

   - Remove unnecessary weights
   - Further size reduction

## 7. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

- [ ] Set up video streaming pipeline
- [ ] Integrate Grounding DINO
- [ ] Integrate SAM 2 + BoT-SORT
- [ ] Basic Arduino firmware
- [ ] Simple PID controller

### Phase 2: Basic Tracking (Week 3-4)

- [ ] Object detection from text
- [ ] Real-time tracking at 30 FPS
- [ ] Visual servoing (servo control)
- [ ] Basic wheel control

### Phase 3: Cinematic Shots (Week 5-6)

- [ ] Implement orbit shot
- [ ] Implement dolly shot
- [ ] Implement tracking shot
- [ ] Test and refine

### Phase 4: Enhancement (Week 7+)

- [ ] RL for motor smoothness
- [ ] Additional shot types
- [ ] VLA integration for shot planning
- [ ] Performance optimization

## 8. Key Technical Decisions

### 8.1 Model Choices Summary

| Component | Model | Rationale |

|----------|-------|-----------|

| Initial Detection | Grounding DINO | Zero-shot, open-vocabulary |

| Segmentation | SAM 2 | Real-time, video-native |

| Tracking | BoT-SORT | Handles occlusions, fast |

| Motor Control | PID + (future) RL | Simple first, RL later |

### 8.2 Architecture Principles

1. **Modularity**: Each layer independent, replaceable
2. **Simplicity**: Start simple, add complexity incrementally
3. **Performance**: Real-time (30 FPS) is non-negotiable
4. **Extensibility**: Design for future enhancements

## 9. Research-Backed Recommendations

### 9.1 Model Performance Estimates (RTX 2080 Ti)

- **Grounding DINO**: ~100-200ms/frame (5-10 FPS) - acceptable for initial detection
- **SAM 2**: ~47ms/frame (21 FPS) - good for segmentation
- **BoT-SORT**: ~10-20ms/frame (50-100 FPS) - excellent for tracking
- **RTMDet**: ~4-5ms/frame (200+ FPS) - excellent for detection

### 9.2 Why Not PaliGemma/SAM3 for First Iteration

- **PaliGemma**: General-purpose VLM, slower than Grounding DINO for detection
- **SAM 3**: Newer but heavier, may not achieve real-time on RTX 2080 Ti
- **Verdict**: Use proven, fast models for first iteration

### 9.3 Why Not YOLO for Tracking

- **YOLO**: Class-limited (80 classes), can't track arbitrary objects
- **BoT-SORT**: Class-agnostic, tracks any detected object
- **Verdict**: BoT-SORT solves the open-vocabulary problem

## 10. Next Steps

1. **Validate Model Performance**: Benchmark Grounding DINO, SAM 2, BoT-SORT on RTX 2080 Ti
2. **Implement Video Pipeline**: Set up phone streaming, OpenCV capture
3. **Build Arduino Firmware**: Motor/servo control with serial interface
4. **Create Basic Tracking Loop**: Grounding DINO → SAM 2 → BoT-SORT
5. **Implement Visual Servoing**: Servo control based on tracking
6. **Add Basic Shots**: Orbit, dolly, tracking shots

This plan provides a clear, research-backed path forward while keeping the first iteration simple and achievable.
