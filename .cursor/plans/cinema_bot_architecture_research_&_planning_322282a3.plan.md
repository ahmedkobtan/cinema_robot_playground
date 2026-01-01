---
name: Cinema Bot Architecture Research & Planning
overview: Comprehensive research and architecture planning for the AI-powered cinema robot, addressing all technical questions raised in the brainstorming document with evidence-based recommendations for model selection, system architecture, and implementation strategy.
todos:
  - id: research-models
    content: Research and document best real-time video detection/segmentation/grounding models for RTX 2080 Ti
    status: pending
  - id: solve-handoff
    content: Design solution for VLM→Tracker handoff that solves YOLO class limitation problem
    status: pending
  - id: analyze-vlm-performance
    content: Analyze real-time VLM performance and determine if direct motor control is feasible
    status: pending
  - id: design-architecture
    content: Design simplified first-iteration architecture balancing simplicity and functionality
    status: pending
    dependencies:
      - research-models
      - solve-handoff
  - id: define-cinema-shots
    content: Research and document implementable cinematic shot types with motion specifications
    status: pending
  - id: plan-rl-usage
    content: "Determine RL application: motion smoothing vs shot planning, and when to implement"
    status: pending
  - id: create-implementation-phases
    content: Break down implementation into manageable phases with clear success criteria
    status: pending
    dependencies:
      - design-architecture
---

# Cinema Bot Architecture: In-Depth Research & Technical Plan

## Executive Summary

This plan addresses all technical questions and concerns raised in the brainstorming phase, providing research-backed recommendations for building an AI-powered cinema robot. The architecture prioritizes simplicity for the first iteration while maintaining a clear path to advanced features.

## 1. Model Selection: Real-Time Video Detection/Segmentation/Grounding

### Research Findings

**Current State (2024-2025):**

1. **RTMDet** (OpenMMLab) - **RECOMMENDED FOR FIRST ITERATION**

   - Performance: 52.8% AP on COCO at 300+ FPS on RTX 3090
   - RTX 2080 Ti equivalent: ~250-280 FPS (sufficient headroom)
   - Variants: RTMDet-Tiny (1020+ FPS), RTMDet-Small (819 FPS)
   - **Why it wins**: Best speed/accuracy tradeoff for real-time tracking
   - **Use case**: Primary object detection and tracking

2. **YOLOv12** (Latest, Feb 2025)

   - Performance: 40.6% mAP at 1.64ms latency on T4 GPU
   - Features: Area Attention, FlashAttention modules
   - **Consideration**: Newer but may have compatibility issues

3. **Grounding DINO** - **RECOMMENDED FOR INITIAL DETECTION**

   - Performance: 52.5% AP on COCO (zero-shot)
   - **Why it wins**: Open-vocabulary detection solves the "YOLO class limitation" problem
   - **Use case**: Initial object grounding from natural language

4. **SAM 2 / SAM 3** - **CONDITIONAL RECOMMENDATION**

   - SAM 2: Video propagation with cross-frame attention
   - SAM 3: Enhanced version with better temporal consistency
   - Performance: ~30 FPS on RTX 2080 Ti (acceptable for tracking)
   - **Why conditional**: Excellent for segmentation but may be overkill for first iteration
   - **Alternative**: FastSAM or MobileSAM for lighter weight

5. **Follow Anything (FAn)** - **STRONG CANDIDATE**

   - Open-vocabulary, multimodal detection/tracking
   - Real-time capable on 6-8GB VRAM (fits RTX 2080 Ti)
   - Supports text, image, or click queries
   - **Why it's interesting**: Solves VLM→Tracker handoff elegantly

6. **Grounded SAM 2** (Meta AI, July 2024)

   - Integrates Florence-2 with SAM 2
   - Zero-shot segmentation from text prompts
   - **Consideration**: Newer, less battle-tested

### Recommendation for First Iteration

**Phase 1 (Simplest):**

- **Grounding DINO** for initial object detection (once per command)
- **RTMDet-Small** for real-time tracking (30 FPS loop)
- **Rationale**: Solves open-vocabulary problem, maintains real-time performance

**Phase 2 (Enhanced):**

- Add **SAM 2** for pixel-level segmentation (smoother centroid calculation)
- Or evaluate **Follow Anything (FAn)** as unified solution

## 2. Solving the VLM→Tracker Handoff Problem

### The Core Issue

**Problem**: YOLO only tracks pre-trained classes. If VLM finds "vintage blue typewriter," YOLO can't track it.

### Solutions (Ranked by Simplicity)

**Solution 1: Promptable Trackers (RECOMMENDED)**

- **SAM 2/SAM 3**: Feed bounding box from VLM as prompt
- SAM tracks pixels, not classes - works for any object
- Performance: 30+ FPS on RTX 2080 Ti
- **Workflow**: VLM finds box → SAM tracks mask → Calculate centroid from mask

**Solution 2: Appearance-Based Tracking**

- **ByteTrack** or **Bot-SORT**: Track by visual appearance, not class
- Feed initial bounding box from VLM
- Tracks based on visual features (color, texture, shape)
- Performance: 30+ FPS, very lightweight

**Solution 3: Unified Models**

- **Follow Anything (FAn)**: Handles detection AND tracking in one model
- Open-vocabulary by design
- Eliminates handoff complexity entirely

### Recommended Approach

**First Iteration:**

```
Grounding DINO (once) → Bounding Box → ByteTrack/Bot-SORT (30 FPS)
```

**Why**: Simple, proven, real-time capable. ByteTrack tracks by appearance, not class.

**Future Enhancement:**

```
Grounding DINO (once) → Bounding Box → SAM 2 (30 FPS) → Mask-based centroid
```

**Why**: Pixel-level segmentation provides smoother tracking for cinematic shots.

## 3. Vision-Language Models (VLMs) for Object Grounding

### Performance Analysis

**Real-Time VLMs (5-15 FPS):**

- **Moondream2**: ~10-15 FPS on RTX 2080 Ti
- **LLaVA-1.5**: ~5-8 FPS on RTX 2080 Ti
- **PaliGemma**: ~8-12 FPS on RTX 2080 Ti

**Conclusion**: Too slow for direct motor control (5-15 FPS = 200-66ms latency)

### Recommended Architecture

**Two-Stage Approach:**

1. **VLM Stage** (Runs once per command, ~100-200ms):

   - Grounding DINO or PaliGemma finds object
   - Outputs: Bounding box coordinates
   - Frequency: Once at start, or when tracker loses object

2. **Tracking Stage** (Runs 30 FPS, ~33ms):

   - Fast tracker (RTMDet, ByteTrack, SAM 2) maintains lock
   - Outputs: Updated coordinates every frame
   - Frequency: Continuous at 30 FPS

**This solves**: VLM provides intelligence, tracker provides speed.

## 4. Real-Time VLM Agent: Feasibility

### Direct VLM Control: NOT RECOMMENDED

**Why**:

- 5-15 FPS = 200-66ms per decision
- Robot moves between frames → overshooting
- Object moves → VLM can't react fast enough

### Hybrid Approach: RECOMMENDED

**Architecture:**

```
Voice Command → VLM (Grounding) → Fast Tracker (30 FPS) → Motor Control
     ↓              ↓                    ↓                    ↓
  Once          ~200ms              ~33ms/frame          ~10ms
```

**VLM Role**: High-level "what to track" decision

**Tracker Role**: Low-level "where is it now" updates

**Controller Role**: "how to move" based on tracker output

## 5. Handling Moving Objects

### Challenge Analysis

**Scenario**: Object moves faster than robot can react

### Solution: Predictive Control

**Current Position Tracking:**

- Tracker provides (x, y) every 33ms
- Calculate velocity: v = (x_t - x_{t-1}) / dt
- Predict future position: x_pred = x + v * t_horizon

**Adaptive Control:**

- If object moving fast → increase motor speed
- If object moving slow → decrease motor speed
- If object stationary → maintain position

**Implementation**: PID controller with velocity feedforward

### Tracker Loss Handling

**When tracker loses object:**

1. Re-run VLM (Grounding DINO) to re-acquire
2. If VLM can't find → alert user or return to last known position
3. Implement "search pattern" (spiral, grid) if needed

## 6. Reinforcement Learning: Motion Smoothing vs Shot Planning

### Two Distinct RL Applications

**Application 1: Motion Smoothing (RECOMMENDED FOR FIRST ITERATION)**

- **Goal**: Learn motor control to overcome jerky DC motors
- **Input**: Desired velocity, current state
- **Output**: Motor PWM values (ramp up/down)
- **Training**: Isaac Sim simulation of robot on carpet
- **Reward**: Smoothness of camera footage (low acceleration)

**Application 2: Shot Planning (FUTURE ENHANCEMENT)**

- **Goal**: Learn optimal camera movements for cinematic shots
- **Input**: Object position, shot type, scene context
- **Output**: Trajectory (path + timing)
- **Training**: Simulated filming scenarios with aesthetic rewards
- **Reward**: Shot quality metrics (framing, stability, composition)

### Recommendation

**First Iteration**: Skip RL entirely, use PID control

- **Why**: Simpler, more predictable, easier to debug
- **When to add RL**: After basic tracking works, for motion smoothing

**Future**: Add RL for shot planning after motion smoothing is solved

## 7. Cinema Shot Types: Professional Standards

### Common Cinematic Shots (Implementable)

**1. Dolly In/Out**

- Motion: Drive straight toward/away from subject
- Servo: Locked forward
- Effect: Dramatic focus, reveals context
- **Implementation**: Control forward/backward speed, maintain center framing

**2. Orbit (Circular Tracking)**

- Motion: Circle around subject
- Servo: Pan to keep subject centered
- Effect: Dynamic "hero shot," shows 3D space
- **Implementation**: Calculate circular path, coordinate wheels + servo

**3. Reveal (Lateral Movement)**

- Motion: Drive sideways (strafing)
- Servo: Pan from edge to center
- Effect: Reveals subject from behind obstacle
- **Implementation**: Lateral wheel control + servo sweep

**4. Dolly Zoom (Vertigo Effect)**

- Motion: Drive backward
- Digital: Zoom in simultaneously
- Effect: Subject stays same size, background expands
- **Implementation**: Coordinate backward motion + phone camera zoom API

**5. Tracking Shot**

- Motion: Follow subject's movement
- Servo: Pan to maintain framing
- Effect: Smooth following, maintains relationship
- **Implementation**: Visual servoing with PID control

### Shot Planning Logic

**Input**: Shot type + object position

**Output**: Trajectory (wheel speeds, servo angles over time)

**Example - Orbit:**

```python
def calculate_orbit(object_center, radius, speed):
    # Calculate circular path
    angles = np.linspace(0, 2*pi, num_frames)
    robot_positions = radius * [cos(θ), sin(θ)]

    # For each position, calculate:
    # - Wheel speeds to reach position
    # - Servo angle to point at object
    return trajectory
```

## 8. System Architecture: Simplified First Iteration

### Recommended Architecture (Minimal Viable)

```
┌─────────────────────────────────────────────────────────┐
│                    PC (RTX 2080 Ti)                      │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Whisper    │  │  Grounding   │  │   RTMDet     │  │
│  │ (Optional)   │→ │     DINO      │→ │  Tracker     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         │                 │                 │            │
│         └─────────────────┴─────────────────┘            │
│                          │                               │
│                   ┌──────▼──────┐                        │
│                   │   Director  │                        │
│                   │   Agent     │                        │
│                   │  (Python)   │                        │
│                   └──────┬──────┘                        │
└──────────────────────────┼──────────────────────────────┘
                           │ Serial (115200 baud)
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  Arduino Uno                            │
│  ┌──────────────┐  ┌──────────────┐                    │
│  │   Motor      │  │    Servo     │                    │
│  │  Controller  │  │   Control    │                    │
│  │  (L298N)     │  │   (FS90)     │                    │
│  └──────────────┘  └──────────────┘                    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Smartphone (Camera)                         │
│              Streams via Wi-Fi → PC                      │
└─────────────────────────────────────────────────────────┘
```

### Component Breakdown

**Layer 1: Perception (PC)**

- **Input**: Phone video stream (720p @ 30fps via IP Webcam)
- **Detection**: Grounding DINO (once per command)
- **Tracking**: RTMDet or ByteTrack (30 FPS continuous)
- **Output**: Object coordinates (x, y, width, height) every frame

**Layer 2: Decision (PC)**

- **Input**: Object coordinates + user command ("orbit", "dolly in", etc.)
- **Logic**: Calculate error (object center vs screen center)
- **Planning**: Generate trajectory for cinematic shot
- **Output**: Motor commands (forward/back/left/right speed, servo angle)

**Layer 3: Control (Arduino)**

- **Input**: Serial commands from PC
- **Protocol**: Simple string format ("M:F:200" = Move Forward speed 200)
- **Output**: PWM signals to L298N (motors) and servo

### Communication Protocol

**PC → Arduino:**

```
M:F:200    # Move Forward at speed 200 (0-255)
M:B:150    # Move Backward at speed 150
M:L:180    # Turn Left at speed 180
M:R:180    # Turn Right at speed 180
M:S        # Stop
S:1500     # Set Servo to 1500 microseconds (center)
```

**Arduino → PC:**

```
OK          # Command received and executed
ERROR:xxx   # Error message
```

## 9. Implementation Phases

### Phase 1: Basic Tracking (Week 1-2)

**Goal**: Track a manually selected object

**Components:**

- Phone video stream → OpenCV
- Manual object selection (click on screen)
- ByteTrack for tracking
- Simple PID controller for centering
- Basic Arduino firmware

**Success Criteria**: Robot keeps selected object centered in frame

### Phase 2: Voice Commands (Week 3)

**Goal**: Accept natural language commands

**Components:**

- Whisper for voice-to-text
- Grounding DINO for object detection
- Integration with Phase 1 tracker

**Success Criteria**: "Find the coffee mug" → Robot finds and tracks mug

### Phase 3: Cinematic Shots (Week 4-5)

**Goal**: Execute predefined shot types

**Components:**

- Shot planning logic (orbit, dolly, etc.)
- Trajectory generation
- Coordinated wheel + servo control

**Success Criteria**: "Orbit the mug" → Robot performs smooth circular motion

### Phase 4: Motion Smoothing (Week 6+)

**Goal**: Eliminate jerky movements

**Components:**

- RL training in Isaac Sim (optional)
- Or: Advanced PID with feedforward control
- Vibration damping (hardware)

**Success Criteria**: Smooth, professional-looking footage

## 10. Hardware-Specific Considerations

### RTX 2080 Ti (11GB VRAM)

- **Capacity**: Sufficient for all recommended models
- **Performance**:
  - RTMDet: ~250-280 FPS
  - Grounding DINO: ~10-15 FPS
  - SAM 2: ~30 FPS
  - Moondream2: ~10-15 FPS

### Arduino Uno Limitations

- **Processing**: No AI models, only motor control
- **Communication**: Serial at 115200 baud (sufficient for 30 FPS commands)
- **Latency**: ~10ms command execution (negligible)

### Phone Camera

- **Streaming**: IP Webcam or DroidCam (MJPEG/RTSP)
- **Resolution**: 720p recommended (balance quality/latency)
- **Latency**: ~50-100ms network + processing (acceptable)

## 11. Key Decisions Summary

### Model Selection

- **Initial Detection**: Grounding DINO (open-vocabulary)
- **Tracking**: RTMDet-Small or ByteTrack (real-time)
- **Future**: SAM 2 for segmentation (if needed)

### Architecture

- **VLM**: One-time use for object grounding
- **Tracker**: Continuous 30 FPS for position updates
- **Controller**: PID-based (RL optional later)

### First Iteration Scope

- ✅ Basic object tracking
- ✅ Voice/text commands
- ✅ Simple cinematic shots (orbit, dolly)
- ❌ RL (defer to Phase 4)
- ❌ Complex shot planning (defer)
- ❌ Pixel-level segmentation (defer)

## 12. Open Questions & Research Gaps

### Areas Needing Further Investigation

1. **Phone Camera API**: Can we control zoom programmatically for dolly zoom?
2. **Network Latency**: Actual measurements of IP Webcam latency on local network
3. **Motor Characteristics**: Detailed PWM response curves for smooth acceleration
4. **Carpet Friction**: Quantify for RL training data

### Recommended Next Steps

1. Benchmark model performance on actual RTX 2080 Ti hardware
2. Test phone streaming latency with IP Webcam
3. Characterize motor response (PWM → actual velocity)
4. Implement Phase 1 (basic tracking) to validate architecture

## Conclusion

This architecture balances simplicity for the first iteration with a clear path to advanced features. The key insight is separating "what to track" (VLM, slow) from "where is it" (tracker, fast), enabling real-time performance while maintaining natural language flexibility.

The recommended model stack (Grounding DINO + RTMDet/ByteTrack) solves the open-vocabulary problem while maintaining 30 FPS performance, making it ideal for a cinema robot that needs to react to moving objects in real-time.
