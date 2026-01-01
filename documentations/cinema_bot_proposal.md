# Cinema Bot: Consolidated Architecture Plan

## Executive Summary

This plan consolidates four previous proposals, addresses all user feedback, and incorporates additional research to create a unified, actionable architecture for the AI-powered cinema robot. The architecture is designed for RTX 2080 Ti (11GB VRAM), Arduino Uno, 4WD DC motors, FS90 servo, and smartphone camera.

## 1. Open-Vocabulary Detection → Tracker Handoff: CORRECTED APPROACH

### 1.1 The Problem with Previous Proposals

**Issue Identified**: Some proposals suggested Grounding DINO → RTMDet/YOLO handoff, which is **incorrect** because:

- RTMDet/YOLO are class-limited (pre-trained on COCO classes)
- Cannot track objects not in their training set
- Defeats the purpose of open-vocabulary detection

### 1.2 Corrected Solution Architecture

**Primary Recommendation: Follow Anything (FAn) - Standalone Solution**

- **What it is**: Unified open-vocabulary detection and tracking model
- **Capabilities**:
  - Accepts text, image, or click queries
  - Performs both detection AND tracking in one model
  - No handoff needed - eliminates complexity
- **Performance**: Real-time capable on 6-8GB VRAM (fits RTX 2080 Ti)
- **Why it wins**: Solves the handoff problem elegantly by design

**Alternative: Grounding DINO → SAM 2 → Bot-SORT Pipeline**

If FAn doesn't meet performance requirements:

1. **Grounding DINO** (once): Finds object from text → bounding box
2. **SAM 2** (30 FPS): Takes bounding box as prompt → generates segmentation mask
3. **Bot-SORT/ByteTrack** (30 FPS): Tracks mask frame-to-frame (class-agnostic)

**Why this works**:

- SAM 2 tracks pixels, not classes
- Bot-SORT tracks by appearance/motion, not semantic class
- Both are class-agnostic, solving the open-vocabulary problem

### 1.3 Model Performance on RTX 2080 Ti

| Model | Task | FPS (Est.) | VRAM | Use Case |

|-------|------|------------|------|----------|

| Follow Anything (FAn) | Detection + Tracking | 20-30 | ~6-8GB | Unified solution |

| Grounding DINO | Grounding | 5-10 | ~4GB | Initial detection |

| SAM 2 | Segmentation | 20-30 | ~6GB | Mask generation |

| Bot-SORT | Tracking | 30+ | <1GB | Frame-to-frame tracking |

| Mobile-Seed | Segmentation | 24 | ~3GB | Lightweight alternative |

**Recommendation**: Start with **Follow Anything (FAn)**. If performance is insufficient, fall back to Grounding DINO → SAM 2 → Bot-SORT pipeline.

## 2. Vision-Language-Action (VLA) Models: Comprehensive Evaluation

### 2.1 VLA Model Analysis

**OpenVLA Evaluation**:

- **Model Size**: ~7B parameters (base model)
- **VRAM Requirements**: ~14-16GB for full precision (exceeds RTX 2080 Ti's 11GB)
- **Inference Speed**: ~200-500ms per inference (2-5 FPS)
- **Use Case**: End-to-end vision-language-action mapping
- **Verdict**: **Too large and too slow for real-time control on RTX 2080 Ti**

**RT-2 (Robotic Transformer 2) Evaluation**:

- **Model Size**: Proprietary, likely 5-10B parameters
- **VRAM Requirements**: Similar to OpenVLA
- **Inference Speed**: Similar latency issues
- **Availability**: Google proprietary, limited access
- **Verdict**: **Not suitable for this setup**

### 2.2 Corrected Architecture: Hybrid Approach

**VLA models are NOT suitable for direct real-time control** due to:

1. Latency (200-500ms) too high for 30 FPS visual servoing
2. VRAM requirements exceed RTX 2080 Ti capacity
3. Designed for discrete actions, not continuous control

**Recommended Architecture**:

```
Voice/Text Command → LLM Agent (Command Parsing) → Grounding DINO/FAn (Object Detection) → Tracker (30 FPS) → PID Controller (Motor Commands)
```

**Why this works**:

- LLM agent handles high-level command understanding (once)
- VLM handles object grounding (once or infrequently)
- Fast tracker maintains real-time updates (30 FPS)
- PID controller generates smooth motor commands

**Future Consideration**: VLA models may be useful for:

- High-level shot planning (offline)
- Complex scene understanding
- Multi-step task planning
- **Not for real-time motor control**

## 3. Agentic AI Architecture: Comprehensive Design

### 3.1 Three-Layer Agentic Architecture

**Layer 1: Director Agent (VLM + LLM Agent)**

- **Purpose**: Understand user intent, ground objects, identify shot types
- **Implementation**: LangChain ReAct agent with LLM (Llama 3.1 8B via Ollama)
- **Components**:
  - **LLM Agent**: Parses natural language commands ("Orbit the red cup")
  - **Grounding DINO/FAn**: Finds object from parsed command
  - **Output**: Object bounding box + shot type
- **Frequency**: Once at start, or when tracker loses object
- **Why Agentic**: Enables complex command understanding, error recovery, multi-step reasoning

**Layer 2: Cinematographer Agent (Motion Planner)**

- **Purpose**: Calculate trajectories for cinematic shots
- **Implementation**: Rule-based initially, can be enhanced with RL later
- **Components**:
  - **Tracker Integration**: Receives object coordinates from Layer 1
  - **Shot Planner**: Calculates trajectory based on shot type
  - **Output**: High-level JSON commands (`{"action": "orbit", "radius": 1.0, "speed": 0.5}`)
- **Frequency**: 30 FPS (real-time)

**Layer 3: Pilot Agent (Controller)**

- **Purpose**: Execute smooth motor commands
- **Implementation**: PID controller (first iteration), RL later
- **Components**:
  - **Visual Servoing**: Calculates error (object position vs desired)
  - **PID Controller**: Generates motor PWM values
  - **Movement Hierarchy**: Servo first, wheels if needed
- **Output**: Serial commands to Arduino
- **Frequency**: 30 FPS (real-time)

### 3.2 Agent Framework Selection

**LangChain ReAct Agent** (Recommended for MVP):

- **Why**: Provides structured agentic reasoning, tool calling, error handling
- **LLM Backend**: Llama 3.1 8B via Ollama (runs locally on RTX 2080 Ti)
- **Tools Available**:
  - Grounding DINO/FAn for object detection
  - Shot planner for trajectory calculation
  - Tracker for position updates
- **Benefits**:
  - Natural language command parsing
  - Error recovery and re-planning
  - Extensible for future agents

**Alternative**: Custom Python state machine (simpler, but less agentic)

### 3.3 Agent Communication Protocol

**Director → Cinematographer**:

```json
{
  "object_id": "red_cup",
  "bounding_box": [x, y, w, h],
  "shot_type": "orbit",
  "parameters": {"radius": 1.0, "speed": 0.5}
}
```

**Cinematographer → Pilot**:

```json
{
  "action": "orbit",
  "target_position": [x, y],
  "current_position": [robot_x, robot_y],
  "servo_angle": 90,
  "wheel_speeds": {"left": 150, "right": 150}
}
```

## 4. Visual Servoing: Unified Design

### 4.1 Movement Hierarchy (Confirmed Across All Proposals)

**Tier 1: Servo (Fine Control - X-Axis)**

- **Range**: ±90° from center (FS90 servo: 900-2100μs pulse width)
- **Speed**: Fast, smooth, low latency
- **Use Case**: Small object movements (< 20% of frame width)
- **Priority**: **Always try servo first**
- **Control**: PID controller with error = `object_center_x - frame_center_x`

**Tier 2: Wheels (Coarse Control - Z-Axis & Rotation)**

- **Trigger Conditions**:

  1. Servo at physical limit (±90°)
  2. Object moving faster than servo can track
  3. Large position error (> 40% of frame width)

- **Use Case**: Large movements, rotation, depth control
- **Priority**: Secondary to servo
- **Control**:
  - **Z-axis (depth)**: PID with error = `current_box_area - desired_box_area`
  - **Rotation**: Differential wheel speeds for turning

**Tier 3: Depth Control (Z-Axis)**

- **Method**: Bounding box area analysis
- **Logic**:
  - Box area increasing → object closer → drive backward
  - Box area decreasing → object farther → drive forward
- **Goal**: Maintain desired shot size (close-up, medium, wide)
- **Integration**: Works in conjunction with Tier 2 wheel control

### 4.2 Visual Servoing Algorithm

**Error Calculation**:

```python
# X-axis error (horizontal)
error_x = object_center_x - frame_center_x

# Z-axis error (depth)
desired_area = frame_width * frame_height * 0.1  # 10% of frame
current_area = bounding_box_width * bounding_box_height
error_z = current_area - desired_area

# Servo control (Tier 1)
if abs(error_x) < threshold_servo:  # Small error
    servo_angle = center_angle + Kp_x * error_x
    wheel_speed = 0  # Don't move wheels
else:  # Large error or servo at limit
    # Use wheels (Tier 2)
    wheel_speed = Kp_z * error_x  # Turn to reduce error
    servo_angle = center_angle  # Reset servo
```

**PID Controller Implementation**:

- **X-axis (Servo)**: `servo_output = Kp*error_x + Ki*integral_x + Kd*derivative_x`
- **Z-axis (Wheels)**: `wheel_output = Kp*error_z + Ki*integral_z + Kd*derivative_z`
- **Tuning**: Start with conservative gains, tune based on real-world performance

## 5. Cinematic Shot Types: Validated and Documented

### 5.1 Research-Validated Shot Types

Based on professional cinematography and camera slider usage:

**Basic Shots (MVP - First Iteration)**:

1. **Follow/Track** (Most Common)

   - **Motion**: Match object's movement, keep centered
   - **Servo**: Pan to maintain center framing
   - **Wheels**: Move forward/backward/rotate to follow
   - **Use Case**: Standard tracking shot, most common in professional work
   - **Implementation**: Visual servoing with PID control

2. **Dolly In/Out** (Very Common)

   - **Motion**: Drive straight toward/away from object
   - **Servo**: Locked forward (no panning)
   - **Wheels**: Forward/backward at constant speed
   - **Use Case**: Dramatic focus, reveals context
   - **Implementation**: Constant velocity control, maintain center framing

3. **Orbit** (Common for Dynamic Shots)

   - **Motion**: Circle around object at constant radius
   - **Servo**: Continuously pan to keep object centered
   - **Wheels**: Circular path, coordinated with servo
   - **Use Case**: Dynamic "hero shot", shows 3D space
   - **Implementation**: Calculate circular trajectory, coordinate wheels + servo

**Advanced Shots (Future Iterations)**:

4. **Reveal** (Lateral Movement)

   - **Motion**: Drive sideways (strafing) while panning
   - **Servo**: Pan from edge to center
   - **Wheels**: Lateral movement
   - **Use Case**: Reveals object from behind obstacle

5. **Dolly Zoom** (Vertigo Effect)

   - **Motion**: Drive backward while digitally zooming in
   - **Servo**: Locked forward
   - **Wheels**: Backward motion
   - **Digital**: Phone camera zoom API
   - **Use Case**: Dramatic effect, subject stays same size

6. **Lead** (Rule of Thirds)

   - **Motion**: Keep object at edge of frame (1/3 position)
   - **Servo**: Pan to maintain 1/3 framing
   - **Wheels**: Follow object's movement
   - **Use Case**: Cinematic composition, shows anticipation

### 5.2 Shot Implementation Strategy

**First Iteration (Rule-Based)**:

- Hard-code trajectory calculations for each shot type
- Input: Shot type + object position
- Output: Pre-calculated path (wheel speeds, servo angles over time)
- **Why**: Simple, predictable, debuggable

**Future Iteration (RL-Enhanced)**:

- RL agent learns optimal trajectories for each shot type
- Reward: Framing quality, smoothness, cinematic appeal
- **Why**: Can adapt to different environments and learn complex patterns

## 6. Moving Objects: Comprehensive Handling Strategy

### 6.1 Velocity Estimation and Prediction

**Velocity Calculation**:

```python
# Calculate object velocity from tracker
velocity_x = (object_x_t - object_x_t1) / dt
velocity_y = (object_y_t - object_y_t1) / dt
speed = sqrt(velocity_x^2 + velocity_y^2)
```

**Predictive Control**:

- **Kalman Filter**: Predicts future object position
- **Feedforward Control**: Adjusts motor speed based on predicted position
- **Adaptive Gains**: Increases PID gains when object moving fast

### 6.2 Fast-Moving Object Handling

**Strategy**:

1. **Velocity-Based Speed Adjustment**: Increase motor speed proportionally to object velocity
2. **Predictive Positioning**: Use Kalman filter to predict where object will be
3. **Servo Priority**: Use servo for fast corrections, wheels for large movements
4. **Maximum Speed Limits**: Hardware constraints (acceptable limitation)

**Implementation**:

```python
if object_speed > threshold_fast:
    # Increase motor speed
    base_speed = 150
    adaptive_speed = base_speed + K_velocity * object_speed
    motor_speed = min(adaptive_speed, max_speed)  # Cap at hardware limit
else:
    motor_speed = base_speed
```

### 6.3 Tracker Loss Handling

**When Tracker Loses Object**:

1. **Immediate Response**:

   - Stop robot movement (safety)
   - Re-run Grounding DINO/FAn to re-acquire
   - Resume tracking if found

2. **If Re-acquisition Fails**:

   - Alert user via UI
   - Option 1: Return to last known position
   - Option 2: Execute search pattern (spiral, grid)
   - Option 3: Wait for user input

3. **Prevention**:

   - Maintain tracker confidence threshold
   - Re-ground before confidence drops too low
   - Use predictive control to handle brief occlusions

## 7. System Architecture: Complete Design

### 7.1 Software Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    PC (RTX 2080 Ti)                      │
│                                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Layer 1: Director Agent (LangChain ReAct)        │   │
│  │  - LLM (Llama 3.1 8B via Ollama)                │   │
│  │  - Grounding DINO / Follow Anything (FAn)        │   │
│  │  - Command Parsing & Object Grounding            │   │
│  └──────────────────┬────────────────────────────────┘   │
│                     │                                      │
│  ┌──────────────────▼────────────────────────────────┐   │
│  │  Layer 2: Cinematographer Agent                    │   │
│  │  - Tracker (FAn / SAM 2 + Bot-SORT)               │   │
│  │  - Shot Planner (Rule-based)                        │   │
│  │  - Trajectory Calculation                           │   │
│  └──────────────────┬────────────────────────────────┘   │
│                     │                                      │
│  ┌──────────────────▼────────────────────────────────┐   │
│  │  Layer 3: Pilot Agent (Visual Servoing)            │   │
│  │  - PID Controller (X-axis: servo, Z-axis: wheels) │   │
│  │  - Movement Hierarchy Logic                         │   │
│  │  - Command Generator (Serial protocol)              │   │
│  └──────────────────┬────────────────────────────────┘   │
└──────────────────────┼────────────────────────────────────┘
                       │ USB Serial (115200 baud)
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  Arduino Uno                            │
│  - Motor Control (L298N, 4WD DC motors)                │
│  - Servo Control (FS90, 900-2100μs)                    │
│  - Command Listener (Serial protocol)                  │
└────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Smartphone (Camera)                         │
│  - IP Webcam / DroidCam (Wi-Fi stream)                  │
│  - 720p @ 30fps → PC via OpenCV                        │
└────────────────────────────────────────────────────────┘
```

### 7.2 Data Flow

1. **User Input**: Text command ("Orbit the red cup")
2. **Director Agent**:

   - LLM parses command → extracts object ("red cup") and shot type ("orbit")
   - Grounding DINO/FAn finds object → bounding box

3. **Cinematographer Agent**:

   - Tracker maintains object position (30 FPS)
   - Shot planner calculates trajectory for "orbit"

4. **Pilot Agent**:

   - Visual servoing calculates error
   - PID controller generates motor commands
   - Movement hierarchy: servo first, wheels if needed

5. **Arduino**: Executes motor commands
6. **Feedback Loop**: Camera stream → tracker → control (30 FPS)

### 7.3 Communication Protocol

**PC → Arduino (Serial Commands)**:

```
M:F:200    # Move Forward at speed 200 (0-255)
M:B:150    # Move Backward at speed 150
M:L:180    # Turn Left at speed 180
M:R:180    # Turn Right at speed 180
M:S        # Stop
S:1500     # Set Servo to 1500 microseconds (center)
```

**Arduino → PC (Acknowledgments)**:

```
OK          # Command received and executed
ERROR:xxx   # Error message
```

## 8. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

- [ ] Set up video streaming pipeline (phone → PC via Wi-Fi)
- [ ] Implement Arduino firmware (motor/servo control, serial listener)
- [ ] Benchmark Follow Anything (FAn) on RTX 2080 Ti
- [ ] Set up LangChain + Ollama (Llama 3.1 8B) for Director Agent

### Phase 2: Detection & Tracking (Week 3-4)

- [ ] Integrate Follow Anything (FAn) OR Grounding DINO → SAM 2 → Bot-SORT
- [ ] Implement Director Agent (LangChain ReAct with LLM)
- [ ] Test object detection from text commands
- [ ] Implement basic tracking loop (30 FPS)

### Phase 3: Visual Servoing (Week 5-6)

- [ ] Implement PID controller (X-axis: servo, Z-axis: wheels)
- [ ] Implement movement hierarchy (servo first, wheels if needed)
- [ ] Add velocity estimation and predictive control
- [ ] Test with static objects

### Phase 4: Cinematic Shots (Week 7-8)

- [ ] Implement shot planner (Follow, Dolly, Orbit)
- [ ] Integrate Cinematographer Agent
- [ ] Test each shot type individually
- [ ] Add tracker loss handling

### Phase 5: Integration & Testing (Week 9-10)

- [ ] Full system integration
- [ ] Test with moving objects
- [ ] Performance optimization
- [ ] Documentation and user guide

### Phase 6: Future Enhancements (Post-MVP)

- [ ] RL for motor smoothness (Isaac Sim training)
- [ ] Additional shot types (Reveal, Dolly Zoom, Lead)
- [ ] Voice command integration (Whisper)
- [ ] Advanced agentic capabilities

## 9. Key Technical Decisions Summary

### Model Selection

- **Initial Detection**: Follow Anything (FAn) OR Grounding DINO
- **Tracking**: FAn (if used) OR SAM 2 + Bot-SORT
- **Command Parsing**: LangChain ReAct Agent with Llama 3.1 8B
- **Control**: PID controller (first iteration), RL later

### Architecture

- **Three-Layer Agentic**: Director → Cinematographer → Pilot
- **VLM**: One-time use for object grounding (not real-time control)
- **Tracker**: Continuous 30 FPS for position updates
- **Controller**: PID-based visual servoing with movement hierarchy

### First Iteration Scope

- ✅ Follow Anything (FAn) OR Grounding DINO → SAM 2 → Bot-SORT
- ✅ LangChain ReAct Director Agent
- ✅ Basic cinematic shots (Follow, Dolly, Orbit)
- ✅ Visual servoing with movement hierarchy
- ✅ Moving object handling
- ❌ VLA models (too large/slow)
- ❌ RL (defer to Phase 6)
- ❌ Complex shot planning (defer)

## 10. Critical Success Factors

1. **Real-time Performance**: Must maintain 30 FPS tracking and control
2. **Open-Vocabulary**: Must track any object, not just pre-trained classes
3. **Smooth Control**: Minimize jitter in footage (movement hierarchy + PID)
4. **Robust Tracking**: Handle occlusion, fast movement, tracker loss
5. **Modular Design**: Easy to add features (RL, more shots, voice commands)
6. **Agentic Capabilities**: At least one agent (Director) for command understanding

This consolidated plan addresses all feedback, corrects previous misconceptions, and provides a clear, actionable roadmap for building the cinema robot.
