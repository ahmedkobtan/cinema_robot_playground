---
name: Cinema Bot Consolidated Architecture Plan
overview: Comprehensive consolidated architecture plan addressing all four proposals, user feedback, and research gaps. Includes validated model selection, agentic AI architecture, visual servoing design, and implementation roadmap.
todos:
  - id: validate-models
    content: Benchmark Follow Anything (FAn), LLM (Llama 3.1 8B via Ollama), SAM 2, and Bot-SORT on RTX 2080 Ti to verify performance claims and VRAM usage
    status: pending
  - id: prototype-tracking
    content: "Implement prototype tracking pipeline: Grounding DINO → SAM 2 → Bot-SORT OR Follow Anything (FAn) standalone for open-vocabulary object tracking"
    status: pending
    dependencies:
      - validate-models
  - id: arduino-firmware
    content: Develop Arduino firmware for motor/servo control with serial command interface (M:F:speed, S:angle, etc.)
    status: pending
  - id: video-pipeline
    content: "Set up video streaming pipeline: phone → Wi-Fi → OpenCV (720p @ 30fps) using IP Webcam or DroidCam"
    status: pending
  - id: director-agent
    content: "Implement Director Agent (Layer 1): LangChain ReAct agent with LLM (Llama 3.1 8B) for agentic command parsing, integrated with Grounding DINO or FAn for object detection"
    status: pending
    dependencies:
      - prototype-tracking
  - id: visual-servoing
    content: "Implement PID-based visual servoing controller with movement hierarchy: servo (X-axis) → wheels (Z-axis/rotation) → depth control"
    status: pending
    dependencies:
      - arduino-firmware
      - video-pipeline
  - id: shot-planner
    content: Implement rule-based shot planner for Follow, Dolly In/Out, and Orbit shots with trajectory calculation
    status: pending
    dependencies:
      - visual-servoing
  - id: cinematographer-agent
    content: "Implement Cinematographer Agent (Layer 2): integrates tracker with shot planner to generate high-level trajectory commands"
    status: pending
    dependencies:
      - director-agent
      - shot-planner
  - id: pilot-agent
    content: "Implement Pilot Agent (Layer 3): visual servoing controller that converts trajectory commands to serial motor commands"
    status: pending
    dependencies:
      - cinematographer-agent
  - id: moving-objects
    content: Add velocity estimation and predictive control for handling fast-moving objects, implement tracker loss handling with re-acquisition
    status: pending
    dependencies:
      - pilot-agent
  - id: test-mvp
    content: "Build and test complete MVP: text command → object detection → tracking → visual servoing → Arduino control on actual hardware"
    status: pending
    dependencies:
      - moving-objects
---
