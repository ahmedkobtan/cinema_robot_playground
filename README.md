# Cinema Robot Playground

AI-powered cinema robot for autonomous cinematic filming using computer vision, object tracking, and visual servoing.

## Architecture

The system uses a three-layer agentic architecture:

1. **Director Agent (Layer 1)**: Command parsing and object grounding
   - Parses natural language commands ("Orbit the red cup")
   - Uses detection models (Follow Anything, Grounding DINO) to find objects
   - Outputs: Object bounding box + shot type

2. **Cinematographer Agent (Layer 2)**: Trajectory planning
   - Tracks objects using SAM 2 + Bot-SORT or Follow Anything
   - Plans cinematic shot trajectories (Follow, Dolly, Orbit)
   - Outputs: High-level trajectory commands

3. **Pilot Agent (Layer 3)**: Visual servoing and motor control
   - Implements PID-based visual servoing
   - Movement hierarchy: Servo (fine) → Wheels (coarse)
   - Outputs: Serial commands to Arduino

## Hardware Requirements

- **PC**: Ubuntu 24.04 with RTX 2080 Ti (11GB VRAM)
- **Arduino**: Arduino Uno
- **Motors**: 4WD DC motors via L298N driver
- **Servo**: FS90 servo (900-2100μs pulse width, 120° range)
- **Camera**: Smartphone with IP Webcam or DroidCam app

## Software Setup

### 1. Install Dependencies

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

### 2. Upload Arduino Firmware

1. Open Arduino IDE
2. Open `ahmedkobtan_cinema_robot_playground/src/utils/arduino_scripts/robot_control/robot_control.ino`
3. Select board: Tools → Board → Arduino Uno
4. Select port: Tools → Port → /dev/ttyACM0 (or your port)
5. Upload the sketch

### 3. Set Up Video Stream

**Option A: IP Webcam (Android)**
1. Install IP Webcam app
2. Start server
3. Note the IP address (e.g., `192.168.1.100:8080`)

**Option B: DroidCam (Android/iOS)**
1. Install DroidCam app
2. Start server
3. Note the IP address

### 4. Test Setup (Optional)

Before running on the PC, test locally:

```bash
poetry run python scripts/test_setup.py
```

### 5. Run Cinema Bot

```bash
# Activate Poetry shell
poetry shell

# Run with mock models (for testing without GPU)
python -m ahmedkobtan_cinema_robot_playground.src.services.cinema_bot \
    --mock \
    --command "Orbit the red cup"

# Run with real models (requires GPU on PC)
python -m ahmedkobtan_cinema_robot_playground.src.services.cinema_bot \
    --stream-url "http://192.168.1.100:8080/video" \
    --serial-port "/dev/ttyACM0" \
    --command "Orbit the red cup"

# Run with LLM command parsing (requires Ollama)
python -m ahmedkobtan_cinema_robot_playground.src.services.cinema_bot \
    --stream-url "http://192.168.1.100:8080/video" \
    --serial-port "/dev/ttyACM0" \
    --use-llm \
    --command "Orbit the red cup"
```

## Usage Examples

### Follow Shot
```bash
python -m ahmedkobtan_cinema_robot_playground.src.services.cinema_bot --command "Follow the cat"
```

### Dolly In
```bash
python -m ahmedkobtan_cinema_robot_playground.src.services.cinema_bot --command "Dolly in on the object"
```

### Orbit Shot
```bash
python -m ahmedkobtan_cinema_robot_playground.src.services.cinema_bot --command "Orbit the red cup"
```

## Project Structure

```
ahmedkobtan_cinema_robot_playground/
├── src/
│   ├── algorithms/          # Control algorithms
│   │   ├── pid_controller.py
│   │   ├── visual_servoing.py
│   │   └── shot_planner.py
│   ├── models/              # ML model interfaces
│   │   ├── detection_models.py
│   │   └── tracking_models.py
│   ├── services/            # High-level services
│   │   ├── director_agent.py
│   │   ├── cinematographer_agent.py
│   │   ├── pilot_agent.py
│   │   ├── video_stream.py
│   │   └── cinema_bot.py
│   └── utils/               # Utilities
│       ├── robot_commander.py
│       └── arduino_scripts/
│           └── robot_control/
│               └── robot_control.ino
└── tests/                   # Tests
    └── src/
        ├── algorithms/
        └── services/
```

## Testing

### Run Unit Tests

```bash
poetry run pytest tests/
```

### Test Setup

Before running on the PC, test the setup locally:

```bash
# Run setup test script
poetry run python scripts/test_setup.py
```

This will verify:
- All imports work correctly
- PID controller functionality
- Visual servoing calculations
- Shot planner
- Director and Cinematographer agents

### Benchmark Models (Phase 1)

Benchmark models on RTX 2080 Ti:

```bash
# Benchmark all models
poetry run python scripts/benchmark_models.py

# Custom benchmark (100 frames, 720p)
poetry run python scripts/benchmark_models.py --num-frames 100 --frame-size 720x1280
```

This will:
- Benchmark Follow Anything (FAn) model
- Benchmark Grounding DINO
- Benchmark Bot-SORT tracker
- Measure FPS, latency, and VRAM usage
- Save results to `benchmark_results.json`

### Test Video Stream (Phase 1)

Test video streaming from phone to PC:

```bash
# Test IP Webcam stream (replace with your phone's IP)
poetry run python scripts/test_video_stream.py \
    --stream-url "http://192.168.1.100:8080/video"

# Test with display window (shows video feed)
poetry run python scripts/test_video_stream.py \
    --stream-url "http://192.168.1.100:8080/video" \
    --test-display

# Test local camera (USB webcam)
poetry run python scripts/test_video_stream.py \
    --test-local \
    --test-display
```

This will:
- Connect to IP Webcam stream
- Measure FPS and latency
- Display video feed (with --test-display)
- Test frame reading performance
- Provide troubleshooting tips if connection fails

**Finding your phone's IP address:**
1. Open IP Webcam app on your phone
2. Tap "Start server"
3. Note the IP address shown (e.g., `192.168.1.100:8080`)
4. Use URL format: `http://192.168.1.100:8080/video`

### Test LLM Agent (Phase 1)

Test LangChain + Ollama setup:

```bash
# Test LLM agent setup
poetry run python scripts/test_llm_agent.py
```

This will:
- Test simple command parsing (no LLM)
- Test LLM-based command parsing (requires Ollama)
- Verify Ollama connection
- Provide setup instructions if Ollama is not available

## Development Status

### ✅ Phase 1: Foundation (COMPLETED)
- ✅ Video streaming pipeline (phone → PC via Wi-Fi)
- ✅ Arduino firmware (motor/servo control, serial listener)
- ✅ Follow Anything (FAn) model integration (with fallback)
- ✅ Model benchmarking scripts for RTX 2080 Ti
- ✅ LangChain + Ollama setup for Director Agent
- ✅ LLM agent testing scripts

### ✅ Phase 2-3: Core Functionality (COMPLETED)
- ✅ Detection models (Grounding DINO, Mock)
- ✅ Tracking models (Bot-SORT, Simple, Mock)
- ✅ PID controller (single and dual-axis)
- ✅ Visual servoing with movement hierarchy
- ✅ Shot planner (Follow, Dolly, Orbit)
- ✅ Three-layer agent architecture
- ✅ Director, Cinematographer, and Pilot agents

### 🚧 Ready for PC Testing
- Real model integration (Follow Anything, Grounding DINO, Bot-SORT)
- LLM command parsing (requires Ollama setup)
- Full hardware integration
- Model benchmarking on RTX 2080 Ti

### 📋 Future Enhancements (Phase 6)
- SAM 2 segmentation integration
- Velocity estimation and predictive control
- Enhanced tracker loss handling
- RL for motor smoothness (Isaac Sim)
- Additional shot types (Reveal, Dolly Zoom, Lead)
- Voice command integration (Whisper)

### 📋 Future (Phase 6)
- RL for motor smoothness (Isaac Sim)
- Additional shot types (Reveal, Dolly Zoom, Lead)
- Voice command integration (Whisper)
- Advanced agentic capabilities

## Configuration

Edit `config/dev.json`, `config/prod.json`, or `config/qa.json` for environment-specific settings.

## Serial Protocol

**PC → Arduino:**
- `M:F:speed` - Move Forward (speed 0-255)
- `M:B:speed` - Move Backward (speed 0-255)
- `M:L:speed` - Turn Left (speed 0-255)
- `M:R:speed` - Turn Right (speed 0-255)
- `M:S` - Stop
- `S:pulse` - Set Servo pulse width (900-2100 microseconds)

**Arduino → PC:**
- `OK` - Command executed successfully
- `ERROR:xxx` - Error message

## License

[Your License Here]

## Authors

Ahmed Kobtan - ahmedkobtan <akobtan@asu.edu>
