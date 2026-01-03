# Phase 1 PC Testing Guide

This guide provides step-by-step instructions for testing all Phase 1 components on your PC with GPU.

## Prerequisites

1. **Hardware Setup**:
   - PC with NVIDIA GPU (RTX 2080 Ti or compatible)
   - Arduino connected via USB (for firmware tests)
   - Smartphone with IP Webcam app installed and connected to same Wi-Fi network

2. **Software Setup**:
   - Python environment with all dependencies installed
   - CUDA drivers installed
   - Model weights downloaded to `ahmedkobtan_cinema_robot_playground/resources/`:
     - `sam2.1_hiera_base_plus.pt` (or other SAM 2 checkpoint)
     - `osnet_x0_25_msmt17.pt` (Bot-SORT ReID weights)

3. **Configuration**:
   - Update `config/dev.json` (or `config/qa.json`, `config/prod.json`) with your IP Webcam URL:
     ```json
     {
       "configResolution": {
         "resolved": {
           "env": "DEV",
           "ip_webcam_url": "http://YOUR_PHONE_IP:8080"
         }
       }
     }
     ```

## Test Execution Order

### Step 1: Verify Environment Setup

```bash
# Activate poetry environment
cd /path/to/cinema_robot_playground
poetry shell

# Verify GPU availability
poetry run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Verify model weights exist
ls -lh ahmedkobtan_cinema_robot_playground/resources/*.pt
```

**Expected Output**:
- CUDA available: True
- GPU: NVIDIA GeForce RTX 2080 Ti (or your GPU name)
- Model weights listed with file sizes

---

### Step 2: Run Model Benchmarks

**Purpose**: Verify all models load correctly and measure performance on GPU.

```bash
# Run benchmark (will use GPU automatically)
poetry run python scripts/benchmark_models.py --device cuda --num-frames 100

# Check results
cat benchmark_results.json
```

**Expected Results**:
- All three models should show `"available": true`
- Follow Anything: Detection FPS > 10, Tracking FPS > 300
- Grounding DINO: FPS > 0.1 (slower model)
- Bot-SORT: FPS > 300

**Troubleshooting**:
- If Grounding DINO shows "not available": Check that transformers library is installed and model can download from HuggingFace
- If SAM 2 fails: Verify SAM2_CHECKPOINT env var or checkpoint in resources directory

---

### Step 3: Test SAM 2 Checkpoint Loading

**Purpose**: Verify SAM 2 checkpoint loads and generates masks correctly.

```bash
# Find your SAM 2 checkpoint
SAM2_CHECKPOINT=$(find ahmedkobtan_cinema_robot_playground/resources -name "sam2.1_*.pt" | head -1)

# Test checkpoint loading
poetry run python scripts/test_sam2_checkpoint.py \
  --checkpoint "$SAM2_CHECKPOINT" \
  --device cuda

# Test mask generation
poetry run python scripts/test_sam2_checkpoint.py \
  --checkpoint "$SAM2_CHECKPOINT" \
  --device cuda \
  --test-masks
```

**Expected Output**:
- ✓ Checkpoint loaded successfully
- ✓ Model built successfully
- ✓ Predictor created successfully
- ✓ Masks generated (if --test-masks used)

---

### Step 4: Test Follow Anything Components

**Purpose**: Verify individual FAn components work correctly.

```bash
# Test all FAn components
poetry run python scripts/test_fan_components.py --device cuda
```

**Expected Output**:
- ✓ Open-CLIP model loaded
- ✓ Open-CLIP tokenizer loaded
- ✓ DINOv2 model loaded
- ✓ SAM 2 modules imported successfully
- ✓ Follow Anything model initialized

---

### Step 5: Test Follow Anything End-to-End

**Purpose**: Comprehensive test of FAn detection and tracking.

```bash
# Run end-to-end tests
poetry run python scripts/test_fan_end_to_end.py --device cuda
```

**Expected Output**:
- All tests pass (clip_text, clip_image, clip_similarity, fan_detection, fan_tracking, sam2)
- Detection finds objects in test images
- Tracking maintains object identity across frames

---

### Step 6: Test Follow Anything Full Flow

**Purpose**: Test complete FAn workflow with SAM 2.

```bash
# Run full flow test
poetry run python scripts/test_fan_full_flow.py --device cuda
```

**Expected Output**:
- ✓ Test image created
- ✓ Detection successful
- ✓ Tracking successful
- ✓ SAM 2 masks generated

---

### Step 7: Test Video Stream Connection

**Purpose**: Verify IP Webcam connection works.

```bash
# Test video stream (uses config file for URL)
poetry run python scripts/test_video_stream.py --test-display

# Or specify URL explicitly
poetry run python scripts/test_video_stream.py \
  --stream-url "http://YOUR_PHONE_IP:8080" \
  --test-display
```

**Expected Output**:
- Video window opens showing live feed from phone
- FPS displayed in console
- No connection errors

**Troubleshooting**:
- If connection fails: Check IP address, ensure phone and PC on same Wi-Fi, verify IP Webcam app is running
- If no video: Try adding `/video` suffix to URL (some IP Webcam versions require it)

---

### Step 8: Test Live Object Tracking

**Purpose**: Test object tracking from live IP Webcam feed.

```bash
# Test with "desk lamp" prompt using FAn method
poetry run python scripts/test_object_tracking_live.py \
  --prompt "desk lamp" \
  --method fan \
  --device cuda \
  --max-frames 100

# Test with "scissors" prompt using fallback method
poetry run python scripts/test_object_tracking_live.py \
  --prompt "scissors" \
  --method fan_fallback \
  --device cuda \
  --max-frames 100

# Test all methods and objects
poetry run python scripts/test_object_tracking_live.py \
  --test-all \
  --device cuda \
  --max-frames 50
```

**Expected Output**:
- Tracking box drawn on detected objects
- FPS displayed
- Object tracked across frames
- Confidence scores shown

**Note**: If objects not detected, try different prompts or ensure objects are visible in camera view.

---

### Step 9: Test Arduino Firmware

**Purpose**: Verify Arduino communication and motor/servo control.

**Prerequisites**: Arduino connected via USB, firmware uploaded.

```bash
# Test Arduino firmware (with hardware)
poetry run python scripts/test_arduino_firmware.py \
  --serial-port /dev/ttyACM0  # Linux/Mac
  # or
  # --serial-port COM3  # Windows

# Test without hardware (skip-hardware flag)
poetry run python scripts/test_arduino_firmware.py --skip-hardware
```

**Expected Output**:
- ✓ Connection established
- ✓ Motor commands (forward, backward, left, right, stop) execute
- ✓ Servo commands execute
- ✓ Context manager works
- ✓ Error handling works

**Troubleshooting**:
- If connection fails: Check serial port name, ensure Arduino is connected, verify baud rate (115200)
- If commands fail: Verify firmware is uploaded correctly, check serial monitor for errors

---

### Step 10: Test LLM Agent (Director Agent)

**Purpose**: Verify Director Agent can parse natural language commands.

```bash
# Test LLM agent
poetry run python scripts/test_llm_agent.py --device cuda

# Test with specific command
poetry run python scripts/test_llm_agent.py \
  --command "Orbit the red cup" \
  --device cuda
```

**Expected Output**:
- ✓ LLM model loaded (Phi-3-mini)
- ✓ Commands parsed correctly
- ✓ Shot types identified (orbit, follow, dolly_in, etc.)
- ✓ Objects extracted from commands

**Note**: Uses HuggingFace Transformers (not Ollama) as specified in Phase 1.

---

### Step 11: Test Setup Script

**Purpose**: Verify all dependencies and models are available.

```bash
# Run setup verification
poetry run python scripts/test_setup.py --device cuda
```

**Expected Output**:
- All dependencies available
- All models can be imported
- GPU detected and available

---

### Step 12: Run All Pytest Tests

**Purpose**: Verify all unit and integration tests pass.

```bash
# Run all tests
poetry run python -m pytest tests/ -v --tb=short

# Run specific test suites
poetry run python -m pytest tests/src/models/ -v
poetry run python -m pytest tests/src/services/ -v
poetry run python -m pytest tests/src/utils/ -v
```

**Expected Output**:
- All tests pass (99+ tests)
- No hanging tests
- All mocks work correctly

**Note**: Pytest tests use mocks and don't require actual hardware/streams.

---

## Test Summary Checklist

After completing all tests, verify:

- [ ] **Model Benchmarks**: All 3 models available and benchmarked
- [ ] **SAM 2**: Checkpoint loads and generates masks
- [ ] **Follow Anything**: All components work, end-to-end tests pass
- [ ] **Video Stream**: IP Webcam connection works
- [ ] **Live Tracking**: Objects tracked from live feed
- [ ] **Arduino**: Serial communication and motor/servo control work
- [ ] **LLM Agent**: Natural language parsing works
- [ ] **Pytest**: All unit tests pass

---

## Common Issues and Solutions

### Issue: Models not available
**Solution**:
- Check model weights in `resources/` directory
- Verify CUDA is available: `torch.cuda.is_available()`
- Check GPU memory: `nvidia-smi`

### Issue: IP Webcam connection fails
**Solution**:
- Verify phone and PC on same Wi-Fi network
- Check IP address in config file
- Try adding `/video` suffix to URL
- Ensure IP Webcam app is running

### Issue: Arduino not detected
**Solution**:
- Check serial port name: `ls /dev/tty*` (Linux/Mac) or Device Manager (Windows)
- Verify baud rate matches (115200)
- Check USB cable connection
- Verify firmware is uploaded

### Issue: Tracking not working
**Solution**:
- Ensure objects are visible and well-lit
- Try different text prompts
- Check model confidence thresholds
- Verify GPU memory is sufficient

---

## Performance Expectations (RTX 2080 Ti)

Based on benchmark results:

- **Follow Anything Detection**: ~10-15 FPS
- **Follow Anything Tracking**: ~300-600 FPS
- **Grounding DINO**: ~0.2-0.5 FPS (slower model)
- **Bot-SORT**: ~400-500 FPS

---

## Next Steps After Phase 1 Testing

Once all Phase 1 tests pass:

1. Document any issues found
2. Verify all models meet performance requirements
3. Proceed to Phase 2: Detection & Tracking integration
4. Begin integration testing with Cinema Bot orchestrator

---

## Notes

- All scripts use `--device cuda` to explicitly use GPU
- Scripts automatically load config from `config/dev.json` (or ENV variable)
- Model weights are auto-detected from `resources/` directory
- Tests that require hardware (Arduino, IP Webcam) can be skipped with flags
- Pytest tests use mocks and don't require actual hardware
