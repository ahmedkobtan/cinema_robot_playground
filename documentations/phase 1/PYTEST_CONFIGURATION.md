# Pytest Configuration Fix

## Issue

When running `poetry run pytest`, pytest was collecting and trying to run scripts from the `scripts/` directory because they match the `test_*.py` pattern. This caused:

1. **Scripts being executed as tests** - Scripts like `test_video_stream.py` opened cameras and tried to connect to hardware
2. **Camera staying open** - Video streams weren't properly closed when pytest tried to run them
3. **Hanging tests** - Scripts waiting for hardware that may not be available

## Root Cause

Pytest by default collects any file matching `test_*.py` or `*_test.py` patterns from the entire project directory. The scripts in `scripts/` are **standalone test scripts** meant to be run directly with `python`, not as pytest unit tests.

## Solution

Added pytest configuration to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
# Only collect tests from tests/ directory, ignore scripts/
testpaths = ["tests"]
# Ignore scripts directory completely
norecursedirs = ["scripts", "*.egg", "*.eggs", ".git", ".tox", "dist", "build"]
# Only collect files that match test patterns in tests/ directory
python_files = ["test_*.py", "*_test.py"]
```

## Result

- ✅ Pytest now only collects tests from `tests/` directory
- ✅ Scripts in `scripts/` are ignored by pytest
- ✅ No more camera/hardware opening during pytest runs
- ✅ Faster test collection (94 tests instead of 132)

## Usage

### Run pytest (unit tests only):
```bash
poetry run pytest
# or
poetry run pytest tests/
```

### Run scripts (standalone tests):
```bash
# These should be run directly, not via pytest
poetry run python scripts/test_video_stream.py
poetry run python scripts/test_arduino_firmware.py
poetry run python scripts/benchmark_models.py
```

## If Camera Stays Open

If you still have a camera open from a previous pytest run:

1. **Kill the process**:
   ```bash
   pkill -f "python.*test_video"
   # or
   pkill -f opencv
   ```

2. **Close any OpenCV windows** manually (they should close when process dies)

3. **Restart terminal** if needed

## Why Scripts Opened Camera

The `test_video_stream.py` script has a `test_local_camera()` function that:
- Opens `camera_index=0` (your laptop camera)
- Reads frames for testing
- Should disconnect when done, but pytest interruption prevented cleanup

This is **expected behavior** when running the script directly, but **not expected** when running pytest.
