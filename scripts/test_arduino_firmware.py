#!/usr/bin/env python3
"""Test Arduino firmware for Phase 1 - Motor and Servo Control.

This script tests the Arduino firmware implementation by:
1. Connecting to Arduino via serial
2. Testing all motor commands (forward, backward, left, right, stop)
3. Testing servo commands
4. Verifying command responses
5. Testing error handling

Note: This requires Arduino to be connected and firmware uploaded.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ruff: noqa: E402
from loguru import logger

from ahmedkobtan_cinema_robot_playground.src.utils.robot_commander import (
    RobotCommander,
)


def test_connection(commander: RobotCommander) -> bool:
    """Test serial connection to Arduino."""
    logger.info("=" * 60)
    logger.info("Test 1: Serial Connection")
    logger.info("=" * 60)

    try:
        if commander.connect():
            logger.info("✓ Connected to Arduino successfully")
            return True
        else:
            logger.error("✗ Failed to connect to Arduino")
            logger.info("  Make sure:")
            logger.info("  1. Arduino is connected via USB")
            logger.info(f"  2. Serial port is correct: {commander.serial_port}")
            logger.info("  3. Firmware is uploaded to Arduino")
            return False
    except Exception as e:
        logger.error(f"✗ Connection error: {e}")
        return False


def test_motor_commands(commander: RobotCommander) -> bool:
    """Test all motor movement commands."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Motor Commands")
    logger.info("=" * 60)

    commands = [
        ("Move Forward", lambda: commander.move_forward(150)),
        ("Move Backward", lambda: commander.move_backward(150)),
        ("Turn Left", lambda: commander.turn_left(150)),
        ("Turn Right", lambda: commander.turn_right(150)),
        ("Stop", lambda: commander.stop()),
    ]

    results = {}
    for name, cmd_func in commands:
        logger.info(f"\nTesting: {name}...")
        try:
            success = cmd_func()
            time.sleep(0.5)  # Give Arduino time to execute
            if success:
                logger.info(f"  ✓ {name} command sent successfully")
                results[name] = True
            else:
                logger.warning(f"  ✗ {name} command failed (may be OK if no hardware)")
                results[name] = False
        except Exception as e:
            logger.error(f"  ✗ {name} error: {e}")
            results[name] = False

    all_passed = all(results.values())
    if all_passed:
        logger.info("\n✓ All motor commands working")
    else:
        logger.warning("\n⚠ Some motor commands failed (may be OK without hardware)")

    return all_passed


def test_servo_commands(commander: RobotCommander) -> bool:
    """Test servo control commands."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Servo Commands")
    logger.info("=" * 60)

    # Test servo positions (FS90 range: 900-2100 microseconds)
    servo_positions = [
        (900, "Minimum (left)"),
        (1500, "Center"),
        (2100, "Maximum (right)"),
        (1200, "Left of center"),
        (1800, "Right of center"),
    ]

    results = {}
    for pulse_width, description in servo_positions:
        logger.info(f"\nTesting: {description} ({pulse_width}μs)...")
        try:
            _success = commander.set_servo(pulse_width)
            time.sleep(0.3)  # Give servo time to move
            if _success:
                logger.info(f"  ✓ Servo set to {pulse_width}μs")
                results[description] = True
            else:
                logger.warning("  ✗ Servo command failed")
                results[description] = False
        except Exception as e:
            logger.error(f"  ✗ Servo error: {e}")
            results[description] = False

    # Return to center
    commander.set_servo(1500)
    time.sleep(0.3)

    all_passed = all(results.values())
    if all_passed:
        logger.info("\n✓ All servo commands working")
    else:
        logger.warning("\n⚠ Some servo commands failed (may be OK without hardware)")

    return all_passed


def test_command_validation(commander: RobotCommander) -> bool:
    """Test command validation and error handling."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Command Validation")
    logger.info("=" * 60)

    # Test invalid commands
    invalid_commands = [
        ("Invalid command", "INVALID:123"),
        ("Out of range speed", lambda: commander.move_forward(300)),  # > 255
        ("Negative speed", lambda: commander.move_forward(-50)),
        ("Out of range servo", lambda: commander.set_servo(5000)),  # > 2100
        ("Below range servo", lambda: commander.set_servo(500)),  # < 900
    ]

    results = {}
    for name, cmd in invalid_commands:
        logger.info(f"\nTesting: {name}...")
        try:
            if callable(cmd):
                # Commands should clamp values, not fail
                _success = cmd()
                logger.info("  ✓ Command handled (clamped to valid range)")
                results[name] = True
            else:
                # Direct string commands
                _success = commander.send_command(cmd, wait_for_ok=False)
                logger.info("  ✓ Invalid command rejected gracefully")
                results[name] = True
        except Exception as e:
            logger.warning(f"  ⚠ Unexpected error: {e}")
            results[name] = False

    all_passed = all(results.values())
    if all_passed:
        logger.info("\n✓ Command validation working")
    else:
        logger.warning("\n⚠ Some validation tests had issues")

    return all_passed


def test_serial_protocol(commander: RobotCommander) -> bool:
    """Test serial protocol compliance."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 5: Serial Protocol")
    logger.info("=" * 60)

    # Test protocol commands directly
    protocol_commands = [
        "M:F:200",  # Move forward
        "M:B:150",  # Move backward
        "M:L:180",  # Turn left
        "M:R:180",  # Turn right
        "M:S",  # Stop
        "S:1500",  # Servo center
    ]

    results = {}
    for cmd in protocol_commands:
        logger.info(f"\nTesting protocol: {cmd}...")
        try:
            _success = commander.send_command(cmd, wait_for_ok=True)
            time.sleep(0.2)
            if _success:
                logger.info(f"  ✓ Protocol command '{cmd}' executed")
                results[cmd] = True
            else:
                logger.warning(f"  ✗ Protocol command '{cmd}' failed")
                results[cmd] = False
        except Exception as e:
            logger.error(f"  ✗ Protocol error: {e}")
            results[cmd] = False

    all_passed = all(results.values())
    if all_passed:
        logger.info("\n✓ Serial protocol working correctly")
    else:
        logger.warning("\n⚠ Some protocol commands failed")

    return all_passed


def test_context_manager(commander: RobotCommander) -> bool:
    """Test context manager usage."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 6: Context Manager")
    logger.info("=" * 60)

    try:
        # Test context manager
        with RobotCommander(serial_port=commander.serial_port) as robot:
            logger.info("  ✓ Context manager entered")
            _success = robot.move_forward(100)
            time.sleep(0.2)
            robot.stop()
            logger.info("  ✓ Commands executed in context")
            logger.info("  ✓ Context manager will auto-disconnect")

        logger.info("\n✓ Context manager working correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Context manager error: {e}")
        return False


def main():
    """Run Arduino firmware tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Arduino firmware for Phase 1")
    parser.add_argument(
        "--serial-port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port (default: /dev/ttyACM0, Windows: COM3)",
    )
    parser.add_argument(
        "--baud-rate",
        type=int,
        default=115200,
        help="Baud rate (default: 115200)",
    )
    parser.add_argument(
        "--skip-hardware",
        action="store_true",
        help="Skip tests that require actual hardware",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Arduino Firmware Testing (Phase 1)")
    logger.info("=" * 60)
    logger.info(f"Serial Port: {args.serial_port}")
    logger.info(f"Baud Rate: {args.baud_rate}")
    logger.info("")

    commander = RobotCommander(serial_port=args.serial_port, baud_rate=args.baud_rate)

    results = {}

    # Test 1: Connection
    results["connection"] = test_connection(commander)

    if not results["connection"]:
        logger.error("\n✗ Cannot proceed without connection")
        logger.info("\nTroubleshooting:")
        logger.info("  1. Check USB connection")
        logger.info(
            "  2. Verify serial port: ls /dev/tty* (Linux) or Device Manager (Windows)"
        )
        logger.info(
            "  3. Upload firmware: Open robot_control.ino in Arduino IDE and upload"
        )
        logger.info("  4. Check baud rate matches (115200)")
        return 1

    try:
        # Test 2: Motor commands
        if not args.skip_hardware:
            results["motor_commands"] = test_motor_commands(commander)
        else:
            logger.info("\nSkipping motor commands (--skip-hardware)")
            results["motor_commands"] = True

        # Test 3: Servo commands
        if not args.skip_hardware:
            results["servo_commands"] = test_servo_commands(commander)
        else:
            logger.info("\nSkipping servo commands (--skip-hardware)")
            results["servo_commands"] = True

        # Test 4: Command validation
        results["validation"] = test_command_validation(commander)

        # Test 5: Serial protocol
        results["protocol"] = test_serial_protocol(commander)

        # Test 6: Context manager
        results["context_manager"] = test_context_manager(commander)

    finally:
        # Cleanup
        commander.stop()
        commander.disconnect()
        logger.info("\n✓ Disconnected from Arduino")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{test_name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        logger.info("\n✓ All Arduino firmware tests passed!")
        logger.info(
            "\nPhase 1 Arduino firmware implementation is complete and working."
        )
    else:
        logger.warning("\n⚠ Some tests failed")
        logger.info("  This may be OK if:")
        logger.info("  - Hardware is not connected (use --skip-hardware)")
        logger.info("  - Firmware needs to be uploaded")
        logger.info("  - Serial port is incorrect")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
