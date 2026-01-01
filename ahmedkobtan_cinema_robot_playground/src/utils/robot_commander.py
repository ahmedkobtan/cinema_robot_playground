"""Robot commander for Arduino communication - replaces 3D printer commander."""

import logging
import time
from typing import Optional

import serial

logger = logging.getLogger(__name__)

# Serial protocol commands
CMD_MOVE_FORWARD = "M:F"
CMD_MOVE_BACKWARD = "M:B"
CMD_TURN_LEFT = "M:L"
CMD_TURN_RIGHT = "M:R"
CMD_STOP = "M:S"
CMD_SERVO = "S"


class RobotCommander:
    """Handles serial communication with Arduino for robot control."""

    def __init__(self, serial_port: str = "/dev/ttyACM0", baud_rate: int = 115200):
        """
        Initialize robot commander.

        Args:
            serial_port: Serial port path (e.g., "/dev/ttyACM0" on Linux, "COM3" on Windows)
            baud_rate: Serial communication baud rate
        """
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.ser: Optional[serial.Serial] = None
        self.is_connected = False

    def connect(self) -> bool:
        """
        Connect to Arduino via serial.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            logger.info(f"Connected to {self.serial_port} @ {self.baud_rate}")
            time.sleep(2)  # Wait for connection to settle

            # Clear startup garbage
            while self.ser.in_waiting:
                line = self.ser.readline().decode().strip()
                logger.debug(f"Robot startup: {line}")

            self.is_connected = True
            return True
        except Exception as e:
            logger.error(f"Error connecting to robot: {e}")
            return False

    def send_command(self, command: str, wait_for_ok: bool = True) -> bool:
        """
        Send command to Arduino.

        Args:
            command: Command string (e.g., "M:F:200")
            wait_for_ok: Whether to wait for "OK" response

        Returns:
            True if command sent successfully, False otherwise
        """
        if not self.is_connected or self.ser is None:
            logger.error("Robot not connected")
            return False

        try:
            # Remove comments and whitespace
            clean_cmd = command.split(";")[0].strip()
            if not clean_cmd:
                return False

            logger.debug(f"Sending command: {clean_cmd}")
            self.ser.write((clean_cmd + "\n").encode())

            if wait_for_ok:
                # Wait for OK response
                timeout = time.time() + 2.0  # 2 second timeout
                while time.time() < timeout:
                    if self.ser.in_waiting:
                        line = self.ser.readline().decode().strip()
                        logger.debug(f"Robot response: {line}")
                        if "OK" in line.upper() or "ok" in line.lower():
                            return True
                        if "ERROR" in line.upper():
                            logger.error(f"Robot error: {line}")
                            return False
                    time.sleep(0.01)
                logger.warning("Timeout waiting for OK response")
                return False

            return True
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return False

    def move_forward(self, speed: int) -> bool:
        """
        Move robot forward.

        Args:
            speed: Motor speed (0-255)

        Returns:
            True if command sent successfully
        """
        speed = max(0, min(255, speed))  # Clamp to valid range
        return self.send_command(f"{CMD_MOVE_FORWARD}:{speed}")

    def move_backward(self, speed: int) -> bool:
        """
        Move robot backward.

        Args:
            speed: Motor speed (0-255)

        Returns:
            True if command sent successfully
        """
        speed = max(0, min(255, speed))  # Clamp to valid range
        return self.send_command(f"{CMD_MOVE_BACKWARD}:{speed}")

    def turn_left(self, speed: int) -> bool:
        """
        Turn robot left.

        Args:
            speed: Motor speed (0-255)

        Returns:
            True if command sent successfully
        """
        speed = max(0, min(255, speed))  # Clamp to valid range
        return self.send_command(f"{CMD_TURN_LEFT}:{speed}")

    def turn_right(self, speed: int) -> bool:
        """
        Turn robot right.

        Args:
            speed: Motor speed (0-255)

        Returns:
            True if command sent successfully
        """
        speed = max(0, min(255, speed))  # Clamp to valid range
        return self.send_command(f"{CMD_TURN_RIGHT}:{speed}")

    def stop(self) -> bool:
        """
        Stop robot movement.

        Returns:
            True if command sent successfully
        """
        return self.send_command(CMD_STOP)

    def set_servo(self, pulse_width_us: int) -> bool:
        """
        Set servo angle via pulse width.

        Args:
            pulse_width_us: Pulse width in microseconds (900-2100 for FS90)

        Returns:
            True if command sent successfully
        """
        pulse_width_us = max(900, min(2100, pulse_width_us))  # Clamp to FS90 range
        return self.send_command(f"{CMD_SERVO}:{pulse_width_us}")

    def disconnect(self) -> None:
        """Disconnect from Arduino."""
        if self.ser is not None:
            self.ser.close()
            self.ser = None
        self.is_connected = False
        logger.info("Disconnected from robot")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
