"""Tests for Robot Commander (Arduino communication)."""

from unittest.mock import MagicMock, Mock, PropertyMock, patch

from ahmedkobtan_cinema_robot_playground.src.utils.robot_commander import (
    RobotCommander,
)


class TestRobotCommander:
    """Test Robot Commander for Arduino communication."""

    def test_initialization(self):
        """Test commander initialization."""
        commander = RobotCommander(serial_port="/dev/ttyACM0", baud_rate=115200)
        assert commander.serial_port == "/dev/ttyACM0"
        assert commander.baud_rate == 115200
        assert commander.is_connected is False

    def test_initialization_defaults(self):
        """Test commander initialization with defaults."""
        commander = RobotCommander()
        assert commander.serial_port == "/dev/ttyACM0"
        assert commander.baud_rate == 115200

    @patch("time.sleep")
    @patch("serial.Serial")
    def test_connect(self, mock_serial, mock_sleep):
        """Test serial connection."""
        mock_ser = Mock()
        mock_ser.in_waiting = 0  # No data waiting
        mock_serial.return_value = mock_ser

        commander = RobotCommander()
        result = commander.connect()

        assert result is True
        assert commander.is_connected is True
        mock_serial.assert_called_once()

    @patch("time.sleep")
    @patch("serial.Serial")
    def test_connect_failure(self, mock_serial, mock_sleep):
        """Test connection failure handling."""
        mock_serial.side_effect = Exception("Connection failed")

        commander = RobotCommander()
        result = commander.connect()

        assert result is False
        assert commander.is_connected is False

    @patch("serial.Serial")
    def test_move_forward(self, mock_serial):
        """Test move forward command."""
        mock_ser = MagicMock()
        mock_ser.in_waiting = 0
        mock_serial.return_value = mock_ser

        commander = RobotCommander()
        commander.connect()

        # Mock send_command to bypass the problematic loop
        with patch.object(commander, "send_command", return_value=True) as mock_send:
            result = commander.move_forward(200)
            assert result is True
            # Verify send_command was called with correct format
            mock_send.assert_called_once()
            call_args = mock_send.call_args[0][0]
            assert "M:F:" in call_args or call_args.startswith("M:F:")
            assert "200" in call_args

        # Cleanup
        commander.disconnect()

    @patch("serial.Serial")
    def test_move_backward(self, mock_serial):
        """Test move backward command."""
        mock_ser = MagicMock()
        mock_ser.in_waiting = 0
        mock_serial.return_value = mock_ser

        commander = RobotCommander()
        commander.connect()

        with patch.object(commander, "send_command", return_value=True) as mock_send:
            result = commander.move_backward(150)
            assert result is True
            mock_send.assert_called_once()
            call_args = mock_send.call_args[0][0]
            assert "M:B:" in call_args or call_args.startswith("M:B:")

        commander.disconnect()

    @patch("serial.Serial")
    def test_turn_left(self, mock_serial):
        """Test turn left command."""
        mock_ser = MagicMock()
        mock_ser.in_waiting = 0
        mock_serial.return_value = mock_ser

        commander = RobotCommander()
        commander.connect()

        with patch.object(commander, "send_command", return_value=True):
            result = commander.turn_left(180)
            assert result is True

        commander.disconnect()

    @patch("serial.Serial")
    def test_turn_right(self, mock_serial):
        """Test turn right command."""
        mock_ser = MagicMock()
        mock_ser.in_waiting = 0
        mock_serial.return_value = mock_ser

        commander = RobotCommander()
        commander.connect()

        with patch.object(commander, "send_command", return_value=True):
            result = commander.turn_right(180)
            assert result is True

        commander.disconnect()

    @patch("serial.Serial")
    def test_stop(self, mock_serial):
        """Test stop command."""
        mock_ser = MagicMock()
        mock_ser.in_waiting = 0
        mock_serial.return_value = mock_ser

        commander = RobotCommander()
        commander.connect()

        with patch.object(commander, "send_command", return_value=True):
            result = commander.stop()
            assert result is True

        commander.disconnect()

    @patch("serial.Serial")
    def test_set_servo(self, mock_serial):
        """Test servo command."""
        mock_ser = MagicMock()
        mock_ser.in_waiting = 0
        mock_serial.return_value = mock_ser

        commander = RobotCommander()
        commander.connect()

        with patch.object(commander, "send_command", return_value=True):
            # Test center position
            result = commander.set_servo(1500)
            assert result is True

            # Test left position
            result = commander.set_servo(900)
            assert result is True

            # Test right position
            result = commander.set_servo(2100)
            assert result is True

        commander.disconnect()

    def test_set_servo_clamping(self):
        """Test servo value clamping."""
        commander = RobotCommander()

        # Values should be clamped to valid range
        with patch.object(commander, "send_command", return_value=True) as mock_send:
            commander.set_servo(500)  # Below minimum
            # Should be clamped to 900
            call_args = mock_send.call_args[0][0]
            assert "S:900" in call_args or call_args == "S:900"

            commander.set_servo(5000)  # Above maximum
            # Should be clamped to 2100
            call_args = mock_send.call_args[0][0]
            assert "S:2100" in call_args or call_args == "S:2100"

    def test_motor_speed_clamping(self):
        """Test motor speed clamping."""
        commander = RobotCommander()

        with patch.object(commander, "send_command", return_value=True) as mock_send:
            commander.move_forward(300)  # Above 255
            # Should be clamped to 255
            call_args = mock_send.call_args[0][0]
            assert ":255" in call_args

            commander.move_forward(-50)  # Negative
            # Should be clamped to 0
            call_args = mock_send.call_args[0][0]
            assert ":0" in call_args

    def test_send_command_without_connection(self):
        """Test sending command without connection."""
        commander = RobotCommander()
        result = commander.send_command("M:F:200")

        assert result is False

    @patch("serial.Serial")
    def test_send_command_error_response(self, mock_serial):
        """Test handling of error response."""
        mock_ser = MagicMock()
        mock_ser.in_waiting = 0
        mock_serial.return_value = mock_ser

        commander = RobotCommander()
        commander.connect()

        # Mock the actual send_command implementation to test error handling
        # We'll test the error path by mocking readline to return ERROR
        with patch.object(commander, "ser") as mock_ser_obj:
            type(mock_ser_obj).in_waiting = PropertyMock(return_value=1)
            mock_ser_obj.readline.return_value = b"ERROR:Invalid command\n"
            mock_ser_obj.write = MagicMock()

            # Mock time to prevent infinite loop
            with (
                patch(
                    "ahmedkobtan_cinema_robot_playground.src.utils.robot_commander.time.time"
                ) as mock_time,
                patch(
                    "ahmedkobtan_cinema_robot_playground.src.utils.robot_commander.time.sleep"
                ),
            ):
                time_calls = [1000.0, 1000.1, 1003.0]
                call_idx = [0]

                def time_side_effect():
                    idx = call_idx[0]
                    call_idx[0] += 1
                    return time_calls[idx] if idx < len(time_calls) else 1003.0

                mock_time.side_effect = time_side_effect

                result = commander.send_command("INVALID:123", wait_for_ok=True)
                assert result is False

        commander.disconnect()

    @patch("time.sleep")
    @patch("serial.Serial")
    def test_context_manager(self, mock_serial, mock_sleep):
        """Test context manager usage."""
        mock_ser = Mock()
        mock_ser.in_waiting = 0  # No data waiting
        mock_serial.return_value = mock_ser

        with RobotCommander() as commander:
            assert commander.is_connected is True

        # Should disconnect on exit
        assert commander.is_connected is False
        mock_ser.close.assert_called_once()

    def test_disconnect(self):
        """Test disconnection."""
        commander = RobotCommander()
        commander.disconnect()
        assert commander.is_connected is False
        assert commander.ser is None
