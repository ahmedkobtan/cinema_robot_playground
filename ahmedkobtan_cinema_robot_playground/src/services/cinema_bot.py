"""Main Cinema Bot service - orchestrates all agents."""

import argparse
import time
from typing import Optional

from loguru import logger

from ahmedkobtan_cinema_robot_playground.src.services.cinematographer_agent import (
    CinematographerAgent,
)
from ahmedkobtan_cinema_robot_playground.src.services.director_agent import (
    DirectorAgent,
)
from ahmedkobtan_cinema_robot_playground.src.services.pilot_agent import PilotAgent
from ahmedkobtan_cinema_robot_playground.src.services.video_stream import VideoStream
from ahmedkobtan_cinema_robot_playground.src.utils.robot_commander import RobotCommander


class CinemaBot:
    """Main Cinema Bot orchestrator."""

    def __init__(
        self,
        stream_url: Optional[str] = None,
        serial_port: str = "/dev/ttyACM0",
        use_llm: bool = False,
        mock: bool = False,
    ):
        """
        Initialize Cinema Bot.

        Args:
            stream_url: Video stream URL (None for local camera)
            serial_port: Serial port for Arduino
            use_llm: Whether to use LLM for command parsing
            mock: Whether to use mock models (for testing without GPU)
        """
        self.stream_url = stream_url
        self.serial_port = serial_port
        self.use_llm = use_llm
        self.mock = mock

        # Initialize components
        self.video_stream: Optional[VideoStream] = None
        self.robot_commander: Optional[RobotCommander] = None
        self.director_agent: Optional[DirectorAgent] = None
        self.cinematographer_agent: Optional[CinematographerAgent] = None
        self.pilot_agent: Optional[PilotAgent] = None

        # State
        self.is_running = False
        self.current_command: Optional[str] = None
        self.tracking_initialized = False

    def initialize(self) -> bool:
        """Initialize all components."""
        try:
            logger.info("Initializing Cinema Bot...")

            # Video stream
            self.video_stream = VideoStream(stream_url=self.stream_url)
            if not self.video_stream.connect():
                logger.error("Failed to connect to video stream")
                return False

            # Get frame size
            frame_width, frame_height = self.video_stream.get_frame_size()

            # Robot commander
            self.robot_commander = RobotCommander(serial_port=self.serial_port)
            if not self.robot_commander.connect():
                logger.warning(
                    "Failed to connect to robot (continuing without hardware)"
                )

            # Agents
            self.director_agent = DirectorAgent(use_llm=self.use_llm)
            self.cinematographer_agent = CinematographerAgent(
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if self.robot_commander.is_connected:
                self.pilot_agent = PilotAgent(
                    self.robot_commander,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )

            logger.info("Cinema Bot initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing Cinema Bot: {e}")
            return False

    def process_command(self, command: str) -> bool:
        """
        Process user command.

        Args:
            command: User command string

        Returns:
            True if command processed successfully
        """
        if not self.video_stream or not self.video_stream.is_streaming:
            logger.error("Video stream not available")
            return False

        # Read frame
        success, frame = self.video_stream.read_frame()
        if not success or frame is None:
            logger.error("Failed to read frame")
            return False

        # Director Agent: Parse command and ground object
        result = self.director_agent.process_command(command, frame)
        if result is None:
            logger.error("Failed to ground object")
            return False

        # Initialize tracking
        bbox = result["bounding_box"]
        initial_bbox = (bbox.x, bbox.y, bbox.width, bbox.height)
        if not self.cinematographer_agent.initialize_tracking(frame, initial_bbox):
            logger.error("Failed to initialize tracking")
            return False

        self.current_command = command
        self.tracking_initialized = True
        logger.info(f"Command processed: {command}")
        return True

    def run(self, command: Optional[str] = None) -> None:
        """
        Run main control loop.

        Args:
            command: Initial command to execute
        """
        if not self.initialize():
            logger.error("Failed to initialize Cinema Bot")
            return

        if command:
            if not self.process_command(command):
                logger.error("Failed to process initial command")
                return

        self.is_running = True
        logger.info("Starting control loop...")

        frame_count = 0
        last_time = time.time()

        try:
            while self.is_running:
                # Read frame
                success, frame = self.video_stream.read_frame()
                if not success or frame is None:
                    logger.warning("Failed to read frame, skipping...")
                    time.sleep(0.1)
                    continue

                # Update tracking
                tracking_state = self.cinematographer_agent.update_tracking(frame)
                if tracking_state is None:
                    logger.warning("Tracking lost, attempting re-acquisition...")
                    # Try to re-acquire
                    if self.current_command:
                        if not self.process_command(self.current_command):
                            logger.error("Failed to re-acquire object")
                            time.sleep(1.0)
                            continue
                    else:
                        time.sleep(0.1)
                        continue

                # Get shot type from director
                if self.current_command:
                    parsed = self.director_agent.parse_command(self.current_command)
                    shot_type = parsed.get("shot_type", "follow")
                else:
                    shot_type = "follow"

                # Cinematographer: Plan trajectory
                trajectory = self.cinematographer_agent.plan_trajectory(
                    shot_type, tracking_state
                )

                # Pilot: Execute trajectory
                if self.pilot_agent:
                    self.pilot_agent.execute_trajectory(trajectory, tracking_state)

                # Performance monitoring
                frame_count += 1
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    fps = frame_count / (current_time - last_time)
                    logger.debug(f"FPS: {fps:.2f}")
                    frame_count = 0
                    last_time = current_time

                # Maintain ~30 FPS
                time.sleep(0.033)

        except KeyboardInterrupt:
            logger.info("Stopping Cinema Bot...")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop Cinema Bot."""
        self.is_running = False
        if self.pilot_agent:
            self.pilot_agent.stop()
        if self.robot_commander:
            self.robot_commander.disconnect()
        if self.video_stream:
            self.video_stream.disconnect()
        logger.info("Cinema Bot stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Cinema Bot - AI-powered cinema robot")
    parser.add_argument(
        "--stream-url",
        type=str,
        default=None,
        help="Video stream URL (e.g., http://192.168.1.100:8080/video)",
    )
    parser.add_argument(
        "--serial-port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for Arduino (default: /dev/ttyACM0)",
    )
    parser.add_argument(
        "--command",
        type=str,
        required=True,
        help="Command to execute (e.g., 'Orbit the red cup')",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for command parsing (requires Ollama)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock models (for testing without GPU)",
    )

    args = parser.parse_args()

    bot = CinemaBot(
        stream_url=args.stream_url,
        serial_port=args.serial_port,
        use_llm=args.use_llm,
        mock=args.mock,
    )

    bot.run(command=args.command)


if __name__ == "__main__":
    main()
