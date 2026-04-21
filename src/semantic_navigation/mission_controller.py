"""
Mission controller — central orchestrator for the navigation pipeline.

State machine:
  IDLE → EXPLORING → MAPPING → IDLE (exploration complete)
  IDLE → NAVIGATING → IDLE (command execution)

Coordinates: command parser, semantic map, grounding DINO, Nav2 goals.
This module contains the pure logic; the ROS node wrapper handles ROS comms.
"""

import enum
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np

from .command_parser import CommandParser, ParsedCommand
from .semantic_map import SemanticMap, SemanticObject
from .bbox_to_3d import BBoxTo3DConverter, CameraIntrinsics, Object3D

# Lazy import — only needed when detector is actually used
try:
    from .grounding_dino_detector import GroundingDINODetector, Detection
except ImportError:
    GroundingDINODetector = None
    Detection = None


class MissionState(enum.Enum):
    IDLE = "idle"
    EXPLORING = "exploring"
    NAVIGATING = "navigating"
    DETECTING = "detecting"
    FAILED = "failed"


@dataclass
class NavigationResult:
    """Result of a navigation command execution."""
    success: bool
    command: str
    target_label: str
    goal_position: Optional[list] = None
    final_position: Optional[list] = None
    distance_to_goal: float = float("inf")
    time_elapsed: float = 0.0
    failure_reason: str = ""


class MissionController:
    """
    Orchestrates the full vision-language navigation pipeline.

    Standalone logic — no ROS dependencies. The ROS node wrapper
    calls these methods and provides sensor data / action execution.
    """

    def __init__(
        self,
        semantic_map: Optional[SemanticMap] = None,
        detector: Optional[GroundingDINODetector] = None,
        converter: Optional[BBoxTo3DConverter] = None,
        command_parser: Optional[CommandParser] = None,
        detection_interval_m: float = 1.5,
        object_list: Optional[str] = None,
    ):
        self.semantic_map = semantic_map or SemanticMap()
        self.detector = detector  # may be None if running without GPU
        self.converter = converter or BBoxTo3DConverter(
            CameraIntrinsics.turtlebot3_default()
        )
        self.command_parser = command_parser or CommandParser()

        self.state = MissionState.IDLE
        self.detection_interval_m = detection_interval_m
        self.last_detection_position: Optional[np.ndarray] = None

        # Object list tuned for the Gazebo single_room world.
        # The world contains geometric primitives (red cylinder, blue box, etc.)
        # so we prompt DINO with color+shape descriptors it can match.
        self.object_list = object_list or (
            "red cylinder . blue box . green cylinder . yellow box . white cylinder . "
            "red pillar . blue cube . colored obstacle . barrel . crate . pillar"
        )

        # Navigation state
        self._current_command: Optional[ParsedCommand] = None
        self._nav_start_time: float = 0.0
        self._navigation_results: list[NavigationResult] = []

    # ---- State machine ----

    def start_exploration(self):
        """Begin autonomous exploration phase."""
        self.state = MissionState.EXPLORING
        self.last_detection_position = None

    def stop_exploration(self):
        """Stop exploration and return to idle."""
        self.state = MissionState.IDLE

    def is_exploring(self) -> bool:
        return self.state == MissionState.EXPLORING

    def is_navigating(self) -> bool:
        return self.state == MissionState.NAVIGATING

    # ---- Exploration: periodic detection ----

    def should_detect(self, robot_position: np.ndarray) -> bool:
        """
        Check if we should run detection at the current position.
        Triggers every detection_interval_m meters of travel.
        """
        if self.state != MissionState.EXPLORING:
            return False
        if self.last_detection_position is None:
            return True
        dist = np.linalg.norm(robot_position[:2] - self.last_detection_position[:2])
        return dist >= self.detection_interval_m

    def run_detection_at_waypoint(
        self,
        image: np.ndarray,
        depth_image: np.ndarray,
        camera_to_world_tf: np.ndarray,
        robot_position: np.ndarray,
        text_prompt: Optional[str] = None,
    ) -> list[Object3D]:
        """
        Run Grounding DINO + 3D conversion at a waypoint during exploration.

        Args:
            image: BGR camera image
            depth_image: HxW float32 depth image (meters)
            camera_to_world_tf: 4x4 camera-to-world transform
            robot_position: [x, y, z] robot position in world frame
            text_prompt: Override object list for detection

        Returns:
            List of Object3D detections added to the semantic map.
        """
        if self.detector is None:
            return []

        prompt = text_prompt or self.object_list
        detections = self.detector.detect(image, prompt)

        # Convert to 3D
        objects_3d = self.converter.convert_detections(
            detections, depth_image, camera_to_world_tf
        )

        # Add to semantic map
        for obj in objects_3d:
            self.semantic_map.add_object(
                label=obj.label,
                position=obj.position_world.tolist(),
                confidence=obj.confidence,
            )

        self.last_detection_position = np.array(robot_position)
        return objects_3d

    # ---- Command execution ----

    def execute_command(
        self, command_text: str
    ) -> tuple[Optional[np.ndarray], ParsedCommand]:
        """
        Parse a command and find the navigation goal.

        Args:
            command_text: Natural language command.

        Returns:
            (goal_position, parsed_command) — goal is None if target not found.
        """
        parsed = self.command_parser.parse(command_text)
        self._current_command = parsed
        self._nav_start_time = time.time()

        # Query semantic map
        target = self.semantic_map.get_best_match(parsed.query_text)

        if target is None:
            # Try with just the target object (no attribute)
            target = self.semantic_map.get_best_match(parsed.target_object)

        if target is None:
            self.state = MissionState.FAILED
            return None, parsed

        goal = np.array(target.position)

        # If there's a spatial constraint, try to refine
        if parsed.spatial_reference:
            ref_obj = self.semantic_map.get_best_match(parsed.spatial_reference)
            if ref_obj is not None:
                # Among all matches for the target, find the one closest to ref
                candidates = self.semantic_map.query(parsed.query_text)
                if not candidates:
                    candidates = self.semantic_map.query(parsed.target_object)
                if candidates:
                    ref_pos = np.array(ref_obj.position)
                    best = min(
                        candidates,
                        key=lambda c: np.linalg.norm(c.position_np - ref_pos),
                    )
                    goal = np.array(best.position)

        self.state = MissionState.NAVIGATING
        return goal, parsed

    def report_navigation_result(
        self,
        success: bool,
        final_position: Optional[list] = None,
        goal_position: Optional[list] = None,
        failure_reason: str = "",
    ) -> NavigationResult:
        """Record the result of a navigation attempt."""
        elapsed = time.time() - self._nav_start_time
        dist = float("inf")
        if final_position is not None and goal_position is not None:
            dist = float(
                np.linalg.norm(
                    np.array(final_position[:2]) - np.array(goal_position[:2])
                )
            )

        result = NavigationResult(
            success=success,
            command=self._current_command.raw_text if self._current_command else "",
            target_label=(
                self._current_command.query_text if self._current_command else ""
            ),
            goal_position=goal_position,
            final_position=final_position,
            distance_to_goal=dist,
            time_elapsed=elapsed,
            failure_reason=failure_reason,
        )
        self._navigation_results.append(result)
        self.state = MissionState.IDLE
        return result

    def get_results(self) -> list[NavigationResult]:
        """Return all navigation results."""
        return list(self._navigation_results)

    def get_success_rate(self, threshold: float = 0.5) -> float:
        """Compute success rate (within threshold meters of goal)."""
        if not self._navigation_results:
            return 0.0
        successes = sum(
            1 for r in self._navigation_results if r.distance_to_goal <= threshold
        )
        return successes / len(self._navigation_results)

    # ---- Active detection during navigation ----

    def detect_target_live(
        self,
        image: np.ndarray,
        depth_image: np.ndarray,
        camera_to_world_tf: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        During navigation, run detection for the current target to refine the goal.

        Returns updated goal position if target is detected, else None.
        """
        if self.detector is None or self._current_command is None:
            return None

        prompt = self._current_command.dino_prompt
        detections = self.detector.detect(image, prompt)

        if not detections:
            return None

        # Convert best detection to 3D
        objects_3d = self.converter.convert_detections(
            detections[:1], depth_image, camera_to_world_tf
        )
        if objects_3d:
            return objects_3d[0].position_world

        return None


# ---- Standalone test ----
if __name__ == "__main__":
    # Test without actual model (semantic map query only)
    smap = SemanticMap()
    smap.add_object("blue vase", [3.0, 1.5, 0.8], 0.85)
    smap.add_object("red chair", [1.0, 2.0, 0.0], 0.90)
    smap.add_object("table", [2.0, 2.5, 0.0], 0.88)
    smap.add_object("sofa", [2.5, 3.0, 0.0], 0.92)
    smap.add_object("chair", [4.0, 1.0, 0.0], 0.75)

    controller = MissionController(semantic_map=smap, detector=None)

    test_commands = [
        "Go to the blue vase",
        "Find the red chair",
        "Navigate to the chair near the sofa",
        "Go to the lamp",  # not in map
    ]

    for cmd in test_commands:
        goal, parsed = controller.execute_command(cmd)
        if goal is not None:
            print(f"Command: \"{cmd}\"")
            print(f"  Target: \"{parsed.query_text}\"")
            print(f"  Goal: [{goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f}]")
            controller.report_navigation_result(True, goal.tolist(), goal.tolist())
        else:
            print(f"Command: \"{cmd}\" → TARGET NOT FOUND")
            controller.report_navigation_result(False, failure_reason="target not in map")
        print()

    print(f"Success rate: {controller.get_success_rate():.0%}")
