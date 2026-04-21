"""
ROS2 node for the mission controller.

Orchestrates exploration, detection, semantic mapping, and command navigation.
Subscribes to sensors, publishes navigation goals, manages state machine.
"""

import json
import time

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from cv_bridge import CvBridge
import tf2_ros

from semantic_navigation.mission_controller import MissionController, MissionState
from semantic_navigation.bbox_to_3d import BBoxTo3DConverter, CameraIntrinsics
from semantic_navigation.semantic_map import SemanticMap
from semantic_navigation.command_parser import CommandParser

try:
    from semantic_navigation.grounding_dino_detector import GroundingDINODetector
except ImportError:
    GroundingDINODetector = None


class MissionControllerNode(Node):
    """
    ROS2 mission controller node.

    Topics:
      - Subscribes: /camera/image_raw, /camera/depth/image_raw, /odom
      - Subscribes: /user_command (String) — navigation commands
      - Publishes:  /navigation_status (String) — status updates
      - Publishes:  /semantic_map/add (String) — detected objects
      - Action client: /navigate_to_pose (Nav2)
    """

    def __init__(self):
        super().__init__("mission_controller_node")

        # Parameters
        self.declare_parameter("detection_interval_m", 1.5)
        self.declare_parameter("object_list", "")
        self.declare_parameter("enable_detector", True)
        self.declare_parameter("semantic_map_path", "")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")

        det_interval = self.get_parameter("detection_interval_m").value
        object_list = self.get_parameter("object_list").value or None
        enable_detector = self.get_parameter("enable_detector").value
        map_path = self.get_parameter("semantic_map_path").value

        # Initialize core modules
        smap = SemanticMap()
        if map_path:
            try:
                smap.load(map_path)
                self.get_logger().info(f"Loaded semantic map: {len(smap.objects)} objects")
            except FileNotFoundError:
                pass

        detector = None
        self.get_logger().info(
            f"enable_detector={enable_detector} (type={type(enable_detector).__name__})"
        )
        # Handle string "true"/"false" from LaunchConfiguration
        if isinstance(enable_detector, str):
            enable_detector = enable_detector.lower() in ("true", "1", "yes")
        if enable_detector:
            if GroundingDINODetector is None:
                self.get_logger().warn(
                    "Grounding DINO unavailable (torch not installed). "
                    "Running without detector — exploration still works."
                )
            else:
                try:
                    detector = GroundingDINODetector()
                    detector.load_model()
                    self.get_logger().info(f"Detector loaded on {detector.device}")
                except Exception as e:
                    self.get_logger().error(f"Failed to load detector: {e}")
                    detector = None
        else:
            self.get_logger().info("Detector disabled by parameter")

        intrinsics = CameraIntrinsics.turtlebot3_default()
        converter = BBoxTo3DConverter(intrinsics)

        self.controller = MissionController(
            semantic_map=smap,
            detector=detector,
            converter=converter,
            detection_interval_m=det_interval,
            object_list=object_list,
        )
        self.map_path = map_path
        self.bridge = CvBridge()

        # Sensor state
        self.latest_image = None
        self.latest_depth = None
        self.robot_position = np.zeros(3)
        self.robot_orientation = np.array([0, 0, 0, 1])  # quaternion

        # TF buffer for camera transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers
        image_topic = self.get_parameter("image_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        self.get_logger().info(f"Subscribing to image={image_topic}, depth={depth_topic}")
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_cb, 10
        )
        self.depth_sub = self.create_subscription(
            Image, depth_topic, self.depth_cb, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_cb, 10
        )
        self.cmd_sub = self.create_subscription(
            String, "/user_command", self.command_cb, 10
        )

        # Publishers
        self.status_pub = self.create_publisher(String, "/navigation_status", 10)
        self.smap_add_pub = self.create_publisher(String, "/semantic_map/add", 10)

        # Nav2 action client
        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        # Detection timer during exploration (check every 0.5s)
        self.create_timer(0.5, self.exploration_tick)
        # Periodic status log (every 30s)
        self._tick_count = 0
        self.create_timer(30.0, self.status_tick)

        # Auto-start exploration so detection runs alongside frontier_explorer_node
        self.controller.start_exploration()
        self.get_logger().info(
            f"Mission controller node ready — exploration auto-started, "
            f"detector={'LOADED' if self.controller.detector is not None else 'NONE'}"
        )

    def image_cb(self, msg: Image):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def depth_cb(self, msg: Image):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.robot_position = np.array([p.x, p.y, p.z])
        q = msg.pose.pose.orientation
        self.robot_orientation = np.array([q.x, q.y, q.z, q.w])

    def get_camera_to_world_tf(self) -> np.ndarray:
        """Get 4x4 camera-to-world transform from TF tree."""
        try:
            tf = self.tf_buffer.lookup_transform(
                "map", "camera_rgb_optical_frame", rclpy.time.Time()
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            # Build 4x4 matrix
            from scipy.spatial.transform import Rotation
            rot = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            mat = np.eye(4)
            mat[:3, :3] = rot
            mat[:3, 3] = [t.x, t.y, t.z]
            return mat
        except Exception:
            # Fallback: use odom pose (approximate)
            from scipy.spatial.transform import Rotation
            rot = Rotation.from_quat(self.robot_orientation).as_matrix()
            mat = np.eye(4)
            mat[:3, :3] = rot
            mat[:3, 3] = self.robot_position
            return mat

    def status_tick(self):
        """Periodic status log for debugging."""
        self.get_logger().info(
            f"[STATUS] state={self.controller.state.value}, "
            f"detector={'YES' if self.controller.detector else 'NO'}, "
            f"image={'YES' if self.latest_image is not None else 'NO'}, "
            f"depth={'YES' if self.latest_depth is not None else 'NO'}, "
            f"robot=[{self.robot_position[0]:.2f}, {self.robot_position[1]:.2f}], "
            f"last_det_pos={self.controller.last_detection_position}, "
            f"map_objects={len(self.controller.semantic_map.objects)}"
        )

    def exploration_tick(self):
        """Periodic check during exploration: run detection if needed."""
        if not self.controller.is_exploring():
            return
        if self.latest_image is None or self.latest_depth is None:
            # Log once to diagnose missing camera data
            if not hasattr(self, '_warned_no_camera'):
                self._warned_no_camera = True
                self.get_logger().warn(
                    f"Waiting for camera data — "
                    f"image={'OK' if self.latest_image is not None else 'MISSING'}, "
                    f"depth={'OK' if self.latest_depth is not None else 'MISSING'}"
                )
            return
        if self.controller.detector is None:
            if not hasattr(self, '_warned_no_detector'):
                self._warned_no_detector = True
                self.get_logger().error("No detector available — skipping detection")
            return
        if not self.controller.should_detect(self.robot_position):
            return
        self.get_logger().info(
            f"Running Grounding DINO detection at robot pos "
            f"[{self.robot_position[0]:.2f}, {self.robot_position[1]:.2f}]..."
        )

        try:
            tf = self.get_camera_to_world_tf()
            objects = self.controller.run_detection_at_waypoint(
                self.latest_image, self.latest_depth, tf, self.robot_position
            )
        except Exception as e:
            self.get_logger().error(f"Detection failed (will retry next waypoint): {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            # Advance last_detection_position so we don't retry immediately
            self.controller.last_detection_position = self.robot_position[:2].copy()
            return

        self.get_logger().info(f"Detection returned {len(objects)} objects")

        MIN_CONFIDENCE = 0.28  # discard low-confidence / spurious detections
        published = 0
        for obj in objects:
            if obj.confidence < MIN_CONFIDENCE:
                self.get_logger().info(
                    f"  Skipping low-conf detection: {obj.label} (conf={obj.confidence:.2f})"
                )
                continue
            msg = String()
            msg.data = json.dumps({
                "label": obj.label,
                "position": obj.position_world.tolist(),
                "confidence": obj.confidence,
            })
            self.smap_add_pub.publish(msg)
            published += 1
            self.get_logger().info(
                f"  Detected: {obj.label} at "
                f"[{obj.position_world[0]:.2f}, {obj.position_world[1]:.2f}, {obj.position_world[2]:.2f}] "
                f"(conf={obj.confidence:.2f})"
            )

        if published == 0:
            self.get_logger().info("No confident objects detected at this position")

        # Auto-save semantic map
        if self.map_path:
            try:
                self.controller.semantic_map.save(self.map_path)
                self.get_logger().info(
                    f"Semantic map saved: {len(self.controller.semantic_map.objects)} objects total"
                )
            except Exception as e:
                self.get_logger().error(f"Failed to save semantic map: {e}")

    def command_cb(self, msg: String):
        """Handle a user navigation command."""
        command_text = msg.data.strip()
        if not command_text:
            return

        self.get_logger().info(f"Received command: '{command_text}'")

        # Special commands
        if command_text.lower() == "start exploration":
            self.controller.start_exploration()
            self.publish_status("Exploration started")
            return
        if command_text.lower() == "stop exploration":
            self.controller.stop_exploration()
            self.publish_status("Exploration stopped")
            return
        if command_text.lower() == "show map":
            stats = self.controller.semantic_map.get_stats()
            self.publish_status(f"Map: {json.dumps(stats)}")
            return

        # Execute navigation command
        goal, parsed = self.controller.execute_command(command_text)
        if goal is None:
            self.publish_status(
                f"Target '{parsed.query_text}' not found in semantic map"
            )
            return

        self.publish_status(
            f"Navigating to '{parsed.query_text}' at "
            f"[{goal[0]:.2f}, {goal[1]:.2f}]"
        )
        self.send_nav_goal(goal)

    def send_nav_goal(self, position: np.ndarray):
        """Send a navigation goal to Nav2."""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(position[0])
        goal_msg.pose.pose.position.y = float(position[1])
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.w = 1.0

        self.nav_client.wait_for_server(timeout_sec=5.0)
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.nav_goal_response_cb)

    def nav_goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.publish_status("Navigation goal rejected by Nav2")
            self.controller.report_navigation_result(
                False, failure_reason="goal rejected"
            )
            return

        self.publish_status("Navigation goal accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.nav_result_cb)

    def nav_result_cb(self, future):
        result = future.result()
        success = result.status == 4  # SUCCEEDED
        self.controller.report_navigation_result(
            success=success,
            final_position=self.robot_position.tolist(),
        )
        if success:
            self.publish_status("Navigation complete — arrived at target")
        else:
            self.publish_status("Navigation failed")

    def publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text)


def main(args=None):
    rclpy.init(args=args)
    node = MissionControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
