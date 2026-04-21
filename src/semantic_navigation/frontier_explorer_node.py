"""
ROS2 node for autonomous frontier-based exploration.
Fixed: adds mandatory coverage sweep of all object positions after frontier exploration.
"""

import os
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from nav_msgs.msg import OccupancyGrid, Odometry
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from semantic_navigation.frontier_explorer import FrontierExplorer


# Known object positions — robot MUST visit near all of these
COVERAGE_WAYPOINTS = [
    [-2.0,  2.0],   # red cylinder
    [ 2.0,  2.0],   # blue box
    [ 2.5,  0.5],   # white cylinder
    [ 2.5, -2.5],   # yellow box
    [-2.0, -2.0],   # green cylinder
    [ 0.0,  0.0],   # center
]


class FrontierExplorerNode(Node):

    def __init__(self):
        super().__init__("frontier_explorer_node")

        self.declare_parameter("min_frontier_size", 3)
        self.declare_parameter("exploration_rate", 0.5)
        self.declare_parameter("goal_timeout_sec", 60.0)
        self.declare_parameter("strategy", "weighted")
        self.declare_parameter("save_dir", "")
        self.declare_parameter("exploration_seed", 42)
        self.declare_parameter("max_exploration_time_sec", 900.0)

        min_frontier = self.get_parameter("min_frontier_size").value
        rate         = self.get_parameter("exploration_rate").value
        self.goal_timeout          = self.get_parameter("goal_timeout_sec").value
        self.strategy              = self.get_parameter("strategy").value
        self.save_dir              = self.get_parameter("save_dir").value
        seed                       = self.get_parameter("exploration_seed").value
        self.max_exploration_time  = self.get_parameter("max_exploration_time_sec").value

        np.random.seed(seed)
        self.get_logger().info(f"Exploration RNG seeded with {seed}")

        self.explorer        = FrontierExplorer(min_frontier_size=min_frontier)
        self.robot_position  = np.zeros(2)
        self.latest_grid     = None
        self.latest_map_msg  = None
        self.grid_info       = None

        self.navigating           = False
        self.current_goal         = None
        self.goal_handle          = None
        self.goal_start_time      = 0.0
        self.goals_sent           = 0
        self.goals_reached        = 0
        self.goals_failed         = 0
        self.consecutive_failures = 0
        self.exploration_active   = True
        self.start_time           = None
        self.map_updates          = 0
        self.min_warmup_sec       = 30.0
        self.min_map_updates      = 5

        # Coverage sweep state
        self.frontier_done        = False
        self.coverage_waypoints   = [np.array(wp) for wp in COVERAGE_WAYPOINTS]
        self.coverage_index       = 0
        self.coverage_done        = False

        self._last_moving_position = np.zeros(2)
        self._last_move_time       = time.time()
        self._stuck_timeout_sec    = 45.0

        self.map_sub  = self.create_subscription(OccupancyGrid, "/map", self.map_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.status_pub = self.create_publisher(String, "/exploration_status", 10)
        self.timer = self.create_timer(1.0 / rate, self.exploration_tick)

        self.get_logger().info(
            f"Frontier explorer ready — strategy={self.strategy}, "
            f"min_size={min_frontier}, timeout={self.goal_timeout}s"
        )

    def map_cb(self, msg):
        info = msg.info
        self.grid_info = {
            "width":    info.width,
            "height":   info.height,
            "resolution": info.resolution,
            "origin_x": info.origin.position.x,
            "origin_y": info.origin.position.y,
        }
        self.latest_grid = np.array(msg.data, dtype=np.int8).reshape(
            info.height, info.width
        )
        self.latest_map_msg = msg
        self.map_updates += 1
        if self.start_time is None:
            self.start_time = time.time()

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        new_pos = np.array([p.x, p.y])
        if np.linalg.norm(new_pos - self._last_moving_position) > 0.3:
            self._last_moving_position = new_pos.copy()
            self._last_move_time = time.time()
        self.robot_position = new_pos

    def exploration_tick(self):
        if not self.exploration_active:
            return
        if self.latest_grid is None or self.grid_info is None:
            return

        # Hard time limit
        if self.start_time is not None:
            elapsed_total = time.time() - self.start_time
            if elapsed_total > self.max_exploration_time:
                self.publish_status("Time limit reached — starting coverage sweep")
                self.frontier_done = True

        # Check for timeout/stuck
        if self.navigating:
            now = time.time()
            elapsed    = now - self.goal_start_time
            stuck_secs = now - self._last_move_time
            if stuck_secs > self._stuck_timeout_sec and self.goal_handle is not None:
                self.publish_status(f"Robot stuck ({stuck_secs:.0f}s) — canceling goal")
                self.goal_handle.cancel_goal_async()
                self.navigating = False
                self.goals_failed += 1
                self.consecutive_failures += 1
                self._last_move_time = now
                self._last_moving_position = self.robot_position.copy()
            elif elapsed > self.goal_timeout and self.goal_handle is not None:
                self.publish_status(f"Goal timed out — canceling")
                self.goal_handle.cancel_goal_async()
                self.navigating = False
                self.goals_failed += 1
                self.consecutive_failures += 1
            return

        # ── PHASE 2: Coverage sweep of all object positions ──────────────────
        if self.frontier_done:
            if self.coverage_index >= len(self.coverage_waypoints):
                self.publish_status("Coverage sweep complete — all object positions visited!")
                self.finish_exploration()
                return
            wp = self.coverage_waypoints[self.coverage_index]
            dist = np.linalg.norm(wp - self.robot_position)
            if dist < 1.0:
                # Already close enough — skip
                self.publish_status(f"Already near waypoint {self.coverage_index+1}/{len(self.coverage_waypoints)} — skipping")
                self.coverage_index += 1
                return
            self.publish_status(
                f"Coverage waypoint {self.coverage_index+1}/{len(self.coverage_waypoints)}: "
                f"[{wp[0]:.2f}, {wp[1]:.2f}]"
            )
            self.send_nav_goal(wp)
            return

        # ── PHASE 1: Frontier exploration ─────────────────────────────────────
        info      = self.grid_info
        frontiers = self.explorer.detect_frontiers(self.latest_grid)

        if not frontiers:
            elapsed = time.time() - self.start_time if self.start_time else 0
            if elapsed < self.min_warmup_sec or self.map_updates < self.min_map_updates:
                return
            if not hasattr(self, '_no_frontier_count'):
                self._no_frontier_count = 0
            self._no_frontier_count += 1
            if self._no_frontier_count < 5:
                return
            # Frontier exploration done — start coverage sweep
            self.publish_status("No more frontiers — starting coverage sweep of object positions")
            self.frontier_done = True
            return

        if hasattr(self, '_no_frontier_count'):
            self._no_frontier_count = 0

        origin     = (info["origin_x"], info["origin_y"])
        resolution = info["resolution"]
        robot_grid = np.array([
            (self.robot_position[1] - origin[1]) / resolution,
            (self.robot_position[0] - origin[0]) / resolution,
        ])

        use_strategy = self.strategy
        if self.consecutive_failures >= 6:
            self.explorer.visited_frontiers.clear()
            self.consecutive_failures = 0
        elif self.consecutive_failures >= 4:
            self.explorer.visited_frontiers.clear()
            use_strategy = "farthest"
        elif self.consecutive_failures >= 2:
            use_strategy = "farthest"

        goal = self.explorer.select_frontier(
            frontiers, robot_grid, origin, resolution,
            strategy=use_strategy, robot_position_world=self.robot_position,
        )
        if goal is None:
            self.explorer.visited_frontiers.clear()
            goal = self.explorer.select_frontier(
                frontiers, robot_grid, origin, resolution,
                strategy="farthest", robot_position_world=self.robot_position,
            )
            if goal is None:
                self.publish_status("All frontiers visited — starting coverage sweep")
                self.frontier_done = True
                return

        self.send_nav_goal(goal)

    def finish_exploration(self):
        self.publish_status(
            f"Exploration complete! Reached {self.goals_reached}/{self.goals_sent} goals."
        )
        self.exploration_active = False
        self.save_map()

    def save_map(self):
        if self.latest_grid is None or self.grid_info is None:
            return
        save_dir = self.save_dir or os.path.join(
            os.path.expanduser("~"), "ros2_ws", "src", "ROB530_Project",
            "semantic_navigation", "data",
        )
        os.makedirs(save_dir, exist_ok=True)
        try:
            import cv2
            grid = self.latest_grid.copy()
            img  = np.full(grid.shape, 205, dtype=np.uint8)
            img[grid == 0] = 254
            img[(grid > 0) & (grid <= 100)] = 0
            img  = np.flipud(img)
            cv2.imwrite(os.path.join(save_dir, "exploration_map.png"), img)
            info = self.grid_info
            with open(os.path.join(save_dir, "exploration_map.yaml"), "w") as f:
                f.write(f"image: exploration_map.png\n")
                f.write(f"resolution: {info['resolution']}\n")
                f.write(f"origin: [{info['origin_x']}, {info['origin_y']}, 0.0]\n")
                f.write(f"negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
            self.publish_status(f"Map saved!")
        except Exception as e:
            self.get_logger().error(f"Failed to save map: {e}")

    def send_nav_goal(self, position):
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("Nav2 not available")
            return
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(position[0])
        goal_msg.pose.pose.position.y = float(position[1])
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.w = 1.0
        self.navigating      = True
        self.current_goal    = position.copy()
        self.goal_start_time = time.time()
        self.goals_sent     += 1
        dist = np.linalg.norm(position - self.robot_position)
        self.publish_status(
            f"Frontier goal #{self.goals_sent} ({self.strategy}): "
            f"[{position[0]:.2f}, {position[1]:.2f}] "
            f"(dist={dist:.2f}m, robot=[{self.robot_position[0]:.2f}, {self.robot_position[1]:.2f}])"
        )
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_cb)

    def goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected by Nav2")
            self.navigating = False
            self.goals_failed += 1
            self.consecutive_failures += 1
            if self.frontier_done:
                self.coverage_index += 1
            return
        self.goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_cb)

    def goal_result_cb(self, future):
        status = future.result().status
        self.navigating  = False
        self.goal_handle = None

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.goals_reached += 1
            self.consecutive_failures = 0
            if self.current_goal is not None and not self.frontier_done:
                self.explorer.mark_visited(self.current_goal)
            if self.frontier_done:
                self.publish_status(
                    f"Coverage waypoint {self.coverage_index+1} reached!"
                )
                self.coverage_index += 1
            else:
                self.publish_status(
                    f"Reached frontier ({self.goals_reached}/{self.goals_sent})"
                )
        elif status == GoalStatus.STATUS_CANCELED:
            self.goals_failed += 1
            self.consecutive_failures += 1
            if self.frontier_done:
                self.coverage_index += 1
        else:
            self.goals_failed += 1
            self.consecutive_failures += 1
            if self.frontier_done:
                self.coverage_index += 1

    def publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
