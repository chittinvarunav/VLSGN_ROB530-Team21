"""
ROS2 node for the semantic map.

Provides services/topics to add objects, query objects, and visualize
the semantic map in RViz as colored markers.
"""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from semantic_navigation.semantic_map import SemanticMap


class SemanticMapNode(Node):
    """
    ROS2 node for semantic map management.

    Topics:
      - Subscribes: /semantic_map/add (String, JSON: {label, position, confidence})
      - Subscribes: /semantic_map/query_request (String, label to query)
      - Publishes:  /semantic_map/query_result (String, JSON list of matches)
      - Publishes:  /semantic_map/markers (MarkerArray, for RViz)
      - Publishes:  /semantic_map/stats (String, JSON stats)
    """

    def __init__(self):
        super().__init__("semantic_map_node")

        self.declare_parameter("merge_distance", 0.5)
        self.declare_parameter("save_path", "")
        self.declare_parameter("marker_publish_rate", 1.0)

        merge_dist = self.get_parameter("merge_distance").value
        self.save_path = self.get_parameter("save_path").value
        rate = self.get_parameter("marker_publish_rate").value

        self.smap = SemanticMap(merge_distance=merge_dist)

        # Load existing map if path specified
        if self.save_path:
            try:
                self.smap.load(self.save_path)
                self.get_logger().info(
                    f"Loaded map with {len(self.smap.objects)} objects"
                )
            except FileNotFoundError:
                pass

        # Subscribers
        self.add_sub = self.create_subscription(
            String, "/semantic_map/add", self.add_callback, 10
        )
        self.query_sub = self.create_subscription(
            String, "/semantic_map/query_request", self.query_callback, 10
        )

        # Publishers
        self.query_pub = self.create_publisher(
            String, "/semantic_map/query_result", 10
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, "/semantic_map/markers", 10
        )
        self.stats_pub = self.create_publisher(
            String, "/semantic_map/stats", 10
        )

        # Timer for periodic marker publishing
        self.create_timer(1.0 / rate, self.publish_markers)

        self.get_logger().info("Semantic map node ready")

    def add_callback(self, msg: String):
        """Add an object to the semantic map."""
        try:
            data = json.loads(msg.data)
            label = data["label"]
            position = data["position"]
            confidence = data.get("confidence", 0.5)

            obj = self.smap.add_object(label, position, confidence)
            self.get_logger().info(
                f"Added/updated '{obj.label}' at {obj.position} "
                f"(obs={obj.observations})"
            )

            # Auto-save
            if self.save_path:
                self.smap.save(self.save_path)

            # Publish stats
            stats_msg = String()
            stats_msg.data = json.dumps(self.smap.get_stats())
            self.stats_pub.publish(stats_msg)

        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().error(f"Invalid add request: {e}")

    def query_callback(self, msg: String):
        """Query the semantic map and publish results."""
        label = msg.data.strip()
        results = self.smap.query(label)

        response = [
            {
                "label": obj.label,
                "position": obj.position,
                "confidence": obj.confidence,
                "observations": obj.observations,
            }
            for obj in results
        ]

        result_msg = String()
        result_msg.data = json.dumps(response)
        self.query_pub.publish(result_msg)

        self.get_logger().info(
            f"Query '{label}': {len(results)} results"
        )

    def publish_markers(self):
        """Publish semantic map as RViz markers."""
        marker_array = MarkerArray()

        # Clear old markers
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        marker_data = self.smap.to_marker_data()
        for i, data in enumerate(marker_data):
            # Sphere marker for object position
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "semantic_objects"
            marker.id = i * 2
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = data["position"][0]
            marker.pose.position.y = data["position"][1]
            marker.pose.position.z = data["position"][2] if len(data["position"]) > 2 else 0.5
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r = data["color"][0]
            marker.color.g = data["color"][1]
            marker.color.b = data["color"][2]
            marker.color.a = data["color"][3]
            marker_array.markers.append(marker)

            # Text marker for label
            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "semantic_labels"
            text_marker.id = i * 2 + 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = data["position"][0]
            text_marker.pose.position.y = data["position"][1]
            text_marker.pose.position.z = (
                data["position"][2] + 0.3 if len(data["position"]) > 2 else 0.8
            )
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.15
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = f"{data['label']} ({data['confidence']:.2f})"
            marker_array.markers.append(text_marker)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
