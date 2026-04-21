"""
ROS2 node wrapping Grounding DINO detector.

Subscribes to camera images, provides a service for object detection.
Publishes annotated images with detection overlays.
"""

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

try:
    from semantic_navigation.grounding_dino_detector import GroundingDINODetector
except ImportError:
    GroundingDINODetector = None

# Custom service definition would go in a separate package.
# For now we use a topic-based request/response pattern.


class GroundingDINONode(Node):
    """
    ROS2 node for Grounding DINO object detection.

    Topics:
      - Subscribes: /camera/image_raw (sensor_msgs/Image)
      - Subscribes: /detect_request (std_msgs/String) — text prompt
      - Publishes:  /detections (std_msgs/String) — JSON detection results
      - Publishes:  /detections_image (sensor_msgs/Image) — annotated image
    """

    def __init__(self):
        super().__init__("grounding_dino_node")

        # Parameters
        self.declare_parameter("model_id", "IDEA-Research/grounding-dino-base")
        self.declare_parameter("device", "")
        self.declare_parameter("box_threshold", 0.3)
        self.declare_parameter("text_threshold", 0.25)

        model_id = self.get_parameter("model_id").value
        device = self.get_parameter("device").value or None
        box_thresh = self.get_parameter("box_threshold").value
        text_thresh = self.get_parameter("text_threshold").value

        # Initialize detector
        if GroundingDINODetector is None:
            self.get_logger().error(
                "PyTorch/transformers not available. "
                "Activate sem_nav_venv before launching, or set enable_detector:=false"
            )
            self.detector = None
        else:
            try:
                self.detector = GroundingDINODetector(
                    model_id=model_id,
                    device=device,
                    box_threshold=box_thresh,
                    text_threshold=text_thresh,
                )
                self.detector.load_model()
                self.get_logger().info(
                    f"Grounding DINO loaded on {self.detector.device}"
                )
            except Exception as e:
                self.get_logger().error(f"Failed to load Grounding DINO: {e}")
                self.detector = None

        self.bridge = CvBridge()
        self.latest_image: np.ndarray | None = None
        self.latest_text_prompt: str = ""

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, 10
        )
        self.detect_sub = self.create_subscription(
            String, "/detect_request", self.detect_callback, 10
        )

        # Publishers
        self.det_pub = self.create_publisher(String, "/detections", 10)
        self.det_img_pub = self.create_publisher(Image, "/detections_image", 10)

        self.get_logger().info("Grounding DINO node ready")

    def image_callback(self, msg: Image):
        """Store latest camera image."""
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def detect_callback(self, msg: String):
        """
        Run detection when a text prompt is received.

        Publishes detection results as JSON and annotated image.
        """
        if self.detector is None:
            return
        if self.latest_image is None:
            self.get_logger().warn("No image available for detection")
            return

        text_prompt = msg.data
        if not text_prompt:
            return

        self.get_logger().info(f"Detecting: '{text_prompt}'")
        detections = self.detector.detect(self.latest_image, text_prompt)

        # Publish results as JSON
        import json
        results = [
            {
                "label": d.label,
                "bbox": d.bbox,
                "score": d.score,
                "center": list(d.center),
            }
            for d in detections
        ]
        result_msg = String()
        result_msg.data = json.dumps(results)
        self.det_pub.publish(result_msg)

        # Publish annotated image
        vis = self.detector.visualize(self.latest_image, detections)
        img_msg = self.bridge.cv2_to_imgmsg(vis, "bgr8")
        self.det_img_pub.publish(img_msg)

        self.get_logger().info(f"Published {len(detections)} detections")


def main(args=None):
    rclpy.init(args=args)
    node = GroundingDINONode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
