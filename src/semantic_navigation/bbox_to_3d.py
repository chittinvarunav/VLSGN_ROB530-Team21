"""
2D bounding box to 3D world coordinate conversion.

Converts 2D bounding box detections into 3D positions using:
  - Camera intrinsics (from URDF or calibration)
  - Depth information (from depth camera or LiDAR projection)
  - Camera-to-world transform (from TF/SLAM)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float  # focal length x
    fy: float  # focal length y
    cx: float  # principal point x
    cy: float  # principal point y
    width: int
    height: int

    @classmethod
    def from_fov(cls, hfov_deg: float, width: int, height: int):
        """Create intrinsics from horizontal field of view (common for Gazebo cameras)."""
        hfov_rad = np.radians(hfov_deg)
        fx = width / (2.0 * np.tan(hfov_rad / 2.0))
        fy = fx  # square pixels
        return cls(fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0,
                   width=width, height=height)

    @classmethod
    def turtlebot3_default(cls):
        """Default TurtleBot3 Waffle Pi camera intrinsics (Gazebo sim)."""
        # RealSense-like simulated camera: 640x480, ~69.4° HFOV
        return cls.from_fov(hfov_deg=69.4, width=640, height=480)

    @property
    def K(self) -> np.ndarray:
        """3x3 camera intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ])


@dataclass
class Object3D:
    """A 3D object detection result."""
    label: str
    position_camera: np.ndarray  # [x, y, z] in camera frame
    position_world: np.ndarray   # [x, y, z] in world/map frame
    confidence: float
    depth: float                 # depth value used (meters)


class BBoxTo3DConverter:
    """
    Converts 2D bounding box detections to 3D world coordinates.

    Supports two depth sources:
      1. Depth image (e.g., from depth camera)
      2. LiDAR point cloud projected to image plane
    """

    def __init__(self, intrinsics: CameraIntrinsics):
        self.intrinsics = intrinsics

    def pixel_to_camera_frame(self, u: float, v: float, depth: float) -> np.ndarray:
        """
        Convert pixel (u, v) + depth to 3D point in camera optical frame.

        Camera optical frame convention: z forward, x right, y down.
        """
        x = (u - self.intrinsics.cx) * depth / self.intrinsics.fx
        y = (v - self.intrinsics.cy) * depth / self.intrinsics.fy
        z = depth
        return np.array([x, y, z])

    def camera_to_world(
        self, point_camera: np.ndarray, camera_to_world_tf: np.ndarray
    ) -> np.ndarray:
        """
        Transform a 3D point from camera frame to world/map frame.

        Args:
            point_camera: [x, y, z] in camera optical frame
            camera_to_world_tf: 4x4 homogeneous transformation matrix

        Returns:
            [x, y, z] in world frame
        """
        point_h = np.append(point_camera, 1.0)
        point_world = camera_to_world_tf @ point_h
        return point_world[:3]

    def get_depth_at_bbox(
        self,
        depth_image: np.ndarray,
        bbox: list,
        method: str = "center_crop",
    ) -> Optional[float]:
        """
        Extract depth value for a bounding box region.

        Args:
            depth_image: HxW float32 depth image (meters).
            bbox: [x_min, y_min, x_max, y_max] in pixels.
            method: 'center_crop' (central 50% region median) or
                    'center_point' (single center pixel).

        Returns:
            Depth in meters, or None if invalid.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = depth_image.shape[:2]

        # Clamp to image bounds
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if method == "center_point":
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            depth = float(depth_image[cy, cx])
        elif method == "center_crop":
            # Use central 50% of the bbox to avoid edge depth noise
            bw, bh = x2 - x1, y2 - y1
            cx1 = x1 + bw // 4
            cx2 = x2 - bw // 4
            cy1 = y1 + bh // 4
            cy2 = y2 - bh // 4
            region = depth_image[cy1:cy2, cx1:cx2]
            valid = region[(region > 0.1) & (region < 10.0)]  # filter invalid
            if len(valid) == 0:
                return None
            depth = float(np.median(valid))
        else:
            raise ValueError(f"Unknown method: {method}")

        # Sanity check
        if depth <= 0.1 or depth > 15.0:
            return None
        return depth

    def lidar_to_depth_image(
        self,
        points: np.ndarray,
        camera_to_world_tf: np.ndarray,
    ) -> np.ndarray:
        """
        Project LiDAR points onto the camera image plane to create a sparse depth image.

        Args:
            points: Nx3 array of LiDAR points in world frame.
            camera_to_world_tf: 4x4 camera-to-world transform.

        Returns:
            HxW float32 sparse depth image.
        """
        world_to_camera = np.linalg.inv(camera_to_world_tf)
        K = self.intrinsics.K
        h, w = self.intrinsics.height, self.intrinsics.width

        # Transform points to camera frame
        ones = np.ones((points.shape[0], 1))
        points_h = np.hstack([points, ones])
        points_cam = (world_to_camera @ points_h.T).T[:, :3]

        # Filter points behind the camera
        mask = points_cam[:, 2] > 0.1
        points_cam = points_cam[mask]

        # Project to image plane
        proj = (K @ points_cam.T).T
        u = (proj[:, 0] / proj[:, 2]).astype(int)
        v = (proj[:, 1] / proj[:, 2]).astype(int)
        z = points_cam[:, 2]

        # Filter points outside image
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, z = u[valid], v[valid], z[valid]

        depth_image = np.zeros((h, w), dtype=np.float32)
        # Use closest point for each pixel (z-buffer)
        for ui, vi, zi in zip(u, v, z):
            if depth_image[vi, ui] == 0 or zi < depth_image[vi, ui]:
                depth_image[vi, ui] = zi

        return depth_image

    def convert(
        self,
        label: str,
        bbox: list,
        confidence: float,
        depth_image: np.ndarray,
        camera_to_world_tf: np.ndarray,
        depth_method: str = "center_crop",
    ) -> Optional[Object3D]:
        """
        Convert a single 2D detection to a 3D object.

        Args:
            label: Detected object label.
            bbox: [x_min, y_min, x_max, y_max] in pixels.
            confidence: Detection confidence.
            depth_image: HxW float32 depth image (meters).
            camera_to_world_tf: 4x4 camera-to-world homogeneous transform.
            depth_method: Depth extraction method.

        Returns:
            Object3D or None if depth is invalid.
        """
        # Get depth
        depth = self.get_depth_at_bbox(depth_image, bbox, method=depth_method)
        if depth is None:
            return None

        # Get bbox center
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        # Project to camera frame
        point_camera = self.pixel_to_camera_frame(cx, cy, depth)

        # Transform to world frame
        point_world = self.camera_to_world(point_camera, camera_to_world_tf)

        return Object3D(
            label=label,
            position_camera=point_camera,
            position_world=point_world,
            confidence=confidence,
            depth=depth,
        )

    def convert_detections(
        self,
        detections: list,
        depth_image: np.ndarray,
        camera_to_world_tf: np.ndarray,
        depth_method: str = "center_crop",
    ) -> list[Object3D]:
        """Convert a list of Detection objects to 3D."""
        results = []
        for det in detections:
            obj = self.convert(
                label=det.label,
                bbox=det.bbox,
                confidence=det.score,
                depth_image=depth_image,
                camera_to_world_tf=camera_to_world_tf,
                depth_method=depth_method,
            )
            if obj is not None:
                results.append(obj)
        return results


# ---- Standalone test ----
if __name__ == "__main__":
    # Test with synthetic data
    intrinsics = CameraIntrinsics.turtlebot3_default()
    converter = BBoxTo3DConverter(intrinsics)

    print("Camera intrinsics:")
    print(f"  fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")
    print(f"  cx={intrinsics.cx:.1f}, cy={intrinsics.cy:.1f}")
    print(f"  K=\n{intrinsics.K}")

    # Simulate a depth image with uniform 3m depth
    depth_img = np.full((480, 640), 3.0, dtype=np.float32)

    # Identity camera-to-world (camera at origin, looking along z)
    tf = np.eye(4)

    # A bbox centered in the image
    bbox = [270, 190, 370, 290]  # roughly center
    obj = converter.convert("test_object", bbox, 0.9, depth_img, tf)

    if obj:
        print(f"\nTest detection:")
        print(f"  Label: {obj.label}")
        print(f"  Camera frame: {obj.position_camera}")
        print(f"  World frame: {obj.position_world}")
        print(f"  Depth: {obj.depth}m")
    else:
        print("No valid 3D conversion (depth invalid)")
