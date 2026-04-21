"""
Frontier-based exploration node.

Detects frontiers (boundaries between known free space and unknown space)
in the occupancy grid and sends the robot to explore them one by one.
"""

import numpy as np
from typing import Optional
from scipy import ndimage


class FrontierExplorer:
    """
    Frontier detection and selection from occupancy grid.

    Standalone logic — no ROS dependencies.
    Frontiers are cells that are free (0) and adjacent to unknown (-1).
    """

    # Occupancy grid thresholds (Cartographer uses 0-100 probability values)
    FREE_THRESHOLD = 50       # cells with value < this are considered free
    OCCUPIED_THRESHOLD = 50   # cells with value >= this are considered occupied
    UNKNOWN = -1

    def __init__(
        self,
        min_frontier_size: int = 10,
        robot_radius_cells: int = 5,
        min_goal_distance: float = 0.5,
    ):
        """
        Args:
            min_frontier_size: Minimum number of cells to consider as a valid frontier.
            robot_radius_cells: Robot radius in grid cells (for filtering unreachable frontiers).
            min_goal_distance: Minimum distance (meters) a goal must be from robot to be selected.
        """
        self.min_frontier_size = min_frontier_size
        self.robot_radius_cells = robot_radius_cells
        self.min_goal_distance = min_goal_distance
        self.visited_frontiers: list[np.ndarray] = []
        self.visit_radius = 0.8  # meters — mark frontier visited if within this

    def detect_frontiers(self, occupancy_grid: np.ndarray) -> list[np.ndarray]:
        """
        Detect frontier cells in the occupancy grid.

        Args:
            occupancy_grid: 2D array where 0-100=probability, -1=unknown.

        Returns:
            List of frontier clusters, each a Nx2 array of (row, col) indices.
        """
        h, w = occupancy_grid.shape

        # Free cells: known cells with low occupancy probability
        free_mask = (occupancy_grid >= 0) & (occupancy_grid < self.FREE_THRESHOLD)

        # Unknown cells
        unknown_mask = occupancy_grid == self.UNKNOWN

        # Frontier cells: free cells adjacent to at least one unknown cell
        # Dilate unknown region by 1 pixel to find adjacency
        unknown_dilated = ndimage.binary_dilation(unknown_mask, iterations=1)
        frontier_mask = free_mask & unknown_dilated

        if not np.any(frontier_mask):
            return []

        # Cluster frontier cells into connected components
        labeled, num_features = ndimage.label(frontier_mask)
        frontiers = []
        for i in range(1, num_features + 1):
            cells = np.argwhere(labeled == i)
            if len(cells) >= self.min_frontier_size:
                frontiers.append(cells)

        return frontiers

    def frontier_centroid(self, frontier_cells: np.ndarray) -> np.ndarray:
        """Compute centroid of a frontier cluster (row, col)."""
        return frontier_cells.mean(axis=0)

    def grid_to_world(
        self, row: float, col: float,
        origin_x: float, origin_y: float, resolution: float,
    ) -> np.ndarray:
        """Convert grid (row, col) to world (x, y)."""
        x = origin_x + col * resolution
        y = origin_y + row * resolution
        return np.array([x, y])

    def select_frontier(
        self,
        frontiers: list[np.ndarray],
        robot_position_grid: np.ndarray,
        origin: tuple[float, float],
        resolution: float,
        strategy: str = "closest",
        robot_position_world: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Select the best frontier to explore.

        Args:
            frontiers: List of frontier clusters.
            robot_position_grid: Robot position in grid coords (row, col).
            origin: Map origin (x, y) in world coords.
            resolution: Map resolution (m/cell).
            strategy: 'closest' (nearest to robot) or 'largest' (most cells).
            robot_position_world: Robot [x, y] in world coords (for min distance filter).

        Returns:
            World position [x, y] of selected frontier, or None.
        """
        if not frontiers:
            return None

        # Filter out already-visited frontiers and too-close frontiers
        candidates = []
        for frontier in frontiers:
            centroid = self.frontier_centroid(frontier)
            world_pos = self.grid_to_world(
                centroid[0], centroid[1], origin[0], origin[1], resolution
            )
            visited = any(
                np.linalg.norm(world_pos - v) < self.visit_radius
                for v in self.visited_frontiers
            )
            if visited:
                continue
            # Skip frontiers too close to robot
            if robot_position_world is not None:
                dist = np.linalg.norm(world_pos - robot_position_world[:2])
                if dist < self.min_goal_distance:
                    continue
            candidates.append((frontier, centroid, world_pos))

        if not candidates:
            return None

        if strategy == "closest":
            # Pick frontier closest to robot.
            # Secondary key: (world_x, world_y) for deterministic tie-breaking.
            best = min(
                candidates,
                key=lambda c: (
                    np.linalg.norm(c[1] - robot_position_grid),
                    float(c[2][0]),
                    float(c[2][1]),
                ),
            )
        elif strategy == "largest":
            # Pick largest frontier; break ties by world position for determinism.
            best = max(
                candidates,
                key=lambda c: (
                    len(c[0]),
                    -float(c[2][0]),
                    -float(c[2][1]),
                ),
            )
        elif strategy == "weighted":
            # Score = frontier_size / (grid_distance + 1).
            # Naturally balances info-gain vs travel cost — best for full-room coverage.
            best = max(
                candidates,
                key=lambda c: (
                    len(c[0]) / (np.linalg.norm(c[1] - robot_position_grid) + 1.0),
                    float(c[2][0]),
                    float(c[2][1]),
                ),
            )
        elif strategy == "farthest":
            # Pick the frontier farthest from the robot — breaks out of stuck corners.
            best = max(
                candidates,
                key=lambda c: (
                    np.linalg.norm(c[1] - robot_position_grid),
                    float(c[2][0]),
                    float(c[2][1]),
                ),
            )
        else:
            best = candidates[0]

        return best[2]  # world position

    def mark_visited(self, position: np.ndarray):
        """Mark a frontier position as visited."""
        self.visited_frontiers.append(position.copy())

    def is_exploration_complete(self, occupancy_grid: np.ndarray) -> bool:
        """Check if there are no more reachable frontiers."""
        frontiers = self.detect_frontiers(occupancy_grid)
        return len(frontiers) == 0


class FrontierExplorerNode:
    """
    ROS2 node wrapper for frontier exploration.

    This is a template — actual ROS2 imports and node setup go here
    when running on Ubuntu with ROS2.
    """

    def __init__(self):
        # This would be: super().__init__("frontier_explorer_node")
        # when inheriting from rclpy.node.Node
        self.explorer = FrontierExplorer()
        self.exploring = False
        self.current_goal = None

    def start(self):
        self.exploring = True

    def stop(self):
        self.exploring = False

    def occupancy_grid_callback(self, grid_data, info):
        """
        Called when a new occupancy grid is received.

        In ROS2, this subscribes to /map (nav_msgs/OccupancyGrid).
        """
        if not self.exploring:
            return

        grid = np.array(grid_data).reshape(info["height"], info["width"])
        resolution = info["resolution"]
        origin = (info["origin_x"], info["origin_y"])

        frontiers = self.explorer.detect_frontiers(grid)

        if not frontiers:
            print("Exploration complete — no more frontiers")
            self.exploring = False
            return

        # Robot position in grid coords
        robot_world = np.array(info.get("robot_position", [0, 0]))
        robot_grid = np.array([
            (robot_world[1] - origin[1]) / resolution,
            (robot_world[0] - origin[0]) / resolution,
        ])

        goal = self.explorer.select_frontier(
            frontiers, robot_grid, origin, resolution
        )

        if goal is not None:
            self.current_goal = goal
            self.explorer.mark_visited(goal)
            print(f"Next frontier goal: [{goal[0]:.2f}, {goal[1]:.2f}]")
            # In ROS2: send goal to Nav2 action server
        else:
            print("No unvisited frontiers remaining")
            self.exploring = False


# ---- Standalone test ----
if __name__ == "__main__":
    # Create a synthetic occupancy grid
    grid = np.full((100, 100), -1, dtype=np.int8)  # all unknown

    # Carve out some free space (explored region)
    grid[30:70, 20:60] = 0  # free
    grid[35:65, 25:55] = 0  # more free

    # Add some walls
    grid[40, 20:60] = 100
    grid[30:70, 40] = 100

    explorer = FrontierExplorer(min_frontier_size=3)
    frontiers = explorer.detect_frontiers(grid)

    print(f"Found {len(frontiers)} frontiers")
    for i, f in enumerate(frontiers):
        centroid = explorer.frontier_centroid(f)
        print(f"  Frontier {i}: {len(f)} cells, centroid=({centroid[0]:.0f}, {centroid[1]:.0f})")

    robot_pos = np.array([50, 40])
    goal = explorer.select_frontier(
        frontiers, robot_pos, origin=(0.0, 0.0), resolution=0.05
    )
    if goal is not None:
        print(f"\nSelected goal: [{goal[0]:.2f}, {goal[1]:.2f}]m")
