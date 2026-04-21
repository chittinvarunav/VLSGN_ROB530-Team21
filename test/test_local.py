"""
Local test script — works on Mac without ROS2.

Tests all standalone modules:
  1. Command parser
  2. Semantic map (add, query, save/load)
  3. BBox to 3D conversion
  4. Frontier explorer
  5. Mission controller (with pre-built semantic map)
  6. Grounding DINO (optional, requires GPU or MPS)

Usage:
  cd semantic_navigation
  python test/test_local.py               # skip Grounding DINO
  python test/test_local.py --with-dino   # include Grounding DINO (needs GPU/MPS)
"""

import sys
import os
import time
import tempfile
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"


def test_command_parser():
    """Test command parsing module."""
    print("\n--- Test: Command Parser ---")
    from semantic_navigation.command_parser import CommandParser

    parser = CommandParser()

    tests = [
        ("Go to the blue vase", "blue", "vase", "navigate"),
        ("Find the red chair in the bedroom", "red", "chair", "find"),
        ("Navigate to the table near the sofa", "", "table", "navigate"),
        ("Explore the hallway", "", "hallway", "explore"),
        ("Where is the white cabinet", "white", "cabinet", "find"),
    ]

    passed = 0
    for cmd, exp_attr, exp_target, exp_action in tests:
        result = parser.parse(cmd)
        ok = (
            result.target_attribute == exp_attr
            and exp_target in result.target_object
            and result.action == exp_action
        )
        status = PASS if ok else FAIL
        print(f"  {status} \"{cmd}\" → attr='{result.target_attribute}', "
              f"target='{result.target_object}', action='{result.action}'")
        if ok:
            passed += 1

    print(f"  {passed}/{len(tests)} tests passed")
    return passed == len(tests)


def test_semantic_map():
    """Test semantic map module."""
    print("\n--- Test: Semantic Map ---")
    from semantic_navigation.semantic_map import SemanticMap

    smap = SemanticMap(merge_distance=0.5)

    # Add objects
    smap.add_object("red chair", [1.0, 2.0, 0.0], 0.85)
    smap.add_object("blue vase", [3.0, 1.5, 0.8], 0.72)
    smap.add_object("table", [2.0, 2.5, 0.0], 0.90)
    smap.add_object("red chair", [1.1, 2.05, 0.0], 0.88)  # should merge

    # Test merge
    stats = smap.get_stats()
    merge_ok = stats["total_objects"] == 3
    print(f"  {PASS if merge_ok else FAIL} Merge: 4 adds → {stats['total_objects']} objects (expect 3)")

    # Test query
    results = smap.query("chair")
    query_ok = len(results) == 1 and results[0].label == "red chair"
    print(f"  {PASS if query_ok else FAIL} Query 'chair': {len(results)} results")

    results = smap.query("blue vase")
    query2_ok = len(results) == 1
    print(f"  {PASS if query2_ok else FAIL} Query 'blue vase': {len(results)} results")

    # Test nearest query
    nearest = smap.query_nearest([1.0, 2.0, 0.0], max_distance=2.0)
    nearest_ok = len(nearest) >= 1
    print(f"  {PASS if nearest_ok else FAIL} Nearest query: {len(nearest)} objects within 2m")

    # Test save/load
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name
    smap.save(tmp_path)
    smap2 = SemanticMap()
    smap2.load(tmp_path)
    save_ok = len(smap2.objects) == len(smap.objects)
    print(f"  {PASS if save_ok else FAIL} Save/Load: {len(smap2.objects)} objects loaded")
    os.unlink(tmp_path)

    return merge_ok and query_ok and query2_ok and nearest_ok and save_ok


def test_bbox_to_3d():
    """Test 2D to 3D conversion."""
    print("\n--- Test: BBox to 3D ---")
    from semantic_navigation.bbox_to_3d import BBoxTo3DConverter, CameraIntrinsics

    intrinsics = CameraIntrinsics.turtlebot3_default()
    converter = BBoxTo3DConverter(intrinsics)

    # Center of image, 3m depth, identity transform
    depth_img = np.full((480, 640), 3.0, dtype=np.float32)
    tf = np.eye(4)

    # Bbox roughly at image center
    bbox = [270, 190, 370, 290]  # center at (320, 240)
    obj = converter.convert("test", bbox, 0.9, depth_img, tf)

    center_ok = obj is not None
    if obj is not None:
        # At image center with 3m depth, x should be ~0, y should be ~0
        x_ok = abs(obj.position_world[0]) < 0.2
        y_ok = abs(obj.position_world[1]) < 0.2
        z_ok = abs(obj.position_world[2] - 3.0) < 0.1
        print(f"  {PASS if x_ok else FAIL} X position: {obj.position_world[0]:.3f} (expect ~0)")
        print(f"  {PASS if y_ok else FAIL} Y position: {obj.position_world[1]:.3f} (expect ~0)")
        print(f"  {PASS if z_ok else FAIL} Z position: {obj.position_world[2]:.3f} (expect ~3.0)")
        return x_ok and y_ok and z_ok
    else:
        print(f"  {FAIL} No 3D conversion result")
        return False


def test_frontier_explorer():
    """Test frontier detection."""
    print("\n--- Test: Frontier Explorer ---")
    from semantic_navigation.frontier_explorer import FrontierExplorer

    # Create synthetic occupancy grid
    grid = np.full((100, 100), -1, dtype=np.int8)  # all unknown
    grid[30:70, 20:60] = 0  # free space
    grid[50, 20:60] = 100    # wall

    explorer = FrontierExplorer(min_frontier_size=3)
    frontiers = explorer.detect_frontiers(grid)

    found_ok = len(frontiers) > 0
    print(f"  {PASS if found_ok else FAIL} Found {len(frontiers)} frontiers")

    # Select a frontier
    robot_pos = np.array([50, 40])
    goal = explorer.select_frontier(
        frontiers, robot_pos, origin=(0.0, 0.0), resolution=0.05
    )
    goal_ok = goal is not None
    print(f"  {PASS if goal_ok else FAIL} Selected goal: {goal}")

    # Mark visited and check completion
    if goal is not None:
        explorer.mark_visited(goal)

    return found_ok and goal_ok


def test_mission_controller():
    """Test mission controller with pre-built semantic map."""
    print("\n--- Test: Mission Controller ---")
    from semantic_navigation.semantic_map import SemanticMap
    from semantic_navigation.mission_controller import MissionController, MissionState

    smap = SemanticMap()
    smap.add_object("blue vase", [3.0, 1.5, 0.8], 0.85)
    smap.add_object("red chair", [1.0, 2.0, 0.0], 0.90)
    smap.add_object("table", [2.0, 2.5, 0.0], 0.88)
    smap.add_object("sofa", [2.5, 3.0, 0.0], 0.92)
    smap.add_object("chair", [4.0, 1.0, 0.0], 0.75)  # another chair

    controller = MissionController(semantic_map=smap, detector=None)

    # Test basic command
    goal, parsed = controller.execute_command("Go to the blue vase")
    basic_ok = goal is not None and abs(goal[0] - 3.0) < 0.1
    print(f"  {PASS if basic_ok else FAIL} 'Go to the blue vase' → goal={goal}")

    controller.state = MissionState.IDLE

    # Test spatial reference: "chair near the sofa"
    goal, parsed = controller.execute_command("Navigate to the chair near the sofa")
    # Should pick chair at (1.0, 2.0) which is closer to sofa at (2.5, 3.0)
    # than chair at (4.0, 1.0)
    spatial_ok = goal is not None
    if spatial_ok:
        dist_to_sofa = np.linalg.norm(goal[:2] - np.array([2.5, 3.0]))
        spatial_ok = dist_to_sofa < 2.0
    print(f"  {PASS if spatial_ok else FAIL} 'chair near sofa' → goal={goal}")

    controller.state = MissionState.IDLE

    # Test not-found
    goal, parsed = controller.execute_command("Go to the lamp")
    notfound_ok = goal is None
    print(f"  {PASS if notfound_ok else FAIL} 'Go to the lamp' → not found (correct)")

    return basic_ok and spatial_ok and notfound_ok


def test_grounding_dino():
    """Test Grounding DINO model loading and inference."""
    print("\n--- Test: Grounding DINO ---")
    from semantic_navigation.grounding_dino_detector import GroundingDINODetector

    detector = GroundingDINODetector()
    print(f"  Device: {detector.device}")

    print("  Loading model (this may take a minute on first run)...")
    t0 = time.time()
    detector.load_model()
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Create a dummy test image (random noise)
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy_image, "chair . table")
    print(f"  Detections on noise image: {len(detections)} (may be 0, that's OK)")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--with-dino", action="store_true",
                        help="Include Grounding DINO test (needs GPU or MPS)")
    args = parser.parse_args()

    print("=" * 60)
    print("LOCAL TEST SUITE — Semantic Navigation")
    print("=" * 60)

    results = {}
    tests = [
        ("Command Parser", test_command_parser),
        ("Semantic Map", test_semantic_map),
        ("BBox to 3D", test_bbox_to_3d),
        ("Frontier Explorer", test_frontier_explorer),
        ("Mission Controller", test_mission_controller),
    ]

    if args.with_dino:
        tests.append(("Grounding DINO", test_grounding_dino))

    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"  {FAIL} Exception: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    for name, ok in results.items():
        print(f"  {PASS if ok else FAIL} {name}")
    print(f"\n{passed}/{total} test suites passed")

    if not args.with_dino:
        print(f"\n  Note: Run with --with-dino to also test the ML model")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
