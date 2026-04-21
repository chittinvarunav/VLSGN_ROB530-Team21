"""
Full system launch for Vision-Language Grounded Navigation.
ROB 530 Project - Team 21

Launches in order:
  1. Gazebo with single_room world (default; override with world_file arg)
  2. TurtleBot3 Waffle Pi robot spawned into that world
  3. Robot state publisher (TF tree from URDF)
  4. Cartographer SLAM
  5. Nav2
  6. RViz
  7. Grounding DINO node
  8. Semantic map node
  9. Mission controller node

Usage:
  export TURTLEBOT3_MODEL=waffle_pi
  source ~/ros2_ws/install/setup.bash
  ros2 launch semantic_navigation full_system.launch.py

  # Headless (no RViz):
  ros2 launch semantic_navigation full_system.launch.py enable_rviz:=false

  # Skip Grounding DINO while testing Nav2:
  ros2 launch semantic_navigation full_system.launch.py enable_detector:=false
"""

import os
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ---- Paths ----
    pkg_dir          = get_package_share_directory("semantic_navigation")
    nav2_bringup_dir = get_package_share_directory("nav2_bringup")
    tb3_gazebo_dir   = get_package_share_directory("turtlebot3_gazebo")
    gazebo_ros_dir   = get_package_share_directory("gazebo_ros")

    config_dir  = os.path.join(pkg_dir, "config")
    default_world = os.path.join(pkg_dir, "worlds", "single_room.world")
    nav2_params = os.path.join(config_dir, "nav2_params.yaml")
    rviz_config = os.path.join(nav2_bringup_dir, "rviz", "nav2_default_view.rviz")

    tb3_desc_dir = get_package_share_directory("turtlebot3_description")
    urdf_file = os.path.join(tb3_desc_dir, "urdf", "turtlebot3_waffle_pi.urdf")
    robot_description = xacro.process_file(urdf_file, mappings={"namespace": ""}).toxml()

    # Custom SDF with depth camera (original + depth sensor added)
    sdf_model_path = os.path.join(
        pkg_dir, "models", "turtlebot3_waffle_pi", "model.sdf"
    )

    # GAZEBO_MODEL_PATH so the SDF can resolve model:// URIs for meshes
    gazebo_model_path = os.path.join(tb3_gazebo_dir, "models")
    existing_model_path = os.environ.get("GAZEBO_MODEL_PATH", "")
    full_model_path = (
        gazebo_model_path + ":" + existing_model_path
        if existing_model_path
        else gazebo_model_path
    )

    map_save_default = os.path.join(os.path.expanduser("~"),
                                    "ros2_ws", "src", "ROB530_Project",
                                    "semantic_navigation", "data",
                                    "semantic_maps", "latest.json")

    # Add sem_nav_venv site-packages to PYTHONPATH so ROS2 nodes can find torch/transformers
    venv_site_packages = os.path.join(
        os.path.expanduser("~"), "sem_nav_venv", "lib", "python3.10", "site-packages"
    )
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    # Append (not prepend) so system numpy/cv2 take priority over venv versions,
    # but torch/transformers (only in venv) are still found.
    full_pythonpath = (
        existing_pythonpath + ":" + venv_site_packages
        if existing_pythonpath
        else venv_site_packages
    )

    # ---- Launch args ----
    use_sim_time      = LaunchConfiguration("use_sim_time",      default="true")
    enable_rviz       = LaunchConfiguration("enable_rviz",       default="true")
    enable_detector   = LaunchConfiguration("enable_detector",   default="true")
    x_pose            = LaunchConfiguration("x_pose",            default="0.0")
    y_pose            = LaunchConfiguration("y_pose",            default="0.0")
    world_file        = LaunchConfiguration("world_file",        default=default_world)
    semantic_map_path = LaunchConfiguration("semantic_map_path", default=map_save_default)

    return LaunchDescription([
        SetEnvironmentVariable("TURTLEBOT3_MODEL", "waffle_pi"),
        SetEnvironmentVariable("GAZEBO_MODEL_PATH", full_model_path),

        DeclareLaunchArgument("use_sim_time",      default_value="true"),
        DeclareLaunchArgument("enable_rviz",       default_value="true"),
        DeclareLaunchArgument("enable_detector",   default_value="true"),
        DeclareLaunchArgument("x_pose",            default_value="0.0"),
        DeclareLaunchArgument("y_pose",            default_value="0.0"),
        DeclareLaunchArgument("world_file",        default_value=default_world,
                              description="Path to .world file"),
        DeclareLaunchArgument("semantic_map_path", default_value=map_save_default),

        # ── 1. Gazebo server with our world ──────────────────────────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros_dir, "launch", "gzserver.launch.py")
            ),
            launch_arguments={
                "world": world_file,
                "verbose": "false",
                "extra_gazebo_args": "-s libgazebo_ros_factory.so -s libgazebo_ros_init.so",
            }.items(),
        ),

        # Gazebo client (the GUI window)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros_dir, "launch", "gzclient.launch.py")
            ),
        ),

        # ── 2. Robot state publisher (publishes URDF / TF tree) ──────────
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{
                "robot_description": robot_description,
                "use_sim_time": use_sim_time,
            }],
        ),

        # ── 3. Spawn TurtleBot3 into Gazebo from SDF ─────────────────────
        # The SDF model includes all Gazebo plugins:
        #   - diff_drive  → publishes /odom topic + odom TF frame
        #   - laser scan  → publishes /scan
        #   - camera      → publishes /camera/image_raw
        #   - joint_state → publishes /joint_states (used by robot_state_publisher)
        Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            name="spawn_turtlebot3",
            output="screen",
            arguments=[
                "-entity", "turtlebot3_waffle_pi",
                "-file",   sdf_model_path,
                "-x", x_pose,
                "-y", y_pose,
                "-z", "0.01",
                "-Y", "0.0",
            ],
        ),

        # ── 4. Cartographer SLAM (after robot is spawned & publishing) ───
        TimerAction(period=6.0, actions=[
            Node(
                package="cartographer_ros",
                executable="cartographer_node",
                name="cartographer_node",
                output="log",
                parameters=[{"use_sim_time": use_sim_time}],
                arguments=[
                    "-configuration_directory", config_dir,
                    "-configuration_basename", "cartographer.lua",
                ],
            ),
            Node(
                package="cartographer_ros",
                executable="cartographer_occupancy_grid_node",
                name="cartographer_occupancy_grid_node",
                output="log",
                parameters=[{"use_sim_time": use_sim_time}],
                arguments=["-resolution", "0.05", "-publish_period_sec", "1.0"],
            ),
        ]),

        # ── 5. Nav2 (after SLAM publishes /map) ──────────────────────────
        TimerAction(period=10.0, actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(nav2_bringup_dir, "launch", "navigation_launch.py")
                ),
                launch_arguments={
                    "use_sim_time": use_sim_time,
                    "params_file":  nav2_params,
                    "autostart":    "true",
                }.items(),
            ),
        ]),

        # ── 6. RViz ──────────────────────────────────────────────────────
        TimerAction(period=8.0, actions=[
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_config],
                parameters=[{"use_sim_time": use_sim_time}],
                output="screen",
                condition=IfCondition(enable_rviz),
            ),
        ]),

        # ── 7. Frontier explorer (independent — must not be blocked by detector nodes)
        TimerAction(period=15.0, actions=[
            Node(
                package="semantic_navigation",
                executable="frontier_explorer_node",
                name="frontier_explorer_node",
                output="screen",
                parameters=[{
                    "use_sim_time":       use_sim_time,
                    "min_frontier_size":  3,
                    "exploration_rate":   1.0,
                    "goal_timeout_sec":   60.0,
                    "strategy":                "weighted",
                    "exploration_seed":        42,
                    "max_exploration_time_sec": 600.0,
                }],
            ),
        ]),

        # ── 8-10. Semantic navigation nodes (detector, semantic map, mission controller)
        TimerAction(period=15.0, actions=[
            Node(
                package="semantic_navigation",
                executable="grounding_dino_node",
                name="grounding_dino_node",
                output="screen",
                parameters=[{
                    "use_sim_time":   use_sim_time,
                    "box_threshold":  0.2,
                    "text_threshold": 0.15,
                }],
                additional_env={"PYTHONPATH": full_pythonpath},
                condition=IfCondition(enable_detector),
            ),
            Node(
                package="semantic_navigation",
                executable="semantic_map_node",
                name="semantic_map_node",
                output="screen",
                parameters=[{
                    "use_sim_time":   use_sim_time,
                    "merge_distance": 0.5,
                    "save_path":      semantic_map_path,
                }],
            ),
            Node(
                package="semantic_navigation",
                executable="mission_controller_node",
                name="mission_controller_node",
                output="screen",
                parameters=[{
                    "use_sim_time":         use_sim_time,
                    "detection_interval_m": 1.5,
                    "enable_detector":      enable_detector,
                    "semantic_map_path":    semantic_map_path,
                    "depth_topic":          "/depth_camera/depth/image_raw",
                }],
                additional_env={"PYTHONPATH": full_pythonpath},
            ),
        ]),
    ])
