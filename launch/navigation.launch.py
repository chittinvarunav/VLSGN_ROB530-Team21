"""
Launch file for navigation-only mode (post-exploration).
Uses a pre-built map and semantic map for command execution.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(pkg_dir, "config")
    nav2_bringup_dir = get_package_share_directory("nav2_bringup")

    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    map_file = LaunchConfiguration("map", default="")
    semantic_map_path = LaunchConfiguration("semantic_map", default="")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("map", default_value=""),
        DeclareLaunchArgument("semantic_map", default_value=""),

        # Map server (load pre-built map)
        Node(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "yaml_filename": map_file,
            }],
        ),

        # Semantic map node
        Node(
            package="semantic_navigation",
            executable="semantic_map_node",
            name="semantic_map_node",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "save_path": semantic_map_path,
            }],
        ),

        # Mission controller (no detector — uses pre-built semantic map)
        Node(
            package="semantic_navigation",
            executable="mission_controller_node",
            name="mission_controller_node",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "enable_detector": False,
                "semantic_map_path": semantic_map_path,
            }],
        ),
    ])
