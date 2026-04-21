"""
Exploration-only launch: Gazebo + TurtleBot3 + SLAM + Nav2.

Use this FIRST to verify the stack works before adding Grounding DINO.
This is the same as full_system but without the semantic navigation nodes.

Usage:
  export TURTLEBOT3_MODEL=waffle_pi
  source ~/ros2_ws/install/setup.bash
  ros2 launch semantic_navigation exploration.launch.py

Then in a new terminal, teleoperate:
  ros2 run teleop_twist_keyboard teleop_twist_keyboard

Watch RViz: the map should build as you drive around.
Then try clicking a Nav2 goal — robot navigates autonomously.
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
    pkg_dir          = get_package_share_directory("semantic_navigation")
    nav2_bringup_dir = get_package_share_directory("nav2_bringup")
    tb3_gazebo_dir   = get_package_share_directory("turtlebot3_gazebo")
    gazebo_ros_dir   = get_package_share_directory("gazebo_ros")

    config_dir  = os.path.join(pkg_dir, "config")
    nav2_params = os.path.join(config_dir, "nav2_params.yaml")
    rviz_config = os.path.join(nav2_bringup_dir, "rviz", "nav2_default_view.rviz")

    # Use the pre-processed URDF from turtlebot3_gazebo (no xacro macros, no Gazebo plugins)
    # Only used by robot_state_publisher for the TF tree.
    tb3_desc_dir = get_package_share_directory("turtlebot3_description")
    urdf_file = os.path.join(tb3_desc_dir, "urdf", "turtlebot3_waffle_pi.urdf")
    robot_description = xacro.process_file(urdf_file, mappings={"namespace": ""}).toxml()
    #urdf_file = os.path.join(tb3_gazebo_dir, "urdf", "turtlebot3_waffle_pi.urdf")
    #with open(urdf_file, "r") as f:
        #robot_description = f.read()

    # Custom SDF with depth camera (original + depth sensor added)
    sdf_model_path = os.path.join(
        pkg_dir, "models", "turtlebot3_waffle_pi", "model.sdf"
    )

    # GAZEBO_MODEL_PATH so SDF can resolve model:// URIs for meshes
    gazebo_model_path = os.path.join(tb3_gazebo_dir, "models")
    existing_model_path = os.environ.get("GAZEBO_MODEL_PATH", "")
    full_model_path = (
        gazebo_model_path + ":" + existing_model_path
        if existing_model_path
        else gazebo_model_path
    )

    # Default world: single_room (robot at origin)
    default_world = os.path.join(pkg_dir, "worlds", "single_room.world")

    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    enable_rviz  = LaunchConfiguration("enable_rviz",  default="true")
    x_pose       = LaunchConfiguration("x_pose",       default="0.0")
    y_pose       = LaunchConfiguration("y_pose",       default="0.0")
    world_file   = LaunchConfiguration("world_file",   default=default_world)

    return LaunchDescription([
        SetEnvironmentVariable("TURTLEBOT3_MODEL", "waffle_pi"),
        SetEnvironmentVariable("GAZEBO_MODEL_PATH", full_model_path),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("enable_rviz",  default_value="true"),
        DeclareLaunchArgument("x_pose",       default_value="0.0"),
        DeclareLaunchArgument("y_pose",       default_value="0.0"),
        DeclareLaunchArgument("world_file",   default_value=default_world,
                              description="Path to .world file"),

        # Gazebo server + client
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros_dir, "launch", "gzserver.launch.py")
            ),
            launch_arguments={"world": world_file, "verbose": "false"}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros_dir, "launch", "gzclient.launch.py")
            ),
        ),

        # Robot state publisher — publishes TF tree from the URDF
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

        # Spawn TurtleBot3 from SDF (provides diff_drive, laser, camera Gazebo plugins)
        Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            name="spawn_turtlebot3",
            output="screen",
            arguments=[
                "-entity", "turtlebot3_waffle_pi",
                "-file",   sdf_model_path,
                "-x", x_pose, "-y", y_pose, "-z", "0.01",
            ],
        ),

        # Cartographer SLAM
        TimerAction(period=6.0, actions=[
            Node(
                package="cartographer_ros",
                executable="cartographer_node",
                name="cartographer_node",
                output="screen",
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
                output="screen",
                parameters=[{"use_sim_time": use_sim_time}],
                arguments=["-resolution", "0.05", "-publish_period_sec", "1.0"],
            ),
        ]),

        # Nav2
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

        # RViz
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

        # Frontier explorer — autonomous exploration (after Nav2 + map are up)
        TimerAction(period=15.0, actions=[
            Node(
                package="semantic_navigation",
                executable="frontier_explorer_node",
                name="frontier_explorer_node",
                output="screen",
                parameters=[{
                    "use_sim_time":      use_sim_time,
                    "min_frontier_size": 3,
                    "exploration_rate":  1.0,
                    "goal_timeout_sec":  30.0,
                    "strategy":          "weighted",
                    "exploration_seed":  42,
                }],
            ),
        ]),
    ])
