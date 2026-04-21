from setuptools import setup, find_packages

package_name = "semantic_navigation"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", [
            "launch/full_system.launch.py",
            "launch/exploration.launch.py",
            "launch/navigation.launch.py",
        ]),
        ("share/" + package_name + "/config", [
            "config/nav2_params.yaml",
            "config/cartographer.lua",
            "config/object_list.yaml",
        ]),
        ("share/" + package_name + "/models/turtlebot3_waffle_pi", [
            "models/turtlebot3_waffle_pi/model.sdf",
        ]),
        ("share/" + package_name + "/worlds", [
            "worlds/single_room.world",
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    author="ROB530 Team 21",
    description="Vision-Language Grounded Navigation",
    license="MIT",
    entry_points={
        "console_scripts": [
            "grounding_dino_node = semantic_navigation.grounding_dino_node:main",
            "semantic_map_node = semantic_navigation.semantic_map_node:main",
            "mission_controller_node = semantic_navigation.mission_controller_node:main",
            "teleop_interface = semantic_navigation.teleop_interface:main",
            "frontier_explorer_node = semantic_navigation.frontier_explorer_node:main",
        ],
    },
)
