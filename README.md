# Semantic Navigation — Vision-Language Grounded Robot Navigation

**ROB 530 — Mobile Robotics · University of Michigan · Team 21**

To see the system in action, check out our demo video: https://www.youtube.com/watch?v=WfOprSfAoKM 

A ROS2 system that enables a TurtleBot3 to autonomously explore an unknown environment, build a semantic map of objects using [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base), and navigate to any object described in plain English — e.g. *"Go to the blue box"* or *"Find the red cylinder"*.

---

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [System Architecture](#system-architecture)
4. [Repository Structure](#repository-structure)
5. [Requirements](#requirements)
6. [Installation](#installation)
   - [Step 1 — System Dependencies](#step-1--system-dependencies)
   - [Step 2 — Environment Variables](#step-2--environment-variables)
   - [Step 3 — Python Virtual Environment](#step-3--python-virtual-environment-ml-dependencies)
   - [Step 4 — ROS2 Workspace Setup](#step-4--ros2-workspace-setup)
   - [Step 5 — Fix URDF Loading](#step-5--fix-urdf-loading-required)
   - [Step 6 — Build the Package](#step-6--build-the-ros2-package)
   - [Step 7 — Verify Installation](#step-7--verify-installation)
7. [Running the System](#running-the-system)
8. [Navigation Commands](#navigation-commands)
9. [Detectable Objects](#detectable-objects)
10. [Evaluation](#evaluation)
11. [Troubleshooting](#troubleshooting)
12. [Team](#team)

---

## Overview

This project integrates several robotics and computer vision components into a single autonomous navigation pipeline:

- **SLAM** via Cartographer builds a 2D occupancy map as the robot explores
- **Frontier-based exploration** drives the robot to unknown areas automatically
- **Grounding DINO** (open-vocabulary object detection) identifies objects in the robot's camera feed using free-form text prompts — no retraining needed
- **Semantic map** stores each detected object's 3D world position, label, and confidence
- **Natural language command parser** (spaCy) extracts navigation targets from plain English commands
- **Nav2** handles path planning and autonomous driving to the target position

The result: tell the robot in plain English what to find, and it will navigate to it.

---

## How It Works

The system runs in two sequential phases:

### Phase 1 — Autonomous Exploration

```
Occupancy Grid (Cartographer SLAM)
          │
          ▼
  Frontier Explorer
  (finds boundaries between free and unknown space)
          │  picks nearest unvisited frontier
          ▼
     Nav2 Goal
  (drives robot to frontier)
          │
          ▼
  Grounding DINO (RGB camera)
  (detects objects at each frontier stop)
          │  bbox + confidence score
          ▼
   BBox → 3D Converter
  (projects 2D detection into 3D world coords
   using camera intrinsics + TF transform)
          │  (x, y, z) world position
          ▼
    Semantic Map
  (stores label → 3D position, persisted to JSON)
```

The robot keeps exploring until all frontiers are exhausted — meaning the full map has been covered.

### Phase 2 — Natural Language Navigation

```
User: "Go to the blue box"
          │
          ▼
  Command Parser (spaCy)
  (extracts: action="navigate", attribute="blue", target="box")
          │  query_text = "blue box"
          ▼
    Semantic Map lookup
  (resolves "blue box" → stored (x, y, z) position)
          │
          ▼
     Nav2 Goal
  (sends position as navigation goal)
          │
          ▼
  TurtleBot3 navigates and arrives at object
          │
          ▼
  /navigation_status topic updated
```

---

## System Architecture

The codebase separates **pure logic** from **ROS2 node wrappers**. Each module has a `module.py` (logic, no ROS imports, fully unit-testable) and a `module_node.py` (ROS subscriber/publisher wiring).

| Module | Files | Role |
|---|---|---|
| **Mission Controller** | `mission_controller.py` + `_node.py` | Central state machine. States: `IDLE → EXPLORING → DETECTING → NAVIGATING → IDLE`. Coordinates all other modules. |
| **Frontier Explorer** | `frontier_explorer.py` + `_node.py` | Reads the occupancy grid, finds frontier cells (free cells adjacent to unknown), clusters them, and returns the best goal. Filters frontiers that are too small or too close to obstacles. |
| **Grounding DINO Detector** | `grounding_dino_detector.py` + `_node.py` | Wraps HuggingFace `IDEA-Research/grounding-dino-base`. Accepts a text prompt and an RGB image, returns a list of `Detection(label, bbox, score)`. Runs on CPU or CUDA. |
| **Semantic Map** | `semantic_map.py` + `_node.py` | Dictionary of `label → SemanticObject(position, confidence, timestamp)`. Supports fuzzy label matching, persists to/from JSON. Bounded to room extents `[-5, 5] × [-5, 5]` m. |
| **Command Parser** | `command_parser.py` | Regex + spaCy NLP pipeline. Parses commands like *"Go to the blue vase"* or *"Find the red chair near the sofa"* into a `ParsedCommand(target_object, target_attribute, spatial_relation, location)`. |
| **BBox → 3D** | `bbox_to_3d.py` | Given a 2D bounding box center and depth value, uses camera intrinsics (`CameraIntrinsics`) and the camera-to-map TF transform to produce a 3D world coordinate. Uses `CameraIntrinsics.turtlebot3_default()` for the simulated Waffle Pi camera (640×480, 69.4° HFOV). |
| **Teleop Interface** | `teleop_interface.py` | Optional keyboard-driven override of the mission controller for manual testing. |

### ROS2 Topics

| Topic | Type | Direction | Description |
|---|---|---|---|
| `/user_command` | `std_msgs/String` | Input | Plain-English navigation commands |
| `/navigation_status` | `std_msgs/String` | Output | Current system status and results |
| `/semantic_map` | custom | Internal | Semantic map state broadcast |
| `/cmd_vel` | `geometry_msgs/Twist` | Output | Robot velocity commands (via Nav2) |
| `/map` | `nav_msgs/OccupancyGrid` | Input | Cartographer SLAM output |
| `/camera/image_raw` | `sensor_msgs/Image` | Input | RGB camera feed |

---

## Repository Structure

```
semantic_navigation/
├── README.md
├── package.xml                  # ROS2 package metadata
├── setup.py / setup.cfg         # Python package build
├── requirements.txt             # Python (ML) dependencies
├── .gitignore
│
├── config/
│   ├── cartographer.lua         # SLAM tuning parameters
│   ├── nav2_params.yaml         # Nav2 planner/controller config
│   └── object_list.yaml         # Default Grounding DINO prompts + ground truth positions
│
├── launch/
│   ├── exploration.launch.py    # Phase 1: Gazebo + SLAM + Nav2 + RViz
│   ├── full_system.launch.py    # Phase 2: Full pipeline with Grounding DINO
│   └── navigation.launch.py    # Nav2 only (no Gazebo, for real robot)
│
├── src/semantic_navigation/     # Core Python source
│   ├── mission_controller.py
│   ├── mission_controller_node.py
│   ├── frontier_explorer.py
│   ├── frontier_explorer_node.py
│   ├── grounding_dino_detector.py
│   ├── grounding_dino_node.py
│   ├── semantic_map.py
│   ├── semantic_map_node.py
│   ├── command_parser.py
│   ├── bbox_to_3d.py
│   └── teleop_interface.py
│
├── scripts/
│   ├── run_full_evaluation.py   # Automated evaluation suite
│   ├── analyze_results.py       # Parse and summarize experiment JSON/CSV
│   ├── plot_semantic_map.py     # Visualize detected vs ground truth positions
│   └── visualize_semantic_map.py
│
├── data/
│   ├── exploration_map.pgm      # Pre-built map (for skipping exploration)
│   └── exploration_map.yaml
│
├── models/turtlebot3_waffle_pi/ # Gazebo SDF model
├── urdf/robot.urdf.xacro        # Robot description
├── worlds/single_room.world     # Gazebo world with 5 objects
└── test/test_local.py           # Unit tests (no ROS required)
```

---

## Requirements

| Requirement | Version |
|---|---|
| OS | Ubuntu 22.04 LTS (native or VM) |
| ROS2 | Humble Hawksbill |
| Python | 3.10+ |
| GPU | Optional — CPU works, ~5–10 s/image |
| VRAM (if GPU) | 4 GB+ for Grounding DINO |
| Disk | ~700 MB for model weights (auto-downloaded) |
| Internet | Required on first run (HuggingFace weight download) |

---

## Installation

### Step 1 — System Dependencies

```bash
sudo apt update
sudo apt install -y \
  ros-humble-turtlebot3 \
  ros-humble-turtlebot3-gazebo \
  ros-humble-turtlebot3-msgs \
  ros-humble-cartographer-ros \
  ros-humble-navigation2 \
  ros-humble-nav2-bringup \
  ros-humble-teleop-twist-keyboard \
  ros-humble-xacro \
  ros-humble-cv-bridge \
  ros-humble-tf2-ros \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-pip \
  python3-venv
```

---

### Step 2 — Environment Variables

Add ROS2 and TurtleBot3 model to your shell startup:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "export TURTLEBOT3_MODEL=waffle_pi" >> ~/.bashrc
source ~/.bashrc
```

---

### Step 3 — Python Virtual Environment (ML dependencies)

The ML stack (PyTorch, Transformers, spaCy) is installed in a dedicated virtual environment to avoid conflicts with system Python.

```bash
python3 -m venv ~/sem_nav_venv
source ~/sem_nav_venv/bin/activate

pip install --upgrade pip

# CPU-only PyTorch — works on any laptop or VM, no GPU needed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# If you have a CUDA-capable GPU (4 GB+ VRAM), use this instead:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# ML and vision dependencies
pip install \
  transformers>=4.35.0 \
  Pillow>=10.0.0 \
  opencv-python>=4.8.0 \
  numpy>=1.24.0 \
  scipy>=1.10.0 \
  matplotlib>=3.7.0 \
  PyYAML>=6.0 \
  spacy>=3.0.0

# NLP model for command parsing
python -m spacy download en_core_web_sm
```

> **Note:** You need to `source ~/sem_nav_venv/bin/activate` in every terminal where you run the full system. The exploration-only launch does not require the venv.

---

### Step 4 — ROS2 Workspace Setup

```bash
mkdir -p ~/ros2_ws/src
mkdir -p ~/ros2_ws/semantic_maps

# Clone from GitHub
cd ~/ros2_ws/src
git clone https://github.com/YOUR_USERNAME/semantic_navigation.git

# Or copy from a local folder / USB:
# cp -r /path/to/semantic_navigation ~/ros2_ws/src/
```

---

### Step 5 — Fix URDF Loading (Required)

The TurtleBot3 description package uses xacro for its URDF, but the exploration launch file reads it as a plain file by default. This one-time fix is required.

Open the exploration launch file:

```bash
nano ~/ros2_ws/src/semantic_navigation/launch/exploration.launch.py
```

At the top of the file, add `import xacro` after `import os`:

```python
import os
import xacro   # ADD THIS LINE
```

Then find these lines (~line 46):

```python
urdf_file = os.path.join(_desc_dir, "urdf", "turtlebot3_waffle_pi.urdf")
with open(urdf_file, "r") as f:
    robot_description = f.read()
```

Replace them with:

```python
urdf_file = os.path.join(tb3_desc_dir, "urdf", "turtlebot3_waffle_pi.urdf")
robot_description = xacro.process_file(urdf_file, mappings={"namespace": ""}).toxml()
```

Save with `Ctrl+O`, exit with `Ctrl+X`.

---

### Step 6 — Build the ROS2 Package

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select semantic_navigation --symlink-install
source install/setup.bash

# Add to bashrc so it persists across terminals
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

> **`--symlink-install`** means edits to Python source files in `src/` take effect immediately without rebuilding. Only rebuild if you change `package.xml`, `setup.py`, or launch files.

---

### Step 7 — Verify Installation

```bash
# Confirm ROS2 can find the package
ros2 pkg list | grep semantic_navigation

# Confirm Grounding DINO loads correctly (downloads weights on first run ~700 MB)
source ~/sem_nav_venv/bin/activate
python3 -c "
from semantic_navigation.grounding_dino_detector import GroundingDINODetector
d = GroundingDINODetector()
d.load_model()
print('Grounding DINO loaded on:', d.device)
"
```

Expected output: `Grounding DINO loaded on: cpu` (or `cuda:0` if you have a GPU).

---

## Running the System

Run each stage in order. Each stage opens in its own terminal unless stated otherwise.

### Stage 1 — Exploration with SLAM (Gazebo + Cartographer + Nav2)

```bash
export TURTLEBOT3_MODEL=waffle_pi
source ~/ros2_ws/install/setup.bash
ros2 launch semantic_navigation exploration.launch.py
```

Gazebo opens with the 5-object single-room world. RViz shows the robot, LiDAR scan, and the map building in real time.

---

### Stage 2 — Manual Teleoperation (new terminal, optional)

Drive the robot manually to explore the map yourself:

```bash
source ~/ros2_ws/install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Use the on-screen key bindings (`i` = forward, `j`/`l` = turn, `,` = back, `k` = stop). Watch the occupancy map expand in RViz as the robot moves.

---

### Stage 3 — Nav2 Click-to-Navigate (in RViz, no extra terminal)

In RViz, click the **"2D Nav Goal"** button in the toolbar, then click a location on the built map. Nav2 will compute a path and drive the robot there autonomously.

---

### Stage 4 — Full System with Grounding DINO (restart required)

First, kill the exploration session:

```bash
pkill -f gzserver
pkill -f gzclient
pkill -f ros2
sleep 5
```

Archive the previous semantic map (if any) and launch the full pipeline:

```bash
source ~/sem_nav_venv/bin/activate
source ~/ros2_ws/install/setup.bash
export TURTLEBOT3_MODEL=waffle_pi

# Archive any previous map
mv ~/ros2_ws/src/semantic_navigation/data/semantic_maps/latest.json \
   ~/ros2_ws/src/semantic_navigation/data/semantic_maps/run_$(date +%Y%m%d_%H%M%S).json 2>/dev/null || true

ros2 launch semantic_navigation full_system.launch.py
```

This launches all 9 nodes in order: Gazebo, TurtleBot3, robot state publisher, Cartographer, Nav2, RViz, Grounding DINO, semantic map, and mission controller.

---

### Stage 5 — Send Navigation Commands (new terminal)

```bash
source ~/sem_nav_venv/bin/activate
source ~/ros2_ws/install/setup.bash

# Kick off autonomous exploration (robot drives around and maps objects)
ros2 topic pub /user_command std_msgs/msg/String '{data: "start exploration"}'

# Once exploration finishes, navigate to any detected object
ros2 topic pub /user_command std_msgs/msg/String '{data: "Go to the blue box"}'
ros2 topic pub /user_command std_msgs/msg/String '{data: "Go to the yellow box"}'
ros2 topic pub /user_command std_msgs/msg/String '{data: "Go to the red cylinder"}'
ros2 topic pub /user_command std_msgs/msg/String '{data: "Go to the white cylinder"}'
ros2 topic pub /user_command std_msgs/msg/String '{data: "Go to the green cylinder"}'

# Query the current semantic map
ros2 topic pub /user_command std_msgs/msg/String '{data: "show map"}'

# Monitor real-time status
ros2 topic echo /navigation_status
```

---

## Navigation Commands

The command parser understands a range of phrasings. All of the following are valid:

| Command | Parsed target |
|---|---|
| `"Go to the blue box"` | blue box |
| `"Navigate to the red cylinder"` | red cylinder |
| `"Find the yellow box"` | yellow box |
| `"Go to the green cylinder near the wall"` | green cylinder |
| `"start exploration"` | triggers exploration mode |
| `"show map"` | prints semantic map contents to status topic |

To add support for new phrasings, edit the regex patterns in `src/semantic_navigation/command_parser.py`.

---

## Detectable Objects

The default Gazebo world (`worlds/single_room.world`) contains 5 objects with known ground-truth positions:

| Label | Position (x, y) |
|---|---|
| `red cylinder` | (-2.0, 2.0) |
| `blue box` | (2.0, 2.0) |
| `yellow box` | (2.5, -2.5) |
| `white cylinder` | (2.5, 0.5) |
| `green cylinder` | (-2.0, -2.0) |

The Grounding DINO prompt is defined in `config/object_list.yaml`:

```yaml
default_prompt: "red cylinder . blue box . yellow box . white cylinder . green cylinder"
```

To detect different objects, edit the prompt and add corresponding entries to `ground_truth`. No model retraining is needed — Grounding DINO is open-vocabulary.

---

## Evaluation

Run the automated evaluation suite to benchmark detection accuracy and navigation success rate:

```bash
cd ~/ros2_ws/src/semantic_navigation/scripts
source ~/sem_nav_venv/bin/activate
python3 run_full_evaluation.py
```

Results are saved as timestamped CSV and JSON files in `scripts/data/experiments/`.

To visualize detected positions vs. ground truth on a map:

```bash
python3 plot_semantic_map.py
# Output: semantic_map_comparison.png
```

To generate a summary table from all experiment runs:

```bash
python3 analyze_results.py
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `colcon build` fails with xacro error | `sudo apt install ros-humble-xacro` |
| `map` frame missing in RViz on startup | Normal — Cartographer delays ~10 s before publishing. Wait for it. |
| Gazebo is very slow in a VM | Enable 3D acceleration in VM settings; set video memory to 128 MB+ |
| Grounding DINO is slow (~5–10 s/image) | Expected on CPU. Use CUDA PyTorch if you have a 4 GB+ GPU. |
| Model weights fail to download | Check internet connection. Weights come from HuggingFace (~700 MB). |
| `ros2 pkg list` doesn't show the package | Run `source ~/ros2_ws/install/setup.bash` first |
| Nav2 goal rejected | Make sure the map has been built enough to include the goal area. Drive around more first. |
| Detected object position is wrong | The robot may not have had a clear view. Run exploration again or manually drive to a better vantage point. |

---

## Team

ROB 530 — Mobile Robotics, University of Michigan
Team 21
