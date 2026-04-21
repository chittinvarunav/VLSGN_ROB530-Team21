#!/bin/bash
# ============================================================
# GreatLakes Setup Script — ROB 530 Semantic Navigation
# TurtleBot3 Waffle Pi + Cartographer SLAM + Nav2 + Grounding DINO
# ============================================================
#
# Step 1: SSH into GreatLakes
#   ssh uniqname@greatlakes.arc-ts.umich.edu
#
# Step 2: Copy project to GreatLakes (run on YOUR Mac):
#   scp -r ~/Documents/ROB530_Project/ greatlakes:~/
#
# Step 3: On GreatLakes — request a GPU node with display:
#   salloc --account=rob530w25_class --partition=gpu \
#          --gpus=1 --mem=32G --cpus-per-task=8 --time=4:00:00
#
# Step 4: Run this script:
#   cd ~/ROB530_Project/semantic_navigation
#   bash scripts/setup_greatlakes.sh
#
# ============================================================
set -e

echo "===== ROB 530 GreatLakes Setup ====="
echo ""

# ---- 1. Load modules ----
module load ros/humble-desktop     2>/dev/null || \
module load ros2/humble            2>/dev/null || \
echo "WARNING: ROS2 module not found — check: module avail ros"

module load python/3.10
module load cuda/11.8
module load gazebo/11              2>/dev/null || echo "Gazebo module may not exist, using ROS-bundled version"

source /opt/ros/humble/setup.bash 2>/dev/null || \
source ~/ros2_humble/setup.bash   2>/dev/null || \
echo "WARNING: Could not source ROS2 — check your ROS2 installation path"

echo "ROS2 sourced: $(which ros2 2>/dev/null || echo 'not found')"
echo "Python: $(python3 --version)"
echo "CUDA: $(nvcc --version 2>/dev/null | head -1 || echo 'not found')"
echo ""

# ---- 2. Install ROS2 TurtleBot3 packages ----
echo "===== Installing TurtleBot3 packages ====="
sudo apt-get update -q
sudo apt-get install -y \
    ros-humble-turtlebot3 \
    ros-humble-turtlebot3-gazebo \
    ros-humble-turtlebot3-msgs \
    ros-humble-cartographer-ros \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-teleop-twist-keyboard \
    python3-colcon-common-extensions \
    python3-rosdep 2>/dev/null || {
  echo "apt-get failed (no sudo?). Trying pip installs instead..."
}

echo "TURTLEBOT3_MODEL=waffle_pi" >> ~/.bashrc
export TURTLEBOT3_MODEL=waffle_pi

# ---- 3. Create ROS2 workspace ----
echo ""
echo "===== Setting up ROS2 workspace ====="
mkdir -p ~/ros2_ws/src
mkdir -p ~/ros2_ws/semantic_maps

# Symlink our package into the workspace
if [ ! -L ~/ros2_ws/src/semantic_navigation ]; then
    ln -s ~/ROB530_Project/semantic_navigation ~/ros2_ws/src/semantic_navigation
    echo "Linked semantic_navigation into workspace"
fi

# ---- 4. Python ML dependencies ----
echo ""
echo "===== Installing Python ML dependencies ====="
VENV_DIR="$HOME/sem_nav_venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install --upgrade pip -q
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q
pip install transformers>=4.35.0 Pillow opencv-python-headless numpy scipy matplotlib PyYAML spacy -q
python -m spacy download en_core_web_sm -q

echo ""
echo "===== Verifying GPU ====="
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

# ---- 5. Build the ROS2 package ----
echo ""
echo "===== Building ROS2 package ====="
cd ~/ros2_ws
source /opt/ros/humble/setup.bash 2>/dev/null || true
colcon build --packages-select semantic_navigation --symlink-install
source install/setup.bash

echo ""
echo "===== Setup complete! ====="
echo ""
echo "========================================"
echo "  TESTING — DO THIS IN ORDER:"
echo "========================================"
echo ""
echo "STAGE 1 — Test Gazebo alone (run this first):"
echo "  export TURTLEBOT3_MODEL=waffle_pi"
echo "  source ~/ros2_ws/install/setup.bash"
echo "  ros2 launch semantic_navigation exploration.launch.py"
echo "  → You should see Gazebo window open with 5-room house"
echo "  → TurtleBot3 robot should appear in the hallway"
echo "  → RViz should open showing the robot and LiDAR scan"
echo ""
echo "STAGE 2 — Test teleoperation (new terminal):"
echo "  source ~/ros2_ws/install/setup.bash"
echo "  ros2 run teleop_twist_keyboard teleop_twist_keyboard"
echo "  → Drive the robot around. Map should build in RViz."
echo ""
echo "STAGE 3 — Test Nav2 click-to-navigate:"
echo "  In RViz, use '2D Nav Goal' button to click a goal"
echo "  → Robot should plan a path and navigate to it"
echo "  → It will avoid walls and furniture"
echo ""
echo "STAGE 4 — Test Grounding DINO (GPU):"
echo "  source $VENV_DIR/bin/activate"
echo "  cd ~/ROB530_Project/semantic_navigation"
echo "  python3 -c \""
echo "  from src.semantic_navigation.grounding_dino_detector import GroundingDINODetector"
echo "  d = GroundingDINODetector(); d.load_model()"
echo "  print('DINO loaded on:', d.device)\""
echo ""
echo "STAGE 5 — Run full system:"
echo "  ros2 launch semantic_navigation full_system.launch.py"
echo "  → All nodes start. Then send a command:"
echo "  ros2 topic pub /user_command std_msgs/String '{data: \"start exploration\"}'"
echo "  ros2 topic pub /user_command std_msgs/String '{data: \"Go to the blue vase\"}'"
echo ""
echo "========================================"
