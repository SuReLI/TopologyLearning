#!/bin/bash

# Install dependencies
echo "Installing mujoco dependencies ...\r"
apt-get update & apt-get install -y libglfw3 curl git libgl1-mesa-dev libgl1-mesa-glx libglew-dev \
    libosmesa6-dev python3-pip python3-numpy python3-scipy net-tools unzip vim wget xpra \
    xserver-xorg-dev patchelf

# Install mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
mv mujoco210 ~/.mujoco/.
rm mujoco210-linux-x86_64.tar.gz

echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:"${HOME}"/.mujoco/mujoco210/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib/nvidia" >> ~/.bashrc

# Install mujoco_py
python3 -m pip install -U 'mujoco-py<2.2,>=2.1'
python3 -m pip install "cython<3"
# Build mujoco_py
python3 -c "import mujoco_py"
