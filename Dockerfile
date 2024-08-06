# Utiliser l'image de base avec CUDA et cuDNN
FROM nvidia/cuda:12.1.0-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

# Mettre à jour les packages et installer les dépendances
RUN apt-get update && apt-get install -y \
    python3.6 python3-pip libglib2.0-0 libsm6 libxext6 libxrender-dev wget libxml2-dev libxslt-dev python-dev

# Installer PyTorch et torchvision avec CUDA support
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install numpy torch lxml opencv-python matplotlib networkx ipdb gym gymnasium scikit-learn \
    scikit-image discord

# Working directory setup
WORKDIR /TopologyLearning
COPY . /TopologyLearning

#####################################################################
########################   ANT-MAZE STUFF   #########################
#####################################################################
# INSTALL MUJOCO
# > Install mujoco dependencies
RUN apt-get update
RUN apt-get install -y libglfw3 curl git libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev python3-pip  \
    python3-numpy python3-scipy net-tools unzip vim wget
RUN apt-get install -y xpra
RUN apt-get install -y xserver-xorg-dev
RUN apt-get install -y patchelf

# > Download and extract mujoco
RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
RUN tar -xf mujoco210-linux-x86_64.tar.gz
RUN mkdir /root/.mujoco
RUN mv mujoco210 /root/.mujoco/.
RUN rm mujoco210-linux-x86_64.tar.gz
# > Setup mandatory environment variables

RUN echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin" >> /root/.bashrc
RUN echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib/nvidia" >> /root/.bashrc
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin:/usr/lib/nvidia"

# > Install and build mujoco_py
RUN python3 -m pip install -U 'mujoco-py<2.2,>=2.1'
RUN python3 -m pip install "cython<3"
RUN python3 -c "import mujoco_py"
#####################################################################

#####################################################################
#########################   VIZDOOM STUFF   #########################
#####################################################################
# Specific to vizdoom experiments
# Install dependencies
RUN apt install -y wget libssl-dev libboost-all-dev libsdl2-dev libsdl2-2.0-0

# Install cmake 3.29
RUN wget https://github.com/Kitware/CMake/releases/download/v3.29.0-rc3/cmake-3.29.0-rc3.tar.gz
RUN tar -xvzf cmake-3.29.0-rc3.tar.gz
RUN bash cmake-3.29.0-rc3/bootstrap
RUN make
RUN make install

# Install vizdoomx
RUN python3 -m pip install vizdoom

# Install pyslam dependencies
#####################################################################

