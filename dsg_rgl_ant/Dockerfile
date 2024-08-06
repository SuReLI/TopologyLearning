

# Utiliser l'image de base avec CUDA et cuDNN
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Mettre à jour les packages et installer les dépendances
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget

# Install python dependencies

# Download thundersvm
RUN wget https://github.com/Xtra-Computing/thundersvm/blob/d38af58e0ceb7e5d948f3ef7d2c241ba50133ee6/python/dist/thundersvm-cpu-0.2.0-py3-none-linux_x86_64.whl

# RUN pyhton3 -m pip install thundersvm-cpu-0.2.0-py3-none-linux_x86_64.whl
RUN python3 -m pip install torch

# Definir le répertoire de travail
WORKDIR /app

# Copier les fichiers source dans le conteneur
COPY . /app


