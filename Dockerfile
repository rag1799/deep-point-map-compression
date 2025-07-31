# Base image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# setup environment
ENV TERM=xterm
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/python3.10/dist-packages/torch/lib/
ENV PYTHONPATH=/depoco/submodules/ChamferDistancePytorch/
ENV PIP_ROOT_USER_ACTION=ignore
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6"

# Provide a data directory to share data across docker and the host system
RUN mkdir -p /data

# Install system packages
RUN apt-get update && apt-get install -y \
    build-essential cmake git \
    python3 python3-pip python3-dev \
    libopencv-dev libeigen3-dev \
 && rm -rf /var/lib/apt/lists/*


# Install Pytorch with CUDA 11 support
RUN pip3 install \
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    torchaudio==2.1.0+cu121 \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install "numpy<2.0"
# Install python dependencies
RUN pip3 install \
    open3d  \
    tensorboard \
    ruamel.yaml \
    jupyterlab

# Copy the libary to the docker image
COPY ./ depoco/
RUN apt-get update && apt-get install -y ninja-build
# Install depoco and 3rdparty dependencies
RUN apt-get update && apt-get install -y \
    pybind11-dev \
    build-essential cmake
RUN cd depoco/ && pip3 install -U -e .
RUN cd depoco/submodules/octree_handler && pip3 install -U .
RUN cd depoco/submodules/ChamferDistancePytorch/chamfer3D/ && pip3 install -U .

WORKDIR /depoco/depoco

