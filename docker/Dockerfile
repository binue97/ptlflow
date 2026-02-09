# PTLFlow - PyTorch Lightning Optical Flow
# Multi-stage Docker build for training and inference

# ============================================================================
# Stage 1: Base image with system dependencies
# ============================================================================
ARG CUDA_VERSION=12.1.1
ARG CUDNN_VERSION=8
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.12 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    wget \
    curl \
    libopencv-dev \
    libhdf5-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default and install pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ============================================================================
# Stage 2: Python dependencies
# ============================================================================
FROM base AS dependencies

WORKDIR /tmp

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install Python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 3: Final image (application code mounted via docker-compose)
# ============================================================================
FROM dependencies AS final

# Set working directory
WORKDIR /workspace/ptlflow

# Set Python path
ENV PYTHONPATH=/workspace/ptlflow:$PYTHONPATH

# Environment variables for better performance
ENV KMP_INIT_AT_FORK=FALSE
ENV OMP_NUM_THREADS=1

# Copy entrypoint script
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Expose ports for TensorBoard
EXPOSE 6006

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command (can be overridden)
CMD ["bash"]
