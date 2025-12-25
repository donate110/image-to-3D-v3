FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# CUDA arch list for torch ( 8.0 A100 - 8.6 A40, A6000 -  8.9 6000 ADA - 12.0 6000 PRO BW)
ENV TORCH_CUDA_ARCH_LIST="8.6;8.0;8.9;12.0"

# Install CUDA keyring and all system dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    gnupg \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    git \
    build-essential \
    cmake \
    ninja-build \
    python3.11 \
    python3-pip \
    python3-wheel \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    cuda-toolkit-12-8 \
    cuda-nvcc-12-8 \
    cuda-libraries-dev-12-8 \
    cuda-nvtx-12-8 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 120 &&  \
update-alternatives --install /usr/bin/python python /usr/bin/python3 120

ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Copy requirements files
COPY docker/requirements.txt /tmp/requirements.txt

# Install basic dependencies from requirements.txt with no cache
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Install flash-attn, SPZ, and ben2 in one layer
RUN pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl \
    && wget https://github.com/404-Repo/spz/releases/download/v.0.1.2/pyspz-wheels-ubuntu-latest.zip \
    && unzip pyspz-wheels-ubuntu-latest.zip \
    && pip install --no-cache-dir pyspz-1.0.0-cp311-cp311-manylinux_2_34_x86_64.whl \
    && rm -rf pyspz-wheels-ubuntu-latest.zip *.whl \
    && pip install --no-cache-dir -e "git+https://github.com/PramaLLC/BEN2.git#egg=ben2"

# Set working directory
WORKDIR /workspace

# Copy the project
COPY pipeline_service /workspace

EXPOSE 10006
CMD ["python",  "serve.py"]