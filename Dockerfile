ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

FROM ${BASE_IMAGE}

ENV APP_ROOT=/opt/sam2
ENV PYTHONUNBUFFERED=1
ENV SAM2_BUILD_ALLOW_ERRORS=0
ENV MODEL_SIZE=${MODEL_SIZE}
ENV CUDA_HOME=/usr/local/cuda
ENV UV_SYSTEM_PYTHON=1
ENV UV_PROJECT_ENVIRONMENT=/opt/conda

ENV TORCH_CUDA_ARCH_LIST="7.0 7.2 7.5 8.0 8.6 8.9 9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavutil-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    pkg-config \
    build-essential \
    g++-11 \
    gcc-11 \
    libffi-dev \
    curl && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-11 100

ADD https://astral.sh/uv/0.9.21/install.sh /uv-installer.sh

# Install uv
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

RUN mkdir -p ${APP_ROOT}/checkpoints

COPY sam2 ${APP_ROOT}/sam2

WORKDIR ${APP_ROOT}/sam2

# Building SAM 2 CUDA extension
RUN uv sync -v
RUN uv add modal "fastapi[standard-no-fastapi-cloud-cli]" peft

# Download SAM 2.1 checkpoints
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_tiny.pt
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_small.pt
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_base_plus.pt
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_large.pt

