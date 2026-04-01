# Use Ubuntu 24.04 with CUDA 12.8 support
FROM nvidia/cuda:12.8.0-base-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_CACHE_DIR="/root/.cache/uv"

# Install system dependencies with apt cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git curl cmake build-essential libglib2.0-0 libglu1-mesa \
    libvulkan1 vulkan-tools pkg-config python3-pip \
    libgl1 \
    libx11-6 \
    libvulkan1 \
    libxt6 libxrender1 libxext6 libsm6 libice6 libfontconfig1 \ 
    && rm -rf /var/lib/apt/lists/*

   
# Install uv (the fast python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create workspace and set as home
WORKDIR /workspace

# Step A: Copy dependency files and source code
# Note: source/relic is required for building the package during uv sync
COPY pyproject.toml /workspace/pyproject.toml
COPY uv.lock /workspace/uv.lock

# Step B: Sync. This takes ~10 mins the FIRST time, 
# but will be CACHED for subsequent builds with same dependencies.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra-index-url https://pypi.nvidia.com

# Set entrypoint to bash for interactive use
ENTRYPOINT ["/bin/bash"]
