# Multi-stage Dockerfile for DCX inference system
# Builds a portable Docker image that runs on any platform

# Stage 1: Builder - compiles with Nuitka
FROM python:3.9-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ccache \
    clang \
    git \
    patchelf \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir nuitka

# Copy source code
COPY inference.py inference_ray.py ./
COPY src ./src
COPY configs ./configs

# Build with Nuitka (creates standalone executables)
RUN python -m nuitka \
    --standalone \
    --onefile \
    --assume-yes-for-downloads \
    --show-progress \
    --enable-plugin=numpy \
    --enable-plugin=torch \
    --include-package=yaml \
    --include-package=pydicom \
    --include-package=nibabel \
    --include-package=cv2 \
    --include-package=PIL \
    --include-package=scipy \
    --include-package=skimage \
    --include-data-dir=configs=configs \
    --include-data-dir=src=src \
    --output-filename=inference \
    inference.py

RUN python -m nuitka \
    --standalone \
    --onefile \
    --assume-yes-for-downloads \
    --show-progress \
    --enable-plugin=numpy \
    --enable-plugin=torch \
    --include-package=ray \
    --include-package=yaml \
    --include-package=pydicom \
    --include-package=nibabel \
    --include-package=cv2 \
    --include-package=PIL \
    --include-package=scipy \
    --include-package=skimage \
    --include-package=pandas \
    --include-data-dir=configs=configs \
    --include-data-dir=src=src \
    --output-filename=inference_ray \
    inference_ray.py

# Stage 2: Runtime - minimal image with just the executables
FROM ubuntu:22.04

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy compiled executables from builder
COPY --from=builder /build/inference /app/
COPY --from=builder /build/inference_ray /app/

# Copy necessary data files
COPY configs /app/configs
COPY checkpoints /app/checkpoints

# Make executables runnable
RUN chmod +x inference inference_ray

# Create data directory for volume mounting
RUN mkdir -p /data /results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4

# Default entrypoint (can be overridden)
ENTRYPOINT ["/app/inference_ray"]
CMD ["--help"]