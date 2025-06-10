# Stage 1 - Base Setup
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# System package installation
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    libasound2-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    sqlite3 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Stage 2 - Dependencies
WORKDIR /app

# Copy requirements from docker folder (build context is project root)
COPY docker/requirements.txt ./

# Install Python packages
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Install ONNX Runtime GPU with provider configuration
RUN python3 -m pip install onnxruntime-gpu==1.15.1

# Install additional packages for the avatar service
RUN python3 -m pip install \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    websockets==11.0.3 \
    pydantic==2.4.2 \
    openai==1.3.0 \
    requests==2.31.0 \
    aiofiles==23.2.0 \
    python-multipart==0.0.6 \
    pillow==10.0.1 \
    pydub==0.25.1 \
    soundfile==0.12.1

# Stage 3 - Application Setup
COPY . .

# Create required directories
RUN mkdir -p /app/assets/models/wav2lip \
    /app/assets/models/insightface \
    /app/assets/avatars/registered \
    /app/data/avatar_registry/face_cache \
    /app/logs \
    /app/temp \
    /app/static

# Set permissions
RUN chmod +x /app/docker/entrypoint.sh \
    /app/docker/model-warmup.sh \
    /app/scripts/*.py

# Stage 4 - Runtime Configuration
# Create non-root user for security
RUN useradd -m -u 1000 avataruser && \
    chown -R avataruser:avataruser /app

# Environment variables for GPU and model optimization
ENV CUDA_VISIBLE_DEVICES=0
ENV ONNX_DISABLE_GLOBAL_THREAD_POOL=1
ENV OMP_NUM_THREADS=8
ENV AVATAR_CACHE_SIZE=2GB
ENV MAX_REGISTERED_AVATARS=100
ENV PRELOAD_MODELS=true
ENV PRELOAD_AVATARS=true
ENV AVATAR_CACHE_WARMUP=true
ENV GPU_MEMORY_LIMIT=20GB
ENV MODEL_CACHE_SIZE=4GB
ENV PYTHONPATH=/app

# Switch to non-root user
USER avataruser

# Expose port
EXPOSE 5002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python /app/scripts/health_check.py --check-models --check-avatars

# Set entrypoint
ENTRYPOINT ["/app/docker/entrypoint.sh"] 