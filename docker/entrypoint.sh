#!/bin/bash

# Avatar Service Entrypoint Script
# Handles cold model loading and avatar cache initialization

set -e

echo "🚀 Starting Avatar Streaming Service..."
echo "=================================================="

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate wav2lip

# Set Python path
export PYTHONPATH=/app:$PYTHONPATH

# Check GPU availability
echo "🔧 Checking GPU availability..."
python -c "import onnxruntime; print(f'ONNX Runtime device: {onnxruntime.get_device()}')"

if [[ "$PRELOAD_MODELS" == "true" ]]; then
    echo "❄️ Starting cold model loading..."
    
    # Run model warmup script
    echo "🔥 Warming up models and GPU..."
    bash /app/docker/model-warmup.sh
    
    if [[ $? -eq 0 ]]; then
        echo "✅ Model warmup completed successfully"
    else
        echo "❌ Model warmup failed"
        exit 1
    fi
fi

if [[ "$PRELOAD_AVATARS" == "true" ]]; then
    echo "👤 Initializing avatar cache system..."
    
    # Initialize avatar database and cache
    python /app/scripts/initialize_avatar_cache.py
    
    if [[ $? -eq 0 ]]; then
        echo "✅ Avatar cache initialization completed"
    else
        echo "❌ Avatar cache initialization failed"
        exit 1
    fi
fi

# Validate system requirements
echo "🔍 Validating system requirements..."
python /app/scripts/validate_system.py

# Start the application
echo "🌐 Starting FastAPI application..."
echo "Service will be available at http://0.0.0.0:5002"
echo "=================================================="

# Use conda environment's uvicorn
exec /opt/conda/envs/wav2lip/bin/uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 5002 \
    --workers 1 \
    --log-level info \
    --access-log \
    --no-use-colors 