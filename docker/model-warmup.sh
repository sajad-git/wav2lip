#!/bin/bash

# Model Warmup Script
# Pre-loads Wav2Lip and InsightFace models into GPU memory

set -e

echo "🔥 Starting model warmup and GPU initialization..."

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate wav2lip

# Set Python path
export PYTHONPATH=/app:$PYTHONPATH

# Check if models directory exists
if [[ ! -d "/app/assets/models" ]]; then
    echo "📁 Creating models directory..."
    mkdir -p /app/assets/models/wav2lip
    mkdir -p /app/assets/models/insightface
fi

# Download models if not present
echo "📥 Checking and downloading required models..."
python /app/scripts/download_models.py --verify-checksums --gpu-optimize

# Validate GPU memory availability
echo "🧠 Checking GPU memory availability..."
python -c "
import onnxruntime as ort
import psutil
import os

# Check GPU memory
if ort.get_device() == 'GPU':
    print(f'✅ GPU available for ONNX Runtime')
    print(f'CUDA devices: {os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"Not set\")}')
else:
    print('⚠️  GPU not available, falling back to CPU')

# Check system memory
memory = psutil.virtual_memory()
print(f'System memory: {memory.total // (1024**3)} GB total, {memory.available // (1024**3)} GB available')
"

# Initialize models and GPU memory pools
echo "🚀 Initializing models and GPU memory pools..."
python -c "
import sys
sys.path.append('/app')

from app.core.model_loader import ColdModelLoader
from app.core.face_cache_manager import FaceCacheManager
import time
import traceback

try:
    start_time = time.time()
    
    # Initialize model loader
    print('📦 Initializing cold model loader...')
    model_loader = ColdModelLoader()
    
    # Load Wav2Lip models
    print('🎭 Loading Wav2Lip models into GPU memory...')
    wav2lip_models = model_loader.load_wav2lip_models()
    print(f'✅ Loaded Wav2Lip models: {list(wav2lip_models.keys())}')
    
    # Load face detection models
    print('👤 Loading InsightFace models...')
    face_detector = model_loader.load_face_detection_models()
    print('✅ InsightFace models loaded successfully')
    
    # Perform baseline performance validation
    print('⚡ Validating model performance...')
    performance_metrics = model_loader.validate_model_performance()
    print(f'📊 Baseline performance established: {performance_metrics}')
    
    # Initialize face cache manager
    print('💾 Initializing face cache manager...')
    face_cache = FaceCacheManager()
    print('✅ Face cache manager initialized')
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f'🎉 Model warmup completed successfully in {total_time:.2f} seconds')
    print('🔥 All models are now loaded in GPU memory and ready for processing')
    
except Exception as e:
    print(f'❌ Model warmup failed: {str(e)}')
    traceback.print_exc()
    sys.exit(1)
"

if [[ $? -eq 0 ]]; then
    echo "✅ Model warmup completed successfully"
    echo "🎯 Models are now resident in GPU memory for instant processing"
else
    echo "❌ Model warmup failed"
    exit 1
fi

echo "🏁 Model warmup process completed" 