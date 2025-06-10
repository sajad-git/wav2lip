# Avatar Streaming Service

A high-performance, dockerized AI avatar system with cold model loading and advanced avatar registration capabilities. This service creates low-latency, Persian-supported AI avatars using pre-loaded Wav2Lip models and cached face data for instant processing.

## 🌟 Key Features

### ❄️ Cold Loading Innovation
- **Pre-loaded Models**: Load Wav2Lip ONNX models once during container startup
- **GPU Memory Optimization**: Keep models in GPU memory throughout service lifetime
- **Instant Processing**: <200ms processing time per chunk (no model loading overhead)
- **Multi-user Support**: 3+ concurrent users sharing pre-loaded model instances

### 🎭 Avatar Registration System
- **Pre-processed Face Detection**: Register avatars with complete face analysis
- **Cached Face Metadata**: Store bounding boxes, landmarks, and cropped face regions
- **Instant Avatar Access**: Zero face detection time during inference
- **Multi-format Support**: Image (JPG, PNG) and video/GIF avatar registration
- **Quality Validation**: Ensure avatar quality during registration, not runtime

### 🚀 Performance Targets
- **Model Loading**: Once at startup (5-10 seconds)
- **Avatar Registration**: 2-5 seconds per avatar (one-time cost)
- **First Chunk Latency**: <500ms (no model loading + no face detection delay)
- **Chunk Processing**: <150ms per 5-15 second segment
- **Inter-chunk Gap**: <50ms (seamless playback)
- **Concurrent Users**: 3+ users with shared pre-loaded models + pre-processed avatars

## 🏗️ Architecture Overview

```
avatar-streaming-service/
├── docker/                    # Docker configuration
├── app/                       # Main application
│   ├── config/               # Configuration files
│   ├── core/                 # Core processing components
│   ├── services/             # Service layer
│   ├── streaming/            # WebSocket streaming
│   ├── utils/                # Utility functions
│   ├── models/               # Data models
│   └── middleware/           # Middleware components
├── assets/                   # Static assets and models
├── data/                     # Avatar registry and cache
├── static/                   # Frontend files
├── tests/                    # Test suite
└── scripts/                  # Utility scripts
```

## 🚀 Quick Start

### Prerequisites

- **GPU**: NVIDIA RTX 4090 or equivalent (20GB+ VRAM recommended)
- **Docker**: Latest version with NVIDIA Container Toolkit
- **System Memory**: 32GB+ RAM recommended
- **Storage**: 50GB+ free space for models and avatars

### 1. Clone Repository

```bash
git clone <repository-url>
cd avatar-streaming-service
```

### 2. Environment Setup

```bash
# Copy environment configuration
cp env.example .env

# Edit configuration with your API keys
nano .env
```

**Required Configuration:**
```bash
# Add your OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here

# Configure GPU settings
CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_LIMIT=20GB
```

### 3. Download Models

```bash
# Download required models
python scripts/download_models.py --verify-checksums --gpu-optimize
```

### 4. Build and Run

```bash
# Build and start the service
docker-compose up --build
```

### 5. Access the Service

- **Web Interface**: http://localhost:5002
- **Avatar Registration**: http://localhost:5002/avatar-registration
- **API Documentation**: http://localhost:5002/docs
- **Health Check**: http://localhost:5002/health

## 📋 Usage Examples

### Avatar Registration

#### Upload New Avatar (Web Interface)
1. Navigate to http://localhost:5002/avatar-registration
2. Drag and drop an image or video file
3. Fill in avatar details (name, description)
4. Wait for face detection and quality assessment
5. Complete registration

#### API Registration
```bash
curl -X POST "http://localhost:5002/avatar/register" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@avatar.jpg" \
  -F "avatar_name=My Avatar" \
  -F "user_id=user123"
```

### Avatar Processing

#### WebSocket Streaming
```javascript
const ws = new WebSocket('ws://localhost:5002/ws/stream');

// Select avatar
ws.send(JSON.stringify({
  type: 'avatar_selection',
  avatar_id: 'avatar_123'
}));

// Send text for processing
ws.send(JSON.stringify({
  type: 'text_input',
  text: 'سلام، چطورید؟',  // Persian text
  language: 'fa'
}));

// Receive video chunks
ws.onmessage = (event) => {
  const chunk = JSON.parse(event.data);
  if (chunk.type === 'video_chunk') {
    // Display video chunk
    displayVideoChunk(chunk.data);
  }
};
```

#### REST API Processing
```bash
curl -X POST "http://localhost:5002/avatar/process" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you today?",
    "avatar_id": "avatar_123",
    "language": "en"
  }'
```

## 🔧 Configuration

### Model Configuration

```python
# app/config/model_config.py
class ModelConfig:
    wav2lip_model_path = "/app/assets/models/wav2lip/wav2lip.onnx"
    wav2lip_gan_model_path = "/app/assets/models/wav2lip/wav2lip_gan.onnx"
    face_detector_path = "/app/assets/models/insightface/antelope"
```

### Avatar Configuration

```python
# app/config/avatar_config.py
class AvatarConfig:
    supported_formats = {".jpg", ".jpeg", ".png", ".gif", ".mp4", ".mov"}
    max_file_size = 50 * 1024 * 1024  # 50MB
    face_detection_threshold = 0.5
    cache_compression = True
```

### Performance Tuning

```bash
# Environment variables for optimization
GPU_MEMORY_LIMIT=20GB
MODEL_CACHE_SIZE=4GB
AVATAR_CACHE_SIZE=2GB
MAX_CONCURRENT_USERS=3
CHUNK_BUFFER_SIZE=5
```

## 🧪 Testing

### Run Test Suite

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/ -v

# Avatar registration tests
pytest tests/unit/test_avatar_registration.py -v

# Performance tests
pytest tests/load/ -m performance

# GPU tests (requires GPU)
ENABLE_GPU_TESTS=true pytest tests/ -m gpu
```

### Performance Benchmarking

```bash
# Run performance benchmark
python scripts/performance_benchmark.py --output benchmark_report.json

# Verbose benchmark with detailed output
python scripts/performance_benchmark.py --verbose
```

## 📊 Monitoring and Metrics

### Health Checks

```bash
# Basic health check
curl http://localhost:5002/health

# Detailed readiness check
curl http://localhost:5002/ready

# Avatar cache status
curl http://localhost:5002/avatar/cache/status
```

### Performance Metrics

- **Model Loading Time**: Tracked during startup
- **Avatar Registration Time**: Per-avatar registration metrics
- **Face Cache Hit Rate**: Cache efficiency tracking
- **Processing Latency**: Real-time chunk processing times
- **Memory Usage**: GPU and system memory monitoring

## 🐳 Docker Deployment

### Production Deployment

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  avatar-service:
    build:
      context: .
      dockerfile: docker/Dockerfile
    runtime: nvidia
    environment:
      - LOG_LEVEL=WARNING
      - MAX_CONCURRENT_USERS=10
      - GPU_MEMORY_LIMIT=40GB
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    ports:
      - "5002:5002"
```

### Environment-specific Configuration

```bash
# Development
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
```

## 🔒 Security Considerations

### File Upload Security
- File format validation
- Size limits enforcement
- Malicious content scanning
- Rate limiting on uploads

### API Security
- CORS configuration
- Rate limiting
- Input validation
- Error message sanitization

### Data Protection
- Avatar data encryption
- Secure cache storage
- User data isolation
- Audit logging

## 🌍 Persian Language Support

### Text Processing
- Unicode normalization
- RTL text handling
- Persian-specific punctuation
- Hazm library integration

### Audio Optimization
- Persian phoneme enhancement
- Consonant clarity improvement
- Speaking rate optimization

### UI Support
- RTL layout support
- Persian fonts
- Persian keyboard support

## 🚨 Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check NVIDIA runtime
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

#### Model Loading Failures
```bash
# Check model files
python scripts/validate_system.py --check-models

# Verify ONNX compatibility
python scripts/download_models.py --verify-only
```

#### Avatar Registration Issues
```bash
# Check face detection
python scripts/validate_system.py --check-face-detection

# Clear avatar cache
python scripts/cleanup_resources.py --clear-avatar-cache
```

#### Memory Issues
```bash
# Monitor memory usage
python scripts/performance_benchmark.py --memory-profile

# Optimize memory settings
export GPU_MEMORY_LIMIT=16GB
export AVATAR_CACHE_SIZE=1GB
```

### Logging and Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Check application logs
docker-compose logs avatar-service

# Monitor performance
docker stats avatar-service
```

## 📚 API Documentation

### Avatar Management
- `POST /avatar/register` - Register new avatar
- `GET /avatar/list` - List available avatars
- `DELETE /avatar/{avatar_id}` - Delete avatar
- `GET /avatar/{avatar_id}/info` - Get avatar details

### Processing
- `POST /avatar/process` - Process text with avatar
- `WS /ws/stream` - WebSocket streaming endpoint

### System
- `GET /health` - Health status
- `GET /ready` - Readiness check
- `GET /metrics` - Performance metrics

## 🔄 Development Workflow

### Local Development

```bash
# Set up development environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r docker/requirements.txt

# Run in development mode
export DEV_MODE=true
export SKIP_MODEL_LOADING=true
python app/main.py
```

### Code Quality

```bash
# Run linting
flake8 app/
black app/

# Type checking
mypy app/

# Security scanning
bandit -r app/
```

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation

## 🎯 Roadmap

### Phase 1 (Current)
- ✅ Core avatar streaming functionality
- ✅ Cold model loading
- ✅ Avatar registration system
- ✅ Persian language support

### Phase 2 (Planned)
- 🔄 Advanced emotion detection
- 🔄 Real-time background replacement
- 🔄 Multi-language expansion
- 🔄 Enhanced quality controls

### Phase 3 (Future)
- 📋 Voice cloning integration
- 📋 Custom model training
- 📋 Advanced analytics dashboard
- 📋 Enterprise features

---

**Built with ❤️ for high-performance AI avatar generation** 