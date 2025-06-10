# Avatar Streaming Service Troubleshooting Guide

## Overview

This guide provides solutions for common issues encountered when deploying and operating the Avatar Streaming Service with cold model loading and avatar registration capabilities.

## Quick Diagnostics

### Health Check Commands

```bash
# Check service health
curl -f http://localhost:5002/health || echo "Service health check failed"

# Check model readiness
curl -f http://localhost:5002/ready || echo "Models not loaded"

# Check GPU status
nvidia-smi

# Check container status
docker ps | grep avatar-service
docker logs avatar-service --tail 50

# Check avatar cache status
curl http://localhost:5002/metrics | grep avatar_cache
```

### Log Analysis

```bash
# View recent logs
docker logs avatar-service --since 1h --follow

# Search for errors
docker logs avatar-service 2>&1 | grep -i error

# Check startup logs
docker logs avatar-service 2>&1 | head -100

# Filter specific components
docker logs avatar-service 2>&1 | grep "model_loader"
docker logs avatar-service 2>&1 | grep "avatar_registrar"
```

---

## GPU and CUDA Issues

### GPU Not Detected

#### Symptoms
- `RuntimeError: CUDA error: no CUDA-capable device is detected`
- Service fails to start with GPU errors
- Models fall back to CPU processing

#### Diagnosis
```bash
# Check GPU visibility
nvidia-smi

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi

# Check NVIDIA container toolkit
sudo systemctl status nvidia-container-toolkit

# Verify GPU in container
docker exec avatar-service nvidia-smi
```

#### Solutions

1. **Install/Update NVIDIA Drivers**
   ```bash
   # Ubuntu
   sudo apt update
   sudo apt install nvidia-driver-525
   sudo reboot
   
   # Check driver version
   nvidia-smi
   ```

2. **Install NVIDIA Container Toolkit**
   ```bash
   # Add repository
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   # Install toolkit
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

3. **Configure Docker Runtime**
   ```bash
   # Add to /etc/docker/daemon.json
   {
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     },
     "default-runtime": "nvidia"
   }
   
   sudo systemctl restart docker
   ```

### GPU Memory Issues

#### Symptoms
- `RuntimeError: CUDA out of memory`
- Service becomes unresponsive
- Processing failures under load

#### Diagnosis
```bash
# Monitor GPU memory
nvidia-smi -l 1

# Check memory usage in container
docker exec avatar-service nvidia-smi

# View memory allocation in logs
docker logs avatar-service 2>&1 | grep -i memory
```

#### Solutions

1. **Optimize GPU Memory Settings**
   ```bash
   # Set GPU memory limit in environment
   export GPU_MEMORY_LIMIT=18GB
   
   # Restart service with new limits
   docker-compose restart avatar-service
   ```

2. **Clear GPU Memory Cache**
   ```bash
   # Clear CUDA cache
   docker exec avatar-service python -c "import torch; torch.cuda.empty_cache()"
   
   # Restart service if needed
   docker-compose restart avatar-service
   ```

3. **Reduce Concurrent Users**
   ```bash
   # Update configuration
   export MAX_CONCURRENT_USERS=2
   docker-compose restart avatar-service
   ```

### CUDA Version Mismatch

#### Symptoms
- `cuDNN error: CUDNN_STATUS_VERSION_MISMATCH`
- Model loading failures
- ONNX Runtime errors

#### Diagnosis
```bash
# Check CUDA version
nvcc --version
nvidia-smi

# Check container CUDA version
docker exec avatar-service nvcc --version

# Check ONNX Runtime GPU support
docker exec avatar-service python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

#### Solutions

1. **Update Base Image**
   ```dockerfile
   # Use correct CUDA version in Dockerfile
   FROM nvidia/cuda:11.8-runtime-ubuntu20.04
   ```

2. **Install Correct ONNX Runtime**
   ```bash
   # In container or requirements.txt
   pip uninstall onnxruntime-gpu
   pip install onnxruntime-gpu==1.16.0
   ```

---

## Model Loading Issues

### Models Fail to Load

#### Symptoms
- Service stuck at startup
- `FileNotFoundError` for model files
- Model validation failures

#### Diagnosis
```bash
# Check model files exist
docker exec avatar-service ls -la /app/assets/models/
docker exec avatar-service ls -la /app/assets/models/wav2lip/
docker exec avatar-service ls -la /app/assets/models/insightface/

# Test model loading manually
docker exec avatar-service python scripts/validate_models.py

# Check model download logs
docker logs avatar-service 2>&1 | grep -i "download\|model"
```

#### Solutions

1. **Download Models Manually**
   ```bash
   # Download models
   docker exec avatar-service python scripts/download_models.py --force

   # Verify checksums
   docker exec avatar-service python scripts/download_models.py --verify-checksums
   ```

2. **Check Model Permissions**
   ```bash
   # Fix permissions
   docker exec avatar-service chown -R app:app /app/assets/models/
   docker exec avatar-service chmod -R 755 /app/assets/models/
   ```

3. **Clear Model Cache**
   ```bash
   # Remove corrupted models
   docker exec avatar-service rm -rf /app/assets/models/wav2lip/*
   docker exec avatar-service rm -rf /app/assets/models/insightface/*
   
   # Re-download
   docker exec avatar-service python scripts/download_models.py
   ```

### Model Loading Timeout

#### Symptoms
- Service startup exceeds timeout
- Cold loading takes longer than expected
- Health checks fail during startup

#### Diagnosis
```bash
# Check startup logs
docker logs avatar-service 2>&1 | grep -i "loading\|timeout"

# Monitor GPU during startup
nvidia-smi -l 1

# Check system resources
htop
free -h
```

#### Solutions

1. **Increase Startup Timeout**
   ```yaml
   # In docker-compose.yml
   services:
     avatar-service:
       healthcheck:
         start_period: 120s  # Increase from 60s
   ```

2. **Optimize Model Loading**
   ```python
   # In model_config.py
   MODEL_WARMUP_TIMEOUT = 120  # Increase timeout
   PARALLEL_MODEL_LOADING = True  # Enable parallel loading
   ```

3. **Pre-warm Models**
   ```bash
   # Pre-load models before service start
   docker run --rm --gpus all -v ./assets:/app/assets \
     avatar-service:latest python scripts/model-warmup.sh
   ```

---

## Avatar Registration Issues

### Face Detection Failures

#### Symptoms
- `No face detected in avatar` errors
- Avatar registration fails
- Low face quality scores

#### Diagnosis
```bash
# Test face detection manually
docker exec avatar-service python -c "
from app.core.avatar_registrar import ColdAvatarRegistrar
registrar = ColdAvatarRegistrar()
result = registrar.test_face_detection('/path/to/test/image.jpg')
print(result)
"

# Check face detection logs
docker logs avatar-service 2>&1 | grep -i "face\|detection"

# Verify InsightFace models
docker exec avatar-service ls -la /app/assets/models/insightface/
```

#### Solutions

1. **Verify Image Quality**
   ```python
   # Test image quality
   from PIL import Image
   import numpy as np
   
   # Check image resolution
   img = Image.open('avatar.jpg')
   print(f"Resolution: {img.size}")
   
   # Check if face is clearly visible
   # Ensure good lighting and frontal view
   ```

2. **Adjust Detection Threshold**
   ```bash
   # Lower detection threshold
   export FACE_DETECTION_THRESHOLD=0.3
   docker-compose restart avatar-service
   ```

3. **Update Detection Models**
   ```bash
   # Re-download InsightFace models
   docker exec avatar-service rm -rf /app/assets/models/insightface/
   docker exec avatar-service python scripts/download_models.py --insightface-only
   ```

### Avatar Cache Corruption

#### Symptoms
- Cache loading failures
- Inconsistent avatar processing
- Cache validation errors

#### Diagnosis
```bash
# Validate avatar cache
docker exec avatar-service python scripts/validate_avatar_cache.py

# Check cache files
docker exec avatar-service ls -la /app/data/avatar_registry/face_cache/

# Check database integrity
docker exec avatar-service sqlite3 /app/data/avatar_registry/avatars.db "PRAGMA integrity_check;"
```

#### Solutions

1. **Rebuild Avatar Cache**
   ```bash
   # Clear cache
   docker exec avatar-service rm -rf /app/data/avatar_registry/face_cache/*
   
   # Rebuild cache
   docker exec avatar-service python scripts/rebuild_avatar_cache.py
   ```

2. **Fix Database Issues**
   ```bash
   # Backup database
   docker exec avatar-service cp /app/data/avatar_registry/avatars.db /app/data/avatars_backup.db
   
   # Repair database
   docker exec avatar-service sqlite3 /app/data/avatar_registry/avatars.db "VACUUM;"
   ```

3. **Reset Avatar Registry**
   ```bash
   # Complete reset (WARNING: loses all registered avatars)
   docker-compose down
   rm -rf ./data/avatar_registry/*
   docker-compose up -d
   ```

---

## Processing and Performance Issues

### High Latency

#### Symptoms
- Chunk processing takes longer than expected
- First chunk delay exceeds targets
- Poor user experience

#### Diagnosis
```bash
# Check processing metrics
curl http://localhost:5002/metrics

# Monitor GPU utilization
nvidia-smi -l 1

# Check system load
htop
iostat -x 1

# Analyze processing logs
docker logs avatar-service 2>&1 | grep -i "processing\|latency"
```

#### Solutions

1. **Optimize GPU Settings**
   ```bash
   # Set optimal GPU clocks
   sudo nvidia-smi -pl 450
   sudo nvidia-smi -ac 1215,2520
   sudo nvidia-smi -pm 1
   ```

2. **Tune Processing Parameters**
   ```bash
   # Optimize chunk size
   export CHUNK_SIZE_SECONDS=8
   
   # Reduce quality for speed
   export PROCESSING_QUALITY=fast
   
   docker-compose restart avatar-service
   ```

3. **Enable Performance Mode**
   ```bash
   # Set CPU governor to performance
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   ```

### Memory Leaks

#### Symptoms
- Gradually increasing memory usage
- Service becomes unresponsive over time
- Out of memory errors

#### Diagnosis
```bash
# Monitor memory usage
docker stats avatar-service

# Check for memory leaks in logs
docker logs avatar-service 2>&1 | grep -i "memory\|leak"

# Profile memory usage
docker exec avatar-service python scripts/memory_profiler.py
```

#### Solutions

1. **Restart Service Regularly**
   ```bash
   # Add restart policy
   # In docker-compose.yml
   services:
     avatar-service:
       restart: unless-stopped
       healthcheck:
         retries: 3
   ```

2. **Clear Caches Periodically**
   ```bash
   # Add cleanup job
   docker exec avatar-service python scripts/cleanup_resources.py --memory-cleanup
   ```

3. **Optimize Memory Settings**
   ```bash
   # Reduce cache sizes
   export AVATAR_CACHE_SIZE=1GB
   export MODEL_CACHE_SIZE=2GB
   
   docker-compose restart avatar-service
   ```

---

## Network and Connectivity Issues

### WebSocket Connection Failures

#### Symptoms
- WebSocket connections drop
- Real-time streaming fails
- Client connection errors

#### Diagnosis
```bash
# Test WebSocket connection
curl --include \
     --no-buffer \
     --header "Connection: Upgrade" \
     --header "Upgrade: websocket" \
     --header "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" \
     --header "Sec-WebSocket-Version: 13" \
     http://localhost:5002/stream

# Check WebSocket logs
docker logs avatar-service 2>&1 | grep -i "websocket\|connection"

# Monitor network connections
netstat -tulpn | grep 5002
```

#### Solutions

1. **Check Firewall Settings**
   ```bash
   # Open required ports
   sudo ufw allow 5002/tcp
   sudo ufw allow 8080/tcp  # metrics port
   ```

2. **Configure Proxy Settings**
   ```nginx
   # In nginx.conf
   location /stream {
       proxy_pass http://avatar-service:5002;
       proxy_http_version 1.1;
       proxy_set_header Upgrade $http_upgrade;
       proxy_set_header Connection "upgrade";
       proxy_read_timeout 86400;
   }
   ```

3. **Increase Connection Limits**
   ```bash
   # In system limits
   echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
   echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf
   ```

### API Request Failures

#### Symptoms
- HTTP 500 errors
- Request timeouts
- Rate limiting issues

#### Diagnosis
```bash
# Test API endpoints
curl -v http://localhost:5002/health
curl -v http://localhost:5002/avatar/list

# Check API logs
docker logs avatar-service 2>&1 | grep -i "api\|request\|response"

# Monitor request metrics
curl http://localhost:5002/metrics | grep http_requests
```

#### Solutions

1. **Check Request Format**
   ```bash
   # Test avatar registration
   curl -X POST \
        -F "file=@avatar.jpg" \
        -F "avatar_name=test" \
        -F "user_id=user123" \
        http://localhost:5002/avatar/register
   ```

2. **Adjust Rate Limits**
   ```bash
   # Increase rate limits
   export RATE_LIMIT_REQUESTS=200
   export RATE_LIMIT_WINDOW=60
   
   docker-compose restart avatar-service
   ```

3. **Enable Debug Mode**
   ```bash
   # Enable detailed logging
   export LOG_LEVEL=DEBUG
   export ENABLE_DEBUG_ENDPOINTS=true
   
   docker-compose restart avatar-service
   ```

---

## File and Storage Issues

### Upload Failures

#### Symptoms
- File upload errors
- Disk space issues
- Permission denied errors

#### Diagnosis
```bash
# Check disk space
df -h

# Check upload directory permissions
docker exec avatar-service ls -la /app/temp/uploads/
docker exec avatar-service ls -la /app/assets/avatars/registered/

# Check upload logs
docker logs avatar-service 2>&1 | grep -i "upload\|file"
```

#### Solutions

1. **Fix Permissions**
   ```bash
   # Fix upload directory permissions
   docker exec avatar-service chown -R app:app /app/temp/uploads/
   docker exec avatar-service chmod -R 755 /app/temp/uploads/
   ```

2. **Clean Up Temporary Files**
   ```bash
   # Clean temporary uploads
   docker exec avatar-service find /app/temp/uploads/ -type f -mtime +1 -delete
   ```

3. **Increase Upload Limits**
   ```bash
   # Increase file size limits
   export MAX_AVATAR_FILE_SIZE=104857600  # 100MB
   
   # Update nginx if using proxy
   client_max_body_size 100M;
   ```

### Database Issues

#### Symptoms
- SQLite database errors
- Avatar metadata corruption
- Database locking issues

#### Diagnosis
```bash
# Check database integrity
docker exec avatar-service sqlite3 /app/data/avatar_registry/avatars.db "PRAGMA integrity_check;"

# Check database permissions
docker exec avatar-service ls -la /app/data/avatar_registry/

# Check database logs
docker logs avatar-service 2>&1 | grep -i "database\|sqlite"
```

#### Solutions

1. **Repair Database**
   ```bash
   # Backup database
   docker cp avatar-service:/app/data/avatar_registry/avatars.db ./backup_avatars.db
   
   # Repair database
   docker exec avatar-service sqlite3 /app/data/avatar_registry/avatars.db "VACUUM;"
   ```

2. **Reset Database**
   ```bash
   # WARNING: This will lose all avatar data
   docker exec avatar-service rm /app/data/avatar_registry/avatars.db
   docker-compose restart avatar-service
   ```

---

## Service Management

### Service Won't Start

#### Diagnosis Steps
```bash
# Check container status
docker ps -a | grep avatar-service

# View startup logs
docker logs avatar-service

# Check resource usage
docker stats --no-stream

# Verify configuration
docker exec avatar-service env | grep -E "(GPU|MODEL|AVATAR)"
```

#### Common Solutions
1. **Resource Issues**: Ensure sufficient GPU memory and system resources
2. **Configuration**: Verify environment variables and file paths
3. **Dependencies**: Check that all required models and files are present
4. **Permissions**: Ensure proper file and directory permissions

### Service Crashes

#### Log Analysis
```bash
# Get crash logs
docker logs avatar-service --since 1h | tail -100

# Check system logs
journalctl -u docker.service --since "1 hour ago"

# Monitor system resources
dmesg | grep -i "killed\|memory\|gpu"
```

#### Recovery Steps
1. **Immediate**: Restart service with `docker-compose restart avatar-service`
2. **Investigate**: Analyze logs for root cause
3. **Preventive**: Implement monitoring and alerts
4. **Long-term**: Optimize resource usage and error handling

---

## Monitoring and Alerts

### Health Monitoring Setup

```bash
# Create monitoring script
cat > monitor_avatar_service.sh << 'EOF'
#!/bin/bash
while true; do
    if ! curl -f http://localhost:5002/health > /dev/null 2>&1; then
        echo "$(date): Avatar service health check failed"
        # Add alerting logic here
    fi
    sleep 30
done
EOF

chmod +x monitor_avatar_service.sh
```

### Performance Monitoring

```bash
# Monitor key metrics
watch -n 5 'curl -s http://localhost:5002/metrics | grep -E "(processing_time|cache_hit_rate|gpu_utilization)"'

# GPU monitoring
watch -n 1 nvidia-smi
```

---

## Emergency Procedures

### Service Recovery

```bash
# Quick recovery steps
docker-compose down
docker system prune -f
docker-compose up -d

# Health check
sleep 60
curl -f http://localhost:5002/health
```

### Data Recovery

```bash
# Restore from backup
docker-compose down
rm -rf ./data/avatar_registry/*
tar -xzf /backup/avatar-service-backup-latest.tar.gz -C ./data/avatar_registry/
docker-compose up -d
```

### Factory Reset

```bash
# Complete system reset (WARNING: loses all data)
docker-compose down
docker rmi avatar-service:latest
rm -rf ./data/*
rm -rf ./logs/*
rm -rf ./temp/*
docker-compose build
docker-compose up -d
```

This troubleshooting guide covers the most common issues and their solutions. For issues not covered here, check the service logs and contact support with detailed error information. 