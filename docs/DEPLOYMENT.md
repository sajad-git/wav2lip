# Avatar Streaming Service Deployment Guide

## Overview

This guide covers the deployment of the Avatar Streaming Service with cold model loading and avatar registration capabilities. The service is designed for GPU-accelerated processing with NVIDIA RTX 4090 optimization.

## Prerequisites

### Hardware Requirements

#### Minimum Requirements
- **GPU**: NVIDIA RTX 3080 or equivalent (12GB VRAM)
- **CPU**: 8-core Intel/AMD processor
- **RAM**: 32GB system memory
- **Storage**: 100GB available space (SSD recommended)
- **Network**: Gigabit ethernet

#### Recommended Requirements (RTX 4090 Optimized)
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **CPU**: 16-core Intel/AMD processor
- **RAM**: 64GB system memory
- **Storage**: 500GB NVMe SSD
- **Network**: 10 Gigabit ethernet

### Software Requirements

#### Host System
- **OS**: Ubuntu 20.04/22.04 LTS or CentOS 8+
- **Docker**: 24.0+ with BuildKit support
- **Docker Compose**: 2.20+
- **NVIDIA Container Toolkit**: Latest version
- **CUDA**: 11.8+ (for RTX 4090 compatibility)

#### GPU Drivers
- **NVIDIA Driver**: 525.85+ for RTX 4090
- **CUDA Toolkit**: 11.8 or 12.0+
- **cuDNN**: 8.6+

---

## Pre-Deployment Setup

### 1. NVIDIA Container Toolkit Installation

#### Ubuntu/Debian
```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

#### CentOS/RHEL
```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo

# Install nvidia-container-toolkit
sudo yum install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

### 2. GPU Validation

```bash
# Test NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Verify GPU accessibility
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 bash -c "nvidia-smi && nvcc --version"
```

### 3. System Optimization

#### GPU Memory Configuration
```bash
# Add to /etc/modprobe.d/nvidia.conf
echo "options nvidia NVreg_PreserveVideoMemoryAllocations=1" | sudo tee -a /etc/modprobe.d/nvidia.conf

# Set GPU persistence mode
sudo nvidia-smi -pm 1

# Set GPU power and clock settings for RTX 4090
sudo nvidia-smi -pl 450  # Set power limit to 450W
sudo nvidia-smi -ac 1215,2520  # Set memory and GPU clocks
```

#### System Limits
```bash
# Add to /etc/security/limits.conf
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* soft memlock unlimited" | sudo tee -a /etc/security/limits.conf
echo "* hard memlock unlimited" | sudo tee -a /etc/security/limits.conf
```

---

## Environment Configuration

### 1. Environment Files

Create environment files based on deployment type:

#### Development (.env.dev)
```env
NODE_ENV=development
GPU_MEMORY_LIMIT=16GB
MAX_CONCURRENT_USERS=2
PRELOAD_MODELS=true
AVATAR_CACHE_WARMUP=true
LOG_LEVEL=DEBUG
```

#### Staging (.env.staging)
```env
NODE_ENV=staging
GPU_MEMORY_LIMIT=20GB
MAX_CONCURRENT_USERS=3
PRELOAD_MODELS=true
AVATAR_CACHE_WARMUP=true
LOG_LEVEL=INFO
```

#### Production (.env.prod)
```env
NODE_ENV=production
GPU_MEMORY_LIMIT=20GB
MAX_CONCURRENT_USERS=5
PRELOAD_MODELS=true
AVATAR_CACHE_WARMUP=true
LOG_LEVEL=WARNING
OPENAI_API_KEY=${OPENAI_API_KEY}
MCP_TABIB_ENDPOINT=${MCP_TABIB_ENDPOINT}
SECRET_KEY=${SECRET_KEY}
```

### 2. Secrets Management

#### Using Docker Secrets
```bash
# Create secrets
echo "your-openai-api-key" | docker secret create openai_api_key -
echo "your-secret-key" | docker secret create app_secret_key -
echo "your-mcp-endpoint" | docker secret create mcp_endpoint -
```

#### Using External Secrets (Kubernetes)
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: avatar-service-secrets
type: Opaque
data:
  openai-api-key: <base64-encoded-key>
  secret-key: <base64-encoded-secret>
  mcp-endpoint: <base64-encoded-endpoint>
```

---

## Docker Deployment

### 1. Build Configuration

#### Development Build
```bash
# Build development image
docker build -f docker/Dockerfile -t avatar-service:dev .

# Or using docker-compose
docker-compose -f docker-compose.dev.yml build
```

#### Production Build
```bash
# Build production image with optimizations
docker build -f docker/Dockerfile \
  --build-arg BUILD_ENV=production \
  --build-arg CUDA_VERSION=11.8 \
  --build-arg PYTHON_VERSION=3.9 \
  -t avatar-service:prod .
```

### 2. Volume Setup

```bash
# Create required directories
mkdir -p ./assets/models/wav2lip
mkdir -p ./assets/models/insightface
mkdir -p ./assets/avatars/registered
mkdir -p ./data/avatar_registry/face_cache
mkdir -p ./logs
mkdir -p ./temp/uploads

# Set permissions
chmod 755 ./assets/avatars/registered
chmod 755 ./data/avatar_registry
chmod 777 ./temp/uploads
```

### 3. Model Download

```bash
# Download models before deployment
docker run --rm -v ./assets/models:/app/assets/models \
  avatar-service:prod python scripts/download_models.py --verify-checksums

# Verify model integrity
docker run --rm -v ./assets/models:/app/assets/models \
  avatar-service:prod python scripts/validate_models.py
```

### 4. Service Deployment

#### Single Container Deployment
```bash
# Run development container
docker run -d \
  --name avatar-service-dev \
  --gpus all \
  --shm-size=2g \
  -p 5002:5002 \
  -v ./assets:/app/assets:ro \
  -v ./data:/app/data:rw \
  -v ./logs:/app/logs:rw \
  --env-file .env.dev \
  avatar-service:dev

# Check container health
docker logs avatar-service-dev
docker exec avatar-service-dev python scripts/health_check.py
```

#### Docker Compose Deployment
```bash
# Development deployment
docker-compose -f docker-compose.dev.yml up -d

# Production deployment
docker-compose -f docker-compose.yml up -d

# Check service health
docker-compose ps
docker-compose logs avatar-service
```

---

## Production Deployment

### 1. Multi-Stage Production Setup

#### Docker Compose with GPU Optimization
```yaml
version: '3.8'

services:
  avatar-service:
    image: avatar-service:prod
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
        limits:
          memory: 32G
          cpus: '16'
    shm_size: 4gb
    ports:
      - "5002:5002"
      - "8080:8080"  # Metrics port
    volumes:
      - ./assets/models:/app/assets/models:ro
      - ./assets/avatars:/app/assets/avatars:ro
      - ./data/avatar_registry:/app/data/avatar_registry:rw
      - ./logs:/app/logs:rw
      - ./temp:/app/temp:rw
    environment:
      - GPU_MEMORY_LIMIT=20GB
      - MAX_CONCURRENT_USERS=5
      - PRELOAD_MODELS=true
      - AVATAR_CACHE_WARMUP=true
    env_file:
      - .env.prod
    healthcheck:
      test: ["CMD", "python", "/app/scripts/health_check.py", "--check-models", "--check-avatars"]
      interval: 30s
      timeout: 15s
      retries: 3
      start_period: 60s

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - avatar-service

  redis:
    image: redis:alpine
    restart: unless-stopped
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

volumes:
  redis_data:
```

### 2. Load Balancer Configuration

#### NGINX Configuration
```nginx
upstream avatar_service {
    least_conn;
    server avatar-service:5002 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    client_max_body_size 100M;
    
    # Avatar upload endpoint
    location /avatar/register {
        proxy_pass http://avatar_service;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_request_buffering off;
        proxy_read_timeout 300s;
    }
    
    # WebSocket endpoint
    location /stream {
        proxy_pass http://avatar_service;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
    
    # API endpoints
    location /api {
        proxy_pass http://avatar_service;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # Static files
    location /static {
        alias /app/static;
        expires 1h;
    }
}
```

### 3. SSL/TLS Configuration

```bash
# Generate SSL certificate (Let's Encrypt)
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com

# Or use custom certificates
mkdir -p ./nginx/ssl
cp your-cert.pem ./nginx/ssl/cert.pem
cp your-key.pem ./nginx/ssl/key.pem
```

---

## Kubernetes Deployment

### 1. Namespace and ConfigMap

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: avatar-service

---

apiVersion: v1
kind: ConfigMap
metadata:
  name: avatar-service-config
  namespace: avatar-service
data:
  GPU_MEMORY_LIMIT: "20GB"
  MAX_CONCURRENT_USERS: "5"
  PRELOAD_MODELS: "true"
  AVATAR_CACHE_WARMUP: "true"
```

### 2. Deployment with GPU

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: avatar-service
  namespace: avatar-service
spec:
  replicas: 1
  selector:
    matchLabels:
      app: avatar-service
  template:
    metadata:
      labels:
        app: avatar-service
    spec:
      containers:
      - name: avatar-service
        image: avatar-service:prod
        ports:
        - containerPort: 5002
        - containerPort: 8080
        resources:
          requests:
            memory: "16Gi"
            cpu: "8"
            nvidia.com/gpu: 1
          limits:
            memory: "32Gi"
            cpu: "16"
            nvidia.com/gpu: 1
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: avatar-service-secrets
              key: openai-api-key
        envFrom:
        - configMapRef:
            name: avatar-service-config
        volumeMounts:
        - name: models-volume
          mountPath: /app/assets/models
          readOnly: true
        - name: avatars-volume
          mountPath: /app/assets/avatars/registered
        - name: data-volume
          mountPath: /app/data
        livenessProbe:
          httpGet:
            path: /health
            port: 5002
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 5002
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: models-volume
        persistentVolumeClaim:
          claimName: models-pvc
      - name: avatars-volume
        persistentVolumeClaim:
          claimName: avatars-pvc
      - name: data-volume
        persistentVolumeClaim:
          claimName: data-pvc
      nodeSelector:
        nvidia.com/gpu.present: "true"
```

### 3. Service and Ingress

```yaml
apiVersion: v1
kind: Service
metadata:
  name: avatar-service
  namespace: avatar-service
spec:
  selector:
    app: avatar-service
  ports:
  - name: http
    port: 5002
    targetPort: 5002
  - name: metrics
    port: 8080
    targetPort: 8080

---

apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: avatar-service-ingress
  namespace: avatar-service
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: avatar-service-tls
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: avatar-service
            port:
              number: 5002
```

---

## Monitoring and Observability

### 1. Health Monitoring

```bash
# Health check script
#!/bin/bash
HEALTH_URL="http://localhost:5002/health"
READY_URL="http://localhost:5002/ready"

# Check health
if curl -f $HEALTH_URL > /dev/null 2>&1; then
    echo "Service is healthy"
else
    echo "Service health check failed"
    exit 1
fi

# Check readiness
if curl -f $READY_URL > /dev/null 2>&1; then
    echo "Service is ready"
else
    echo "Service is not ready"
    exit 1
fi
```

### 2. Prometheus Metrics

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'avatar-service'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: /metrics
    scrape_interval: 30s
```

### 3. Grafana Dashboard

Key metrics to monitor:
- GPU utilization and memory usage
- Model loading status and performance
- Avatar registration and processing times
- WebSocket connection counts
- Error rates and response times
- Avatar cache hit rates

---

## Backup and Recovery

### 1. Data Backup

```bash
#!/bin/bash
# Backup script for avatar data

BACKUP_DIR="/backup/avatar-service"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR/$DATE

# Backup avatar registry database
docker exec avatar-service sqlite3 /app/data/avatar_registry/avatars.db .dump > $BACKUP_DIR/$DATE/avatars_db.sql

# Backup registered avatars
docker cp avatar-service:/app/assets/avatars/registered $BACKUP_DIR/$DATE/

# Backup face cache
docker cp avatar-service:/app/data/avatar_registry/face_cache $BACKUP_DIR/$DATE/

# Compress backup
tar -czf $BACKUP_DIR/avatar-service-backup-$DATE.tar.gz -C $BACKUP_DIR/$DATE .

# Cleanup old backups (keep last 7 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

### 2. Disaster Recovery

```bash
#!/bin/bash
# Recovery script

BACKUP_FILE="$1"
RECOVERY_DIR="/tmp/avatar-recovery"

# Extract backup
mkdir -p $RECOVERY_DIR
tar -xzf $BACKUP_FILE -C $RECOVERY_DIR

# Stop service
docker-compose stop avatar-service

# Restore database
cat $RECOVERY_DIR/avatars_db.sql | docker exec -i avatar-service sqlite3 /app/data/avatar_registry/avatars.db

# Restore files
docker cp $RECOVERY_DIR/registered avatar-service:/app/assets/avatars/
docker cp $RECOVERY_DIR/face_cache avatar-service:/app/data/avatar_registry/

# Restart service
docker-compose start avatar-service

# Verify recovery
docker exec avatar-service python scripts/validate_avatar_cache.py
```

---

## Performance Tuning

### 1. GPU Optimization

```bash
# GPU performance tuning
nvidia-smi -pl 450  # Set power limit for RTX 4090
nvidia-smi -ac 1215,2520  # Set memory and GPU clocks

# Docker GPU configuration
echo '{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}' | sudo tee /etc/docker/daemon.json

sudo systemctl restart docker
```

### 2. Memory Optimization

```bash
# System memory tuning
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.dirty_ratio=15' | sudo tee -a /etc/sysctl.conf
echo 'vm.dirty_background_ratio=5' | sudo tee -a /etc/sysctl.conf

sudo sysctl -p
```

### 3. Container Optimization

```dockerfile
# Production Dockerfile optimizations
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Optimize Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=random

# Optimize CUDA
ENV CUDA_CACHE_DISABLE=0
ENV CUDA_CACHE_MAXSIZE=2147483648

# Optimize memory allocation
ENV MALLOC_ARENA_MAX=2
```

---

## Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check GPU visibility
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi

# Check NVIDIA container toolkit
sudo systemctl status nvidia-container-toolkit
```

#### Model Loading Failures
```bash
# Check model files
docker exec avatar-service ls -la /app/assets/models/
docker exec avatar-service python scripts/validate_models.py
```

#### Memory Issues
```bash
# Monitor GPU memory
nvidia-smi -l 1

# Check container memory
docker stats avatar-service
```

#### Performance Issues
```bash
# Check processing metrics
curl http://localhost:5002/metrics

# Monitor logs
docker logs -f avatar-service
```

---

## Security Considerations

### 1. Container Security

```bash
# Run container as non-root user
docker run --user 1000:1000 avatar-service

# Limit container capabilities
docker run --cap-drop ALL --cap-add CHOWN avatar-service
```

### 2. Network Security

```bash
# Use custom networks
docker network create avatar-network --driver bridge

# Limit port exposure
docker run -p 127.0.0.1:5002:5002 avatar-service
```

### 3. Data Security

```bash
# Encrypt volumes
docker volume create --driver local \
  --opt type=tmpfs \
  --opt device=tmpfs \
  --opt o=size=1000m,uid=1000 \
  secure-temp

# Use secrets for sensitive data
docker secret create openai_key /path/to/key
```

This deployment guide provides comprehensive instructions for deploying the Avatar Streaming Service in various environments with optimal performance and security configurations. 