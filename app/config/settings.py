"""
Settings Configuration for Avatar Streaming Service
Handles environment variables and application configuration
"""

import os
from typing import Set, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Service Configuration
    service_name: str = "Avatar Streaming Service"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 5002
    workers: int = 1
    log_level: str = "INFO"
    
    # Model Loading Configuration
    preload_models: bool = True
    preload_avatars: bool = True
    avatar_cache_warmup: bool = True
    
    # GPU and Memory Configuration
    gpu_memory_limit: str = "20GB"
    model_cache_size: str = "4GB"
    avatar_cache_size: str = "2GB"
    max_registered_avatars: int = 100
    
    # CUDA Configuration
    cuda_visible_devices: str = "0"
    onnx_disable_global_thread_pool: int = 1
    omp_num_threads: int = 8
    
    # Performance Settings
    max_concurrent_users: int = 3
    chunk_buffer_size: int = 5
    processing_timeout: int = 30
    
    # API Keys and External Services
    openai_api_key: Optional[str] = None
    mcp_server_url: str = "http://5.9.72.171:8007/mcp"
    
    # File Paths
    avatar_storage_path: str = "/app/assets/avatars/registered"
    cache_storage_path: str = "/app/data/avatar_registry/face_cache"
    model_storage_path: str = "/app/assets/models"
    log_storage_path: str = "/app/logs"
    temp_storage_path: str = "/app/temp"
    
    # Avatar Configuration
    supported_avatar_formats: Set[str] = {".jpg", ".jpeg", ".png", ".gif", ".mp4", ".mov"}
    max_avatar_file_size: int = 50 * 1024 * 1024  # 50MB
    face_detection_threshold: float = 0.5
    cache_compression: bool = True
    auto_cleanup_days: int = 30
    
    # Audio Configuration
    audio_sample_rate: int = 16000
    chunk_duration_seconds: float = 5.0
    max_audio_duration: int = 300  # 5 minutes
    
    # Video Configuration
    video_fps: int = 25
    video_resolution: str = "256x256"
    video_quality: str = "medium"
    
    # Security Configuration
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_origins: list = ["*"]
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    
    # Database Configuration
    database_url: str = "sqlite:///./data/avatar_registry/avatars.db"
    database_echo: bool = False
    
    # Monitoring Configuration
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    
    # Development Configuration
    debug: bool = False
    reload: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Ensure required directories exist
        os.makedirs(self.avatar_storage_path, exist_ok=True)
        os.makedirs(self.cache_storage_path, exist_ok=True)
        os.makedirs(self.log_storage_path, exist_ok=True)
        os.makedirs(self.temp_storage_path, exist_ok=True)
        
        # Set environment variables for CUDA
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)
        os.environ["ONNX_DISABLE_GLOBAL_THREAD_POOL"] = str(self.onnx_disable_global_thread_pool)
        os.environ["OMP_NUM_THREADS"] = str(self.omp_num_threads)
    
    @property
    def gpu_memory_limit_bytes(self) -> int:
        """Convert GPU memory limit to bytes"""
        limit_str = self.gpu_memory_limit.upper()
        if limit_str.endswith("GB"):
            return int(limit_str[:-2]) * 1024 * 1024 * 1024
        elif limit_str.endswith("MB"):
            return int(limit_str[:-2]) * 1024 * 1024
        else:
            return int(limit_str)
    
    @property
    def model_cache_size_bytes(self) -> int:
        """Convert model cache size to bytes"""
        cache_str = self.model_cache_size.upper()
        if cache_str.endswith("GB"):
            return int(cache_str[:-2]) * 1024 * 1024 * 1024
        elif cache_str.endswith("MB"):
            return int(cache_str[:-2]) * 1024 * 1024
        else:
            return int(cache_str)
    
    @property
    def avatar_cache_size_bytes(self) -> int:
        """Convert avatar cache size to bytes"""
        cache_str = self.avatar_cache_size.upper()
        if cache_str.endswith("GB"):
            return int(cache_str[:-2]) * 1024 * 1024 * 1024
        elif cache_str.endswith("MB"):
            return int(cache_str[:-2]) * 1024 * 1024
        else:
            return int(cache_str)


# Global settings instance
settings = Settings() 