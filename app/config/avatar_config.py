"""
Avatar Configuration for Avatar Streaming Service
Avatar registration and caching configuration
"""

import os
from typing import Set, Optional
from pathlib import Path


class AvatarConfig:
    """Avatar registration and caching configuration"""
    
    def __init__(self):
        # Avatar storage configuration
        self.avatar_storage_path: str = os.getenv("AVATAR_STORAGE_PATH", "/app/assets/avatars/registered")
        self.cache_storage_path: str = os.getenv("CACHE_STORAGE_PATH", "/app/data/avatar_registry/face_cache")
        
        # Supported file formats
        self.supported_formats: Set[str] = {".jpg", ".jpeg", ".png", ".gif", ".mp4", ".mov"}
        
        # File size limits
        self.max_file_size: int = int(os.getenv("MAX_AVATAR_FILE_SIZE", 50 * 1024 * 1024))  # 50MB
        self.min_file_size: int = 1024  # 1KB
        
        # Face detection configuration
        self.face_detection_threshold: float = float(os.getenv("FACE_DETECTION_THRESHOLD", 0.5))
        self.min_face_size: int = 64  # Minimum face size in pixels
        self.max_faces_per_avatar: int = 1  # Only allow single face avatars
        
        # Cache configuration
        self.cache_compression: bool = os.getenv("CACHE_COMPRESSION", "true").lower() == "true"
        self.cache_format: str = "lz4"  # Compression format
        self.auto_cleanup_days: int = int(os.getenv("AUTO_CLEANUP_DAYS", 30))
        
        # Processing configuration
        self.target_resolution: tuple = (256, 256)  # Target image resolution
        self.face_crop_size: tuple = (96, 96)  # Face crop size for wav2lip
        self.video_max_frames: int = 100  # Maximum frames to process from video
        
        # Quality thresholds
        self.min_quality_score: float = 0.7  # Minimum face quality score
        self.min_clarity_score: float = 0.6  # Minimum image clarity
        self.min_lighting_score: float = 0.5  # Minimum lighting quality
        
        # Database configuration
        self.database_path: str = os.path.join(
            os.path.dirname(self.cache_storage_path), 
            "avatars.db"
        )
        
        # Ensure directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        Path(self.avatar_storage_path).mkdir(parents=True, exist_ok=True)
        Path(self.cache_storage_path).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.database_path)).mkdir(parents=True, exist_ok=True)
    
    def get_avatar_path(self, avatar_id: str) -> str:
        """Get avatar directory path"""
        return os.path.join(self.avatar_storage_path, avatar_id)
    
    def get_cache_path(self, avatar_id: str) -> str:
        """Get face cache file path"""
        return os.path.join(self.cache_storage_path, f"{avatar_id}_face_data.pkl")
    
    def get_metadata_path(self, avatar_id: str) -> str:
        """Get avatar metadata file path"""
        return os.path.join(self.get_avatar_path(avatar_id), "metadata.json")
    
    def get_original_file_path(self, avatar_id: str, file_extension: str) -> str:
        """Get original avatar file path"""
        return os.path.join(self.get_avatar_path(avatar_id), f"original{file_extension}")
    
    def get_processed_face_path(self, avatar_id: str) -> str:
        """Get processed face image path"""
        return os.path.join(self.get_avatar_path(avatar_id), "processed_face.jpg")
    
    def validate_avatar_format(self, file_path: str) -> bool:
        """Validate avatar file format"""
        file_extension = os.path.splitext(file_path)[1].lower()
        return file_extension in self.supported_formats
    
    def validate_file_size(self, file_size: int) -> bool:
        """Validate avatar file size"""
        return self.min_file_size <= file_size <= self.max_file_size
    
    def is_video_format(self, file_path: str) -> bool:
        """Check if file is video format"""
        file_extension = os.path.splitext(file_path)[1].lower()
        return file_extension in {".gif", ".mp4", ".mov"}
    
    def is_image_format(self, file_path: str) -> bool:
        """Check if file is image format"""
        file_extension = os.path.splitext(file_path)[1].lower()
        return file_extension in {".jpg", ".jpeg", ".png"}
    
    def get_target_dimensions(self, format_type: str) -> tuple:
        """Get target dimensions based on format"""
        if format_type == "face_crop":
            return self.face_crop_size
        else:
            return self.target_resolution
    
    def should_compress_cache(self) -> bool:
        """Check if cache compression is enabled"""
        return self.cache_compression
    
    def get_cleanup_threshold_date(self):
        """Get threshold date for cache cleanup"""
        from datetime import datetime, timedelta
        return datetime.now() - timedelta(days=self.auto_cleanup_days)


# Global avatar configuration instance
avatar_config = AvatarConfig() 