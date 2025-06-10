"""
Model Configuration and Management
Handles model paths, loading settings, and optimization parameters for Wav2Lip and InsightFace models.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model loading and management."""
    
    # Base paths
    models_base_path: str = "/app/assets/models"
    wav2lip_model_path: str = "/app/assets/models/wav2lip"
    insightface_model_path: str = "/app/assets/models/insightface_func/models"
    
    # Wav2Lip Models
    wav2lip_model_file: str = "wav2lip.onnx"
    wav2lip_gan_model_file: str = "wav2lip_gan.onnx"
    
    # InsightFace Models
    insightface_model_pack: str = "antelope"
    face_detection_model: str = "scrfd_2.5g_bnkps.onnx"
    face_recognition_model: str = "glintr100.onnx"
    
    # Loading Settings
    preload_models: bool = True
    model_warmup: bool = True
    validate_models: bool = True
    cache_models: bool = True
    
    # Performance Settings
    use_gpu: bool = True
    batch_size: int = 1
    input_size: Tuple[int, int] = (96, 96)  # Face input size for Wav2Lip
    output_size: Tuple[int, int] = (96, 96)  # Face output size
    
    # Quality Settings
    quality_mode: str = "balanced"  # "fast", "balanced", "quality"
    detection_threshold: float = 0.5
    recognition_threshold: float = 0.6
    
    # Cache Settings
    model_cache_ttl: int = 3600  # 1 hour
    memory_optimization: bool = True
    lazy_loading: bool = False

@dataclass
class Wav2LipModelInfo:
    """Information about Wav2Lip model configuration."""
    
    model_name: str
    model_path: str
    input_shape: Dict[str, List[int]]
    output_shape: Dict[str, List[int]]
    supported_audio_length: int = 1600  # Audio mel-spectrogram length
    supported_face_size: int = 96
    frame_rate: int = 25
    
    def validate_model_file(self) -> bool:
        """Validate model file exists and is accessible."""
        return Path(self.model_path).exists() and Path(self.model_path).is_file()

@dataclass
class InsightFaceModelInfo:
    """Information about InsightFace model configuration."""
    
    model_name: str
    model_path: str
    input_size: Tuple[int, int]
    detection_threshold: float
    nms_threshold: float = 0.4
    
    def validate_model_file(self) -> bool:
        """Validate model file exists and is accessible."""
        return Path(self.model_path).exists() and Path(self.model_path).is_file()

class ModelRegistry:
    """Registry for managing model configurations and metadata."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.wav2lip_models: Dict[str, Wav2LipModelInfo] = {}
        self.insightface_models: Dict[str, InsightFaceModelInfo] = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize model registry with default configurations."""
        
        # Wav2Lip Models
        self.wav2lip_models["wav2lip"] = Wav2LipModelInfo(
            model_name="wav2lip",
            model_path=os.path.join(self.config.wav2lip_model_path, self.config.wav2lip_model_file),
            input_shape={
                "audio": [1, 1, 80, 16],  # [batch, channels, mel_bins, time_steps]
                "face": [1, 6, 96, 96]    # [batch, channels, height, width]
            },
            output_shape={
                "output": [1, 3, 96, 96]  # [batch, channels, height, width]
            }
        )
        
        self.wav2lip_models["wav2lip_gan"] = Wav2LipModelInfo(
            model_name="wav2lip_gan",
            model_path=os.path.join(self.config.wav2lip_model_path, self.config.wav2lip_gan_model_file),
            input_shape={
                "audio": [1, 1, 80, 16],
                "face": [1, 6, 96, 96]
            },
            output_shape={
                "output": [1, 3, 96, 96]
            }
        )
        
        # InsightFace Models
        antelope_path = os.path.join(self.config.insightface_model_path, self.config.insightface_model_pack)
        
        self.insightface_models["detection"] = InsightFaceModelInfo(
            model_name="face_detection",
            model_path=os.path.join(antelope_path, self.config.face_detection_model),
            input_size=(640, 640),
            detection_threshold=self.config.detection_threshold
        )
        
        # Only include recognition model if we have it
        recognition_model_path = os.path.join(antelope_path, self.config.face_recognition_model)
        if os.path.exists(recognition_model_path):
            self.insightface_models["recognition"] = InsightFaceModelInfo(
                model_name="face_recognition",
                model_path=recognition_model_path,
                input_size=(112, 112),
                detection_threshold=self.config.recognition_threshold
            )
    
    def get_wav2lip_model_info(self, model_name: str) -> Optional[Wav2LipModelInfo]:
        """Get Wav2Lip model information by name."""
        return self.wav2lip_models.get(model_name)
    
    def get_insightface_model_info(self, model_name: str) -> Optional[InsightFaceModelInfo]:
        """Get InsightFace model information by name."""
        return self.insightface_models.get(model_name)
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models by category."""
        return {
            "wav2lip": list(self.wav2lip_models.keys()),
            "insightface": list(self.insightface_models.keys())
        }
    
    def validate_all_models(self) -> Dict[str, bool]:
        """Validate all registered models exist."""
        validation_results = {}
        
        for name, model_info in self.wav2lip_models.items():
            validation_results[f"wav2lip_{name}"] = model_info.validate_model_file()
        
        for name, model_info in self.insightface_models.items():
            validation_results[f"insightface_{name}"] = model_info.validate_model_file()
        
        return validation_results
    
    def get_model_download_urls(self) -> Dict[str, str]:
        """Get download URLs for models."""
        return {
            "wav2lip.onnx": "https://github.com/AliaksandrSiarohin/wav2lip-ONNX/releases/download/v1.0/wav2lip.onnx",
            "wav2lip_gan.onnx": "https://github.com/AliaksandrSiarohin/wav2lip-ONNX/releases/download/v1.0/wav2lip_gan.onnx",
            "scrfd_2.5g_bnkps.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_2.5g_bnkps.onnx",
            "antelope": "https://github.com/deepinsight/insightface/releases/download/v0.7/antelope.zip"
        }

class ModelLoadingStrategy:
    """Strategy for model loading based on configuration."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def should_preload_models(self) -> bool:
        """Determine if models should be preloaded."""
        return self.config.preload_models
    
    def should_warmup_models(self) -> bool:
        """Determine if models should be warmed up."""
        return self.config.model_warmup
    
    def get_loading_priority(self) -> List[str]:
        """Get model loading priority order."""
        if self.config.quality_mode == "fast":
            return ["wav2lip", "insightface_detection"]
        elif self.config.quality_mode == "quality":
            return ["wav2lip_gan", "insightface_detection", "insightface_recognition"]
        else:  # balanced
            return ["wav2lip", "wav2lip_gan", "insightface_detection"]
    
    def get_memory_optimization_settings(self) -> Dict[str, Any]:
        """Get memory optimization settings based on configuration."""
        settings = {
            "enable_memory_pattern": True,
            "enable_mem_reuse": True,
            "memory_limit_mb": 4096 if self.config.memory_optimization else None,
            "graph_optimization_level": "all" if self.config.memory_optimization else "basic"
        }
        return settings

def get_model_config() -> ModelConfig:
    """
    Get model configuration from environment variables.
    
    Returns:
        ModelConfig: Configured model settings
    """
    return ModelConfig(
        models_base_path=os.getenv("MODELS_BASE_PATH", "/app/assets/models"),
        preload_models=os.getenv("PRELOAD_MODELS", "true").lower() == "true",
        model_warmup=os.getenv("MODEL_WARMUP", "true").lower() == "true",
        validate_models=os.getenv("VALIDATE_MODELS", "true").lower() == "true",
        use_gpu=os.getenv("USE_GPU", "true").lower() == "true",
        quality_mode=os.getenv("QUALITY_MODE", "balanced"),
        detection_threshold=float(os.getenv("DETECTION_THRESHOLD", "0.5")),
        memory_optimization=os.getenv("MEMORY_OPTIMIZATION", "true").lower() == "true",
    )

# Global model registry instance
model_registry: Optional[ModelRegistry] = None

def initialize_model_registry() -> ModelRegistry:
    """
    Initialize the global model registry.
    
    Returns:
        ModelRegistry: Initialized model registry instance
    """
    global model_registry
    if model_registry is None:
        config = get_model_config()
        model_registry = ModelRegistry(config)
    return model_registry

def get_model_registry() -> Optional[ModelRegistry]:
    """
    Get the global model registry instance.
    
    Returns:
        Optional[ModelRegistry]: Model registry instance or None if not initialized
    """
    return model_registry

def create_model_metadata_file(model_path: str, model_info: Dict[str, Any]):
    """
    Create metadata file for a model.
    
    Args:
        model_path: Path to the model file
        model_info: Model metadata information
    """
    metadata_path = model_path.replace('.onnx', '_metadata.json')
    
    import json
    try:
        with open(metadata_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        logger.info(f"Created metadata file: {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to create metadata file {metadata_path}: {e}")

def validate_model_integrity(model_path: str) -> bool:
    """
    Validate model file integrity.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        bool: True if model is valid, False otherwise
    """
    try:
        import onnx
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        logger.info(f"Model validation successful: {model_path}")
        return True
    except Exception as e:
        logger.error(f"Model validation failed for {model_path}: {e}")
        return False

# Model file checksums for integrity verification
MODEL_CHECKSUMS = {
    "wav2lip.onnx": "a1b2c3d4e5f6789",  # Replace with actual checksums
    "wav2lip_gan.onnx": "f6e5d4c3b2a1987",  # Replace with actual checksums
}

def verify_model_checksum(model_path: str) -> bool:
    """
    Verify model file checksum.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        bool: True if checksum matches, False otherwise
    """
    import hashlib
    
    model_name = os.path.basename(model_path)
    expected_checksum = MODEL_CHECKSUMS.get(model_name)
    
    if not expected_checksum:
        logger.warning(f"No checksum available for {model_name}")
        return True
    
    try:
        with open(model_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        if file_hash == expected_checksum:
            logger.info(f"Checksum verification successful: {model_name}")
            return True
        else:
            logger.error(f"Checksum mismatch for {model_name}: expected {expected_checksum}, got {file_hash}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to verify checksum for {model_path}: {e}")
        return False 