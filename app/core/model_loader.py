"""
Cold Model Loader for Avatar Streaming Service
Pre-loads wav2lip and face detection models into GPU memory
"""

import os
import time
import logging
import psutil
import pickle
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sys
sys.path.append('/app/assets/models')
from insightface_func.face_detect_crop_single import Face_detect_crop

from app.config.settings import settings


@dataclass
class PerformanceMetrics:
    """Performance metrics for model operations"""
    load_time_seconds: float
    inference_time_ms: float
    memory_usage_mb: int
    gpu_memory_mb: int


@dataclass
class ModelInstance:
    """Model instance with metadata"""
    session: Any
    input_specs: Dict[str, Any]
    output_specs: Dict[str, Any]
    performance_profile: PerformanceMetrics
    memory_usage: int


class GPUMemoryPool:
    """GPU memory management for model instances"""
    
    def __init__(self, total_memory_limit: int):
        self.total_memory_limit = total_memory_limit
        self.allocated_memory = 0
        self.memory_pools = {}
        self.logger = logging.getLogger(__name__)
    
    def allocate_pool(self, pool_name: str, size_bytes: int) -> bool:
        """Allocate GPU memory pool"""
        if self.allocated_memory + size_bytes > self.total_memory_limit:
            self.logger.warning(f"Cannot allocate {size_bytes} bytes for {pool_name}")
            return False
        
        self.memory_pools[pool_name] = size_bytes
        self.allocated_memory += size_bytes
        self.logger.info(f"Allocated {size_bytes // (1024*1024)} MB for {pool_name}")
        return True
    
    def deallocate_pool(self, pool_name: str) -> None:
        """Deallocate GPU memory pool"""
        if pool_name in self.memory_pools:
            size_bytes = self.memory_pools.pop(pool_name)
            self.allocated_memory -= size_bytes
            self.logger.info(f"Deallocated {size_bytes // (1024*1024)} MB from {pool_name}")
    
    def get_available_memory(self) -> int:
        """Get available GPU memory"""
        return self.total_memory_limit - self.allocated_memory
    
    def optimize_allocation(self) -> None:
        """Optimize memory allocation"""
        # Implement memory defragmentation if needed
        pass


class ColdModelLoader:
    """Cold model loader with GPU memory management"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.loaded_models: Dict[str, ModelInstance] = {}
        self.models_loaded = False
        
        # Initialize GPU memory pool
        gpu_memory_limit = settings.gpu_memory_limit_bytes
        self.gpu_memory_pool = GPUMemoryPool(gpu_memory_limit)
        
        # Model paths
        self.model_paths = {
            "wav2lip": os.path.join(settings.model_storage_path, "wav2lip", "wav2lip.onnx"),
            "wav2lip_gan": os.path.join(settings.model_storage_path, "wav2lip", "wav2lip_gan.onnx"),
            "face_detector": os.path.join(settings.model_storage_path, "insightface_func", "models")
        }
    
    async def load_all_models(self) -> None:
        """Load all models into GPU memory"""
        start_time = time.time()
        self.logger.info("🔥 Starting cold model loading...")
        
        try:
            # Load wav2lip models
            wav2lip_models = await self.load_wav2lip_models()
            
            # Load face detection models
            face_detector = await self.load_face_detection_models()
            
            # Validate all models
            await self.validate_model_performance()
            
            self.models_loaded = True
            load_time = time.time() - start_time
            self.logger.info(f"✅ All models loaded successfully in {load_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"❌ Model loading failed: {str(e)}")
            raise
    
    async def load_wav2lip_models(self) -> Dict[str, ort.InferenceSession]:
        """Load wav2lip ONNX models into GPU memory"""
        self.logger.info("🎭 Loading Wav2Lip models...")
        
        wav2lip_models = {}
        
        for model_name in ["wav2lip", "wav2lip_gan"]:
            model_path = self.model_paths[model_name]
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Wav2Lip model not found: {model_path}")
            
            # Create ONNX session with GPU optimization
            start_time = time.time()
            session = self._create_onnx_session(model_path, model_name)
            load_time = time.time() - start_time
            
            # Get model specifications
            input_specs = {inp.name: inp.shape for inp in session.get_inputs()}
            output_specs = {out.name: out.shape for out in session.get_outputs()}
            
            # Measure performance
            perf_metrics = await self._measure_model_performance(session, model_name)
            perf_metrics.load_time_seconds = load_time
            
            # Create model instance
            model_instance = ModelInstance(
                session=session,
                input_specs=input_specs,
                output_specs=output_specs,
                performance_profile=perf_metrics,
                memory_usage=self._estimate_model_memory(session)
            )
            
            self.loaded_models[model_name] = model_instance
            wav2lip_models[model_name] = session
            
            self.logger.info(f"✅ Loaded {model_name} - Load time: {load_time:.2f}s")
        
        return wav2lip_models
    
    async def load_face_detection_models(self) -> Face_detect_crop:
        """Load InsightFace models for face detection"""
        self.logger.info("👤 Loading InsightFace models...")
        
        start_time = time.time()
        
        # Initialize face detector
        face_detector = Face_detect_crop(
            name='antelope', 
            root=self.model_paths["face_detector"]
        )
        
        # Prepare face detector with GPU
        ctx_id = 0 if ort.get_device() == 'GPU' else -1
        face_detector.prepare(
            ctx_id=ctx_id, 
            det_thresh=settings.face_detection_threshold, 
            det_size=(320, 320),
            mode='none'
        )
        
        load_time = time.time() - start_time
        
        # Create dummy performance metrics
        perf_metrics = PerformanceMetrics(
            load_time_seconds=load_time,
            inference_time_ms=0.0,  # Will be measured during warmup
            memory_usage_mb=500,  # Estimated
            gpu_memory_mb=1000   # Estimated
        )
        
        # Create model instance
        model_instance = ModelInstance(
            session=face_detector,
            input_specs={"image": [320, 320, 3]},
            output_specs={"bbox": [4], "kps": [5, 2]},
            performance_profile=perf_metrics,
            memory_usage=500 * 1024 * 1024  # 500MB estimated
        )
        
        self.loaded_models["face_detector"] = model_instance
        
        self.logger.info(f"✅ Loaded face detector - Load time: {load_time:.2f}s")
        return face_detector
    
    def _create_onnx_session(self, model_path: str, model_name: str) -> ort.InferenceSession:
        """Create optimized ONNX Runtime session"""
        
        # Session options for optimization
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.inter_op_num_threads = settings.omp_num_threads
        session_options.intra_op_num_threads = settings.omp_num_threads
        
        # Allocate GPU memory for this model
        model_memory = 2 * 1024 * 1024 * 1024  # 2GB per model
        if not self.gpu_memory_pool.allocate_pool(model_name, model_memory):
            self.logger.warning(f"Could not allocate GPU memory for {model_name}")
        
        # Configure providers (GPU first, CPU fallback)
        providers = ["CPUExecutionProvider"]
        if ort.get_device() == 'GPU':
            providers = [
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": model_memory,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                }),
                "CPUExecutionProvider"
            ]
        
        # Create session
        session = ort.InferenceSession(
            model_path, 
            sess_options=session_options, 
            providers=providers
        )
        
        return session
    
    async def _measure_model_performance(self, session: ort.InferenceSession, model_name: str) -> PerformanceMetrics:
        """Measure model inference performance"""
        
        # Create dummy inputs based on model type
        if "wav2lip" in model_name:
            # Wav2lip input shapes
            dummy_video = np.random.rand(1, 6, 96, 96).astype(np.float32)
            dummy_audio = np.random.rand(1, 1, 80, 16).astype(np.float32)
            inputs = {"video_frames": dummy_video, "mel_spectrogram": dummy_audio}
        else:
            # Other models - create appropriate dummy inputs
            inputs = {}
        
        # Warm up with a few inferences
        for _ in range(3):
            if inputs:
                _ = session.run(None, inputs)
        
        # Measure inference time
        if inputs:
            start_time = time.time()
            for _ in range(10):
                _ = session.run(None, inputs)
            avg_inference_time = (time.time() - start_time) / 10 * 1000  # ms
        else:
            avg_inference_time = 0.0
        
        # Get memory usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss // (1024 * 1024)  # MB
        
        return PerformanceMetrics(
            load_time_seconds=0.0,  # Will be set by caller
            inference_time_ms=avg_inference_time,
            memory_usage_mb=memory_usage,
            gpu_memory_mb=1000  # Estimated GPU memory
        )
    
    def _estimate_model_memory(self, session: ort.InferenceSession) -> int:
        """Estimate model memory usage"""
        # Simple estimation based on model parameters
        # In practice, you might want more sophisticated estimation
        return 1024 * 1024 * 1024  # 1GB per model (estimated)
    
    async def validate_model_performance(self) -> Dict[str, PerformanceMetrics]:
        """Validate performance of all loaded models"""
        self.logger.info("⚡ Validating model performance...")
        
        performance_metrics = {}
        
        for model_name, model_instance in self.loaded_models.items():
            if model_name in ["wav2lip", "wav2lip_gan"]:
                # Test wav2lip inference
                session = model_instance.session
                dummy_video = np.random.rand(1, 6, 96, 96).astype(np.float32)
                dummy_audio = np.random.rand(1, 1, 80, 16).astype(np.float32)
                
                start_time = time.time()
                output = session.run(None, {
                    "video_frames": dummy_video, 
                    "mel_spectrogram": dummy_audio
                })
                inference_time = (time.time() - start_time) * 1000
                
                self.logger.info(f"✅ {model_name} inference: {inference_time:.2f}ms")
                
            elif model_name == "face_detector":
                # Test face detection
                face_detector = model_instance.session
                dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                
                start_time = time.time()
                bbox = face_detector.get_bbox(dummy_image)
                inference_time = (time.time() - start_time) * 1000
                
                self.logger.info(f"✅ face_detector inference: {inference_time:.2f}ms")
            
            performance_metrics[model_name] = model_instance.performance_profile
        
        return performance_metrics
    
    def get_model_instance(self, model_name: str) -> Any:
        """Get pre-loaded model instance"""
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
        
        return self.loaded_models[model_name].session
    
    def get_model_metadata(self, model_name: str) -> ModelInstance:
        """Get complete model metadata"""
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
        
        return self.loaded_models[model_name]
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage for all loaded models"""
        memory_usage = {}
        
        for model_name, model_instance in self.loaded_models.items():
            memory_usage[model_name] = model_instance.memory_usage
        
        return memory_usage
    
    def cleanup_models(self) -> None:
        """Clean up loaded models and free memory"""
        self.logger.info("🧹 Cleaning up loaded models...")
        
        for model_name in list(self.loaded_models.keys()):
            # Deallocate GPU memory
            self.gpu_memory_pool.deallocate_pool(model_name)
            
            # Remove model instance
            del self.loaded_models[model_name]
        
        self.models_loaded = False
        self.logger.info("✅ Model cleanup completed") 