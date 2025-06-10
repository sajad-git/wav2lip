"""
GPU Configuration and Management
Handles GPU resource allocation, memory management, and CUDA optimization settings.
"""

import os
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import torch
import onnxruntime as ort

logger = logging.getLogger(__name__)

@dataclass
class GPUConfig:
    """GPU configuration settings for avatar processing."""
    
    # CUDA Settings
    cuda_device_id: int = 0
    cuda_visible_devices: str = "0"
    gpu_memory_limit: str = "20GB"
    model_cache_size: str = "4GB"
    
    # ONNX Runtime Settings
    onnx_disable_global_thread_pool: bool = True
    omp_num_threads: int = 8
    onnx_graph_optimization_level: int = 3  # ORT_ENABLE_ALL
    
    # Memory Management
    gpu_memory_fraction: float = 0.8
    allow_memory_growth: bool = True
    memory_pool_initial_size: int = 2 * 1024 * 1024 * 1024  # 2GB
    
    # Performance Settings
    enable_tensor_rt: bool = True
    enable_cuda_provider: bool = True
    enable_cpu_fallback: bool = False
    
    # Cache Settings
    preload_models: bool = True
    model_warmup: bool = True
    cache_optimization: bool = True
    
    def __post_init__(self):
        """Set environment variables based on configuration."""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        os.environ["ONNX_DISABLE_GLOBAL_THREAD_POOL"] = str(int(self.onnx_disable_global_thread_pool))
        os.environ["OMP_NUM_THREADS"] = str(self.omp_num_threads)

class GPUManager:
    """GPU resource management and optimization."""
    
    def __init__(self, config: GPUConfig):
        self.config = config
        self.available_providers = []
        self.memory_pools = {}
        
    def initialize_gpu(self) -> bool:
        """
        Initialize GPU and validate accessibility.
        
        Returns:
            bool: GPU initialization success status
        """
        try:
            # Check CUDA availability
            if not torch.cuda.is_available():
                logger.error("CUDA is not available")
                return False
                
            # Set CUDA device
            torch.cuda.set_device(self.config.cuda_device_id)
            
            # Check GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            logger.info(f"GPU Total Memory: {total_memory / (1024**3):.2f}GB")
            
            # Initialize ONNX Runtime providers
            self._initialize_onnx_providers()
            
            # Allocate initial memory pools
            self._allocate_memory_pools()
            
            logger.info("GPU initialization successful")
            return True
            
        except Exception as e:
            logger.error(f"GPU initialization failed: {e}")
            return False
    
    def _initialize_onnx_providers(self):
        """Initialize ONNX Runtime execution providers."""
        available_providers = ort.get_available_providers()
        logger.info(f"Available ONNX providers: {available_providers}")
        
        if 'CUDAExecutionProvider' in available_providers and self.config.enable_cuda_provider:
            cuda_options = {
                'device_id': self.config.cuda_device_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': self._parse_memory_size(self.config.gpu_memory_limit),
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }
            self.available_providers = [('CUDAExecutionProvider', cuda_options)]
            
        if 'TensorrtExecutionProvider' in available_providers and self.config.enable_tensor_rt:
            tensorrt_options = {
                'device_id': self.config.cuda_device_id,
                'trt_max_workspace_size': 2147483648,  # 2GB
                'trt_fp16_enable': True,
            }
            self.available_providers.insert(0, ('TensorrtExecutionProvider', tensorrt_options))
        
        if self.config.enable_cpu_fallback:
            self.available_providers.append('CPUExecutionProvider')
            
        logger.info(f"Configured ONNX providers: {[p[0] if isinstance(p, tuple) else p for p in self.available_providers]}")
    
    def _allocate_memory_pools(self):
        """Allocate GPU memory pools for efficient processing."""
        try:
            if torch.cuda.is_available():
                # Pre-allocate memory for model caching
                cache_size = self._parse_memory_size(self.config.model_cache_size)
                self.memory_pools['model_cache'] = torch.cuda.memory.MemoryPool()
                
                # Set memory fraction
                torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)
                
                if self.config.allow_memory_growth:
                    torch.cuda.empty_cache()
                    
                logger.info(f"Memory pools allocated successfully")
                
        except Exception as e:
            logger.error(f"Memory pool allocation failed: {e}")
    
    def _parse_memory_size(self, size_str: str) -> int:
        """
        Parse memory size string to bytes.
        
        Args:
            size_str: Memory size string (e.g., "4GB", "2048MB")
            
        Returns:
            int: Memory size in bytes
        """
        size_str = size_str.upper()
        if size_str.endswith('GB'):
            return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
        elif size_str.endswith('MB'):
            return int(float(size_str[:-2]) * 1024 * 1024)
        elif size_str.endswith('KB'):
            return int(float(size_str[:-2]) * 1024)
        else:
            return int(size_str)
    
    def get_onnx_session_options(self) -> Tuple[list, dict]:
        """
        Get optimized ONNX session options.
        
        Returns:
            Tuple of providers and session options
        """
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel(self.config.onnx_graph_optimization_level)
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_reuse = True
        
        if self.config.omp_num_threads > 0:
            session_options.intra_op_num_threads = self.config.omp_num_threads
            session_options.inter_op_num_threads = 1
        
        return self.available_providers, session_options
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        Get current GPU memory usage information.
        
        Returns:
            Dict containing memory usage statistics
        """
        if not torch.cuda.is_available():
            return {}
            
        try:
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device)
            cached = torch.cuda.memory_reserved(device)
            total = torch.cuda.get_device_properties(device).total_memory
            
            return {
                'allocated_gb': allocated / (1024**3),
                'cached_gb': cached / (1024**3),
                'total_gb': total / (1024**3),
                'utilization_percent': (allocated / total) * 100,
            }
        except Exception as e:
            logger.error(f"Failed to get memory info: {e}")
            return {}
    
    def cleanup_memory(self):
        """Clean up GPU memory and caches."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("GPU memory cleanup completed")
        except Exception as e:
            logger.error(f"GPU memory cleanup failed: {e}")

def get_gpu_config() -> GPUConfig:
    """
    Get GPU configuration from environment variables.
    
    Returns:
        GPUConfig: Configured GPU settings
    """
    return GPUConfig(
        cuda_device_id=int(os.getenv("CUDA_DEVICE_ID", "0")),
        cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES", "0"),
        gpu_memory_limit=os.getenv("GPU_MEMORY_LIMIT", "20GB"),
        model_cache_size=os.getenv("MODEL_CACHE_SIZE", "4GB"),
        onnx_disable_global_thread_pool=os.getenv("ONNX_DISABLE_GLOBAL_THREAD_POOL", "1") == "1",
        omp_num_threads=int(os.getenv("OMP_NUM_THREADS", "8")),
        preload_models=os.getenv("PRELOAD_MODELS", "true").lower() == "true",
        model_warmup=os.getenv("MODEL_WARMUP", "true").lower() == "true",
    )

# Global GPU manager instance
gpu_manager: Optional[GPUManager] = None

def initialize_gpu_manager() -> GPUManager:
    """
    Initialize the global GPU manager.
    
    Returns:
        GPUManager: Initialized GPU manager instance
    """
    global gpu_manager
    if gpu_manager is None:
        config = get_gpu_config()
        gpu_manager = GPUManager(config)
        gpu_manager.initialize_gpu()
    return gpu_manager

def get_gpu_manager() -> Optional[GPUManager]:
    """
    Get the global GPU manager instance.
    
    Returns:
        Optional[GPUManager]: GPU manager instance or None if not initialized
    """
    return gpu_manager 