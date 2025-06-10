"""
Health Endpoints for Avatar Streaming Service Monitoring
Provides comprehensive health check endpoints for models, avatar cache, GPU, and system components.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
import time
import psutil
import GPUtil
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import asyncio

from app.core.model_loader import ColdModelLoader
from app.core.face_cache_manager import FaceCacheManager
from app.core.resource_manager import GPUResourceManager
from app.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class HealthMetrics:
    """Health metrics data structure"""
    timestamp: datetime
    status: str
    response_time_ms: float
    details: Dict[str, Any]


@dataclass
class ComponentHealth:
    """Individual component health status"""
    name: str
    status: str  # "healthy", "degraded", "error"
    last_check: datetime
    response_time_ms: float
    details: Dict[str, Any]
    errors: List[str]


class HealthChecker:
    """Comprehensive health checking system"""
    
    def __init__(self):
        self.settings = get_settings()
        self.model_loader: Optional[ColdModelLoader] = None
        self.face_cache_manager: Optional[FaceCacheManager] = None
        self.gpu_resource_manager: Optional[GPUResourceManager] = None
        self.last_health_check = {}
        self.health_history = []
        
    def initialize_dependencies(self, 
                              model_loader: ColdModelLoader,
                              face_cache_manager: FaceCacheManager,
                              gpu_resource_manager: GPUResourceManager):
        """Initialize health checker with service dependencies"""
        self.model_loader = model_loader
        self.face_cache_manager = face_cache_manager
        self.gpu_resource_manager = gpu_resource_manager
        
    async def check_system_health(self) -> HealthMetrics:
        """Comprehensive system health check"""
        start_time = time.time()
        
        try:
            # Check all components
            components = await self._check_all_components()
            
            # Determine overall status
            overall_status = self._determine_overall_status(components)
            
            # Calculate response time
            response_time = (time.time() - start_time) * 1000
            
            # Create health metrics
            health_metrics = HealthMetrics(
                timestamp=datetime.utcnow(),
                status=overall_status,
                response_time_ms=response_time,
                details={
                    "components": {comp.name: {
                        "status": comp.status,
                        "response_time_ms": comp.response_time_ms,
                        "details": comp.details,
                        "errors": comp.errors
                    } for comp in components},
                    "system_metrics": await self._get_system_metrics(),
                    "service_uptime": self._get_service_uptime()
                }
            )
            
            # Store in history
            self._store_health_history(health_metrics)
            
            return health_metrics
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthMetrics(
                timestamp=datetime.utcnow(),
                status="error",
                response_time_ms=(time.time() - start_time) * 1000,
                details={"error": str(e)}
            )
    
    async def _check_all_components(self) -> List[ComponentHealth]:
        """Check all system components"""
        components = []
        
        # Check components in parallel
        tasks = [
            self._check_models_health(),
            self._check_avatar_cache_health(),
            self._check_gpu_health(),
            self._check_database_health(),
            self._check_storage_health(),
            self._check_network_health()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, ComponentHealth):
                components.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Component check failed: {result}")
                components.append(ComponentHealth(
                    name="unknown",
                    status="error",
                    last_check=datetime.utcnow(),
                    response_time_ms=0,
                    details={},
                    errors=[str(result)]
                ))
        
        return components
    
    async def _check_models_health(self) -> ComponentHealth:
        """Check model loading and inference health"""
        start_time = time.time()
        errors = []
        details = {}
        
        try:
            if not self.model_loader:
                raise Exception("Model loader not initialized")
            
            # Check if models are loaded
            models_status = self.model_loader.get_models_status()
            details["models_loaded"] = models_status
            
            # Test model inference
            if models_status.get("wav2lip_loaded", False):
                inference_time = await self._test_model_inference()
                details["inference_test_ms"] = inference_time
            
            # Check GPU memory usage
            gpu_memory = self._get_gpu_memory_usage()
            details["gpu_memory_usage"] = gpu_memory
            
            # Determine status
            if all(models_status.values()):
                status = "healthy"
            elif any(models_status.values()):
                status = "degraded"
                errors.append("Some models not loaded")
            else:
                status = "error"
                errors.append("No models loaded")
                
        except Exception as e:
            status = "error"
            errors.append(str(e))
            logger.error(f"Model health check failed: {e}")
        
        return ComponentHealth(
            name="models",
            status=status,
            last_check=datetime.utcnow(),
            response_time_ms=(time.time() - start_time) * 1000,
            details=details,
            errors=errors
        )
    
    async def _check_avatar_cache_health(self) -> ComponentHealth:
        """Check avatar cache system health"""
        start_time = time.time()
        errors = []
        details = {}
        
        try:
            if not self.face_cache_manager:
                raise Exception("Avatar cache manager not initialized")
            
            # Check cache system status
            cache_stats = await self.face_cache_manager.get_cache_statistics()
            details.update(cache_stats)
            
            # Test cache operations
            test_result = await self._test_avatar_cache_operations()
            details["cache_test_ms"] = test_result
            
            # Check cache storage health
            storage_health = await self._check_cache_storage()
            details["storage_health"] = storage_health
            
            # Determine status based on cache performance
            if cache_stats.get("hit_rate", 0) > 0.8 and test_result < 50:
                status = "healthy"
            elif cache_stats.get("hit_rate", 0) > 0.5:
                status = "degraded"
                errors.append("Cache performance below optimal")
            else:
                status = "error"
                errors.append("Cache system not functioning properly")
                
        except Exception as e:
            status = "error"
            errors.append(str(e))
            logger.error(f"Avatar cache health check failed: {e}")
        
        return ComponentHealth(
            name="avatar_cache",
            status=status,
            last_check=datetime.utcnow(),
            response_time_ms=(time.time() - start_time) * 1000,
            details=details,
            errors=errors
        )
    
    async def _check_gpu_health(self) -> ComponentHealth:
        """Check GPU availability and performance"""
        start_time = time.time()
        errors = []
        details = {}
        
        try:
            # Check GPU availability
            gpus = GPUtil.getGPUs()
            if not gpus:
                raise Exception("No GPUs detected")
            
            gpu = gpus[0]  # Assuming RTX 4090 is first GPU
            
            details["gpu_name"] = gpu.name
            details["gpu_memory_used_gb"] = round(gpu.memoryUsed / 1024, 2)
            details["gpu_memory_total_gb"] = round(gpu.memoryTotal / 1024, 2)
            details["gpu_memory_utilization"] = round(gpu.memoryUtil * 100, 2)
            details["gpu_utilization"] = round(gpu.load * 100, 2)
            details["gpu_temperature"] = gpu.temperature
            
            # Determine GPU health status
            if gpu.memoryUtil < 0.9 and gpu.temperature < 80:
                status = "healthy"
            elif gpu.memoryUtil < 0.95 and gpu.temperature < 85:
                status = "degraded"
                if gpu.memoryUtil >= 0.9:
                    errors.append("GPU memory usage high")
                if gpu.temperature >= 80:
                    errors.append("GPU temperature elevated")
            else:
                status = "error"
                if gpu.memoryUtil >= 0.95:
                    errors.append("GPU memory critical")
                if gpu.temperature >= 85:
                    errors.append("GPU overheating")
                
        except Exception as e:
            status = "error"
            errors.append(str(e))
            logger.error(f"GPU health check failed: {e}")
        
        return ComponentHealth(
            name="gpu",
            status=status,
            last_check=datetime.utcnow(),
            response_time_ms=(time.time() - start_time) * 1000,
            details=details,
            errors=errors
        )
    
    async def _check_database_health(self) -> ComponentHealth:
        """Check avatar database health"""
        start_time = time.time()
        errors = []
        details = {}
        
        try:
            # Test database connectivity and operations
            # This would integrate with your actual database
            details["database_connected"] = True
            details["avatar_count"] = await self._get_avatar_count()
            details["database_size_mb"] = await self._get_database_size()
            
            status = "healthy"
                
        except Exception as e:
            status = "error"
            errors.append(str(e))
            logger.error(f"Database health check failed: {e}")
        
        return ComponentHealth(
            name="database",
            status=status,
            last_check=datetime.utcnow(),
            response_time_ms=(time.time() - start_time) * 1000,
            details=details,
            errors=errors
        )
    
    async def _check_storage_health(self) -> ComponentHealth:
        """Check storage system health"""
        start_time = time.time()
        errors = []
        details = {}
        
        try:
            # Check disk usage
            disk_usage = psutil.disk_usage('/')
            details["disk_total_gb"] = round(disk_usage.total / (1024**3), 2)
            details["disk_used_gb"] = round(disk_usage.used / (1024**3), 2)
            details["disk_free_gb"] = round(disk_usage.free / (1024**3), 2)
            details["disk_usage_percent"] = round((disk_usage.used / disk_usage.total) * 100, 2)
            
            # Check avatar storage
            avatar_storage_size = await self._get_avatar_storage_size()
            details["avatar_storage_gb"] = round(avatar_storage_size / (1024**3), 2)
            
            # Determine storage health
            disk_usage_percent = (disk_usage.used / disk_usage.total) * 100
            if disk_usage_percent < 80:
                status = "healthy"
            elif disk_usage_percent < 90:
                status = "degraded"
                errors.append("Disk usage high")
            else:
                status = "error"
                errors.append("Disk usage critical")
                
        except Exception as e:
            status = "error"
            errors.append(str(e))
            logger.error(f"Storage health check failed: {e}")
        
        return ComponentHealth(
            name="storage",
            status=status,
            last_check=datetime.utcnow(),
            response_time_ms=(time.time() - start_time) * 1000,
            details=details,
            errors=errors
        )
    
    async def _check_network_health(self) -> ComponentHealth:
        """Check network connectivity health"""
        start_time = time.time()
        errors = []
        details = {}
        
        try:
            # Check network interfaces
            network_stats = psutil.net_io_counters()
            details["bytes_sent"] = network_stats.bytes_sent
            details["bytes_recv"] = network_stats.bytes_recv
            details["packets_sent"] = network_stats.packets_sent
            details["packets_recv"] = network_stats.packets_recv
            
            # Test external connectivity (OpenAI API)
            connectivity_test = await self._test_external_connectivity()
            details["external_connectivity"] = connectivity_test
            
            status = "healthy" if connectivity_test else "degraded"
            if not connectivity_test:
                errors.append("External API connectivity issues")
                
        except Exception as e:
            status = "error"
            errors.append(str(e))
            logger.error(f"Network health check failed: {e}")
        
        return ComponentHealth(
            name="network",
            status=status,
            last_check=datetime.utcnow(),
            response_time_ms=(time.time() - start_time) * 1000,
            details=details,
            errors=errors
        )
    
    def _determine_overall_status(self, components: List[ComponentHealth]) -> str:
        """Determine overall system status from component statuses"""
        if not components:
            return "error"
        
        statuses = [comp.status for comp in components]
        
        if "error" in statuses:
            return "error"
        elif "degraded" in statuses:
            return "degraded"
        else:
            return "healthy"
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "active_sessions": await self._get_active_sessions_count(),
            "requests_per_minute": await self._get_request_rate(),
            "average_response_time": await self._get_average_response_time()
        }
    
    def _get_service_uptime(self) -> float:
        """Get service uptime in hours"""
        # This would track from service start time
        return 0.0  # Placeholder
    
    async def _test_model_inference(self) -> float:
        """Test model inference performance"""
        # Placeholder for model inference test
        return 150.0  # ms
    
    async def _test_avatar_cache_operations(self) -> float:
        """Test avatar cache read/write operations"""
        # Placeholder for cache operations test
        return 25.0  # ms
    
    async def _test_external_connectivity(self) -> bool:
        """Test external API connectivity"""
        # Placeholder for connectivity test
        return True
    
    def _get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get GPU memory usage details"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    "used_gb": round(gpu.memoryUsed / 1024, 2),
                    "total_gb": round(gpu.memoryTotal / 1024, 2),
                    "utilization_percent": round(gpu.memoryUtil * 100, 2)
                }
        except:
            pass
        return {"error": "GPU info unavailable"}
    
    async def _check_cache_storage(self) -> Dict[str, Any]:
        """Check avatar cache storage health"""
        return {"status": "healthy", "size_mb": 0}
    
    async def _get_avatar_count(self) -> int:
        """Get total number of registered avatars"""
        return 0  # Placeholder
    
    async def _get_database_size(self) -> float:
        """Get database size in MB"""
        return 0.0  # Placeholder
    
    async def _get_avatar_storage_size(self) -> int:
        """Get avatar storage size in bytes"""
        return 0  # Placeholder
    
    async def _get_active_sessions_count(self) -> int:
        """Get number of active user sessions"""
        return 0  # Placeholder
    
    async def _get_request_rate(self) -> float:
        """Get requests per minute"""
        return 0.0  # Placeholder
    
    async def _get_average_response_time(self) -> float:
        """Get average response time in ms"""
        return 0.0  # Placeholder
    
    def _store_health_history(self, metrics: HealthMetrics):
        """Store health metrics in history"""
        self.health_history.append(metrics)
        # Keep only last 100 entries
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]


# Global health checker instance
health_checker = HealthChecker()


# FastAPI router
router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_class=JSONResponse)
async def get_health_status():
    """Get comprehensive system health status"""
    try:
        health_metrics = await health_checker.check_system_health()
        
        response_data = {
            "status": health_metrics.status,
            "timestamp": health_metrics.timestamp.isoformat(),
            "response_time_ms": health_metrics.response_time_ms,
            "details": health_metrics.details
        }
        
        status_code = (
            status.HTTP_200_OK if health_metrics.status == "healthy"
            else status.HTTP_206_PARTIAL_CONTENT if health_metrics.status == "degraded"
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        
        return JSONResponse(content=response_data, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Health check endpoint failed: {e}")
        return JSONResponse(
            content={
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


@router.get("/models", response_class=JSONResponse)
async def get_models_health():
    """Get model-specific health status"""
    try:
        model_health = await health_checker._check_models_health()
        
        return JSONResponse(
            content={
                "component": model_health.name,
                "status": model_health.status,
                "timestamp": model_health.last_check.isoformat(),
                "response_time_ms": model_health.response_time_ms,
                "details": model_health.details,
                "errors": model_health.errors
            },
            status_code=status.HTTP_200_OK if model_health.status == "healthy" else status.HTTP_206_PARTIAL_CONTENT
        )
        
    except Exception as e:
        logger.error(f"Model health check failed: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


@router.get("/avatar-cache", response_class=JSONResponse)
async def get_avatar_cache_health():
    """Get avatar cache system health status"""
    try:
        cache_health = await health_checker._check_avatar_cache_health()
        
        return JSONResponse(
            content={
                "component": cache_health.name,
                "status": cache_health.status,
                "timestamp": cache_health.last_check.isoformat(),
                "response_time_ms": cache_health.response_time_ms,
                "details": cache_health.details,
                "errors": cache_health.errors
            },
            status_code=status.HTTP_200_OK if cache_health.status == "healthy" else status.HTTP_206_PARTIAL_CONTENT
        )
        
    except Exception as e:
        logger.error(f"Avatar cache health check failed: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


@router.get("/gpu", response_class=JSONResponse)
async def get_gpu_health():
    """Get GPU health status"""
    try:
        gpu_health = await health_checker._check_gpu_health()
        
        return JSONResponse(
            content={
                "component": gpu_health.name,
                "status": gpu_health.status,
                "timestamp": gpu_health.last_check.isoformat(),
                "response_time_ms": gpu_health.response_time_ms,
                "details": gpu_health.details,
                "errors": gpu_health.errors
            },
            status_code=status.HTTP_200_OK if gpu_health.status == "healthy" else status.HTTP_206_PARTIAL_CONTENT
        )
        
    except Exception as e:
        logger.error(f"GPU health check failed: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


@router.get("/ready", response_class=JSONResponse)
async def get_readiness_status():
    """Get service readiness status for load balancer"""
    try:
        # Quick readiness check
        models_ready = health_checker.model_loader and health_checker.model_loader.are_models_loaded()
        cache_ready = health_checker.face_cache_manager and health_checker.face_cache_manager.is_cache_ready()
        
        ready = models_ready and cache_ready
        
        return JSONResponse(
            content={
                "ready": ready,
                "models_loaded": models_ready,
                "cache_ready": cache_ready,
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            content={"ready": False, "error": str(e)},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


@router.get("/history", response_class=JSONResponse)
async def get_health_history(limit: int = 10):
    """Get recent health check history"""
    try:
        recent_history = health_checker.health_history[-limit:]
        
        return JSONResponse(
            content={
                "history": [
                    {
                        "timestamp": metrics.timestamp.isoformat(),
                        "status": metrics.status,
                        "response_time_ms": metrics.response_time_ms
                    }
                    for metrics in recent_history
                ],
                "count": len(recent_history)
            },
            status_code=status.HTTP_200_OK
        )
        
    except Exception as e:
        logger.error(f"Health history retrieval failed: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


def initialize_health_checker(model_loader, face_cache_manager, gpu_resource_manager):
    """Initialize health checker with service dependencies"""
    health_checker.initialize_dependencies(
        model_loader=model_loader,
        face_cache_manager=face_cache_manager,
        gpu_resource_manager=gpu_resource_manager
    ) 