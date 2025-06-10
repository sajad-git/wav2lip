"""
API response structures and status information
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AvatarProcessingResponse(BaseModel):
    """Avatar processing response model"""
    request_id: str = Field(..., description="Unique request identifier")
    status: str = Field(..., description="Processing status (processing, streaming, completed, error)")
    total_chunks: int = Field(..., description="Expected number of chunks")
    estimated_duration: float = Field(..., description="Estimated total processing time")
    first_chunk_eta: float = Field(..., description="Time to first chunk delivery")
    avatar_id: str = Field(..., description="Avatar used for processing")
    face_cache_hit: bool = Field(..., description="Whether face data was cached")
    websocket_url: str = Field(..., description="WebSocket endpoint for streaming")
    session_id: str = Field(..., description="Client session identifier")


class ProcessingQualityMetrics(BaseModel):
    """Processing quality metrics"""
    video_quality: float = Field(..., description="Video output quality score")
    audio_sync_accuracy: float = Field(..., description="Audio-video synchronization accuracy")
    lip_sync_quality: float = Field(..., description="Lip synchronization quality")
    frame_consistency: float = Field(..., description="Frame-to-frame consistency")
    overall_quality: float = Field(..., description="Overall processing quality")


class StreamingStatusResponse(BaseModel):
    """Streaming status response model"""
    chunk_id: str = Field(..., description="Current chunk identifier")
    chunks_completed: int = Field(..., description="Number of completed chunks")
    chunks_remaining: int = Field(..., description="Remaining chunks to process")
    current_latency: float = Field(..., description="Current processing latency")
    average_latency: float = Field(..., description="Average processing time per chunk")
    avatar_processing_time: float = Field(..., description="Time saved by cached face data")
    quality_metrics: ProcessingQualityMetrics = Field(..., description="Quality information")
    next_chunk_eta: float = Field(..., description="Estimated time to next chunk")


class ServiceHealthResponse(BaseModel):
    """Service health response model"""
    service_status: str = Field(..., description="Overall service health (healthy, degraded, error)")
    models_loaded: bool = Field(..., description="Model loading status")
    avatar_cache_loaded: bool = Field(..., description="Avatar cache status")
    gpu_available: bool = Field(..., description="GPU accessibility")
    active_sessions: int = Field(..., description="Number of active user sessions")
    registered_avatars_count: int = Field(..., description="Number of registered avatars")
    average_response_time: float = Field(..., description="Recent average response time")
    last_health_check: datetime = Field(..., description="Last health check timestamp")
    available_capacity: int = Field(..., description="Remaining user capacity")


class ProcessingError(BaseModel):
    """Processing error model"""
    error_code: str = Field(..., description="Specific error code")
    error_message: str = Field(..., description="Human-readable error message")
    error_category: str = Field(..., description="Error category (validation, processing, system, avatar)")
    avatar_related: bool = Field(default=False, description="Whether error is avatar-related")
    recovery_suggestion: str = Field(..., description="Suggested recovery action")
    retry_possible: bool = Field(..., description="Whether retry is recommended")
    fallback_available: bool = Field(..., description="Fallback options available")


class ErrorResponse(BaseModel):
    """General error response model"""
    success: bool = Field(default=False, description="Operation success status")
    error: ProcessingError = Field(..., description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Original request identifier")


class SuccessResponse(BaseModel):
    """General success response model"""
    success: bool = Field(default=True, description="Operation success status")
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class PerformanceMetrics(BaseModel):
    """Performance metrics model"""
    avg_processing_time: float = Field(..., description="Average processing time per chunk")
    avg_first_chunk_latency: float = Field(..., description="Average first chunk latency")
    cache_hit_rate: float = Field(..., description="Avatar cache hit rate")
    gpu_utilization: float = Field(..., description="GPU utilization percentage")
    memory_usage: float = Field(..., description="Memory usage percentage")
    concurrent_users: int = Field(..., description="Current concurrent users")
    throughput_chunks_per_second: float = Field(..., description="Processing throughput")


class SystemStatusResponse(BaseModel):
    """System status response model"""
    system_health: str = Field(..., description="Overall system health")
    gpu_status: str = Field(..., description="GPU status")
    model_status: str = Field(..., description="Model loading status")
    avatar_cache_status: str = Field(..., description="Avatar cache status")
    storage_status: str = Field(..., description="Storage system status")
    performance_metrics: PerformanceMetrics = Field(..., description="Current performance metrics")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    last_restart: Optional[datetime] = Field(None, description="Last service restart")


class ReadinessResponse(BaseModel):
    """Service readiness response model"""
    ready: bool = Field(..., description="Service readiness status")
    models_loaded: bool = Field(..., description="Models loaded and ready")
    avatar_cache_ready: bool = Field(..., description="Avatar cache ready")
    gpu_ready: bool = Field(..., description="GPU ready for processing")
    dependencies_ready: bool = Field(..., description="External dependencies ready")
    estimated_ready_time: Optional[float] = Field(None, description="Time until ready (if not ready)")


class MetricsResponse(BaseModel):
    """Metrics response model"""
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")
    performance: PerformanceMetrics = Field(..., description="Performance metrics")
    resource_usage: Dict[str, float] = Field(..., description="Resource usage metrics")
    error_rates: Dict[str, float] = Field(..., description="Error rate metrics")
    avatar_cache_metrics: Dict[str, Any] = Field(..., description="Avatar cache metrics")
    model_metrics: Dict[str, Any] = Field(..., description="Model performance metrics")


class ValidationResult(BaseModel):
    """Input validation result model"""
    is_valid: bool = Field(..., description="Validation success status")
    error_messages: List[str] = Field(default_factory=list, description="Validation error details")
    warnings: List[str] = Field(default_factory=list, description="Non-critical warnings")
    sanitized_data: Optional[Any] = Field(None, description="Cleaned input data")
    security_score: float = Field(default=1.0, description="Security assessment score")


class SecurityScanResult(BaseModel):
    """Security scan result model"""
    is_safe: bool = Field(..., description="Security scan result")
    threat_level: str = Field(..., description="Threat level assessment")
    detected_threats: List[str] = Field(default_factory=list, description="Detected security threats")
    scan_time: float = Field(..., description="Scan processing time")
    recommendations: List[str] = Field(default_factory=list, description="Security recommendations")


class CleanupReport(BaseModel):
    """Cache cleanup operation report"""
    files_removed: int = Field(..., description="Number of files removed")
    space_freed: int = Field(..., description="Disk space freed in bytes")
    cleanup_time: float = Field(..., description="Cleanup operation duration")
    errors_encountered: List[str] = Field(default_factory=list, description="Cleanup errors")
    next_cleanup_scheduled: Optional[datetime] = Field(None, description="Next scheduled cleanup") 