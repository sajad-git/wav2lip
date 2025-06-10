"""
Error models for Avatar Streaming Service.
Defines error response structures and exception handling.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories."""
    
    VALIDATION = "validation"
    PROCESSING = "processing"
    SYSTEM = "system"
    AVATAR = "avatar"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    GPU = "gpu"
    MODEL = "model"


class ProcessingError(BaseModel):
    """Main error response model."""
    
    error_code: str = Field(..., description="Specific error code")
    error_message: str = Field(..., description="Human-readable error message")
    error_category: ErrorCategory = Field(..., description="Error category")
    severity: ErrorSeverity = Field(..., description="Error severity level")
    avatar_related: bool = Field(default=False, description="Whether error is avatar-related")
    recovery_suggestion: str = Field(..., description="Suggested recovery action")
    retry_possible: bool = Field(..., description="Whether retry is recommended")
    fallback_available: bool = Field(..., description="Fallback options available")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error occurrence time")
    request_id: Optional[str] = Field(default=None, description="Request identifier")
    client_id: Optional[str] = Field(default=None, description="Client identifier")
    avatar_id: Optional[str] = Field(default=None, description="Related avatar ID")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")


class ValidationError(ProcessingError):
    """Validation-specific error model."""
    
    field_name: Optional[str] = Field(default=None, description="Field that failed validation")
    validation_rule: Optional[str] = Field(default=None, description="Validation rule that failed")
    provided_value: Optional[str] = Field(default=None, description="Value that was provided")
    expected_format: Optional[str] = Field(default=None, description="Expected format or range")
    
    def __init__(self, **data):
        data.setdefault('error_category', ErrorCategory.VALIDATION)
        data.setdefault('severity', ErrorSeverity.LOW)
        super().__init__(**data)


class AvatarError(ProcessingError):
    """Avatar-specific error model."""
    
    avatar_operation: Optional[str] = Field(default=None, description="Avatar operation that failed")
    face_detection_error: bool = Field(default=False, description="Whether face detection failed")
    cache_error: bool = Field(default=False, description="Whether cache operation failed")
    quality_issue: bool = Field(default=False, description="Whether avatar quality is insufficient")
    
    def __init__(self, **data):
        data.setdefault('error_category', ErrorCategory.AVATAR)
        data.setdefault('avatar_related', True)
        super().__init__(**data)


class GPUError(ProcessingError):
    """GPU-specific error model."""
    
    gpu_memory_issue: bool = Field(default=False, description="GPU memory problem")
    cuda_error: bool = Field(default=False, description="CUDA-related error")
    driver_issue: bool = Field(default=False, description="GPU driver problem")
    model_loading_error: bool = Field(default=False, description="Model loading failure")
    
    def __init__(self, **data):
        data.setdefault('error_category', ErrorCategory.GPU)
        data.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(**data)


class ModelError(ProcessingError):
    """Model-specific error model."""
    
    model_name: Optional[str] = Field(default=None, description="Model that failed")
    model_loading_failed: bool = Field(default=False, description="Model loading failure")
    inference_failed: bool = Field(default=False, description="Model inference failure")
    model_not_loaded: bool = Field(default=False, description="Model not pre-loaded")
    
    def __init__(self, **data):
        data.setdefault('error_category', ErrorCategory.MODEL)
        data.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(**data)


class SystemError(ProcessingError):
    """System-level error model."""
    
    memory_issue: bool = Field(default=False, description="System memory problem")
    disk_space_issue: bool = Field(default=False, description="Disk space problem")
    permission_issue: bool = Field(default=False, description="File permission problem")
    dependency_missing: bool = Field(default=False, description="Missing dependency")
    
    def __init__(self, **data):
        data.setdefault('error_category', ErrorCategory.SYSTEM)
        data.setdefault('severity', ErrorSeverity.CRITICAL)
        super().__init__(**data)


class NetworkError(ProcessingError):
    """Network-related error model."""
    
    connection_timeout: bool = Field(default=False, description="Connection timeout")
    api_unavailable: bool = Field(default=False, description="External API unavailable")
    rate_limited: bool = Field(default=False, description="Rate limit exceeded")
    websocket_error: bool = Field(default=False, description="WebSocket connection error")
    
    def __init__(self, **data):
        data.setdefault('error_category', ErrorCategory.NETWORK)
        data.setdefault('severity', ErrorSeverity.MEDIUM)
        super().__init__(**data)


class ErrorContext(BaseModel):
    """Error context information."""
    
    service_name: str = Field(..., description="Source service identifier")
    operation_name: str = Field(..., description="Failed operation")
    client_id: Optional[str] = Field(default=None, description="Affected client")
    avatar_id: Optional[str] = Field(default=None, description="Related avatar ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error occurrence time")
    stack_trace: Optional[str] = Field(default=None, description="Full error stack trace")
    request_data: Optional[Dict[str, Any]] = Field(default=None, description="Request data at time of error")
    system_state: Optional[Dict[str, Any]] = Field(default=None, description="System state at time of error")


class ErrorResponse(BaseModel):
    """Complete error response structure."""
    
    success: bool = Field(default=False, description="Request success status")
    error: ProcessingError = Field(..., description="Error details")
    context: Optional[ErrorContext] = Field(default=None, description="Error context")
    fallback_data: Optional[Dict[str, Any]] = Field(default=None, description="Fallback response data")
    retry_after: Optional[int] = Field(default=None, description="Retry delay in seconds")


class SecurityScanResult(BaseModel):
    """Security scan result model."""
    
    is_safe: bool = Field(..., description="Whether content is safe")
    threat_level: ErrorSeverity = Field(..., description="Threat severity level")
    threats_detected: List[str] = Field(default_factory=list, description="Detected threats")
    scan_details: Dict[str, Any] = Field(default_factory=dict, description="Detailed scan results")
    action_required: bool = Field(..., description="Whether action is required")
    recommendations: List[str] = Field(default_factory=list, description="Security recommendations")


class ValidationResult(BaseModel):
    """Input validation result model."""
    
    is_valid: bool = Field(..., description="Validation success status")
    error_messages: List[str] = Field(default_factory=list, description="Validation error details")
    warnings: List[str] = Field(default_factory=list, description="Non-critical warnings")
    sanitized_data: Optional[Any] = Field(default=None, description="Cleaned input data")
    security_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Security assessment score")


class RecoveryResult(BaseModel):
    """Error recovery attempt result."""
    
    recovery_successful: bool = Field(..., description="Recovery success status")
    recovery_method: str = Field(..., description="Recovery method used")
    recovery_time: float = Field(..., description="Time taken for recovery")
    service_status: str = Field(..., description="Service status after recovery")
    fallback_activated: bool = Field(..., description="Whether fallback was activated")
    manual_intervention_required: bool = Field(..., description="Whether manual intervention is needed")


class FallbackResponse(BaseModel):
    """Fallback response model."""
    
    fallback_type: str = Field(..., description="Type of fallback used")
    reduced_functionality: List[str] = Field(default_factory=list, description="Features not available")
    estimated_recovery_time: Optional[int] = Field(default=None, description="Estimated recovery time in seconds")
    fallback_data: Optional[Dict[str, Any]] = Field(default=None, description="Fallback response data")
    user_message: str = Field(..., description="User-friendly fallback message")


# Common error code definitions
class ErrorCodes:
    """Standard error codes."""
    
    # Validation errors (1000-1999)
    INVALID_INPUT = "1001"
    MISSING_REQUIRED_FIELD = "1002"
    INVALID_FORMAT = "1003"
    VALUE_OUT_OF_RANGE = "1004"
    INVALID_AVATAR_ID = "1005"
    
    # Avatar errors (2000-2999)
    AVATAR_NOT_FOUND = "2001"
    AVATAR_REGISTRATION_FAILED = "2002"
    FACE_DETECTION_FAILED = "2003"
    AVATAR_QUALITY_INSUFFICIENT = "2004"
    AVATAR_CACHE_ERROR = "2005"
    AVATAR_FILE_CORRUPTED = "2006"
    
    # Processing errors (3000-3999)
    PROCESSING_TIMEOUT = "3001"
    CHUNK_PROCESSING_FAILED = "3002"
    AUDIO_PROCESSING_FAILED = "3003"
    VIDEO_GENERATION_FAILED = "3004"
    SYNCHRONIZATION_FAILED = "3005"
    
    # System errors (4000-4999)
    GPU_NOT_AVAILABLE = "4001"
    INSUFFICIENT_MEMORY = "4002"
    MODEL_NOT_LOADED = "4003"
    SERVICE_UNAVAILABLE = "4004"
    DISK_SPACE_INSUFFICIENT = "4005"
    
    # Network errors (5000-5999)
    CONNECTION_TIMEOUT = "5001"
    API_RATE_LIMITED = "5002"
    EXTERNAL_SERVICE_ERROR = "5003"
    WEBSOCKET_DISCONNECTED = "5004"
    
    # Authentication errors (6000-6999)
    UNAUTHORIZED = "6001"
    INVALID_SESSION = "6002"
    RATE_LIMIT_EXCEEDED = "6003" 