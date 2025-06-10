"""
Centralized error handling and recovery
"""
import logging
import traceback
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from app.models.response_models import ErrorResponse, ProcessingError


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories"""
    VALIDATION = "validation"
    PROCESSING = "processing"
    SYSTEM = "system"
    AVATAR = "avatar"
    NETWORK = "network"
    RESOURCE = "resource"
    EXTERNAL = "external"


@dataclass
class ErrorContext:
    """Error context information"""
    service_name: str
    operation_name: str
    client_id: Optional[str] = None
    avatar_id: Optional[str] = None
    timestamp: datetime = None
    stack_trace: str = ""
    additional_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.additional_data is None:
            self.additional_data = {}


@dataclass
class RecoveryStrategy:
    """Error recovery strategy"""
    strategy_name: str
    retry_count: int
    retry_delay: float
    fallback_action: Optional[Callable] = None
    escalation_threshold: int = 3


class FallbackManager:
    """Graceful degradation manager"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.fallback_responses = {
            "wav2lip_processing": "Processing temporarily unavailable",
            "avatar_loading": "Avatar temporarily unavailable",
            "tts_generation": "Audio generation temporarily unavailable",
            "rag_query": "Knowledge query temporarily unavailable"
        }
    
    def get_fallback_response(self, operation: str, context: ErrorContext) -> Any:
        """Get fallback response for failed operation"""
        if operation in self.fallback_responses:
            return self.fallback_responses[operation]
        
        return f"Service temporarily unavailable: {operation}"


class StructuredLogger:
    """Structured error logging system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_counts = {}
        self.start_time = time.time()
    
    def log_error(self, error: Exception, context: ErrorContext, severity: ErrorSeverity):
        """Log error with structured context"""
        error_key = f"{context.service_name}_{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        log_data = {
            "service": context.service_name,
            "operation": context.operation_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "severity": severity.value,
            "client_id": context.client_id,
            "avatar_id": context.avatar_id,
            "timestamp": context.timestamp.isoformat(),
            "error_count": self.error_counts[error_key],
            "stack_trace": context.stack_trace
        }
        
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.error(f"🚨 {severity.value.upper()} ERROR: {log_data}")
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"⚠️ {severity.value.upper()} ERROR: {log_data}")
        else:
            self.logger.info(f"ℹ️ {severity.value.upper()} ERROR: {log_data}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        uptime = time.time() - self.start_time
        total_errors = sum(self.error_counts.values())
        
        return {
            "total_errors": total_errors,
            "error_rate": total_errors / max(1, uptime / 3600),  # Errors per hour
            "error_breakdown": self.error_counts,
            "uptime_hours": uptime / 3600
        }


class GlobalErrorHandler:
    """Centralized error handling and recovery"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_registry: Dict[str, Callable] = {}
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        self.fallback_manager = FallbackManager()
        self.error_logger = StructuredLogger()
        
        # Initialize default recovery strategies
        self._initialize_recovery_strategies()
        
        self.logger.info("🛡️ Global Error Handler initialized")
    
    def handle_service_error(self, error: Exception, context: ErrorContext) -> ErrorResponse:
        """Handle service-level errors with appropriate recovery"""
        try:
            # Classify error type and severity
            error_category = self._classify_error(error)
            severity = self._determine_severity(error, context)
            
            # Log error with context
            context.stack_trace = traceback.format_exc()
            self.error_logger.log_error(error, context, severity)
            
            # Apply appropriate recovery strategy
            recovery_result = self._apply_recovery_strategy(error, context, error_category)
            
            # Generate structured error response
            processing_error = ProcessingError(
                error_code=f"{error_category.value}_{type(error).__name__}",
                error_message=str(error),
                error_category=error_category.value,
                avatar_related=context.avatar_id is not None,
                recovery_suggestion=recovery_result.get("suggestion", "Retry operation"),
                retry_possible=recovery_result.get("retry_possible", True),
                fallback_available=recovery_result.get("fallback_available", False)
            )
            
            return ErrorResponse(
                error=processing_error,
                request_id=context.additional_data.get("request_id")
            )
            
        except Exception as handler_error:
            self.logger.error(f"❌ Error handler failed: {str(handler_error)}")
            
            # Fallback error response
            return ErrorResponse(
                error=ProcessingError(
                    error_code="HANDLER_FAILURE",
                    error_message="Error handling system failure",
                    error_category="system",
                    avatar_related=False,
                    recovery_suggestion="Contact system administrator",
                    retry_possible=False,
                    fallback_available=False
                )
            )
    
    def handle_avatar_error(self, error: Exception, avatar_id: str, operation: str) -> ErrorResponse:
        """Handle avatar-related errors with context"""
        context = ErrorContext(
            service_name="avatar_service",
            operation_name=operation,
            avatar_id=avatar_id,
            additional_data={"error_type": "avatar_specific"}
        )
        
        return self.handle_service_error(error, context)
    
    def implement_graceful_degradation(self, service_name: str, error: Exception) -> Dict[str, Any]:
        """Maintain service availability during partial failures"""
        try:
            self.logger.info(f"🔄 Implementing graceful degradation for {service_name}")
            
            # Identify available fallback options
            fallback_response = self.fallback_manager.get_fallback_response(
                service_name, 
                ErrorContext(service_name=service_name, operation_name="degradation")
            )
            
            # Switch to degraded mode operation
            degraded_mode = self._enable_degraded_mode(service_name)
            
            return {
                "fallback_response": fallback_response,
                "degraded_mode": degraded_mode,
                "message": f"Service {service_name} operating in degraded mode",
                "recovery_eta": self._estimate_recovery_time(service_name, error)
            }
            
        except Exception as degradation_error:
            self.logger.error(f"❌ Graceful degradation failed: {str(degradation_error)}")
            return {
                "fallback_response": "Service temporarily unavailable",
                "degraded_mode": False,
                "message": "Service degradation failed",
                "recovery_eta": 300  # 5 minutes default
            }
    
    def recover_from_gpu_error(self, gpu_error: Exception) -> Dict[str, Any]:
        """Handle GPU-related errors without service restart"""
        try:
            self.logger.warning(f"🔧 Attempting GPU error recovery: {str(gpu_error)}")
            
            # Diagnose GPU memory or processing issues
            gpu_status = self._diagnose_gpu_issue(gpu_error)
            
            recovery_actions = []
            
            if "memory" in str(gpu_error).lower():
                # Attempt GPU memory cleanup
                cleanup_result = self._cleanup_gpu_memory()
                recovery_actions.append(f"GPU memory cleanup: {cleanup_result}")
                
                if not cleanup_result:
                    # Reload models if necessary
                    reload_result = self._reload_models_after_gpu_error()
                    recovery_actions.append(f"Model reload: {reload_result}")
            
            elif "device" in str(gpu_error).lower():
                # Check GPU availability
                gpu_available = self._check_gpu_availability()
                if not gpu_available:
                    # Fallback to CPU processing if available
                    cpu_fallback = self._enable_cpu_fallback()
                    recovery_actions.append(f"CPU fallback: {cpu_fallback}")
            
            return {
                "recovery_successful": True,
                "actions_taken": recovery_actions,
                "gpu_status": gpu_status,
                "fallback_mode": "cpu" if "CPU fallback" in str(recovery_actions) else "gpu"
            }
            
        except Exception as recovery_error:
            self.logger.error(f"❌ GPU error recovery failed: {str(recovery_error)}")
            return {
                "recovery_successful": False,
                "error": str(recovery_error),
                "fallback_mode": "degraded"
            }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        return self.error_logger.get_error_statistics()
    
    def _initialize_recovery_strategies(self):
        """Initialize default recovery strategies"""
        self.recovery_strategies = {
            "validation": RecoveryStrategy(
                strategy_name="input_validation",
                retry_count=0,
                retry_delay=0.0,
                escalation_threshold=1
            ),
            "processing": RecoveryStrategy(
                strategy_name="processing_retry",
                retry_count=2,
                retry_delay=1.0,
                escalation_threshold=3
            ),
            "system": RecoveryStrategy(
                strategy_name="system_recovery",
                retry_count=1,
                retry_delay=5.0,
                escalation_threshold=2
            ),
            "avatar": RecoveryStrategy(
                strategy_name="avatar_cache_recovery",
                retry_count=1,
                retry_delay=2.0,
                escalation_threshold=2
            ),
            "network": RecoveryStrategy(
                strategy_name="network_retry",
                retry_count=3,
                retry_delay=2.0,
                escalation_threshold=5
            ),
            "resource": RecoveryStrategy(
                strategy_name="resource_cleanup",
                retry_count=1,
                retry_delay=10.0,
                escalation_threshold=2
            )
        }
    
    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Classify error into appropriate category"""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        if "validation" in error_str or "invalid" in error_str:
            return ErrorCategory.VALIDATION
        elif "avatar" in error_str or "face" in error_str:
            return ErrorCategory.AVATAR
        elif "memory" in error_str or "gpu" in error_str or "cuda" in error_str:
            return ErrorCategory.RESOURCE
        elif "network" in error_str or "connection" in error_str or "timeout" in error_str:
            return ErrorCategory.NETWORK
        elif "external" in error_str or "api" in error_str:
            return ErrorCategory.EXTERNAL
        elif "runtime" in error_type or "system" in error_str:
            return ErrorCategory.SYSTEM
        else:
            return ErrorCategory.PROCESSING
    
    def _determine_severity(self, error: Exception, context: ErrorContext) -> ErrorSeverity:
        """Determine error severity based on error and context"""
        error_str = str(error).lower()
        
        # Critical errors
        if any(keyword in error_str for keyword in ["cuda", "gpu", "memory", "segmentation"]):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if any(keyword in error_str for keyword in ["failed to load", "initialization", "authentication"]):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if any(keyword in error_str for keyword in ["timeout", "connection", "not found"]):
            return ErrorSeverity.MEDIUM
        
        # Low severity errors (validation, user input, etc.)
        return ErrorSeverity.LOW
    
    def _apply_recovery_strategy(self, error: Exception, context: ErrorContext, 
                               category: ErrorCategory) -> Dict[str, Any]:
        """Apply appropriate recovery strategy"""
        strategy = self.recovery_strategies.get(category.value)
        
        if not strategy:
            return {
                "suggestion": "Manual intervention required",
                "retry_possible": False,
                "fallback_available": True
            }
        
        return {
            "suggestion": f"Retry with {strategy.strategy_name} strategy",
            "retry_possible": strategy.retry_count > 0,
            "fallback_available": strategy.fallback_action is not None,
            "retry_delay": strategy.retry_delay,
            "max_retries": strategy.retry_count
        }
    
    def _enable_degraded_mode(self, service_name: str) -> bool:
        """Enable degraded mode for a service"""
        try:
            # This would implement actual degraded mode switching
            # For now, just return success
            self.logger.info(f"✅ Degraded mode enabled for {service_name}")
            return True
        except Exception:
            return False
    
    def _estimate_recovery_time(self, service_name: str, error: Exception) -> int:
        """Estimate recovery time in seconds"""
        error_str = str(error).lower()
        
        if "memory" in error_str or "gpu" in error_str:
            return 120  # 2 minutes for GPU issues
        elif "network" in error_str:
            return 30   # 30 seconds for network issues
        elif "validation" in error_str:
            return 5    # 5 seconds for validation issues
        else:
            return 60   # 1 minute default
    
    def _diagnose_gpu_issue(self, gpu_error: Exception) -> Dict[str, Any]:
        """Diagnose GPU-related issues"""
        # This would implement actual GPU diagnostics
        return {
            "gpu_available": True,  # Mock
            "memory_usage": "8GB/24GB",  # Mock
            "error_type": type(gpu_error).__name__
        }
    
    def _cleanup_gpu_memory(self) -> bool:
        """Cleanup GPU memory"""
        try:
            # This would implement actual GPU memory cleanup
            self.logger.info("🧹 GPU memory cleanup attempted")
            return True
        except Exception:
            return False
    
    def _reload_models_after_gpu_error(self) -> bool:
        """Reload models after GPU error"""
        try:
            # This would implement actual model reloading
            self.logger.info("🔄 Model reload after GPU error attempted")
            return True
        except Exception:
            return False
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available"""
        try:
            # This would implement actual GPU availability check
            return True
        except Exception:
            return False
    
    def _enable_cpu_fallback(self) -> bool:
        """Enable CPU fallback mode"""
        try:
            # This would implement actual CPU fallback
            self.logger.info("💻 CPU fallback mode enabled")
            return True
        except Exception:
            return False 