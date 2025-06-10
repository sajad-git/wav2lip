"""
Model Performance Monitor for Avatar Streaming Service
Monitors model loading status, inference performance, GPU utilization, and model health.
"""

import time
import GPUtil
import threading
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging
import asyncio
import json

from app.core.model_loader import ColdModelLoader
from monitoring.metrics_collector import MetricsCollector, MetricType

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceMetrics:
    """Model performance metrics data structure"""
    model_name: str
    inference_count: int = 0
    total_inference_time: float = 0.0
    average_inference_time: float = 0.0
    min_inference_time: float = float('inf')
    max_inference_time: float = 0.0
    error_count: int = 0
    success_rate: float = 1.0
    gpu_memory_usage: float = 0.0
    last_inference_time: Optional[datetime] = None
    inference_times_history: deque = field(default_factory=lambda: deque(maxlen=1000))


@dataclass
class ModelLoadingMetrics:
    """Model loading and initialization metrics"""
    model_name: str
    loading_time: float
    memory_footprint: float
    loading_timestamp: datetime
    is_loaded: bool
    load_errors: List[str] = field(default_factory=list)
    warm_up_time: float = 0.0
    initialization_success: bool = True


@dataclass
class ModelHealthStatus:
    """Model health assessment"""
    model_name: str
    status: str  # "healthy", "degraded", "error", "not_loaded"
    health_score: float  # 0.0 to 1.0
    last_check: datetime
    performance_trend: str  # "improving", "stable", "degrading"
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class ModelMonitor:
    """Comprehensive model performance monitoring system"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.model_loader: Optional[ColdModelLoader] = None
        
        # Performance tracking
        self.model_metrics: Dict[str, ModelPerformanceMetrics] = {}
        self.loading_metrics: Dict[str, ModelLoadingMetrics] = {}
        self.health_history: deque = deque(maxlen=1440)  # 24 hours of health checks
        
        # Monitoring configuration
        self.monitoring_active = False
        self.health_check_interval = 300  # 5 minutes
        self.performance_thresholds = {
            "max_inference_time": 500.0,  # ms
            "min_success_rate": 0.95,
            "max_gpu_memory": 20.0,  # GB
            "max_error_rate": 0.05
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Baseline performance
        self.performance_baselines = {}
        
    def initialize(self, model_loader: ColdModelLoader):
        """Initialize monitor with model loader"""
        self.model_loader = model_loader
        self._establish_performance_baselines()
        logger.info("Model monitor initialized")
    
    def start_monitoring(self):
        """Start continuous model monitoring"""
        self.monitoring_active = True
        self._start_health_monitoring()
        logger.info("Model monitoring started")
    
    def stop_monitoring(self):
        """Stop model monitoring"""
        self.monitoring_active = False
        logger.info("Model monitoring stopped")
    
    def record_model_loading(self, model_name: str, loading_time: float, 
                           memory_footprint: float, success: bool, 
                           errors: List[str] = None, warm_up_time: float = 0.0):
        """Record model loading metrics"""
        with self.lock:
            loading_metrics = ModelLoadingMetrics(
                model_name=model_name,
                loading_time=loading_time,
                memory_footprint=memory_footprint,
                loading_timestamp=datetime.utcnow(),
                is_loaded=success,
                load_errors=errors or [],
                warm_up_time=warm_up_time,
                initialization_success=success
            )
            
            self.loading_metrics[model_name] = loading_metrics
            
            # Record in metrics collector
            tags = {"model": model_name, "success": str(success)}
            self.metrics_collector.record_metric("model_loading_time", loading_time, tags, MetricType.TIMING)
            self.metrics_collector.record_metric("model_memory_footprint", memory_footprint, tags, MetricType.GAUGE)
            self.metrics_collector.record_metric("model_warmup_time", warm_up_time, tags, MetricType.TIMING)
            
            if not success:
                self.metrics_collector.record_metric("model_loading_errors", 1, tags, MetricType.COUNTER)
            
            logger.info(f"Recorded loading metrics for {model_name}: {loading_time:.2f}ms, {memory_footprint:.2f}GB")
    
    def record_inference(self, model_name: str, inference_time: float, 
                        success: bool = True, gpu_memory_used: float = 0.0,
                        request_id: str = None):
        """Record model inference performance"""
        with self.lock:
            # Initialize model metrics if not exists
            if model_name not in self.model_metrics:
                self.model_metrics[model_name] = ModelPerformanceMetrics(model_name=model_name)
            
            metrics = self.model_metrics[model_name]
            
            # Update inference metrics
            metrics.inference_count += 1
            metrics.last_inference_time = datetime.utcnow()
            metrics.gpu_memory_usage = gpu_memory_used
            
            if success:
                metrics.total_inference_time += inference_time
                metrics.average_inference_time = metrics.total_inference_time / metrics.inference_count
                metrics.min_inference_time = min(metrics.min_inference_time, inference_time)
                metrics.max_inference_time = max(metrics.max_inference_time, inference_time)
                metrics.inference_times_history.append(inference_time)
            else:
                metrics.error_count += 1
            
            # Calculate success rate
            metrics.success_rate = (metrics.inference_count - metrics.error_count) / metrics.inference_count
            
            # Record in metrics collector
            tags = {"model": model_name, "success": str(success)}
            if request_id:
                tags["request_id"] = request_id
            
            self.metrics_collector.record_model_inference(
                request_id or "unknown", model_name, inference_time, success
            )
            
            # Record GPU memory usage
            if gpu_memory_used > 0:
                self.metrics_collector.record_metric("model_gpu_memory", gpu_memory_used, tags, MetricType.GAUGE)
    
    def assess_model_health(self, model_name: str) -> ModelHealthStatus:
        """Assess individual model health"""
        with self.lock:
            current_time = datetime.utcnow()
            
            # Get model metrics
            if model_name not in self.model_metrics:
                return ModelHealthStatus(
                    model_name=model_name,
                    status="not_loaded",
                    health_score=0.0,
                    last_check=current_time,
                    performance_trend="unknown",
                    alerts=["Model not found in metrics"],
                    recommendations=["Check if model is properly loaded"]
                )
            
            metrics = self.model_metrics[model_name]
            loading_metrics = self.loading_metrics.get(model_name)
            
            # Calculate health score
            health_score = self._calculate_health_score(metrics, loading_metrics)
            
            # Determine status
            status = self._determine_health_status(health_score, metrics)
            
            # Analyze performance trend
            trend = self._analyze_performance_trend(metrics)
            
            # Generate alerts and recommendations
            alerts = self._generate_health_alerts(metrics)
            recommendations = self._generate_health_recommendations(metrics)
            
            health_status = ModelHealthStatus(
                model_name=model_name,
                status=status,
                health_score=health_score,
                last_check=current_time,
                performance_trend=trend,
                alerts=alerts,
                recommendations=recommendations
            )
            
            return health_status
    
    def get_all_models_health(self) -> Dict[str, ModelHealthStatus]:
        """Get health status for all monitored models"""
        health_statuses = {}
        
        # Check all models that have metrics
        for model_name in self.model_metrics.keys():
            health_statuses[model_name] = self.assess_model_health(model_name)
        
        # Check loaded models that might not have inference metrics yet
        if self.model_loader:
            loaded_models = self.model_loader.get_loaded_models()
            for model_name in loaded_models:
                if model_name not in health_statuses:
                    health_statuses[model_name] = self.assess_model_health(model_name)
        
        return health_statuses
    
    def get_performance_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for all models"""
        with self.lock:
            cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
            
            summary = {
                "monitoring_window_minutes": time_window_minutes,
                "models": {},
                "overall_metrics": {
                    "total_models": len(self.model_metrics),
                    "healthy_models": 0,
                    "degraded_models": 0,
                    "error_models": 0
                }
            }
            
            # Analyze each model
            for model_name, metrics in self.model_metrics.items():
                health = self.assess_model_health(model_name)
                
                # Update overall counters
                if health.status == "healthy":
                    summary["overall_metrics"]["healthy_models"] += 1
                elif health.status == "degraded":
                    summary["overall_metrics"]["degraded_models"] += 1
                else:
                    summary["overall_metrics"]["error_models"] += 1
                
                # Calculate recent performance
                recent_inferences = [t for t in metrics.inference_times_history 
                                   if metrics.last_inference_time and 
                                   (metrics.last_inference_time - timedelta(minutes=time_window_minutes)) <= datetime.utcnow()]
                
                model_summary = {
                    "health_status": health.status,
                    "health_score": health.health_score,
                    "performance_trend": health.performance_trend,
                    "total_inferences": metrics.inference_count,
                    "recent_inferences": len(recent_inferences),
                    "average_inference_time": metrics.average_inference_time,
                    "success_rate": metrics.success_rate,
                    "gpu_memory_usage": metrics.gpu_memory_usage,
                    "last_inference": metrics.last_inference_time.isoformat() if metrics.last_inference_time else None,
                    "alerts": health.alerts,
                    "recommendations": health.recommendations
                }
                
                if recent_inferences:
                    model_summary["recent_avg_inference_time"] = sum(recent_inferences) / len(recent_inferences)
                    model_summary["recent_min_inference_time"] = min(recent_inferences)
                    model_summary["recent_max_inference_time"] = max(recent_inferences)
                
                summary["models"][model_name] = model_summary
            
            return summary
    
    def _calculate_health_score(self, metrics: ModelPerformanceMetrics, 
                              loading_metrics: Optional[ModelLoadingMetrics]) -> float:
        """Calculate overall health score for a model"""
        score_components = []
        
        # Success rate component (40% weight)
        success_score = metrics.success_rate
        score_components.append(("success_rate", success_score, 0.4))
        
        # Performance component (30% weight)
        if metrics.average_inference_time > 0:
            baseline_time = self.performance_baselines.get(metrics.model_name, {}).get("inference_time", 200.0)
            performance_ratio = baseline_time / metrics.average_inference_time
            performance_score = min(1.0, max(0.0, performance_ratio))
        else:
            performance_score = 1.0
        score_components.append(("performance", performance_score, 0.3))
        
        # Stability component (20% weight) - based on inference time variance
        if len(metrics.inference_times_history) > 10:
            times = list(metrics.inference_times_history)
            mean_time = sum(times) / len(times)
            variance = sum((t - mean_time) ** 2 for t in times) / len(times)
            std_dev = variance ** 0.5
            # Lower coefficient of variation = higher stability
            cv = std_dev / mean_time if mean_time > 0 else 0
            stability_score = max(0.0, 1.0 - cv)
        else:
            stability_score = 1.0
        score_components.append(("stability", stability_score, 0.2))
        
        # Loading success component (10% weight)
        if loading_metrics:
            loading_score = 1.0 if loading_metrics.initialization_success else 0.0
        else:
            loading_score = 0.5  # Unknown loading status
        score_components.append(("loading", loading_score, 0.1))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in score_components)
        
        return max(0.0, min(1.0, total_score))
    
    def _determine_health_status(self, health_score: float, metrics: ModelPerformanceMetrics) -> str:
        """Determine health status based on score and metrics"""
        if health_score >= 0.8 and metrics.success_rate >= 0.95:
            return "healthy"
        elif health_score >= 0.6 and metrics.success_rate >= 0.8:
            return "degraded"
        else:
            return "error"
    
    def _analyze_performance_trend(self, metrics: ModelPerformanceMetrics) -> str:
        """Analyze performance trend over recent inferences"""
        if len(metrics.inference_times_history) < 20:
            return "insufficient_data"
        
        # Take recent samples for trend analysis
        recent_times = list(metrics.inference_times_history)[-20:]
        
        # Simple linear trend analysis
        x = np.arange(len(recent_times))
        y = np.array(recent_times)
        
        # Calculate linear regression slope
        n = len(recent_times)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x * x) - np.sum(x) ** 2)
        
        # Determine trend based on slope
        if slope < -5:  # Improving (getting faster)
            return "improving"
        elif slope > 5:  # Degrading (getting slower)
            return "degrading"
        else:
            return "stable"
    
    def _generate_health_alerts(self, metrics: ModelPerformanceMetrics) -> List[str]:
        """Generate health alerts based on model metrics"""
        alerts = []
        
        # Success rate alerts
        if metrics.success_rate < self.performance_thresholds["min_success_rate"]:
            alerts.append(f"Low success rate: {metrics.success_rate:.2%}")
        
        # Performance alerts
        if metrics.average_inference_time > self.performance_thresholds["max_inference_time"]:
            alerts.append(f"High average inference time: {metrics.average_inference_time:.2f}ms")
        
        # GPU memory alerts
        if metrics.gpu_memory_usage > self.performance_thresholds["max_gpu_memory"]:
            alerts.append(f"High GPU memory usage: {metrics.gpu_memory_usage:.2f}GB")
        
        # Error rate alerts
        error_rate = metrics.error_count / metrics.inference_count if metrics.inference_count > 0 else 0
        if error_rate > self.performance_thresholds["max_error_rate"]:
            alerts.append(f"High error rate: {error_rate:.2%}")
        
        # Recent activity alerts
        if metrics.last_inference_time:
            time_since_last = (datetime.utcnow() - metrics.last_inference_time).total_seconds()
            if time_since_last > 3600:  # 1 hour
                alerts.append(f"No recent activity: {time_since_last/3600:.1f} hours since last inference")
        
        return alerts
    
    def _generate_health_recommendations(self, metrics: ModelPerformanceMetrics) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []
        
        # Performance recommendations
        if metrics.average_inference_time > self.performance_thresholds["max_inference_time"]:
            recommendations.append("Consider optimizing model or reducing input size")
            recommendations.append("Check GPU utilization and memory allocation")
        
        # Success rate recommendations
        if metrics.success_rate < 0.9:
            recommendations.append("Review error logs for common failure patterns")
            recommendations.append("Validate input data quality and preprocessing")
        
        # Stability recommendations
        if len(metrics.inference_times_history) > 10:
            times = list(metrics.inference_times_history)
            variance = np.var(times)
            if variance > 10000:  # High variance in inference times
                recommendations.append("Investigate inference time variance")
                recommendations.append("Consider batch size optimization")
        
        # Memory recommendations
        if metrics.gpu_memory_usage > 15.0:  # High but not critical
            recommendations.append("Monitor GPU memory usage for potential optimization")
        
        return recommendations
    
    def _establish_performance_baselines(self):
        """Establish performance baselines for models"""
        # These would typically be learned from historical data
        # For now, setting reasonable defaults
        self.performance_baselines = {
            "wav2lip": {
                "inference_time": 150.0,  # ms
                "gpu_memory": 2.0,  # GB
                "success_rate": 0.98
            },
            "face_detection": {
                "inference_time": 50.0,  # ms
                "gpu_memory": 1.0,  # GB
                "success_rate": 0.99
            }
        }
        
        logger.info("Performance baselines established")
    
    def _start_health_monitoring(self):
        """Start background health monitoring"""
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Perform health checks
                    health_statuses = self.get_all_models_health()
                    
                    # Store health history
                    health_summary = {
                        "timestamp": datetime.utcnow(),
                        "models": {name: status.health_score for name, status in health_statuses.items()}
                    }
                    self.health_history.append(health_summary)
                    
                    # Log any critical alerts
                    for model_name, health in health_statuses.items():
                        if health.status == "error":
                            logger.warning(f"Model {model_name} health critical: {health.alerts}")
                        elif health.alerts:
                            logger.info(f"Model {model_name} alerts: {health.alerts}")
                    
                    time.sleep(self.health_check_interval)
                    
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    time.sleep(self.health_check_interval)
        
        thread = threading.Thread(target=monitoring_loop, daemon=True)
        thread.start()
        logger.info("Health monitoring thread started")
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export model metrics"""
        with self.lock:
            if format_type == "json":
                export_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "performance_summary": self.get_performance_summary(),
                    "health_statuses": {name: {
                        "status": health.status,
                        "score": health.health_score,
                        "trend": health.performance_trend,
                        "alerts": health.alerts
                    } for name, health in self.get_all_models_health().items()},
                    "loading_metrics": {name: {
                        "loading_time": metrics.loading_time,
                        "memory_footprint": metrics.memory_footprint,
                        "is_loaded": metrics.is_loaded,
                        "warm_up_time": metrics.warm_up_time
                    } for name, metrics in self.loading_metrics.items()}
                }
                return json.dumps(export_data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")


# Global model monitor instance
model_monitor: Optional[ModelMonitor] = None


def initialize_model_monitor(metrics_collector: MetricsCollector) -> ModelMonitor:
    """Initialize global model monitor"""
    global model_monitor
    model_monitor = ModelMonitor(metrics_collector)
    logger.info("Model monitor initialized")
    return model_monitor


def get_model_monitor() -> Optional[ModelMonitor]:
    """Get global model monitor instance"""
    return model_monitor 