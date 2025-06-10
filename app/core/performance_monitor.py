"""
Performance Monitor for Avatar Streaming Service.
Tracks system performance, GPU utilization, and processing metrics.
"""

import time
import threading
import psutil
import GPUtil
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging
import asyncio
import statistics
import json

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemResourceMetrics:
    """System resource utilization metrics."""
    
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    gpu_utilization: float
    gpu_memory_percent: float
    gpu_memory_used_gb: float
    gpu_temperature: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ProcessingMetrics:
    """Processing performance metrics."""
    
    chunk_processing_time: float
    model_inference_time: float
    face_cache_access_time: float
    audio_processing_time: float
    video_generation_time: float
    total_pipeline_time: float
    queue_wait_time: float
    gpu_memory_usage: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MetricsCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics_history: deque = deque(maxlen=retention_hours * 3600)  # 1 metric per second
        self.processing_metrics: deque = deque(maxlen=10000)  # Last 10k processing events
        self.system_metrics: deque = deque(maxlen=3600)  # Last hour of system metrics
        self.metric_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.running = False
        self.collection_thread = None
        
    def start_collection(self):
        """Start metrics collection."""
        if self.running:
            return
            
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Performance metrics collection started")
        
    def stop_collection(self):
        """Stop metrics collection."""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Performance metrics collection stopped")
        
    def _collection_loop(self):
        """Main metrics collection loop."""
        while self.running:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                if system_metrics:
                    self.system_metrics.append(system_metrics)
                    
                # Notify subscribers
                self._notify_subscribers("system_metrics", system_metrics)
                
                time.sleep(1)  # Collect every second
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(5)  # Wait before retrying
                
    def _collect_system_metrics(self) -> Optional[SystemResourceMetrics]:
        """Collect current system resource metrics."""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # GPU metrics (assuming single GPU)
            gpu_metrics = {
                "utilization": 0.0,
                "memory_percent": 0.0,
                "memory_used_gb": 0.0,
                "temperature": 0.0
            }
            
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_metrics.update({
                        "utilization": gpu.load * 100,
                        "memory_percent": gpu.memoryUtil * 100,
                        "memory_used_gb": gpu.memoryUsed / 1024,
                        "temperature": gpu.temperature
                    })
            except Exception as e:
                logger.debug(f"GPU metrics collection failed: {e}")
                
            return SystemResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                disk_usage_percent=disk.percent,
                gpu_utilization=gpu_metrics["utilization"],
                gpu_memory_percent=gpu_metrics["memory_percent"],
                gpu_memory_used_gb=gpu_metrics["memory_used_gb"],
                gpu_temperature=gpu_metrics["temperature"]
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return None
            
    def record_processing_metric(self, metrics: ProcessingMetrics):
        """Record processing performance metrics."""
        self.processing_metrics.append(metrics)
        self._notify_subscribers("processing_metrics", metrics)
        
    def record_custom_metric(self, name: str, value: float, unit: str, context: Dict[str, Any] = None):
        """Record custom performance metric."""
        metric = PerformanceMetric(
            timestamp=datetime.utcnow(),
            metric_name=name,
            value=value,
            unit=unit,
            context=context or {}
        )
        self.metrics_history.append(metric)
        self._notify_subscribers("custom_metric", metric)
        
    def subscribe_to_metric(self, metric_type: str, callback: Callable):
        """Subscribe to metric updates."""
        self.metric_subscribers[metric_type].append(callback)
        
    def _notify_subscribers(self, metric_type: str, data: Any):
        """Notify metric subscribers."""
        for callback in self.metric_subscribers.get(metric_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error notifying metric subscriber: {e}")


class PerformanceAnalyzer:
    """Analyzes performance metrics and generates insights."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.performance_thresholds = {
            "cpu_warning": 80.0,
            "memory_warning": 85.0,
            "gpu_warning": 90.0,
            "processing_time_warning": 2.0,  # seconds
            "cache_miss_rate_warning": 0.1  # 10%
        }
        
    def analyze_current_performance(self) -> Dict[str, Any]:
        """Analyze current system performance."""
        analysis = {
            "timestamp": datetime.utcnow(),
            "overall_health": "healthy",
            "warnings": [],
            "recommendations": [],
            "metrics": {}
        }
        
        # Analyze system metrics
        if self.metrics_collector.system_metrics:
            latest_system = self.metrics_collector.system_metrics[-1]
            analysis["metrics"]["system"] = self._analyze_system_metrics(latest_system)
            
            # Check for warnings
            if latest_system.cpu_percent > self.performance_thresholds["cpu_warning"]:
                analysis["warnings"].append(f"High CPU usage: {latest_system.cpu_percent:.1f}%")
                analysis["overall_health"] = "warning"
                
            if latest_system.memory_percent > self.performance_thresholds["memory_warning"]:
                analysis["warnings"].append(f"High memory usage: {latest_system.memory_percent:.1f}%")
                analysis["overall_health"] = "warning"
                
            if latest_system.gpu_utilization > self.performance_thresholds["gpu_warning"]:
                analysis["warnings"].append(f"High GPU usage: {latest_system.gpu_utilization:.1f}%")
                analysis["overall_health"] = "warning"
                
        # Analyze processing metrics
        if self.metrics_collector.processing_metrics:
            analysis["metrics"]["processing"] = self._analyze_processing_metrics()
            
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
        
    def _analyze_system_metrics(self, metrics: SystemResourceMetrics) -> Dict[str, Any]:
        """Analyze system resource metrics."""
        return {
            "cpu_status": self._get_resource_status(metrics.cpu_percent, self.performance_thresholds["cpu_warning"]),
            "memory_status": self._get_resource_status(metrics.memory_percent, self.performance_thresholds["memory_warning"]),
            "gpu_status": self._get_resource_status(metrics.gpu_utilization, self.performance_thresholds["gpu_warning"]),
            "gpu_memory_available_gb": 24 - metrics.gpu_memory_used_gb,  # Assuming RTX 4090
            "system_load": (metrics.cpu_percent + metrics.memory_percent + metrics.gpu_utilization) / 3
        }
        
    def _analyze_processing_metrics(self) -> Dict[str, Any]:
        """Analyze processing performance metrics."""
        recent_metrics = list(self.metrics_collector.processing_metrics)[-100:]  # Last 100 processing events
        
        if not recent_metrics:
            return {"status": "no_data"}
            
        # Calculate averages
        avg_processing_time = statistics.mean([m.total_pipeline_time for m in recent_metrics])
        avg_inference_time = statistics.mean([m.model_inference_time for m in recent_metrics])
        avg_cache_time = statistics.mean([m.face_cache_access_time for m in recent_metrics])
        
        # Calculate cache hit rate (assuming cache_access_time < 0.05s indicates hit)
        cache_hits = sum(1 for m in recent_metrics if m.face_cache_access_time < 0.05)
        cache_hit_rate = cache_hits / len(recent_metrics) if recent_metrics else 0
        
        return {
            "average_processing_time": avg_processing_time,
            "average_inference_time": avg_inference_time,
            "average_cache_access_time": avg_cache_time,
            "cache_hit_rate": cache_hit_rate,
            "processing_stability": self._calculate_stability(recent_metrics),
            "throughput_per_minute": len(recent_metrics) * (60 / max(1, (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds()))
        }
        
    def _calculate_stability(self, metrics: List[ProcessingMetrics]) -> float:
        """Calculate processing time stability (lower variance = higher stability)."""
        if len(metrics) < 2:
            return 1.0
            
        times = [m.total_pipeline_time for m in metrics]
        variance = statistics.variance(times)
        mean_time = statistics.mean(times)
        
        # Normalized stability score (0-1, where 1 is perfectly stable)
        cv = variance / mean_time if mean_time > 0 else 0  # Coefficient of variation
        stability = max(0.0, 1.0 - cv)
        
        return stability
        
    def _get_resource_status(self, usage: float, warning_threshold: float) -> str:
        """Get resource status based on usage and threshold."""
        if usage < warning_threshold * 0.7:
            return "healthy"
        elif usage < warning_threshold:
            return "moderate"
        elif usage < warning_threshold * 1.2:
            return "warning"
        else:
            return "critical"
            
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if "system" in analysis["metrics"]:
            system_metrics = analysis["metrics"]["system"]
            
            if system_metrics["cpu_status"] in ["warning", "critical"]:
                recommendations.append("Consider reducing processing quality or chunk size to lower CPU usage")
                
            if system_metrics["memory_status"] in ["warning", "critical"]:
                recommendations.append("Clear model cache or reduce buffer sizes to free memory")
                
            if system_metrics["gpu_status"] in ["warning", "critical"]:
                recommendations.append("Reduce concurrent users or model quality to lower GPU usage")
                
        if "processing" in analysis["metrics"]:
            processing_metrics = analysis["metrics"]["processing"]
            
            if processing_metrics.get("cache_hit_rate", 1.0) < 0.8:
                recommendations.append("Improve avatar cache warming to increase cache hit rate")
                
            if processing_metrics.get("processing_stability", 1.0) < 0.7:
                recommendations.append("Investigate processing time variance - consider adjusting chunk sizes")
                
            if processing_metrics.get("average_processing_time", 0) > 2.0:
                recommendations.append("Processing time is high - consider optimizing pipeline or reducing quality")
                
        return recommendations
        
    def get_performance_trends(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance trends over specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Filter recent system metrics
        recent_system = [m for m in self.metrics_collector.system_metrics if m.timestamp > cutoff_time]
        recent_processing = [m for m in self.metrics_collector.processing_metrics if m.timestamp > cutoff_time]
        
        trends = {
            "time_period_hours": hours,
            "data_points": len(recent_system),
            "system_trends": {},
            "processing_trends": {}
        }
        
        if recent_system:
            trends["system_trends"] = {
                "cpu_trend": self._calculate_trend([m.cpu_percent for m in recent_system]),
                "memory_trend": self._calculate_trend([m.memory_percent for m in recent_system]),
                "gpu_trend": self._calculate_trend([m.gpu_utilization for m in recent_system]),
                "gpu_memory_trend": self._calculate_trend([m.gpu_memory_used_gb for m in recent_system])
            }
            
        if recent_processing:
            trends["processing_trends"] = {
                "processing_time_trend": self._calculate_trend([m.total_pipeline_time for m in recent_processing]),
                "inference_time_trend": self._calculate_trend([m.model_inference_time for m in recent_processing]),
                "cache_access_trend": self._calculate_trend([m.face_cache_access_time for m in recent_processing])
            }
            
        return trends
        
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend direction and magnitude for a metric."""
        if len(values) < 2:
            return {"direction": "stable", "magnitude": 0.0, "confidence": 0.0}
            
        # Simple linear trend calculation
        x = list(range(len(values)))
        n = len(values)
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        # Linear regression slope
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2) if (n * sum_x2 - sum_x ** 2) != 0 else 0
        
        # Determine trend direction
        if abs(slope) < 0.01:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
            
        # Calculate confidence based on R-squared
        mean_y = sum_y / n
        ss_tot = sum((values[i] - mean_y) ** 2 for i in range(n))
        ss_res = sum((values[i] - (slope * x[i] + (sum_y - slope * sum_x) / n)) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            "direction": direction,
            "magnitude": abs(slope),
            "confidence": max(0.0, r_squared)
        }


class AlertManager:
    """Manages performance alerts and notifications."""
    
    def __init__(self, analyzer: PerformanceAnalyzer):
        self.analyzer = analyzer
        self.alert_handlers: List[Callable] = []
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_cooldown: Dict[str, datetime] = {}
        self.cooldown_duration = timedelta(minutes=5)
        
    def add_alert_handler(self, handler: Callable):
        """Add alert handler function."""
        self.alert_handlers.append(handler)
        
    def check_and_trigger_alerts(self):
        """Check performance and trigger alerts if needed."""
        analysis = self.analyzer.analyze_current_performance()
        
        current_time = datetime.utcnow()
        
        for warning in analysis.get("warnings", []):
            alert_key = f"warning_{hash(warning)}"
            
            # Check cooldown
            if alert_key in self.alert_cooldown:
                if current_time - self.alert_cooldown[alert_key] < self.cooldown_duration:
                    continue  # Still in cooldown
                    
            # Trigger alert
            alert_data = {
                "type": "warning",
                "message": warning,
                "timestamp": current_time,
                "analysis": analysis
            }
            
            self._trigger_alert(alert_data)
            self.alert_cooldown[alert_key] = current_time
            
        # Check for critical conditions
        if analysis["overall_health"] == "critical":
            alert_key = "critical_system_health"
            
            if alert_key not in self.alert_cooldown or current_time - self.alert_cooldown[alert_key] > self.cooldown_duration:
                alert_data = {
                    "type": "critical",
                    "message": "System health is critical",
                    "timestamp": current_time,
                    "analysis": analysis
                }
                
                self._trigger_alert(alert_data)
                self.alert_cooldown[alert_key] = current_time
                
    def _trigger_alert(self, alert_data: Dict[str, Any]):
        """Trigger alert to all handlers."""
        self.alert_history.append(alert_data)
        
        for handler in self.alert_handlers:
            try:
                handler(alert_data)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
                
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts within specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert["timestamp"] > cutoff_time]


class PerformanceMonitor:
    """Main performance monitoring orchestrator."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.analyzer = PerformanceAnalyzer(self.metrics_collector)
        self.alert_manager = AlertManager(self.analyzer)
        self.monitoring_active = False
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.metrics_collector.start_collection()
        self.monitoring_active = True
        logger.info("Performance monitoring started")
        
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.metrics_collector.stop_collection()
        self.monitoring_active = False
        logger.info("Performance monitoring stopped")
        
    def record_processing_event(self, **metrics):
        """Record a processing event with timing metrics."""
        processing_metrics = ProcessingMetrics(**metrics)
        self.metrics_collector.record_processing_metric(processing_metrics)
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "current_analysis": self.analyzer.analyze_current_performance(),
            "trends_1h": self.analyzer.get_performance_trends(hours=1),
            "recent_alerts": self.alert_manager.get_recent_alerts(hours=24),
            "monitoring_status": "active" if self.monitoring_active else "inactive"
        }
        
    def add_alert_handler(self, handler: Callable):
        """Add custom alert handler."""
        self.alert_manager.add_alert_handler(handler) 