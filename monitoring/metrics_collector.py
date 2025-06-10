"""
Metrics Collector for Avatar Streaming Service
Collects and aggregates performance metrics for models, avatar cache, processing pipeline, and system resources.
"""

import time
import psutil
import GPUtil
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import asyncio
import json
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric type enumeration"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMING = "timing"


@dataclass
class MetricPoint:
    """Individual metric data point"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class ProcessingMetrics:
    """Processing pipeline metrics"""
    request_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    processing_stages: Dict[str, float] = field(default_factory=dict)
    model_inference_time: float = 0.0
    avatar_cache_hit: bool = False
    avatar_cache_access_time: float = 0.0
    chunk_count: int = 0
    total_audio_duration: float = 0.0
    total_video_frames: int = 0
    gpu_memory_peak: float = 0.0
    quality_score: float = 0.0
    errors: List[str] = field(default_factory=list)


@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    gpu_utilization: float
    gpu_memory_used: float
    gpu_memory_total: float
    gpu_temperature: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_sessions: int
    processing_queue_size: int


@dataclass
class AvatarCacheMetrics:
    """Avatar cache performance metrics"""
    cache_hits: int = 0
    cache_misses: int = 0
    cache_size: int = 0
    cache_memory_usage: float = 0.0
    average_access_time: float = 0.0
    cache_hit_rate: float = 0.0
    registered_avatars: int = 0
    active_avatars: int = 0
    cache_evictions: int = 0


class MetricsCollector:
    """Central metrics collection and aggregation system"""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics_buffer = deque(maxlen=10000)
        self.processing_metrics = {}
        self.system_metrics_history = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.avatar_cache_metrics = AvatarCacheMetrics()
        
        # Aggregated metrics
        self.request_counters = defaultdict(int)
        self.timing_histograms = defaultdict(list)
        self.gauge_values = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Collection state
        self.collection_active = False
        self.collection_interval = 60  # seconds
        
        # Performance tracking
        self.performance_baselines = {}
        self.alerts_config = {}
        
    def start_collection(self):
        """Start automated metrics collection"""
        self.collection_active = True
        self._start_system_metrics_collection()
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop automated metrics collection"""
        self.collection_active = False
        logger.info("Metrics collection stopped")
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None, 
                     metric_type: MetricType = MetricType.GAUGE):
        """Record a single metric point"""
        with self.lock:
            metric = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.utcnow(),
                tags=tags or {},
                metric_type=metric_type
            )
            
            self.metrics_buffer.append(metric)
            
            # Update aggregated metrics
            if metric_type == MetricType.COUNTER:
                self.request_counters[name] += value
            elif metric_type == MetricType.GAUGE:
                self.gauge_values[name] = value
            elif metric_type == MetricType.TIMING:
                self.timing_histograms[name].append(value)
                # Keep only recent timing data
                if len(self.timing_histograms[name]) > 1000:
                    self.timing_histograms[name] = self.timing_histograms[name][-500:]
    
    def start_processing_request(self, request_id: str) -> ProcessingMetrics:
        """Start tracking a processing request"""
        with self.lock:
            metrics = ProcessingMetrics(
                request_id=request_id,
                start_time=datetime.utcnow()
            )
            self.processing_metrics[request_id] = metrics
            
            # Record request start
            self.record_metric("requests_started", 1, 
                             {"request_id": request_id}, MetricType.COUNTER)
            
            return metrics
    
    def record_processing_stage(self, request_id: str, stage_name: str, duration_ms: float):
        """Record processing stage timing"""
        with self.lock:
            if request_id in self.processing_metrics:
                self.processing_metrics[request_id].processing_stages[stage_name] = duration_ms
                
                # Record stage timing
                self.record_metric(f"stage_{stage_name}_duration", duration_ms,
                                 {"request_id": request_id}, MetricType.TIMING)
    
    def record_model_inference(self, request_id: str, model_name: str, 
                             inference_time_ms: float, success: bool = True):
        """Record model inference metrics"""
        with self.lock:
            if request_id in self.processing_metrics:
                self.processing_metrics[request_id].model_inference_time += inference_time_ms
            
            # Record inference metrics
            tags = {"model": model_name, "success": str(success)}
            self.record_metric("model_inference_duration", inference_time_ms, tags, MetricType.TIMING)
            self.record_metric("model_inference_count", 1, tags, MetricType.COUNTER)
            
            if not success:
                self.record_metric("model_inference_errors", 1, tags, MetricType.COUNTER)
    
    def record_avatar_cache_access(self, request_id: str, avatar_id: str, 
                                 access_time_ms: float, cache_hit: bool):
        """Record avatar cache access metrics"""
        with self.lock:
            # Update request metrics
            if request_id in self.processing_metrics:
                self.processing_metrics[request_id].avatar_cache_hit = cache_hit
                self.processing_metrics[request_id].avatar_cache_access_time = access_time_ms
            
            # Update cache metrics
            if cache_hit:
                self.avatar_cache_metrics.cache_hits += 1
            else:
                self.avatar_cache_metrics.cache_misses += 1
            
            # Calculate hit rate
            total_accesses = self.avatar_cache_metrics.cache_hits + self.avatar_cache_metrics.cache_misses
            if total_accesses > 0:
                self.avatar_cache_metrics.cache_hit_rate = self.avatar_cache_metrics.cache_hits / total_accesses
            
            # Record access metrics
            tags = {"avatar_id": avatar_id, "cache_hit": str(cache_hit)}
            self.record_metric("avatar_cache_access_time", access_time_ms, tags, MetricType.TIMING)
            self.record_metric("avatar_cache_accesses", 1, tags, MetricType.COUNTER)
    
    def record_chunk_processing(self, request_id: str, chunk_id: str, 
                              processing_time_ms: float, frame_count: int):
        """Record chunk processing metrics"""
        with self.lock:
            if request_id in self.processing_metrics:
                self.processing_metrics[request_id].chunk_count += 1
                self.processing_metrics[request_id].total_video_frames += frame_count
            
            # Record chunk metrics
            tags = {"request_id": request_id, "chunk_id": chunk_id}
            self.record_metric("chunk_processing_time", processing_time_ms, tags, MetricType.TIMING)
            self.record_metric("chunk_frame_count", frame_count, tags, MetricType.GAUGE)
            self.record_metric("chunks_processed", 1, tags, MetricType.COUNTER)
    
    def record_gpu_memory_usage(self, request_id: str, memory_used_gb: float, memory_total_gb: float):
        """Record GPU memory usage during processing"""
        with self.lock:
            if request_id in self.processing_metrics:
                self.processing_metrics[request_id].gpu_memory_peak = max(
                    self.processing_metrics[request_id].gpu_memory_peak, memory_used_gb
                )
            
            # Record GPU metrics
            self.record_metric("gpu_memory_used", memory_used_gb, {"request_id": request_id})
            self.record_metric("gpu_memory_utilization", 
                             (memory_used_gb / memory_total_gb) * 100 if memory_total_gb > 0 else 0,
                             {"request_id": request_id})
    
    def finish_processing_request(self, request_id: str, success: bool = True, 
                                quality_score: float = 0.0, errors: List[str] = None):
        """Finish tracking a processing request"""
        with self.lock:
            if request_id not in self.processing_metrics:
                return
            
            metrics = self.processing_metrics[request_id]
            metrics.end_time = datetime.utcnow()
            metrics.quality_score = quality_score
            metrics.errors = errors or []
            
            # Calculate total processing time
            total_time = (metrics.end_time - metrics.start_time).total_seconds() * 1000
            
            # Record completion metrics
            tags = {"success": str(success), "request_id": request_id}
            self.record_metric("request_total_time", total_time, tags, MetricType.TIMING)
            self.record_metric("request_quality_score", quality_score, tags, MetricType.GAUGE)
            self.record_metric("requests_completed", 1, tags, MetricType.COUNTER)
            
            if not success:
                self.record_metric("requests_failed", 1, tags, MetricType.COUNTER)
            
            # Clean up old metrics
            if len(self.processing_metrics) > 1000:
                oldest_keys = sorted(self.processing_metrics.keys(), 
                                   key=lambda k: self.processing_metrics[k].start_time)[:500]
                for key in oldest_keys:
                    del self.processing_metrics[key]
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system resource metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # GPU metrics
            gpu_utilization = 0.0
            gpu_memory_used = 0.0
            gpu_memory_total = 0.0
            gpu_temperature = 0.0
            
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_utilization = gpu.load * 100
                    gpu_memory_used = gpu.memoryUsed / 1024  # GB
                    gpu_memory_total = gpu.memoryTotal / 1024  # GB
                    gpu_temperature = gpu.temperature
            except Exception as e:
                logger.warning(f"Failed to collect GPU metrics: {e}")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            # Network stats
            network = psutil.net_io_counters()
            
            system_metrics = SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                gpu_utilization=gpu_utilization,
                gpu_memory_used=gpu_memory_used,
                gpu_memory_total=gpu_memory_total,
                gpu_temperature=gpu_temperature,
                disk_usage_percent=disk_usage_percent,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                active_sessions=len(self.processing_metrics),
                processing_queue_size=0  # Would be populated by queue manager
            )
            
            # Store in history
            with self.lock:
                self.system_metrics_history.append(system_metrics)
            
            # Record as individual metrics
            self.record_metric("system_cpu_percent", cpu_percent)
            self.record_metric("system_memory_percent", memory.percent)
            self.record_metric("system_gpu_utilization", gpu_utilization)
            self.record_metric("system_gpu_memory_used", gpu_memory_used)
            self.record_metric("system_gpu_temperature", gpu_temperature)
            self.record_metric("system_disk_usage_percent", disk_usage_percent)
            self.record_metric("system_active_sessions", len(self.processing_metrics))
            
            return system_metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(timestamp=datetime.utcnow(), cpu_percent=0, memory_percent=0,
                               gpu_utilization=0, gpu_memory_used=0, gpu_memory_total=0,
                               gpu_temperature=0, disk_usage_percent=0, network_bytes_sent=0,
                               network_bytes_recv=0, active_sessions=0, processing_queue_size=0)
    
    def update_avatar_cache_metrics(self, cache_size: int, memory_usage_mb: float, 
                                  registered_avatars: int, active_avatars: int):
        """Update avatar cache system metrics"""
        with self.lock:
            self.avatar_cache_metrics.cache_size = cache_size
            self.avatar_cache_metrics.cache_memory_usage = memory_usage_mb
            self.avatar_cache_metrics.registered_avatars = registered_avatars
            self.avatar_cache_metrics.active_avatars = active_avatars
            
            # Record as metrics
            self.record_metric("avatar_cache_size", cache_size)
            self.record_metric("avatar_cache_memory_usage_mb", memory_usage_mb)
            self.record_metric("avatar_cache_hit_rate", self.avatar_cache_metrics.cache_hit_rate)
            self.record_metric("avatar_registered_count", registered_avatars)
            self.record_metric("avatar_active_count", active_avatars)
    
    def get_performance_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for specified time window"""
        with self.lock:
            cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
            
            # Filter recent metrics
            recent_metrics = [m for m in self.metrics_buffer if m.timestamp >= cutoff_time]
            
            # Calculate aggregations
            summary = {
                "time_window_minutes": time_window_minutes,
                "metrics_count": len(recent_metrics),
                "request_metrics": self._calculate_request_metrics(cutoff_time),
                "performance_metrics": self._calculate_performance_metrics(recent_metrics),
                "avatar_cache_metrics": self._get_avatar_cache_summary(),
                "system_metrics": self._get_system_metrics_summary(cutoff_time),
                "alerts": self._check_performance_alerts()
            }
            
            return summary
    
    def _calculate_request_metrics(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Calculate request-related metrics"""
        recent_requests = [m for m in self.processing_metrics.values() 
                          if m.start_time >= cutoff_time]
        
        if not recent_requests:
            return {"total_requests": 0}
        
        completed_requests = [r for r in recent_requests if r.end_time is not None]
        
        # Calculate request rate
        time_span_hours = (datetime.utcnow() - cutoff_time).total_seconds() / 3600
        request_rate = len(recent_requests) / time_span_hours if time_span_hours > 0 else 0
        
        # Calculate average processing time
        if completed_requests:
            processing_times = [(r.end_time - r.start_time).total_seconds() * 1000 
                              for r in completed_requests]
            avg_processing_time = sum(processing_times) / len(processing_times)
            min_processing_time = min(processing_times)
            max_processing_time = max(processing_times)
        else:
            avg_processing_time = min_processing_time = max_processing_time = 0
        
        # Calculate success rate
        successful_requests = [r for r in completed_requests if not r.errors]
        success_rate = len(successful_requests) / len(completed_requests) if completed_requests else 0
        
        return {
            "total_requests": len(recent_requests),
            "completed_requests": len(completed_requests),
            "requests_per_hour": request_rate,
            "average_processing_time_ms": avg_processing_time,
            "min_processing_time_ms": min_processing_time,
            "max_processing_time_ms": max_processing_time,
            "success_rate": success_rate,
            "cache_hit_rate": self.avatar_cache_metrics.cache_hit_rate
        }
    
    def _calculate_performance_metrics(self, recent_metrics: List[MetricPoint]) -> Dict[str, Any]:
        """Calculate performance-related metrics"""
        timing_metrics = [m for m in recent_metrics if m.metric_type == MetricType.TIMING]
        
        performance = {}
        
        # Group by metric name
        timing_by_name = defaultdict(list)
        for metric in timing_metrics:
            timing_by_name[metric.name].append(metric.value)
        
        # Calculate statistics for each timing metric
        for name, values in timing_by_name.items():
            if values:
                performance[name] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "p95": self._calculate_percentile(values, 95),
                    "p99": self._calculate_percentile(values, 99)
                }
        
        return performance
    
    def _get_avatar_cache_summary(self) -> Dict[str, Any]:
        """Get avatar cache metrics summary"""
        return {
            "cache_hits": self.avatar_cache_metrics.cache_hits,
            "cache_misses": self.avatar_cache_metrics.cache_misses,
            "hit_rate": self.avatar_cache_metrics.cache_hit_rate,
            "cache_size": self.avatar_cache_metrics.cache_size,
            "memory_usage_mb": self.avatar_cache_metrics.cache_memory_usage,
            "registered_avatars": self.avatar_cache_metrics.registered_avatars,
            "active_avatars": self.avatar_cache_metrics.active_avatars,
            "cache_evictions": self.avatar_cache_metrics.cache_evictions
        }
    
    def _get_system_metrics_summary(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Get system metrics summary"""
        recent_system_metrics = [m for m in self.system_metrics_history 
                               if m.timestamp >= cutoff_time]
        
        if not recent_system_metrics:
            return {}
        
        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in recent_system_metrics) / len(recent_system_metrics)
        avg_memory = sum(m.memory_percent for m in recent_system_metrics) / len(recent_system_metrics)
        avg_gpu_util = sum(m.gpu_utilization for m in recent_system_metrics) / len(recent_system_metrics)
        avg_gpu_temp = sum(m.gpu_temperature for m in recent_system_metrics) / len(recent_system_metrics)
        
        # Get current values
        current = recent_system_metrics[-1]
        
        return {
            "current_cpu_percent": current.cpu_percent,
            "current_memory_percent": current.memory_percent,
            "current_gpu_utilization": current.gpu_utilization,
            "current_gpu_memory_used": current.gpu_memory_used,
            "current_gpu_temperature": current.gpu_temperature,
            "average_cpu_percent": avg_cpu,
            "average_memory_percent": avg_memory,
            "average_gpu_utilization": avg_gpu_util,
            "average_gpu_temperature": avg_gpu_temp
        }
    
    def _check_performance_alerts(self) -> List[Dict[str, Any]]:
        """Check for performance alerts"""
        alerts = []
        
        # Check recent system metrics
        if self.system_metrics_history:
            current = self.system_metrics_history[-1]
            
            # CPU alert
            if current.cpu_percent > 90:
                alerts.append({
                    "type": "high_cpu",
                    "severity": "critical" if current.cpu_percent > 95 else "warning",
                    "value": current.cpu_percent,
                    "threshold": 90,
                    "message": f"High CPU usage: {current.cpu_percent:.1f}%"
                })
            
            # Memory alert
            if current.memory_percent > 85:
                alerts.append({
                    "type": "high_memory",
                    "severity": "critical" if current.memory_percent > 95 else "warning",
                    "value": current.memory_percent,
                    "threshold": 85,
                    "message": f"High memory usage: {current.memory_percent:.1f}%"
                })
            
            # GPU temperature alert
            if current.gpu_temperature > 80:
                alerts.append({
                    "type": "high_gpu_temperature",
                    "severity": "critical" if current.gpu_temperature > 85 else "warning",
                    "value": current.gpu_temperature,
                    "threshold": 80,
                    "message": f"High GPU temperature: {current.gpu_temperature:.1f}°C"
                })
        
        # Check cache hit rate
        if self.avatar_cache_metrics.cache_hit_rate < 0.7:
            alerts.append({
                "type": "low_cache_hit_rate",
                "severity": "warning",
                "value": self.avatar_cache_metrics.cache_hit_rate,
                "threshold": 0.7,
                "message": f"Low cache hit rate: {self.avatar_cache_metrics.cache_hit_rate:.2f}"
            })
        
        return alerts
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        index = max(0, min(index, len(sorted_values) - 1))
        return sorted_values[index]
    
    def _start_system_metrics_collection(self):
        """Start automated system metrics collection in background"""
        def collect_loop():
            while self.collection_active:
                try:
                    self.collect_system_metrics()
                    time.sleep(self.collection_interval)
                except Exception as e:
                    logger.error(f"System metrics collection error: {e}")
                    time.sleep(self.collection_interval)
        
        thread = threading.Thread(target=collect_loop, daemon=True)
        thread.start()
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format"""
        with self.lock:
            if format_type == "json":
                return self._export_json()
            elif format_type == "prometheus":
                return self._export_prometheus()
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_json(self) -> str:
        """Export metrics as JSON"""
        export_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": self.get_performance_summary(),
            "recent_metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "timestamp": m.timestamp.isoformat(),
                    "tags": m.tags,
                    "type": m.metric_type.value
                }
                for m in list(self.metrics_buffer)[-100:]  # Last 100 metrics
            ]
        }
        return json.dumps(export_data, indent=2)
    
    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        # Export gauge values
        for name, value in self.gauge_values.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        
        # Export counters
        for name, value in self.request_counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        
        return "\n".join(lines)


# Global metrics collector instance
metrics_collector = MetricsCollector()


def initialize_metrics_collector():
    """Initialize and start metrics collection"""
    metrics_collector.start_collection()
    logger.info("Metrics collector initialized and started")


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance"""
    return metrics_collector 