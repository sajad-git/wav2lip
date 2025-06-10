"""
Monitoring Middleware for Avatar Streaming Service
Tracks performance metrics, API usage, and system health
"""

import time
import logging
import asyncio
import psutil
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.config.settings import settings


@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    timestamp: float
    method: str
    path: str
    status_code: int
    response_time_ms: float
    request_size_bytes: int
    response_size_bytes: int
    client_id: str
    user_agent: str
    error_message: Optional[str] = None


@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: float
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    active_connections: int
    gpu_usage_percent: float = 0.0
    gpu_memory_used_gb: float = 0.0


@dataclass
class PerformanceStats:
    """Aggregated performance statistics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0
    min_response_time_ms: float = float('inf')
    max_response_time_ms: float = 0.0
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    unique_clients: int = 0


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive monitoring and metrics collection"""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.request_metrics: deque = deque(maxlen=10000)  # Last 10k requests
        self.system_metrics: deque = deque(maxlen=1000)   # Last 1k system snapshots
        
        # Performance tracking
        self.endpoint_stats: Dict[str, PerformanceStats] = defaultdict(PerformanceStats)
        self.client_stats: Dict[str, PerformanceStats] = defaultdict(PerformanceStats)
        
        # Real-time counters
        self.active_requests = 0
        self.total_requests = 0
        self.start_time = time.time()
        
        # Monitoring tasks
        self.system_monitor_task = None
        self.cleanup_task = None
        
        # Start background monitoring
        self.start_background_monitoring()
    
    def start_background_monitoring(self):
        """Start background monitoring tasks"""
        async def system_monitor():
            """Monitor system resources"""
            while True:
                try:
                    await asyncio.sleep(10)  # Collect every 10 seconds
                    await self.collect_system_metrics()
                except Exception as e:
                    self.logger.error(f"System monitoring error: {e}")
        
        async def cleanup_old_metrics():
            """Clean up old metrics"""
            while True:
                try:
                    await asyncio.sleep(300)  # Cleanup every 5 minutes
                    await self.cleanup_old_metrics()
                except Exception as e:
                    self.logger.error(f"Metrics cleanup error: {e}")
        
        self.system_monitor_task = asyncio.create_task(system_monitor())
        self.cleanup_task = asyncio.create_task(cleanup_old_metrics())
    
    async def collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # GPU metrics (if available)
            gpu_usage = 0.0
            gpu_memory = 0.0
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Primary GPU
                    gpu_usage = gpu.load * 100
                    gpu_memory = gpu.memoryUsed / 1024  # Convert MB to GB
            except ImportError:
                pass  # GPU monitoring not available
            
            # Create metrics record
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_available_gb=memory.available / (1024**3),
                disk_usage_percent=disk.percent,
                active_connections=self.active_requests,
                gpu_usage_percent=gpu_usage,
                gpu_memory_used_gb=gpu_memory
            )
            
            self.system_metrics.append(metrics)
            
            # Log critical resource usage
            if cpu_percent > 90:
                self.logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
            if memory.percent > 90:
                self.logger.warning(f"High memory usage: {memory.percent:.1f}%")
            if gpu_usage > 95:
                self.logger.warning(f"High GPU usage: {gpu_usage:.1f}%")
                
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    async def cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory bloat"""
        current_time = time.time()
        
        # Clean request metrics older than 1 hour
        cutoff_time = current_time - 3600
        
        # Filter request metrics
        filtered_requests = deque(maxlen=10000)
        for metric in self.request_metrics:
            if metric.timestamp > cutoff_time:
                filtered_requests.append(metric)
        
        self.request_metrics = filtered_requests
        
        # Filter system metrics (keep last hour)
        filtered_system = deque(maxlen=1000)
        for metric in self.system_metrics:
            if metric.timestamp > cutoff_time:
                filtered_system.append(metric)
        
        self.system_metrics = filtered_system
        
        self.logger.debug(f"Cleaned up metrics. Requests: {len(self.request_metrics)}, System: {len(self.system_metrics)}")
    
    def get_client_identifier(self, request: Request) -> str:
        """Extract client identifier from request"""
        client_id = request.headers.get("X-Client-ID")
        if client_id:
            return f"client_{client_id}"
        
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return f"ip_{forwarded_for.split(',')[0].strip()}"
        
        return f"ip_{request.client.host}" if request.client else "unknown"
    
    def get_endpoint_key(self, path: str, method: str) -> str:
        """Get normalized endpoint key for metrics"""
        # Normalize paths with IDs
        if "/avatar/" in path and path.count("/") >= 3:
            parts = path.split("/")
            if len(parts) >= 4:
                # Replace avatar ID with placeholder
                parts[2] = "{avatar_id}"
                return f"{method}:" + "/".join(parts[:4])
        
        # Other dynamic paths
        normalized_path = path
        for pattern in [r'/\d+', r'/[a-f0-9-]{36}']:  # Numbers and UUIDs
            import re
            normalized_path = re.sub(pattern, '/{id}', normalized_path)
        
        return f"{method}:{normalized_path}"
    
    def calculate_request_size(self, request: Request) -> int:
        """Estimate request size"""
        size = 0
        
        # Headers
        for name, value in request.headers.items():
            size += len(name) + len(value) + 4  # ": " and "\r\n"
        
        # URL
        size += len(str(request.url))
        
        # Content-Length header if available
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size += int(content_length)
            except ValueError:
                pass
        
        return size
    
    def calculate_response_size(self, response: Response) -> int:
        """Estimate response size"""
        size = 0
        
        # Headers
        for name, value in response.headers.items():
            size += len(name) + len(value) + 4
        
        # Content-Length header if available
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                size += int(content_length)
            except ValueError:
                pass
        
        return size
    
    def update_performance_stats(self, endpoint_key: str, client_id: str, 
                               response_time_ms: float, success: bool):
        """Update performance statistics"""
        # Update endpoint stats
        endpoint_stats = self.endpoint_stats[endpoint_key]
        endpoint_stats.total_requests += 1
        if success:
            endpoint_stats.successful_requests += 1
        else:
            endpoint_stats.failed_requests += 1
        
        # Update response time stats
        endpoint_stats.min_response_time_ms = min(endpoint_stats.min_response_time_ms, response_time_ms)
        endpoint_stats.max_response_time_ms = max(endpoint_stats.max_response_time_ms, response_time_ms)
        
        # Calculate rolling average
        if endpoint_stats.total_requests == 1:
            endpoint_stats.avg_response_time_ms = response_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            endpoint_stats.avg_response_time_ms = (
                alpha * response_time_ms + 
                (1 - alpha) * endpoint_stats.avg_response_time_ms
            )
        
        # Calculate error rate
        endpoint_stats.error_rate = (
            endpoint_stats.failed_requests / endpoint_stats.total_requests * 100
        )
        
        # Update client stats (similar logic)
        client_stats = self.client_stats[client_id]
        client_stats.total_requests += 1
        if success:
            client_stats.successful_requests += 1
        else:
            client_stats.failed_requests += 1
        
        client_stats.error_rate = (
            client_stats.failed_requests / client_stats.total_requests * 100
        )
    
    async def dispatch(self, request: Request, call_next):
        """Process request with monitoring"""
        start_time = time.time()
        self.active_requests += 1
        self.total_requests += 1
        
        # Extract request info
        client_id = self.get_client_identifier(request)
        endpoint_key = self.get_endpoint_key(request.url.path, request.method)
        user_agent = request.headers.get("user-agent", "unknown")
        request_size = self.calculate_request_size(request)
        
        error_message = None
        response = None
        
        try:
            # Process request
            response = await call_next(request)
            success = 200 <= response.status_code < 400
            
        except Exception as e:
            self.logger.error(f"Request processing error: {e}")
            error_message = str(e)
            success = False
            
            # Create error response
            from fastapi.responses import JSONResponse
            response = JSONResponse(
                status_code=500,
                content={
                    "error": "internal_error",
                    "message": "Internal server error"
                }
            )
        
        finally:
            # Calculate metrics
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            self.active_requests -= 1
            
            # Calculate response size
            response_size = self.calculate_response_size(response) if response else 0
            
            # Create request metrics
            metrics = RequestMetrics(
                timestamp=start_time,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code if response else 500,
                response_time_ms=response_time_ms,
                request_size_bytes=request_size,
                response_size_bytes=response_size,
                client_id=client_id,
                user_agent=user_agent,
                error_message=error_message
            )
            
            # Store metrics
            self.request_metrics.append(metrics)
            
            # Update performance stats
            self.update_performance_stats(endpoint_key, client_id, response_time_ms, success)
            
            # Add monitoring headers
            if response:
                response.headers["X-Response-Time"] = f"{response_time_ms:.2f}ms"
                response.headers["X-Request-ID"] = f"req_{int(start_time * 1000)}"
                
                # Log slow requests
                if response_time_ms > 5000:  # 5 seconds
                    self.logger.warning(
                        f"Slow request: {request.method} {request.url.path} "
                        f"took {response_time_ms:.2f}ms (client: {client_id})"
                    )
        
        return response
    
    def get_performance_summary(self, time_window_minutes: int = 60) -> Dict:
        """Get performance summary for the specified time window"""
        current_time = time.time()
        cutoff_time = current_time - (time_window_minutes * 60)
        
        # Filter recent requests
        recent_requests = [m for m in self.request_metrics if m.timestamp > cutoff_time]
        
        if not recent_requests:
            return {
                "time_window_minutes": time_window_minutes,
                "total_requests": 0,
                "metrics": {}
            }
        
        # Calculate summary metrics
        total_requests = len(recent_requests)
        successful_requests = len([r for r in recent_requests if 200 <= r.status_code < 400])
        failed_requests = total_requests - successful_requests
        
        response_times = [r.response_time_ms for r in recent_requests]
        avg_response_time = sum(response_times) / len(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        
        # Calculate percentiles
        sorted_times = sorted(response_times)
        p50_response_time = sorted_times[int(len(sorted_times) * 0.5)]
        p95_response_time = sorted_times[int(len(sorted_times) * 0.95)]
        p99_response_time = sorted_times[int(len(sorted_times) * 0.99)]
        
        # Calculate RPS
        time_span = max(r.timestamp for r in recent_requests) - min(r.timestamp for r in recent_requests)
        requests_per_second = total_requests / max(time_span, 1)
        
        # Unique clients
        unique_clients = len(set(r.client_id for r in recent_requests))
        
        # Error breakdown
        error_codes = {}
        for request in recent_requests:
            if request.status_code >= 400:
                code = str(request.status_code)
                error_codes[code] = error_codes.get(code, 0) + 1
        
        # Top endpoints by volume
        endpoint_counts = {}
        for request in recent_requests:
            endpoint = f"{request.method}:{request.path}"
            endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
        
        top_endpoints = sorted(endpoint_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "time_window_minutes": time_window_minutes,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "error_rate_percent": (failed_requests / total_requests * 100) if total_requests > 0 else 0,
            "requests_per_second": requests_per_second,
            "unique_clients": unique_clients,
            "response_time_ms": {
                "avg": avg_response_time,
                "min": min_response_time,
                "max": max_response_time,
                "p50": p50_response_time,
                "p95": p95_response_time,
                "p99": p99_response_time
            },
            "error_breakdown": error_codes,
            "top_endpoints": top_endpoints,
            "uptime_seconds": current_time - self.start_time,
            "total_requests_served": self.total_requests
        }
    
    def get_system_health(self) -> Dict:
        """Get current system health metrics"""
        if not self.system_metrics:
            return {"status": "no_data", "message": "No system metrics available"}
        
        latest_metrics = self.system_metrics[-1]
        
        # Determine health status
        status = "healthy"
        warnings = []
        
        if latest_metrics.cpu_usage_percent > 90:
            status = "critical"
            warnings.append(f"High CPU usage: {latest_metrics.cpu_usage_percent:.1f}%")
        elif latest_metrics.cpu_usage_percent > 70:
            status = "warning"
            warnings.append(f"Elevated CPU usage: {latest_metrics.cpu_usage_percent:.1f}%")
        
        if latest_metrics.memory_usage_percent > 90:
            status = "critical"
            warnings.append(f"High memory usage: {latest_metrics.memory_usage_percent:.1f}%")
        elif latest_metrics.memory_usage_percent > 80:
            status = "warning"
            warnings.append(f"Elevated memory usage: {latest_metrics.memory_usage_percent:.1f}%")
        
        if latest_metrics.gpu_usage_percent > 95:
            status = "critical"
            warnings.append(f"High GPU usage: {latest_metrics.gpu_usage_percent:.1f}%")
        
        return {
            "status": status,
            "warnings": warnings,
            "metrics": {
                "cpu_usage_percent": latest_metrics.cpu_usage_percent,
                "memory_usage_percent": latest_metrics.memory_usage_percent,
                "memory_used_gb": latest_metrics.memory_used_gb,
                "memory_available_gb": latest_metrics.memory_available_gb,
                "disk_usage_percent": latest_metrics.disk_usage_percent,
                "gpu_usage_percent": latest_metrics.gpu_usage_percent,
                "gpu_memory_used_gb": latest_metrics.gpu_memory_used_gb,
                "active_connections": self.active_requests,
                "timestamp": latest_metrics.timestamp
            }
        } 