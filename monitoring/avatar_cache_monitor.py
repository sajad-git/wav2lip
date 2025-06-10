"""
Avatar Cache Monitor for Avatar Streaming Service
Monitors avatar cache performance, hit rates, usage patterns, and cache health.
"""

import time
import threading
import os
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import asyncio

from app.core.face_cache_manager import FaceCacheManager
from monitoring.metrics_collector import MetricsCollector, MetricType

logger = logging.getLogger(__name__)


@dataclass
class AvatarUsageMetrics:
    """Avatar usage metrics"""
    avatar_id: str
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    total_access_time: float = 0.0
    average_access_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    processing_requests: int = 0
    quality_score: float = 0.0
    file_size: int = 0
    registration_date: Optional[datetime] = None
    user_id: Optional[str] = None


@dataclass
class CachePerformanceMetrics:
    """Cache system performance metrics"""
    total_hits: int = 0
    total_misses: int = 0
    hit_rate: float = 0.0
    average_access_time: float = 0.0
    cache_size_entries: int = 0
    cache_size_bytes: int = 0
    memory_usage_mb: float = 0.0
    evictions: int = 0
    last_cleanup: Optional[datetime] = None
    storage_health: str = "unknown"


@dataclass
class AvatarCacheHealth:
    """Avatar cache health assessment"""
    status: str  # "healthy", "degraded", "error"
    health_score: float  # 0.0 to 1.0
    last_check: datetime
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    performance_trend: str = "stable"  # "improving", "stable", "degrading"


class AvatarCacheMonitor:
    """Comprehensive avatar cache monitoring system"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.face_cache_manager: Optional[FaceCacheManager] = None
        
        # Usage tracking
        self.avatar_metrics: Dict[str, AvatarUsageMetrics] = {}
        self.cache_performance = CachePerformanceMetrics()
        self.health_history: deque = deque(maxlen=1440)  # 24 hours of health checks
        
        # Monitoring configuration
        self.monitoring_active = False
        self.health_check_interval = 300  # 5 minutes
        self.usage_tracking_enabled = True
        
        # Performance thresholds
        self.performance_thresholds = {
            "min_hit_rate": 0.7,
            "max_access_time": 50.0,  # ms
            "max_memory_usage": 2048.0,  # MB
            "max_cache_size": 1000  # entries
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Usage patterns analysis
        self.usage_patterns = {
            "popular_avatars": [],
            "frequent_users": defaultdict(int),
            "peak_usage_hours": defaultdict(int),
            "usage_trends": deque(maxlen=168)  # 1 week of hourly data
        }
        
    def initialize(self, face_cache_manager: FaceCacheManager):
        """Initialize monitor with face cache manager"""
        self.face_cache_manager = face_cache_manager
        self._load_existing_avatar_metrics()
        logger.info("Avatar cache monitor initialized")
    
    def start_monitoring(self):
        """Start continuous cache monitoring"""
        self.monitoring_active = True
        self._start_health_monitoring()
        self._start_usage_pattern_analysis()
        logger.info("Avatar cache monitoring started")
    
    def stop_monitoring(self):
        """Stop cache monitoring"""
        self.monitoring_active = False
        logger.info("Avatar cache monitoring stopped")
    
    def record_avatar_access(self, avatar_id: str, access_time_ms: float, 
                           cache_hit: bool, user_id: str = None):
        """Record avatar cache access"""
        with self.lock:
            current_time = datetime.utcnow()
            
            # Initialize avatar metrics if not exists
            if avatar_id not in self.avatar_metrics:
                self.avatar_metrics[avatar_id] = AvatarUsageMetrics(
                    avatar_id=avatar_id,
                    user_id=user_id
                )
            
            metrics = self.avatar_metrics[avatar_id]
            
            # Update avatar-specific metrics
            metrics.access_count += 1
            metrics.last_accessed = current_time
            metrics.total_access_time += access_time_ms
            metrics.average_access_time = metrics.total_access_time / metrics.access_count
            
            if cache_hit:
                metrics.cache_hits += 1
                self.cache_performance.total_hits += 1
            else:
                metrics.cache_misses += 1
                self.cache_performance.total_misses += 1
            
            # Update overall cache performance
            total_accesses = self.cache_performance.total_hits + self.cache_performance.total_misses
            if total_accesses > 0:
                self.cache_performance.hit_rate = self.cache_performance.total_hits / total_accesses
            
            # Update average access time
            total_time = sum(m.total_access_time for m in self.avatar_metrics.values())
            total_accesses_all = sum(m.access_count for m in self.avatar_metrics.values())
            if total_accesses_all > 0:
                self.cache_performance.average_access_time = total_time / total_accesses_all
            
            # Record in metrics collector
            self.metrics_collector.record_avatar_cache_access(
                "cache_access", avatar_id, access_time_ms, cache_hit
            )
            
            # Track usage patterns
            self._update_usage_patterns(avatar_id, user_id, current_time)
    
    def record_avatar_registration(self, avatar_id: str, file_size: int, 
                                 quality_score: float, user_id: str = None):
        """Record new avatar registration"""
        with self.lock:
            if avatar_id not in self.avatar_metrics:
                self.avatar_metrics[avatar_id] = AvatarUsageMetrics(avatar_id=avatar_id)
            
            metrics = self.avatar_metrics[avatar_id]
            metrics.file_size = file_size
            metrics.quality_score = quality_score
            metrics.registration_date = datetime.utcnow()
            metrics.user_id = user_id
            
            # Record registration metrics
            tags = {"avatar_id": avatar_id, "user_id": user_id or "unknown"}
            self.metrics_collector.record_metric("avatar_registered", 1, tags, MetricType.COUNTER)
            self.metrics_collector.record_metric("avatar_file_size", file_size, tags, MetricType.GAUGE)
            self.metrics_collector.record_metric("avatar_quality_score", quality_score, tags, MetricType.GAUGE)
    
    def record_cache_eviction(self, avatar_id: str, reason: str = "lru"):
        """Record cache eviction event"""
        with self.lock:
            self.cache_performance.evictions += 1
            
            # Record eviction metrics
            tags = {"avatar_id": avatar_id, "reason": reason}
            self.metrics_collector.record_metric("cache_evictions", 1, tags, MetricType.COUNTER)
            
            logger.info(f"Cache eviction recorded for avatar {avatar_id}, reason: {reason}")
    
    def update_cache_system_metrics(self, cache_size_entries: int, cache_size_bytes: int, 
                                  memory_usage_mb: float, storage_health: str = "healthy"):
        """Update overall cache system metrics"""
        with self.lock:
            self.cache_performance.cache_size_entries = cache_size_entries
            self.cache_performance.cache_size_bytes = cache_size_bytes
            self.cache_performance.memory_usage_mb = memory_usage_mb
            self.cache_performance.storage_health = storage_health
            
            # Record system metrics
            self.metrics_collector.record_metric("cache_size_entries", cache_size_entries)
            self.metrics_collector.record_metric("cache_size_bytes", cache_size_bytes)
            self.metrics_collector.record_metric("cache_memory_usage_mb", memory_usage_mb)
            self.metrics_collector.record_metric("cache_hit_rate", self.cache_performance.hit_rate)
    
    def assess_cache_health(self) -> AvatarCacheHealth:
        """Assess overall cache health"""
        with self.lock:
            current_time = datetime.utcnow()
            
            # Calculate health score
            health_score = self._calculate_cache_health_score()
            
            # Determine status
            status = self._determine_cache_health_status(health_score)
            
            # Analyze performance trend
            trend = self._analyze_cache_performance_trend()
            
            # Generate alerts and recommendations
            alerts = self._generate_cache_alerts()
            recommendations = self._generate_cache_recommendations()
            
            health = AvatarCacheHealth(
                status=status,
                health_score=health_score,
                last_check=current_time,
                alerts=alerts,
                recommendations=recommendations,
                performance_trend=trend
            )
            
            return health
    
    def get_avatar_usage_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get avatar usage summary"""
        with self.lock:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            # Filter recent activity
            recent_avatars = {
                avatar_id: metrics for avatar_id, metrics in self.avatar_metrics.items()
                if metrics.last_accessed and metrics.last_accessed >= cutoff_time
            }
            
            # Calculate summary statistics
            total_avatars = len(self.avatar_metrics)
            active_avatars = len(recent_avatars)
            
            # Most popular avatars
            popular_avatars = sorted(
                self.avatar_metrics.items(),
                key=lambda x: x[1].access_count,
                reverse=True
            )[:10]
            
            # Performance statistics
            if recent_avatars:
                avg_access_time = sum(m.average_access_time for m in recent_avatars.values()) / len(recent_avatars)
                total_accesses = sum(m.access_count for m in recent_avatars.values())
            else:
                avg_access_time = 0.0
                total_accesses = 0
            
            summary = {
                "time_window_hours": time_window_hours,
                "total_registered_avatars": total_avatars,
                "active_avatars": active_avatars,
                "total_accesses": total_accesses,
                "average_access_time_ms": avg_access_time,
                "cache_hit_rate": self.cache_performance.hit_rate,
                "most_popular_avatars": [
                    {
                        "avatar_id": avatar_id,
                        "access_count": metrics.access_count,
                        "hit_rate": metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses) 
                                   if (metrics.cache_hits + metrics.cache_misses) > 0 else 0,
                        "avg_access_time": metrics.average_access_time,
                        "quality_score": metrics.quality_score
                    }
                    for avatar_id, metrics in popular_avatars
                ],
                "usage_patterns": self._get_usage_patterns_summary(),
                "cache_performance": {
                    "hit_rate": self.cache_performance.hit_rate,
                    "average_access_time": self.cache_performance.average_access_time,
                    "cache_size_entries": self.cache_performance.cache_size_entries,
                    "memory_usage_mb": self.cache_performance.memory_usage_mb,
                    "evictions": self.cache_performance.evictions
                }
            }
            
            return summary
    
    def get_user_avatar_usage(self, user_id: str) -> Dict[str, Any]:
        """Get avatar usage for specific user"""
        with self.lock:
            user_avatars = {
                avatar_id: metrics for avatar_id, metrics in self.avatar_metrics.items()
                if metrics.user_id == user_id
            }
            
            if not user_avatars:
                return {"user_id": user_id, "avatars": [], "total_usage": 0}
            
            total_usage = sum(metrics.access_count for metrics in user_avatars.values())
            avg_quality = sum(metrics.quality_score for metrics in user_avatars.values()) / len(user_avatars)
            
            return {
                "user_id": user_id,
                "total_avatars": len(user_avatars),
                "total_usage": total_usage,
                "average_quality_score": avg_quality,
                "avatars": [
                    {
                        "avatar_id": avatar_id,
                        "access_count": metrics.access_count,
                        "last_accessed": metrics.last_accessed.isoformat() if metrics.last_accessed else None,
                        "quality_score": metrics.quality_score,
                        "file_size": metrics.file_size,
                        "hit_rate": metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses)
                                   if (metrics.cache_hits + metrics.cache_misses) > 0 else 0
                    }
                    for avatar_id, metrics in user_avatars.items()
                ]
            }
    
    def cleanup_stale_metrics(self, days_threshold: int = 30):
        """Clean up metrics for avatars not accessed recently"""
        with self.lock:
            cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
            
            stale_avatars = [
                avatar_id for avatar_id, metrics in self.avatar_metrics.items()
                if not metrics.last_accessed or metrics.last_accessed < cutoff_date
            ]
            
            for avatar_id in stale_avatars:
                del self.avatar_metrics[avatar_id]
                logger.info(f"Cleaned up stale metrics for avatar {avatar_id}")
            
            return len(stale_avatars)
    
    def _calculate_cache_health_score(self) -> float:
        """Calculate overall cache health score"""
        score_components = []
        
        # Hit rate component (40% weight)
        hit_rate_score = min(1.0, self.cache_performance.hit_rate / 0.9)  # Target 90% hit rate
        score_components.append(("hit_rate", hit_rate_score, 0.4))
        
        # Access time component (30% weight)
        if self.cache_performance.average_access_time > 0:
            access_time_score = max(0.0, 1.0 - (self.cache_performance.average_access_time / 100.0))
        else:
            access_time_score = 1.0
        score_components.append(("access_time", access_time_score, 0.3))
        
        # Memory usage component (20% weight)
        memory_usage_score = max(0.0, 1.0 - (self.cache_performance.memory_usage_mb / 2048.0))
        score_components.append(("memory_usage", memory_usage_score, 0.2))
        
        # Storage health component (10% weight)
        storage_score = 1.0 if self.cache_performance.storage_health == "healthy" else 0.5
        score_components.append(("storage_health", storage_score, 0.1))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in score_components)
        
        return max(0.0, min(1.0, total_score))
    
    def _determine_cache_health_status(self, health_score: float) -> str:
        """Determine cache health status"""
        if health_score >= 0.8 and self.cache_performance.hit_rate >= 0.7:
            return "healthy"
        elif health_score >= 0.6 and self.cache_performance.hit_rate >= 0.5:
            return "degraded"
        else:
            return "error"
    
    def _analyze_cache_performance_trend(self) -> str:
        """Analyze cache performance trend"""
        if len(self.health_history) < 10:
            return "insufficient_data"
        
        # Get recent health scores
        recent_scores = [entry.get("health_score", 0.0) for entry in list(self.health_history)[-10:]]
        
        # Simple trend analysis
        if len(recent_scores) >= 2:
            trend_slope = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
            if trend_slope > 0.05:
                return "improving"
            elif trend_slope < -0.05:
                return "degrading"
        
        return "stable"
    
    def _generate_cache_alerts(self) -> List[str]:
        """Generate cache performance alerts"""
        alerts = []
        
        # Hit rate alerts
        if self.cache_performance.hit_rate < self.performance_thresholds["min_hit_rate"]:
            alerts.append(f"Low cache hit rate: {self.cache_performance.hit_rate:.2%}")
        
        # Access time alerts
        if self.cache_performance.average_access_time > self.performance_thresholds["max_access_time"]:
            alerts.append(f"High average access time: {self.cache_performance.average_access_time:.2f}ms")
        
        # Memory usage alerts
        if self.cache_performance.memory_usage_mb > self.performance_thresholds["max_memory_usage"]:
            alerts.append(f"High memory usage: {self.cache_performance.memory_usage_mb:.2f}MB")
        
        # Cache size alerts
        if self.cache_performance.cache_size_entries > self.performance_thresholds["max_cache_size"]:
            alerts.append(f"Large cache size: {self.cache_performance.cache_size_entries} entries")
        
        # Storage health alerts
        if self.cache_performance.storage_health != "healthy":
            alerts.append(f"Storage health issue: {self.cache_performance.storage_health}")
        
        return alerts
    
    def _generate_cache_recommendations(self) -> List[str]:
        """Generate cache optimization recommendations"""
        recommendations = []
        
        # Hit rate recommendations
        if self.cache_performance.hit_rate < 0.8:
            recommendations.append("Consider increasing cache size or improving eviction policy")
            recommendations.append("Analyze avatar access patterns for optimization")
        
        # Performance recommendations
        if self.cache_performance.average_access_time > 30.0:
            recommendations.append("Optimize cache data structure or storage mechanism")
            recommendations.append("Consider cache warming for frequently accessed avatars")
        
        # Memory recommendations
        if self.cache_performance.memory_usage_mb > 1500.0:
            recommendations.append("Consider implementing cache compression")
            recommendations.append("Review cache size limits and eviction policies")
        
        # Eviction recommendations
        if self.cache_performance.evictions > 100:
            recommendations.append("Review cache eviction patterns and policies")
            recommendations.append("Consider increasing cache capacity")
        
        return recommendations
    
    def _update_usage_patterns(self, avatar_id: str, user_id: str, access_time: datetime):
        """Update usage pattern analytics"""
        # Track hourly usage
        hour_key = access_time.hour
        self.usage_patterns["peak_usage_hours"][hour_key] += 1
        
        # Track user frequency
        if user_id:
            self.usage_patterns["frequent_users"][user_id] += 1
        
        # Update hourly trend data
        current_hour = access_time.replace(minute=0, second=0, microsecond=0)
        self.usage_patterns["usage_trends"].append({
            "timestamp": current_hour,
            "avatar_id": avatar_id,
            "user_id": user_id
        })
    
    def _get_usage_patterns_summary(self) -> Dict[str, Any]:
        """Get usage patterns summary"""
        # Peak hours analysis
        peak_hours = sorted(
            self.usage_patterns["peak_usage_hours"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Top users analysis
        top_users = sorted(
            self.usage_patterns["frequent_users"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "peak_usage_hours": [{"hour": hour, "usage_count": count} for hour, count in peak_hours],
            "most_active_users": [{"user_id": user_id, "usage_count": count} for user_id, count in top_users],
            "total_unique_users": len(self.usage_patterns["frequent_users"]),
            "hourly_trend_data_points": len(self.usage_patterns["usage_trends"])
        }
    
    def _load_existing_avatar_metrics(self):
        """Load existing avatar metrics if available"""
        # This would load from persistent storage if implemented
        # For now, starting with empty metrics
        logger.info("Avatar metrics initialized (no persistent storage configured)")
    
    def _start_health_monitoring(self):
        """Start background health monitoring"""
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Assess cache health
                    health = self.assess_cache_health()
                    
                    # Store health history
                    health_entry = {
                        "timestamp": datetime.utcnow(),
                        "health_score": health.health_score,
                        "status": health.status,
                        "hit_rate": self.cache_performance.hit_rate,
                        "memory_usage": self.cache_performance.memory_usage_mb
                    }
                    self.health_history.append(health_entry)
                    
                    # Log critical issues
                    if health.status == "error":
                        logger.warning(f"Cache health critical: {health.alerts}")
                    elif health.alerts:
                        logger.info(f"Cache health alerts: {health.alerts}")
                    
                    time.sleep(self.health_check_interval)
                    
                except Exception as e:
                    logger.error(f"Cache health monitoring error: {e}")
                    time.sleep(self.health_check_interval)
        
        thread = threading.Thread(target=monitoring_loop, daemon=True)
        thread.start()
        logger.info("Cache health monitoring thread started")
    
    def _start_usage_pattern_analysis(self):
        """Start background usage pattern analysis"""
        def analysis_loop():
            while self.monitoring_active:
                try:
                    # Update popular avatars list
                    popular = sorted(
                        self.avatar_metrics.items(),
                        key=lambda x: x[1].access_count,
                        reverse=True
                    )[:20]
                    
                    self.usage_patterns["popular_avatars"] = [
                        {"avatar_id": avatar_id, "access_count": metrics.access_count}
                        for avatar_id, metrics in popular
                    ]
                    
                    # Cleanup old usage trend data
                    cutoff_time = datetime.utcnow() - timedelta(days=7)
                    self.usage_patterns["usage_trends"] = deque([
                        entry for entry in self.usage_patterns["usage_trends"]
                        if entry["timestamp"] >= cutoff_time
                    ], maxlen=168)
                    
                    time.sleep(3600)  # Run every hour
                    
                except Exception as e:
                    logger.error(f"Usage pattern analysis error: {e}")
                    time.sleep(3600)
        
        thread = threading.Thread(target=analysis_loop, daemon=True)
        thread.start()
        logger.info("Usage pattern analysis thread started")
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export cache metrics"""
        with self.lock:
            if format_type == "json":
                export_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "cache_health": self.assess_cache_health().__dict__,
                    "usage_summary": self.get_avatar_usage_summary(),
                    "cache_performance": {
                        "hit_rate": self.cache_performance.hit_rate,
                        "average_access_time": self.cache_performance.average_access_time,
                        "cache_size_entries": self.cache_performance.cache_size_entries,
                        "memory_usage_mb": self.cache_performance.memory_usage_mb,
                        "evictions": self.cache_performance.evictions
                    },
                    "usage_patterns": self.usage_patterns
                }
                return json.dumps(export_data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")


# Global avatar cache monitor instance
avatar_cache_monitor: Optional[AvatarCacheMonitor] = None


def initialize_avatar_cache_monitor(metrics_collector: MetricsCollector) -> AvatarCacheMonitor:
    """Initialize global avatar cache monitor"""
    global avatar_cache_monitor
    avatar_cache_monitor = AvatarCacheMonitor(metrics_collector)
    logger.info("Avatar cache monitor initialized")
    return avatar_cache_monitor


def get_avatar_cache_monitor() -> Optional[AvatarCacheMonitor]:
    """Get global avatar cache monitor instance"""
    return avatar_cache_monitor 