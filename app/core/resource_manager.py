"""
GPU resource allocation for concurrent users + avatar cache management
"""
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
from queue import PriorityQueue
import threading
import psutil

from app.core.face_cache_manager import FaceCacheManager
from app.models.chunk_models import ProcessingTask
from app.models.avatar_models import CachedFaceData


@dataclass
class ResourceQuota:
    """Resource allocation quota for a user"""
    max_concurrent_chunks: int = 3
    max_gpu_memory_mb: int = 2048
    max_processing_time_per_chunk: float = 5.0
    priority_level: int = 1


@dataclass
class SessionMetrics:
    """User session usage metrics"""
    chunks_processed: int = 0
    total_processing_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0
    last_activity: datetime = None


class UserSession:
    """User session with resource allocation"""
    
    def __init__(self, user_id: str, resource_quota: ResourceQuota):
        self.user_id = user_id
        self.resource_quota = resource_quota
        self.allowed_avatars: Set[str] = set()
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.metrics = SessionMetrics()
        self.last_activity = datetime.now()
        self.active_tasks: List[ProcessingTask] = []
        
    def add_avatar_access(self, avatar_id: str):
        """Add avatar access permission"""
        self.allowed_avatars.add(avatar_id)
        
    def has_avatar_access(self, avatar_id: str) -> bool:
        """Check if user has access to avatar"""
        return avatar_id in self.allowed_avatars or len(self.allowed_avatars) == 0  # Empty means all access
        
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
        self.metrics.last_activity = self.last_activity


class GPUResourceManager:
    """GPU resource allocation for concurrent users + avatar cache management"""
    
    def __init__(self, model_instances: Dict[str, Any], avatar_cache: FaceCacheManager):
        self.logger = logging.getLogger(__name__)
        self.model_instances = model_instances
        self.avatar_cache = avatar_cache
        self.user_sessions: Dict[str, UserSession] = {}
        self.resource_allocation: Dict[str, ResourceQuota] = {}
        self.processing_queue = PriorityQueue()
        self.active_processing: Dict[str, List[ProcessingTask]] = {}
        self.resource_lock = threading.Lock()
        
        # Resource limits
        self.max_concurrent_users = 3
        self.max_total_gpu_memory_mb = 18000  # Leave 2GB buffer on RTX 4090
        self.max_concurrent_chunks = 9  # 3 chunks per user max
        
        # Performance tracking
        self.total_chunks_processed = 0
        self.total_processing_time = 0.0
        self.start_time = time.time()
        
        self.logger.info("🔧 GPU Resource Manager initialized")
    
    async def allocate_user_session(self, user_id: str, priority: int = 1) -> UserSession:
        """Create user session with resource allocation"""
        try:
            with self.resource_lock:
                if user_id in self.user_sessions:
                    session = self.user_sessions[user_id]
                    session.update_activity()
                    return session
                
                # Check if we can accept new user
                if len(self.user_sessions) >= self.max_concurrent_users:
                    # Try to cleanup inactive sessions
                    await self._cleanup_inactive_sessions()
                    
                    if len(self.user_sessions) >= self.max_concurrent_users:
                        raise RuntimeError(f"Maximum concurrent users ({self.max_concurrent_users}) reached")
                
                # Create resource quota based on priority
                quota = self._calculate_resource_quota(priority, len(self.user_sessions))
                
                # Create new session
                session = UserSession(user_id, quota)
                self.user_sessions[user_id] = session
                self.resource_allocation[user_id] = quota
                self.active_processing[user_id] = []
                
                self.logger.info(f"✅ User session allocated for {user_id} with quota: {quota}")
                return session
                
        except Exception as e:
            self.logger.error(f"❌ Failed to allocate user session for {user_id}: {str(e)}")
            raise
    
    def get_shared_model_instance(self, model_name: str, user_id: str) -> Any:
        """Provide model access with resource tracking"""
        try:
            # Validate user session exists
            if user_id not in self.user_sessions:
                raise ValueError(f"No active session for user {user_id}")
            
            session = self.user_sessions[user_id]
            session.update_activity()
            
            # Check if user has available resources
            current_tasks = len(self.active_processing.get(user_id, []))
            if current_tasks >= session.resource_quota.max_concurrent_chunks:
                raise RuntimeError(f"User {user_id} has reached concurrent processing limit")
            
            # Return cached model instance
            if model_name not in self.model_instances:
                raise ValueError(f"Model {model_name} not available")
            
            model_instance = self.model_instances[model_name]
            self.logger.debug(f"🔗 Model {model_name} accessed by user {user_id}")
            
            return model_instance
            
        except Exception as e:
            self.logger.error(f"❌ Failed to provide model access for {user_id}: {str(e)}")
            raise
    
    async def get_shared_avatar_cache(self, avatar_id: str, user_id: str) -> Optional[CachedFaceData]:
        """Provide avatar face data with usage tracking"""
        try:
            # Validate user session and avatar access
            if user_id not in self.user_sessions:
                raise ValueError(f"No active session for user {user_id}")
            
            session = self.user_sessions[user_id]
            
            if not session.has_avatar_access(avatar_id):
                raise PermissionError(f"User {user_id} does not have access to avatar {avatar_id}")
            
            session.update_activity()
            
            # Retrieve cached face data
            face_data = await self.avatar_cache.retrieve_face_cache(avatar_id)
            
            if face_data:
                session.metrics.cache_hits += 1
                self.logger.debug(f"🎯 Avatar cache hit for {avatar_id} by user {user_id}")
            else:
                session.metrics.cache_misses += 1
                self.logger.warning(f"💔 Avatar cache miss for {avatar_id} by user {user_id}")
            
            return face_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to provide avatar cache for {user_id}: {str(e)}")
            if user_id in self.user_sessions:
                self.user_sessions[user_id].metrics.errors += 1
            raise
    
    async def manage_concurrent_processing(self, tasks: List[ProcessingTask]) -> None:
        """Coordinate GPU usage across users with shared resources"""
        try:
            # Sort tasks by priority and user quotas
            sorted_tasks = self._prioritize_tasks(tasks)
            
            # Process tasks in batches to prevent GPU overload
            batch_size = min(self.max_concurrent_chunks, len(sorted_tasks))
            
            for i in range(0, len(sorted_tasks), batch_size):
                batch = sorted_tasks[i:i + batch_size]
                
                # Check resource availability
                if not self._check_resource_availability(batch):
                    # Apply backpressure - delay processing
                    await asyncio.sleep(0.1)
                    continue
                
                # Process batch concurrently
                await self._process_task_batch(batch)
                
                # Monitor GPU utilization
                await self._monitor_gpu_utilization()
                
        except Exception as e:
            self.logger.error(f"❌ Concurrent processing management failed: {str(e)}")
            raise
    
    async def cleanup_user_session(self, user_id: str) -> None:
        """Clean up user session and release resources"""
        try:
            with self.resource_lock:
                if user_id in self.user_sessions:
                    session = self.user_sessions[user_id]
                    
                    # Log session metrics
                    self.logger.info(
                        f"📊 Session metrics for {user_id}: "
                        f"chunks={session.metrics.chunks_processed}, "
                        f"time={session.metrics.total_processing_time:.2f}s, "
                        f"cache_hits={session.metrics.cache_hits}, "
                        f"errors={session.metrics.errors}"
                    )
                    
                    # Remove session
                    del self.user_sessions[user_id]
                    del self.resource_allocation[user_id]
                    if user_id in self.active_processing:
                        del self.active_processing[user_id]
                    
                    self.logger.info(f"🧹 User session cleaned up for {user_id}")
                    
        except Exception as e:
            self.logger.error(f"❌ Failed to cleanup session for {user_id}: {str(e)}")
    
    async def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource allocation status"""
        try:
            gpu_memory = self._get_gpu_memory_usage()
            system_memory = psutil.virtual_memory()
            
            return {
                "active_users": len(self.user_sessions),
                "max_users": self.max_concurrent_users,
                "active_chunks": sum(len(tasks) for tasks in self.active_processing.values()),
                "max_chunks": self.max_concurrent_chunks,
                "gpu_memory_used_mb": gpu_memory.get("used_mb", 0),
                "gpu_memory_total_mb": gpu_memory.get("total_mb", 0),
                "system_memory_percent": system_memory.percent,
                "total_chunks_processed": self.total_chunks_processed,
                "average_processing_time": self._get_average_processing_time(),
                "uptime_seconds": time.time() - self.start_time
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get resource status: {str(e)}")
            return {}
    
    def _calculate_resource_quota(self, priority: int, current_users: int) -> ResourceQuota:
        """Calculate resource quota based on priority and current load"""
        base_quota = ResourceQuota()
        
        # Adjust based on current load
        load_factor = max(0.5, 1.0 - (current_users * 0.2))
        
        quota = ResourceQuota(
            max_concurrent_chunks=max(1, int(base_quota.max_concurrent_chunks * load_factor)),
            max_gpu_memory_mb=int(base_quota.max_gpu_memory_mb * load_factor),
            max_processing_time_per_chunk=base_quota.max_processing_time_per_chunk,
            priority_level=priority
        )
        
        return quota
    
    async def _cleanup_inactive_sessions(self) -> None:
        """Clean up inactive user sessions"""
        current_time = datetime.now()
        inactive_users = []
        
        for user_id, session in self.user_sessions.items():
            if session.last_activity:
                inactive_time = (current_time - session.last_activity).total_seconds()
                if inactive_time > 300:  # 5 minutes
                    inactive_users.append(user_id)
        
        for user_id in inactive_users:
            await self.cleanup_user_session(user_id)
            self.logger.info(f"🧹 Cleaned up inactive session for {user_id}")
    
    def _prioritize_tasks(self, tasks: List[ProcessingTask]) -> List[ProcessingTask]:
        """Sort tasks by priority and user quotas"""
        return sorted(tasks, key=lambda t: (
            -t.priority,  # Higher priority first
            t.created_at.timestamp()  # Earlier tasks first
        ))
    
    def _check_resource_availability(self, tasks: List[ProcessingTask]) -> bool:
        """Check if resources are available for task batch"""
        # Check concurrent chunk limit
        total_active = sum(len(tasks) for tasks in self.active_processing.values())
        if total_active + len(tasks) > self.max_concurrent_chunks:
            return False
        
        # Check GPU memory
        gpu_memory = self._get_gpu_memory_usage()
        if gpu_memory.get("used_mb", 0) > self.max_total_gpu_memory_mb * 0.9:
            return False
        
        return True
    
    async def _process_task_batch(self, tasks: List[ProcessingTask]) -> None:
        """Process a batch of tasks"""
        # This would be implemented to coordinate with the actual processing services
        # For now, just track the tasks
        for task in tasks:
            user_id = task.user_id
            if user_id in self.active_processing:
                self.active_processing[user_id].append(task)
    
    async def _monitor_gpu_utilization(self) -> None:
        """Monitor GPU utilization and apply throttling if needed"""
        gpu_memory = self._get_gpu_memory_usage()
        utilization = gpu_memory.get("used_mb", 0) / gpu_memory.get("total_mb", 1)
        
        if utilization > 0.95:
            # High utilization - apply backpressure
            await asyncio.sleep(0.2)
            self.logger.warning(f"⚠️ High GPU utilization: {utilization:.1%}")
    
    def _get_gpu_memory_usage(self) -> Dict[str, int]:
        """Get current GPU memory usage"""
        try:
            # This would integrate with actual GPU monitoring
            # For now, return mock data
            return {
                "used_mb": 8000,  # Mock value
                "total_mb": 24000  # RTX 4090
            }
        except Exception:
            return {"used_mb": 0, "total_mb": 24000}
    
    def _get_average_processing_time(self) -> float:
        """Get average processing time per chunk"""
        if self.total_chunks_processed == 0:
            return 0.0
        return self.total_processing_time / self.total_chunks_processed 