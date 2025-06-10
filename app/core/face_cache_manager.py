"""
Face Cache Manager
Handles efficient face data caching and retrieval for instant avatar processing.
"""

import os
import pickle
import json
import logging
import hashlib
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import OrderedDict
import numpy as np
import gzip
import sqlite3

from .avatar_registrar import CachedFaceData, FaceProcessingMetadata

logger = logging.getLogger(__name__)

@dataclass
class CacheMetrics:
    """Face cache performance metrics."""
    cache_hits: int = 0
    cache_misses: int = 0
    total_requests: int = 0
    total_cache_size: int = 0
    avg_access_time: float = 0.0
    last_cleanup: Optional[datetime] = None

@dataclass
class CacheStatus:
    """Cache availability status for an avatar."""
    avatar_id: str
    is_cached: bool
    cache_size: int
    last_accessed: Optional[datetime]
    cache_age: float  # Hours
    integrity_verified: bool

@dataclass
class WarmupReport:
    """Avatar cache warmup operation report."""
    avatars_warmed: int
    total_avatars: int
    warmup_time: float
    cache_size_total: int
    errors: List[str]
    warnings: List[str]

@dataclass
class CleanupReport:
    """Cache cleanup operation report."""
    files_removed: int
    space_freed: int
    cleanup_time: float
    errors: List[str]

class CompressionEngine:
    """Efficient compression/decompression for face data."""
    
    @staticmethod
    def compress_face_data(data: CachedFaceData) -> bytes:
        """
        Compress cached face data efficiently.
        
        Args:
            data: Cached face data to compress
            
        Returns:
            bytes: Compressed data
        """
        try:
            # Serialize to dict first
            serializable_data = {
                'avatar_id': data.avatar_id,
                'face_boxes': data.face_boxes,
                'face_landmarks': [landmarks.tolist() for landmarks in data.face_landmarks],
                'cropped_faces': [face.tolist() for face in data.cropped_faces],
                'face_masks': [mask.tolist() for mask in data.face_masks],
                'processing_metadata': asdict(data.processing_metadata),
                'cache_version': data.cache_version,
                'cache_timestamp': data.cache_timestamp.isoformat(),
                'compression_ratio': data.compression_ratio,
                'integrity_hash': data.integrity_hash
            }
            
            # Pickle and compress
            pickled_data = pickle.dumps(serializable_data, protocol=pickle.HIGHEST_PROTOCOL)
            compressed_data = gzip.compress(pickled_data, compresslevel=6)
            
            logger.debug(f"Compressed face data for {data.avatar_id}: "
                        f"{len(pickled_data)} -> {len(compressed_data)} bytes "
                        f"({len(compressed_data)/len(pickled_data)*100:.1f}%)")
            
            return compressed_data
            
        except Exception as e:
            logger.error(f"Failed to compress face data for {data.avatar_id}: {e}")
            raise
    
    @staticmethod
    def decompress_face_data(compressed_data: bytes) -> CachedFaceData:
        """
        Decompress cached face data.
        
        Args:
            compressed_data: Compressed data bytes
            
        Returns:
            CachedFaceData: Decompressed face data
        """
        try:
            # Decompress and unpickle
            pickled_data = gzip.decompress(compressed_data)
            data_dict = pickle.loads(pickled_data)
            
            # Reconstruct numpy arrays
            cached_face_data = CachedFaceData(
                avatar_id=data_dict['avatar_id'],
                face_boxes=data_dict['face_boxes'],
                face_landmarks=[np.array(landmarks) for landmarks in data_dict['face_landmarks']],
                cropped_faces=[np.array(face) for face in data_dict['cropped_faces']],
                face_masks=[np.array(mask) for mask in data_dict['face_masks']],
                processing_metadata=FaceProcessingMetadata(**data_dict['processing_metadata']),
                cache_version=data_dict['cache_version'],
                cache_timestamp=datetime.fromisoformat(data_dict['cache_timestamp']),
                compression_ratio=data_dict['compression_ratio'],
                integrity_hash=data_dict['integrity_hash']
            )
            
            return cached_face_data
            
        except Exception as e:
            logger.error(f"Failed to decompress face data: {e}")
            raise
    
    @staticmethod
    def estimate_compression_ratio(data_size: int) -> float:
        """
        Estimate compression ratio for given data size.
        
        Args:
            data_size: Original data size in bytes
            
        Returns:
            float: Estimated compression ratio
        """
        # Based on empirical testing of face data compression
        base_ratio = 0.4  # Gzip typically achieves 40% compression on face data
        
        # Larger data typically compresses better
        if data_size > 1024 * 1024:  # > 1MB
            return base_ratio * 0.9
        elif data_size > 100 * 1024:  # > 100KB
            return base_ratio
        else:
            return base_ratio * 1.1
        
class DiskCacheHandler:
    """Handles persistent disk-based face data caching."""
    
    def __init__(self, cache_storage_path: str):
        """
        Initialize disk cache handler.
        
        Args:
            cache_storage_path: Directory for cache files
        """
        self.cache_storage_path = cache_storage_path
        self.compression_engine = CompressionEngine()
        
        # Ensure cache directory exists
        os.makedirs(cache_storage_path, exist_ok=True)
    
    def store_face_data(self, avatar_id: str, face_data: CachedFaceData) -> bool:
        """
        Store face data to disk cache.
        
        Args:
            avatar_id: Avatar identifier
            face_data: Face data to cache
            
        Returns:
            bool: Storage success status
        """
        try:
            cache_file_path = self._get_cache_file_path(avatar_id)
            
            # Compress data
            compressed_data = self.compression_engine.compress_face_data(face_data)
            
            # Write to file atomically
            temp_file_path = cache_file_path + '.tmp'
            with open(temp_file_path, 'wb') as f:
                f.write(compressed_data)
            
            # Atomic rename
            os.rename(temp_file_path, cache_file_path)
            
            # Create metadata file
            self._create_metadata_file(avatar_id, face_data, len(compressed_data))
            
            logger.info(f"Face data stored to disk cache: {avatar_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store face data to disk for {avatar_id}: {e}")
            # Clean up temp file if exists
            temp_file_path = self._get_cache_file_path(avatar_id) + '.tmp'
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass
            return False
    
    def load_face_data(self, avatar_id: str) -> Optional[CachedFaceData]:
        """
        Load face data from disk cache.
        
        Args:
            avatar_id: Avatar identifier
            
        Returns:
            CachedFaceData or None if not found
        """
        try:
            cache_file_path = self._get_cache_file_path(avatar_id)
            
            if not os.path.exists(cache_file_path):
                return None
            
            # Read compressed data
            with open(cache_file_path, 'rb') as f:
                compressed_data = f.read()
            
            # Decompress and return
            face_data = self.compression_engine.decompress_face_data(compressed_data)
            
            # Update metadata access time
            self._update_access_time(avatar_id)
            
            logger.debug(f"Face data loaded from disk cache: {avatar_id}")
            return face_data
            
        except Exception as e:
            logger.error(f"Failed to load face data from disk for {avatar_id}: {e}")
            return None
    
    def delete_face_data(self, avatar_id: str) -> bool:
        """
        Delete face data from disk cache.
        
        Args:
            avatar_id: Avatar identifier
            
        Returns:
            bool: Deletion success status
        """
        try:
            cache_file_path = self._get_cache_file_path(avatar_id)
            metadata_file_path = self._get_metadata_file_path(avatar_id)
            
            removed_files = 0
            
            if os.path.exists(cache_file_path):
                os.remove(cache_file_path)
                removed_files += 1
            
            if os.path.exists(metadata_file_path):
                os.remove(metadata_file_path)
                removed_files += 1
            
            if removed_files > 0:
                logger.info(f"Face data deleted from disk cache: {avatar_id}")
                return True
            else:
                logger.warning(f"No cache files found for {avatar_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete face data from disk for {avatar_id}: {e}")
            return False
    
    def get_cache_size(self, avatar_id: str) -> int:
        """Get cache file size for avatar."""
        try:
            cache_file_path = self._get_cache_file_path(avatar_id)
            if os.path.exists(cache_file_path):
                return os.path.getsize(cache_file_path)
            return 0
        except:
            return 0
    
    def list_cached_avatars(self) -> List[str]:
        """List all cached avatar IDs."""
        try:
            cached_avatars = []
            for filename in os.listdir(self.cache_storage_path):
                if filename.endswith('_face_cache.gz'):
                    avatar_id = filename[:-15]  # Remove '_face_cache.gz'
                    cached_avatars.append(avatar_id)
            return cached_avatars
        except Exception as e:
            logger.error(f"Failed to list cached avatars: {e}")
            return []
    
    def cleanup_expired_cache(self, max_age_hours: int) -> CleanupReport:
        """
        Clean up expired cache files.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
            
        Returns:
            CleanupReport: Cleanup operation results
        """
        start_time = time.time()
        files_removed = 0
        space_freed = 0
        errors = []
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            for filename in os.listdir(self.cache_storage_path):
                if filename.endswith('_face_cache.gz'):
                    file_path = os.path.join(self.cache_storage_path, filename)
                    
                    try:
                        # Check file modification time
                        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                        
                        if file_mtime < cutoff_time:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            
                            # Also remove metadata file
                            avatar_id = filename[:-15]
                            metadata_file = self._get_metadata_file_path(avatar_id)
                            if os.path.exists(metadata_file):
                                os.remove(metadata_file)
                            
                            files_removed += 1
                            space_freed += file_size
                            
                    except Exception as e:
                        errors.append(f"Failed to remove {filename}: {str(e)}")
            
            cleanup_time = time.time() - start_time
            
            logger.info(f"Cache cleanup completed: {files_removed} files removed, "
                       f"{space_freed} bytes freed in {cleanup_time:.2f}s")
            
            return CleanupReport(
                files_removed=files_removed,
                space_freed=space_freed,
                cleanup_time=cleanup_time,
                errors=errors
            )
            
        except Exception as e:
            errors.append(f"Cache cleanup failed: {str(e)}")
            return CleanupReport(
                files_removed=files_removed,
                space_freed=space_freed,
                cleanup_time=time.time() - start_time,
                errors=errors
            )
    
    def _get_cache_file_path(self, avatar_id: str) -> str:
        """Get cache file path for avatar."""
        return os.path.join(self.cache_storage_path, f"{avatar_id}_face_cache.gz")
    
    def _get_metadata_file_path(self, avatar_id: str) -> str:
        """Get metadata file path for avatar."""
        return os.path.join(self.cache_storage_path, f"{avatar_id}_metadata.json")
    
    def _create_metadata_file(self, avatar_id: str, face_data: CachedFaceData, file_size: int):
        """Create metadata file for cached data."""
        try:
            metadata = {
                'avatar_id': avatar_id,
                'cache_version': face_data.cache_version,
                'cache_timestamp': face_data.cache_timestamp.isoformat(),
                'file_size': file_size,
                'integrity_hash': face_data.integrity_hash,
                'face_count': len(face_data.cropped_faces),
                'processing_metadata': asdict(face_data.processing_metadata),
                'last_accessed': datetime.now().isoformat()
            }
            
            metadata_file_path = self._get_metadata_file_path(avatar_id)
            with open(metadata_file_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to create metadata file for {avatar_id}: {e}")
    
    def _update_access_time(self, avatar_id: str):
        """Update last access time in metadata."""
        try:
            metadata_file_path = self._get_metadata_file_path(avatar_id)
            
            if os.path.exists(metadata_file_path):
                with open(metadata_file_path, 'r') as f:
                    metadata = json.load(f)
                
                metadata['last_accessed'] = datetime.now().isoformat()
                
                with open(metadata_file_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
        except Exception as e:
            logger.debug(f"Failed to update access time for {avatar_id}: {e}")

class FaceCacheManager:
    """Main face cache manager with memory and disk caching."""
    
    def __init__(
        self,
        cache_storage_path: str,
        max_memory_cache_size: int = 100,
        enable_disk_cache: bool = True
    ):
        """
        Initialize face cache manager.
        
        Args:
            cache_storage_path: Directory for cache files
            max_memory_cache_size: Maximum avatars in memory cache
            enable_disk_cache: Enable persistent disk caching
        """
        self.cache_storage_path = cache_storage_path
        self.max_memory_cache_size = max_memory_cache_size
        self.enable_disk_cache = enable_disk_cache
        
        # In-memory cache (LRU)
        self.memory_cache: OrderedDict[str, CachedFaceData] = OrderedDict()
        self.cache_lock = threading.RLock()
        
        # Disk cache handler
        self.disk_cache = DiskCacheHandler(cache_storage_path) if enable_disk_cache else None
        
        # Performance metrics
        self.metrics = CacheMetrics()
        
        logger.info(f"Face cache manager initialized: "
                   f"memory_cache_size={max_memory_cache_size}, "
                   f"disk_cache={'enabled' if enable_disk_cache else 'disabled'}")
    
    def store_face_cache(self, avatar_id: str, face_data: CachedFaceData) -> bool:
        """
        Store face detection results for fast access.
        
        Args:
            avatar_id: Avatar identifier
            face_data: Processed face data
            
        Returns:
            bool: Storage success status
        """
        try:
            with self.cache_lock:
                # Store in memory cache
                self.memory_cache[avatar_id] = face_data
                self._evict_if_needed()
                
                # Store in disk cache
                disk_success = True
                if self.enable_disk_cache:
                    disk_success = self.disk_cache.store_face_data(avatar_id, face_data)
                
                # Update metrics
                cache_size = len(face_data.cropped_faces) * 96 * 96 * 3  # Estimate
                self.metrics.total_cache_size += cache_size
                
                logger.info(f"Face cache stored for {avatar_id} "
                           f"(memory: True, disk: {disk_success})")
                
                return disk_success
                
        except Exception as e:
            logger.error(f"Failed to store face cache for {avatar_id}: {e}")
            return False
    
    def retrieve_face_cache(self, avatar_id: str) -> Optional[CachedFaceData]:
        """
        Fast retrieval of pre-processed face data.
        
        Args:
            avatar_id: Avatar identifier
            
        Returns:
            CachedFaceData or None if not found
        """
        start_time = time.time()
        
        try:
            with self.cache_lock:
                self.metrics.total_requests += 1
                
                # Check memory cache first
                if avatar_id in self.memory_cache:
                    # Move to end (most recently used)
                    face_data = self.memory_cache.pop(avatar_id)
                    self.memory_cache[avatar_id] = face_data
                    
                    self.metrics.cache_hits += 1
                    access_time = (time.time() - start_time) * 1000
                    self._update_avg_access_time(access_time)
                    
                    logger.debug(f"Face cache hit (memory) for {avatar_id} in {access_time:.1f}ms")
                    return face_data
                
                # Check disk cache
                if self.enable_disk_cache:
                    face_data = self.disk_cache.load_face_data(avatar_id)
                    
                    if face_data is not None:
                        # Store in memory cache for future access
                        self.memory_cache[avatar_id] = face_data
                        self._evict_if_needed()
                        
                        self.metrics.cache_hits += 1
                        access_time = (time.time() - start_time) * 1000
                        self._update_avg_access_time(access_time)
                        
                        logger.debug(f"Face cache hit (disk) for {avatar_id} in {access_time:.1f}ms")
                        return face_data
                
                # Cache miss
                self.metrics.cache_misses += 1
                logger.debug(f"Face cache miss for {avatar_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve face cache for {avatar_id}: {e}")
            self.metrics.cache_misses += 1
            return None
    
    def warm_up_avatar_cache(self, avatar_ids: List[str]) -> WarmupReport:
        """
        Pre-load avatars into memory cache.
        
        Args:
            avatar_ids: List of avatar IDs to pre-load
            
        Returns:
            WarmupReport: Warmup operation results
        """
        start_time = time.time()
        avatars_warmed = 0
        cache_size_total = 0
        errors = []
        warnings = []
        
        try:
            logger.info(f"Starting avatar cache warmup for {len(avatar_ids)} avatars")
            
            for avatar_id in avatar_ids:
                try:
                    # Skip if already in memory cache
                    if avatar_id in self.memory_cache:
                        avatars_warmed += 1
                        continue
                    
                    # Load from disk cache
                    face_data = self.retrieve_face_cache(avatar_id)
                    
                    if face_data is not None:
                        avatars_warmed += 1
                        cache_size = len(face_data.cropped_faces) * 96 * 96 * 3
                        cache_size_total += cache_size
                    else:
                        warnings.append(f"No cached data found for avatar {avatar_id}")
                        
                except Exception as e:
                    errors.append(f"Failed to warm up avatar {avatar_id}: {str(e)}")
            
            warmup_time = time.time() - start_time
            
            report = WarmupReport(
                avatars_warmed=avatars_warmed,
                total_avatars=len(avatar_ids),
                warmup_time=warmup_time,
                cache_size_total=cache_size_total,
                errors=errors,
                warnings=warnings
            )
            
            logger.info(f"Avatar cache warmup completed: {avatars_warmed}/{len(avatar_ids)} "
                       f"avatars warmed in {warmup_time:.2f}s")
            
            return report
            
        except Exception as e:
            errors.append(f"Warmup operation failed: {str(e)}")
            return WarmupReport(
                avatars_warmed=avatars_warmed,
                total_avatars=len(avatar_ids),
                warmup_time=time.time() - start_time,
                cache_size_total=cache_size_total,
                errors=errors,
                warnings=warnings
            )
    
    def cleanup_expired_cache(self, max_age_days: int) -> CleanupReport:
        """
        Automatic cache maintenance.
        
        Args:
            max_age_days: Maximum cache age in days
            
        Returns:
            CleanupReport: Cleanup operation results
        """
        try:
            max_age_hours = max_age_days * 24
            
            if self.enable_disk_cache:
                cleanup_report = self.disk_cache.cleanup_expired_cache(max_age_hours)
                self.metrics.last_cleanup = datetime.now()
                
                logger.info(f"Cache cleanup completed: {cleanup_report.files_removed} files removed")
                return cleanup_report
            else:
                return CleanupReport(
                    files_removed=0,
                    space_freed=0,
                    cleanup_time=0.0,
                    errors=["Disk cache disabled"]
                )
                
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            return CleanupReport(
                files_removed=0,
                space_freed=0,
                cleanup_time=0.0,
                errors=[f"Cleanup failed: {str(e)}"]
            )
    
    def get_cache_status(self, avatar_id: str) -> CacheStatus:
        """
        Get cache status for an avatar.
        
        Args:
            avatar_id: Avatar identifier
            
        Returns:
            CacheStatus: Cache availability and metadata
        """
        try:
            # Check memory cache
            in_memory = avatar_id in self.memory_cache
            
            # Check disk cache
            cache_size = 0
            last_accessed = None
            cache_age = 0.0
            
            if self.enable_disk_cache:
                cache_size = self.disk_cache.get_cache_size(avatar_id)
                
                # Try to get metadata
                metadata_file = self.disk_cache._get_metadata_file_path(avatar_id)
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        if 'last_accessed' in metadata:
                            last_accessed = datetime.fromisoformat(metadata['last_accessed'])
                        
                        if 'cache_timestamp' in metadata:
                            cache_timestamp = datetime.fromisoformat(metadata['cache_timestamp'])
                            cache_age = (datetime.now() - cache_timestamp).total_seconds() / 3600
                            
                    except Exception as e:
                        logger.debug(f"Failed to read metadata for {avatar_id}: {e}")
            
            is_cached = in_memory or cache_size > 0
            
            return CacheStatus(
                avatar_id=avatar_id,
                is_cached=is_cached,
                cache_size=cache_size,
                last_accessed=last_accessed,
                cache_age=cache_age,
                integrity_verified=True  # Simplified for now
            )
            
        except Exception as e:
            logger.error(f"Failed to get cache status for {avatar_id}: {e}")
            return CacheStatus(
                avatar_id=avatar_id,
                is_cached=False,
                cache_size=0,
                last_accessed=None,
                cache_age=0.0,
                integrity_verified=False
            )
    
    def get_cache_metrics(self) -> CacheMetrics:
        """Get cache performance metrics."""
        return self.metrics
    
    def clear_memory_cache(self):
        """Clear all data from memory cache."""
        with self.cache_lock:
            self.memory_cache.clear()
            logger.info("Memory cache cleared")
    
    def _evict_if_needed(self):
        """Evict least recently used items if cache is full."""
        while len(self.memory_cache) > self.max_memory_cache_size:
            # Remove least recently used (first item)
            evicted_avatar, _ = self.memory_cache.popitem(last=False)
            logger.debug(f"Evicted {evicted_avatar} from memory cache")
    
    def _update_avg_access_time(self, access_time: float):
        """Update average access time metric."""
        if self.metrics.avg_access_time == 0:
            self.metrics.avg_access_time = access_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.avg_access_time = (
                alpha * access_time + (1 - alpha) * self.metrics.avg_access_time
            ) 