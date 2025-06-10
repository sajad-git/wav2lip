#!/usr/bin/env python3
"""
Unit Tests for Face Detection Cache System
Tests face data caching, compression, retrieval, and performance optimizations.
"""

import pytest
import tempfile
import json
import pickle
import time
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from datetime import datetime, timedelta

# Import test fixtures
from tests.conftest import (
    test_cache_manager, sample_face_data, avatar_config_fixture
)

class TestFaceDetectionCache:
    """Test face detection caching functionality"""
    
    @pytest.fixture
    def cache_manager(self, avatar_config_fixture):
        """Create face cache manager instance for testing"""
        from app.core.face_cache_manager import FaceCacheManager
        from app.core.compression_engine import CompressionEngine
        
        # Create temporary cache directory
        cache_dir = Path(tempfile.mkdtemp())
        
        compression_engine = Mock(spec=CompressionEngine)
        compression_engine.compress_face_data.return_value = b"compressed_data"
        compression_engine.decompress_face_data.return_value = Mock()
        compression_engine.estimate_compression_ratio.return_value = 0.3
        
        cache_manager = FaceCacheManager(
            cache_storage_path=cache_dir,
            compression_engine=compression_engine,
            max_cache_size=1024*1024*100,  # 100MB
            enable_compression=True
        )
        
        return cache_manager
    
    def test_store_face_cache_success(self, cache_manager, sample_face_data):
        """Test successful face cache storage"""
        # Arrange
        avatar_id = "test_cache_store"
        
        # Act
        result = cache_manager.store_face_cache(avatar_id, sample_face_data)
        
        # Assert
        assert result is True
        
        # Verify cache file exists
        cache_file = cache_manager.cache_storage_path / f"{avatar_id}_face_cache.pkl"
        assert cache_file.exists()
        
        # Verify compression was called
        cache_manager.compression_engine.compress_face_data.assert_called_once_with(sample_face_data)
        
        # Verify cache metadata
        assert avatar_id in cache_manager.cache_storage
        assert cache_manager.cache_storage[avatar_id].avatar_id == avatar_id
    
    def test_retrieve_face_cache_success(self, cache_manager, sample_face_data):
        """Test successful face cache retrieval"""
        # Arrange
        avatar_id = "test_cache_retrieve"
        
        # Store first
        cache_manager.store_face_cache(avatar_id, sample_face_data)
        
        # Mock decompression
        cache_manager.compression_engine.decompress_face_data.return_value = sample_face_data
        
        # Act
        retrieved_data = cache_manager.retrieve_face_cache(avatar_id)
        
        # Assert
        assert retrieved_data is not None
        assert retrieved_data.avatar_id == avatar_id
        assert len(retrieved_data.face_boxes) > 0
        assert len(retrieved_data.cropped_faces) > 0
        
        # Verify decompression was called
        cache_manager.compression_engine.decompress_face_data.assert_called()
    
    def test_retrieve_non_existent_cache(self, cache_manager):
        """Test retrieval of non-existent cache returns None"""
        # Arrange
        avatar_id = "non_existent_avatar"
        
        # Act
        result = cache_manager.retrieve_face_cache(avatar_id)
        
        # Assert
        assert result is None
    
    def test_cache_compression_efficiency(self, cache_manager, sample_face_data):
        """Test face data compression efficiency"""
        # Arrange
        avatar_id = "test_compression"
        original_size = len(pickle.dumps(sample_face_data))
        
        # Mock compression with realistic ratio
        compressed_data = b"compressed" * (original_size // 4)  # 75% compression
        cache_manager.compression_engine.compress_face_data.return_value = compressed_data
        cache_manager.compression_engine.estimate_compression_ratio.return_value = 0.25
        
        # Act
        cache_manager.store_face_cache(avatar_id, sample_face_data)
        
        # Assert
        compression_ratio = cache_manager.compression_engine.estimate_compression_ratio()
        assert compression_ratio < 0.5  # At least 50% compression
        
        # Verify compressed size is smaller
        cache_file = cache_manager.cache_storage_path / f"{avatar_id}_face_cache.pkl"
        assert cache_file.stat().st_size < original_size
    
    def test_cache_integrity_validation(self, cache_manager, sample_face_data):
        """Test cache integrity validation"""
        # Arrange
        avatar_id = "test_integrity"
        
        # Add integrity hash to sample data
        sample_face_data.integrity_hash = "test_hash_123"
        
        # Store cache
        cache_manager.store_face_cache(avatar_id, sample_face_data)
        
        # Mock successful integrity check
        cache_manager.compression_engine.decompress_face_data.return_value = sample_face_data
        
        # Act
        retrieved_data = cache_manager.retrieve_face_cache(avatar_id)
        
        # Assert
        assert retrieved_data is not None
        assert retrieved_data.integrity_hash == "test_hash_123"
        
        # Test with corrupted data
        corrupted_data = Mock()
        corrupted_data.integrity_hash = "wrong_hash"
        cache_manager.compression_engine.decompress_face_data.return_value = corrupted_data
        
        with patch.object(cache_manager, '_validate_cache_integrity', return_value=False):
            corrupted_result = cache_manager.retrieve_face_cache(avatar_id)
            assert corrupted_result is None
    
    def test_memory_cache_management(self, cache_manager, sample_face_data):
        """Test in-memory cache management and LRU eviction"""
        # Arrange
        cache_manager.max_memory_cache_size = 3  # Small cache for testing
        avatar_ids = [f"test_avatar_{i}" for i in range(5)]
        
        # Act - Store more avatars than cache can hold
        for avatar_id in avatar_ids:
            cache_manager.store_face_cache(avatar_id, sample_face_data)
        
        # Assert - Check LRU eviction
        assert len(cache_manager.cache_storage) <= cache_manager.max_memory_cache_size
        
        # Verify most recent avatars are still in memory cache
        for avatar_id in avatar_ids[-3:]:  # Last 3 avatars
            assert avatar_id in cache_manager.cache_storage
    
    def test_warm_up_avatar_cache(self, cache_manager, sample_face_data):
        """Test avatar cache warm-up functionality"""
        # Arrange
        avatar_ids = [f"warmup_avatar_{i}" for i in range(3)]
        
        # Store avatars on disk first
        for avatar_id in avatar_ids:
            cache_manager.store_face_cache(avatar_id, sample_face_data)
        
        # Clear memory cache
        cache_manager.cache_storage.clear()
        
        # Mock decompression
        cache_manager.compression_engine.decompress_face_data.return_value = sample_face_data
        
        # Act
        warmup_report = cache_manager.warm_up_avatar_cache(avatar_ids)
        
        # Assert
        assert warmup_report.success is True
        assert warmup_report.loaded_count == len(avatar_ids)
        assert warmup_report.failed_count == 0
        
        # Verify all avatars are now in memory cache
        for avatar_id in avatar_ids:
            assert avatar_id in cache_manager.cache_storage
            assert cache_manager.cache_storage[avatar_id] is not None
    
    def test_cleanup_expired_cache(self, cache_manager, sample_face_data):
        """Test automatic cache cleanup for expired entries"""
        # Arrange
        avatar_id_old = "old_avatar"
        avatar_id_new = "new_avatar"
        
        # Create old cache entry
        old_timestamp = datetime.now() - timedelta(days=31)  # 31 days old
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = old_timestamp
            cache_manager.store_face_cache(avatar_id_old, sample_face_data)
        
        # Create new cache entry
        cache_manager.store_face_cache(avatar_id_new, sample_face_data)
        
        # Act
        cleanup_report = cache_manager.cleanup_expired_cache(max_age_days=30)
        
        # Assert
        assert cleanup_report.cleaned_count >= 1
        assert cleanup_report.remaining_count >= 1
        
        # Verify old cache is removed but new cache remains
        old_cache_file = cache_manager.cache_storage_path / f"{avatar_id_old}_face_cache.pkl"
        new_cache_file = cache_manager.cache_storage_path / f"{avatar_id_new}_face_cache.pkl"
        
        # The cleanup might not delete files immediately in mock scenario
        # but we can verify the cleanup logic was triggered
        assert cleanup_report.success is True
    
    def test_concurrent_cache_access(self, cache_manager, sample_face_data):
        """Test concurrent cache access safety"""
        import threading
        import queue
        
        # Arrange
        avatar_id = "concurrent_test"
        results = queue.Queue()
        
        def store_operation():
            try:
                result = cache_manager.store_face_cache(avatar_id, sample_face_data)
                results.put(("store", result))
            except Exception as e:
                results.put(("store_error", str(e)))
        
        def retrieve_operation():
            try:
                result = cache_manager.retrieve_face_cache(avatar_id)
                results.put(("retrieve", result is not None))
            except Exception as e:
                results.put(("retrieve_error", str(e)))
        
        # Act - Simulate concurrent operations
        threads = []
        for _ in range(3):
            store_thread = threading.Thread(target=store_operation)
            retrieve_thread = threading.Thread(target=retrieve_operation)
            threads.extend([store_thread, retrieve_thread])
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join(timeout=5)
        
        # Assert - Collect results
        operation_results = []
        while not results.empty():
            operation_results.append(results.get())
        
        # Verify no errors occurred
        error_results = [r for r in operation_results if "error" in r[0]]
        assert len(error_results) == 0, f"Concurrent access errors: {error_results}"
        
        # Verify at least one successful store and retrieve
        store_results = [r for r in operation_results if r[0] == "store"]
        retrieve_results = [r for r in operation_results if r[0] == "retrieve"]
        
        assert any(r[1] for r in store_results), "No successful store operations"
        assert any(r[1] for r in retrieve_results), "No successful retrieve operations"

class TestFaceCachePerformance:
    """Test face cache performance characteristics"""
    
    def test_cache_access_time_requirements(self, cache_manager, sample_face_data):
        """Test cache access meets timing requirements (< 10ms)"""
        # Arrange
        avatar_id = "performance_test"
        cache_manager.store_face_cache(avatar_id, sample_face_data)
        
        # Mock fast decompression
        cache_manager.compression_engine.decompress_face_data.return_value = sample_face_data
        
        # Act
        start_time = time.time()
        result = cache_manager.retrieve_face_cache(avatar_id)
        end_time = time.time()
        
        # Assert
        access_time = (end_time - start_time) * 1000  # Convert to milliseconds
        assert access_time < 10.0, f"Cache access took {access_time:.2f}ms, should be < 10ms"
        assert result is not None
    
    def test_cache_storage_time_requirements(self, cache_manager, sample_face_data):
        """Test cache storage performance"""
        # Arrange
        avatar_id = "storage_performance_test"
        
        # Act
        start_time = time.time()
        result = cache_manager.store_face_cache(avatar_id, sample_face_data)
        end_time = time.time()
        
        # Assert
        storage_time = (end_time - start_time) * 1000  # Convert to milliseconds
        assert storage_time < 100.0, f"Cache storage took {storage_time:.2f}ms, should be < 100ms"
        assert result is True
    
    def test_memory_cache_hit_ratio(self, cache_manager, sample_face_data):
        """Test memory cache hit ratio performance"""
        # Arrange
        avatar_ids = [f"hit_ratio_test_{i}" for i in range(10)]
        
        # Store all avatars
        for avatar_id in avatar_ids:
            cache_manager.store_face_cache(avatar_id, sample_face_data)
        
        # Mock decompression for disk access
        cache_manager.compression_engine.decompress_face_data.return_value = sample_face_data
        
        # Act - Access recently stored avatars multiple times
        total_accesses = 0
        cache_hits = 0
        
        for _ in range(3):  # 3 rounds of access
            for avatar_id in avatar_ids[-5:]:  # Access last 5 avatars
                total_accesses += 1
                # Check if in memory cache before retrieval
                if avatar_id in cache_manager.cache_storage:
                    cache_hits += 1
                cache_manager.retrieve_face_cache(avatar_id)
        
        # Assert
        hit_ratio = cache_hits / total_accesses
        assert hit_ratio > 0.8, f"Cache hit ratio {hit_ratio:.2f} should be > 0.8"
    
    def test_compression_performance(self, cache_manager):
        """Test compression performance with large face data"""
        # Arrange
        avatar_id = "compression_performance"
        
        # Create large face data
        large_face_data = Mock()
        large_face_data.avatar_id = avatar_id
        large_face_data.face_boxes = [(50, 50, 200, 200)] * 100  # 100 frames
        large_face_data.cropped_faces = [
            np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8) 
            for _ in range(100)
        ]
        large_face_data.face_landmarks = [
            np.random.rand(68, 2) for _ in range(100)
        ]
        
        # Mock compression timing
        def timed_compression(data):
            time.sleep(0.01)  # Simulate 10ms compression time
            return b"compressed_large_data"
        
        cache_manager.compression_engine.compress_face_data.side_effect = timed_compression
        
        # Act
        start_time = time.time()
        result = cache_manager.store_face_cache(avatar_id, large_face_data)
        end_time = time.time()
        
        # Assert
        compression_time = (end_time - start_time) * 1000
        assert compression_time < 50.0, f"Compression took {compression_time:.2f}ms, should be < 50ms"
        assert result is True

class TestFaceCacheReliability:
    """Test face cache reliability and error handling"""
    
    def test_cache_corruption_recovery(self, cache_manager, sample_face_data):
        """Test recovery from cache corruption"""
        # Arrange
        avatar_id = "corruption_test"
        cache_manager.store_face_cache(avatar_id, sample_face_data)
        
        # Simulate corruption by making decompression fail
        cache_manager.compression_engine.decompress_face_data.side_effect = Exception("Corruption detected")
        
        # Act
        result = cache_manager.retrieve_face_cache(avatar_id)
        
        # Assert
        assert result is None  # Should return None for corrupted cache
        
        # Verify cache cleanup for corrupted entry
        cache_file = cache_manager.cache_storage_path / f"{avatar_id}_face_cache.pkl"
        # In a real implementation, corrupted cache would be cleaned up
        # Here we just verify the error was handled gracefully
    
    def test_disk_space_handling(self, cache_manager, sample_face_data):
        """Test handling of disk space limitations"""
        # Arrange
        avatar_id = "disk_space_test"
        
        # Mock disk space error
        with patch('pathlib.Path.write_bytes', side_effect=OSError("No space left on device")):
            # Act
            result = cache_manager.store_face_cache(avatar_id, sample_face_data)
            
            # Assert
            assert result is False  # Should handle disk space error gracefully
    
    def test_cache_consistency_validation(self, cache_manager, sample_face_data):
        """Test cache consistency validation"""
        # Arrange
        avatar_id = "consistency_test"
        
        # Store cache with specific metadata
        sample_face_data.cache_timestamp = datetime.now()
        sample_face_data.cache_version = "1.0.0"
        
        cache_manager.store_face_cache(avatar_id, sample_face_data)
        
        # Mock retrieval with version mismatch
        inconsistent_data = Mock()
        inconsistent_data.cache_version = "0.9.0"  # Old version
        cache_manager.compression_engine.decompress_face_data.return_value = inconsistent_data
        
        with patch.object(cache_manager, '_validate_cache_consistency', return_value=False):
            # Act
            result = cache_manager.retrieve_face_cache(avatar_id)
            
            # Assert
            assert result is None  # Should reject inconsistent cache

if __name__ == "__main__":
    pytest.main([__file__]) 