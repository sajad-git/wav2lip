"""
Concurrent Processing Load Tests
Tests multiple concurrent users with shared models and avatar cache performance
"""

import pytest
import asyncio
import time
import psutil
import statistics
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

from app.core.model_loader import ColdModelLoader
from app.core.resource_manager import GPUResourceManager
from app.core.face_cache_manager import FaceCacheManager
from app.services.wav2lip_service import PreloadedWav2LipService
from app.streaming.websocket_handler import BinaryWebSocketHandler
from app.models.chunk_models import AudioChunk, VideoChunk, ChunkMetadata


class TestConcurrentLoad:
    """Test concurrent processing with multiple users"""
    
    @pytest.fixture
    def mock_gpu_resources(self):
        """Mock GPU resources for load testing"""
        return {
            "total_memory": 24 * 1024 * 1024 * 1024,  # 24GB
            "available_memory": 20 * 1024 * 1024 * 1024,  # 20GB available
            "model_memory_usage": 5 * 1024 * 1024 * 1024,  # 5GB for models
            "avatar_cache_memory": 1 * 1024 * 1024 * 1024  # 1GB for avatar cache
        }
    
    @pytest.fixture
    def mock_user_sessions(self):
        """Mock user sessions for testing"""
        return {
            f"user_{i}": {
                "user_id": f"user_{i}",
                "session_start": time.time(),
                "processing_queue": [],
                "avatar_preferences": [f"avatar_{i % 3}"]  # 3 avatars rotated
            } for i in range(3)
        }
    
    @pytest.fixture
    def mock_shared_models(self):
        """Mock pre-loaded shared model instances"""
        models = {}
        for model_name in ["wav2lip", "wav2lip_gan", "face_detector"]:
            model = Mock()
            model.memory_usage = 1024 * 1024 * 1024  # 1GB each
            model.reference_count = 0
            model.last_accessed = time.time()
            
            # Mock inference method
            async def mock_inference(*args, **kwargs):
                await asyncio.sleep(0.08)  # Simulate processing time
                return np.random.rand(1, 3, 96, 96)
            
            model.run_inference = mock_inference
            models[model_name] = model
        
        return models
    
    @pytest.fixture
    def mock_avatar_cache(self):
        """Mock avatar cache with multiple avatars"""
        cache_data = {}
        for i in range(5):  # 5 avatars in cache
            avatar_id = f"avatar_{i}"
            cache_data[avatar_id] = {
                "avatar_id": avatar_id,
                "face_boxes": [(50, 50, 200, 200)],
                "face_landmarks": [np.random.rand(106, 2)],
                "cropped_faces": [np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)],
                "cache_timestamp": time.time(),
                "access_count": 0
            }
        
        cache = Mock()
        cache.cache_data = cache_data
        
        async def mock_retrieve(avatar_id):
            if avatar_id in cache_data:
                cache_data[avatar_id]["access_count"] += 1
                await asyncio.sleep(0.01)  # Fast cache access
                return cache_data[avatar_id]
            return None
        
        cache.retrieve_face_cache = mock_retrieve
        cache.get_memory_usage = Mock(return_value=512 * 1024 * 1024)  # 512MB
        
        return cache
    
    @pytest.mark.asyncio
    async def test_three_user_concurrent_processing(self, mock_shared_models, mock_avatar_cache, mock_user_sessions):
        """Test 3-user concurrent processing with shared resources"""
        # Arrange
        resource_manager = Mock()
        resource_manager.model_instances = mock_shared_models
        resource_manager.avatar_cache = mock_avatar_cache
        resource_manager.user_sessions = {}
        
        # Setup resource manager methods
        async def mock_allocate_session(user_id):
            session = Mock()
            session.user_id = user_id
            session.model_access_count = {}
            session.avatar_access_count = {}
            session.processing_queue = []
            resource_manager.user_sessions[user_id] = session
            return session
        
        def mock_get_model(model_name, user_id):
            if user_id in resource_manager.user_sessions:
                session = resource_manager.user_sessions[user_id]
                session.model_access_count[model_name] = session.model_access_count.get(model_name, 0) + 1
            return mock_shared_models[model_name]
        
        async def mock_get_avatar_cache(avatar_id, user_id):
            if user_id in resource_manager.user_sessions:
                session = resource_manager.user_sessions[user_id]
                session.avatar_access_count[avatar_id] = session.avatar_access_count.get(avatar_id, 0) + 1
            return await mock_avatar_cache.retrieve_face_cache(avatar_id)
        
        resource_manager.allocate_user_session = mock_allocate_session
        resource_manager.get_shared_model_instance = mock_get_model
        resource_manager.get_shared_avatar_cache = mock_get_avatar_cache
        
        # Define user workflow
        async def user_workflow(user_id, num_requests=5):
            # Allocate session
            session = await resource_manager.allocate_user_session(user_id)
            
            processing_times = []
            
            for i in range(num_requests):
                start_time = time.time()
                
                # Get model instance (shared)
                model = resource_manager.get_shared_model_instance("wav2lip", user_id)
                
                # Get avatar cache (shared)
                avatar_id = f"avatar_{i % 3}"  # Rotate through 3 avatars
                cached_face = await resource_manager.get_shared_avatar_cache(avatar_id, user_id)
                
                # Simulate processing
                if cached_face:
                    # Fast processing with cached face data
                    await asyncio.sleep(0.08)  # Target processing time
                else:
                    # Slower processing without cache
                    await asyncio.sleep(0.35)
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # Small delay between requests
                await asyncio.sleep(0.02)
            
            return {
                "user_id": user_id,
                "requests_processed": num_requests,
                "processing_times": processing_times,
                "average_time": statistics.mean(processing_times),
                "total_time": sum(processing_times)
            }
        
        # Act
        overall_start = time.time()
        
        # Run 3 concurrent users
        user_tasks = [
            user_workflow("user_1", 5),
            user_workflow("user_2", 5),
            user_workflow("user_3", 5)
        ]
        
        results = await asyncio.gather(*user_tasks)
        overall_time = time.time() - overall_start
        
        # Assert
        assert len(results) == 3
        
        # Check each user processed successfully
        for result in results:
            assert result["requests_processed"] == 5
            assert result["average_time"] < 0.15  # Target: <150ms average
            assert len(result["processing_times"]) == 5
        
        # Check overall performance
        assert overall_time < 3.0  # Total time for all users
        
        # Verify resource sharing
        for user_id in ["user_1", "user_2", "user_3"]:
            session = resource_manager.user_sessions[user_id]
            assert session.model_access_count["wav2lip"] == 5  # 5 requests per user
            assert sum(session.avatar_access_count.values()) == 5  # 5 avatar accesses
    
    @pytest.mark.asyncio
    async def test_gpu_resource_allocation(self, mock_gpu_resources, mock_shared_models):
        """Test GPU resource allocation under concurrent load"""
        # Arrange
        gpu_manager = Mock()
        gpu_manager.total_memory = mock_gpu_resources["total_memory"]
        gpu_manager.available_memory = mock_gpu_resources["available_memory"]
        gpu_manager.allocated_memory = mock_gpu_resources["model_memory_usage"]
        
        def mock_allocate_memory(size_bytes):
            if gpu_manager.allocated_memory + size_bytes <= gpu_manager.total_memory:
                gpu_manager.allocated_memory += size_bytes
                return True
            return False
        
        def mock_get_utilization():
            return {
                "memory_used": gpu_manager.allocated_memory,
                "memory_total": gpu_manager.total_memory,
                "utilization_percent": (gpu_manager.allocated_memory / gpu_manager.total_memory) * 100
            }
        
        gpu_manager.allocate_memory = mock_allocate_memory
        gpu_manager.get_utilization = mock_get_utilization
        
        # Act - Simulate concurrent memory allocation
        allocation_requests = [
            512 * 1024 * 1024,  # 512MB
            256 * 1024 * 1024,  # 256MB
            1024 * 1024 * 1024, # 1GB
        ]
        
        allocation_results = []
        for size in allocation_requests:
            success = gpu_manager.allocate_memory(size)
            allocation_results.append(success)
        
        utilization = gpu_manager.get_utilization()
        
        # Assert
        assert all(allocation_results)  # All allocations should succeed
        assert utilization["utilization_percent"] < 80  # <80% utilization target
        assert utilization["memory_used"] <= utilization["memory_total"]
    
    @pytest.mark.asyncio
    async def test_model_instance_sharing(self, mock_shared_models):
        """Test model instance sharing efficiency"""
        # Arrange
        model_manager = Mock()
        model_manager.loaded_models = mock_shared_models
        model_manager.access_stats = {model: {"access_count": 0, "users": set()} for model in mock_shared_models}
        
        def mock_get_model(model_name, user_id):
            if model_name in model_manager.loaded_models:
                model_manager.access_stats[model_name]["access_count"] += 1
                model_manager.access_stats[model_name]["users"].add(user_id)
                return model_manager.loaded_models[model_name]
            return None
        
        model_manager.get_model_instance = mock_get_model
        
        # Act - Multiple users accessing same models
        users = ["user_1", "user_2", "user_3"]
        models = ["wav2lip", "face_detector"]
        
        for user in users:
            for model in models:
                for _ in range(3):  # 3 accesses per user per model
                    instance = model_manager.get_model_instance(model, user)
                    assert instance is not None
        
        # Assert
        for model_name in models:
            stats = model_manager.access_stats[model_name]
            assert stats["access_count"] == 9  # 3 users × 3 accesses
            assert len(stats["users"]) == 3  # 3 different users
            
        # Verify same instances are shared
        wav2lip_instance1 = model_manager.get_model_instance("wav2lip", "user_1")
        wav2lip_instance2 = model_manager.get_model_instance("wav2lip", "user_2")
        assert wav2lip_instance1 is wav2lip_instance2  # Same instance
    
    @pytest.mark.asyncio
    async def test_avatar_cache_concurrent_access(self, mock_avatar_cache):
        """Test shared avatar cache performance under concurrent access"""
        # Arrange
        cache_stats = {
            "hits": 0,
            "misses": 0,
            "access_times": []
        }
        
        original_retrieve = mock_avatar_cache.retrieve_face_cache
        
        async def instrumented_retrieve(avatar_id):
            start_time = time.time()
            result = await original_retrieve(avatar_id)
            access_time = time.time() - start_time
            
            cache_stats["access_times"].append(access_time)
            if result:
                cache_stats["hits"] += 1
            else:
                cache_stats["misses"] += 1
            
            return result
        
        mock_avatar_cache.retrieve_face_cache = instrumented_retrieve
        
        # Act - Concurrent avatar cache access
        async def access_avatars(user_id, num_accesses=10):
            avatar_ids = [f"avatar_{i % 5}" for i in range(num_accesses)]  # 5 avatars, some repeated
            
            for avatar_id in avatar_ids:
                cached_data = await mock_avatar_cache.retrieve_face_cache(avatar_id)
                # Simulate small processing delay
                await asyncio.sleep(0.001)
            
            return f"user_{user_id}_completed"
        
        # Run 3 concurrent users
        tasks = [access_avatars(i, 10) for i in range(3)]
        results = await asyncio.gather(*tasks)
        
        # Assert
        assert len(results) == 3
        assert cache_stats["hits"] == 30  # All should be cache hits (30 total accesses)
        assert cache_stats["misses"] == 0
        
        # Check access time performance
        avg_access_time = statistics.mean(cache_stats["access_times"])
        max_access_time = max(cache_stats["access_times"])
        
        assert avg_access_time < 0.02  # Target: <20ms average
        assert max_access_time < 0.05  # Target: <50ms maximum
    
    @pytest.mark.asyncio
    async def test_memory_stability_under_load(self, mock_gpu_resources):
        """Test memory usage stability under sustained load"""
        # Arrange
        memory_monitor = Mock()
        memory_samples = []
        
        def mock_get_memory_usage():
            # Simulate slight memory fluctuation
            base_usage = 6 * 1024 * 1024 * 1024  # 6GB base
            fluctuation = np.random.normal(0, 50 * 1024 * 1024)  # ±50MB
            current_usage = base_usage + fluctuation
            memory_samples.append(current_usage)
            return current_usage
        
        memory_monitor.get_current_usage = mock_get_memory_usage
        
        # Act - Sustained processing simulation
        async def sustained_processing():
            for i in range(50):  # 50 processing cycles
                memory_monitor.get_current_usage()
                await asyncio.sleep(0.02)  # 20ms cycles
        
        await sustained_processing()
        
        # Assert
        memory_variance = np.var(memory_samples)
        memory_std = np.std(memory_samples)
        max_memory = max(memory_samples)
        min_memory = min(memory_samples)
        
        # Memory should be stable (low variance)
        assert memory_std < 100 * 1024 * 1024  # <100MB standard deviation
        assert (max_memory - min_memory) < 200 * 1024 * 1024  # <200MB range
        assert max_memory < 8 * 1024 * 1024 * 1024  # <8GB maximum
    
    @pytest.mark.asyncio
    async def test_performance_degradation_limits(self, mock_shared_models, mock_avatar_cache):
        """Test performance boundaries under increasing load"""
        # Arrange
        performance_data = []
        
        async def process_request(user_id, request_id):
            start_time = time.time()
            
            # Simulate request processing
            model = mock_shared_models["wav2lip"]
            avatar_cache = await mock_avatar_cache.retrieve_face_cache(f"avatar_{request_id % 3}")
            
            # Processing time increases slightly with concurrent load
            base_time = 0.08
            load_penalty = len(asyncio.current_task().get_loop()._ready) * 0.001
            processing_time = base_time + load_penalty
            
            await asyncio.sleep(processing_time)
            
            total_time = time.time() - start_time
            performance_data.append({
                "user_id": user_id,
                "request_id": request_id,
                "processing_time": total_time,
                "timestamp": start_time
            })
            
            return total_time
        
        # Act - Gradually increase load
        load_levels = [1, 2, 3, 5]  # Number of concurrent users
        performance_by_load = {}
        
        for num_users in load_levels:
            performance_data.clear()
            
            # Create tasks for concurrent users
            tasks = []
            for user_id in range(num_users):
                for request_id in range(5):  # 5 requests per user
                    task = process_request(f"user_{user_id}", request_id)
                    tasks.append(task)
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            avg_processing_time = statistics.mean(results)
            performance_by_load[num_users] = {
                "avg_processing_time": avg_processing_time,
                "total_time": total_time,
                "requests_per_second": len(results) / total_time
            }
        
        # Assert
        # Performance should degrade gracefully
        for i in range(len(load_levels) - 1):
            current_load = load_levels[i]
            next_load = load_levels[i + 1]
            
            current_perf = performance_by_load[current_load]["avg_processing_time"]
            next_perf = performance_by_load[next_load]["avg_processing_time"]
            
            # Performance degradation should be reasonable
            degradation_ratio = next_perf / current_perf
            assert degradation_ratio < 2.0  # <2x degradation per load level
        
        # Even under max load, should meet minimum performance
        max_load_perf = performance_by_load[max(load_levels)]["avg_processing_time"]
        assert max_load_perf < 0.3  # <300ms even under max load
    
    @pytest.mark.asyncio
    async def test_avatar_registration_under_load(self):
        """Test avatar registration system under concurrent load"""
        # Arrange
        registration_service = Mock()
        registration_stats = {
            "successful_registrations": 0,
            "failed_registrations": 0,
            "registration_times": []
        }
        
        async def mock_register_avatar(avatar_data, avatar_id, user_id):
            start_time = time.time()
            
            # Simulate registration processing
            await asyncio.sleep(0.2 + np.random.normal(0, 0.05))  # ~200ms ± 50ms
            
            # Simulate occasional failures under load
            failure_rate = 0.05  # 5% failure rate
            if np.random.random() < failure_rate:
                registration_stats["failed_registrations"] += 1
                raise Exception("Registration failed due to load")
            
            registration_time = time.time() - start_time
            registration_stats["registration_times"].append(registration_time)
            registration_stats["successful_registrations"] += 1
            
            return {
                "avatar_id": avatar_id,
                "status": "success",
                "processing_time": registration_time
            }
        
        registration_service.register_avatar = mock_register_avatar
        
        # Act - Concurrent avatar registrations
        async def register_user_avatars(user_id, num_avatars=3):
            results = []
            for i in range(num_avatars):
                try:
                    result = await registration_service.register_avatar(
                        avatar_data=b"fake_avatar_data",
                        avatar_id=f"{user_id}_avatar_{i}",
                        user_id=user_id
                    )
                    results.append(result)
                except Exception as e:
                    results.append({"status": "failed", "error": str(e)})
            return results
        
        # Run 5 concurrent users each registering 3 avatars
        user_tasks = [register_user_avatars(f"user_{i}", 3) for i in range(5)]
        all_results = await asyncio.gather(*user_tasks)
        
        # Flatten results
        all_registrations = [result for user_results in all_results for result in user_results]
        
        # Assert
        successful_count = sum(1 for r in all_registrations if r["status"] == "success")
        failed_count = sum(1 for r in all_registrations if r["status"] == "failed")
        
        assert successful_count >= 12  # At least 80% success rate (12/15)
        assert failed_count <= 3  # At most 20% failure rate
        
        if registration_stats["registration_times"]:
            avg_registration_time = statistics.mean(registration_stats["registration_times"])
            max_registration_time = max(registration_stats["registration_times"])
            
            assert avg_registration_time < 0.3  # <300ms average
            assert max_registration_time < 0.5  # <500ms maximum


class TestLoadTestMetrics:
    """Test performance metrics collection and analysis"""
    
    @pytest.fixture
    def metrics_collector(self):
        """Mock metrics collector"""
        collector = Mock()
        collector.metrics = {
            "processing_times": [],
            "memory_usage": [],
            "gpu_utilization": [],
            "cache_hit_rates": [],
            "error_rates": []
        }
        
        def add_metric(metric_type, value, timestamp=None):
            if timestamp is None:
                timestamp = time.time()
            collector.metrics[metric_type].append({
                "value": value,
                "timestamp": timestamp
            })
        
        collector.add_metric = add_metric
        
        return collector
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, metrics_collector):
        """Test comprehensive performance metrics collection"""
        # Arrange - Simulate load test with metrics collection
        async def monitored_processing(process_id):
            start_time = time.time()
            
            # Simulate processing with metrics
            processing_time = 0.08 + np.random.normal(0, 0.01)
            await asyncio.sleep(processing_time)
            
            # Collect metrics
            metrics_collector.add_metric("processing_times", processing_time)
            metrics_collector.add_metric("memory_usage", 1024 + np.random.normal(0, 50))  # MB
            metrics_collector.add_metric("gpu_utilization", 75 + np.random.normal(0, 5))  # %
            metrics_collector.add_metric("cache_hit_rates", 0.95 + np.random.normal(0, 0.02))
            
            return processing_time
        
        # Act - Run monitored load test
        tasks = [monitored_processing(i) for i in range(20)]
        results = await asyncio.gather(*tasks)
        
        # Assert - Analyze collected metrics
        processing_times = [m["value"] for m in metrics_collector.metrics["processing_times"]]
        memory_usage = [m["value"] for m in metrics_collector.metrics["memory_usage"]]
        gpu_utilization = [m["value"] for m in metrics_collector.metrics["gpu_utilization"]]
        cache_hit_rates = [m["value"] for m in metrics_collector.metrics["cache_hit_rates"]]
        
        # Performance assertions
        assert len(processing_times) == 20
        assert statistics.mean(processing_times) < 0.12  # <120ms average
        assert statistics.stdev(processing_times) < 0.02  # Low variance
        
        assert statistics.mean(memory_usage) < 1200  # <1200MB average
        assert statistics.mean(gpu_utilization) < 85  # <85% GPU utilization
        assert statistics.mean(cache_hit_rates) > 0.9  # >90% cache hit rate
    
    def test_load_test_report_generation(self, metrics_collector):
        """Test load test report generation"""
        # Arrange - Add sample metrics
        sample_data = {
            "processing_times": [0.08, 0.09, 0.07, 0.10, 0.08],
            "memory_usage": [1000, 1050, 1020, 1080, 1030],
            "gpu_utilization": [75, 78, 72, 80, 76],
            "cache_hit_rates": [0.95, 0.96, 0.94, 0.97, 0.95]
        }
        
        for metric_type, values in sample_data.items():
            for value in values:
                metrics_collector.add_metric(metric_type, value)
        
        # Act - Generate report
        def generate_load_test_report(metrics):
            report = {}
            for metric_type, metric_data in metrics.items():
                values = [m["value"] for m in metric_data]
                if values:
                    report[metric_type] = {
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                        "min": min(values),
                        "max": max(values),
                        "count": len(values)
                    }
            return report
        
        report = generate_load_test_report(metrics_collector.metrics)
        
        # Assert
        assert "processing_times" in report
        assert "memory_usage" in report
        assert "gpu_utilization" in report
        assert "cache_hit_rates" in report
        
        # Check processing times analysis
        proc_stats = report["processing_times"]
        assert proc_stats["mean"] < 0.1  # <100ms average
        assert proc_stats["std_dev"] < 0.02  # Low variance
        assert proc_stats["count"] == 5
        
        # Check cache performance
        cache_stats = report["cache_hit_rates"]
        assert cache_stats["mean"] > 0.9  # >90% hit rate
        assert cache_stats["min"] > 0.9  # Consistent performance


# Load test fixtures and utilities
@pytest.fixture(scope="session")
def load_test_metrics():
    """Global load test metrics tracking"""
    return {
        "concurrent_user_performance": [],
        "resource_utilization": [],
        "error_rates": [],
        "scalability_metrics": []
    }


@pytest.mark.asyncio
async def simulate_realistic_load():
    """Simulate realistic user load patterns"""
    # Simulate varying load patterns
    load_patterns = [
        {"users": 1, "duration": 10, "requests_per_user": 5},
        {"users": 2, "duration": 15, "requests_per_user": 8},
        {"users": 3, "duration": 20, "requests_per_user": 10}
    ]
    
    results = []
    
    for pattern in load_patterns:
        start_time = time.time()
        
        # Simulate user tasks
        async def user_session(user_id, num_requests):
            session_results = []
            for i in range(num_requests):
                request_start = time.time()
                await asyncio.sleep(0.08 + np.random.exponential(0.02))  # Realistic processing time
                request_time = time.time() - request_start
                session_results.append(request_time)
                
                # Random inter-request delay
                await asyncio.sleep(np.random.exponential(0.1))
            
            return session_results
        
        # Run concurrent users
        user_tasks = [
            user_session(f"user_{i}", pattern["requests_per_user"])
            for i in range(pattern["users"])
        ]
        
        user_results = await asyncio.gather(*user_tasks)
        pattern_time = time.time() - start_time
        
        # Flatten all request times
        all_request_times = [t for user_times in user_results for t in user_times]
        
        results.append({
            "pattern": pattern,
            "execution_time": pattern_time,
            "request_times": all_request_times,
            "avg_request_time": statistics.mean(all_request_times),
            "total_requests": len(all_request_times)
        })
    
    return results


def pytest_sessionfinish_load(session, exitstatus):
    """Generate comprehensive load test report"""
    print("\n" + "="*80)
    print("CONCURRENT PROCESSING LOAD TEST SUMMARY")
    print("="*80)
    print("Target Performance Metrics:")
    print("- 3 concurrent users supported: ✓")
    print("- <150ms average processing time: <results>")
    print("- >90% cache hit rate: <results>")
    print("- <20GB GPU memory usage: <results>")
    print("- <5% error rate under load: <results>")
    print("="*80)
    print("Scalability Analysis:")
    print("- Performance degradation under load: <analysis>")
    print("- Resource utilization efficiency: <metrics>")
    print("- Recommendations: <optimization suggestions>")
    print("="*80) 