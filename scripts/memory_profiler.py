#!/usr/bin/env python3
"""
Memory Profiler for Avatar Streaming Service
Monitors GPU and system memory usage, tracks allocations, and detects memory leaks
"""

import os
import sys
import time
import json
import argparse
import logging
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import traceback

# GPU monitoring imports
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not available. GPU monitoring disabled.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a point in time"""
    timestamp: datetime
    system_memory_mb: float
    system_memory_percent: float
    gpu_memory_mb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_temperature: Optional[float] = None
    process_memory_mb: float = 0.0
    process_memory_percent: float = 0.0
    model_memory_mb: Optional[float] = None
    cache_memory_mb: Optional[float] = None
    active_connections: int = 0
    processing_chunks: int = 0

@dataclass
class MemoryStats:
    """Memory statistics over time"""
    avg_system_memory_mb: float
    peak_system_memory_mb: float
    avg_gpu_memory_mb: Optional[float]
    peak_gpu_memory_mb: Optional[float]
    memory_growth_rate_mb_per_hour: float
    potential_leak_detected: bool
    efficiency_score: float

class GPUMonitor:
    """GPU memory and performance monitoring"""
    
    def __init__(self):
        self.initialized = False
        self.device_count = 0
        self.device_handles = []
        
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                
                for i in range(self.device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self.device_handles.append(handle)
                
                self.initialized = True
                logger.info(f"GPU monitoring initialized for {self.device_count} devices")
                
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitoring: {e}")
    
    def get_gpu_memory_info(self, device_id: int = 0) -> Dict[str, float]:
        """Get GPU memory information for specific device"""
        if not self.initialized or device_id >= self.device_count:
            return {}
        
        try:
            handle = self.device_handles[device_id]
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            total_mb = memory_info.total / (1024 * 1024)
            used_mb = memory_info.used / (1024 * 1024)
            free_mb = memory_info.free / (1024 * 1024)
            
            return {
                'total_mb': total_mb,
                'used_mb': used_mb,
                'free_mb': free_mb,
                'percent_used': (used_mb / total_mb) * 100
            }
            
        except Exception as e:
            logger.error(f"Error getting GPU memory info: {e}")
            return {}
    
    def get_gpu_temperature(self, device_id: int = 0) -> Optional[float]:
        """Get GPU temperature"""
        if not self.initialized or device_id >= self.device_count:
            return None
        
        try:
            handle = self.device_handles[device_id]
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            return float(temp)
            
        except Exception as e:
            logger.error(f"Error getting GPU temperature: {e}")
            return None
    
    def get_gpu_processes(self, device_id: int = 0) -> List[Dict[str, Any]]:
        """Get processes using GPU"""
        if not self.initialized or device_id >= self.device_count:
            return []
        
        try:
            handle = self.device_handles[device_id]
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            
            process_info = []
            for proc in processes:
                process_info.append({
                    'pid': proc.pid,
                    'memory_mb': proc.usedGpuMemory / (1024 * 1024),
                })
            
            return process_info
            
        except Exception as e:
            logger.error(f"Error getting GPU processes: {e}")
            return []

class TorchMemoryMonitor:
    """PyTorch-specific memory monitoring"""
    
    def __init__(self):
        self.available = TORCH_AVAILABLE and torch.cuda.is_available()
    
    def get_torch_memory_info(self, device_id: int = 0) -> Dict[str, float]:
        """Get PyTorch CUDA memory information"""
        if not self.available:
            return {}
        
        try:
            device = f"cuda:{device_id}"
            allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
            cached = torch.cuda.memory_reserved(device) / (1024 * 1024)
            
            # Get memory summary
            memory_summary = torch.cuda.memory_summary(device)
            
            return {
                'allocated_mb': allocated,
                'cached_mb': cached,
                'max_allocated_mb': torch.cuda.max_memory_allocated(device) / (1024 * 1024),
                'max_cached_mb': torch.cuda.max_memory_reserved(device) / (1024 * 1024),
                'memory_summary': memory_summary
            }
            
        except Exception as e:
            logger.error(f"Error getting PyTorch memory info: {e}")
            return {}
    
    def reset_peak_stats(self, device_id: int = 0) -> None:
        """Reset peak memory statistics"""
        if self.available:
            try:
                device = f"cuda:{device_id}"
                torch.cuda.reset_peak_memory_stats(device)
                logger.info(f"Reset PyTorch peak memory stats for {device}")
            except Exception as e:
                logger.error(f"Error resetting PyTorch memory stats: {e}")
    
    def empty_cache(self, device_id: int = 0) -> None:
        """Empty PyTorch CUDA cache"""
        if self.available:
            try:
                torch.cuda.empty_cache()
                logger.info("Emptied PyTorch CUDA cache")
            except Exception as e:
                logger.error(f"Error emptying PyTorch cache: {e}")

class MemoryProfiler:
    """Main memory profiler class"""
    
    def __init__(self, output_path: str = None):
        self.output_path = Path(output_path) if output_path else Path("memory_profile.json")
        self.snapshots: List[MemorySnapshot] = []
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Initialize monitors
        self.gpu_monitor = GPUMonitor()
        self.torch_monitor = TorchMemoryMonitor()
        
        # Get process info
        self.process = psutil.Process()
        self.start_time = datetime.now()
        
        logger.info("Memory profiler initialized")
    
    def take_snapshot(self, metadata: Dict[str, Any] = None) -> MemorySnapshot:
        """Take a memory usage snapshot"""
        try:
            # System memory
            system_memory = psutil.virtual_memory()
            
            # Process memory
            process_memory = self.process.memory_info()
            
            # GPU memory
            gpu_memory_info = self.gpu_monitor.get_gpu_memory_info()
            gpu_temp = self.gpu_monitor.get_gpu_temperature()
            
            # Create snapshot
            snapshot = MemorySnapshot(
                timestamp=datetime.now(),
                system_memory_mb=system_memory.used / (1024 * 1024),
                system_memory_percent=system_memory.percent,
                gpu_memory_mb=gpu_memory_info.get('used_mb'),
                gpu_memory_percent=gpu_memory_info.get('percent_used'),
                gpu_temperature=gpu_temp,
                process_memory_mb=process_memory.rss / (1024 * 1024),
                process_memory_percent=self.process.memory_percent()
            )
            
            # Add metadata if provided
            if metadata:
                for key, value in metadata.items():
                    if hasattr(snapshot, key):
                        setattr(snapshot, key, value)
            
            self.snapshots.append(snapshot)
            return snapshot
            
        except Exception as e:
            logger.error(f"Error taking memory snapshot: {e}")
            raise
    
    def start_monitoring(self, interval_seconds: float = 5.0) -> None:
        """Start continuous memory monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        
        def monitor_loop():
            logger.info(f"Started memory monitoring (interval: {interval_seconds}s)")
            
            while self.monitoring_active:
                try:
                    self.take_snapshot()
                    time.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(interval_seconds)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop continuous memory monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        logger.info("Stopped memory monitoring")
    
    def analyze_memory_usage(self) -> MemoryStats:
        """Analyze memory usage patterns"""
        if not self.snapshots:
            raise ValueError("No snapshots available for analysis")
        
        # System memory analysis
        system_memory_values = [s.system_memory_mb for s in self.snapshots]
        avg_system = sum(system_memory_values) / len(system_memory_values)
        peak_system = max(system_memory_values)
        
        # GPU memory analysis
        gpu_memory_values = [s.gpu_memory_mb for s in self.snapshots if s.gpu_memory_mb is not None]
        avg_gpu = sum(gpu_memory_values) / len(gpu_memory_values) if gpu_memory_values else None
        peak_gpu = max(gpu_memory_values) if gpu_memory_values else None
        
        # Memory growth analysis
        if len(self.snapshots) >= 2:
            time_span_hours = (self.snapshots[-1].timestamp - self.snapshots[0].timestamp).total_seconds() / 3600
            memory_growth = self.snapshots[-1].system_memory_mb - self.snapshots[0].system_memory_mb
            growth_rate = memory_growth / max(time_span_hours, 0.1)  # Avoid division by zero
        else:
            growth_rate = 0.0
        
        # Leak detection (simple heuristic)
        leak_detected = growth_rate > 100  # More than 100MB/hour growth
        
        # Efficiency score (inverse of memory waste)
        efficiency_score = 1.0 - min(peak_system / (avg_system * 2), 1.0) if avg_system > 0 else 0.0
        
        return MemoryStats(
            avg_system_memory_mb=avg_system,
            peak_system_memory_mb=peak_system,
            avg_gpu_memory_mb=avg_gpu,
            peak_gpu_memory_mb=peak_gpu,
            memory_growth_rate_mb_per_hour=growth_rate,
            potential_leak_detected=leak_detected,
            efficiency_score=efficiency_score
        )
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory usage report"""
        if not self.snapshots:
            return {"error": "No data available"}
        
        try:
            stats = self.analyze_memory_usage()
            
            # Recent snapshots for trend analysis
            recent_snapshots = self.snapshots[-10:] if len(self.snapshots) > 10 else self.snapshots
            
            # GPU process information
            gpu_processes = self.gpu_monitor.get_gpu_processes()
            
            # PyTorch memory info
            torch_memory = self.torch_monitor.get_torch_memory_info()
            
            report = {
                "profile_duration_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
                "total_snapshots": len(self.snapshots),
                "analysis": asdict(stats),
                "current_usage": {
                    "system_memory_mb": recent_snapshots[-1].system_memory_mb,
                    "system_memory_percent": recent_snapshots[-1].system_memory_percent,
                    "gpu_memory_mb": recent_snapshots[-1].gpu_memory_mb,
                    "gpu_memory_percent": recent_snapshots[-1].gpu_memory_percent,
                    "process_memory_mb": recent_snapshots[-1].process_memory_mb,
                    "gpu_temperature": recent_snapshots[-1].gpu_temperature,
                },
                "gpu_processes": gpu_processes,
                "torch_memory": torch_memory,
                "recent_trend": [asdict(s) for s in recent_snapshots],
                "recommendations": self._generate_recommendations(stats)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, stats: MemoryStats) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if stats.potential_leak_detected:
            recommendations.append("Potential memory leak detected. Monitor for continuous growth.")
        
        if stats.avg_gpu_memory_mb and stats.peak_gpu_memory_mb:
            if stats.peak_gpu_memory_mb > stats.avg_gpu_memory_mb * 1.5:
                recommendations.append("High GPU memory spikes detected. Consider reducing batch sizes.")
        
        if stats.efficiency_score < 0.5:
            recommendations.append("Low memory efficiency. Consider optimizing memory allocation patterns.")
        
        if stats.avg_system_memory_mb > 16000:  # 16GB
            recommendations.append("High system memory usage. Monitor for optimization opportunities.")
        
        if not recommendations:
            recommendations.append("Memory usage appears optimal.")
        
        return recommendations
    
    def save_profile(self, include_raw_data: bool = False) -> None:
        """Save profiling data to file"""
        try:
            report = self.generate_report()
            
            if include_raw_data:
                report["raw_snapshots"] = [asdict(s) for s in self.snapshots]
            
            # Convert datetime objects to strings for JSON serialization
            def datetime_converter(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object {obj} is not JSON serializable")
            
            with open(self.output_path, 'w') as f:
                json.dump(report, f, indent=2, default=datetime_converter)
            
            logger.info(f"Memory profile saved to {self.output_path}")
            
        except Exception as e:
            logger.error(f"Error saving profile: {e}")
            raise
    
    def cleanup_gpu_memory(self) -> Dict[str, Any]:
        """Perform GPU memory cleanup"""
        cleanup_results = {
            "actions_taken": [],
            "memory_freed_mb": 0,
            "errors": []
        }
        
        try:
            # Get memory before cleanup
            gpu_memory_before = self.gpu_monitor.get_gpu_memory_info()
            
            # Empty PyTorch cache
            if self.torch_monitor.available:
                self.torch_monitor.empty_cache()
                cleanup_results["actions_taken"].append("Emptied PyTorch CUDA cache")
            
            # Force garbage collection
            import gc
            collected = gc.collect()
            cleanup_results["actions_taken"].append(f"Garbage collection freed {collected} objects")
            
            # Get memory after cleanup
            time.sleep(1)  # Allow time for cleanup
            gpu_memory_after = self.gpu_monitor.get_gpu_memory_info()
            
            if gpu_memory_before and gpu_memory_after:
                freed_mb = gpu_memory_before.get('used_mb', 0) - gpu_memory_after.get('used_mb', 0)
                cleanup_results["memory_freed_mb"] = max(0, freed_mb)
            
        except Exception as e:
            error_msg = f"Error during GPU cleanup: {e}"
            logger.error(error_msg)
            cleanup_results["errors"].append(error_msg)
        
        return cleanup_results

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Avatar Service Memory Profiler")
    parser.add_argument('--duration', type=int, default=60,
                        help='Monitoring duration in seconds (default: 60)')
    parser.add_argument('--interval', type=float, default=5.0,
                        help='Sampling interval in seconds (default: 5.0)')
    parser.add_argument('--output', type=str, default='memory_profile.json',
                        help='Output file path (default: memory_profile.json)')
    parser.add_argument('--continuous', action='store_true',
                        help='Run continuous monitoring until interrupted')
    parser.add_argument('--cleanup', action='store_true',
                        help='Perform GPU memory cleanup')
    parser.add_argument('--report-only', action='store_true',
                        help='Generate report from existing snapshots without monitoring')
    parser.add_argument('--include-raw', action='store_true',
                        help='Include raw snapshot data in output')
    
    args = parser.parse_args()
    
    # Initialize profiler
    profiler = MemoryProfiler(args.output)
    
    try:
        if args.cleanup:
            logger.info("Performing GPU memory cleanup...")
            cleanup_results = profiler.cleanup_gpu_memory()
            print(f"Cleanup completed: {cleanup_results}")
            return 0
        
        if args.report_only:
            # Load existing data if available
            if profiler.output_path.exists():
                logger.info("Loading existing profile data...")
                # This would require implementing data loading
            else:
                logger.error("No existing profile data found")
                return 1
        else:
            # Start monitoring
            profiler.start_monitoring(args.interval)
            
            if args.continuous:
                logger.info("Starting continuous monitoring. Press Ctrl+C to stop.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Stopping monitoring...")
            else:
                logger.info(f"Monitoring for {args.duration} seconds...")
                time.sleep(args.duration)
            
            profiler.stop_monitoring()
        
        # Generate and save report
        logger.info("Generating memory usage report...")
        profiler.save_profile(args.include_raw)
        
        # Print summary
        report = profiler.generate_report()
        if "analysis" in report:
            analysis = report["analysis"]
            print(f"\nMemory Profile Summary:")
            print(f"  Duration: {report['profile_duration_hours']:.2f} hours")
            print(f"  Snapshots: {report['total_snapshots']}")
            print(f"  Avg System Memory: {analysis['avg_system_memory_mb']:.2f} MB")
            print(f"  Peak System Memory: {analysis['peak_system_memory_mb']:.2f} MB")
            if analysis['avg_gpu_memory_mb']:
                print(f"  Avg GPU Memory: {analysis['avg_gpu_memory_mb']:.2f} MB")
                print(f"  Peak GPU Memory: {analysis['peak_gpu_memory_mb']:.2f} MB")
            print(f"  Memory Growth Rate: {analysis['memory_growth_rate_mb_per_hour']:.2f} MB/hour")
            print(f"  Efficiency Score: {analysis['efficiency_score']:.2f}")
            print(f"  Potential Leak: {analysis['potential_leak_detected']}")
            
            if report["recommendations"]:
                print(f"\nRecommendations:")
                for rec in report["recommendations"]:
                    print(f"  • {rec}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
        profiler.stop_monitoring()
        return 0
    except Exception as e:
        logger.error(f"Error during profiling: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 