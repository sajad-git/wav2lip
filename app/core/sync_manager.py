"""
Synchronization Manager for Avatar Streaming Service.
Handles audio-video synchronization and timing coordination.
"""

import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TimingMetadata:
    """Timing metadata for chunks."""
    
    chunk_id: str
    start_time: float
    end_time: float
    duration: float
    sequence_number: int
    expected_arrival: datetime
    actual_arrival: Optional[datetime] = None
    processing_start: Optional[datetime] = None
    processing_end: Optional[datetime] = None
    transmission_start: Optional[datetime] = None
    client_buffer_target: float = 0.5  # Target buffer time on client


@dataclass
class SyncState:
    """Synchronization state tracking."""
    
    reference_time: datetime = field(default_factory=datetime.utcnow)
    last_chunk_time: float = 0.0
    cumulative_drift: float = 0.0
    chunk_count: int = 0
    average_processing_time: float = 0.0
    timing_variance: float = 0.0
    sync_quality_score: float = 1.0


class TimingCoordinator:
    """Coordinates timing across multiple processing stages."""
    
    def __init__(self):
        self.chunk_timings: Dict[str, TimingMetadata] = {}
        self.sync_states: Dict[str, SyncState] = {}
        self.timing_history: deque = deque(maxlen=100)
        self.target_frame_rate = 25.0  # FPS
        self.target_chunk_overlap = 0.1  # seconds
        self.max_drift_tolerance = 0.2  # seconds
        
    def create_timing_metadata(
        self, 
        chunk_id: str, 
        start_time: float, 
        duration: float, 
        sequence_number: int
    ) -> TimingMetadata:
        """Create timing metadata for a chunk."""
        
        end_time = start_time + duration
        expected_arrival = datetime.utcnow() + timedelta(seconds=duration * 0.8)
        
        metadata = TimingMetadata(
            chunk_id=chunk_id,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            sequence_number=sequence_number,
            expected_arrival=expected_arrival
        )
        
        self.chunk_timings[chunk_id] = metadata
        return metadata
        
    def update_processing_timing(self, chunk_id: str, stage: str, timestamp: datetime = None):
        """Update timing information for processing stages."""
        
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        if chunk_id not in self.chunk_timings:
            logger.warning(f"Timing metadata not found for chunk {chunk_id}")
            return
            
        metadata = self.chunk_timings[chunk_id]
        
        if stage == "processing_start":
            metadata.processing_start = timestamp
        elif stage == "processing_end":
            metadata.processing_end = timestamp
        elif stage == "transmission_start":
            metadata.transmission_start = timestamp
        elif stage == "actual_arrival":
            metadata.actual_arrival = timestamp
            
    def calculate_sync_drift(self, client_id: str, chunk_id: str) -> float:
        """Calculate synchronization drift for a chunk."""
        
        if chunk_id not in self.chunk_timings:
            return 0.0
            
        metadata = self.chunk_timings[chunk_id]
        
        if not metadata.actual_arrival or not metadata.expected_arrival:
            return 0.0
            
        drift = (metadata.actual_arrival - metadata.expected_arrival).total_seconds()
        
        # Update sync state
        if client_id not in self.sync_states:
            self.sync_states[client_id] = SyncState()
            
        sync_state = self.sync_states[client_id]
        sync_state.cumulative_drift += drift
        sync_state.chunk_count += 1
        
        # Calculate timing variance
        if len(self.timing_history) > 1:
            processing_times = [t.processing_time for t in self.timing_history if t.processing_time]
            if processing_times:
                sync_state.timing_variance = np.std(processing_times)
                
        return drift
        
    def optimize_chunk_timing(self, chunks: List[Any], client_buffer_status: float) -> List[Any]:
        """Optimize chunk timing based on client buffer status."""
        
        optimized_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Adjust timing based on buffer status
            if client_buffer_status < 0.3:  # Low buffer
                # Prioritize faster processing
                chunk.priority = "high"
                chunk.quality_setting = "fast"
            elif client_buffer_status > 0.8:  # High buffer
                # Allow higher quality processing
                chunk.quality_setting = "quality"
                
            # Calculate optimal delivery time
            if i > 0:
                prev_chunk = optimized_chunks[-1]
                # Ensure smooth transition
                chunk.delivery_time = prev_chunk.delivery_time + prev_chunk.duration - self.target_chunk_overlap
            else:
                chunk.delivery_time = time.time() + 0.5  # 500ms buffer
                
            optimized_chunks.append(chunk)
            
        return optimized_chunks
        
    def get_timing_recommendations(self, client_id: str) -> Dict[str, Any]:
        """Get timing recommendations for optimization."""
        
        if client_id not in self.sync_states:
            return {"status": "no_data"}
            
        sync_state = self.sync_states[client_id]
        
        recommendations = {
            "current_drift": sync_state.cumulative_drift / max(sync_state.chunk_count, 1),
            "timing_stability": 1.0 - min(sync_state.timing_variance / 0.5, 1.0),
            "sync_quality": sync_state.sync_quality_score
        }
        
        # Generate specific recommendations
        if abs(recommendations["current_drift"]) > self.max_drift_tolerance:
            recommendations["action"] = "adjust_buffer_size"
            recommendations["suggested_buffer"] = 0.8 if recommendations["current_drift"] > 0 else 0.3
        elif recommendations["timing_stability"] < 0.7:
            recommendations["action"] = "stabilize_processing"
            recommendations["suggested_quality"] = "balanced"
        else:
            recommendations["action"] = "maintain_current"
            
        return recommendations


class AudioVideoSyncManager:
    """Manages audio-video synchronization."""
    
    def __init__(self):
        self.timing_coordinator = TimingCoordinator()
        self.sync_corrections: Dict[str, float] = {}
        self.frame_timing_cache: Dict[str, List[float]] = {}
        
    def synchronize_audio_video(
        self, 
        audio_chunks: List[Any], 
        video_chunks: List[Any]
    ) -> Tuple[List[Any], List[Any]]:
        """Synchronize audio and video chunks."""
        
        if len(audio_chunks) != len(video_chunks):
            logger.warning(f"Chunk count mismatch: {len(audio_chunks)} audio, {len(video_chunks)} video")
            
        synchronized_audio = []
        synchronized_video = []
        
        for i, (audio_chunk, video_chunk) in enumerate(zip(audio_chunks, video_chunks)):
            # Ensure matching durations
            if hasattr(audio_chunk, 'duration') and hasattr(video_chunk, 'duration'):
                duration_diff = abs(audio_chunk.duration - video_chunk.duration)
                
                if duration_diff > 0.1:  # More than 100ms difference
                    logger.warning(f"Duration mismatch in chunk {i}: {duration_diff:.3f}s")
                    
                    # Adjust video to match audio
                    video_chunk = self._adjust_video_duration(video_chunk, audio_chunk.duration)
                    
            # Add synchronization metadata
            sync_id = f"sync_{i}_{int(time.time() * 1000)}"
            audio_chunk.sync_id = sync_id
            video_chunk.sync_id = sync_id
            
            # Calculate frame timing
            if hasattr(video_chunk, 'frame_count'):
                frame_times = self._calculate_frame_timing(
                    video_chunk.frame_count, 
                    audio_chunk.duration if hasattr(audio_chunk, 'duration') else 1.0
                )
                video_chunk.frame_timing = frame_times
                self.frame_timing_cache[sync_id] = frame_times
                
            synchronized_audio.append(audio_chunk)
            synchronized_video.append(video_chunk)
            
        return synchronized_audio, synchronized_video
        
    def _adjust_video_duration(self, video_chunk: Any, target_duration: float) -> Any:
        """Adjust video chunk duration to match target."""
        
        if not hasattr(video_chunk, 'frame_count') or not hasattr(video_chunk, 'frame_rate'):
            return video_chunk
            
        current_duration = video_chunk.frame_count / video_chunk.frame_rate
        speed_factor = current_duration / target_duration
        
        # Adjust frame rate to match duration
        video_chunk.adjusted_frame_rate = video_chunk.frame_rate * speed_factor
        
        # Recalculate frame timing
        video_chunk.frame_timing = self._calculate_frame_timing(
            video_chunk.frame_count, 
            target_duration
        )
        
        logger.info(f"Adjusted video duration from {current_duration:.3f}s to {target_duration:.3f}s")
        
        return video_chunk
        
    def _calculate_frame_timing(self, frame_count: int, duration: float) -> List[float]:
        """Calculate timing for individual frames."""
        
        if frame_count <= 0:
            return []
            
        frame_interval = duration / frame_count
        return [i * frame_interval for i in range(frame_count)]
        
    def validate_sync_quality(self, chunks: List[Any]) -> Dict[str, Any]:
        """Validate synchronization quality across chunks."""
        
        quality_metrics = {
            "overall_quality": 1.0,
            "timing_consistency": 1.0,
            "duration_accuracy": 1.0,
            "frame_stability": 1.0,
            "issues": []
        }
        
        if not chunks:
            return quality_metrics
            
        # Check timing consistency
        durations = [getattr(chunk, 'duration', 0) for chunk in chunks if hasattr(chunk, 'duration')]
        if durations:
            duration_variance = np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 0
            quality_metrics["timing_consistency"] = max(0.0, 1.0 - duration_variance * 2)
            
            if duration_variance > 0.2:
                quality_metrics["issues"].append("High duration variance detected")
                
        # Check frame timing stability
        frame_timings = []
        for chunk in chunks:
            if hasattr(chunk, 'frame_timing') and chunk.frame_timing:
                intervals = [chunk.frame_timing[i+1] - chunk.frame_timing[i] 
                           for i in range(len(chunk.frame_timing)-1)]
                frame_timings.extend(intervals)
                
        if frame_timings:
            frame_variance = np.std(frame_timings) / np.mean(frame_timings) if np.mean(frame_timings) > 0 else 0
            quality_metrics["frame_stability"] = max(0.0, 1.0 - frame_variance * 5)
            
            if frame_variance > 0.1:
                quality_metrics["issues"].append("Frame timing instability detected")
                
        # Calculate overall quality
        quality_metrics["overall_quality"] = (
            quality_metrics["timing_consistency"] * 0.4 +
            quality_metrics["duration_accuracy"] * 0.3 +
            quality_metrics["frame_stability"] * 0.3
        )
        
        return quality_metrics


class BufferSyncManager:
    """Manages client buffer synchronization."""
    
    def __init__(self):
        self.client_buffers: Dict[str, Dict[str, Any]] = {}
        self.sync_adjustments: Dict[str, float] = {}
        
    def update_client_buffer_status(
        self, 
        client_id: str, 
        buffer_level: float, 
        playback_position: float,
        latency: float
    ):
        """Update client buffer status."""
        
        if client_id not in self.client_buffers:
            self.client_buffers[client_id] = {
                "level": 0.0,
                "position": 0.0,
                "latency": 0.0,
                "last_update": datetime.utcnow(),
                "stability_score": 1.0
            }
            
        buffer_info = self.client_buffers[client_id]
        
        # Calculate buffer stability
        level_change = abs(buffer_level - buffer_info["level"])
        stability_factor = max(0.0, 1.0 - level_change)
        
        buffer_info.update({
            "level": buffer_level,
            "position": playback_position,
            "latency": latency,
            "last_update": datetime.utcnow(),
            "stability_score": buffer_info["stability_score"] * 0.9 + stability_factor * 0.1
        })
        
        # Calculate sync adjustment
        if buffer_level < 0.2:  # Low buffer
            self.sync_adjustments[client_id] = -0.1  # Speed up slightly
        elif buffer_level > 0.8:  # High buffer
            self.sync_adjustments[client_id] = 0.05  # Slow down slightly
        else:
            self.sync_adjustments[client_id] = 0.0  # Normal speed
            
    def get_optimal_delivery_timing(self, client_id: str, chunk_count: int) -> List[float]:
        """Calculate optimal delivery timing for chunks."""
        
        if client_id not in self.client_buffers:
            # Default timing for new clients
            return [i * 0.8 for i in range(chunk_count)]
            
        buffer_info = self.client_buffers[client_id]
        adjustment = self.sync_adjustments.get(client_id, 0.0)
        
        base_interval = 0.8 - adjustment  # Base 800ms interval
        
        return [i * base_interval for i in range(chunk_count)]
        
    def should_pause_streaming(self, client_id: str) -> bool:
        """Determine if streaming should be paused for buffer management."""
        
        if client_id not in self.client_buffers:
            return False
            
        buffer_info = self.client_buffers[client_id]
        
        # Pause if buffer is very high or very low with poor stability
        if buffer_info["level"] > 0.9:
            return True
        elif buffer_info["level"] < 0.1 and buffer_info["stability_score"] < 0.5:
            return True
            
        return False
        
    def get_sync_report(self, client_id: str) -> Dict[str, Any]:
        """Get synchronization report for client."""
        
        if client_id not in self.client_buffers:
            return {"status": "no_data"}
            
        buffer_info = self.client_buffers[client_id]
        adjustment = self.sync_adjustments.get(client_id, 0.0)
        
        return {
            "buffer_level": buffer_info["level"],
            "stability_score": buffer_info["stability_score"],
            "current_adjustment": adjustment,
            "latency": buffer_info["latency"],
            "last_update": buffer_info["last_update"],
            "status": "healthy" if buffer_info["stability_score"] > 0.7 else "unstable"
        } 