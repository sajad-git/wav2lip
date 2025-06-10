"""
Buffer Management Component
Manages client-side buffering and flow control for smooth streaming
"""

import asyncio
import logging
import time
from collections import deque
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from ..models.chunk_models import VideoChunk, AudioChunk, ChunkMetadata
from ..utils.error_handler import GlobalErrorHandler


class BufferState(Enum):
    """Buffer state enumeration"""
    EMPTY = "empty"
    LOW = "low"
    OPTIMAL = "optimal"
    HIGH = "high"
    FULL = "full"
    OVERFLOW = "overflow"


@dataclass
class BufferConfiguration:
    """Buffer configuration parameters"""
    min_buffer_size: int = 2  # Minimum chunks to buffer
    target_buffer_size: int = 5  # Target optimal buffer size
    max_buffer_size: int = 10  # Maximum buffer capacity
    
    # Timing thresholds (seconds)
    low_threshold: float = 1.0  # Buffer low warning threshold
    high_threshold: float = 3.0  # Buffer high threshold
    
    # Flow control parameters
    enable_adaptive_buffering: bool = True
    enable_quality_adaptation: bool = True
    enable_predictive_buffering: bool = True
    
    # Performance settings
    buffer_check_interval: float = 0.1  # How often to check buffer status
    flow_control_sensitivity: float = 0.5  # Flow control responsiveness


@dataclass
class BufferMetrics:
    """Buffer performance metrics"""
    total_chunks_buffered: int = 0
    buffer_underruns: int = 0
    buffer_overflows: int = 0
    average_buffer_level: float = 0.0
    peak_buffer_level: int = 0
    buffer_efficiency: float = 0.0  # % of time in optimal range
    
    # Timing metrics
    total_buffer_time: float = 0.0
    average_chunk_duration: float = 0.0
    playback_continuity: float = 1.0  # % of smooth playback
    
    # Flow control metrics
    flow_control_adjustments: int = 0
    quality_adaptations: int = 0
    predictive_hits: int = 0
    predictive_misses: int = 0


@dataclass
class ChunkBuffer:
    """Individual chunk buffer with metadata"""
    chunks: deque = field(default_factory=deque)
    total_duration: float = 0.0
    earliest_timestamp: float = 0.0
    latest_timestamp: float = 0.0
    average_chunk_size: int = 0
    state: BufferState = BufferState.EMPTY


class FlowController:
    """Flow control for adaptive streaming"""
    
    def __init__(self, config: BufferConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Flow control state
        self.current_flow_rate = 1.0  # Normal rate
        self.target_flow_rate = 1.0
        self.adjustment_step = 0.1
        self.min_flow_rate = 0.5
        self.max_flow_rate = 2.0
        
    def calculate_target_flow_rate(self, buffer_state: BufferState, 
                                 buffer_level: int, target_level: int) -> float:
        """Calculate target flow rate based on buffer state"""
        if not self.config.enable_adaptive_buffering:
            return 1.0
            
        if buffer_state == BufferState.EMPTY:
            return self.max_flow_rate  # Maximum speed to fill buffer
        elif buffer_state == BufferState.LOW:
            return min(1.5, self.max_flow_rate)  # Increase rate
        elif buffer_state == BufferState.HIGH:
            return max(0.7, self.min_flow_rate)  # Decrease rate
        elif buffer_state == BufferState.FULL:
            return self.min_flow_rate  # Minimum rate to prevent overflow
        else:
            return 1.0  # Normal rate for optimal state
    
    def update_flow_rate(self, target_rate: float) -> float:
        """Update flow rate with smoothing"""
        # Apply smoothing to prevent abrupt changes
        rate_diff = target_rate - self.current_flow_rate
        max_change = self.adjustment_step * self.config.flow_control_sensitivity
        
        if abs(rate_diff) <= max_change:
            self.current_flow_rate = target_rate
        else:
            change = max_change if rate_diff > 0 else -max_change
            self.current_flow_rate += change
            
        # Ensure rate stays within bounds
        self.current_flow_rate = max(
            self.min_flow_rate, 
            min(self.max_flow_rate, self.current_flow_rate)
        )
        
        return self.current_flow_rate


class PredictiveBuffering:
    """Predictive buffering based on usage patterns"""
    
    def __init__(self, config: BufferConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Prediction state
        self.usage_history: List[Tuple[float, int]] = []  # (timestamp, buffer_level)
        self.prediction_window = 30.0  # seconds
        self.prediction_accuracy = 0.5
        
    def record_usage(self, timestamp: float, buffer_level: int) -> None:
        """Record buffer usage for prediction"""
        self.usage_history.append((timestamp, buffer_level))
        
        # Keep only recent history
        cutoff_time = timestamp - self.prediction_window
        self.usage_history = [
            (t, level) for t, level in self.usage_history if t >= cutoff_time
        ]
    
    def predict_buffer_needs(self, current_timestamp: float) -> Optional[int]:
        """Predict future buffer needs"""
        if not self.config.enable_predictive_buffering or len(self.usage_history) < 10:
            return None
            
        try:
            # Simple trend analysis
            recent_levels = [level for _, level in self.usage_history[-10:]]
            if len(recent_levels) < 5:
                return None
                
            # Calculate trend
            trend = (recent_levels[-1] - recent_levels[0]) / len(recent_levels)
            
            # Predict buffer level in next few seconds
            prediction_horizon = 2.0  # seconds
            predicted_level = recent_levels[-1] + (trend * prediction_horizon)
            
            return max(0, int(predicted_level))
            
        except Exception as e:
            self.logger.warning(f"Prediction calculation failed: {str(e)}")
            return None


class ClientBufferManager:
    """Client-side buffer management"""
    
    def __init__(self, client_id: str, config: Optional[BufferConfiguration] = None):
        self.client_id = client_id
        self.config = config or BufferConfiguration()
        self.logger = logging.getLogger(__name__)
        
        # Buffer components
        self.video_buffer = ChunkBuffer()
        self.audio_buffer = ChunkBuffer()
        self.flow_controller = FlowController(self.config)
        self.predictive_buffering = PredictiveBuffering(self.config)
        
        # Metrics and monitoring
        self.metrics = BufferMetrics()
        self.last_buffer_check = time.time()
        self.buffer_monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Flow control state
        self.flow_control_callbacks = []
        self.quality_adaptation_callbacks = []
        
    async def initialize(self) -> None:
        """Initialize buffer manager"""
        self.buffer_monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitor_buffer_status())
        self.logger.info(f"✅ Buffer manager initialized for client {self.client_id}")
    
    async def shutdown(self) -> None:
        """Shutdown buffer manager"""
        self.buffer_monitoring_active = False
        
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
                
        self.logger.info(f"🛑 Buffer manager shut down for client {self.client_id}")
    
    def add_video_chunk(self, chunk: VideoChunk) -> bool:
        """Add video chunk to buffer"""
        try:
            # Check buffer capacity
            if len(self.video_buffer.chunks) >= self.config.max_buffer_size:
                self.logger.warning(f"Video buffer full for client {self.client_id}")
                self.metrics.buffer_overflows += 1
                return False
            
            # Add chunk to buffer
            self.video_buffer.chunks.append(chunk)
            self.video_buffer.total_duration += chunk.duration_seconds
            
            # Update timestamps
            chunk_timestamp = time.time()
            if not self.video_buffer.earliest_timestamp:
                self.video_buffer.earliest_timestamp = chunk_timestamp
            self.video_buffer.latest_timestamp = chunk_timestamp
            
            # Update metrics
            self.metrics.total_chunks_buffered += 1
            self._update_buffer_state()
            
            self.logger.debug(f"Added video chunk {chunk.chunk_id} to buffer")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add video chunk: {str(e)}")
            return False
    
    def get_next_video_chunk(self) -> Optional[VideoChunk]:
        """Get next video chunk for playback"""
        if not self.video_buffer.chunks:
            self.logger.debug(f"No video chunks available for client {self.client_id}")
            if self.video_buffer.state != BufferState.EMPTY:
                self.metrics.buffer_underruns += 1
                self._update_buffer_state()
            return None
        
        try:
            chunk = self.video_buffer.chunks.popleft()
            self.video_buffer.total_duration -= chunk.duration_seconds
            
            # Update buffer state
            self._update_buffer_state()
            
            self.logger.debug(f"Retrieved video chunk {chunk.chunk_id} from buffer")
            return chunk
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve video chunk: {str(e)}")
            return None
    
    def _update_buffer_state(self) -> None:
        """Update buffer state based on current level"""
        current_level = len(self.video_buffer.chunks)
        
        # Determine buffer state
        if current_level == 0:
            self.video_buffer.state = BufferState.EMPTY
        elif current_level < self.config.min_buffer_size:
            self.video_buffer.state = BufferState.LOW
        elif current_level > self.config.max_buffer_size * 0.9:
            self.video_buffer.state = BufferState.FULL
        elif current_level > self.config.target_buffer_size * 1.5:
            self.video_buffer.state = BufferState.HIGH
        else:
            self.video_buffer.state = BufferState.OPTIMAL
        
        # Update metrics
        self.metrics.peak_buffer_level = max(self.metrics.peak_buffer_level, current_level)
        
        # Record for predictive buffering
        current_time = time.time()
        self.predictive_buffering.record_usage(current_time, current_level)
    
    async def _monitor_buffer_status(self) -> None:
        """Monitor buffer status and apply flow control"""
        while self.buffer_monitoring_active:
            try:
                current_time = time.time()
                
                # Update buffer state
                self._update_buffer_state()
                
                # Calculate flow control
                current_level = len(self.video_buffer.chunks)
                target_flow_rate = self.flow_controller.calculate_target_flow_rate(
                    self.video_buffer.state, current_level, self.config.target_buffer_size
                )
                
                # Update flow rate
                new_flow_rate = self.flow_controller.update_flow_rate(target_flow_rate)
                
                # Apply flow control if rate changed significantly
                if abs(new_flow_rate - self.flow_controller.current_flow_rate) > 0.1:
                    await self._apply_flow_control(new_flow_rate)
                    self.metrics.flow_control_adjustments += 1
                
                # Check for quality adaptation needs
                if self.config.enable_quality_adaptation:
                    await self._check_quality_adaptation()
                
                # Update metrics
                self._update_metrics(current_time)
                
                # Sleep until next check
                await asyncio.sleep(self.config.buffer_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in buffer monitoring: {str(e)}")
                await asyncio.sleep(1.0)
    
    async def _apply_flow_control(self, flow_rate: float) -> None:
        """Apply flow control adjustments"""
        try:
            # Notify flow control callbacks
            for callback in self.flow_control_callbacks:
                await callback(self.client_id, flow_rate)
                
            self.logger.debug(f"Applied flow control rate {flow_rate:.2f} for client {self.client_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply flow control: {str(e)}")
    
    async def _check_quality_adaptation(self) -> None:
        """Check if quality adaptation is needed"""
        try:
            buffer_level = len(self.video_buffer.chunks)
            
            # Determine if quality adjustment is needed
            quality_adjustment = None
            
            if self.video_buffer.state == BufferState.LOW:
                quality_adjustment = "reduce"
            elif self.video_buffer.state == BufferState.HIGH:
                quality_adjustment = "increase"
            
            if quality_adjustment:
                # Notify quality adaptation callbacks
                for callback in self.quality_adaptation_callbacks:
                    await callback(self.client_id, quality_adjustment)
                    
                self.metrics.quality_adaptations += 1
                self.logger.debug(f"Applied quality adaptation: {quality_adjustment}")
                
        except Exception as e:
            self.logger.error(f"Failed to check quality adaptation: {str(e)}")
    
    def _update_metrics(self, current_time: float) -> None:
        """Update buffer metrics"""
        current_level = len(self.video_buffer.chunks)
        
        # Update average buffer level
        if self.metrics.total_chunks_buffered > 0:
            self.metrics.average_buffer_level = (
                (self.metrics.average_buffer_level * (self.metrics.total_chunks_buffered - 1) 
                 + current_level) / self.metrics.total_chunks_buffered
            )
        
        # Calculate buffer efficiency
        time_since_last_check = current_time - self.last_buffer_check
        if time_since_last_check > 0:
            self.metrics.total_buffer_time += time_since_last_check
            
            # Check if buffer is in optimal range
            is_optimal = (
                self.config.min_buffer_size <= current_level <= 
                self.config.target_buffer_size * 1.2
            )
            
            if is_optimal:
                # This time period counts as efficient
                pass  # Would update efficiency metric
        
        self.last_buffer_check = current_time
    
    def register_flow_control_callback(self, callback) -> None:
        """Register callback for flow control events"""
        self.flow_control_callbacks.append(callback)
    
    def register_quality_adaptation_callback(self, callback) -> None:
        """Register callback for quality adaptation events"""
        self.quality_adaptation_callbacks.append(callback)
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status"""
        return {
            "client_id": self.client_id,
            "video_buffer_level": len(self.video_buffer.chunks),
            "buffer_state": self.video_buffer.state.value,
            "total_duration": self.video_buffer.total_duration,
            "flow_rate": self.flow_controller.current_flow_rate,
            "metrics": {
                "total_chunks_buffered": self.metrics.total_chunks_buffered,
                "buffer_underruns": self.metrics.buffer_underruns,
                "buffer_overflows": self.metrics.buffer_overflows,
                "average_buffer_level": self.metrics.average_buffer_level,
                "peak_buffer_level": self.metrics.peak_buffer_level,
                "flow_control_adjustments": self.metrics.flow_control_adjustments,
                "quality_adaptations": self.metrics.quality_adaptations
            }
        }
    
    def reset_metrics(self) -> None:
        """Reset buffer metrics"""
        self.metrics = BufferMetrics()
        self.logger.info(f"Reset buffer metrics for client {self.client_id}")


class BufferManagerRegistry:
    """Registry for managing multiple client buffers"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client_buffers: Dict[str, ClientBufferManager] = {}
        self.global_config = BufferConfiguration()
        
    async def register_client(self, client_id: str, 
                            config: Optional[BufferConfiguration] = None) -> ClientBufferManager:
        """Register new client buffer manager"""
        if client_id in self.client_buffers:
            self.logger.warning(f"Client {client_id} already registered")
            return self.client_buffers[client_id]
        
        # Create buffer manager for client
        buffer_manager = ClientBufferManager(
            client_id=client_id,
            config=config or self.global_config
        )
        
        await buffer_manager.initialize()
        self.client_buffers[client_id] = buffer_manager
        
        self.logger.info(f"📝 Registered buffer manager for client {client_id}")
        return buffer_manager
    
    async def unregister_client(self, client_id: str) -> None:
        """Unregister client buffer manager"""
        if client_id not in self.client_buffers:
            return
        
        buffer_manager = self.client_buffers[client_id]
        await buffer_manager.shutdown()
        
        del self.client_buffers[client_id]
        self.logger.info(f"🗑️ Unregistered buffer manager for client {client_id}")
    
    def get_client_buffer(self, client_id: str) -> Optional[ClientBufferManager]:
        """Get buffer manager for specific client"""
        return self.client_buffers.get(client_id)
    
    def get_all_buffer_status(self) -> Dict[str, Dict[str, Any]]:
        """Get buffer status for all clients"""
        return {
            client_id: buffer_manager.get_buffer_status()
            for client_id, buffer_manager in self.client_buffers.items()
        }
    
    async def shutdown_all(self) -> None:
        """Shutdown all buffer managers"""
        for buffer_manager in self.client_buffers.values():
            await buffer_manager.shutdown()
        self.client_buffers.clear()
        self.logger.info("🛑 All buffer managers shut down") 