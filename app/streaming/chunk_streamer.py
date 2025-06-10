"""
Sequential Chunk Streaming Component
Manages sequential chunk delivery with timing optimization for smooth playback
"""

import asyncio
import logging
import time
from collections import deque
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from queue import Queue, Empty

from ..models.chunk_models import VideoChunk, ChunkMetadata
from ..models.response_models import StreamingStatusResponse
from ..utils.error_handler import GlobalErrorHandler


@dataclass
class StreamingMetrics:
    """Streaming performance metrics"""
    chunks_streamed: int = 0
    total_streaming_time: float = 0.0
    average_chunk_time: float = 0.0
    buffer_underruns: int = 0
    late_deliveries: int = 0
    quality_adjustments: int = 0


@dataclass
class ClientQualitySettings:
    """Per-client quality configuration"""
    target_bitrate: int = 1000000  # 1Mbps default
    max_resolution: tuple = (512, 512)
    compression_level: int = 5  # 1-10 scale
    frame_rate: int = 25
    adaptive_quality: bool = True


@dataclass
class BufferStatus:
    """Client buffer status information"""
    current_level: int = 0
    max_capacity: int = 10
    target_level: int = 3
    is_full: bool = False
    is_empty: bool = True
    last_update: float = 0.0


class TimingCoordinator:
    """Manages timing coordination across multiple clients"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.reference_timeline = 0.0
        self.client_timelines: Dict[str, float] = {}
        self.sync_tolerance = 0.1  # 100ms tolerance
        
    def register_client(self, client_id: str) -> None:
        """Register new client for timing coordination"""
        self.client_timelines[client_id] = time.time()
        self.logger.debug(f"Registered client {client_id} for timing coordination")
    
    def unregister_client(self, client_id: str) -> None:
        """Remove client from timing coordination"""
        if client_id in self.client_timelines:
            del self.client_timelines[client_id]
            self.logger.debug(f"Unregistered client {client_id} from timing coordination")
    
    def get_target_timestamp(self, client_id: str) -> float:
        """Get target timestamp for next chunk delivery"""
        if client_id not in self.client_timelines:
            return time.time()
        return self.client_timelines[client_id]
    
    def update_client_timeline(self, client_id: str, timestamp: float) -> None:
        """Update client timeline after chunk delivery"""
        if client_id in self.client_timelines:
            self.client_timelines[client_id] = timestamp


class QualityAdapter:
    """Adaptive quality management for streaming"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client_settings: Dict[str, ClientQualitySettings] = {}
        
    def register_client(self, client_id: str, settings: ClientQualitySettings) -> None:
        """Register client quality settings"""
        self.client_settings[client_id] = settings
        self.logger.debug(f"Registered quality settings for client {client_id}")
    
    def adapt_chunk_quality(self, chunk: VideoChunk, client_id: str, 
                          buffer_status: BufferStatus) -> VideoChunk:
        """Adapt chunk quality based on client status"""
        if client_id not in self.client_settings:
            return chunk
            
        settings = self.client_settings[client_id]
        
        # Skip adaptation if not enabled
        if not settings.adaptive_quality:
            return chunk
            
        # Adapt based on buffer status
        if buffer_status.current_level < buffer_status.target_level:
            # Buffer running low, reduce quality for faster processing
            chunk.compression_level = min(chunk.compression_level + 2, 10)
            self.logger.debug(f"Increased compression for client {client_id} due to low buffer")
        elif buffer_status.current_level > buffer_status.target_level + 2:
            # Buffer healthy, can increase quality
            chunk.compression_level = max(chunk.compression_level - 1, 1)
            
        return chunk


class SequentialChunkStreamer:
    """Sequential chunk delivery with timing optimization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.streaming_queues: Dict[str, Queue[VideoChunk]] = {}
        self.buffer_status: Dict[str, BufferStatus] = {}
        self.client_metrics: Dict[str, StreamingMetrics] = {}
        self.timing_coordinator = TimingCoordinator()
        self.quality_adapter = QualityAdapter()
        self.error_handler = GlobalErrorHandler()
        
        # Streaming control
        self.max_queue_size = 10
        self.streaming_active = False
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        
    async def initialize(self) -> None:
        """Initialize the chunk streamer"""
        self.streaming_active = True
        self.logger.info("✅ Sequential chunk streamer initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the chunk streamer"""
        self.streaming_active = False
        
        # Cancel all streaming tasks
        for task in self.streaming_tasks.values():
            if not task.done():
                task.cancel()
                
        # Wait for tasks to complete
        if self.streaming_tasks:
            await asyncio.gather(*self.streaming_tasks.values(), return_exceptions=True)
            
        self.logger.info("🛑 Sequential chunk streamer shut down")
    
    def register_client(self, client_id: str, 
                       quality_settings: Optional[ClientQualitySettings] = None) -> None:
        """Register new client for streaming"""
        # Initialize client streaming queue
        self.streaming_queues[client_id] = Queue(maxsize=self.max_queue_size)
        
        # Initialize buffer status
        self.buffer_status[client_id] = BufferStatus()
        
        # Initialize metrics
        self.client_metrics[client_id] = StreamingMetrics()
        
        # Register with timing coordinator
        self.timing_coordinator.register_client(client_id)
        
        # Register quality settings
        if quality_settings:
            self.quality_adapter.register_client(client_id, quality_settings)
        
        self.logger.info(f"📝 Registered client {client_id} for streaming")
    
    def unregister_client(self, client_id: str) -> None:
        """Unregister client from streaming"""
        # Cancel streaming task if active
        if client_id in self.streaming_tasks:
            task = self.streaming_tasks[client_id]
            if not task.done():
                task.cancel()
            del self.streaming_tasks[client_id]
        
        # Clean up client data
        if client_id in self.streaming_queues:
            del self.streaming_queues[client_id]
        if client_id in self.buffer_status:
            del self.buffer_status[client_id]
        if client_id in self.client_metrics:
            del self.client_metrics[client_id]
            
        # Unregister from coordinators
        self.timing_coordinator.unregister_client(client_id)
        
        self.logger.info(f"🗑️ Unregistered client {client_id} from streaming")
    
    def queue_chunk_for_streaming(self, chunk: VideoChunk, client_id: str) -> bool:
        """Queue processed chunk for sequential streaming"""
        if client_id not in self.streaming_queues:
            self.logger.error(f"Client {client_id} not registered for streaming")
            return False
            
        queue = self.streaming_queues[client_id]
        buffer_status = self.buffer_status[client_id]
        
        try:
            # Check if queue is full
            if queue.full():
                self.logger.warning(f"Streaming queue full for client {client_id}")
                self.client_metrics[client_id].buffer_underruns += 1
                return False
            
            # Add timing metadata
            chunk.metadata.queue_timestamp = time.time()
            
            # Queue the chunk
            queue.put_nowait(chunk)
            
            # Update buffer status
            buffer_status.current_level = queue.qsize()
            buffer_status.is_full = queue.full()
            buffer_status.is_empty = queue.empty()
            buffer_status.last_update = time.time()
            
            # Start streaming task if not active
            if client_id not in self.streaming_tasks:
                self.streaming_tasks[client_id] = asyncio.create_task(
                    self._stream_client_chunks(client_id)
                )
            
            self.logger.debug(f"Queued chunk {chunk.chunk_id} for client {client_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to queue chunk for client {client_id}: {str(e)}")
            return False
    
    async def _stream_client_chunks(self, client_id: str) -> None:
        """Stream chunks for a specific client"""
        queue = self.streaming_queues[client_id]
        metrics = self.client_metrics[client_id]
        
        self.logger.info(f"🎬 Started streaming for client {client_id}")
        
        try:
            while self.streaming_active:
                try:
                    # Get next chunk with timeout
                    chunk = queue.get(timeout=1.0)
                    
                    # Record streaming start time
                    stream_start = time.time()
                    
                    # Apply quality adaptation
                    buffer_status = self.buffer_status[client_id]
                    adapted_chunk = self.quality_adapter.adapt_chunk_quality(
                        chunk, client_id, buffer_status
                    )
                    
                    # Stream the chunk
                    success = await self._deliver_chunk_to_client(adapted_chunk, client_id)
                    
                    if success:
                        # Update metrics
                        stream_time = time.time() - stream_start
                        metrics.chunks_streamed += 1
                        metrics.total_streaming_time += stream_time
                        metrics.average_chunk_time = (
                            metrics.total_streaming_time / metrics.chunks_streamed
                        )
                        
                        # Update timing coordinator
                        self.timing_coordinator.update_client_timeline(
                            client_id, time.time()
                        )
                        
                        # Update buffer status
                        buffer_status.current_level = queue.qsize()
                        buffer_status.is_empty = queue.empty()
                        
                        self.logger.debug(
                            f"Streamed chunk {chunk.chunk_id} to client {client_id} "
                            f"in {stream_time:.3f}s"
                        )
                    else:
                        metrics.late_deliveries += 1
                        self.logger.warning(f"Failed to deliver chunk to client {client_id}")
                    
                    # Mark queue task as done
                    queue.task_done()
                    
                except Empty:
                    # No chunks available, continue waiting
                    continue
                except asyncio.CancelledError:
                    self.logger.info(f"Streaming cancelled for client {client_id}")
                    break
                except Exception as e:
                    self.logger.error(f"Error streaming to client {client_id}: {str(e)}")
                    await asyncio.sleep(0.1)  # Brief pause before retry
                    
        finally:
            # Clean up streaming task reference
            if client_id in self.streaming_tasks:
                del self.streaming_tasks[client_id]
            self.logger.info(f"🏁 Stopped streaming for client {client_id}")
    
    async def _deliver_chunk_to_client(self, chunk: VideoChunk, client_id: str) -> bool:
        """Deliver chunk to specific client via WebSocket"""
        try:
            # This would be implemented by the WebSocket handler
            # For now, simulate delivery
            await asyncio.sleep(0.01)  # Simulate network latency
            
            # In real implementation, this would call:
            # await websocket_handler.stream_chunk_to_client(chunk, client_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deliver chunk to client {client_id}: {str(e)}")
            return False
    
    def get_streaming_status(self, client_id: str) -> Optional[StreamingStatusResponse]:
        """Get current streaming status for client"""
        if client_id not in self.client_metrics:
            return None
            
        metrics = self.client_metrics[client_id]
        buffer_status = self.buffer_status[client_id]
        
        return StreamingStatusResponse(
            chunk_id=f"current_{int(time.time())}",
            chunks_completed=metrics.chunks_streamed,
            chunks_remaining=buffer_status.current_level,
            current_latency=metrics.average_chunk_time,
            average_latency=metrics.average_chunk_time,
            avatar_processing_time=0.0,  # Would be provided by processing pipeline
            quality_metrics=None,  # Would include quality metrics
            next_chunk_eta=metrics.average_chunk_time
        )
    
    async def coordinate_multi_client_streaming(self) -> None:
        """Manage streaming across multiple clients"""
        while self.streaming_active:
            try:
                # Monitor all client queues
                for client_id, queue in self.streaming_queues.items():
                    buffer_status = self.buffer_status[client_id]
                    
                    # Check for buffer issues
                    if buffer_status.current_level == 0 and not buffer_status.is_empty:
                        self.logger.warning(f"Buffer underrun for client {client_id}")
                        self.client_metrics[client_id].buffer_underruns += 1
                    
                    # Apply quality adjustments if needed
                    if buffer_status.current_level < buffer_status.target_level:
                        self.client_metrics[client_id].quality_adjustments += 1
                
                # Sleep before next check
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in multi-client coordination: {str(e)}")
                await asyncio.sleep(1.0)
    
    def get_all_client_metrics(self) -> Dict[str, StreamingMetrics]:
        """Get streaming metrics for all clients"""
        return self.client_metrics.copy()
    
    def reset_client_metrics(self, client_id: str) -> None:
        """Reset metrics for specific client"""
        if client_id in self.client_metrics:
            self.client_metrics[client_id] = StreamingMetrics()
            self.logger.info(f"Reset metrics for client {client_id}") 