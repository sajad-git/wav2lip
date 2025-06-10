"""
Chunk Manager - Sequential chunk processing with pre-loaded models + cached faces
Manages efficient chunk processing workflow with timing optimization
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass
from queue import Queue, PriorityQueue
from enum import Enum
import numpy as np

from app.config.settings import Settings
from app.core.audio_processor import AudioProcessor
from app.core.silence_detector import PersianSilenceDetector, ChunkBoundary
from app.core.face_cache_manager import FaceCacheManager, CachedFaceData
from app.models.chunk_models import AudioChunk, VideoChunk, ChunkMetadata

class ChunkStatus(Enum):
    """Chunk processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class ChunkPriority(Enum):
    """Chunk processing priority levels"""
    HIGH = 1
    NORMAL = 2
    LOW = 3

@dataclass
class ChunkTask:
    """Chunk processing task"""
    chunk_id: str
    audio_data: bytes
    avatar_id: str
    metadata: ChunkMetadata
    priority: ChunkPriority
    retry_count: int = 0
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    def __lt__(self, other):
        """Priority queue comparison"""
        return self.priority.value < other.priority.value

@dataclass
class ProcessingTimeline:
    """Timeline for chunk processing coordination"""
    total_chunks: int
    target_latency: float
    max_latency: float
    chunk_durations: List[float]
    processing_schedule: List[float]

class TimingController:
    """Controls chunk timing for smooth playback"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.target_buffer_size = 3  # 3 chunks ahead
        self.max_processing_time = 2.0  # 2 seconds max per chunk
        
    def calculate_optimal_timing(
        self,
        chunk_boundaries: List[ChunkBoundary],
        target_first_chunk_latency: float = 0.5
    ) -> ProcessingTimeline:
        """Calculate optimal processing timeline"""
        total_chunks = len(chunk_boundaries)
        chunk_durations = [boundary.duration for boundary in chunk_boundaries]
        
        # Calculate processing schedule
        processing_schedule = []
        current_time = 0.0
        
        for i, duration in enumerate(chunk_durations):
            if i == 0:
                # First chunk - target latency
                processing_schedule.append(target_first_chunk_latency)
            else:
                # Subsequent chunks - based on playback timing
                processing_schedule.append(current_time + duration - 0.5)
            
            current_time += duration
        
        return ProcessingTimeline(
            total_chunks=total_chunks,
            target_latency=target_first_chunk_latency,
            max_latency=target_first_chunk_latency * 2,
            chunk_durations=chunk_durations,
            processing_schedule=processing_schedule
        )

class BufferManager:
    """Manages client buffer coordination"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client_buffers: Dict[str, Dict[str, Any]] = {}
        
    def update_client_buffer_status(
        self,
        client_id: str,
        buffer_level: int,
        buffer_capacity: int
    ):
        """Update client buffer status"""
        self.client_buffers[client_id] = {
            'level': buffer_level,
            'capacity': buffer_capacity,
            'utilization': buffer_level / buffer_capacity,
            'last_update': time.time()
        }
    
    def should_pause_processing(self, client_id: str) -> bool:
        """Check if processing should be paused for client"""
        if client_id not in self.client_buffers:
            return False
        
        buffer_info = self.client_buffers[client_id]
        return buffer_info['utilization'] > 0.9  # Pause if >90% full
    
    def get_recommended_chunk_count(self, client_id: str) -> int:
        """Get recommended number of chunks to process"""
        if client_id not in self.client_buffers:
            return 3  # Default
        
        buffer_info = self.client_buffers[client_id]
        utilization = buffer_info['utilization']
        
        if utilization < 0.3:
            return 5  # Aggressive processing
        elif utilization < 0.6:
            return 3  # Normal processing
        else:
            return 1  # Conservative processing

class SequentialChunkProcessor:
    """Sequential chunk processing with pre-loaded models + cached faces"""
    
    def __init__(
        self,
        model_instances: Dict[str, Any],
        face_cache_manager: FaceCacheManager
    ):
        self.settings = Settings()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.model_instances = model_instances
        self.face_cache_manager = face_cache_manager
        self.audio_processor = AudioProcessor()
        self.silence_detector = PersianSilenceDetector()
        
        # Processing management
        self.processing_queue: PriorityQueue[ChunkTask] = PriorityQueue()
        self.timing_controller = TimingController()
        self.buffer_manager = BufferManager()
        
        # State tracking
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.processing_metrics: Dict[str, Any] = {}
        
        # Performance settings
        self.max_concurrent_chunks = 3
        self.chunk_timeout = 5.0  # 5 seconds
        self.retry_limit = 2
        
    async def process_chunk_sequence(
        self,
        audio_data: bytes,
        avatar_id: str,
        client_id: str,
        session_id: str
    ) -> AsyncGenerator[VideoChunk, None]:
        """
        Process audio chunks sequentially with pre-loaded models + cached face data
        
        Args:
            audio_data: Complete audio data
            avatar_id: Avatar identifier for cached face data
            client_id: Client identifier for buffer management
            session_id: Session identifier for tracking
            
        Yields:
            Processed video chunks in sequence
        """
        try:
            self.logger.info(f"Starting chunk sequence processing for session {session_id}")
            
            # Step 1: Validate avatar readiness
            if not await self.validate_avatar_readiness(avatar_id):
                raise ValueError(f"Avatar {avatar_id} not ready for processing")
            
            # Step 2: Detect optimal chunk boundaries
            chunk_boundaries = await self.silence_detector.detect_optimal_chunk_boundaries(
                audio_data=audio_data,
                target_chunk_duration=5.0,
                max_chunk_duration=15.0,
                min_chunk_duration=2.0
            )
            
            # Step 3: Create processing timeline
            timeline = self.timing_controller.calculate_optimal_timing(
                chunk_boundaries=chunk_boundaries,
                target_first_chunk_latency=0.5
            )
            
            # Step 4: Split audio into chunks
            audio_chunks = await self._create_audio_chunks(
                audio_data, chunk_boundaries, avatar_id, session_id
            )
            
            # Step 5: Initialize session tracking
            self._initialize_session(session_id, timeline, len(audio_chunks))
            
            # Step 6: Process chunks sequentially
            async for video_chunk in self._process_chunks_with_timing(
                audio_chunks, timeline, client_id, session_id
            ):
                yield video_chunk
                
        except Exception as e:
            self.logger.error(f"Chunk sequence processing failed: {str(e)}")
            await self._handle_sequence_error(session_id, e)
            raise
    
    async def validate_avatar_readiness(self, avatar_id: str) -> bool:
        """
        Validate avatar is ready for immediate processing
        
        Args:
            avatar_id: Avatar identifier
            
        Returns:
            True if avatar is ready, False otherwise
        """
        try:
            # Check face cache availability
            cached_face_data = await self.face_cache_manager.retrieve_face_cache(avatar_id)
            
            if not cached_face_data:
                self.logger.warning(f"Face cache not available for avatar {avatar_id}")
                return False
            
            # Validate cache integrity
            if not self.face_cache_manager.validate_cache_integrity(cached_face_data):
                self.logger.warning(f"Face cache integrity check failed for avatar {avatar_id}")
                return False
            
            # Pre-load cache if not in memory
            if not self.face_cache_manager.is_in_memory_cache(avatar_id):
                await self.face_cache_manager.load_to_memory_cache(avatar_id)
            
            self.logger.debug(f"Avatar {avatar_id} validated and ready for processing")
            return True
            
        except Exception as e:
            self.logger.error(f"Avatar readiness validation failed: {str(e)}")
            return False
    
    async def _create_audio_chunks(
        self,
        audio_data: bytes,
        chunk_boundaries: List[ChunkBoundary],
        avatar_id: str,
        session_id: str
    ) -> List[AudioChunk]:
        """Create audio chunks from boundaries"""
        audio_chunks = []
        
        # Load audio for processing
        import librosa
        import io
        audio_array, sr = librosa.load(
            io.BytesIO(audio_data),
            sr=self.audio_processor.target_sample_rate
        )
        
        for i, boundary in enumerate(chunk_boundaries):
            # Extract audio segment
            start_sample = int(boundary.start_time * sr)
            end_sample = int(boundary.end_time * sr)
            chunk_audio = audio_array[start_sample:end_sample]
            
            # Convert to bytes
            chunk_bytes = self.audio_processor._array_to_wav_bytes(chunk_audio)
            
            # Create chunk metadata
            metadata = ChunkMetadata(
                processing_time=0.0,
                model_used="wav2lip_preloaded",
                avatar_id=avatar_id,
                face_cache_hit=True,
                quality_settings="high",
                gpu_memory_used=0,
                timestamp_created=time.time()
            )
            
            # Create audio chunk
            audio_chunk = AudioChunk(
                chunk_id=f"{session_id}_chunk_{i:03d}",
                audio_data=chunk_bytes,
                duration_seconds=boundary.duration,
                start_time=boundary.start_time,
                end_time=boundary.end_time,
                sample_rate=sr,
                metadata=metadata,
                quality_metrics={}
            )
            
            audio_chunks.append(audio_chunk)
        
        return audio_chunks
    
    async def _process_chunks_with_timing(
        self,
        audio_chunks: List[AudioChunk],
        timeline: ProcessingTimeline,
        client_id: str,
        session_id: str
    ) -> AsyncGenerator[VideoChunk, None]:
        """Process chunks with optimal timing"""
        
        # Create processing tasks
        tasks = []
        for i, audio_chunk in enumerate(audio_chunks):
            task = ChunkTask(
                chunk_id=audio_chunk.chunk_id,
                audio_data=audio_chunk.audio_data,
                avatar_id=audio_chunk.metadata.avatar_id,
                metadata=audio_chunk.metadata,
                priority=ChunkPriority.HIGH if i == 0 else ChunkPriority.NORMAL
            )
            tasks.append(task)
        
        # Process first chunk immediately for low latency
        first_chunk = await self._process_single_chunk(tasks[0])
        if first_chunk:
            self.logger.info(f"First chunk processed in {time.time() - tasks[0].created_at:.3f}s")
            yield first_chunk
        
        # Process remaining chunks with timing coordination
        for i, task in enumerate(tasks[1:], 1):
            # Check client buffer status
            if self.buffer_manager.should_pause_processing(client_id):
                self.logger.debug(f"Pausing processing for client {client_id} - buffer full")
                await asyncio.sleep(0.1)
            
            # Process chunk
            video_chunk = await self._process_single_chunk(task)
            if video_chunk:
                # Update session metrics
                self._update_session_metrics(session_id, i + 1, len(tasks))
                yield video_chunk
            else:
                self.logger.warning(f"Failed to process chunk {task.chunk_id}")
    
    async def _process_single_chunk(self, task: ChunkTask) -> Optional[VideoChunk]:
        """Process a single chunk with error handling"""
        try:
            start_time = time.time()
            
            # Get cached face data (no detection delay)
            cached_face_data = await self.face_cache_manager.retrieve_face_cache(task.avatar_id)
            if not cached_face_data:
                raise ValueError(f"Face cache not available for avatar {task.avatar_id}")
            
            # Get pre-loaded model instance (no loading delay)
            wav2lip_model = self.model_instances.get("wav2lip")
            if not wav2lip_model:
                raise ValueError("Wav2Lip model not available")
            
            # Process chunk using cached models and face data
            video_chunk = await self._run_wav2lip_inference(
                task, cached_face_data, wav2lip_model
            )
            
            # Update timing metadata
            processing_time = time.time() - start_time
            task.metadata.processing_time = processing_time
            
            self.logger.debug(f"Chunk {task.chunk_id} processed in {processing_time:.3f}s")
            return video_chunk
            
        except Exception as e:
            return await self.handle_chunk_processing_error(task, e)
    
    async def _run_wav2lip_inference(
        self,
        task: ChunkTask,
        cached_face_data: CachedFaceData,
        model_instance: Any
    ) -> VideoChunk:
        """Run Wav2Lip inference with cached data"""
        
        # This is a simplified implementation
        # In practice, this would call the actual Wav2Lip model
        
        # Simulate processing time based on chunk duration
        processing_time = min(0.5, task.metadata.duration_seconds * 0.1)
        await asyncio.sleep(processing_time)
        
        # Create mock video frames
        # In real implementation, this would be actual Wav2Lip output
        frame_count = int(task.metadata.duration_seconds * 25)  # 25 FPS
        video_frames = [
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            for _ in range(frame_count)
        ]
        
        # Create video chunk
        video_chunk = VideoChunk(
            chunk_id=task.chunk_id,
            video_frames=video_frames,
            frame_rate=25,
            duration_seconds=task.metadata.duration_seconds,
            sync_timestamp=time.time(),
            encoding_format="raw",
            compression_level=0,
            avatar_id=task.avatar_id,
            metadata={}
        )
        
        return video_chunk
    
    async def handle_chunk_processing_error(
        self,
        chunk: ChunkTask,
        error: Exception
    ) -> Optional[VideoChunk]:
        """
        Handle chunk processing errors with retry logic
        
        Args:
            chunk: Failed chunk task
            error: Error details
            
        Returns:
            Recovered video chunk or None if recovery fails
        """
        try:
            self.logger.warning(f"Chunk {chunk.chunk_id} failed: {str(error)}")
            
            # Check retry limit
            if chunk.retry_count >= self.retry_limit:
                self.logger.error(f"Chunk {chunk.chunk_id} exceeded retry limit")
                return None
            
            # Increment retry count
            chunk.retry_count += 1
            chunk.priority = ChunkPriority.HIGH  # Prioritize retry
            
            # Wait before retry
            await asyncio.sleep(0.1 * chunk.retry_count)
            
            # Retry processing
            self.logger.info(f"Retrying chunk {chunk.chunk_id} (attempt {chunk.retry_count})")
            return await self._process_single_chunk(chunk)
            
        except Exception as retry_error:
            self.logger.error(f"Chunk retry failed: {str(retry_error)}")
            return None
    
    def _initialize_session(
        self,
        session_id: str,
        timeline: ProcessingTimeline,
        chunk_count: int
    ):
        """Initialize session tracking"""
        self.active_sessions[session_id] = {
            'timeline': timeline,
            'total_chunks': chunk_count,
            'processed_chunks': 0,
            'failed_chunks': 0,
            'start_time': time.time(),
            'last_chunk_time': 0.0,
            'average_processing_time': 0.0
        }
    
    def _update_session_metrics(
        self,
        session_id: str,
        processed_count: int,
        total_count: int
    ):
        """Update session processing metrics"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session['processed_chunks'] = processed_count
            session['last_chunk_time'] = time.time()
            
            # Calculate average processing time
            elapsed_time = time.time() - session['start_time']
            session['average_processing_time'] = elapsed_time / processed_count
    
    async def _handle_sequence_error(self, session_id: str, error: Exception):
        """Handle sequence-level errors"""
        self.logger.error(f"Sequence processing failed for session {session_id}: {str(error)}")
        
        # Clean up session
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current session status"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        return {
            'session_id': session_id,
            'total_chunks': session['total_chunks'],
            'processed_chunks': session['processed_chunks'],
            'failed_chunks': session['failed_chunks'],
            'progress_percentage': (session['processed_chunks'] / session['total_chunks']) * 100,
            'average_processing_time': session['average_processing_time'],
            'estimated_completion': session['start_time'] + (
                session['average_processing_time'] * session['total_chunks']
            )
        }
    
    def optimize_chunk_timing(
        self,
        chunks: List[ChunkMetadata],
        target_latency: float = 0.5
    ) -> List[ChunkMetadata]:
        """
        Optimize chunk timing for minimal gaps
        
        Args:
            chunks: List of chunk metadata
            target_latency: Target latency for first chunk
            
        Returns:
            Optimized chunk metadata with timing
        """
        optimized_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Calculate optimal start time
            if i == 0:
                optimal_start = target_latency
            else:
                previous_chunk = optimized_chunks[i - 1]
                optimal_start = previous_chunk.processing_time + chunk.processing_time
            
            # Create optimized metadata
            optimized_metadata = ChunkMetadata(
                processing_time=chunk.processing_time,
                model_used=chunk.model_used,
                avatar_id=chunk.avatar_id,
                face_cache_hit=chunk.face_cache_hit,
                quality_settings=chunk.quality_settings,
                gpu_memory_used=chunk.gpu_memory_used,
                timestamp_created=optimal_start
            )
            
            optimized_chunks.append(optimized_metadata)
        
        return optimized_chunks 