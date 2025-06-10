"""
Chunk processing and streaming data structures
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import numpy as np


class AudioQualityMetrics(BaseModel):
    """Audio quality assessment metrics"""
    snr_ratio: float = Field(..., description="Signal-to-noise ratio")
    clarity_score: float = Field(..., description="Speech clarity assessment")
    duration_seconds: float = Field(..., description="Audio duration")
    quality_grade: str = Field(..., description="Overall quality grade")
    recommendations: List[str] = Field(default_factory=list, description="Improvement suggestions")


class ChunkMetadata(BaseModel):
    """Processing metadata for chunks"""
    processing_time: float = Field(..., description="Time to process chunk")
    model_used: str = Field(..., description="Wav2lip model identifier")
    avatar_id: str = Field(..., description="Avatar used for processing")
    face_cache_hit: bool = Field(..., description="Whether face data was cached")
    quality_settings: Dict[str, Any] = Field(default_factory=dict, description="Processing quality")
    gpu_memory_used: int = Field(..., description="GPU memory consumption")
    timestamp_created: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


class AudioChunk(BaseModel):
    """Audio chunk data model"""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    audio_data: bytes = Field(..., description="Raw audio segment")
    duration_seconds: float = Field(..., description="Chunk duration")
    start_time: float = Field(..., description="Start time in original audio")
    end_time: float = Field(..., description="End time in original audio")
    sample_rate: int = Field(..., description="Audio sample rate")
    metadata: ChunkMetadata = Field(..., description="Processing metadata")
    quality_metrics: AudioQualityMetrics = Field(..., description="Quality assessment")

    class Config:
        arbitrary_types_allowed = True


class VideoMetadata(BaseModel):
    """Video processing metadata"""
    frame_rate: int = Field(default=25, description="Video frame rate")
    encoding_format: str = Field(default="mp4", description="Video encoding format")
    compression_level: int = Field(default=1, description="Compression applied")
    color_space: str = Field(default="RGB", description="Color space")
    resolution: tuple = Field(default=(256, 256), description="Video resolution")


class VideoChunk(BaseModel):
    """Video chunk data model"""
    chunk_id: str = Field(..., description="Matching audio chunk ID")
    video_frames: Optional[List[Any]] = Field(None, description="Processed video frames")  # np.ndarray handled separately
    frame_rate: int = Field(default=25, description="Video frame rate (25 FPS)")
    duration_seconds: float = Field(..., description="Video duration")
    sync_timestamp: float = Field(..., description="Synchronization reference")
    encoding_format: str = Field(default="mp4", description="Video encoding format")
    compression_level: int = Field(default=1, description="Compression applied")
    avatar_id: str = Field(..., description="Avatar used for processing")
    metadata: VideoMetadata = Field(..., description="Processing metadata")

    class Config:
        arbitrary_types_allowed = True


class StreamingSequence(BaseModel):
    """Streaming coordination model"""
    sequence_id: str = Field(..., description="Unique sequence identifier")
    total_chunks: int = Field(..., description="Expected total chunks")
    completed_chunks: int = Field(default=0, description="Successfully processed chunks")
    current_chunk: int = Field(default=0, description="Currently streaming chunk")
    avatar_id: str = Field(..., description="Avatar used for sequence")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    processing_start: datetime = Field(default_factory=datetime.now, description="Processing start time")


class ChunkTask(BaseModel):
    """Chunk processing task model"""
    chunk_id: str = Field(..., description="Unique identifier")
    audio_data: bytes = Field(..., description="Audio segment data")
    avatar_id: str = Field(..., description="Avatar identifier for cached face data")
    metadata: ChunkMetadata = Field(..., description="Processing metadata")
    priority: int = Field(default=1, description="Processing priority")
    retry_count: int = Field(default=0, description="Retry attempt counter")
    created_at: datetime = Field(default_factory=datetime.now, description="Task creation time")

    class Config:
        arbitrary_types_allowed = True


class ProcessingTask(BaseModel):
    """Generic processing task model"""
    task_id: str = Field(..., description="Unique task identifier")
    user_id: str = Field(..., description="User identifier")
    task_type: str = Field(..., description="Type of processing task")
    avatar_id: Optional[str] = Field(None, description="Associated avatar ID")
    priority: int = Field(default=1, description="Task priority")
    estimated_duration: float = Field(..., description="Estimated processing time")
    created_at: datetime = Field(default_factory=datetime.now, description="Task creation time")
    started_at: Optional[datetime] = Field(None, description="Task start time")
    completed_at: Optional[datetime] = Field(None, description="Task completion time")


class BufferStatus(BaseModel):
    """Client buffer status model"""
    buffer_size: int = Field(..., description="Current buffer size")
    max_buffer_size: int = Field(..., description="Maximum buffer capacity")
    fill_percentage: float = Field(..., description="Buffer fill percentage")
    chunks_buffered: int = Field(..., description="Number of chunks in buffer")
    estimated_playback_time: float = Field(..., description="Estimated playback time remaining")
    buffer_health: str = Field(..., description="Buffer health status")


class QualitySettings(BaseModel):
    """Processing quality settings"""
    video_quality: str = Field(default="balanced", description="Video quality level")
    frame_rate: int = Field(default=25, description="Target frame rate")
    resolution: tuple = Field(default=(256, 256), description="Target resolution")
    compression_level: int = Field(default=1, description="Compression level")
    optimization_flags: List[str] = Field(default_factory=list, description="Optimization flags")


class SyncMetadata(BaseModel):
    """Audio-video synchronization metadata"""
    audio_offset: float = Field(default=0.0, description="Audio offset in seconds")
    video_offset: float = Field(default=0.0, description="Video offset in seconds")
    sync_confidence: float = Field(default=1.0, description="Synchronization confidence")
    drift_correction: float = Field(default=0.0, description="Applied drift correction")
    timing_reference: str = Field(..., description="Timing reference point")


class ChunkProcessingResult(BaseModel):
    """Result of chunk processing operation"""
    success: bool = Field(..., description="Processing success status")
    chunk_id: str = Field(..., description="Processed chunk identifier")
    processing_time: float = Field(..., description="Actual processing time")
    quality_score: float = Field(..., description="Output quality score")
    face_cache_used: bool = Field(..., description="Whether cached face data was used")
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")


class StreamingMetrics(BaseModel):
    """Streaming performance metrics"""
    total_chunks_streamed: int = Field(default=0, description="Total chunks streamed")
    average_chunk_size: float = Field(default=0.0, description="Average chunk size in bytes")
    streaming_rate: float = Field(default=0.0, description="Chunks per second")
    total_streaming_time: float = Field(default=0.0, description="Total streaming duration")
    buffer_underruns: int = Field(default=0, description="Number of buffer underruns")
    quality_degradations: int = Field(default=0, description="Quality degradation events")
    connection_drops: int = Field(default=0, description="Connection drop events") 