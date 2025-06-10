"""
Request models for Avatar Streaming Service.
Defines input data structures for API requests.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime


class TextProcessingRequest(BaseModel):
    """Request model for text-to-avatar processing."""
    
    text: str = Field(..., min_length=1, max_length=5000, description="Text content to process")
    language: str = Field(default="fa", description="Language code (fa for Persian)")
    avatar_id: str = Field(..., description="Selected avatar identifier")
    voice: str = Field(default="alloy", description="TTS voice selection")
    quality: str = Field(default="balanced", regex="^(fast|balanced|quality)$", description="Processing quality setting")
    client_id: str = Field(..., description="Client session identifier")
    
    @validator('text')
    def validate_text_content(cls, v):
        if not v.strip():
            raise ValueError("Text content cannot be empty or whitespace only")
        return v.strip()
    
    @validator('avatar_id')
    def validate_avatar_id(cls, v):
        if not v or len(v) < 8:
            raise ValueError("Avatar ID must be at least 8 characters long")
        return v


class AudioProcessingRequest(BaseModel):
    """Request model for audio-to-avatar processing."""
    
    audio_data: bytes = Field(..., description="Raw audio data")
    audio_format: str = Field(..., description="Audio format (wav, mp3, etc.)")
    avatar_id: str = Field(..., description="Selected avatar identifier")
    language: str = Field(default="fa", description="Language for STT processing")
    quality: str = Field(default="balanced", regex="^(fast|balanced|quality)$", description="Processing quality setting")
    client_id: str = Field(..., description="Client session identifier")
    enable_rag: bool = Field(default=True, description="Enable RAG knowledge integration")
    
    @validator('audio_data')
    def validate_audio_data(cls, v):
        if not v or len(v) < 1000:  # Minimum 1KB
            raise ValueError("Audio data too small or empty")
        if len(v) > 50 * 1024 * 1024:  # Maximum 50MB
            raise ValueError("Audio data too large (max 50MB)")
        return v


class AvatarSelectionRequest(BaseModel):
    """Request model for avatar selection."""
    
    avatar_id: str = Field(..., description="Avatar identifier to select")
    client_id: str = Field(..., description="Client session identifier")
    
    @validator('avatar_id')
    def validate_avatar_id(cls, v):
        if not v or len(v) < 8:
            raise ValueError("Avatar ID must be at least 8 characters long")
        return v


class WebSocketMessage(BaseModel):
    """WebSocket message structure."""
    
    message_type: str = Field(..., regex="^(text|audio|control|avatar_selection)$", description="Message type")
    message_id: str = Field(..., description="Unique message identifier")
    client_id: str = Field(..., description="Client session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    data: Dict[str, Any] = Field(..., description="Message payload")
    
    @validator('message_id')
    def validate_message_id(cls, v):
        if not v or len(v) < 8:
            raise ValueError("Message ID must be at least 8 characters long")
        return v


class ControlMessage(BaseModel):
    """Control message for WebSocket communication."""
    
    action: str = Field(..., regex="^(start|stop|pause|resume|quality_change)$", description="Control action")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Action parameters")


class QualitySettings(BaseModel):
    """Quality settings for processing."""
    
    video_quality: str = Field(default="balanced", regex="^(fast|balanced|quality)$", description="Video processing quality")
    audio_quality: str = Field(default="standard", regex="^(low|standard|high)$", description="Audio processing quality")
    chunk_size: int = Field(default=10, ge=5, le=30, description="Chunk size in seconds")
    frame_rate: int = Field(default=25, ge=15, le=30, description="Output video frame rate")


class ClientSettings(BaseModel):
    """Client-specific settings."""
    
    buffer_size: int = Field(default=3, ge=1, le=10, description="Client buffer size in chunks")
    max_latency: float = Field(default=1.0, ge=0.1, le=5.0, description="Maximum acceptable latency in seconds")
    preferred_quality: str = Field(default="balanced", description="Preferred quality setting")
    enable_compression: bool = Field(default=True, description="Enable video compression")


class SessionInitRequest(BaseModel):
    """Session initialization request."""
    
    client_id: str = Field(..., description="Client identifier")
    user_agent: str = Field(..., description="Client user agent")
    capabilities: Dict[str, Any] = Field(default_factory=dict, description="Client capabilities")
    preferred_settings: Optional[ClientSettings] = Field(default=None, description="Client preferences")
    
    @validator('client_id')
    def validate_client_id(cls, v):
        if not v or len(v) < 8:
            raise ValueError("Client ID must be at least 8 characters long")
        return v


class HealthCheckRequest(BaseModel):
    """Health check request model."""
    
    check_models: bool = Field(default=True, description="Check model loading status")
    check_avatars: bool = Field(default=True, description="Check avatar cache status")
    check_gpu: bool = Field(default=True, description="Check GPU availability")
    detailed: bool = Field(default=False, description="Return detailed health information")


class MetricsRequest(BaseModel):
    """Metrics request model."""
    
    time_range: str = Field(default="1h", regex="^(5m|15m|1h|4h|24h)$", description="Metrics time range")
    include_performance: bool = Field(default=True, description="Include performance metrics")
    include_usage: bool = Field(default=True, description="Include usage statistics")
    include_errors: bool = Field(default=False, description="Include error statistics") 