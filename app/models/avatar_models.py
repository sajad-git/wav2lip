"""
Avatar registration and management data models
"""
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field
import numpy as np


class AvatarRegistrationRequest(BaseModel):
    """Request model for avatar registration"""
    file_data: bytes = Field(..., description="Avatar file data")
    filename: str = Field(..., description="Original filename")
    avatar_name: str = Field(..., description="User-friendly name")
    user_id: str = Field(..., description="Owner user identifier")
    file_format: str = Field(..., description="File format (jpg, png, gif, mp4)")
    description: Optional[str] = Field(None, description="Avatar description")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")


class FaceDetectionSummary(BaseModel):
    """Summary of face detection results"""
    faces_detected: int = Field(..., description="Number of faces found")
    primary_face_confidence: float = Field(..., description="Main face detection confidence")
    face_consistency_score: float = Field(..., description="Consistency across frames (for video)")
    bounding_boxes: List[Tuple[int, int, int, int]] = Field(..., description="Face bounding boxes")
    landmarks: Optional[List[Any]] = Field(None, description="Facial landmarks")  # np.ndarray serialization handled separately
    quality_metrics: Dict[str, float] = Field(default_factory=dict, description="Quality assessment")


class QualityAssessment(BaseModel):
    """Avatar quality assessment"""
    overall_score: float = Field(..., description="Overall quality score (0-1)")
    face_clarity: float = Field(..., description="Face clarity score")
    lighting_quality: float = Field(..., description="Lighting quality score")
    resolution_score: float = Field(..., description="Resolution adequacy score")
    processing_ready: bool = Field(..., description="Ready for wav2lip processing")
    recommendations: List[str] = Field(default_factory=list, description="Quality improvement suggestions")


class AvatarRegistrationResponse(BaseModel):
    """Response model for avatar registration"""
    avatar_id: str = Field(..., description="Generated unique identifier")
    registration_status: str = Field(..., description="Success/failure status")
    face_detection_results: FaceDetectionSummary = Field(..., description="Face processing summary")
    quality_assessment: QualityAssessment = Field(..., description="Avatar quality metrics")
    processing_time: float = Field(..., description="Registration processing time")
    cache_status: str = Field(..., description="Face cache creation status")
    errors: List[str] = Field(default_factory=list, description="Any registration errors")
    warnings: List[str] = Field(default_factory=list, description="Non-critical warnings")


class AvatarInfo(BaseModel):
    """Avatar information model"""
    avatar_id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="User-friendly name")
    file_format: str = Field(..., description="Original file format")
    file_size: int = Field(..., description="File size in bytes")
    resolution: Tuple[int, int] = Field(..., description="Image/video resolution")
    frame_count: int = Field(default=1, description="Number of frames (for video)")
    registration_date: datetime = Field(..., description="Registration timestamp")
    last_used: Optional[datetime] = Field(None, description="Last usage timestamp")
    usage_count: int = Field(default=0, description="Number of times used")
    face_quality_score: float = Field(..., description="Face detection quality (0-1)")
    processing_ready: bool = Field(..., description="Ready for immediate processing")
    cache_size: int = Field(default=0, description="Face cache data size")
    owner_id: str = Field(..., description="Avatar owner user ID")


class FaceProcessingMetadata(BaseModel):
    """Face processing metadata"""
    processing_version: str = Field(..., description="Processing algorithm version")
    detection_model: str = Field(..., description="Face detection model used")
    face_count: int = Field(..., description="Number of faces detected")
    primary_face_bbox: Tuple[int, int, int, int] = Field(..., description="Primary face bounding box")
    processing_quality: str = Field(..., description="Processing quality level")
    optimization_flags: List[str] = Field(default_factory=list, description="Applied optimizations")


class CachedFaceData(BaseModel):
    """Cached face data model"""
    avatar_id: str = Field(..., description="Avatar identifier")
    face_boxes: List[Tuple[int, int, int, int]] = Field(..., description="Bounding boxes per frame")
    face_landmarks: Optional[List[Any]] = Field(None, description="Face landmarks per frame")  # Handled separately
    cropped_faces: Optional[List[Any]] = Field(None, description="Pre-cropped face regions")  # Handled separately
    face_masks: Optional[List[Any]] = Field(None, description="Face mask regions for wav2lip")  # Handled separately
    processing_metadata: FaceProcessingMetadata = Field(..., description="Processing details")
    cache_version: str = Field(..., description="Cache format version")
    cache_timestamp: datetime = Field(..., description="Cache creation time")
    compression_ratio: float = Field(default=1.0, description="Data compression ratio")
    integrity_hash: str = Field(..., description="Data integrity checksum")

    class Config:
        arbitrary_types_allowed = True


class AvatarDatabaseEntry(BaseModel):
    """Avatar database entry model"""
    id: Optional[int] = Field(None, description="Database primary key")
    avatar_id: str = Field(..., description="Unique avatar identifier")
    name: str = Field(..., description="Avatar name")
    file_path: str = Field(..., description="Storage file path")
    cache_path: str = Field(..., description="Face cache file path")
    owner_id: str = Field(..., description="Owner user ID")
    file_format: str = Field(..., description="File format")
    file_size: int = Field(..., description="File size in bytes")
    face_quality_score: float = Field(..., description="Quality assessment")
    registration_date: datetime = Field(..., description="Registration timestamp")
    last_accessed: Optional[datetime] = Field(None, description="Last access time")
    access_count: int = Field(default=0, description="Usage counter")
    is_active: bool = Field(default=True, description="Active status")
    metadata_json: str = Field(default="{}", description="Additional metadata as JSON")


class AvatarValidationResult(BaseModel):
    """Avatar validation result"""
    is_valid: bool = Field(..., description="Overall validation status")
    file_format: str = Field(..., description="Detected file format")
    file_size: int = Field(..., description="File size in bytes")
    resolution: Tuple[int, int] = Field(..., description="Image/video resolution")
    frame_count: int = Field(default=1, description="Number of frames (for video)")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Non-critical warnings")
    face_detected: bool = Field(default=False, description="Face detection success")
    face_confidence: float = Field(default=0.0, description="Face detection confidence")


class AvatarReadinessReport(BaseModel):
    """Avatar processing readiness assessment"""
    avatar_id: str = Field(..., description="Avatar identifier")
    is_ready: bool = Field(..., description="Ready for processing")
    cache_available: bool = Field(..., description="Face cache available")
    cache_integrity: bool = Field(..., description="Cache integrity valid")
    face_quality_sufficient: bool = Field(..., description="Face quality meets standards")
    error_details: List[str] = Field(default_factory=list, description="Readiness issues")
    estimated_processing_time: float = Field(..., description="Estimated processing time per chunk")


class WarmupReport(BaseModel):
    """Avatar cache warmup report"""
    avatars_processed: int = Field(..., description="Number of avatars processed")
    successful_warmups: int = Field(..., description="Successful warmup operations")
    failed_warmups: int = Field(..., description="Failed warmup operations")
    total_time: float = Field(..., description="Total warmup time")
    cache_hit_improvement: float = Field(..., description="Expected cache hit improvement")
    errors: List[str] = Field(default_factory=list, description="Warmup errors")


class ProcessingStats(BaseModel):
    """Avatar processing statistics"""
    total_usage_count: int = Field(default=0, description="Total times used")
    average_processing_time: float = Field(default=0.0, description="Average processing time")
    cache_hit_rate: float = Field(default=0.0, description="Cache hit rate")
    last_used: Optional[datetime] = Field(None, description="Last usage timestamp")
    quality_score_history: List[float] = Field(default_factory=list, description="Quality score over time")


class CacheStatus(BaseModel):
    """Face cache status"""
    is_cached: bool = Field(..., description="Cache available")
    cache_size: int = Field(default=0, description="Cache size in bytes")
    cache_age: float = Field(default=0.0, description="Cache age in hours")
    integrity_valid: bool = Field(default=True, description="Cache integrity status")
    last_validated: Optional[datetime] = Field(None, description="Last validation time") 