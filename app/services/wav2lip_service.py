"""
Wav2Lip processing with pre-loaded models + cached faces
"""
import asyncio
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from app.core.model_loader import ColdModelLoader
from app.core.face_cache_manager import FaceCacheManager
from app.models.chunk_models import AudioChunk, VideoChunk, VideoMetadata, ChunkProcessingResult
from app.models.avatar_models import AvatarReadinessReport, CachedFaceData


@dataclass
class QualitySettings:
    """Processing quality configuration"""
    quality_level: str = "balanced"  # fast, balanced, quality
    frame_rate: int = 25
    resolution: tuple = (256, 256)
    compression_level: int = 1
    optimization_flags: List[str] = None
    
    def __post_init__(self):
        if self.optimization_flags is None:
            self.optimization_flags = []


@dataclass
class ProcessingMetrics:
    """Performance tracking metrics"""
    total_chunks_processed: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    cache_hit_rate: float = 0.0
    error_count: int = 0
    last_processing_time: float = 0.0


class PreloadedWav2LipService:
    """Wav2Lip processing with pre-loaded models + cached faces"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_instances: Dict[str, Any] = {}
        self.face_cache_manager: Optional[FaceCacheManager] = None
        self.processing_metrics = ProcessingMetrics()
        self.quality_settings = QualitySettings()
        self.is_initialized = False
        
        self.logger.info("🎬 Preloaded Wav2Lip Service created")
    
    async def initialize_with_preloaded_models(self, model_loader: ColdModelLoader, face_cache: FaceCacheManager) -> None:
        """Initialize service with cached models and face data"""
        try:
            self.logger.info("🔄 Initializing Wav2Lip service with pre-loaded models...")
            
            # Get pre-loaded model instances
            self.model_instances = {
                "wav2lip_gan": model_loader.get_model_instance("wav2lip_gan"),
                "wav2lip": model_loader.get_model_instance("wav2lip"),
                "face_detector": model_loader.get_model_instance("face_detector")
            }
            
            # Link to face cache manager
            self.face_cache_manager = face_cache
            
            # Validate model functionality
            await self._validate_model_functionality()
            
            self.is_initialized = True
            self.logger.info("✅ Wav2Lip service initialized with pre-loaded models")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Wav2Lip service: {str(e)}")
            raise
    
    async def process_audio_chunk_with_cached_face(self, audio_chunk: AudioChunk, avatar_id: str) -> VideoChunk:
        """Ultra-fast processing using pre-loaded models + cached face data"""
        if not self.is_initialized:
            raise RuntimeError("Service not initialized")
        
        start_time = time.time()
        
        try:
            self.logger.debug(f"🎬 Processing chunk {audio_chunk.chunk_id} with avatar {avatar_id}")
            
            # Step 1: Retrieve cached model instance (no loading delay)
            wav2lip_model = self.model_instances["wav2lip_gan"]
            
            # Step 2: Get pre-processed face data from cache (no face detection delay)
            cached_face_data = await self.face_cache_manager.retrieve_face_cache(avatar_id)
            if not cached_face_data:
                raise RuntimeError(f"No cached face data available for avatar {avatar_id}")
            
            # Step 3: Convert audio to mel spectrogram
            mel_spectrogram = await self._audio_to_mel_spectrogram(audio_chunk.audio_data)
            
            # Step 4: Use cached face bounding boxes and cropped face regions
            face_regions = cached_face_data.cropped_faces
            face_boxes = cached_face_data.face_boxes
            
            if not face_regions or len(face_regions) == 0:
                raise RuntimeError(f"No face regions in cached data for avatar {avatar_id}")
            
            # Step 5: Run ONNX inference with CUDA acceleration
            video_frames = await self._run_wav2lip_inference(
                wav2lip_model, 
                face_regions, 
                mel_spectrogram,
                face_boxes
            )
            
            # Step 6: Post-process video frames using cached metadata
            processed_frames = await self._post_process_frames(
                video_frames, 
                cached_face_data.processing_metadata
            )
            
            # Step 7: Create video chunk with timing metadata
            processing_time = time.time() - start_time
            
            video_chunk = VideoChunk(
                chunk_id=audio_chunk.chunk_id,
                video_frames=processed_frames,
                frame_rate=self.quality_settings.frame_rate,
                duration_seconds=audio_chunk.duration_seconds,
                sync_timestamp=time.time(),
                encoding_format="mp4",
                compression_level=self.quality_settings.compression_level,
                avatar_id=avatar_id,
                metadata=VideoMetadata(
                    frame_rate=self.quality_settings.frame_rate,
                    encoding_format="mp4",
                    compression_level=self.quality_settings.compression_level,
                    resolution=self.quality_settings.resolution
                )
            )
            
            # Update metrics
            self._update_processing_metrics(processing_time, True)
            
            self.logger.debug(f"✅ Chunk {audio_chunk.chunk_id} processed in {processing_time:.3f}s")
            return video_chunk
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_processing_metrics(processing_time, False)
            self.logger.error(f"❌ Failed to process chunk {audio_chunk.chunk_id}: {str(e)}")
            raise
    
    async def batch_process_chunks_with_avatar(self, chunks: List[AudioChunk], avatar_id: str) -> List[VideoChunk]:
        """Efficient batch processing with shared models and cached face data"""
        if not self.is_initialized:
            raise RuntimeError("Service not initialized")
        
        try:
            self.logger.info(f"🔄 Batch processing {len(chunks)} chunks with avatar {avatar_id}")
            
            # Validate chunk sequence and timing
            self._validate_chunk_sequence(chunks)
            
            # Load avatar face cache once for entire batch
            cached_face_data = await self.face_cache_manager.retrieve_face_cache(avatar_id)
            if not cached_face_data:
                raise RuntimeError(f"No cached face data available for avatar {avatar_id}")
            
            # Process chunks sequentially using cached models and face data
            video_chunks = []
            
            for chunk in chunks:
                try:
                    video_chunk = await self.process_audio_chunk_with_cached_face(chunk, avatar_id)
                    video_chunks.append(video_chunk)
                    
                    # Maintain timing continuity
                    await self._ensure_timing_continuity(video_chunks)
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to process chunk {chunk.chunk_id}: {str(e)}")
                    # Continue with other chunks
                    continue
            
            # Apply quality optimization
            optimized_chunks = await self._optimize_batch_quality(video_chunks)
            
            self.logger.info(f"✅ Batch processing completed: {len(optimized_chunks)}/{len(chunks)} chunks")
            return optimized_chunks
            
        except Exception as e:
            self.logger.error(f"❌ Batch processing failed: {str(e)}")
            raise
    
    async def validate_avatar_readiness(self, avatar_id: str) -> AvatarReadinessReport:
        """Verify avatar is ready for immediate processing"""
        try:
            # Check face cache availability and integrity
            cache_available = await self.face_cache_manager.is_cache_available(avatar_id)
            
            if not cache_available:
                return AvatarReadinessReport(
                    avatar_id=avatar_id,
                    is_ready=False,
                    cache_available=False,
                    cache_integrity=False,
                    face_quality_sufficient=False,
                    error_details=["Face cache not available"],
                    estimated_processing_time=5.0  # Would need face detection
                )
            
            # Validate cached face data completeness
            cached_face_data = await self.face_cache_manager.retrieve_face_cache(avatar_id)
            
            face_quality_sufficient = True
            error_details = []
            
            if not cached_face_data.cropped_faces or len(cached_face_data.cropped_faces) == 0:
                face_quality_sufficient = False
                error_details.append("No cropped face regions in cache")
            
            if not cached_face_data.face_boxes or len(cached_face_data.face_boxes) == 0:
                face_quality_sufficient = False
                error_details.append("No face bounding boxes in cache")
            
            # Verify face quality meets processing standards
            quality_score = cached_face_data.processing_metadata.face_count
            if quality_score < 0.5:
                face_quality_sufficient = False
                error_details.append("Face quality below minimum threshold")
            
            cache_integrity = await self.face_cache_manager.validate_cache_integrity(avatar_id)
            
            is_ready = cache_available and cache_integrity and face_quality_sufficient
            estimated_time = 0.15 if is_ready else 2.0  # Fast with cache, slow without
            
            return AvatarReadinessReport(
                avatar_id=avatar_id,
                is_ready=is_ready,
                cache_available=cache_available,
                cache_integrity=cache_integrity,
                face_quality_sufficient=face_quality_sufficient,
                error_details=error_details,
                estimated_processing_time=estimated_time
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to validate avatar readiness: {str(e)}")
            return AvatarReadinessReport(
                avatar_id=avatar_id,
                is_ready=False,
                cache_available=False,
                cache_integrity=False,
                face_quality_sufficient=False,
                error_details=[f"Validation error: {str(e)}"],
                estimated_processing_time=10.0
            )
    
    async def optimize_inference_settings(self, model_name: str, quality_target: str) -> None:
        """Optimize model settings without reloading"""
        try:
            if quality_target not in ["fast", "balanced", "quality"]:
                raise ValueError("Quality target must be one of: fast, balanced, quality")
            
            # Update quality settings
            if quality_target == "fast":
                self.quality_settings = QualitySettings(
                    quality_level="fast",
                    frame_rate=20,
                    resolution=(192, 192),
                    compression_level=2,
                    optimization_flags=["fast_inference", "reduced_precision"]
                )
            elif quality_target == "balanced":
                self.quality_settings = QualitySettings(
                    quality_level="balanced",
                    frame_rate=25,
                    resolution=(256, 256),
                    compression_level=1,
                    optimization_flags=["balanced_inference"]
                )
            else:  # quality
                self.quality_settings = QualitySettings(
                    quality_level="quality",
                    frame_rate=30,
                    resolution=(512, 512),
                    compression_level=0,
                    optimization_flags=["high_quality", "full_precision"]
                )
            
            self.logger.info(f"✅ Inference settings optimized for {quality_target}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to optimize inference settings: {str(e)}")
            raise
    
    async def get_processing_metrics(self) -> ProcessingMetrics:
        """Get current processing performance metrics"""
        return self.processing_metrics
    
    async def _validate_model_functionality(self) -> None:
        """Validate that all required models are functional"""
        required_models = ["wav2lip_gan", "wav2lip", "face_detector"]
        
        for model_name in required_models:
            if model_name not in self.model_instances:
                raise RuntimeError(f"Required model {model_name} not available")
            
            model = self.model_instances[model_name]
            if model is None:
                raise RuntimeError(f"Model {model_name} is None")
        
        self.logger.info("✅ All required models validated")
    
    async def _audio_to_mel_spectrogram(self, audio_data: bytes) -> np.ndarray:
        """Convert audio to mel spectrogram for wav2lip"""
        try:
            # This would implement actual audio processing
            # For now, return mock mel spectrogram
            mel_spec = np.random.rand(80, 16)  # Mock mel spectrogram
            return mel_spec
            
        except Exception as e:
            self.logger.error(f"❌ Failed to convert audio to mel spectrogram: {str(e)}")
            raise
    
    async def _run_wav2lip_inference(self, model: Any, face_regions: List[np.ndarray], 
                                   mel_spec: np.ndarray, face_boxes: List[tuple]) -> List[np.ndarray]:
        """Run wav2lip inference with CUDA acceleration"""
        try:
            # This would implement actual ONNX inference
            # For now, return mock video frames
            num_frames = len(face_regions)
            frames = []
            
            for i in range(num_frames):
                # Mock frame generation
                frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                frames.append(frame)
            
            return frames
            
        except Exception as e:
            self.logger.error(f"❌ Wav2lip inference failed: {str(e)}")
            raise
    
    async def _post_process_frames(self, frames: List[np.ndarray], metadata: Any) -> List[np.ndarray]:
        """Post-process video frames using cached metadata"""
        try:
            # Apply post-processing based on quality settings
            processed_frames = []
            
            for frame in frames:
                # Apply quality-specific processing
                if "fast_inference" in self.quality_settings.optimization_flags:
                    # Fast processing - minimal post-processing
                    processed_frame = frame
                else:
                    # Standard processing
                    processed_frame = frame
                
                processed_frames.append(processed_frame)
            
            return processed_frames
            
        except Exception as e:
            self.logger.error(f"❌ Frame post-processing failed: {str(e)}")
            raise
    
    def _validate_chunk_sequence(self, chunks: List[AudioChunk]) -> None:
        """Validate chunk sequence and timing"""
        if not chunks:
            raise ValueError("No chunks provided")
        
        # Sort chunks by start time
        sorted_chunks = sorted(chunks, key=lambda c: c.start_time)
        
        # Check for gaps or overlaps
        for i in range(1, len(sorted_chunks)):
            prev_chunk = sorted_chunks[i-1]
            curr_chunk = sorted_chunks[i]
            
            gap = curr_chunk.start_time - prev_chunk.end_time
            if gap > 0.1:  # More than 100ms gap
                self.logger.warning(f"⚠️ Large gap between chunks: {gap:.3f}s")
    
    async def _ensure_timing_continuity(self, video_chunks: List[VideoChunk]) -> None:
        """Maintain timing continuity across chunks"""
        if len(video_chunks) < 2:
            return
        
        # Adjust timing to ensure smooth playback
        for i in range(1, len(video_chunks)):
            prev_chunk = video_chunks[i-1]
            curr_chunk = video_chunks[i]
            
            # Ensure continuous timing
            expected_start = prev_chunk.sync_timestamp + prev_chunk.duration_seconds
            if abs(curr_chunk.sync_timestamp - expected_start) > 0.05:  # 50ms tolerance
                curr_chunk.sync_timestamp = expected_start
    
    async def _optimize_batch_quality(self, video_chunks: List[VideoChunk]) -> List[VideoChunk]:
        """Apply quality optimization across the batch"""
        # This would implement batch-level quality optimization
        # For now, just return the chunks as-is
        return video_chunks
    
    def _update_processing_metrics(self, processing_time: float, success: bool) -> None:
        """Update processing performance metrics"""
        self.processing_metrics.last_processing_time = processing_time
        
        if success:
            self.processing_metrics.total_chunks_processed += 1
            self.processing_metrics.total_processing_time += processing_time
            
            if self.processing_metrics.total_chunks_processed > 0:
                self.processing_metrics.average_processing_time = (
                    self.processing_metrics.total_processing_time / 
                    self.processing_metrics.total_chunks_processed
                )
        else:
            self.processing_metrics.error_count += 1 