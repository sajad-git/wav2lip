"""
Silence Detector - Persian-aware audio segmentation
Detects optimal chunk boundaries based on Persian speech patterns
"""

import asyncio
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import librosa
from dataclasses import dataclass
from enum import Enum

from app.config.settings import Settings
from app.core.audio_processor import AudioProcessor

class SilenceType(Enum):
    """Types of silence in speech"""
    WORD_BOUNDARY = "word_boundary"
    PHRASE_BOUNDARY = "phrase_boundary"
    SENTENCE_BOUNDARY = "sentence_boundary"
    BREATH_PAUSE = "breath_pause"
    HESITATION = "hesitation"

@dataclass
class SilenceSegment:
    """Silence segment information"""
    start_time: float
    end_time: float
    duration: float
    silence_type: SilenceType
    confidence: float
    energy_level: float

@dataclass
class ChunkBoundary:
    """Optimal chunk boundary information"""
    start_time: float
    end_time: float
    duration: float
    confidence: float
    boundary_quality: str  # excellent, good, fair, poor
    persian_score: float  # How well this boundary respects Persian speech patterns

class PersianSilenceDetector:
    """Persian-aware silence detection for optimal audio chunking"""
    
    def __init__(self):
        self.settings = Settings()
        self.logger = logging.getLogger(__name__)
        self.audio_processor = AudioProcessor()
        
        # Persian speech characteristics
        self.persian_pause_patterns = {
            'word_boundary': (0.05, 0.15),  # 50-150ms
            'phrase_boundary': (0.15, 0.4),  # 150-400ms
            'sentence_boundary': (0.4, 1.0),  # 400ms-1s
            'breath_pause': (0.2, 0.6),  # 200-600ms
            'hesitation': (0.1, 0.3)  # 100-300ms
        }
        
        # Persian phoneme characteristics
        self.persian_vowel_formants = {
            'a': (730, 1090),   # /a/
            'e': (530, 1840),   # /e/
            'i': (270, 2290),   # /i/
            'o': (570, 840),    # /o/
            'u': (300, 870),    # /u/
        }
        
        # Detection parameters
        self.energy_threshold_factor = 0.02  # 2% of peak energy
        self.min_silence_duration = 0.05  # 50ms minimum
        self.max_silence_duration = 2.0   # 2s maximum
        self.frame_length = 0.025  # 25ms frames
        self.hop_length = 0.010    # 10ms hop
        
    async def detect_optimal_chunk_boundaries(
        self,
        audio_data: bytes,
        target_chunk_duration: float = 5.0,
        max_chunk_duration: float = 15.0,
        min_chunk_duration: float = 2.0
    ) -> List[ChunkBoundary]:
        """
        Detect optimal chunk boundaries for Persian speech
        
        Args:
            audio_data: Audio data bytes
            target_chunk_duration: Preferred chunk duration
            max_chunk_duration: Maximum allowed chunk duration
            min_chunk_duration: Minimum allowed chunk duration
            
        Returns:
            List of optimal chunk boundaries
        """
        try:
            self.logger.debug("Detecting optimal chunk boundaries for Persian speech")
            
            # Load and preprocess audio
            audio_array, sr = librosa.load(
                io.BytesIO(audio_data),
                sr=self.audio_processor.target_sample_rate
            )
            
            # Detect all silence segments
            silence_segments = await self.detect_silence_segments(audio_array, sr)
            
            # Classify silence types based on Persian patterns
            classified_segments = self._classify_persian_silence_types(
                silence_segments, audio_array, sr
            )
            
            # Find optimal chunk boundaries
            chunk_boundaries = self._find_optimal_boundaries(
                classified_segments,
                len(audio_array) / sr,
                target_chunk_duration,
                max_chunk_duration,
                min_chunk_duration
            )
            
            # Score boundaries based on Persian speech patterns
            scored_boundaries = self._score_persian_boundaries(
                chunk_boundaries, classified_segments
            )
            
            self.logger.debug(f"Found {len(scored_boundaries)} optimal chunk boundaries")
            return scored_boundaries
            
        except Exception as e:
            self.logger.error(f"Chunk boundary detection failed: {str(e)}")
            # Fallback to simple time-based chunking
            return self._create_fallback_boundaries(
                len(audio_array) / sr,
                target_chunk_duration
            )
    
    async def detect_silence_segments(
        self,
        audio_array: np.ndarray,
        sample_rate: int
    ) -> List[SilenceSegment]:
        """
        Detect silence segments in audio
        
        Args:
            audio_array: Audio data array
            sample_rate: Audio sample rate
            
        Returns:
            List of detected silence segments
        """
        try:
            # Calculate frame parameters
            frame_length = int(self.frame_length * sample_rate)
            hop_length = int(self.hop_length * sample_rate)
            
            # Calculate energy per frame using RMS
            energy = librosa.feature.rms(
                y=audio_array,
                frame_length=frame_length,
                hop_length=hop_length
            )[0]
            
            # Calculate adaptive threshold
            energy_threshold = self._calculate_adaptive_threshold(energy)
            
            # Find silent frames
            silent_frames = energy < energy_threshold
            
            # Convert frame indices to time segments
            silence_segments = self._frames_to_time_segments(
                silent_frames, hop_length, sample_rate
            )
            
            # Filter by duration
            filtered_segments = [
                seg for seg in silence_segments
                if self.min_silence_duration <= seg.duration <= self.max_silence_duration
            ]
            
            self.logger.debug(f"Detected {len(filtered_segments)} silence segments")
            return filtered_segments
            
        except Exception as e:
            self.logger.error(f"Silence detection failed: {str(e)}")
            return []
    
    def _classify_persian_silence_types(
        self,
        silence_segments: List[SilenceSegment],
        audio_array: np.ndarray,
        sample_rate: int
    ) -> List[SilenceSegment]:
        """Classify silence types based on Persian speech patterns"""
        classified_segments = []
        
        for segment in silence_segments:
            # Analyze context around silence
            context_before, context_after = self._extract_silence_context(
                segment, audio_array, sample_rate
            )
            
            # Classify based on duration and context
            silence_type = self._determine_persian_silence_type(
                segment.duration, context_before, context_after
            )
            
            # Calculate confidence based on pattern matching
            confidence = self._calculate_silence_confidence(
                segment, silence_type, context_before, context_after
            )
            
            # Create classified segment
            classified_segment = SilenceSegment(
                start_time=segment.start_time,
                end_time=segment.end_time,
                duration=segment.duration,
                silence_type=silence_type,
                confidence=confidence,
                energy_level=segment.energy_level
            )
            
            classified_segments.append(classified_segment)
        
        return classified_segments
    
    def _find_optimal_boundaries(
        self,
        silence_segments: List[SilenceSegment],
        total_duration: float,
        target_duration: float,
        max_duration: float,
        min_duration: float
    ) -> List[ChunkBoundary]:
        """Find optimal chunk boundaries from silence segments"""
        boundaries = []
        current_start = 0.0
        
        while current_start < total_duration:
            # Find the best boundary within the target duration range
            target_end = current_start + target_duration
            max_end = min(current_start + max_duration, total_duration)
            min_end = current_start + min_duration
            
            # Find suitable silence segments within range
            candidate_segments = [
                seg for seg in silence_segments
                if min_end <= seg.start_time <= max_end
            ]
            
            if candidate_segments:
                # Choose the best candidate based on Persian patterns
                best_segment = self._select_best_boundary_segment(
                    candidate_segments, target_end
                )
                
                # Create boundary
                boundary = ChunkBoundary(
                    start_time=current_start,
                    end_time=best_segment.end_time,
                    duration=best_segment.end_time - current_start,
                    confidence=best_segment.confidence,
                    boundary_quality="good",
                    persian_score=0.8
                )
                
                boundaries.append(boundary)
                current_start = best_segment.end_time
            else:
                # No suitable silence found, use fixed duration
                end_time = min(current_start + target_duration, total_duration)
                
                boundary = ChunkBoundary(
                    start_time=current_start,
                    end_time=end_time,
                    duration=end_time - current_start,
                    confidence=0.5,
                    boundary_quality="fair",
                    persian_score=0.3
                )
                
                boundaries.append(boundary)
                current_start = end_time
        
        return boundaries
    
    def _score_persian_boundaries(
        self,
        boundaries: List[ChunkBoundary],
        silence_segments: List[SilenceSegment]
    ) -> List[ChunkBoundary]:
        """Score boundaries based on Persian speech patterns"""
        scored_boundaries = []
        
        for boundary in boundaries:
            # Find associated silence segment
            associated_silence = self._find_associated_silence(
                boundary, silence_segments
            )
            
            # Calculate Persian score
            persian_score = self._calculate_persian_score(
                boundary, associated_silence
            )
            
            # Determine quality grade
            quality = self._determine_boundary_quality(persian_score)
            
            # Update boundary with scores
            scored_boundary = ChunkBoundary(
                start_time=boundary.start_time,
                end_time=boundary.end_time,
                duration=boundary.duration,
                confidence=boundary.confidence,
                boundary_quality=quality,
                persian_score=persian_score
            )
            
            scored_boundaries.append(scored_boundary)
        
        return scored_boundaries
    
    def _calculate_adaptive_threshold(self, energy: np.ndarray) -> float:
        """Calculate adaptive threshold for silence detection"""
        # Use percentile-based threshold
        base_threshold = np.percentile(energy, 20)  # 20th percentile
        peak_energy = np.max(energy)
        
        # Adaptive threshold based on dynamic range
        threshold = max(
            base_threshold,
            peak_energy * self.energy_threshold_factor
        )
        
        return threshold
    
    def _frames_to_time_segments(
        self,
        silent_frames: np.ndarray,
        hop_length: int,
        sample_rate: int
    ) -> List[SilenceSegment]:
        """Convert silent frame indices to time segments"""
        segments = []
        
        # Find contiguous silent regions
        silent_regions = self._find_contiguous_regions(silent_frames)
        
        for start_frame, end_frame in silent_regions:
            start_time = start_frame * hop_length / sample_rate
            end_time = end_frame * hop_length / sample_rate
            duration = end_time - start_time
            
            # Calculate average energy level for the segment
            energy_level = np.mean(silent_frames[start_frame:end_frame])
            
            segment = SilenceSegment(
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                silence_type=SilenceType.WORD_BOUNDARY,  # Default, will be classified later
                confidence=0.5,
                energy_level=energy_level
            )
            
            segments.append(segment)
        
        return segments
    
    def _find_contiguous_regions(self, boolean_array: np.ndarray) -> List[Tuple[int, int]]:
        """Find contiguous True regions in boolean array"""
        regions = []
        
        # Find transitions
        transitions = np.diff(np.concatenate(([False], boolean_array, [False])).astype(int))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]
        
        for start, end in zip(starts, ends):
            regions.append((start, end))
        
        return regions
    
    def _extract_silence_context(
        self,
        silence_segment: SilenceSegment,
        audio_array: np.ndarray,
        sample_rate: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract audio context before and after silence"""
        context_duration = 0.5  # 500ms context
        context_samples = int(context_duration * sample_rate)
        
        # Calculate sample indices
        silence_start_sample = int(silence_segment.start_time * sample_rate)
        silence_end_sample = int(silence_segment.end_time * sample_rate)
        
        # Extract context before silence
        before_start = max(0, silence_start_sample - context_samples)
        context_before = audio_array[before_start:silence_start_sample]
        
        # Extract context after silence
        after_end = min(len(audio_array), silence_end_sample + context_samples)
        context_after = audio_array[silence_end_sample:after_end]
        
        return context_before, context_after
    
    def _determine_persian_silence_type(
        self,
        duration: float,
        context_before: np.ndarray,
        context_after: np.ndarray
    ) -> SilenceType:
        """Determine silence type based on Persian speech patterns"""
        # Duration-based classification
        if duration < 0.1:
            return SilenceType.WORD_BOUNDARY
        elif duration < 0.25:
            return SilenceType.PHRASE_BOUNDARY
        elif duration < 0.7:
            return SilenceType.SENTENCE_BOUNDARY
        else:
            return SilenceType.BREATH_PAUSE
    
    def _calculate_silence_confidence(
        self,
        segment: SilenceSegment,
        silence_type: SilenceType,
        context_before: np.ndarray,
        context_after: np.ndarray
    ) -> float:
        """Calculate confidence score for silence classification"""
        # Base confidence from duration matching
        duration = segment.duration
        expected_range = self.persian_pause_patterns.get(silence_type.value, (0.1, 1.0))
        
        if expected_range[0] <= duration <= expected_range[1]:
            duration_confidence = 1.0
        else:
            # Calculate how far outside the expected range
            if duration < expected_range[0]:
                duration_confidence = duration / expected_range[0]
            else:
                duration_confidence = expected_range[1] / duration
        
        # Context-based confidence (simplified)
        context_confidence = 0.8  # Placeholder for more sophisticated analysis
        
        # Combined confidence
        return (duration_confidence + context_confidence) / 2.0
    
    def _select_best_boundary_segment(
        self,
        candidates: List[SilenceSegment],
        target_end: float
    ) -> SilenceSegment:
        """Select the best boundary segment from candidates"""
        if not candidates:
            return None
        
        # Score each candidate
        scored_candidates = []
        for candidate in candidates:
            # Distance from target
            distance_score = 1.0 - abs(candidate.start_time - target_end) / 2.0
            
            # Type preference (sentence > phrase > word)
            type_scores = {
                SilenceType.SENTENCE_BOUNDARY: 1.0,
                SilenceType.BREATH_PAUSE: 0.9,
                SilenceType.PHRASE_BOUNDARY: 0.7,
                SilenceType.WORD_BOUNDARY: 0.5,
                SilenceType.HESITATION: 0.3
            }
            type_score = type_scores.get(candidate.silence_type, 0.5)
            
            # Combined score
            total_score = (
                distance_score * 0.4 +
                candidate.confidence * 0.3 +
                type_score * 0.3
            )
            
            scored_candidates.append((candidate, total_score))
        
        # Return best candidate
        return max(scored_candidates, key=lambda x: x[1])[0]
    
    def _find_associated_silence(
        self,
        boundary: ChunkBoundary,
        silence_segments: List[SilenceSegment]
    ) -> Optional[SilenceSegment]:
        """Find silence segment associated with boundary"""
        for segment in silence_segments:
            if abs(segment.end_time - boundary.end_time) < 0.1:  # 100ms tolerance
                return segment
        return None
    
    def _calculate_persian_score(
        self,
        boundary: ChunkBoundary,
        silence_segment: Optional[SilenceSegment]
    ) -> float:
        """Calculate Persian-specific boundary score"""
        if not silence_segment:
            return 0.3  # Low score for non-silence boundaries
        
        # Type-based scoring
        type_scores = {
            SilenceType.SENTENCE_BOUNDARY: 1.0,
            SilenceType.BREATH_PAUSE: 0.9,
            SilenceType.PHRASE_BOUNDARY: 0.7,
            SilenceType.WORD_BOUNDARY: 0.5,
            SilenceType.HESITATION: 0.3
        }
        
        type_score = type_scores.get(silence_segment.silence_type, 0.5)
        confidence_score = silence_segment.confidence
        
        # Duration appropriateness
        duration_score = 1.0
        if boundary.duration < 2.0:
            duration_score = 0.5
        elif boundary.duration > 15.0:
            duration_score = 0.3
        
        return (type_score + confidence_score + duration_score) / 3.0
    
    def _determine_boundary_quality(self, persian_score: float) -> str:
        """Determine boundary quality grade"""
        if persian_score >= 0.8:
            return "excellent"
        elif persian_score >= 0.6:
            return "good"
        elif persian_score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _create_fallback_boundaries(
        self,
        total_duration: float,
        target_duration: float
    ) -> List[ChunkBoundary]:
        """Create fallback boundaries using simple time division"""
        boundaries = []
        current_start = 0.0
        
        while current_start < total_duration:
            end_time = min(current_start + target_duration, total_duration)
            
            boundary = ChunkBoundary(
                start_time=current_start,
                end_time=end_time,
                duration=end_time - current_start,
                confidence=0.3,
                boundary_quality="poor",
                persian_score=0.2
            )
            
            boundaries.append(boundary)
            current_start = end_time
        
        return boundaries 