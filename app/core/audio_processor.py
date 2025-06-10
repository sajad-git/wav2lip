"""
Audio Processor - Audio processing and format standardization
Handles audio conversion, validation, and Persian language optimization
"""

import asyncio
import io
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import librosa
import soundfile as sf
from pydub import AudioSegment
import wave
import struct

from app.config.settings import Settings

class AudioQualityReport:
    """Audio quality assessment report"""
    def __init__(self):
        self.snr_ratio: float = 0.0
        self.clarity_score: float = 0.0
        self.duration_seconds: float = 0.0
        self.quality_grade: str = "unknown"
        self.recommendations: List[str] = []
        self.sample_rate: int = 16000
        self.channels: int = 1
        self.bit_depth: int = 16

class AudioProcessor:
    """Audio processing and format standardization"""
    
    def __init__(self):
        self.settings = Settings()
        self.logger = logging.getLogger(__name__)
        
        # Audio processing constants
        self.target_sample_rate = 16000  # 16kHz for wav2lip
        self.target_channels = 1  # Mono
        self.target_bit_depth = 16
        self.min_duration = 0.1  # Minimum 100ms
        self.max_duration = 30.0  # Maximum 30 seconds
        
        # Persian language specific settings
        self.persian_frequency_range = (80, 8000)  # Hz
        self.persian_speech_rate_range = (2.0, 6.0)  # syllables per second
        
    async def convert_to_wav2lip_format(self, audio_data: bytes, source_format: str) -> bytes:
        """
        Convert audio to wav2lip-compatible format
        
        Args:
            audio_data: Raw audio data bytes
            source_format: Source format identifier (wav, mp3, m4a, etc.)
            
        Returns:
            Processed audio in wav2lip-compatible format
        """
        try:
            self.logger.debug(f"Converting audio from {source_format} to wav2lip format")
            
            # Step 1: Load audio using librosa
            audio_array, original_sr = librosa.load(
                io.BytesIO(audio_data),
                sr=None,
                mono=False
            )
            
            # Step 2: Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = librosa.to_mono(audio_array)
            
            # Step 3: Resample to target sample rate
            if original_sr != self.target_sample_rate:
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=original_sr,
                    target_sr=self.target_sample_rate
                )
            
            # Step 4: Normalize audio levels
            audio_array = self._normalize_audio_levels(audio_array)
            
            # Step 5: Remove silence padding
            audio_array = self._remove_silence_padding(audio_array)
            
            # Step 6: Apply Persian-specific optimization
            audio_array = await self.optimize_for_persian_speech(audio_array)
            
            # Step 7: Convert to bytes format
            audio_bytes = self._array_to_wav_bytes(audio_array)
            
            self.logger.debug("Audio conversion completed successfully")
            return audio_bytes
            
        except Exception as e:
            self.logger.error(f"Audio conversion failed: {str(e)}")
            raise
    
    async def validate_audio_quality(self, audio_data: bytes) -> AudioQualityReport:
        """
        Validate audio quality for processing
        
        Args:
            audio_data: Audio data bytes
            
        Returns:
            Quality assessment report
        """
        report = AudioQualityReport()
        
        try:
            # Load audio for analysis
            audio_array, sr = librosa.load(io.BytesIO(audio_data), sr=None)
            
            # Basic metrics
            report.duration_seconds = len(audio_array) / sr
            report.sample_rate = sr
            report.channels = 1 if len(audio_array.shape) == 1 else audio_array.shape[0]
            
            # Signal-to-noise ratio
            report.snr_ratio = self._calculate_snr(audio_array)
            
            # Speech clarity assessment
            report.clarity_score = self._assess_speech_clarity(audio_array, sr)
            
            # Overall quality grade
            report.quality_grade = self._determine_quality_grade(report)
            
            # Generate recommendations
            report.recommendations = self._generate_recommendations(report)
            
            self.logger.debug(f"Audio quality assessed: {report.quality_grade}")
            return report
            
        except Exception as e:
            self.logger.error(f"Audio quality validation failed: {str(e)}")
            report.quality_grade = "error"
            report.recommendations.append(f"Quality assessment failed: {str(e)}")
            return report
    
    async def optimize_for_persian_speech(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Optimize audio for Persian speech processing
        
        Args:
            audio_array: Audio data array
            
        Returns:
            Persian-optimized audio array
        """
        try:
            # Apply Persian-specific frequency filtering
            audio_array = self._apply_persian_frequency_filter(audio_array)
            
            # Enhance consonant clarity
            audio_array = self._enhance_consonant_clarity(audio_array)
            
            # Optimize for Persian phonemes
            audio_array = self._optimize_persian_phonemes(audio_array)
            
            # Reduce background noise
            audio_array = self._reduce_background_noise(audio_array)
            
            # Normalize speaking rate
            audio_array = self._normalize_speaking_rate(audio_array)
            
            return audio_array
            
        except Exception as e:
            self.logger.warning(f"Persian optimization failed: {str(e)}")
            return audio_array
    
    async def split_audio_for_chunking(
        self, 
        audio_data: bytes, 
        chunk_duration: float = 5.0,
        overlap_duration: float = 0.5
    ) -> List[Tuple[bytes, float, float]]:
        """
        Split audio into chunks for processing
        
        Args:
            audio_data: Audio data bytes
            chunk_duration: Duration of each chunk in seconds
            overlap_duration: Overlap between chunks in seconds
            
        Returns:
            List of tuples (chunk_bytes, start_time, end_time)
        """
        try:
            # Load audio
            audio_array, sr = librosa.load(io.BytesIO(audio_data), sr=self.target_sample_rate)
            
            total_duration = len(audio_array) / sr
            chunks = []
            
            start_time = 0.0
            chunk_samples = int(chunk_duration * sr)
            overlap_samples = int(overlap_duration * sr)
            
            while start_time < total_duration:
                # Calculate chunk boundaries
                start_sample = int(start_time * sr)
                end_sample = min(start_sample + chunk_samples, len(audio_array))
                end_time = end_sample / sr
                
                # Extract chunk
                chunk_array = audio_array[start_sample:end_sample]
                
                # Convert to bytes
                chunk_bytes = self._array_to_wav_bytes(chunk_array)
                
                chunks.append((chunk_bytes, start_time, end_time))
                
                # Move to next chunk with overlap
                start_time += chunk_duration - overlap_duration
                
                # Break if remaining audio is too short
                if end_sample >= len(audio_array):
                    break
            
            self.logger.debug(f"Audio split into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Audio chunking failed: {str(e)}")
            raise
    
    def _normalize_audio_levels(self, audio_array: np.ndarray) -> np.ndarray:
        """Normalize audio levels to prevent clipping"""
        # Peak normalization
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array = audio_array / max_val * 0.95
        
        # RMS normalization for consistent volume
        rms = np.sqrt(np.mean(audio_array ** 2))
        if rms > 0:
            target_rms = 0.1
            audio_array = audio_array * (target_rms / rms)
        
        return audio_array
    
    def _remove_silence_padding(self, audio_array: np.ndarray) -> np.ndarray:
        """Remove silence from beginning and end"""
        # Simple energy-based trimming
        frame_length = int(0.025 * self.target_sample_rate)  # 25ms frames
        hop_length = frame_length // 2
        
        # Calculate energy per frame
        energy = librosa.feature.rms(
            y=audio_array,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Find non-silent regions
        threshold = np.max(energy) * 0.01  # 1% of peak energy
        non_silent = energy > threshold
        
        if np.any(non_silent):
            start_frame = np.argmax(non_silent)
            end_frame = len(non_silent) - np.argmax(non_silent[::-1]) - 1
            
            start_sample = start_frame * hop_length
            end_sample = min((end_frame + 1) * hop_length, len(audio_array))
            
            return audio_array[start_sample:end_sample]
        
        return audio_array
    
    def _apply_persian_frequency_filter(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply frequency filtering optimized for Persian speech"""
        # Bandpass filter for Persian speech frequencies
        from scipy.signal import butter, filtfilt
        
        nyquist = self.target_sample_rate / 2
        low = self.persian_frequency_range[0] / nyquist
        high = min(self.persian_frequency_range[1] / nyquist, 0.99)
        
        b, a = butter(4, [low, high], btype='band')
        filtered_audio = filtfilt(b, a, audio_array)
        
        return filtered_audio
    
    def _enhance_consonant_clarity(self, audio_array: np.ndarray) -> np.ndarray:
        """Enhance consonant clarity for Persian speech"""
        # High-frequency emphasis for consonants
        from scipy.signal import butter, filtfilt
        
        nyquist = self.target_sample_rate / 2
        high_freq = 2000 / nyquist  # Emphasize 2kHz+
        
        b, a = butter(2, high_freq, btype='high')
        high_freq_component = filtfilt(b, a, audio_array)
        
        # Subtle enhancement
        enhanced_audio = audio_array + 0.1 * high_freq_component
        
        return enhanced_audio
    
    def _optimize_persian_phonemes(self, audio_array: np.ndarray) -> np.ndarray:
        """Optimize for Persian phoneme characteristics"""
        # Persian-specific spectral shaping
        # This is a simplified implementation
        return audio_array
    
    def _reduce_background_noise(self, audio_array: np.ndarray) -> np.ndarray:
        """Reduce background noise using spectral subtraction"""
        # Simple noise reduction using spectral subtraction
        # Estimate noise from first 0.5 seconds
        noise_duration = min(0.5, len(audio_array) / self.target_sample_rate / 4)
        noise_samples = int(noise_duration * self.target_sample_rate)
        
        if noise_samples > 0:
            noise_profile = np.mean(np.abs(audio_array[:noise_samples]))
            
            # Apply gentle noise reduction
            noise_threshold = noise_profile * 2
            audio_array = np.where(
                np.abs(audio_array) > noise_threshold,
                audio_array,
                audio_array * 0.1
            )
        
        return audio_array
    
    def _normalize_speaking_rate(self, audio_array: np.ndarray) -> np.ndarray:
        """Normalize speaking rate for consistent processing"""
        # This is a placeholder for speaking rate normalization
        # In practice, this would involve sophisticated speech analysis
        return audio_array
    
    def _calculate_snr(self, audio_array: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        # Simple SNR calculation
        signal_power = np.mean(audio_array ** 2)
        
        # Estimate noise from quiet segments
        frame_length = int(0.025 * self.target_sample_rate)
        hop_length = frame_length // 2
        
        energy = librosa.feature.rms(
            y=audio_array,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Assume bottom 10% of frames are noise
        noise_threshold = np.percentile(energy, 10)
        noise_power = noise_threshold ** 2
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 60.0  # Very high SNR if no noise detected
        
        return max(0.0, min(60.0, snr))
    
    def _assess_speech_clarity(self, audio_array: np.ndarray, sr: int) -> float:
        """Assess speech clarity score"""
        # Calculate spectral centroid as a measure of clarity
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_array, sr=sr)[0]
        
        # Higher centroid generally indicates clearer speech
        mean_centroid = np.mean(spectral_centroids)
        
        # Normalize to 0-1 scale
        clarity_score = min(1.0, mean_centroid / 3000.0)
        
        return clarity_score
    
    def _determine_quality_grade(self, report: AudioQualityReport) -> str:
        """Determine overall quality grade"""
        score = (report.snr_ratio / 30.0 + report.clarity_score) / 2.0
        
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _generate_recommendations(self, report: AudioQualityReport) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        if report.snr_ratio < 20:
            recommendations.append("Improve recording environment to reduce background noise")
        
        if report.clarity_score < 0.5:
            recommendations.append("Use better microphone or improve recording quality")
        
        if report.duration_seconds < self.min_duration:
            recommendations.append(f"Audio too short, minimum {self.min_duration} seconds required")
        
        if report.duration_seconds > self.max_duration:
            recommendations.append(f"Audio too long, maximum {self.max_duration} seconds recommended")
        
        return recommendations
    
    def _array_to_wav_bytes(self, audio_array: np.ndarray) -> bytes:
        """Convert numpy array to WAV bytes"""
        # Ensure 16-bit PCM format
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        # Create WAV file in memory
        buffer = io.BytesIO()
        
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.target_sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        buffer.seek(0)
        return buffer.read() 