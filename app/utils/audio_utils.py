"""
Audio Processing Utilities
Audio format standardization and processing for Avatar service
"""

import os
import logging
import tempfile
import numpy as np
import wave
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
import io

try:
    import librosa
    import soundfile as sf
    from pydub import AudioSegment
    from pydub.utils import which
    AUDIO_LIBRARIES_AVAILABLE = True
except ImportError:
    AUDIO_LIBRARIES_AVAILABLE = False
    logging.warning("Audio processing libraries not available. Install librosa, soundfile, and pydub for full functionality.")

from .validation import ValidationResult


@dataclass
class AudioMetadata:
    """Audio file metadata"""
    duration: float
    sample_rate: int
    channels: int
    bit_depth: int
    format: str
    size_bytes: int
    is_valid: bool = True
    error_message: Optional[str] = None


@dataclass
class AudioQualityReport:
    """Audio quality assessment report"""
    snr_ratio: float
    clarity_score: float
    duration_seconds: float
    quality_grade: str  # A, B, C, D, F
    recommendations: List[str]
    is_suitable_for_processing: bool


class AudioProcessor:
    """Audio processing and format standardization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Target specifications for wav2lip
        self.target_sample_rate = 16000  # 16kHz
        self.target_channels = 1  # Mono
        self.target_format = "wav"
        
        # Quality thresholds
        self.min_duration = 0.1  # 100ms minimum
        self.max_duration = 300.0  # 5 minutes maximum
        self.min_snr = 10.0  # 10dB minimum SNR
        
        # Persian speech optimization
        self.persian_frequency_range = (80, 8000)  # Hz
        self.persian_emphasis_frequencies = [200, 400, 800, 1600, 3200]  # Hz
        
        if not AUDIO_LIBRARIES_AVAILABLE:
            self.logger.warning("Audio processing capabilities limited without librosa and soundfile")
    
    def get_audio_metadata(self, audio_path: Union[str, Path]) -> Optional[AudioMetadata]:
        """Extract comprehensive audio metadata"""
        try:
            if not AUDIO_LIBRARIES_AVAILABLE:
                return self._get_basic_metadata(audio_path)
            
            # Use librosa for detailed analysis
            y, sr = librosa.load(str(audio_path), sr=None)
            
            # Get file info
            audio_path = Path(audio_path)
            file_size = audio_path.stat().st_size
            
            # Calculate metadata
            duration = librosa.get_duration(y=y, sr=sr)
            channels = 1 if y.ndim == 1 else y.shape[0]
            
            # Estimate bit depth (librosa loads as float32, original might be different)
            with sf.SoundFile(str(audio_path)) as f:
                bit_depth = f.subtype_info.bits if hasattr(f.subtype_info, 'bits') else 16
            
            return AudioMetadata(
                duration=duration,
                sample_rate=sr,
                channels=channels,
                bit_depth=bit_depth,
                format=audio_path.suffix.lower(),
                size_bytes=file_size
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get audio metadata for {audio_path}: {str(e)}")
            return AudioMetadata(
                duration=0.0,
                sample_rate=0,
                channels=0,
                bit_depth=0,
                format="unknown",
                size_bytes=0,
                is_valid=False,
                error_message=str(e)
            )
    
    def _get_basic_metadata(self, audio_path: Union[str, Path]) -> Optional[AudioMetadata]:
        """Get basic metadata without advanced libraries"""
        try:
            audio_path = Path(audio_path)
            file_size = audio_path.stat().st_size
            
            # Try to read WAV file headers
            if audio_path.suffix.lower() == '.wav':
                with wave.open(str(audio_path), 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    bit_depth = wav_file.getsampwidth() * 8
                    duration = frames / sample_rate
                    
                    return AudioMetadata(
                        duration=duration,
                        sample_rate=sample_rate,
                        channels=channels,
                        bit_depth=bit_depth,
                        format=".wav",
                        size_bytes=file_size
                    )
            
            # For other formats, return limited info
            return AudioMetadata(
                duration=0.0,
                sample_rate=0,
                channels=0,
                bit_depth=0,
                format=audio_path.suffix.lower(),
                size_bytes=file_size,
                is_valid=False,
                error_message="Advanced audio libraries required for this format"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get basic metadata for {audio_path}: {str(e)}")
            return None
    
    def convert_to_wav2lip_format(self, audio_input: Union[str, Path, bytes], 
                                 source_format: Optional[str] = None) -> Optional[bytes]:
        """Convert audio to wav2lip-compatible format"""
        if not AUDIO_LIBRARIES_AVAILABLE:
            self.logger.error("Audio conversion requires librosa and soundfile libraries")
            return None
        
        try:
            # Load audio data
            if isinstance(audio_input, bytes):
                # Handle bytes input
                temp_file = tempfile.NamedTemporaryFile(
                    suffix=f".{source_format}" if source_format else ".wav",
                    delete=False
                )
                temp_file.write(audio_input)
                temp_file.close()
                audio_path = temp_file.name
                cleanup_temp = True
            else:
                audio_path = str(audio_input)
                cleanup_temp = False
            
            try:
                # Load with librosa
                y, sr = librosa.load(audio_path, sr=self.target_sample_rate)
                
                # Ensure mono
                if y.ndim > 1:
                    y = librosa.to_mono(y)
                
                # Normalize audio levels
                y = self._normalize_audio(y)
                
                # Apply Persian speech optimization
                y = self._optimize_for_persian_speech(y, sr)
                
                # Convert to bytes
                with io.BytesIO() as buffer:
                    sf.write(buffer, y, sr, format='WAV', subtype='PCM_16')
                    buffer.seek(0)
                    wav_data = buffer.read()
                
                self.logger.info(f"Converted audio to wav2lip format: {len(wav_data)} bytes")
                return wav_data
                
            finally:
                if cleanup_temp:
                    os.unlink(audio_path)
            
        except Exception as e:
            self.logger.error(f"Audio conversion failed: {str(e)}")
            return None
    
    def _normalize_audio(self, audio: np.ndarray, target_level: float = -3.0) -> np.ndarray:
        """Normalize audio to target level in dB"""
        try:
            # Calculate RMS
            rms = np.sqrt(np.mean(audio**2))
            
            if rms == 0:
                return audio
            
            # Convert target level from dB to linear
            target_linear = 10**(target_level / 20)
            
            # Calculate normalization factor
            normalization_factor = target_linear / rms
            
            # Apply normalization
            normalized = audio * normalization_factor
            
            # Prevent clipping
            max_val = np.max(np.abs(normalized))
            if max_val > 1.0:
                normalized = normalized / max_val * 0.95
            
            return normalized
            
        except Exception as e:
            self.logger.warning(f"Audio normalization failed: {str(e)}")
            return audio
    
    def _optimize_for_persian_speech(self, audio: np.ndarray, 
                                   sample_rate: int) -> np.ndarray:
        """Apply Persian-specific audio optimizations"""
        try:
            # Apply bandpass filter for Persian speech frequency range
            low_freq, high_freq = self.persian_frequency_range
            audio_filtered = librosa.effects.preemphasis(audio)
            
            # Enhance Persian phoneme frequencies
            for freq in self.persian_emphasis_frequencies:
                # Apply gentle emphasis at key frequencies
                audio_filtered = self._apply_frequency_emphasis(
                    audio_filtered, sample_rate, freq, gain_db=1.0
                )
            
            # Reduce background noise
            audio_filtered = self._reduce_noise(audio_filtered, sample_rate)
            
            return audio_filtered
            
        except Exception as e:
            self.logger.warning(f"Persian optimization failed: {str(e)}")
            return audio
    
    def _apply_frequency_emphasis(self, audio: np.ndarray, sample_rate: int,
                                center_freq: float, gain_db: float = 1.0,
                                q_factor: float = 1.0) -> np.ndarray:
        """Apply frequency emphasis using a peaking filter"""
        try:
            # Simple frequency emphasis (could be improved with proper filter)
            stft = librosa.stft(audio)
            freqs = librosa.fft_frequencies(sr=sample_rate)
            
            # Find closest frequency bin
            freq_bin = np.argmin(np.abs(freqs - center_freq))
            
            # Apply gentle gain
            gain_linear = 10**(gain_db / 20)
            bandwidth = max(1, int(len(freqs) / (sample_rate / center_freq / q_factor)))
            start_bin = max(0, freq_bin - bandwidth // 2)
            end_bin = min(len(freqs), freq_bin + bandwidth // 2)
            
            stft[start_bin:end_bin] *= gain_linear
            
            return librosa.istft(stft)
            
        except Exception as e:
            self.logger.warning(f"Frequency emphasis failed: {str(e)}")
            return audio
    
    def _reduce_noise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply noise reduction"""
        try:
            # Simple noise reduction using spectral gating
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            
            # Estimate noise floor from quiet sections
            power = magnitude**2
            noise_floor = np.percentile(power, 10)  # Bottom 10% as noise estimate
            
            # Create mask for noise reduction
            mask = power / (power + noise_floor * 0.1)
            
            # Apply mask
            stft_clean = stft * mask
            
            return librosa.istft(stft_clean)
            
        except Exception as e:
            self.logger.warning(f"Noise reduction failed: {str(e)}")
            return audio
    
    def validate_audio_quality(self, audio_input: Union[str, Path, bytes]) -> AudioQualityReport:
        """Comprehensive audio quality assessment"""
        try:
            # Load audio for analysis
            if isinstance(audio_input, bytes):
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_file.write(audio_input)
                temp_file.close()
                audio_path = temp_file.name
                cleanup_temp = True
            else:
                audio_path = str(audio_input)
                cleanup_temp = False
            
            try:
                if not AUDIO_LIBRARIES_AVAILABLE:
                    # Basic validation without advanced libraries
                    metadata = self._get_basic_metadata(audio_path)
                    return AudioQualityReport(
                        snr_ratio=0.0,
                        clarity_score=0.5,
                        duration_seconds=metadata.duration if metadata else 0.0,
                        quality_grade="C",
                        recommendations=["Install audio processing libraries for detailed analysis"],
                        is_suitable_for_processing=True
                    )
                
                # Load audio
                y, sr = librosa.load(audio_path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                
                # Calculate SNR
                snr = self._calculate_snr(y)
                
                # Calculate clarity score
                clarity = self._calculate_clarity_score(y, sr)
                
                # Determine quality grade
                grade = self._determine_quality_grade(snr, clarity, duration)
                
                # Generate recommendations
                recommendations = self._generate_recommendations(snr, clarity, duration)
                
                # Check suitability
                is_suitable = (
                    duration >= self.min_duration and
                    duration <= self.max_duration and
                    snr >= self.min_snr
                )
                
                return AudioQualityReport(
                    snr_ratio=snr,
                    clarity_score=clarity,
                    duration_seconds=duration,
                    quality_grade=grade,
                    recommendations=recommendations,
                    is_suitable_for_processing=is_suitable
                )
                
            finally:
                if cleanup_temp:
                    os.unlink(audio_path)
                    
        except Exception as e:
            self.logger.error(f"Audio quality validation failed: {str(e)}")
            return AudioQualityReport(
                snr_ratio=0.0,
                clarity_score=0.0,
                duration_seconds=0.0,
                quality_grade="F",
                recommendations=[f"Validation error: {str(e)}"],
                is_suitable_for_processing=False
            )
    
    def _calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio"""
        try:
            # Simple SNR calculation
            # Assume signal is in the middle portion, noise in quiet sections
            signal_power = np.mean(audio**2)
            
            # Estimate noise from quietest 10% of frames
            frame_size = int(len(audio) * 0.01)  # 1% frame size
            frame_powers = []
            
            for i in range(0, len(audio) - frame_size, frame_size):
                frame = audio[i:i + frame_size]
                frame_powers.append(np.mean(frame**2))
            
            noise_power = np.percentile(frame_powers, 10)
            
            if noise_power == 0:
                return 60.0  # Very high SNR if no noise detected
            
            snr_linear = signal_power / noise_power
            snr_db = 10 * np.log10(snr_linear)
            
            return max(0.0, min(60.0, snr_db))  # Clamp between 0 and 60 dB
            
        except Exception as e:
            self.logger.warning(f"SNR calculation failed: {str(e)}")
            return 20.0  # Default moderate SNR
    
    def _calculate_clarity_score(self, audio: np.ndarray, sample_rate: int) -> float:
        """Calculate speech clarity score (0-1)"""
        try:
            # Calculate spectral features
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            
            # Spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(S=magnitude)[0]
            centroid_mean = np.mean(spectral_centroid)
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude)[0]
            rolloff_mean = np.mean(spectral_rolloff)
            
            # Zero crossing rate (indication of speech vs noise)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            zcr_mean = np.mean(zcr)
            
            # Combine features into clarity score
            # Normalize features to 0-1 range
            centroid_score = min(1.0, centroid_mean / (sample_rate * 0.25))
            rolloff_score = min(1.0, rolloff_mean / (sample_rate * 0.5))
            zcr_score = min(1.0, zcr_mean * 100)  # ZCR is usually very small
            
            # Weighted combination
            clarity = (centroid_score * 0.4 + rolloff_score * 0.3 + zcr_score * 0.3)
            
            return max(0.0, min(1.0, clarity))
            
        except Exception as e:
            self.logger.warning(f"Clarity calculation failed: {str(e)}")
            return 0.5  # Default moderate clarity
    
    def _determine_quality_grade(self, snr: float, clarity: float, 
                               duration: float) -> str:
        """Determine quality grade based on metrics"""
        # Calculate composite score
        snr_score = min(1.0, snr / 30.0)  # Normalize SNR to 0-1 (30dB = perfect)
        duration_score = 1.0 if self.min_duration <= duration <= self.max_duration else 0.5
        
        composite_score = (snr_score * 0.4 + clarity * 0.4 + duration_score * 0.2)
        
        if composite_score >= 0.9:
            return "A"
        elif composite_score >= 0.8:
            return "B"
        elif composite_score >= 0.6:
            return "C"
        elif composite_score >= 0.4:
            return "D"
        else:
            return "F"
    
    def _generate_recommendations(self, snr: float, clarity: float, 
                                duration: float) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        if snr < self.min_snr:
            recommendations.append("Improve recording environment to reduce background noise")
        
        if clarity < 0.5:
            recommendations.append("Ensure clear speech with proper microphone positioning")
        
        if duration < self.min_duration:
            recommendations.append("Audio too short for processing")
        elif duration > self.max_duration:
            recommendations.append("Consider splitting long audio into smaller segments")
        
        if snr > 40 and clarity > 0.8:
            recommendations.append("Excellent audio quality for processing")
        elif not recommendations:
            recommendations.append("Audio quality is suitable for processing")
        
        return recommendations
    
    def extract_audio_features_for_sync(self, audio_input: Union[str, Path, bytes]) -> Optional[Dict[str, Any]]:
        """Extract audio features for lip-sync alignment"""
        if not AUDIO_LIBRARIES_AVAILABLE:
            self.logger.warning("Feature extraction requires librosa")
            return None
        
        try:
            # Load audio
            if isinstance(audio_input, bytes):
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_file.write(audio_input)
                temp_file.close()
                audio_path = temp_file.name
                cleanup_temp = True
            else:
                audio_path = str(audio_input)
                cleanup_temp = False
            
            try:
                y, sr = librosa.load(audio_path, sr=self.target_sample_rate)
                
                # Extract features important for lip-sync
                features = {}
                
                # MFCC features (for phoneme detection)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                features['mfcc'] = mfcc.tolist()
                
                # Spectral centroid (for vowel/consonant detection)
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                features['spectral_centroid'] = spectral_centroid.tolist()
                
                # Zero crossing rate (for voiced/unvoiced detection)
                zcr = librosa.feature.zero_crossing_rate(y)
                features['zero_crossing_rate'] = zcr.tolist()
                
                # RMS energy (for speech intensity)
                rms = librosa.feature.rms(y=y)
                features['rms_energy'] = rms.tolist()
                
                # Temporal features
                features['duration'] = librosa.get_duration(y=y, sr=sr)
                features['sample_rate'] = sr
                features['hop_length'] = 512  # Default hop length
                
                return features
                
            finally:
                if cleanup_temp:
                    os.unlink(audio_path)
                    
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            return None
    
    def validate_format_compatibility(self, file_path: Union[str, Path]) -> ValidationResult:
        """Validate audio format compatibility"""
        try:
            file_path = Path(file_path)
            
            # Check file extension
            allowed_extensions = {'.wav', '.mp3', '.aac', '.ogg', '.flac', '.m4a'}
            if file_path.suffix.lower() not in allowed_extensions:
                return ValidationResult(
                    is_valid=False,
                    error_messages=[f"Unsupported audio format: {file_path.suffix}"],
                    security_score=0.5
                )
            
            # Get metadata
            metadata = self.get_audio_metadata(file_path)
            if not metadata or not metadata.is_valid:
                return ValidationResult(
                    is_valid=False,
                    error_messages=["Could not read audio file metadata"],
                    security_score=0.5
                )
            
            # Check constraints
            errors = []
            warnings = []
            
            if metadata.duration < self.min_duration:
                errors.append(f"Audio too short: {metadata.duration:.2f}s (minimum {self.min_duration}s)")
            
            if metadata.duration > self.max_duration:
                warnings.append(f"Audio very long: {metadata.duration:.2f}s (consider splitting)")
            
            if metadata.sample_rate < 8000:
                warnings.append("Low sample rate may affect quality")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                error_messages=errors,
                warnings=warnings,
                security_score=1.0
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_messages=[f"Validation error: {str(e)}"],
                security_score=0.0
            ) 