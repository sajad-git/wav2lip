"""
Speech-to-Text Service - Persian language optimized STT
Uses OpenAI Whisper API with Persian language optimization
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List
import openai
from dataclasses import dataclass
import io
import tempfile
import os

from app.config.settings import Settings
from app.core.audio_processor import AudioProcessor, AudioQualityReport

@dataclass
class STTResult:
    """Speech-to-text result"""
    text: str
    language: str
    confidence: float
    processing_time: float
    segments: List[Dict[str, Any]] = None
    word_timestamps: List[Dict[str, Any]] = None

@dataclass
class PersianTextMetrics:
    """Persian text processing metrics"""
    text_length: int
    word_count: int
    persian_word_ratio: float
    text_complexity: str
    estimated_speech_duration: float

class PersianTextProcessor:
    """Persian text processing and optimization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Persian language patterns
        self.persian_chars = set('ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی')
        self.persian_digits = set('۰۱۲۳۴۵۶۷۸۹')
        
        # Common Persian words for validation
        self.common_persian_words = {
            'و', 'در', 'به', 'از', 'که', 'این', 'با', 'را', 'یک', 'برای',
            'تا', 'است', 'آن', 'بر', 'کرد', 'دو', 'می', 'گفت', 'هم', 'نیز'
        }
    
    def analyze_persian_text(self, text: str) -> PersianTextMetrics:
        """Analyze Persian text characteristics"""
        # Basic metrics
        text_length = len(text)
        words = text.split()
        word_count = len(words)
        
        # Calculate Persian word ratio
        persian_words = 0
        for word in words:
            if any(char in self.persian_chars for char in word):
                persian_words += 1
        
        persian_ratio = persian_words / word_count if word_count > 0 else 0.0
        
        # Estimate complexity
        complexity = self._estimate_text_complexity(text, words)
        
        # Estimate speech duration (Persian: ~3.5 words per second)
        estimated_duration = word_count / 3.5
        
        return PersianTextMetrics(
            text_length=text_length,
            word_count=word_count,
            persian_word_ratio=persian_ratio,
            text_complexity=complexity,
            estimated_speech_duration=estimated_duration
        )
    
    def normalize_persian_text(self, text: str) -> str:
        """Normalize Persian text for better processing"""
        # Convert Arabic digits to Persian
        arabic_to_persian = str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')
        text = text.translate(arabic_to_persian)
        
        # Normalize Persian characters
        text = text.replace('ك', 'ک')  # Arabic kaf to Persian kaf
        text = text.replace('ي', 'ی')  # Arabic yeh to Persian yeh
        text = text.replace('ة', 'ه')  # Arabic teh marbuta to heh
        
        # Clean up extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def validate_persian_text(self, text: str) -> bool:
        """Validate if text is primarily Persian"""
        metrics = self.analyze_persian_text(text)
        return metrics.persian_word_ratio > 0.7  # 70% Persian words
    
    def _estimate_text_complexity(self, text: str, words: List[str]) -> str:
        """Estimate text complexity for Persian"""
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        if avg_word_length < 4:
            return "simple"
        elif avg_word_length < 6:
            return "medium"
        else:
            return "complex"

class OptimizedSTTService:
    """Optimized Speech-to-Text service with Persian support"""
    
    def __init__(self):
        self.settings = Settings()
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(
            api_key=self.settings.openai_api_key
        )
        
        # Core components
        self.audio_processor = AudioProcessor()
        self.persian_processor = PersianTextProcessor()
        
        # Service configuration
        self.supported_languages = ['fa', 'en', 'auto']
        self.default_language = 'fa'
        self.max_audio_duration = 30.0  # 30 seconds max
        self.min_audio_duration = 0.5   # 0.5 seconds min
        
        # Quality settings
        self.whisper_model = "whisper-1"
        self.temperature = 0.2  # Low temperature for more consistent results
        
        # Caching
        self.result_cache: Dict[str, STTResult] = {}
        self.cache_ttl = 300  # 5 minutes
        
    async def transcribe_audio(
        self,
        audio_data: bytes,
        language: str = "fa",
        enable_word_timestamps: bool = False
    ) -> STTResult:
        """
        Transcribe audio to text with Persian optimization
        
        Args:
            audio_data: Audio data bytes
            language: Target language ('fa', 'en', 'auto')
            enable_word_timestamps: Whether to include word-level timestamps
            
        Returns:
            STT transcription result
        """
        try:
            start_time = time.time()
            self.logger.debug(f"Starting STT transcription for language: {language}")
            
            # Step 1: Validate and preprocess audio
            audio_quality = await self.audio_processor.validate_audio_quality(audio_data)
            if audio_quality.quality_grade == "poor":
                self.logger.warning("Audio quality is poor, results may be degraded")
            
            # Step 2: Convert audio to optimal format
            processed_audio = await self.audio_processor.convert_to_wav2lip_format(
                audio_data, "wav"
            )
            
            # Step 3: Check cache
            cache_key = self._generate_cache_key(processed_audio, language)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.logger.debug("Returning cached STT result")
                return cached_result
            
            # Step 4: Perform transcription
            transcription_result = await self._call_whisper_api(
                processed_audio, language, enable_word_timestamps
            )
            
            # Step 5: Process and optimize results
            stt_result = await self._process_transcription_result(
                transcription_result, language, start_time
            )
            
            # Step 6: Cache result
            self._cache_result(cache_key, stt_result)
            
            self.logger.info(f"STT completed in {stt_result.processing_time:.3f}s")
            return stt_result
            
        except Exception as e:
            self.logger.error(f"STT transcription failed: {str(e)}")
            raise
    
    async def transcribe_chunked_audio(
        self,
        audio_chunks: List[bytes],
        language: str = "fa"
    ) -> List[STTResult]:
        """
        Transcribe multiple audio chunks efficiently
        
        Args:
            audio_chunks: List of audio chunk data
            language: Target language
            
        Returns:
            List of STT results for each chunk
        """
        try:
            self.logger.debug(f"Transcribing {len(audio_chunks)} audio chunks")
            
            # Process chunks concurrently (with rate limiting)
            semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests
            
            async def transcribe_chunk(chunk_data: bytes) -> STTResult:
                async with semaphore:
                    return await self.transcribe_audio(chunk_data, language)
            
            # Create tasks for all chunks
            tasks = [transcribe_chunk(chunk) for chunk in audio_chunks]
            
            # Execute with progress tracking
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and log errors
            stt_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Chunk {i} transcription failed: {str(result)}")
                    # Create empty result for failed chunk
                    stt_results.append(STTResult(
                        text="",
                        language=language,
                        confidence=0.0,
                        processing_time=0.0
                    ))
                else:
                    stt_results.append(result)
            
            return stt_results
            
        except Exception as e:
            self.logger.error(f"Chunked STT transcription failed: {str(e)}")
            raise
    
    async def _call_whisper_api(
        self,
        audio_data: bytes,
        language: str,
        enable_timestamps: bool
    ) -> Dict[str, Any]:
        """Call OpenAI Whisper API"""
        try:
            # Create temporary file for API call
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_data)
                temp_file.flush()
                
                try:
                    # Prepare API parameters
                    api_params = {
                        "model": self.whisper_model,
                        "temperature": self.temperature,
                        "response_format": "verbose_json" if enable_timestamps else "json"
                    }
                    
                    # Set language if not auto-detect
                    if language != "auto":
                        api_params["language"] = language
                    
                    # Call Whisper API
                    with open(temp_file.name, "rb") as audio_file:
                        response = await asyncio.to_thread(
                            self.openai_client.audio.transcriptions.create,
                            file=audio_file,
                            **api_params
                        )
                    
                    # Convert response to dict
                    if hasattr(response, 'model_dump'):
                        return response.model_dump()
                    else:
                        return dict(response)
                        
                finally:
                    # Clean up temporary file
                    os.unlink(temp_file.name)
                    
        except Exception as e:
            self.logger.error(f"Whisper API call failed: {str(e)}")
            raise
    
    async def _process_transcription_result(
        self,
        api_response: Dict[str, Any],
        requested_language: str,
        start_time: float
    ) -> STTResult:
        """Process and optimize transcription result"""
        
        # Extract basic information
        text = api_response.get("text", "")
        detected_language = api_response.get("language", requested_language)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Process Persian text if applicable
        if detected_language == "fa" or requested_language == "fa":
            text = self.persian_processor.normalize_persian_text(text)
            
            # Validate Persian text quality
            if not self.persian_processor.validate_persian_text(text):
                self.logger.warning("Transcribed text does not appear to be primarily Persian")
        
        # Extract segments and word timestamps if available
        segments = api_response.get("segments", [])
        word_timestamps = self._extract_word_timestamps(segments)
        
        # Estimate confidence based on available data
        confidence = self._estimate_confidence(api_response, text)
        
        return STTResult(
            text=text,
            language=detected_language,
            confidence=confidence,
            processing_time=processing_time,
            segments=segments,
            word_timestamps=word_timestamps
        )
    
    def _extract_word_timestamps(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract word-level timestamps from segments"""
        word_timestamps = []
        
        for segment in segments:
            if "words" in segment:
                for word_info in segment["words"]:
                    word_timestamps.append({
                        "word": word_info.get("word", ""),
                        "start": word_info.get("start", 0.0),
                        "end": word_info.get("end", 0.0),
                        "confidence": word_info.get("probability", 0.0)
                    })
        
        return word_timestamps
    
    def _estimate_confidence(self, api_response: Dict[str, Any], text: str) -> float:
        """Estimate transcription confidence"""
        # If segments available, use average of segment probabilities
        segments = api_response.get("segments", [])
        if segments:
            total_confidence = 0.0
            total_duration = 0.0
            
            for segment in segments:
                segment_duration = segment.get("end", 0) - segment.get("start", 0)
                segment_confidence = segment.get("avg_logprob", -1.0)
                
                # Convert log probability to confidence score (0-1)
                if segment_confidence > -1.0:
                    confidence = min(1.0, max(0.0, (segment_confidence + 3.0) / 3.0))
                else:
                    confidence = 0.5  # Default confidence
                
                total_confidence += confidence * segment_duration
                total_duration += segment_duration
            
            if total_duration > 0:
                return total_confidence / total_duration
        
        # Fallback: estimate based on text characteristics
        if not text.strip():
            return 0.0
        elif len(text.split()) < 3:
            return 0.6  # Low confidence for very short text
        else:
            return 0.8  # Default confidence for reasonable text
    
    def _generate_cache_key(self, audio_data: bytes, language: str) -> str:
        """Generate cache key for audio data"""
        import hashlib
        
        # Create hash of audio data and language
        hasher = hashlib.md5()
        hasher.update(audio_data)
        hasher.update(language.encode())
        
        return hasher.hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[STTResult]:
        """Get cached STT result if available"""
        if cache_key in self.result_cache:
            cached_result, timestamp = self.result_cache[cache_key]
            
            # Check if cache is still valid
            if time.time() - timestamp < self.cache_ttl:
                return cached_result
            else:
                # Remove expired cache entry
                del self.result_cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: STTResult):
        """Cache STT result"""
        self.result_cache[cache_key] = (result, time.time())
        
        # Clean up old cache entries periodically
        if len(self.result_cache) > 100:  # Max 100 cached results
            self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.result_cache.items()
            if current_time - timestamp > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.result_cache[key]
        
        self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def validate_audio_for_stt(self, audio_data: bytes) -> bool:
        """Validate audio data for STT processing"""
        try:
            # Check audio quality
            quality_report = await self.audio_processor.validate_audio_quality(audio_data)
            
            # Check duration
            if quality_report.duration_seconds < self.min_audio_duration:
                self.logger.warning(f"Audio too short: {quality_report.duration_seconds}s")
                return False
            
            if quality_report.duration_seconds > self.max_audio_duration:
                self.logger.warning(f"Audio too long: {quality_report.duration_seconds}s")
                return False
            
            # Check quality grade
            if quality_report.quality_grade == "poor":
                self.logger.warning("Audio quality is poor for STT")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Audio validation failed: {str(e)}")
            return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get STT service status"""
        return {
            "service_name": "STT Service",
            "status": "active",
            "supported_languages": self.supported_languages,
            "default_language": self.default_language,
            "whisper_model": self.whisper_model,
            "cache_size": len(self.result_cache),
            "max_audio_duration": self.max_audio_duration,
            "min_audio_duration": self.min_audio_duration
        } 