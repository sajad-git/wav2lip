"""
OpenAI TTS with Persian language optimization
"""
import asyncio
import logging
import time
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from cachetools import TTLCache
import openai

from app.models.chunk_models import AudioChunk, AudioQualityMetrics, ChunkMetadata


@dataclass 
class TTSQualitySettings:
    """Voice and quality configuration"""
    voice: str = "alloy"  # alloy, echo, fable, onyx, nova, shimmer
    model: str = "tts-1"  # tts-1, tts-1-hd
    speed: float = 1.0
    response_format: str = "mp3"
    quality_level: str = "standard"  # standard, hd


class PersianTextProcessor:
    """Persian text optimization for TTS"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Persian-specific patterns
        self.persian_numbers = {
            '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
            '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
        }
        
        # Arabic numbers to Persian
        self.arabic_numbers = {
            '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
            '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
        }
        
        # Common Persian abbreviations
        self.persian_abbreviations = {
            'ج.ا.ا': 'جمهوری اسلامی ایران',
            'ع': 'علیه السلام',
            'ص': 'صلی الله علیه وآله',
            'ره': 'رحمة الله علیه',
            'قدس': 'قدس سره'
        }
    
    def normalize_unicode(self, text: str) -> str:
        """Unicode normalization for Persian text"""
        import unicodedata
        
        # Normalize to NFD (decomposed form) then back to NFC
        text = unicodedata.normalize('NFD', text)
        text = unicodedata.normalize('NFC', text)
        
        # Fix Arabic Yeh and Keh to Persian
        text = text.replace('ي', 'ی')  # Arabic Yeh to Persian Yeh
        text = text.replace('ك', 'ک')  # Arabic Keh to Persian Keh
        
        # Fix Arabic numbers to Persian
        for arabic, persian in self.arabic_numbers.items():
            text = text.replace(arabic, persian)
        
        return text
    
    def optimize_punctuation(self, text: str) -> str:
        """Optimize punctuation for natural speech pauses"""
        # Add pauses after Persian punctuation
        text = re.sub(r'([۔،؛؟!])', r'\1 ', text)
        
        # Optimize for speech rhythm
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'([\.]{2,})', '... ', text)  # Multiple dots
        
        # Add pauses for better speech flow
        text = re.sub(r'(\d+)', r' \1 ', text)  # Numbers
        text = re.sub(r'([A-Za-z]+)', r' \1 ', text)  # English words
        
        return text.strip()
    
    def handle_numbers(self, text: str) -> str:
        """Convert Persian numbers for better TTS pronunciation"""
        # Convert Persian digits to words for better pronunciation
        persian_digit_words = {
            '۰': 'صفر', '۱': 'یک', '۲': 'دو', '۳': 'سه', '۴': 'چهار',
            '۵': 'پنج', '۶': 'شش', '۷': 'هفت', '۸': 'هشت', '۹': 'نه'
        }
        
        # Replace isolated digits with words
        for digit, word in persian_digit_words.items():
            text = re.sub(rf'\b{digit}\b', word, text)
        
        return text
    
    def validate_rtl_text(self, text: str) -> bool:
        """Validate RTL text direction"""
        persian_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha():
                total_chars += 1
                if '\u0600' <= char <= '\u06FF':  # Persian/Arabic Unicode range
                    persian_chars += 1
        
        if total_chars == 0:
            return True  # Empty or no alphabetic characters
        
        return persian_chars / total_chars > 0.5  # Majority Persian


class OptimizedTTSService:
    """OpenAI TTS with Persian language optimization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.openai_client = None
        self.persian_processor = PersianTextProcessor()
        self.audio_cache = TTLCache(maxsize=100, ttl=3600)  # 1 hour TTL
        self.quality_settings = TTSQualitySettings()
        
        # Performance metrics
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        self.errors = 0
        
        self.logger.info("🎤 Optimized TTS Service created")
    
    async def initialize(self, openai_api_key: str) -> None:
        """Initialize OpenAI client"""
        try:
            self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
            
            # Test connection
            await self._test_connection()
            
            self.logger.info("✅ TTS Service initialized with OpenAI")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize TTS service: {str(e)}")
            raise
    
    async def process_persian_text_for_tts(self, text: str) -> str:
        """Preprocess Persian text for optimal speech generation"""
        try:
            if not text or not text.strip():
                return ""
            
            # Step 1: Normalize Persian Unicode characters
            normalized_text = self.persian_processor.normalize_unicode(text)
            
            # Step 2: Handle right-to-left text direction
            if not self.persian_processor.validate_rtl_text(normalized_text):
                self.logger.warning("⚠️ Text appears to be mixed or non-Persian")
            
            # Step 3: Optimize punctuation for natural speech pauses
            punctuation_optimized = self.persian_processor.optimize_punctuation(normalized_text)
            
            # Step 4: Handle numbers for better pronunciation
            number_optimized = self.persian_processor.handle_numbers(punctuation_optimized)
            
            # Step 5: Validate text length and structure
            if len(number_optimized) > 4000:  # OpenAI TTS limit
                self.logger.warning(f"⚠️ Text length ({len(number_optimized)}) exceeds recommended limit")
                number_optimized = number_optimized[:4000] + "..."
            
            # Step 6: Apply Persian-specific pronunciation rules
            final_text = self._apply_persian_pronunciation_rules(number_optimized)
            
            self.logger.debug(f"📝 Persian text processed: {len(text)} -> {len(final_text)} chars")
            return final_text
            
        except Exception as e:
            self.logger.error(f"❌ Persian text processing failed: {str(e)}")
            return text  # Return original text as fallback
    
    async def generate_chunked_audio_optimized(self, text: str, voice: str = "alloy") -> List[AudioChunk]:
        """Generate audio chunks optimized for Persian speech"""
        if not self.openai_client:
            raise RuntimeError("TTS service not initialized")
        
        start_time = time.time()
        
        try:
            self.logger.info(f"🎤 Generating TTS audio for text length: {len(text)}")
            
            # Process Persian text
            processed_text = await self.process_persian_text_for_tts(text)
            
            # Split text at natural Persian pause points
            text_chunks = self._split_text_for_persian_speech(processed_text)
            
            # Generate TTS audio for each chunk
            audio_chunks = []
            total_duration = 0.0
            
            for i, chunk_text in enumerate(text_chunks):
                try:
                    # Check cache first
                    cache_key = self._generate_cache_key(chunk_text, voice)
                    
                    if cache_key in self.audio_cache:
                        audio_data = self.audio_cache[cache_key]
                        self.cache_hits += 1
                        self.logger.debug(f"📦 Cache hit for chunk {i}")
                    else:
                        # Generate audio using OpenAI API
                        audio_data = await self._generate_tts_audio(chunk_text, voice)
                        
                        # Cache result
                        self.audio_cache[cache_key] = audio_data
                    
                    # Create audio chunk with metadata
                    chunk_duration = self._estimate_audio_duration(chunk_text)
                    
                    audio_chunk = AudioChunk(
                        chunk_id=f"tts_chunk_{i}",
                        audio_data=audio_data,
                        duration_seconds=chunk_duration,
                        start_time=total_duration,
                        end_time=total_duration + chunk_duration,
                        sample_rate=24000,  # OpenAI TTS output
                        metadata=ChunkMetadata(
                            processing_time=time.time() - start_time,
                            model_used=self.quality_settings.model,
                            avatar_id="",  # Will be set later
                            face_cache_hit=False,
                            quality_settings={"voice": voice, "model": self.quality_settings.model},
                            gpu_memory_used=0
                        ),
                        quality_metrics=AudioQualityMetrics(
                            snr_ratio=30.0,  # Estimated for TTS
                            clarity_score=0.95,  # High for TTS
                            duration_seconds=chunk_duration,
                            quality_grade="excellent",
                            recommendations=[]
                        )
                    )
                    
                    audio_chunks.append(audio_chunk)
                    total_duration += chunk_duration
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to generate chunk {i}: {str(e)}")
                    continue
            
            processing_time = time.time() - start_time
            self.total_requests += 1
            self.total_processing_time += processing_time
            
            self.logger.info(f"✅ Generated {len(audio_chunks)} TTS chunks in {processing_time:.2f}s")
            return audio_chunks
            
        except Exception as e:
            self.errors += 1
            self.logger.error(f"❌ TTS generation failed: {str(e)}")
            raise
    
    async def handle_tts_api_errors(self, error: Exception, text: str, retry_count: int) -> Optional[bytes]:
        """Robust error handling with retry logic"""
        try:
            self.logger.warning(f"⚠️ TTS API error (attempt {retry_count}): {str(error)}")
            
            # Analyze error type
            if "rate_limit" in str(error).lower():
                # Rate limiting - exponential backoff
                delay = min(2 ** retry_count, 60)  # Max 60 seconds
                self.logger.info(f"💤 Rate limited, waiting {delay}s before retry")
                await asyncio.sleep(delay)
                
                if retry_count < 3:
                    return await self._generate_tts_audio(text, self.quality_settings.voice)
            
            elif "quota" in str(error).lower():
                # Quota exceeded - check for cached alternatives
                self.logger.error("❌ OpenAI quota exceeded")
                return self._get_cached_fallback(text)
            
            elif "content_policy" in str(error).lower():
                # Content policy violation
                self.logger.warning("⚠️ Content policy violation, sanitizing text")
                sanitized_text = self._sanitize_text_for_tts(text)
                
                if sanitized_text != text and retry_count < 2:
                    return await self._generate_tts_audio(sanitized_text, self.quality_settings.voice)
            
            # Fallback to alternative voice or quality
            if retry_count < 2:
                alternative_voice = self._get_alternative_voice()
                self.logger.info(f"🔄 Retrying with alternative voice: {alternative_voice}")
                return await self._generate_tts_audio(text, alternative_voice)
            
            # Final fallback - check cache for similar content
            return self._get_cached_fallback(text)
            
        except Exception as fallback_error:
            self.logger.error(f"❌ Fallback handling failed: {str(fallback_error)}")
            return None
    
    async def get_service_metrics(self) -> Dict[str, Any]:
        """Get TTS service performance metrics"""
        avg_processing_time = 0.0
        if self.total_requests > 0:
            avg_processing_time = self.total_processing_time / self.total_requests
        
        cache_hit_rate = 0.0
        if self.total_requests > 0:
            cache_hit_rate = self.cache_hits / self.total_requests
        
        return {
            "total_requests": self.total_requests,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "errors": self.errors,
            "cache_size": len(self.audio_cache),
            "quality_settings": {
                "voice": self.quality_settings.voice,
                "model": self.quality_settings.model,
                "speed": self.quality_settings.speed
            }
        }
    
    async def _test_connection(self) -> None:
        """Test OpenAI TTS connection"""
        try:
            test_text = "تست اتصال"
            await self._generate_tts_audio(test_text, "alloy")
            self.logger.info("✅ OpenAI TTS connection test successful")
            
        except Exception as e:
            self.logger.error(f"❌ OpenAI TTS connection test failed: {str(e)}")
            raise
    
    async def _generate_tts_audio(self, text: str, voice: str) -> bytes:
        """Generate TTS audio using OpenAI API"""
        try:
            response = await self.openai_client.audio.speech.create(
                model=self.quality_settings.model,
                voice=voice,
                input=text,
                response_format=self.quality_settings.response_format,
                speed=self.quality_settings.speed
            )
            
            return response.content
            
        except Exception as e:
            self.logger.error(f"❌ OpenAI TTS API call failed: {str(e)}")
            raise
    
    def _split_text_for_persian_speech(self, text: str) -> List[str]:
        """Split text at natural Persian pause points"""
        # Split on Persian sentence endings
        sentences = re.split(r'[۔؟!]', text)
        
        chunks = []
        current_chunk = ""
        max_chunk_length = 500  # Characters per chunk
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed limit, finalize current chunk
            if len(current_chunk) + len(sentence) > max_chunk_length and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def _apply_persian_pronunciation_rules(self, text: str) -> str:
        """Apply Persian-specific pronunciation optimization"""
        # Handle common Persian pronunciation issues
        replacements = {
            'خواهید': 'خواهد',  # Simplified pronunciation
            'میخواهم': 'می خوام',  # Colloquial form
            'هستید': 'هستین',  # Common pronunciation
        }
        
        for formal, colloquial in replacements.items():
            text = text.replace(formal, colloquial)
        
        return text
    
    def _estimate_audio_duration(self, text: str) -> float:
        """Estimate audio duration based on text length"""
        # Persian speech rate: approximately 2.5 characters per second
        words = len(text.split())
        chars = len(text)
        
        # Estimate based on character count (more accurate for Persian)
        duration = chars / 12.0  # Adjusted for Persian speech rate
        
        # Add pauses for punctuation
        punctuation_count = sum(1 for char in text if char in '،؛۔؟!')
        duration += punctuation_count * 0.3  # 300ms per punctuation pause
        
        return max(0.5, duration)  # Minimum 0.5 seconds
    
    def _generate_cache_key(self, text: str, voice: str) -> str:
        """Generate cache key for text and voice combination"""
        import hashlib
        content = f"{text}_{voice}_{self.quality_settings.model}_{self.quality_settings.speed}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_fallback(self, text: str) -> Optional[bytes]:
        """Get cached audio for similar text as fallback"""
        # Try to find similar cached content
        for cache_key, audio_data in self.audio_cache.items():
            # This is a simplified similarity check
            # In production, you might use more sophisticated text similarity
            if len(text) > 10 and text[:10] in cache_key:
                self.logger.info("📦 Using cached fallback audio")
                return audio_data
        
        return None
    
    def _get_alternative_voice(self) -> str:
        """Get alternative voice for fallback"""
        voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        current_voice = self.quality_settings.voice
        
        # Return next voice in the list
        try:
            current_index = voices.index(current_voice)
            return voices[(current_index + 1) % len(voices)]
        except ValueError:
            return "alloy"  # Default fallback
    
    def _sanitize_text_for_tts(self, text: str) -> str:
        """Sanitize text to avoid content policy violations"""
        # Remove potentially problematic content
        # This is a basic implementation - extend as needed
        sanitized = re.sub(r'[^\u0600-\u06FF\u0020-\u007E\s]', '', text)
        return sanitized.strip() 