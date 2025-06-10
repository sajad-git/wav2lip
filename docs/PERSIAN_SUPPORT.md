# Persian Language Support Documentation

## Overview

The Avatar Streaming Service includes comprehensive support for Persian (Farsi) language processing, including text handling, speech recognition, text-to-speech generation, and cultural considerations for Persian speakers.

## Persian Language Features

### Text Processing Capabilities
- **Unicode Support**: Full UTF-8 Persian character set support
- **RTL (Right-to-Left) Layout**: Proper text direction handling
- **Character Normalization**: Standardization of Persian Unicode variations
- **Diacritic Handling**: Support for Persian diacritical marks
- **Number Conversion**: Persian-Arabic numeral handling
- **Punctuation Optimization**: Persian-specific punctuation rules

### Speech Technology Integration
- **STT (Speech-to-Text)**: Persian-optimized Whisper model integration
- **TTS (Text-to-Speech)**: OpenAI TTS with Persian pronunciation optimization
- **Phonetic Processing**: Persian phoneme recognition and generation
- **Prosody Optimization**: Natural Persian speech patterns

---

## Persian Text Processing

### Unicode and Character Handling

#### Supported Character Sets
```python
# Persian Unicode ranges
PERSIAN_RANGES = [
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
]

# Core Persian letters
PERSIAN_LETTERS = {
    'ا': 'ALEF',
    'ب': 'BEH',
    'پ': 'PEH',
    'ت': 'TEH',
    'ث': 'THEH',
    'ج': 'JEEM',
    'چ': 'CHEH',
    'ح': 'HAH',
    'خ': 'KHAH',
    'د': 'DAL',
    'ذ': 'ZAL',
    'ر': 'REH',
    'ز': 'ZAIN',
    'ژ': 'JEH',
    'س': 'SEEN',
    'ش': 'SHEEN',
    'ص': 'SAD',
    'ض': 'ZAD',
    'ط': 'TAH',
    'ظ': 'ZAH',
    'ع': 'AIN',
    'غ': 'GHAIN',
    'ف': 'FEH',
    'ق': 'QAF',
    'ک': 'KAF',
    'گ': 'GAF',
    'ل': 'LAM',
    'م': 'MEEM',
    'ن': 'NOON',
    'و': 'VAV',
    'ه': 'HEH',
    'ی': 'YEH'
}
```

#### Text Normalization
```python
class PersianTextNormalizer:
    def __init__(self):
        self.normalization_map = {
            # Arabic-Indic digits to Persian
            '٠': '۰', '١': '۱', '٢': '۲', '٣': '۳', '٤': '۴',
            '٥': '۵', '٦': '۶', '٧': '۷', '٨': '۸', '٩': '۹',
            
            # Arabic characters to Persian equivalents
            'ي': 'ی',  # Arabic YEH to Persian YEH
            'ك': 'ک',  # Arabic KAF to Persian KAF
            'ة': 'ه',  # Arabic TEH MARBUTA to Persian HEH
            
            # Zero-width characters
            '\u200C': '\u200C',  # ZWNJ (half-space)
            '\u200D': '',        # ZWJ (remove)
        }
    
    def normalize(self, text: str) -> str:
        """Normalize Persian text for consistent processing"""
        # Apply character normalization
        for old, new in self.normalization_map.items():
            text = text.replace(old, new)
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Clean extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def is_persian_text(self, text: str) -> bool:
        """Check if text contains Persian characters"""
        persian_char_count = 0
        total_chars = len([c for c in text if c.isalpha()])
        
        for char in text:
            if any(start <= ord(char) <= end for start, end in PERSIAN_RANGES):
                persian_char_count += 1
        
        return persian_char_count / max(total_chars, 1) > 0.5
```

### RTL (Right-to-Left) Support

#### Text Direction Handling
```python
class RTLTextProcessor:
    def __init__(self):
        self.rtl_chars = set()
        for start, end in PERSIAN_RANGES:
            self.rtl_chars.update(chr(i) for i in range(start, end + 1))
    
    def detect_text_direction(self, text: str) -> str:
        """Detect primary text direction"""
        rtl_count = sum(1 for char in text if char in self.rtl_chars)
        ltr_count = sum(1 for char in text if char.isascii() and char.isalpha())
        
        if rtl_count > ltr_count:
            return 'rtl'
        elif ltr_count > rtl_count:
            return 'ltr'
        else:
            return 'mixed'
    
    def apply_bidi_algorithm(self, text: str) -> str:
        """Apply bidirectional text algorithm"""
        # Implementation of Unicode Bidirectional Algorithm
        # for proper RTL/LTR text mixing
        return text  # Simplified for example
    
    def format_for_display(self, text: str) -> str:
        """Format text for proper RTL display"""
        direction = self.detect_text_direction(text)
        
        if direction == 'rtl':
            # Apply RTL formatting
            text = f"‏{text}‎"  # RLM + text + LRM
        
        return text
```

### Persian-Specific Text Processing

#### Sentence and Word Segmentation
```python
class PersianSegmenter:
    def __init__(self):
        self.sentence_endings = ['؟', '!', '؛', '.']
        self.word_separators = [' ', '\u200C']  # Space and ZWNJ
    
    def split_sentences(self, text: str) -> List[str]:
        """Split Persian text into sentences"""
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in self.sentence_endings:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return sentences
    
    def split_words(self, text: str) -> List[str]:
        """Split Persian text into words considering ZWNJ"""
        # Persian words can be connected with ZWNJ (half-space)
        words = re.split(r'[\s]+', text)
        return [word.strip() for word in words if word.strip()]
    
    def optimize_for_tts(self, text: str) -> str:
        """Optimize Persian text for TTS processing"""
        # Add appropriate pauses for Persian TTS
        text = re.sub(r'([.!؟؛])', r'\1 ', text)  # Add space after punctuation
        text = re.sub(r'(\d+)', r' \1 ', text)    # Add spaces around numbers
        text = re.sub(r'\s+', ' ', text)          # Normalize spaces
        
        return text.strip()
```

---

## Speech Recognition (STT)

### Whisper Persian Integration

#### Model Configuration
```python
class PersianSTTService:
    def __init__(self):
        self.model_size = "medium"  # Optimal for Persian
        self.language = "fa"
        self.model = whisper.load_model(self.model_size)
        
        # Persian-specific options
        self.decode_options = {
            "language": "fa",
            "task": "transcribe",
            "temperature": 0.2,
            "no_speech_threshold": 0.6,
            "logprob_threshold": -1.0,
            "compression_ratio_threshold": 2.4,
            "condition_on_previous_text": True,
            "fp16": True  # Enable for GPU acceleration
        }
    
    def transcribe_persian_audio(self, audio_data: bytes) -> dict:
        """Transcribe Persian audio with optimizations"""
        # Load audio
        audio = whisper.load_audio(io.BytesIO(audio_data))
        
        # Transcribe with Persian-specific settings
        result = self.model.transcribe(
            audio,
            **self.decode_options
        )
        
        # Post-process Persian text
        if result.get("text"):
            result["text"] = self.post_process_persian_text(result["text"])
        
        return result
    
    def post_process_persian_text(self, text: str) -> str:
        """Post-process Persian transcription"""
        normalizer = PersianTextNormalizer()
        
        # Normalize text
        text = normalizer.normalize(text)
        
        # Fix common transcription errors
        text = self.fix_persian_transcription_errors(text)
        
        return text
    
    def fix_persian_transcription_errors(self, text: str) -> str:
        """Fix common Persian transcription errors"""
        corrections = {
            # Common Whisper mistakes for Persian
            'می کنم': 'می‌کنم',
            'می کند': 'می‌کند',
            'می باشد': 'می‌باشد',
            'می توان': 'می‌توان',
            'می شود': 'می‌شود',
            
            # Number corrections
            'یک': '۱',
            'دو': '۲',
            'سه': '۳',
            'چهار': '۴',
            'پنج': '۵',
        }
        
        for old, new in corrections.items():
            text = text.replace(old, new)
        
        return text
```

### Audio Preprocessing for Persian

#### Persian-Specific Audio Optimization
```python
class PersianAudioProcessor:
    def __init__(self):
        self.sample_rate = 16000
        self.persian_freq_range = (80, 8000)  # Optimal for Persian speech
    
    def preprocess_persian_audio(self, audio_data: bytes) -> np.ndarray:
        """Preprocess audio for Persian speech recognition"""
        # Load audio
        audio, sr = librosa.load(io.BytesIO(audio_data), sr=self.sample_rate)
        
        # Apply Persian-specific filtering
        audio = self.apply_persian_frequency_filter(audio)
        
        # Enhance Persian consonants
        audio = self.enhance_persian_consonants(audio)
        
        # Normalize for Persian speech patterns
        audio = self.normalize_persian_speech(audio)
        
        return audio
    
    def apply_persian_frequency_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply frequency filtering optimal for Persian"""
        # Bandpass filter for Persian speech
        from scipy.signal import butter, filtfilt
        
        nyquist = self.sample_rate // 2
        low = self.persian_freq_range[0] / nyquist
        high = self.persian_freq_range[1] / nyquist
        
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, audio)
    
    def enhance_persian_consonants(self, audio: np.ndarray) -> np.ndarray:
        """Enhance Persian consonant clarity"""
        # Apply spectral enhancement for Persian consonants
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Enhance high-frequency components (consonants)
        magnitude[magnitude.shape[0]//2:] *= 1.2
        
        # Reconstruct audio
        enhanced_stft = magnitude * np.exp(1j * phase)
        return librosa.istft(enhanced_stft)
    
    def normalize_persian_speech(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio for Persian speech patterns"""
        # Apply dynamic range compression suitable for Persian
        return librosa.util.normalize(audio)
```

---

## Text-to-Speech (TTS)

### OpenAI TTS Persian Optimization

#### Persian TTS Configuration
```python
class PersianTTSService:
    def __init__(self):
        self.client = OpenAI()
        self.voice = "alloy"  # Best voice for Persian
        self.model = "tts-1"
        self.persian_processor = PersianTextProcessor()
    
    def generate_persian_speech(self, text: str) -> bytes:
        """Generate speech for Persian text"""
        # Preprocess Persian text for TTS
        processed_text = self.preprocess_for_tts(text)
        
        # Generate speech
        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=processed_text,
            response_format="mp3",
            speed=0.9  # Slightly slower for Persian clarity
        )
        
        # Post-process audio for Persian
        audio_data = response.content
        return self.post_process_persian_audio(audio_data)
    
    def preprocess_for_tts(self, text: str) -> str:
        """Preprocess Persian text for optimal TTS"""
        # Normalize text
        text = self.persian_processor.normalize_text(text)
        
        # Handle Persian numbers
        text = self.convert_persian_numbers_to_words(text)
        
        # Add pronunciation hints
        text = self.add_persian_pronunciation_hints(text)
        
        # Optimize punctuation for natural pauses
        text = self.optimize_persian_punctuation(text)
        
        return text
    
    def convert_persian_numbers_to_words(self, text: str) -> str:
        """Convert Persian numbers to words for better TTS"""
        number_words = {
            '۰': 'صفر',
            '۱': 'یک',
            '۲': 'دو',
            '۳': 'سه',
            '۴': 'چهار',
            '۵': 'پنج',
            '۶': 'شش',
            '۷': 'هفت',
            '۸': 'هشت',
            '۹': 'نه'
        }
        
        for digit, word in number_words.items():
            text = text.replace(digit, word)
        
        return text
    
    def add_persian_pronunciation_hints(self, text: str) -> str:
        """Add pronunciation hints for difficult Persian words"""
        pronunciation_hints = {
            'می‌شود': 'می شَوَد',
            'می‌کند': 'می کُنَد',
            'می‌باشد': 'می باشَد',
            'خواهد': 'خواهَد',
            'هستند': 'هَستَند'
        }
        
        for word, hint in pronunciation_hints.items():
            text = text.replace(word, hint)
        
        return text
    
    def optimize_persian_punctuation(self, text: str) -> str:
        """Optimize punctuation for natural Persian speech"""
        # Add pauses after specific Persian conjunctions
        text = re.sub(r'(اما|ولی|لیکن|امّا)', r'\1، ', text)
        
        # Add emphasis to question words
        text = re.sub(r'(چرا|چه|کی|کجا|چطور)', r'\1 ', text)
        
        # Optimize sentence endings
        text = re.sub(r'([.!؟])', r'\1 ', text)
        
        return text
```

### Persian Audio Enhancement

#### Post-Processing for Persian TTS
```python
class PersianAudioEnhancer:
    def __init__(self):
        self.target_sample_rate = 22050
        
    def enhance_persian_tts_audio(self, audio_data: bytes) -> bytes:
        """Enhance TTS audio for Persian speech"""
        # Load audio
        audio, sr = librosa.load(io.BytesIO(audio_data), sr=self.target_sample_rate)
        
        # Apply Persian-specific enhancements
        audio = self.enhance_persian_phonemes(audio)
        audio = self.adjust_persian_prosody(audio)
        audio = self.normalize_persian_volume(audio)
        
        # Convert back to bytes
        return self.audio_to_bytes(audio, sr)
    
    def enhance_persian_phonemes(self, audio: np.ndarray) -> np.ndarray:
        """Enhance Persian phoneme clarity"""
        # Apply spectral enhancement for Persian sounds
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Enhance Persian-specific frequency ranges
        # Persian vowels: 300-2000 Hz
        # Persian consonants: 2000-8000 Hz
        freq_bins = magnitude.shape[0]
        vowel_range = slice(int(freq_bins * 300/11025), int(freq_bins * 2000/11025))
        consonant_range = slice(int(freq_bins * 2000/11025), int(freq_bins * 8000/11025))
        
        magnitude[vowel_range] *= 1.1      # Enhance vowels
        magnitude[consonant_range] *= 1.15  # Enhance consonants
        
        # Reconstruct audio
        enhanced_stft = magnitude * np.exp(1j * phase)
        return librosa.istft(enhanced_stft)
    
    def adjust_persian_prosody(self, audio: np.ndarray) -> np.ndarray:
        """Adjust prosody for natural Persian speech"""
        # Extract pitch
        f0 = librosa.yin(audio, fmin=80, fmax=400)
        
        # Apply Persian prosody patterns
        # Persian typically has falling intonation at sentence end
        # and rising intonation for questions
        
        return audio  # Simplified for example
    
    def normalize_persian_volume(self, audio: np.ndarray) -> np.ndarray:
        """Normalize volume for consistent Persian speech"""
        # Apply dynamic range compression
        compressed = librosa.util.normalize(audio)
        
        # Apply gentle limiting
        peak_threshold = 0.95
        compressed = np.clip(compressed, -peak_threshold, peak_threshold)
        
        return compressed
```

---

## Persian Language Validation

### Text Validation
```python
class PersianValidator:
    def __init__(self):
        self.min_persian_ratio = 0.7
        self.max_text_length = 5000
        
    def validate_persian_text(self, text: str) -> dict:
        """Validate Persian text input"""
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Check text length
        if len(text) > self.max_text_length:
            results['errors'].append(f"Text too long: {len(text)} chars")
            results['is_valid'] = False
        
        # Check Persian content ratio
        persian_ratio = self.calculate_persian_ratio(text)
        results['metrics']['persian_ratio'] = persian_ratio
        
        if persian_ratio < self.min_persian_ratio:
            results['warnings'].append(
                f"Low Persian content: {persian_ratio:.2%}"
            )
        
        # Check for problematic characters
        problematic_chars = self.find_problematic_characters(text)
        if problematic_chars:
            results['warnings'].append(
                f"Problematic chars: {problematic_chars}"
            )
        
        # Check text direction consistency
        direction_issues = self.check_text_direction(text)
        if direction_issues:
            results['warnings'].extend(direction_issues)
        
        return results
    
    def calculate_persian_ratio(self, text: str) -> float:
        """Calculate ratio of Persian characters"""
        persian_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha():
                total_chars += 1
                if any(start <= ord(char) <= end for start, end in PERSIAN_RANGES):
                    persian_chars += 1
        
        return persian_chars / max(total_chars, 1)
    
    def find_problematic_characters(self, text: str) -> List[str]:
        """Find characters that might cause processing issues"""
        problematic = []
        
        # Check for mixed scripts that might cause issues
        for char in text:
            code_point = ord(char)
            if 0x0600 <= code_point <= 0x06FF:  # Arabic block
                if char not in PERSIAN_LETTERS:
                    problematic.append(char)
        
        return list(set(problematic))
    
    def check_text_direction(self, text: str) -> List[str]:
        """Check for text direction inconsistencies"""
        issues = []
        
        # Check for improper mixing of RTL and LTR
        rtl_processor = RTLTextProcessor()
        direction = rtl_processor.detect_text_direction(text)
        
        if direction == 'mixed':
            issues.append("Mixed text direction detected")
        
        return issues
```

---

## Persian Cultural Considerations

### Cultural Context Processing
```python
class PersianCulturalProcessor:
    def __init__(self):
        self.formal_indicators = [
            'جناب', 'خانم', 'آقای', 'سرکار خانم', 'محترم'
        ]
        self.informal_indicators = [
            'داداش', 'رفیق', 'دوست', 'عزیز'
        ]
        
    def detect_formality_level(self, text: str) -> str:
        """Detect formality level in Persian text"""
        formal_count = sum(1 for word in self.formal_indicators if word in text)
        informal_count = sum(1 for word in self.informal_indicators if word in text)
        
        if formal_count > informal_count:
            return 'formal'
        elif informal_count > formal_count:
            return 'informal'
        else:
            return 'neutral'
    
    def adapt_response_style(self, text: str, formality: str) -> str:
        """Adapt response style based on formality"""
        if formality == 'formal':
            # Use formal Persian expressions
            text = text.replace('تو', 'شما')
            text = text.replace('می‌کنی', 'می‌کنید')
        elif formality == 'informal':
            # Use informal Persian expressions
            text = text.replace('شما', 'تو')
            text = text.replace('می‌کنید', 'می‌کنی')
        
        return text
    
    def handle_persian_honorifics(self, text: str) -> str:
        """Properly handle Persian honorifics and titles"""
        honorifics = {
            'حضرت': 'His/Her Holiness',
            'جناب آقای': 'Mr.',
            'سرکار خانم': 'Ms.',
            'دکتر': 'Dr.',
            'استاد': 'Professor'
        }
        
        # Process honorifics appropriately for TTS
        for persian, english in honorifics.items():
            if persian in text:
                # Add appropriate pauses after honorifics
                text = text.replace(persian, f"{persian}، ")
        
        return text
```

### Persian Date and Time Handling
```python
class PersianDateTimeProcessor:
    def __init__(self):
        self.persian_months = [
            'فروردین', 'اردیبهشت', 'خرداد', 'تیر',
            'مرداد', 'شهریور', 'مهر', 'آبان',
            'آذر', 'دی', 'بهمن', 'اسفند'
        ]
        
        self.persian_weekdays = [
            'شنبه', 'یکشنبه', 'دوشنبه', 'سه‌شنبه',
            'چهارشنبه', 'پنج‌شنبه', 'جمعه'
        ]
    
    def convert_to_persian_date(self, gregorian_date: datetime) -> str:
        """Convert Gregorian date to Persian solar date"""
        # Implementation would use jdatetime or similar library
        # This is a simplified example
        return f"{gregorian_date.day} {self.persian_months[gregorian_date.month-1]} {gregorian_date.year}"
    
    def format_time_for_persian(self, time_str: str) -> str:
        """Format time strings for Persian pronunciation"""
        # Convert 24-hour to Persian time format
        return time_str  # Simplified for example
```

---

## Testing Persian Functionality

### Persian Language Test Suite
```python
class PersianLanguageTests:
    def __init__(self):
        self.test_texts = [
            "سلام، چطورید؟",
            "امروز هوا خیلی خوب است.",
            "می‌توانید کمکم کنید؟",
            "من از ایران هستم و فارسی صحبت می‌کنم.",
            "۱۲۳۴۵۶۷۸۹۰",  # Persian numbers
            "What is the weather like today?",  # Mixed language
            "این یک متن آزمایشی برای بررسی عملکرد سیستم است."
        ]
    
    def test_text_normalization(self):
        """Test Persian text normalization"""
        normalizer = PersianTextNormalizer()
        
        for text in self.test_texts:
            normalized = normalizer.normalize(text)
            print(f"Original: {text}")
            print(f"Normalized: {normalized}")
            print(f"Is Persian: {normalizer.is_persian_text(text)}")
            print("---")
    
    def test_stt_pipeline(self):
        """Test Persian STT pipeline"""
        stt_service = PersianSTTService()
        
        # Test with sample Persian audio files
        test_audio_files = [
            "persian_greeting.wav",
            "persian_question.wav",
            "persian_numbers.wav"
        ]
        
        for audio_file in test_audio_files:
            if os.path.exists(audio_file):
                with open(audio_file, 'rb') as f:
                    audio_data = f.read()
                
                result = stt_service.transcribe_persian_audio(audio_data)
                print(f"Audio: {audio_file}")
                print(f"Transcription: {result.get('text', 'No transcription')}")
                print("---")
    
    def test_tts_pipeline(self):
        """Test Persian TTS pipeline"""
        tts_service = PersianTTSService()
        
        for text in self.test_texts[:5]:  # Test first 5 texts
            if PersianTextNormalizer().is_persian_text(text):
                try:
                    audio_data = tts_service.generate_persian_speech(text)
                    print(f"Generated TTS for: {text}")
                    print(f"Audio size: {len(audio_data)} bytes")
                except Exception as e:
                    print(f"TTS failed for '{text}': {e}")
                print("---")
    
    def test_avatar_processing_persian(self):
        """Test avatar processing with Persian content"""
        test_scenarios = [
            {
                'text': 'سلام، من یک آواتار هستم.',
                'expected_chunks': 1,
                'avatar_id': 'persian_male_avatar'
            },
            {
                'text': 'امروز هوا خیلی خوب است و می‌توانیم بیرون برویم.',
                'expected_chunks': 2,
                'avatar_id': 'persian_female_avatar'
            }
        ]
        
        for scenario in test_scenarios:
            print(f"Testing: {scenario['text']}")
            # Mock avatar processing test
            print(f"Expected chunks: {scenario['expected_chunks']}")
            print("---")

# Run tests
if __name__ == "__main__":
    tests = PersianLanguageTests()
    print("=== Persian Text Normalization Tests ===")
    tests.test_text_normalization()
    
    print("\n=== Persian TTS Tests ===")
    tests.test_tts_pipeline()
    
    print("\n=== Persian Avatar Processing Tests ===")
    tests.test_avatar_processing_persian()
```

---

## Configuration and Best Practices

### Recommended Settings for Persian
```python
# Configuration for optimal Persian support
PERSIAN_CONFIG = {
    'text_processing': {
        'normalization': True,
        'rtl_support': True,
        'unicode_form': 'NFKC',
        'max_text_length': 5000
    },
    'stt': {
        'model': 'medium',
        'language': 'fa',
        'temperature': 0.2,
        'audio_enhancement': True
    },
    'tts': {
        'voice': 'alloy',
        'speed': 0.9,
        'model': 'tts-1',
        'persian_optimization': True
    },
    'avatar': {
        'chunk_size': 8,  # Longer chunks for Persian
        'quality': 'balanced',
        'persian_prosody': True
    }
}
```

This documentation provides comprehensive coverage of Persian language support in the Avatar Streaming Service, including technical implementation details and cultural considerations for optimal Persian language processing. 