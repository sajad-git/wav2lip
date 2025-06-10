# Avatar Registration System Documentation

## Overview

The Avatar Registration System enables users to upload and register custom avatars with advanced face detection, caching, and optimization for real-time lip-sync processing. This system eliminates face detection latency during runtime through pre-processing and caching strategies.

## Key Features

### Cold Avatar Registration
- **One-time Processing**: Face detection performed once during registration
- **Pre-computed Face Data**: Bounding boxes, landmarks, and cropped regions cached
- **Quality Validation**: Comprehensive face quality assessment
- **Multi-format Support**: Images (JPG, PNG) and videos (MP4, GIF, MOV)
- **Instant Processing**: Zero face detection delay during runtime

### Performance Benefits
- **Registration Time**: 2-5 seconds per avatar (one-time cost)
- **Runtime Performance**: <10ms face data retrieval
- **Cache Hit Rate**: >95% for registered avatars
- **Quality Assurance**: Face validation during registration, not runtime
- **Memory Efficiency**: Compressed face data storage

---

## Architecture Overview

### Components

```
Avatar Registration System
├── Avatar Registrar (Cold Registration Engine)
├── Face Cache Manager (Pre-processed Data Storage)
├── Avatar Validator (File and Quality Validation)
├── Face Detector Utils (InsightFace Integration)
├── Avatar Database (Metadata Storage)
└── Avatar Management Service (CRUD Operations)
```

### Data Flow

```
File Upload → Validation → Face Detection → Quality Assessment → Cache Generation → Database Storage
     ↓              ↓             ↓               ↓                ↓                 ↓
 Format/Size    File Integrity  InsightFace    Quality Metrics   Compressed      Avatar ID
 Validation     Verification    Processing     Face Analysis     Face Data       Generation
```

---

## Registration Process

### Step-by-Step Workflow

#### 1. File Upload and Validation
```python
class AvatarUploadHandler:
    def __init__(self):
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mov'}
        
    def validate_upload(self, file_data: bytes, filename: str) -> ValidationResult:
        """Validate uploaded avatar file"""
        # Check file size
        if len(file_data) > self.max_file_size:
            raise ValidationError(f"File too large: {len(file_data)} bytes")
        
        # Check file format
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.supported_formats:
            raise ValidationError(f"Unsupported format: {file_ext}")
        
        # Verify file integrity
        try:
            if file_ext in {'.jpg', '.jpeg', '.png'}:
                Image.open(io.BytesIO(file_data)).verify()
            elif file_ext in {'.mp4', '.mov', '.gif'}:
                # Video validation
                self.validate_video_file(file_data)
        except Exception as e:
            raise ValidationError(f"File corruption detected: {e}")
        
        return ValidationResult(is_valid=True)
```

#### 2. Frame Extraction and Processing
```python
class FrameExtractor:
    def __init__(self):
        self.max_frames = 100
        self.target_resolution = (512, 512)
        
    def extract_frames(self, file_data: bytes, file_format: str) -> List[np.ndarray]:
        """Extract frames from image or video"""
        frames = []
        
        if file_format.lower() in ['jpg', 'jpeg', 'png']:
            # Single image
            image = Image.open(io.BytesIO(file_data))
            frame = np.array(image.convert('RGB'))
            frame = self.resize_frame(frame)
            frames.append(frame)
            
        elif file_format.lower() in ['mp4', 'mov', 'gif']:
            # Video/animated content
            frames = self.extract_video_frames(file_data)
            
        return frames
    
    def extract_video_frames(self, video_data: bytes) -> List[np.ndarray]:
        """Extract frames from video file"""
        # Save temporary file for OpenCV
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_file:
            temp_file.write(video_data)
            temp_file.flush()
            
            cap = cv2.VideoCapture(temp_file.name)
            frames = []
            frame_count = 0
            
            while cap.isOpened() and frame_count < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.resize_frame(frame)
                frames.append(frame)
                frame_count += 1
                
            cap.release()
            
        return frames
```

#### 3. Face Detection and Analysis
```python
class FaceDetectionProcessor:
    def __init__(self):
        self.face_detector = insightface.app.FaceAnalysis(
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.face_detector.prepare(ctx_id=0, det_size=(640, 640))
        self.confidence_threshold = 0.5
        
    def detect_faces_in_frames(self, frames: List[np.ndarray]) -> FaceDetectionResult:
        """Detect faces in all frames with consistency checking"""
        detection_results = []
        
        for frame_idx, frame in enumerate(frames):
            # Detect faces in frame
            faces = self.face_detector.get(frame)
            
            if not faces:
                continue
                
            # Use the most confident face
            primary_face = max(faces, key=lambda x: x.det_score)
            
            if primary_face.det_score < self.confidence_threshold:
                continue
                
            # Extract face data
            face_data = self.extract_face_data(frame, primary_face)
            detection_results.append({
                'frame_index': frame_idx,
                'face_data': face_data,
                'confidence': primary_face.det_score
            })
        
        return self.analyze_detection_consistency(detection_results)
    
    def extract_face_data(self, frame: np.ndarray, face) -> FaceData:
        """Extract comprehensive face data for caching"""
        # Get bounding box
        bbox = face.bbox.astype(int)
        
        # Get facial landmarks
        landmarks = face.kps.astype(int)
        
        # Crop face region
        cropped_face = self.crop_face_region(frame, bbox, target_size=(96, 96))
        
        # Generate face mask
        face_mask = self.generate_face_mask(frame.shape[:2], landmarks)
        
        return FaceData(
            bbox=bbox,
            landmarks=landmarks,
            cropped_face=cropped_face,
            face_mask=face_mask,
            confidence=face.det_score
        )
    
    def analyze_detection_consistency(self, results: List[dict]) -> FaceDetectionResult:
        """Analyze face detection consistency across frames"""
        if not results:
            raise FaceDetectionError("No faces detected in any frame")
        
        # Calculate consistency metrics
        confidences = [r['confidence'] for r in results]
        avg_confidence = np.mean(confidences)
        
        # Check bbox consistency for videos
        consistency_score = 1.0
        if len(results) > 1:
            consistency_score = self.calculate_bbox_consistency(results)
        
        return FaceDetectionResult(
            face_data_list=[r['face_data'] for r in results],
            average_confidence=avg_confidence,
            consistency_score=consistency_score,
            frame_count=len(results),
            detection_success=True
        )
```

#### 4. Quality Assessment
```python
class AvatarQualityAssessor:
    def __init__(self):
        self.min_resolution = (256, 256)
        self.min_face_size = 64
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'acceptable': 0.5,
            'poor': 0.3
        }
        
    def assess_avatar_quality(self, frames: List[np.ndarray], 
                            detection_result: FaceDetectionResult) -> QualityAssessment:
        """Comprehensive avatar quality assessment"""
        scores = {}
        
        # Face detection quality
        scores['face_detection'] = detection_result.average_confidence
        scores['face_consistency'] = detection_result.consistency_score
        
        # Image quality assessment
        scores['image_clarity'] = self.assess_image_clarity(frames)
        scores['lighting_quality'] = self.assess_lighting(frames)
        scores['face_orientation'] = self.assess_face_orientation(detection_result)
        scores['resolution_adequacy'] = self.assess_resolution(frames)
        
        # Calculate overall score
        weights = {
            'face_detection': 0.3,
            'face_consistency': 0.2,
            'image_clarity': 0.2,
            'lighting_quality': 0.15,
            'face_orientation': 0.1,
            'resolution_adequacy': 0.05
        }
        
        overall_score = sum(scores[metric] * weight 
                          for metric, weight in weights.items())
        
        # Determine quality grade
        quality_grade = self.determine_quality_grade(overall_score)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(scores)
        
        return QualityAssessment(
            overall_score=overall_score,
            quality_grade=quality_grade,
            individual_scores=scores,
            recommendations=recommendations,
            processing_ready=overall_score >= self.quality_thresholds['acceptable']
        )
    
    def assess_image_clarity(self, frames: List[np.ndarray]) -> float:
        """Assess image sharpness and clarity"""
        clarity_scores = []
        
        for frame in frames:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Calculate Laplacian variance (measure of sharpness)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1 range
            clarity_score = min(laplacian_var / 1000.0, 1.0)
            clarity_scores.append(clarity_score)
        
        return np.mean(clarity_scores)
    
    def assess_lighting(self, frames: List[np.ndarray]) -> float:
        """Assess lighting quality and uniformity"""
        lighting_scores = []
        
        for frame in frames:
            # Convert to HSV for better lighting analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            v_channel = hsv[:, :, 2]  # Value channel
            
            # Calculate lighting metrics
            mean_brightness = np.mean(v_channel) / 255.0
            brightness_std = np.std(v_channel) / 255.0
            
            # Optimal lighting: moderate brightness with good contrast
            brightness_score = 1.0 - abs(mean_brightness - 0.6) / 0.6
            contrast_score = min(brightness_std * 2, 1.0)
            
            lighting_score = (brightness_score + contrast_score) / 2
            lighting_scores.append(lighting_score)
        
        return np.mean(lighting_scores)
```

#### 5. Face Data Caching
```python
class FaceCacheGenerator:
    def __init__(self):
        self.compression_enabled = True
        self.cache_version = "1.0"
        
    def generate_face_cache(self, avatar_id: str, 
                          detection_result: FaceDetectionResult,
                          frames: List[np.ndarray]) -> CachedFaceData:
        """Generate optimized face cache for runtime processing"""
        # Prepare face data for caching
        face_boxes = []
        face_landmarks = []
        cropped_faces = []
        face_masks = []
        
        for face_data in detection_result.face_data_list:
            face_boxes.append(face_data.bbox)
            face_landmarks.append(face_data.landmarks)
            cropped_faces.append(face_data.cropped_face)
            face_masks.append(face_data.face_mask)
        
        # Create cache object
        cache_data = CachedFaceData(
            avatar_id=avatar_id,
            face_boxes=np.array(face_boxes),
            face_landmarks=np.array(face_landmarks),
            cropped_faces=np.array(cropped_faces),
            face_masks=np.array(face_masks),
            processing_metadata=FaceProcessingMetadata(
                frame_count=len(frames),
                detection_confidence=detection_result.average_confidence,
                consistency_score=detection_result.consistency_score,
                cache_version=self.cache_version
            ),
            cache_timestamp=datetime.now(),
            compression_ratio=0.0,  # Will be set after compression
            integrity_hash=""       # Will be set after compression
        )
        
        # Compress cache data if enabled
        if self.compression_enabled:
            cache_data = self.compress_cache_data(cache_data)
        
        return cache_data
    
    def compress_cache_data(self, cache_data: CachedFaceData) -> CachedFaceData:
        """Compress face cache data for storage efficiency"""
        original_size = self.calculate_cache_size(cache_data)
        
        # Compress arrays using lz4
        compressed_data = {
            'face_boxes': lz4.compress(cache_data.face_boxes.tobytes()),
            'face_landmarks': lz4.compress(cache_data.face_landmarks.tobytes()),
            'cropped_faces': lz4.compress(cache_data.cropped_faces.tobytes()),
            'face_masks': lz4.compress(cache_data.face_masks.tobytes())
        }
        
        # Calculate compression ratio
        compressed_size = sum(len(data) for data in compressed_data.values())
        compression_ratio = compressed_size / original_size
        
        # Generate integrity hash
        integrity_hash = hashlib.sha256(
            b''.join(compressed_data.values())
        ).hexdigest()
        
        # Update cache data
        cache_data.compressed_data = compressed_data
        cache_data.compression_ratio = compression_ratio
        cache_data.integrity_hash = integrity_hash
        
        return cache_data
```

#### 6. Database Storage
```python
class AvatarDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize avatar registry database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS avatars (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    avatar_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    cache_path TEXT NOT NULL,
                    file_format TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    face_quality_score REAL NOT NULL,
                    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1,
                    metadata_json TEXT
                )
            ''')
            
            # Create indices for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_avatar_id ON avatars(avatar_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_owner_id ON avatars(owner_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_active ON avatars(is_active)')
    
    def store_avatar_registration(self, registration_data: AvatarRegistrationData) -> str:
        """Store avatar registration in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO avatars (
                    avatar_id, name, owner_id, file_path, cache_path,
                    file_format, file_size, face_quality_score, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                registration_data.avatar_id,
                registration_data.name,
                registration_data.owner_id,
                registration_data.file_path,
                registration_data.cache_path,
                registration_data.file_format,
                registration_data.file_size,
                registration_data.quality_score,
                json.dumps(registration_data.metadata)
            ))
            
            return registration_data.avatar_id
```

---

## Face Cache System

### Cache Structure and Storage

#### Optimized Cache Format
```python
@dataclass
class CachedFaceData:
    avatar_id: str
    face_boxes: np.ndarray          # Shape: (n_frames, 4)
    face_landmarks: np.ndarray      # Shape: (n_frames, 5, 2)
    cropped_faces: np.ndarray       # Shape: (n_frames, 96, 96, 3)
    face_masks: np.ndarray          # Shape: (n_frames, H, W)
    processing_metadata: FaceProcessingMetadata
    cache_timestamp: datetime
    compression_ratio: float
    integrity_hash: str
    compressed_data: Optional[Dict[str, bytes]] = None
```

#### Cache Access Patterns
```python
class FaceCacheManager:
    def __init__(self):
        self.memory_cache: Dict[str, CachedFaceData] = {}
        self.cache_metrics = CacheMetrics()
        self.max_memory_cache_size = 100
        
    async def get_face_cache(self, avatar_id: str) -> Optional[CachedFaceData]:
        """Retrieve face cache with memory and disk fallback"""
        start_time = time.time()
        
        # Check memory cache first (fastest access)
        if avatar_id in self.memory_cache:
            access_time = time.time() - start_time
            self.cache_metrics.record_hit('memory', access_time)
            await self.update_access_stats(avatar_id)
            return self.memory_cache[avatar_id]
        
        # Load from disk cache
        cache_data = await self.load_from_disk_cache(avatar_id)
        if cache_data:
            # Store in memory cache for future access
            self.store_in_memory_cache(avatar_id, cache_data)
            access_time = time.time() - start_time
            self.cache_metrics.record_hit('disk', access_time)
            await self.update_access_stats(avatar_id)
            return cache_data
        
        # Cache miss
        access_time = time.time() - start_time
        self.cache_metrics.record_miss(access_time)
        return None
    
    async def store_face_cache(self, avatar_id: str, cache_data: CachedFaceData) -> bool:
        """Store face cache with compression and validation"""
        try:
            # Store in memory cache
            self.store_in_memory_cache(avatar_id, cache_data)
            
            # Store on disk
            await self.store_to_disk_cache(avatar_id, cache_data)
            
            return True
        except Exception as e:
            logger.error(f"Failed to store face cache for {avatar_id}: {e}")
            return False
    
    def store_in_memory_cache(self, avatar_id: str, cache_data: CachedFaceData):
        """Store cache data in memory with LRU eviction"""
        if len(self.memory_cache) >= self.max_memory_cache_size:
            # Evict least recently used cache
            lru_avatar_id = min(
                self.memory_cache.keys(),
                key=lambda aid: self.get_last_access_time(aid)
            )
            del self.memory_cache[lru_avatar_id]
        
        self.memory_cache[avatar_id] = cache_data
```

### Cache Performance Optimization

#### Memory Management
```python
class CacheMemoryManager:
    def __init__(self):
        self.memory_limit = 2 * 1024 * 1024 * 1024  # 2GB
        self.current_usage = 0
        self.compression_threshold = 0.8  # Compress when 80% full
        
    def manage_memory_usage(self):
        """Optimize memory usage with intelligent caching"""
        if self.current_usage > self.memory_limit * self.compression_threshold:
            # Apply more aggressive compression
            self.compress_cached_data()
            
        if self.current_usage > self.memory_limit:
            # Evict least frequently used caches
            self.evict_lfu_caches()
    
    def compress_cached_data(self):
        """Apply compression to cached face data"""
        for avatar_id, cache_data in self.memory_cache.items():
            if not cache_data.compressed_data:
                compressed_cache = self.compress_face_cache(cache_data)
                self.memory_cache[avatar_id] = compressed_cache
```

---

## API Integration

### Registration Endpoints

#### Register Avatar
```python
@router.post("/avatar/register")
async def register_avatar(
    file: UploadFile = File(...),
    avatar_name: str = Form(...),
    user_id: str = Form(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None)
) -> AvatarRegistrationResponse:
    """Register new avatar with face detection and caching"""
    
    try:
        # Read file data
        file_data = await file.read()
        
        # Initialize registration service
        registrar = ColdAvatarRegistrar()
        
        # Process registration
        result = await registrar.register_avatar(
            file_data=file_data,
            filename=file.filename,
            avatar_name=avatar_name,
            user_id=user_id,
            description=description,
            tags=tags.split(',') if tags else []
        )
        
        return AvatarRegistrationResponse(
            avatar_id=result.avatar_id,
            registration_status="success",
            face_detection_results=result.face_detection_summary,
            quality_assessment=result.quality_assessment,
            processing_time=result.processing_time,
            cache_status="created",
            errors=[],
            warnings=result.warnings
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FaceDetectionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Avatar registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")
```

#### List Avatars
```python
@router.get("/avatar/list")
async def list_avatars(
    user_id: Optional[str] = None,
    include_inactive: bool = False
) -> List[AvatarInfo]:
    """List available avatars with metadata"""
    
    avatar_service = AvatarManagementService()
    
    avatars = await avatar_service.get_avatar_list(
        user_id=user_id,
        include_inactive=include_inactive
    )
    
    return avatars
```

#### Avatar Information
```python
@router.get("/avatar/{avatar_id}/info")
async def get_avatar_info(avatar_id: str) -> AvatarInfo:
    """Get detailed avatar information"""
    
    avatar_service = AvatarManagementService()
    avatar_info = await avatar_service.get_avatar_info(avatar_id)
    
    if not avatar_info:
        raise HTTPException(status_code=404, detail="Avatar not found")
    
    return avatar_info
```

#### Delete Avatar
```python
@router.delete("/avatar/{avatar_id}")
async def delete_avatar(
    avatar_id: str,
    user_id: str = Body(...),
    confirm: bool = Body(...)
) -> dict:
    """Delete avatar and associated data"""
    
    if not confirm:
        raise HTTPException(status_code=400, detail="Deletion not confirmed")
    
    avatar_service = AvatarManagementService()
    success = await avatar_service.delete_avatar(avatar_id, user_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Avatar not found or unauthorized")
    
    return {"success": True, "message": "Avatar deleted successfully"}
```

---

## Quality Assurance

### Quality Metrics and Thresholds

#### Face Quality Assessment
```python
class FaceQualityMetrics:
    def __init__(self):
        self.quality_criteria = {
            'face_detection_confidence': {
                'excellent': 0.95,
                'good': 0.8,
                'acceptable': 0.6,
                'poor': 0.4
            },
            'face_size_ratio': {
                'excellent': 0.3,  # Face occupies 30% of image
                'good': 0.2,
                'acceptable': 0.15,
                'poor': 0.1
            },
            'image_sharpness': {
                'excellent': 800,   # Laplacian variance
                'good': 400,
                'acceptable': 200,
                'poor': 100
            },
            'lighting_quality': {
                'excellent': 0.9,
                'good': 0.7,
                'acceptable': 0.5,
                'poor': 0.3
            }
        }
```

#### Quality Validation Pipeline
```python
class QualityValidator:
    def __init__(self):
        self.rejection_thresholds = {
            'min_face_confidence': 0.5,
            'min_face_size': 64,
            'max_blur_threshold': 50,
            'min_lighting_score': 0.3
        }
    
    def validate_avatar_quality(self, 
                               frames: List[np.ndarray],
                               detection_result: FaceDetectionResult) -> ValidationResult:
        """Comprehensive quality validation"""
        issues = []
        warnings = []
        
        # Check face detection confidence
        if detection_result.average_confidence < self.rejection_thresholds['min_face_confidence']:
            issues.append(f"Low face detection confidence: {detection_result.average_confidence:.2f}")
        
        # Check face size
        for face_data in detection_result.face_data_list:
            face_width = face_data.bbox[2] - face_data.bbox[0]
            face_height = face_data.bbox[3] - face_data.bbox[1]
            min_face_dimension = min(face_width, face_height)
            
            if min_face_dimension < self.rejection_thresholds['min_face_size']:
                issues.append(f"Face too small: {min_face_dimension}px")
        
        # Check image quality
        quality_scores = self.assess_image_quality(frames)
        if quality_scores['sharpness'] < self.rejection_thresholds['max_blur_threshold']:
            issues.append(f"Image too blurry: {quality_scores['sharpness']}")
        
        if quality_scores['lighting'] < self.rejection_thresholds['min_lighting_score']:
            warnings.append(f"Poor lighting quality: {quality_scores['lighting']:.2f}")
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            errors=issues,
            warnings=warnings,
            quality_scores=quality_scores
        )
```

---

## Performance Monitoring

### Registration Performance Metrics

#### Processing Time Tracking
```python
class RegistrationMetrics:
    def __init__(self):
        self.timing_data = {
            'file_validation': [],
            'frame_extraction': [],
            'face_detection': [],
            'quality_assessment': [],
            'cache_generation': [],
            'database_storage': [],
            'total_registration': []
        }
    
    def record_registration_timing(self, avatar_id: str, timings: dict):
        """Record registration processing times"""
        for stage, duration in timings.items():
            if stage in self.timing_data:
                self.timing_data[stage].append(duration)
        
        logger.info(f"Avatar {avatar_id} registration timings: {timings}")
    
    def get_performance_summary(self) -> dict:
        """Get registration performance summary"""
        summary = {}
        
        for stage, times in self.timing_data.items():
            if times:
                summary[stage] = {
                    'average': np.mean(times),
                    'p95': np.percentile(times, 95),
                    'p99': np.percentile(times, 99),
                    'count': len(times)
                }
        
        return summary
```

#### Cache Performance Analysis
```python
class CachePerformanceAnalyzer:
    def __init__(self):
        self.cache_access_times = []
        self.cache_hit_rates = {'memory': 0, 'disk': 0, 'miss': 0}
        
    def analyze_cache_performance(self) -> dict:
        """Analyze cache system performance"""
        total_requests = sum(self.cache_hit_rates.values())
        
        if total_requests == 0:
            return {'error': 'No cache access data'}
        
        return {
            'cache_hit_rate': (self.cache_hit_rates['memory'] + self.cache_hit_rates['disk']) / total_requests,
            'memory_hit_rate': self.cache_hit_rates['memory'] / total_requests,
            'disk_hit_rate': self.cache_hit_rates['disk'] / total_requests,
            'miss_rate': self.cache_hit_rates['miss'] / total_requests,
            'average_access_time': np.mean(self.cache_access_times) if self.cache_access_times else 0,
            'p95_access_time': np.percentile(self.cache_access_times, 95) if self.cache_access_times else 0
        }
```

---

## Error Handling and Recovery

### Registration Error Management

#### Error Classification
```python
class AvatarRegistrationError(Exception):
    """Base class for avatar registration errors"""
    pass

class FileValidationError(AvatarRegistrationError):
    """File validation failed"""
    pass

class FaceDetectionError(AvatarRegistrationError):
    """Face detection failed"""
    pass

class QualityAssessmentError(AvatarRegistrationError):
    """Quality assessment failed"""
    pass

class CacheGenerationError(AvatarRegistrationError):
    """Cache generation failed"""
    pass

class DatabaseError(AvatarRegistrationError):
    """Database operation failed"""
    pass
```

#### Error Recovery Strategies
```python
class RegistrationErrorHandler:
    def __init__(self):
        self.max_retry_attempts = 3
        self.fallback_strategies = {
            'face_detection': self.fallback_face_detection,
            'quality_assessment': self.fallback_quality_assessment,
            'cache_generation': self.fallback_cache_generation
        }
    
    async def handle_registration_error(self, 
                                      error: Exception,
                                      context: dict) -> Optional[dict]:
        """Handle registration errors with appropriate recovery"""
        error_type = type(error).__name__
        
        logger.error(f"Registration error: {error_type} - {error}")
        
        # Attempt error-specific recovery
        if error_type in self.fallback_strategies:
            try:
                recovery_result = await self.fallback_strategies[error_type](context)
                logger.info(f"Successfully recovered from {error_type}")
                return recovery_result
            except Exception as recovery_error:
                logger.error(f"Recovery failed for {error_type}: {recovery_error}")
        
        # Cleanup on unrecoverable error
        await self.cleanup_failed_registration(context.get('avatar_id'))
        return None
    
    async def fallback_face_detection(self, context: dict) -> dict:
        """Fallback face detection with relaxed parameters"""
        # Try with lower confidence threshold
        detector = FaceDetectionProcessor()
        detector.confidence_threshold = 0.3
        
        frames = context['frames']
        result = detector.detect_faces_in_frames(frames)
        
        return {'detection_result': result}
```

---

## Best Practices and Recommendations

### Avatar Registration Guidelines

#### For Users
1. **Image Quality**: Use high-resolution images (minimum 512x512)
2. **Lighting**: Ensure good, even lighting on the face
3. **Face Position**: Face should be centered and clearly visible
4. **Background**: Use simple, non-distracting backgrounds
5. **File Format**: Prefer JPG for photos, MP4 for videos
6. **File Size**: Keep files under 50MB for optimal processing

#### For Developers
1. **Error Handling**: Implement comprehensive error handling
2. **Progress Feedback**: Provide real-time registration progress
3. **Quality Validation**: Validate quality before processing
4. **Cache Optimization**: Monitor cache performance and optimize
5. **Resource Management**: Clean up failed registrations
6. **Security**: Validate and sanitize all file uploads

### Performance Optimization

#### Registration Speed
```python
# Optimize for speed
FAST_REGISTRATION_CONFIG = {
    'max_frames': 50,           # Reduce frames for videos
    'detection_size': (320, 320), # Smaller detection size
    'quality_checks': 'basic',   # Simplified quality assessment
    'compression_level': 'high'  # Aggressive compression
}

# Optimize for quality
QUALITY_REGISTRATION_CONFIG = {
    'max_frames': 100,          # More frames for better consistency
    'detection_size': (640, 640), # Higher detection resolution
    'quality_checks': 'comprehensive', # Full quality assessment
    'compression_level': 'balanced'     # Balanced compression
}
```

This documentation provides comprehensive coverage of the Avatar Registration System, including technical implementation details, API integration, quality assurance, and best practices for optimal avatar processing performance. 