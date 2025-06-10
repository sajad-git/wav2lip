"""
Cold Avatar Registration Engine
Handles avatar registration with face pre-processing and caching for instant processing.
"""

import os
import cv2
import numpy as np
import logging
import sqlite3
import pickle
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio

# Import wav2lip face detection components
import sys
sys.path.append('/app/assets/models')
from insightface_func.face_detect_crop_single import Face_detect_crop

logger = logging.getLogger(__name__)

@dataclass
class FaceDetectionSummary:
    """Summary of face detection results for avatar registration."""
    faces_detected: int
    primary_face_confidence: float
    face_consistency_score: float
    bounding_boxes: List[Tuple[int, int, int, int]]
    landmarks: List[List[float]]
    quality_metrics: Dict[str, float]

@dataclass
class FaceProcessingMetadata:
    """Metadata from face processing operations."""
    processing_time: float
    detection_model_used: str
    face_resolution: Tuple[int, int]
    preprocessing_steps: List[str]
    quality_score: float
    cache_version: str = "1.0"

@dataclass
class CachedFaceData:
    """Pre-processed face data for instant avatar processing."""
    avatar_id: str
    face_boxes: List[Tuple[int, int, int, int]]
    face_landmarks: List[np.ndarray]
    cropped_faces: List[np.ndarray]
    face_masks: List[np.ndarray]
    processing_metadata: FaceProcessingMetadata
    cache_version: str
    cache_timestamp: datetime
    compression_ratio: float
    integrity_hash: str

@dataclass
class QualityAssessment:
    """Avatar quality assessment results."""
    overall_score: float
    face_clarity_score: float
    lighting_quality_score: float
    orientation_score: float
    consistency_score: float
    processing_ready: bool
    recommendations: List[str]

@dataclass
class AvatarRegistrationResult:
    """Complete avatar registration result."""
    avatar_id: str
    registration_status: str
    face_detection_summary: FaceDetectionSummary
    quality_assessment: QualityAssessment
    processing_time: float
    cache_status: str
    file_info: Dict[str, Any]
    errors: List[str]
    warnings: List[str]

class ColdAvatarRegistrar:
    """Cold avatar registration with face pre-processing."""
    
    def __init__(self, avatar_config, face_detector: Optional[Face_detect_crop] = None):
        """
        Initialize avatar registrar.
        
        Args:
            avatar_config: Avatar configuration settings
            face_detector: Pre-loaded face detection model (optional)
        """
        self.config = avatar_config
        self.face_detector = face_detector or self._initialize_face_detector()
        self.avatar_database_path = os.path.join(avatar_config.cache_storage_path, "../avatars.db")
        self.cache_storage_path = avatar_config.cache_storage_path
        self.avatar_storage_path = avatar_config.avatar_storage_path
        
        # Ensure directories exist
        os.makedirs(self.cache_storage_path, exist_ok=True)
        os.makedirs(self.avatar_storage_path, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_face_detector(self) -> Face_detect_crop:
        """Initialize face detection model if not provided."""
        try:
            face_detector = Face_detect_crop(name='antelope', root='/app/assets/models/insightface_func/models')
            face_detector.prepare(ctx_id=0, det_thresh=self.config.face_detection_threshold, det_size=(640, 640))
            logger.info("Face detector initialized successfully")
            return face_detector
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            raise
    
    def _initialize_database(self):
        """Initialize SQLite database for avatar metadata."""
        try:
            with sqlite3.connect(self.avatar_database_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS avatars (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        avatar_id TEXT UNIQUE NOT NULL,
                        name TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        cache_path TEXT NOT NULL,
                        owner_id TEXT,
                        file_format TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        resolution_width INTEGER,
                        resolution_height INTEGER,
                        frame_count INTEGER DEFAULT 1,
                        face_quality_score REAL NOT NULL,
                        registration_date DATETIME NOT NULL,
                        last_accessed DATETIME,
                        access_count INTEGER DEFAULT 0,
                        is_active BOOLEAN DEFAULT 1,
                        metadata_json TEXT
                    )
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_avatar_id ON avatars (avatar_id);
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_owner_id ON avatars (owner_id);
                ''')
                
                conn.commit()
                logger.info("Avatar database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize avatar database: {e}")
            raise
    
    async def register_avatar(
        self,
        file_data: bytes,
        avatar_id: str,
        file_format: str,
        avatar_name: str = None,
        owner_id: str = None
    ) -> AvatarRegistrationResult:
        """
        Register new avatar with complete face pre-processing.
        
        Args:
            file_data: Avatar file data bytes
            avatar_id: Unique avatar identifier
            file_format: File format (jpg, png, gif, mp4, etc.)
            avatar_name: User-friendly name for avatar
            owner_id: Owner user identifier
            
        Returns:
            AvatarRegistrationResult: Complete registration results
        """
        start_time = datetime.now()
        errors = []
        warnings = []
        
        try:
            logger.info(f"Starting avatar registration for {avatar_id}")
            
            # Create avatar directory
            avatar_dir = os.path.join(self.avatar_storage_path, avatar_id)
            os.makedirs(avatar_dir, exist_ok=True)
            
            # Save original file
            file_extension = f".{file_format.lower()}"
            original_file_path = os.path.join(avatar_dir, f"original{file_extension}")
            
            with open(original_file_path, 'wb') as f:
                f.write(file_data)
            
            # Extract frames from file
            frames = self._extract_frames_from_file(original_file_path, file_format)
            if not frames:
                errors.append("Failed to extract frames from avatar file")
                return self._create_error_result(avatar_id, errors, start_time)
            
            # Validate avatar quality
            quality_assessment = await self._validate_avatar_quality(frames)
            if not quality_assessment.processing_ready:
                errors.append("Avatar quality insufficient for processing")
                warnings.extend(quality_assessment.recommendations)
            
            # Process face data
            face_processing_result = await self._preprocess_face_data(frames, avatar_id)
            
            if not face_processing_result:
                errors.append("Failed to process face data")
                return self._create_error_result(avatar_id, errors, start_time)
            
            # Create cached face data
            cached_face_data = self._create_cached_face_data(
                avatar_id, face_processing_result, frames
            )
            
            # Save face cache
            cache_success = self._save_face_cache(cached_face_data)
            if not cache_success:
                warnings.append("Failed to save face cache, processing will be slower")
            
            # Store avatar in database
            file_info = {
                'file_size': len(file_data),
                'resolution': (frames[0].shape[1], frames[0].shape[0]) if frames else (0, 0),
                'frame_count': len(frames),
                'file_format': file_format
            }
            
            db_success = self._store_avatar_in_database(
                avatar_id, avatar_name or avatar_id, original_file_path,
                self._get_cache_path(avatar_id), owner_id, file_info,
                quality_assessment.overall_score
            )
            
            if not db_success:
                errors.append("Failed to store avatar in database")
            
            # Create face detection summary
            face_detection_summary = FaceDetectionSummary(
                faces_detected=len(face_processing_result.get('face_boxes', [])),
                primary_face_confidence=face_processing_result.get('primary_confidence', 0.0),
                face_consistency_score=quality_assessment.consistency_score,
                bounding_boxes=face_processing_result.get('face_boxes', []),
                landmarks=[],  # Simplified for response
                quality_metrics={
                    'clarity': quality_assessment.face_clarity_score,
                    'lighting': quality_assessment.lighting_quality_score,
                    'orientation': quality_assessment.orientation_score
                }
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = AvatarRegistrationResult(
                avatar_id=avatar_id,
                registration_status="success" if not errors else "partial_success",
                face_detection_summary=face_detection_summary,
                quality_assessment=quality_assessment,
                processing_time=processing_time,
                cache_status="cached" if cache_success else "not_cached",
                file_info=file_info,
                errors=errors,
                warnings=warnings
            )
            
            logger.info(f"Avatar registration completed for {avatar_id} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Avatar registration failed for {avatar_id}: {e}")
            errors.append(f"Registration failed: {str(e)}")
            return self._create_error_result(avatar_id, errors, start_time)
    
    def _extract_frames_from_file(self, file_path: str, file_format: str) -> List[np.ndarray]:
        """Extract frames from image or video file."""
        frames = []
        
        try:
            if file_format.lower() in ['jpg', 'jpeg', 'png', 'bmp']:
                # Single image
                image = cv2.imread(file_path)
                if image is not None:
                    frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    
            elif file_format.lower() in ['gif', 'mp4', 'avi', 'mov']:
                # Video/GIF
                cap = cv2.VideoCapture(file_path)
                frame_count = 0
                max_frames = 100  # Limit frames for processing
                
                while cap.isOpened() and frame_count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(rgb_frame)
                    frame_count += 1
                
                cap.release()
            
            logger.info(f"Extracted {len(frames)} frames from {file_path}")
            return frames
            
        except Exception as e:
            logger.error(f"Failed to extract frames from {file_path}: {e}")
            return []
    
    async def _validate_avatar_quality(self, frames: List[np.ndarray]) -> QualityAssessment:
        """Validate avatar quality for processing suitability."""
        try:
            if not frames:
                return QualityAssessment(
                    overall_score=0.0,
                    face_clarity_score=0.0,
                    lighting_quality_score=0.0,
                    orientation_score=0.0,
                    consistency_score=0.0,
                    processing_ready=False,
                    recommendations=["No frames found in avatar file"]
                )
            
            # Analyze first frame for primary quality metrics
            first_frame = frames[0]
            
            # Face detection test
            try:
                bboxes, kpss = self.face_detector.detect(first_frame, max_num=1, metric='default')
                faces_detected = len(bboxes) if bboxes is not None else 0
            except:
                faces_detected = 0
            
            # Quality scoring
            clarity_score = self._assess_image_clarity(first_frame)
            lighting_score = self._assess_lighting_quality(first_frame)
            orientation_score = self._assess_face_orientation(first_frame)
            consistency_score = self._assess_frame_consistency(frames) if len(frames) > 1 else 1.0
            
            # Overall score calculation
            weights = [0.3, 0.25, 0.25, 0.2]  # clarity, lighting, orientation, consistency
            scores = [clarity_score, lighting_score, orientation_score, consistency_score]
            overall_score = sum(w * s for w, s in zip(weights, scores))
            
            # Processing readiness
            processing_ready = (
                faces_detected > 0 and
                clarity_score > 0.5 and
                lighting_score > 0.4 and
                overall_score > 0.5
            )
            
            # Recommendations
            recommendations = []
            if faces_detected == 0:
                recommendations.append("No face detected in avatar")
            if clarity_score < 0.5:
                recommendations.append("Image clarity could be improved")
            if lighting_score < 0.4:
                recommendations.append("Lighting quality could be better")
            if orientation_score < 0.6:
                recommendations.append("Face orientation not optimal")
            
            return QualityAssessment(
                overall_score=overall_score,
                face_clarity_score=clarity_score,
                lighting_quality_score=lighting_score,
                orientation_score=orientation_score,
                consistency_score=consistency_score,
                processing_ready=processing_ready,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return QualityAssessment(
                overall_score=0.0,
                face_clarity_score=0.0,
                lighting_quality_score=0.0,
                orientation_score=0.0,
                consistency_score=0.0,
                processing_ready=False,
                recommendations=[f"Quality assessment failed: {str(e)}"]
            )
    
    def _assess_image_clarity(self, image: np.ndarray) -> float:
        """Assess image clarity using Laplacian variance."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Normalize to 0-1 scale (empirically determined thresholds)
            return min(laplacian_var / 1000.0, 1.0)
        except:
            return 0.5
    
    def _assess_lighting_quality(self, image: np.ndarray) -> float:
        """Assess lighting quality based on brightness distribution."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            mean_brightness = np.mean(gray)
            brightness_std = np.std(gray)
            
            # Optimal range: 80-180 brightness, high variance
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128
            contrast_score = min(brightness_std / 50.0, 1.0)
            
            return (brightness_score + contrast_score) / 2
        except:
            return 0.5
    
    def _assess_face_orientation(self, image: np.ndarray) -> float:
        """Assess face orientation suitability."""
        try:
            # Simple heuristic based on face detection confidence
            bboxes, kpss = self.face_detector.detect(image, max_num=1, metric='default')
            if bboxes is not None and len(bboxes) > 0:
                # Higher confidence typically indicates better orientation
                confidence = bboxes[0][4] if len(bboxes[0]) > 4 else 0.5
                return confidence
            return 0.0
        except:
            return 0.5
    
    def _assess_frame_consistency(self, frames: List[np.ndarray]) -> float:
        """Assess consistency across video frames."""
        if len(frames) < 2:
            return 1.0
        
        try:
            # Sample frames for consistency check
            sample_frames = frames[::max(1, len(frames) // 5)][:5]
            consistencies = []
            
            for i in range(len(sample_frames) - 1):
                # Simple structural similarity
                gray1 = cv2.cvtColor(sample_frames[i], cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(sample_frames[i + 1], cv2.COLOR_RGB2GRAY)
                
                # Resize for comparison
                h, w = min(gray1.shape[0], gray2.shape[0]), min(gray1.shape[1], gray2.shape[1])
                gray1_resized = cv2.resize(gray1, (w, h))
                gray2_resized = cv2.resize(gray2, (w, h))
                
                # Simple correlation-based similarity
                correlation = cv2.matchTemplate(gray1_resized, gray2_resized, cv2.TM_CCOEFF_NORMED)[0][0]
                consistencies.append(max(0, correlation))
            
            return np.mean(consistencies) if consistencies else 0.5
            
        except Exception as e:
            logger.error(f"Frame consistency assessment failed: {e}")
            return 0.5
    
    async def _preprocess_face_data(self, frames: List[np.ndarray], avatar_id: str) -> Dict[str, Any]:
        """Preprocess face data for all frames."""
        try:
            face_boxes = []
            face_landmarks = []
            cropped_faces = []
            confidences = []
            
            for frame in frames:
                # Detect faces
                bboxes, kpss = self.face_detector.detect(frame, max_num=1, metric='default')
                
                if bboxes is not None and len(bboxes) > 0:
                    bbox = bboxes[0]
                    kps = kpss[0] if kpss is not None else None
                    
                    # Extract face region
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    confidence = bbox[4] if len(bbox) > 4 else 0.5
                    
                    # Crop and resize face
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        face_resized = cv2.resize(face_crop, (96, 96))
                        cropped_faces.append(face_resized)
                        face_boxes.append((x1, y1, x2, y2))
                        face_landmarks.append(kps if kps is not None else np.zeros((5, 2)))
                        confidences.append(confidence)
            
            return {
                'face_boxes': face_boxes,
                'face_landmarks': face_landmarks,
                'cropped_faces': cropped_faces,
                'primary_confidence': max(confidences) if confidences else 0.0,
                'processing_metadata': {
                    'total_frames': len(frames),
                    'faces_detected': len(face_boxes),
                    'avg_confidence': np.mean(confidences) if confidences else 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Face preprocessing failed for {avatar_id}: {e}")
            return {}
    
    def _create_cached_face_data(
        self,
        avatar_id: str,
        face_processing_result: Dict[str, Any],
        frames: List[np.ndarray]
    ) -> CachedFaceData:
        """Create cached face data structure."""
        
        processing_metadata = FaceProcessingMetadata(
            processing_time=face_processing_result.get('processing_time', 0.0),
            detection_model_used="antelope",
            face_resolution=(96, 96),
            preprocessing_steps=["detection", "cropping", "resizing"],
            quality_score=face_processing_result.get('primary_confidence', 0.0),
            cache_version="1.0"
        )
        
        # Create face masks (simplified for now)
        face_masks = []
        for cropped_face in face_processing_result.get('cropped_faces', []):
            mask = np.ones((96, 96), dtype=np.uint8) * 255
            face_masks.append(mask)
        
        cached_data = CachedFaceData(
            avatar_id=avatar_id,
            face_boxes=face_processing_result.get('face_boxes', []),
            face_landmarks=face_processing_result.get('face_landmarks', []),
            cropped_faces=face_processing_result.get('cropped_faces', []),
            face_masks=face_masks,
            processing_metadata=processing_metadata,
            cache_version="1.0",
            cache_timestamp=datetime.now(),
            compression_ratio=0.8,  # Placeholder
            integrity_hash=self._calculate_data_hash(face_processing_result)
        )
        
        return cached_data
    
    def _save_face_cache(self, cached_face_data: CachedFaceData) -> bool:
        """Save face cache data to disk."""
        try:
            cache_file_path = self._get_cache_path(cached_face_data.avatar_id)
            
            # Serialize cached data
            cache_data = {
                'avatar_id': cached_face_data.avatar_id,
                'face_boxes': cached_face_data.face_boxes,
                'face_landmarks': [landmarks.tolist() for landmarks in cached_face_data.face_landmarks],
                'cropped_faces': [face.tolist() for face in cached_face_data.cropped_faces],
                'face_masks': [mask.tolist() for mask in cached_face_data.face_masks],
                'processing_metadata': asdict(cached_face_data.processing_metadata),
                'cache_version': cached_face_data.cache_version,
                'cache_timestamp': cached_face_data.cache_timestamp.isoformat(),
                'compression_ratio': cached_face_data.compression_ratio,
                'integrity_hash': cached_face_data.integrity_hash
            }
            
            # Save as compressed pickle
            with open(cache_file_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"Face cache saved successfully for {cached_face_data.avatar_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save face cache for {cached_face_data.avatar_id}: {e}")
            return False
    
    def _get_cache_path(self, avatar_id: str) -> str:
        """Get cache file path for avatar."""
        return os.path.join(self.cache_storage_path, f"{avatar_id}_face_cache.pkl")
    
    def _calculate_data_hash(self, data: Dict[str, Any]) -> str:
        """Calculate hash for data integrity verification."""
        try:
            data_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.md5(data_str.encode()).hexdigest()
        except:
            return "unknown"
    
    def _store_avatar_in_database(
        self,
        avatar_id: str,
        name: str,
        file_path: str,
        cache_path: str,
        owner_id: str,
        file_info: Dict[str, Any],
        quality_score: float
    ) -> bool:
        """Store avatar metadata in database."""
        try:
            with sqlite3.connect(self.avatar_database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO avatars
                    (avatar_id, name, file_path, cache_path, owner_id, file_format,
                     file_size, resolution_width, resolution_height, frame_count,
                     face_quality_score, registration_date, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    avatar_id, name, file_path, cache_path, owner_id,
                    file_info['file_format'], file_info['file_size'],
                    file_info['resolution'][0], file_info['resolution'][1],
                    file_info['frame_count'], quality_score,
                    datetime.now().isoformat(), json.dumps(file_info)
                ))
                
                conn.commit()
                logger.info(f"Avatar {avatar_id} stored in database successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store avatar {avatar_id} in database: {e}")
            return False
    
    def _create_error_result(
        self,
        avatar_id: str,
        errors: List[str],
        start_time: datetime
    ) -> AvatarRegistrationResult:
        """Create error result for failed registration."""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AvatarRegistrationResult(
            avatar_id=avatar_id,
            registration_status="failed",
            face_detection_summary=FaceDetectionSummary(
                faces_detected=0,
                primary_face_confidence=0.0,
                face_consistency_score=0.0,
                bounding_boxes=[],
                landmarks=[],
                quality_metrics={}
            ),
            quality_assessment=QualityAssessment(
                overall_score=0.0,
                face_clarity_score=0.0,
                lighting_quality_score=0.0,
                orientation_score=0.0,
                consistency_score=0.0,
                processing_ready=False,
                recommendations=[]
            ),
            processing_time=processing_time,
            cache_status="not_cached",
            file_info={},
            errors=errors,
            warnings=[]
        )

    def get_cached_face_data(self, avatar_id: str) -> Optional[CachedFaceData]:
        """
        Retrieve cached face data for avatar.
        
        Args:
            avatar_id: Avatar identifier
            
        Returns:
            CachedFaceData or None if not found
        """
        try:
            cache_file_path = self._get_cache_path(avatar_id)
            
            if not os.path.exists(cache_file_path):
                return None
            
            with open(cache_file_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Reconstruct numpy arrays
            cached_face_data = CachedFaceData(
                avatar_id=cache_data['avatar_id'],
                face_boxes=cache_data['face_boxes'],
                face_landmarks=[np.array(landmarks) for landmarks in cache_data['face_landmarks']],
                cropped_faces=[np.array(face) for face in cache_data['cropped_faces']],
                face_masks=[np.array(mask) for mask in cache_data['face_masks']],
                processing_metadata=FaceProcessingMetadata(**cache_data['processing_metadata']),
                cache_version=cache_data['cache_version'],
                cache_timestamp=datetime.fromisoformat(cache_data['cache_timestamp']),
                compression_ratio=cache_data['compression_ratio'],
                integrity_hash=cache_data['integrity_hash']
            )
            
            logger.info(f"Cached face data retrieved for {avatar_id}")
            return cached_face_data
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached face data for {avatar_id}: {e}")
            return None 