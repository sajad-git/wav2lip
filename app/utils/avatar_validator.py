"""
Avatar Validator for Avatar Streaming Service.
Validates avatar files and assesses quality for lip-sync processing.
"""

import os
import io
import cv2
import numpy as np
from PIL import Image, ImageSequence
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import hashlib
import magic
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Avatar file validation result."""
    
    is_valid: bool
    file_format: str
    file_size: int
    resolution: Tuple[int, int]
    frame_count: int
    errors: List[str]
    warnings: List[str]
    quality_score: float = 0.0
    processing_ready: bool = False
    
    
@dataclass
class QualityAssessment:
    """Avatar quality assessment result."""
    
    face_detected: bool
    face_count: int
    face_confidence: float
    image_clarity: float
    lighting_quality: float
    face_size_ratio: float
    orientation_score: float
    overall_quality: float
    processing_ready: bool
    recommendations: List[str]


class AvatarValidator:
    """Validates avatar files and assesses quality."""
    
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mov', '.avi', '.webm'}
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.min_resolution = (64, 64)
        self.max_resolution = (4096, 4096)
        self.target_resolution = (256, 256)
        self.max_frames = 1000  # For video/GIF files
        
        # Quality thresholds
        self.quality_thresholds = {
            'face_confidence': 0.5,
            'image_clarity': 0.6,
            'lighting_quality': 0.5,
            'face_size_ratio': 0.15,  # Minimum face size relative to image
            'orientation_score': 0.7,
            'overall_quality': 0.6
        }
        
        # Initialize face detector if available
        try:
            from insightface.app import FaceAnalysis
            self.face_detector = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.face_detector.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("Face detector initialized for avatar validation")
        except Exception as e:
            logger.warning(f"Face detector initialization failed: {e}")
            self.face_detector = None
            
    def validate_avatar_file(self, file_data: bytes, filename: str) -> ValidationResult:
        """Comprehensive avatar file validation."""
        
        errors = []
        warnings = []
        
        # Basic file validation
        file_size = len(file_data)
        if file_size == 0:
            errors.append("File is empty")
            return ValidationResult(
                is_valid=False,
                file_format="unknown",
                file_size=0,
                resolution=(0, 0),
                frame_count=0,
                errors=errors,
                warnings=warnings
            )
            
        if file_size > self.max_file_size:
            errors.append(f"File size {file_size / (1024*1024):.1f}MB exceeds maximum {self.max_file_size / (1024*1024)}MB")
            
        # File format validation
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension not in self.supported_formats:
            errors.append(f"Unsupported file format: {file_extension}")
            
        # MIME type validation
        try:
            mime_type = magic.from_buffer(file_data, mime=True)
            if not self._is_valid_mime_type(mime_type, file_extension):
                errors.append(f"MIME type {mime_type} doesn't match file extension {file_extension}")
        except Exception as e:
            warnings.append(f"Could not verify MIME type: {e}")
            
        # Early return if basic validation fails
        if errors:
            return ValidationResult(
                is_valid=False,
                file_format=file_extension,
                file_size=file_size,
                resolution=(0, 0),
                frame_count=0,
                errors=errors,
                warnings=warnings
            )
            
        # Content validation
        try:
            if file_extension in {'.jpg', '.jpeg', '.png'}:
                resolution, frame_count = self._validate_image(file_data)
            elif file_extension == '.gif':
                resolution, frame_count = self._validate_gif(file_data)
            elif file_extension in {'.mp4', '.mov', '.avi', '.webm'}:
                resolution, frame_count = self._validate_video(file_data)
            else:
                errors.append(f"Validation not implemented for format: {file_extension}")
                resolution, frame_count = (0, 0), 0
                
        except Exception as e:
            errors.append(f"Content validation failed: {e}")
            resolution, frame_count = (0, 0), 0
            
        # Resolution validation
        if resolution[0] < self.min_resolution[0] or resolution[1] < self.min_resolution[1]:
            errors.append(f"Resolution {resolution} is below minimum {self.min_resolution}")
        elif resolution[0] > self.max_resolution[0] or resolution[1] > self.max_resolution[1]:
            warnings.append(f"Resolution {resolution} is very high, consider resizing for better performance")
            
        # Frame count validation for videos/GIFs
        if frame_count > self.max_frames:
            warnings.append(f"Frame count {frame_count} is high, processing may be slow")
            
        # Calculate quality score
        quality_score = self._calculate_basic_quality_score(file_data, file_extension, resolution, frame_count)
        
        is_valid = len(errors) == 0
        processing_ready = is_valid and quality_score >= self.quality_thresholds['overall_quality']
        
        return ValidationResult(
            is_valid=is_valid,
            file_format=file_extension,
            file_size=file_size,
            resolution=resolution,
            frame_count=frame_count,
            errors=errors,
            warnings=warnings,
            quality_score=quality_score,
            processing_ready=processing_ready
        )
        
    def _is_valid_mime_type(self, mime_type: str, file_extension: str) -> bool:
        """Check if MIME type matches file extension."""
        
        valid_combinations = {
            '.jpg': ['image/jpeg'],
            '.jpeg': ['image/jpeg'],
            '.png': ['image/png'],
            '.gif': ['image/gif'],
            '.mp4': ['video/mp4'],
            '.mov': ['video/quicktime'],
            '.avi': ['video/x-msvideo', 'video/avi'],
            '.webm': ['video/webm']
        }
        
        return mime_type in valid_combinations.get(file_extension, [])
        
    def _validate_image(self, file_data: bytes) -> Tuple[Tuple[int, int], int]:
        """Validate image file."""
        
        try:
            image = Image.open(io.BytesIO(file_data))
            return image.size, 1
        except Exception as e:
            raise ValueError(f"Invalid image file: {e}")
            
    def _validate_gif(self, file_data: bytes) -> Tuple[Tuple[int, int], int]:
        """Validate GIF file."""
        
        try:
            image = Image.open(io.BytesIO(file_data))
            if not getattr(image, 'is_animated', False):
                return image.size, 1
                
            frame_count = 0
            for frame in ImageSequence.Iterator(image):
                frame_count += 1
                if frame_count > self.max_frames:
                    break
                    
            return image.size, frame_count
            
        except Exception as e:
            raise ValueError(f"Invalid GIF file: {e}")
            
    def _validate_video(self, file_data: bytes) -> Tuple[Tuple[int, int], int]:
        """Validate video file."""
        
        try:
            # Save to temporary file for OpenCV
            temp_path = f"/tmp/temp_video_{hashlib.md5(file_data).hexdigest()}.mp4"
            with open(temp_path, 'wb') as f:
                f.write(file_data)
                
            cap = cv2.VideoCapture(temp_path)
            
            if not cap.isOpened():
                raise ValueError("Could not open video file")
                
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            cap.release()
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass
                
            return (width, height), frame_count
            
        except Exception as e:
            raise ValueError(f"Invalid video file: {e}")
            
    def _calculate_basic_quality_score(
        self, 
        file_data: bytes, 
        file_format: str, 
        resolution: Tuple[int, int], 
        frame_count: int
    ) -> float:
        """Calculate basic quality score without face detection."""
        
        score = 0.0
        
        # Resolution score (0.3 weight)
        resolution_score = min(1.0, (resolution[0] * resolution[1]) / (256 * 256))
        score += resolution_score * 0.3
        
        # File size score (0.2 weight) - not too small, not too large
        file_size_mb = len(file_data) / (1024 * 1024)
        if file_size_mb < 0.1:
            size_score = 0.3  # Too small
        elif file_size_mb > 20:
            size_score = 0.7  # Too large
        else:
            size_score = 1.0  # Good size
        score += size_score * 0.2
        
        # Format score (0.2 weight)
        format_scores = {
            '.jpg': 0.9, '.jpeg': 0.9,
            '.png': 1.0,
            '.gif': 0.8,
            '.mp4': 0.9, '.mov': 0.8, '.avi': 0.7, '.webm': 0.8
        }
        score += format_scores.get(file_format, 0.5) * 0.2
        
        # Frame count score (0.3 weight)
        if file_format in {'.gif', '.mp4', '.mov', '.avi', '.webm'}:
            if frame_count == 0:
                frame_score = 0.0
            elif frame_count < 10:
                frame_score = 0.5  # Too few frames
            elif frame_count > 500:
                frame_score = 0.7  # Too many frames
            else:
                frame_score = 1.0  # Good frame count
        else:
            frame_score = 1.0  # Single image
        score += frame_score * 0.3
        
        return min(1.0, score)
        
    def assess_avatar_quality(self, frames: List[np.ndarray]) -> QualityAssessment:
        """Assess avatar quality using face detection and image analysis."""
        
        if not frames:
            return QualityAssessment(
                face_detected=False,
                face_count=0,
                face_confidence=0.0,
                image_clarity=0.0,
                lighting_quality=0.0,
                face_size_ratio=0.0,
                orientation_score=0.0,
                overall_quality=0.0,
                processing_ready=False,
                recommendations=["No frames provided for analysis"]
            )
            
        # Analyze first frame and a few samples
        sample_frames = frames[:min(5, len(frames))]
        
        face_results = []
        clarity_scores = []
        lighting_scores = []
        
        for frame in sample_frames:
            # Face detection
            face_result = self._detect_faces_in_frame(frame)
            face_results.append(face_result)
            
            # Image quality metrics
            clarity_scores.append(self._assess_image_clarity(frame))
            lighting_scores.append(self._assess_lighting_quality(frame))
            
        # Aggregate results
        face_detected = any(r['detected'] for r in face_results)
        face_count = max(r['count'] for r in face_results) if face_results else 0
        face_confidence = max(r['confidence'] for r in face_results) if face_results else 0.0
        face_size_ratio = max(r['size_ratio'] for r in face_results) if face_results else 0.0
        
        image_clarity = np.mean(clarity_scores) if clarity_scores else 0.0
        lighting_quality = np.mean(lighting_scores) if lighting_scores else 0.0
        
        # Orientation assessment (check if faces are properly oriented)
        orientation_score = self._assess_face_orientation(face_results)
        
        # Calculate overall quality
        overall_quality = self._calculate_overall_quality(
            face_detected, face_confidence, image_clarity, 
            lighting_quality, face_size_ratio, orientation_score
        )
        
        # Generate recommendations
        recommendations = self._generate_quality_recommendations(
            face_detected, face_confidence, image_clarity,
            lighting_quality, face_size_ratio, orientation_score
        )
        
        processing_ready = (
            face_detected and 
            face_confidence >= self.quality_thresholds['face_confidence'] and
            overall_quality >= self.quality_thresholds['overall_quality']
        )
        
        return QualityAssessment(
            face_detected=face_detected,
            face_count=face_count,
            face_confidence=face_confidence,
            image_clarity=image_clarity,
            lighting_quality=lighting_quality,
            face_size_ratio=face_size_ratio,
            orientation_score=orientation_score,
            overall_quality=overall_quality,
            processing_ready=processing_ready,
            recommendations=recommendations
        )
        
    def _detect_faces_in_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect faces in a single frame."""
        
        if self.face_detector is None:
            # Fallback: use OpenCV Haar cascade
            return self._detect_faces_opencv(frame)
            
        try:
            faces = self.face_detector.get(frame)
            
            if not faces:
                return {
                    'detected': False,
                    'count': 0,
                    'confidence': 0.0,
                    'size_ratio': 0.0
                }
                
            # Use the largest/most confident face
            best_face = max(faces, key=lambda f: f.det_score)
            
            # Calculate face size ratio
            bbox = best_face.bbox
            face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            frame_area = frame.shape[0] * frame.shape[1]
            size_ratio = face_area / frame_area
            
            return {
                'detected': True,
                'count': len(faces),
                'confidence': float(best_face.det_score),
                'size_ratio': size_ratio,
                'bbox': bbox
            }
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return {
                'detected': False,
                'count': 0,
                'confidence': 0.0,
                'size_ratio': 0.0
            }
            
    def _detect_faces_opencv(self, frame: np.ndarray) -> Dict[str, Any]:
        """Fallback face detection using OpenCV."""
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return {
                    'detected': False,
                    'count': 0,
                    'confidence': 0.0,
                    'size_ratio': 0.0
                }
                
            # Use the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            face_area = largest_face[2] * largest_face[3]
            frame_area = frame.shape[0] * frame.shape[1]
            size_ratio = face_area / frame_area
            
            return {
                'detected': True,
                'count': len(faces),
                'confidence': 0.7,  # OpenCV doesn't provide confidence
                'size_ratio': size_ratio,
                'bbox': largest_face
            }
            
        except Exception as e:
            logger.error(f"OpenCV face detection failed: {e}")
            return {
                'detected': False,
                'count': 0,
                'confidence': 0.0,
                'size_ratio': 0.0
            }
            
    def _assess_image_clarity(self, frame: np.ndarray) -> float:
        """Assess image clarity using Laplacian variance."""
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1 scale (empirically determined thresholds)
            clarity_score = min(1.0, laplacian_var / 1000.0)
            return clarity_score
            
        except Exception as e:
            logger.error(f"Clarity assessment failed: {e}")
            return 0.0
            
    def _assess_lighting_quality(self, frame: np.ndarray) -> float:
        """Assess lighting quality based on histogram distribution."""
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # Calculate entropy as measure of lighting distribution
            hist_norm = hist / hist.sum()
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
            
            # Normalize entropy to 0-1 scale
            lighting_score = min(1.0, entropy / 8.0)
            
            # Penalize extreme dark or bright images
            mean_brightness = gray.mean()
            if mean_brightness < 50 or mean_brightness > 200:
                lighting_score *= 0.7
                
            return lighting_score
            
        except Exception as e:
            logger.error(f"Lighting assessment failed: {e}")
            return 0.0
            
    def _assess_face_orientation(self, face_results: List[Dict[str, Any]]) -> float:
        """Assess face orientation consistency."""
        
        # This is a simplified assessment
        # In practice, you might use landmark analysis for more accurate orientation detection
        
        detected_faces = [r for r in face_results if r['detected']]
        if not detected_faces:
            return 0.0
            
        # For now, assume frontal orientation if face is detected with good confidence
        avg_confidence = np.mean([r['confidence'] for r in detected_faces])
        orientation_score = min(1.0, avg_confidence * 1.2)  # Boost confidence for orientation
        
        return orientation_score
        
    def _calculate_overall_quality(
        self,
        face_detected: bool,
        face_confidence: float,
        image_clarity: float,
        lighting_quality: float,
        face_size_ratio: float,
        orientation_score: float
    ) -> float:
        """Calculate overall avatar quality score."""
        
        if not face_detected:
            return 0.0
            
        # Weighted combination of quality metrics
        quality_score = (
            face_confidence * 0.3 +
            image_clarity * 0.25 +
            lighting_quality * 0.2 +
            min(1.0, face_size_ratio * 5) * 0.15 +  # Face should be at least 20% of image
            orientation_score * 0.1
        )
        
        return min(1.0, quality_score)
        
    def _generate_quality_recommendations(
        self,
        face_detected: bool,
        face_confidence: float,
        image_clarity: float,
        lighting_quality: float,
        face_size_ratio: float,
        orientation_score: float
    ) -> List[str]:
        """Generate quality improvement recommendations."""
        
        recommendations = []
        
        if not face_detected:
            recommendations.append("No face detected. Ensure the image clearly shows a face.")
            return recommendations
            
        if face_confidence < self.quality_thresholds['face_confidence']:
            recommendations.append("Face detection confidence is low. Use a clearer image with better face visibility.")
            
        if image_clarity < self.quality_thresholds['image_clarity']:
            recommendations.append("Image appears blurry. Use a sharper, higher quality image.")
            
        if lighting_quality < self.quality_thresholds['lighting_quality']:
            recommendations.append("Lighting quality is poor. Use better lighting or a different image.")
            
        if face_size_ratio < self.quality_thresholds['face_size_ratio']:
            recommendations.append("Face is too small in the image. Use an image where the face takes up more space.")
            
        if orientation_score < self.quality_thresholds['orientation_score']:
            recommendations.append("Face orientation may not be optimal. Use a front-facing image.")
            
        if not recommendations:
            recommendations.append("Avatar quality is good for processing.")
            
        return recommendations
        
    def extract_frames_from_video(self, file_data: bytes, max_frames: int = 100) -> List[np.ndarray]:
        """Extract representative frames from video/GIF."""
        
        # Save to temporary file
        temp_path = f"/tmp/temp_extract_{hashlib.md5(file_data).hexdigest()}.mp4"
        
        try:
            with open(temp_path, 'wb') as f:
                f.write(file_data)
                
            cap = cv2.VideoCapture(temp_path)
            
            if not cap.isOpened():
                raise ValueError("Could not open video file")
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = max(1, total_frames // max_frames)
            
            frames = []
            frame_idx = 0
            
            while len(frames) < max_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    break
                    
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                frame_idx += frame_interval
                
            cap.release()
            return frames
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return []
            
        finally:
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass 