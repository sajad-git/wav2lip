"""
Face Detection Utilities for Avatar Streaming Service.
Optimized face detection and processing utilities.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Face detection result for a single frame."""
    
    frame_index: int
    faces_detected: int
    primary_face_bbox: Optional[Tuple[int, int, int, int]]
    primary_face_confidence: float
    all_face_bboxes: List[Tuple[int, int, int, int]]
    all_face_confidences: List[float]
    landmarks: Optional[np.ndarray]
    processing_time: float
    quality_score: float


@dataclass
class DetectionSettings:
    """Face detection configuration settings."""
    
    confidence_threshold: float = 0.5
    batch_size: int = 4
    target_size: Tuple[int, int] = (640, 640)
    smoothing_window: int = 5
    max_faces: int = 1
    use_landmarks: bool = True
    optimize_for_speed: bool = False


class FaceDetectionUtils:
    """Utilities for face detection and optimization."""
    
    def __init__(self, settings: DetectionSettings = None):
        self.settings = settings or DetectionSettings()
        self.detection_cache: Dict[str, DetectionResult] = {}
        self.face_detector = None
        self.opencv_detector = None
        
        # Performance tracking
        self.detection_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        self._initialize_detectors()
        
    def _initialize_detectors(self):
        """Initialize face detection models."""
        
        # Initialize InsightFace detector
        try:
            from insightface.app import FaceAnalysis
            self.face_detector = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.face_detector.prepare(ctx_id=0, det_size=self.settings.target_size)
            logger.info("InsightFace detector initialized")
        except Exception as e:
            logger.warning(f"InsightFace initialization failed: {e}")
            
        # Initialize OpenCV detector as fallback
        try:
            self.opencv_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            logger.info("OpenCV face detector initialized as fallback")
        except Exception as e:
            logger.error(f"OpenCV detector initialization failed: {e}")
            
    def detect_faces_in_frames(
        self, 
        frames: List[np.ndarray], 
        settings: DetectionSettings = None
    ) -> List[DetectionResult]:
        """Batch face detection with optimization."""
        
        if not frames:
            return []
            
        detection_settings = settings or self.settings
        results = []
        
        # Process frames in batches for efficiency
        batch_size = detection_settings.batch_size
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_results = self._process_frame_batch(batch_frames, i, detection_settings)
            results.extend(batch_results)
            
        # Apply temporal smoothing if enabled
        if detection_settings.smoothing_window > 1:
            results = self._apply_temporal_smoothing(results, detection_settings.smoothing_window)
            
        return results
        
    def _process_frame_batch(
        self, 
        frames: List[np.ndarray], 
        start_index: int,
        settings: DetectionSettings
    ) -> List[DetectionResult]:
        """Process a batch of frames."""
        
        batch_results = []
        
        for i, frame in enumerate(frames):
            frame_index = start_index + i
            start_time = time.time()
            
            # Generate cache key
            cache_key = self._generate_cache_key(frame, settings)
            
            # Check cache first
            if cache_key in self.detection_cache:
                cached_result = self.detection_cache[cache_key]
                cached_result.frame_index = frame_index
                batch_results.append(cached_result)
                self.cache_hits += 1
                continue
                
            self.cache_misses += 1
            
            # Perform face detection
            detection_result = self._detect_faces_single_frame(frame, frame_index, settings)
            detection_result.processing_time = time.time() - start_time
            
            # Cache result
            self.detection_cache[cache_key] = detection_result
            batch_results.append(detection_result)
            
            # Track performance
            self.detection_times.append(detection_result.processing_time)
            if len(self.detection_times) > 100:  # Keep last 100 measurements
                self.detection_times.pop(0)
                
        return batch_results
        
    def _detect_faces_single_frame(
        self, 
        frame: np.ndarray, 
        frame_index: int,
        settings: DetectionSettings
    ) -> DetectionResult:
        """Detect faces in a single frame."""
        
        # Try InsightFace first
        if self.face_detector is not None:
            try:
                return self._detect_with_insightface(frame, frame_index, settings)
            except Exception as e:
                logger.warning(f"InsightFace detection failed: {e}")
                
        # Fallback to OpenCV
        if self.opencv_detector is not None:
            try:
                return self._detect_with_opencv(frame, frame_index, settings)
            except Exception as e:
                logger.error(f"OpenCV detection failed: {e}")
                
        # Return empty result if all detectors fail
        return DetectionResult(
            frame_index=frame_index,
            faces_detected=0,
            primary_face_bbox=None,
            primary_face_confidence=0.0,
            all_face_bboxes=[],
            all_face_confidences=[],
            landmarks=None,
            processing_time=0.0,
            quality_score=0.0
        )
        
    def _detect_with_insightface(
        self, 
        frame: np.ndarray, 
        frame_index: int,
        settings: DetectionSettings
    ) -> DetectionResult:
        """Face detection using InsightFace."""
        
        faces = self.face_detector.get(frame)
        
        if not faces:
            return DetectionResult(
                frame_index=frame_index,
                faces_detected=0,
                primary_face_bbox=None,
                primary_face_confidence=0.0,
                all_face_bboxes=[],
                all_face_confidences=[],
                landmarks=None,
                processing_time=0.0,
                quality_score=0.0
            )
            
        # Filter by confidence threshold
        valid_faces = [f for f in faces if f.det_score >= settings.confidence_threshold]
        
        if not valid_faces:
            return DetectionResult(
                frame_index=frame_index,
                faces_detected=0,
                primary_face_bbox=None,
                primary_face_confidence=0.0,
                all_face_bboxes=[],
                all_face_confidences=[],
                landmarks=None,
                processing_time=0.0,
                quality_score=0.0
            )
            
        # Sort by confidence and take top faces
        valid_faces.sort(key=lambda f: f.det_score, reverse=True)
        valid_faces = valid_faces[:settings.max_faces]
        
        # Primary face (highest confidence)
        primary_face = valid_faces[0]
        primary_bbox = tuple(map(int, primary_face.bbox))
        primary_confidence = float(primary_face.det_score)
        
        # All faces
        all_bboxes = [tuple(map(int, f.bbox)) for f in valid_faces]
        all_confidences = [float(f.det_score) for f in valid_faces]
        
        # Landmarks for primary face
        landmarks = primary_face.kps if settings.use_landmarks else None
        
        # Calculate quality score
        quality_score = self._calculate_detection_quality(
            frame, primary_bbox, primary_confidence, landmarks
        )
        
        return DetectionResult(
            frame_index=frame_index,
            faces_detected=len(valid_faces),
            primary_face_bbox=primary_bbox,
            primary_face_confidence=primary_confidence,
            all_face_bboxes=all_bboxes,
            all_face_confidences=all_confidences,
            landmarks=landmarks,
            processing_time=0.0,  # Will be set by caller
            quality_score=quality_score
        )
        
    def _detect_with_opencv(
        self, 
        frame: np.ndarray, 
        frame_index: int,
        settings: DetectionSettings
    ) -> DetectionResult:
        """Face detection using OpenCV Haar cascades."""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.opencv_detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=4,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return DetectionResult(
                frame_index=frame_index,
                faces_detected=0,
                primary_face_bbox=None,
                primary_face_confidence=0.0,
                all_face_bboxes=[],
                all_face_confidences=[],
                landmarks=None,
                processing_time=0.0,
                quality_score=0.0
            )
            
        # Convert to (x, y, x2, y2) format and sort by size
        converted_faces = []
        for (x, y, w, h) in faces:
            bbox = (x, y, x + w, y + h)
            area = w * h
            converted_faces.append((bbox, area))
            
        converted_faces.sort(key=lambda f: f[1], reverse=True)
        converted_faces = converted_faces[:settings.max_faces]
        
        # Primary face (largest)
        primary_bbox = converted_faces[0][0]
        primary_confidence = 0.8  # OpenCV doesn't provide confidence
        
        # All faces
        all_bboxes = [f[0] for f in converted_faces]
        all_confidences = [0.8] * len(converted_faces)  # Dummy confidence
        
        # Calculate quality score
        quality_score = self._calculate_detection_quality(
            frame, primary_bbox, primary_confidence, None
        )
        
        return DetectionResult(
            frame_index=frame_index,
            faces_detected=len(converted_faces),
            primary_face_bbox=primary_bbox,
            primary_face_confidence=primary_confidence,
            all_face_bboxes=all_bboxes,
            all_face_confidences=all_confidences,
            landmarks=None,  # OpenCV doesn't provide landmarks
            processing_time=0.0,  # Will be set by caller
            quality_score=quality_score
        )
        
    def _calculate_detection_quality(
        self, 
        frame: np.ndarray, 
        bbox: Tuple[int, int, int, int], 
        confidence: float,
        landmarks: Optional[np.ndarray]
    ) -> float:
        """Calculate detection quality score."""
        
        # Base score from confidence
        quality_score = confidence
        
        # Face size factor
        x1, y1, x2, y2 = bbox
        face_area = (x2 - x1) * (y2 - y1)
        frame_area = frame.shape[0] * frame.shape[1]
        size_ratio = face_area / frame_area
        
        # Optimal face size is 15-50% of frame
        if 0.15 <= size_ratio <= 0.5:
            size_factor = 1.0
        elif size_ratio < 0.15:
            size_factor = size_ratio / 0.15  # Penalize small faces
        else:
            size_factor = 0.5 / size_ratio  # Penalize large faces
            
        quality_score *= size_factor
        
        # Position factor (faces near center are better)
        face_center_x = (x1 + x2) / 2
        face_center_y = (y1 + y2) / 2
        frame_center_x = frame.shape[1] / 2
        frame_center_y = frame.shape[0] / 2
        
        distance_from_center = np.sqrt(
            ((face_center_x - frame_center_x) / frame.shape[1]) ** 2 +
            ((face_center_y - frame_center_y) / frame.shape[0]) ** 2
        )
        
        position_factor = max(0.5, 1.0 - distance_from_center)
        quality_score *= position_factor
        
        # Landmarks bonus (if available)
        if landmarks is not None:
            quality_score *= 1.1  # 10% bonus for landmark detection
            
        return min(1.0, quality_score)
        
    def _apply_temporal_smoothing(
        self, 
        results: List[DetectionResult], 
        window_size: int
    ) -> List[DetectionResult]:
        """Apply temporal smoothing to detection results."""
        
        if window_size <= 1 or len(results) < window_size:
            return results
            
        smoothed_results = []
        
        for i, result in enumerate(results):
            if result.faces_detected == 0:
                smoothed_results.append(result)
                continue
                
            # Get window of nearby results
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(results), i + window_size // 2 + 1)
            window_results = results[start_idx:end_idx]
            
            # Filter to results with faces
            valid_results = [r for r in window_results if r.faces_detected > 0]
            
            if not valid_results:
                smoothed_results.append(result)
                continue
                
            # Smooth bounding box coordinates
            if result.primary_face_bbox:
                smoothed_bbox = self._smooth_bounding_box(result.primary_face_bbox, valid_results)
                
                # Create smoothed result
                smoothed_result = DetectionResult(
                    frame_index=result.frame_index,
                    faces_detected=result.faces_detected,
                    primary_face_bbox=smoothed_bbox,
                    primary_face_confidence=result.primary_face_confidence,
                    all_face_bboxes=result.all_face_bboxes,
                    all_face_confidences=result.all_face_confidences,
                    landmarks=result.landmarks,
                    processing_time=result.processing_time,
                    quality_score=result.quality_score
                )
                
                smoothed_results.append(smoothed_result)
            else:
                smoothed_results.append(result)
                
        return smoothed_results
        
    def _smooth_bounding_box(
        self, 
        current_bbox: Tuple[int, int, int, int], 
        nearby_results: List[DetectionResult]
    ) -> Tuple[int, int, int, int]:
        """Smooth bounding box coordinates using nearby frames."""
        
        # Collect bounding boxes from nearby frames
        nearby_bboxes = []
        for result in nearby_results:
            if result.primary_face_bbox:
                nearby_bboxes.append(result.primary_face_bbox)
                
        if len(nearby_bboxes) <= 1:
            return current_bbox
            
        # Calculate weighted average (current frame has higher weight)
        weights = []
        for i, bbox in enumerate(nearby_bboxes):
            if bbox == current_bbox:
                weights.append(2.0)  # Higher weight for current frame
            else:
                weights.append(1.0)
                
        total_weight = sum(weights)
        
        # Weighted average of coordinates
        avg_x1 = sum(bbox[0] * w for bbox, w in zip(nearby_bboxes, weights)) / total_weight
        avg_y1 = sum(bbox[1] * w for bbox, w in zip(nearby_bboxes, weights)) / total_weight
        avg_x2 = sum(bbox[2] * w for bbox, w in zip(nearby_bboxes, weights)) / total_weight
        avg_y2 = sum(bbox[3] * w for bbox, w in zip(nearby_bboxes, weights)) / total_weight
        
        return (int(avg_x1), int(avg_y1), int(avg_x2), int(avg_y2))
        
    def _generate_cache_key(self, frame: np.ndarray, settings: DetectionSettings) -> str:
        """Generate cache key for frame and settings."""
        
        # Use frame hash and settings hash
        frame_hash = hash(frame.tobytes())
        settings_hash = hash(str(settings))
        
        return f"{frame_hash}_{settings_hash}"
        
    def optimize_face_boxes_sequence(
        self, 
        detections: List[DetectionResult]
    ) -> List[DetectionResult]:
        """Optimize face detection sequence for smooth processing."""
        
        if not detections:
            return detections
            
        optimized_detections = []
        
        for i, detection in enumerate(detections):
            if detection.faces_detected == 0:
                # Try to interpolate from nearby frames
                interpolated_detection = self._interpolate_missing_detection(detections, i)
                optimized_detections.append(interpolated_detection)
            else:
                optimized_detections.append(detection)
                
        return optimized_detections
        
    def _interpolate_missing_detection(
        self, 
        detections: List[DetectionResult], 
        missing_index: int
    ) -> DetectionResult:
        """Interpolate detection for frame with no face detected."""
        
        # Find nearest frames with detections
        before_detection = None
        after_detection = None
        
        # Search backwards
        for i in range(missing_index - 1, -1, -1):
            if detections[i].faces_detected > 0:
                before_detection = detections[i]
                break
                
        # Search forwards
        for i in range(missing_index + 1, len(detections)):
            if detections[i].faces_detected > 0:
                after_detection = detections[i]
                break
                
        original_detection = detections[missing_index]
        
        # If no nearby detections, return original
        if not before_detection and not after_detection:
            return original_detection
            
        # Use one nearby detection if only one is available
        if before_detection and not after_detection:
            nearby_detection = before_detection
        elif after_detection and not before_detection:
            nearby_detection = after_detection
        else:
            # Interpolate between before and after
            before_bbox = before_detection.primary_face_bbox
            after_bbox = after_detection.primary_face_bbox
            
            if before_bbox and after_bbox:
                # Linear interpolation
                alpha = 0.5  # Simple midpoint interpolation
                interpolated_bbox = (
                    int(before_bbox[0] * (1 - alpha) + after_bbox[0] * alpha),
                    int(before_bbox[1] * (1 - alpha) + after_bbox[1] * alpha),
                    int(before_bbox[2] * (1 - alpha) + after_bbox[2] * alpha),
                    int(before_bbox[3] * (1 - alpha) + after_bbox[3] * alpha)
                )
                
                # Create interpolated detection
                return DetectionResult(
                    frame_index=original_detection.frame_index,
                    faces_detected=1,
                    primary_face_bbox=interpolated_bbox,
                    primary_face_confidence=0.5,  # Lower confidence for interpolated
                    all_face_bboxes=[interpolated_bbox],
                    all_face_confidences=[0.5],
                    landmarks=None,
                    processing_time=0.0,
                    quality_score=0.5
                )
            else:
                nearby_detection = before_detection
                
        # Use nearby detection as fallback
        return DetectionResult(
            frame_index=original_detection.frame_index,
            faces_detected=nearby_detection.faces_detected,
            primary_face_bbox=nearby_detection.primary_face_bbox,
            primary_face_confidence=nearby_detection.primary_face_confidence * 0.8,  # Reduced confidence
            all_face_bboxes=nearby_detection.all_face_bboxes,
            all_face_confidences=[c * 0.8 for c in nearby_detection.all_face_confidences],
            landmarks=nearby_detection.landmarks,
            processing_time=0.0,
            quality_score=nearby_detection.quality_score * 0.8
        )
        
    def crop_face_regions(
        self, 
        frames: List[np.ndarray], 
        detections: List[DetectionResult], 
        target_size: Tuple[int, int] = (96, 96)
    ) -> List[np.ndarray]:
        """Extract and resize face regions for wav2lip processing."""
        
        if len(frames) != len(detections):
            raise ValueError("Number of frames must match number of detections")
            
        cropped_faces = []
        
        for frame, detection in zip(frames, detections):
            if detection.primary_face_bbox is None:
                # Create placeholder face if no detection
                placeholder = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                cropped_faces.append(placeholder)
                continue
                
            try:
                # Extract face region
                x1, y1, x2, y2 = detection.primary_face_bbox
                
                # Add padding to include more context
                padding = 0.2  # 20% padding
                width = x2 - x1
                height = y2 - y1
                
                pad_x = int(width * padding)
                pad_y = int(height * padding)
                
                x1_padded = max(0, x1 - pad_x)
                y1_padded = max(0, y1 - pad_y)
                x2_padded = min(frame.shape[1], x2 + pad_x)
                y2_padded = min(frame.shape[0], y2 + pad_y)
                
                # Crop face region
                face_region = frame[y1_padded:y2_padded, x1_padded:x2_padded]
                
                # Resize to target size
                if face_region.size > 0:
                    resized_face = cv2.resize(face_region, target_size)
                    cropped_faces.append(resized_face)
                else:
                    # Fallback placeholder
                    placeholder = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                    cropped_faces.append(placeholder)
                    
            except Exception as e:
                logger.warning(f"Face cropping failed for frame: {e}")
                placeholder = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                cropped_faces.append(placeholder)
                
        return cropped_faces
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for face detection."""
        
        avg_detection_time = np.mean(self.detection_times) if self.detection_times else 0.0
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "average_detection_time": avg_detection_time,
            "cache_hit_rate": cache_hit_rate,
            "total_cache_hits": self.cache_hits,
            "total_cache_misses": self.cache_misses,
            "total_requests": total_requests,
            "cache_size": len(self.detection_cache)
        }
        
    def clear_cache(self):
        """Clear detection cache."""
        self.detection_cache.clear()
        logger.info("Face detection cache cleared") 