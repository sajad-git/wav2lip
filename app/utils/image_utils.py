"""
Avatar Image Processing and Optimization
Handles image preprocessing, quality assessment, and format conversion for avatars
"""

import cv2
import numpy as np
import logging
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import imageio

from app.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class FaceQualityReport:
    """Face quality assessment report"""
    face_detected: bool
    face_confidence: float
    image_clarity: float
    lighting_quality: float
    processing_ready: bool
    recommendations: List[str]


class AvatarImageProcessor:
    """Avatar image processing and optimization"""
    
    def __init__(self):
        self.target_size = (256, 256)  # Standard size for wav2lip
        self.face_crop_size = (96, 96)  # Face crop size for wav2lip
        self.logger = logging.getLogger(__name__)
    
    def preprocess_avatar_for_wav2lip(self, image_path: str) -> np.ndarray:
        """
        Optimize avatar image for lip-sync processing
        
        Args:
            image_path: Path to avatar image file
            
        Returns:
            np.ndarray: Preprocessed image array ready for wav2lip
        """
        try:
            # Load and validate image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Normalize pixel values to 0-1 range
            image = image.astype(np.float32) / 255.0
            
            # Apply quality enhancement
            image = self._enhance_image_quality(image)
            
            self.logger.debug(f"Preprocessed avatar image: {image.shape}")
            return image
            
        except Exception as e:
            self.logger.error(f"Failed to preprocess avatar image: {str(e)}")
            raise
    
    def preprocess_avatar_from_bytes(self, image_data: bytes, format: str) -> np.ndarray:
        """
        Process avatar from raw bytes data
        
        Args:
            image_data: Image data bytes
            format: Image format (jpg, png, etc.)
            
        Returns:
            np.ndarray: Preprocessed image array
        """
        try:
            # Convert bytes to PIL Image
            image_pil = Image.open(BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image_pil.mode != 'RGB':
                image_pil = image_pil.convert('RGB')
            
            # Convert PIL to numpy array
            image = np.array(image_pil)
            
            # Resize to target size
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            # Apply quality enhancement
            image = self._enhance_image_quality(image)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Failed to process avatar from bytes: {str(e)}")
            raise
    
    def validate_face_quality(self, image: np.ndarray) -> FaceQualityReport:
        """
        Ensure avatar image meets quality standards
        
        Args:
            image: Image array
            
        Returns:
            FaceQualityReport: Face quality assessment
        """
        recommendations = []
        
        try:
            # Convert to uint8 for OpenCV operations
            if image.dtype == np.float32:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image
            
            # Check image clarity
            clarity_score = self._assess_image_clarity(image_uint8)
            if clarity_score < 0.5:
                recommendations.append("Image appears blurry, consider using a higher quality image")
            
            # Check lighting quality
            lighting_score = self._assess_lighting_quality(image_uint8)
            if lighting_score < 0.5:
                recommendations.append("Improve lighting conditions for better results")
            
            # Overall assessment
            processing_ready = clarity_score > 0.3 and lighting_score > 0.3
            
            return FaceQualityReport(
                face_detected=True,  # This will be updated by face detector
                face_confidence=0.0,  # This will be updated by face detector
                image_clarity=clarity_score,
                lighting_quality=lighting_score,
                processing_ready=processing_ready,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Failed to validate face quality: {str(e)}")
            return FaceQualityReport(
                face_detected=False,
                face_confidence=0.0,
                image_clarity=0.0,
                lighting_quality=0.0,
                processing_ready=False,
                recommendations=["Failed to analyze image quality"]
            )
    
    def enhance_avatar_image(self, image: np.ndarray, enhancement_level: str = "medium") -> np.ndarray:
        """
        Improve avatar image quality for better lip-sync
        
        Args:
            image: Raw image array
            enhancement_level: Enhancement level ("low", "medium", "high")
            
        Returns:
            np.ndarray: Enhanced image array
        """
        try:
            # Convert to PIL for enhancement
            if image.dtype == np.float32:
                image_pil = Image.fromarray((image * 255).astype(np.uint8))
            else:
                image_pil = Image.fromarray(image)
            
            # Enhancement settings based on level
            enhancement_settings = {
                "low": {"sharpness": 1.1, "contrast": 1.05, "brightness": 1.0},
                "medium": {"sharpness": 1.2, "contrast": 1.1, "brightness": 1.02},
                "high": {"sharpness": 1.3, "contrast": 1.15, "brightness": 1.05}
            }
            
            settings_dict = enhancement_settings.get(enhancement_level, enhancement_settings["medium"])
            
            # Apply enhancements
            if settings_dict["sharpness"] != 1.0:
                enhancer = ImageEnhance.Sharpness(image_pil)
                image_pil = enhancer.enhance(settings_dict["sharpness"])
            
            if settings_dict["contrast"] != 1.0:
                enhancer = ImageEnhance.Contrast(image_pil)
                image_pil = enhancer.enhance(settings_dict["contrast"])
            
            if settings_dict["brightness"] != 1.0:
                enhancer = ImageEnhance.Brightness(image_pil)
                image_pil = enhancer.enhance(settings_dict["brightness"])
            
            # Convert back to numpy array
            enhanced_image = np.array(image_pil)
            
            # Normalize if needed
            if image.dtype == np.float32:
                enhanced_image = enhanced_image.astype(np.float32) / 255.0
            
            return enhanced_image
            
        except Exception as e:
            self.logger.error(f"Failed to enhance avatar image: {str(e)}")
            return image
    
    def extract_frames_from_video(self, file_data: bytes, max_frames: int = 100) -> List[np.ndarray]:
        """
        Extract representative frames from video/GIF
        
        Args:
            file_data: Video file data
            max_frames: Maximum number of frames to extract
            
        Returns:
            List[np.ndarray]: List of extracted frames
        """
        try:
            # Save temporary file
            temp_path = "/tmp/temp_video.mp4"
            with open(temp_path, 'wb') as f:
                f.write(file_data)
            
            # Read video
            reader = imageio.get_reader(temp_path)
            
            # Calculate frame step
            total_frames = len(reader)
            frame_step = max(1, total_frames // max_frames)
            
            frames = []
            for i, frame in enumerate(reader):
                if i % frame_step == 0 and len(frames) < max_frames:
                    # Convert to RGB if needed
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frames.append(frame)
                    elif len(frame.shape) == 3 and frame.shape[2] == 4:
                        # Convert RGBA to RGB
                        frames.append(frame[:, :, :3])
            
            # Clean up
            reader.close()
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            self.logger.info(f"Extracted {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            self.logger.error(f"Failed to extract frames from video: {str(e)}")
            return []
    
    def create_avatar_thumbnail(self, image: np.ndarray, size: Tuple[int, int] = (128, 128)) -> bytes:
        """
        Create thumbnail for avatar preview
        
        Args:
            image: Source image array
            size: Thumbnail size
            
        Returns:
            bytes: Thumbnail image data
        """
        try:
            # Convert to PIL
            if image.dtype == np.float32:
                image_pil = Image.fromarray((image * 255).astype(np.uint8))
            else:
                image_pil = Image.fromarray(image)
            
            # Create thumbnail
            image_pil.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Convert to bytes
            buffer = BytesIO()
            image_pil.save(buffer, format='JPEG', quality=85)
            
            return buffer.getvalue()
            
        except Exception as e:
            self.logger.error(f"Failed to create thumbnail: {str(e)}")
            return b''
    
    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Apply quality enhancement to image"""
        try:
            # Convert to uint8 for processing
            if image.dtype == np.float32:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image
            
            # Apply bilateral filter for noise reduction while preserving edges
            filtered = cv2.bilateralFilter(image_uint8, 9, 75, 75)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(filtered, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Convert back to original dtype
            if image.dtype == np.float32:
                enhanced = enhanced.astype(np.float32) / 255.0
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"Failed to enhance image quality: {str(e)}")
            return image
    
    def _assess_image_clarity(self, image: np.ndarray) -> float:
        """Assess image clarity using Laplacian variance"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Calculate Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1 range (empirically determined thresholds)
            clarity_score = min(1.0, laplacian_var / 1000.0)
            
            return clarity_score
            
        except Exception as e:
            self.logger.error(f"Failed to assess image clarity: {str(e)}")
            return 0.0
    
    def _assess_lighting_quality(self, image: np.ndarray) -> float:
        """Assess lighting quality based on histogram analysis"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()
            
            # Check for good distribution (avoid too dark or too bright)
            # Good lighting should have distribution across the range
            mid_range_sum = hist[64:192].sum()  # Middle range
            
            # Check for contrast
            std_dev = np.std(gray)
            contrast_score = min(1.0, std_dev / 64.0)
            
            # Combine metrics
            lighting_score = (mid_range_sum + contrast_score) / 2.0
            
            return lighting_score
            
        except Exception as e:
            self.logger.error(f"Failed to assess lighting quality: {str(e)}")
            return 0.0
    
    def convert_to_base64(self, image: np.ndarray, format: str = "JPEG") -> str:
        """Convert image array to base64 string"""
        try:
            # Convert to PIL
            if image.dtype == np.float32:
                image_pil = Image.fromarray((image * 255).astype(np.uint8))
            else:
                image_pil = Image.fromarray(image)
            
            # Convert to bytes
            buffer = BytesIO()
            image_pil.save(buffer, format=format, quality=90)
            
            # Encode to base64
            base64_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/{format.lower()};base64,{base64_str}"
            
        except Exception as e:
            self.logger.error(f"Failed to convert image to base64: {str(e)}")
            return ""
    
    def validate_image_format(self, file_data: bytes, expected_format: str) -> bool:
        """Validate image file format"""
        try:
            # Try to open with PIL
            image = Image.open(BytesIO(file_data))
            actual_format = image.format.lower() if image.format else ""
            
            # Check if format matches expectation
            expected_format = expected_format.lower().replace(".", "")
            
            return actual_format == expected_format or actual_format in ["jpeg", "jpg"] and expected_format in ["jpeg", "jpg"]
            
        except Exception as e:
            self.logger.error(f"Failed to validate image format: {str(e)}")
            return False


# Global image processor instance
image_processor = AvatarImageProcessor() 