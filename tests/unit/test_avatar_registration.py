#!/usr/bin/env python3
"""
Unit Tests for Avatar Registration System
Tests avatar registration, face detection caching, and validation workflows.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from PIL import Image
import io
import sqlite3
from datetime import datetime

# Import test fixtures
from tests.conftest import (
    sample_avatar_image, sample_avatar_video, mock_face_detector,
    test_database, avatar_config_fixture
)

class TestAvatarRegistration:
    """Test avatar registration functionality"""
    
    @pytest.fixture
    def avatar_registrar(self, mock_face_detector, test_database, avatar_config_fixture):
        """Create avatar registrar instance for testing"""
        from app.core.avatar_registrar import ColdAvatarRegistrar
        from app.core.face_cache_manager import FaceCacheManager
        from app.utils.avatar_validator import AvatarValidator
        
        cache_manager = Mock(spec=FaceCacheManager)
        validator = Mock(spec=AvatarValidator)
        
        registrar = ColdAvatarRegistrar(
            face_detector=mock_face_detector,
            avatar_database=test_database,
            face_cache=cache_manager,
            validation_suite=validator
        )
        
        return registrar
    
    def test_register_image_avatar_success(self, avatar_registrar, sample_avatar_image):
        """Test successful image avatar registration"""
        # Arrange
        avatar_id = "test_avatar_001"
        file_format = "jpg"
        
        # Mock face detection results
        mock_face_result = {
            'faces_detected': 1,
            'primary_face_confidence': 0.95,
            'bounding_boxes': [(50, 50, 200, 200)],
            'landmarks': [np.array([[100, 100], [150, 100], [125, 150]])]
        }
        
        avatar_registrar.face_detector.detect_faces.return_value = mock_face_result
        avatar_registrar.validation_suite.validate_avatar_file.return_value = Mock(is_valid=True)
        avatar_registrar.face_cache.store_face_cache.return_value = True
        
        # Act
        result = avatar_registrar.register_avatar(
            file_data=sample_avatar_image,
            avatar_id=avatar_id,
            file_format=file_format
        )
        
        # Assert
        assert result.registration_status == "success"
        assert result.avatar_id == avatar_id
        assert result.face_detection_results.faces_detected == 1
        assert result.quality_assessment.processing_ready is True
        
        # Verify face detector was called
        avatar_registrar.face_detector.detect_faces.assert_called_once()
        
        # Verify cache storage
        avatar_registrar.face_cache.store_face_cache.assert_called_once()
    
    def test_register_video_avatar_success(self, avatar_registrar, sample_avatar_video):
        """Test successful video avatar registration"""
        # Arrange
        avatar_id = "test_video_001"
        file_format = "mp4"
        
        # Mock video frame extraction and face detection
        mock_frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(5)]
        mock_face_results = [
            {
                'faces_detected': 1,
                'primary_face_confidence': 0.92,
                'bounding_boxes': [(45, 45, 195, 195)],
                'landmarks': [np.array([[95, 95], [145, 95], [120, 145]])]
            } for _ in range(5)
        ]
        
        with patch('app.utils.avatar_validator.AvatarValidator.extract_frames_from_video') as mock_extract:
            mock_extract.return_value = mock_frames
            avatar_registrar.face_detector.detect_faces.side_effect = mock_face_results
            avatar_registrar.validation_suite.validate_avatar_file.return_value = Mock(is_valid=True)
            avatar_registrar.face_cache.store_face_cache.return_value = True
            
            # Act
            result = avatar_registrar.register_avatar(
                file_data=sample_avatar_video,
                avatar_id=avatar_id,
                file_format=file_format
            )
            
            # Assert
            assert result.registration_status == "success"
            assert result.avatar_id == avatar_id
            assert result.face_detection_results.faces_detected == 1
            assert result.face_detection_results.face_consistency_score > 0.8
    
    def test_register_avatar_no_face_detected(self, avatar_registrar, sample_avatar_image):
        """Test avatar registration failure when no face is detected"""
        # Arrange
        avatar_id = "test_no_face"
        file_format = "jpg"
        
        # Mock no face detection
        mock_face_result = {
            'faces_detected': 0,
            'primary_face_confidence': 0.0,
            'bounding_boxes': [],
            'landmarks': []
        }
        
        avatar_registrar.face_detector.detect_faces.return_value = mock_face_result
        avatar_registrar.validation_suite.validate_avatar_file.return_value = Mock(is_valid=True)
        
        # Act
        result = avatar_registrar.register_avatar(
            file_data=sample_avatar_image,
            avatar_id=avatar_id,
            file_format=file_format
        )
        
        # Assert
        assert result.registration_status == "failed"
        assert "No face detected" in result.errors
        assert result.quality_assessment.processing_ready is False
    
    def test_register_avatar_invalid_file(self, avatar_registrar):
        """Test avatar registration failure with invalid file"""
        # Arrange
        avatar_id = "test_invalid"
        file_format = "jpg"
        invalid_data = b"invalid image data"
        
        # Mock validation failure
        avatar_registrar.validation_suite.validate_avatar_file.return_value = Mock(
            is_valid=False,
            errors=["Invalid image format", "File corrupted"]
        )
        
        # Act
        result = avatar_registrar.register_avatar(
            file_data=invalid_data,
            avatar_id=avatar_id,
            file_format=file_format
        )
        
        # Assert
        assert result.registration_status == "failed"
        assert "Invalid image format" in result.errors
        assert "File corrupted" in result.errors
    
    def test_register_avatar_low_quality_face(self, avatar_registrar, sample_avatar_image):
        """Test avatar registration with low quality face detection"""
        # Arrange
        avatar_id = "test_low_quality"
        file_format = "jpg"
        
        # Mock low quality face detection
        mock_face_result = {
            'faces_detected': 1,
            'primary_face_confidence': 0.3,  # Low confidence
            'bounding_boxes': [(50, 50, 200, 200)],
            'landmarks': [np.array([[100, 100], [150, 100], [125, 150]])]
        }
        
        avatar_registrar.face_detector.detect_faces.return_value = mock_face_result
        avatar_registrar.validation_suite.validate_avatar_file.return_value = Mock(is_valid=True)
        
        # Act
        result = avatar_registrar.register_avatar(
            file_data=sample_avatar_image,
            avatar_id=avatar_id,
            file_format=file_format
        )
        
        # Assert
        assert result.registration_status == "warning"
        assert "Low face detection confidence" in result.warnings
        assert result.quality_assessment.face_quality_score < 0.5
    
    def test_preprocess_face_data(self, avatar_registrar):
        """Test face data preprocessing functionality"""
        # Arrange
        avatar_id = "test_preprocess"
        mock_frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)]
        
        mock_face_results = [
            {
                'bounding_box': (50, 50, 200, 200),
                'landmarks': np.array([[100, 100], [150, 100], [125, 150]]),
                'confidence': 0.95
            } for _ in range(3)
        ]
        
        avatar_registrar.face_detector.detect_faces.side_effect = mock_face_results
        
        # Act
        result = avatar_registrar.preprocess_face_data(mock_frames, avatar_id)
        
        # Assert
        assert result.avatar_id == avatar_id
        assert len(result.face_boxes) == 3
        assert len(result.face_landmarks) == 3
        assert len(result.cropped_faces) == 3
        assert result.processing_metadata.total_frames == 3
        
        # Verify cropped faces are correct size (96x96 for wav2lip)
        for cropped_face in result.cropped_faces:
            assert cropped_face.shape == (96, 96, 3)
    
    def test_validate_avatar_quality(self, avatar_registrar):
        """Test avatar quality validation"""
        # Arrange
        # High quality frames
        good_frames = [np.random.randint(50, 205, (256, 256, 3), dtype=np.uint8) for _ in range(3)]
        
        # Mock good face detection results
        mock_good_results = [
            {
                'faces_detected': 1,
                'primary_face_confidence': 0.95,
                'clarity_score': 0.9,
                'lighting_quality': 0.85
            } for _ in range(3)
        ]
        
        avatar_registrar.face_detector.assess_quality.side_effect = mock_good_results
        
        # Act
        quality_result = avatar_registrar.validate_avatar_quality(good_frames)
        
        # Assert
        assert quality_result.overall_quality_score > 0.8
        assert quality_result.processing_ready is True
        assert quality_result.face_detectability > 0.9
        assert len(quality_result.recommendations) == 0
    
    def test_get_cached_face_data(self, avatar_registrar):
        """Test cached face data retrieval"""
        # Arrange
        avatar_id = "test_cached"
        mock_cached_data = Mock()
        mock_cached_data.avatar_id = avatar_id
        mock_cached_data.face_boxes = [(50, 50, 200, 200)]
        mock_cached_data.cropped_faces = [np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)]
        
        avatar_registrar.face_cache.retrieve_face_cache.return_value = mock_cached_data
        
        # Act
        result = avatar_registrar.get_cached_face_data(avatar_id)
        
        # Assert
        assert result is not None
        assert result.avatar_id == avatar_id
        assert len(result.face_boxes) == 1
        assert len(result.cropped_faces) == 1
        
        # Verify cache was accessed
        avatar_registrar.face_cache.retrieve_face_cache.assert_called_once_with(avatar_id)
    
    def test_registration_database_integration(self, avatar_registrar, test_database):
        """Test avatar registration database integration"""
        # Arrange
        avatar_id = "test_db_integration"
        avatar_name = "Test Avatar"
        file_path = "/test/path/avatar.jpg"
        
        # Act
        success = test_database.store_avatar_metadata(
            avatar_id=avatar_id,
            name=avatar_name,
            file_path=file_path,
            file_format="jpg",
            file_size=1024,
            face_quality_score=0.95
        )
        
        # Assert
        assert success is True
        
        # Verify data can be retrieved
        stored_data = test_database.get_avatar_info(avatar_id)
        assert stored_data is not None
        assert stored_data['avatar_id'] == avatar_id
        assert stored_data['name'] == avatar_name
        assert stored_data['face_quality_score'] == 0.95
    
    def test_concurrent_registration_handling(self, avatar_registrar, sample_avatar_image):
        """Test handling of concurrent avatar registrations"""
        # Arrange
        avatar_ids = [f"concurrent_test_{i}" for i in range(3)]
        
        # Mock successful registrations
        avatar_registrar.face_detector.detect_faces.return_value = {
            'faces_detected': 1,
            'primary_face_confidence': 0.95,
            'bounding_boxes': [(50, 50, 200, 200)],
            'landmarks': [np.array([[100, 100], [150, 100], [125, 150]])]
        }
        avatar_registrar.validation_suite.validate_avatar_file.return_value = Mock(is_valid=True)
        avatar_registrar.face_cache.store_face_cache.return_value = True
        
        # Act - Simulate concurrent registrations
        async def register_avatar(avatar_id):
            return avatar_registrar.register_avatar(
                file_data=sample_avatar_image,
                avatar_id=avatar_id,
                file_format="jpg"
            )
        
        # This would be run with asyncio in a real concurrent scenario
        results = []
        for avatar_id in avatar_ids:
            result = register_avatar(avatar_id)
            results.append(result)
        
        # Assert
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.registration_status == "success"
            assert result.avatar_id == avatar_ids[i]
    
    def test_registration_error_recovery(self, avatar_registrar, sample_avatar_image):
        """Test error recovery during avatar registration"""
        # Arrange
        avatar_id = "test_error_recovery"
        
        # Mock face detection failure followed by success
        avatar_registrar.face_detector.detect_faces.side_effect = [
            Exception("Face detection failed"),  # First attempt fails
            {  # Second attempt succeeds
                'faces_detected': 1,
                'primary_face_confidence': 0.95,
                'bounding_boxes': [(50, 50, 200, 200)],
                'landmarks': [np.array([[100, 100], [150, 100], [125, 150]])]
            }
        ]
        avatar_registrar.validation_suite.validate_avatar_file.return_value = Mock(is_valid=True)
        avatar_registrar.face_cache.store_face_cache.return_value = True
        
        # Act - First registration should fail
        result1 = avatar_registrar.register_avatar(
            file_data=sample_avatar_image,
            avatar_id=avatar_id,
            file_format="jpg"
        )
        
        # Reset for retry
        avatar_registrar.face_detector.detect_faces.side_effect = None
        avatar_registrar.face_detector.detect_faces.return_value = {
            'faces_detected': 1,
            'primary_face_confidence': 0.95,
            'bounding_boxes': [(50, 50, 200, 200)],
            'landmarks': [np.array([[100, 100], [150, 100], [125, 150]])]
        }
        
        # Second registration should succeed
        result2 = avatar_registrar.register_avatar(
            file_data=sample_avatar_image,
            avatar_id=avatar_id + "_retry",
            file_format="jpg"
        )
        
        # Assert
        assert result1.registration_status == "failed"
        assert "Face detection failed" in result1.errors
        assert result2.registration_status == "success"

class TestAvatarRegistrationPerformance:
    """Test avatar registration performance characteristics"""
    
    def test_registration_timing_requirements(self, avatar_registrar, sample_avatar_image):
        """Test that avatar registration meets timing requirements (< 5 seconds)"""
        import time
        
        # Arrange
        avatar_id = "test_timing"
        
        # Mock fast responses
        avatar_registrar.face_detector.detect_faces.return_value = {
            'faces_detected': 1,
            'primary_face_confidence': 0.95,
            'bounding_boxes': [(50, 50, 200, 200)],
            'landmarks': [np.array([[100, 100], [150, 100], [125, 150]])]
        }
        avatar_registrar.validation_suite.validate_avatar_file.return_value = Mock(is_valid=True)
        avatar_registrar.face_cache.store_face_cache.return_value = True
        
        # Act
        start_time = time.time()
        result = avatar_registrar.register_avatar(
            file_data=sample_avatar_image,
            avatar_id=avatar_id,
            file_format="jpg"
        )
        end_time = time.time()
        
        # Assert
        registration_time = end_time - start_time
        assert registration_time < 5.0  # Must complete within 5 seconds
        assert result.processing_time < 5.0
        assert result.registration_status == "success"
    
    def test_cache_storage_performance(self, avatar_registrar):
        """Test face cache storage performance"""
        import time
        
        # Arrange
        avatar_id = "test_cache_performance"
        mock_face_data = Mock()
        mock_face_data.avatar_id = avatar_id
        mock_face_data.face_boxes = [(50, 50, 200, 200)] * 10  # Multiple frames
        mock_face_data.cropped_faces = [np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)] * 10
        
        # Act
        start_time = time.time()
        success = avatar_registrar.face_cache.store_face_cache(avatar_id, mock_face_data)
        end_time = time.time()
        
        # Assert
        cache_time = end_time - start_time
        assert cache_time < 1.0  # Cache storage should be very fast
        assert success is True 

class TestAvatarRegistration:
    """Test avatar registration workflow and validation"""
    
    @pytest.fixture
    def mock_face_detector(self):
        """Mock face detection model"""
        detector = Mock()
        detector.get.return_value = [(
            np.array([0.1, 0.1, 0.9, 0.9]),  # bounding box
            0.95,  # confidence
            np.random.rand(106, 2)  # landmarks
        )]
        return detector
    
    @pytest.fixture
    def mock_face_cache_manager(self):
        """Mock face cache manager"""
        cache_manager = Mock(spec=FaceCacheManager)
        cache_manager.store_face_cache = AsyncMock(return_value=True)
        cache_manager.retrieve_face_cache = AsyncMock(return_value=None)
        cache_manager.warm_up_avatar_cache = AsyncMock()
        return cache_manager
    
    @pytest.fixture
    def avatar_registrar(self, mock_face_detector, mock_face_cache_manager):
        """Create avatar registrar with mocked dependencies"""
        registrar = ColdAvatarRegistrar(
            face_detector=mock_face_detector,
            face_cache_manager=mock_face_cache_manager
        )
        return registrar
    
    @pytest.fixture
    def sample_image_data(self):
        """Generate sample image data for testing"""
        # Create a simple 256x256x3 RGB image
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        return image.tobytes()
    
    @pytest.fixture
    def sample_avatar_metadata(self):
        """Sample avatar metadata for testing"""
        return {
            "avatar_name": "Test Avatar",
            "user_id": "test_user_123",
            "description": "Test avatar for unit testing",
            "tags": ["test", "avatar"]
        }
    
    @pytest.mark.asyncio
    async def test_image_avatar_registration(self, avatar_registrar, sample_image_data, sample_avatar_metadata):
        """Test single image avatar registration workflow"""
        # Arrange
        avatar_id = "test_avatar_001"
        file_format = "jpg"
        
        # Act
        result = await avatar_registrar.register_avatar(
            file_data=sample_image_data,
            avatar_id=avatar_id,
            file_format=file_format,
            metadata=sample_avatar_metadata
        )
        
        # Assert
        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["avatar_id"] == avatar_id
        assert "face_detection_results" in result
        assert "processing_time" in result
        assert result["processing_time"] < 5.0  # Target: <5 seconds
    
    @pytest.mark.asyncio
    async def test_video_avatar_registration(self, avatar_registrar):
        """Test video/GIF avatar registration"""
        # Arrange
        avatar_id = "test_video_001"
        video_data = b"fake_video_data"  # Mock video data
        
        with patch('app.core.avatar_registrar.cv2') as mock_cv2:
            # Mock video reading
            mock_cap = Mock()
            mock_cap.read.side_effect = [
                (True, np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)),
                (True, np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)),
                (False, None)  # End of video
            ]
            mock_cv2.VideoCapture.return_value = mock_cap
            
            # Act
            result = await avatar_registrar.register_avatar(
                file_data=video_data,
                avatar_id=avatar_id,
                file_format="mp4"
            )
            
            # Assert
            assert result["status"] == "success"
            assert "frame_count" in result
    
    @pytest.mark.asyncio
    async def test_face_detection_validation(self, avatar_registrar, mock_face_detector):
        """Test face detection quality validation"""
        # Arrange
        image_frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)]
        avatar_id = "test_face_detection"
        
        # Configure mock to return high confidence detection
        mock_face_detector.get.return_value = [(
            np.array([50, 50, 200, 200]),  # centered face
            0.98,  # high confidence
            np.random.rand(106, 2)  # landmarks
        )]
        
        # Act
        result = await avatar_registrar.preprocess_face_data(image_frames, avatar_id)
        
        # Assert
        assert result["face_detected"] == True
        assert result["face_confidence"] >= 0.9
        assert len(result["face_boxes"]) > 0
        assert len(result["cropped_faces"]) > 0
    
    @pytest.mark.asyncio
    async def test_avatar_cache_creation(self, avatar_registrar, mock_face_cache_manager):
        """Test face cache generation and storage"""
        # Arrange
        avatar_id = "test_cache_001"
        image_data = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8).tobytes()
        
        # Act
        result = await avatar_registrar.register_avatar(
            file_data=image_data,
            avatar_id=avatar_id,
            file_format="jpg"
        )
        
        # Assert
        # Verify cache manager was called
        mock_face_cache_manager.store_face_cache.assert_called_once()
        cache_call_args = mock_face_cache_manager.store_face_cache.call_args[0]
        assert cache_call_args[0] == avatar_id
        assert isinstance(cache_call_args[1], dict)  # CachedFaceData
    
    @pytest.mark.asyncio
    async def test_registration_error_handling(self, avatar_registrar):
        """Test error scenarios during registration"""
        # Test invalid file data
        with pytest.raises(ValueError):
            await avatar_registrar.register_avatar(
                file_data=b"invalid_image_data",
                avatar_id="test_error",
                file_format="jpg"
            )
        
        # Test unsupported format
        with pytest.raises(ValueError):
            await avatar_registrar.register_avatar(
                file_data=b"some_data",
                avatar_id="test_error",
                file_format="txt"
            )
    
    @pytest.mark.asyncio
    async def test_duplicate_avatar_handling(self, avatar_registrar):
        """Test handling of duplicate avatar registrations"""
        # Arrange
        avatar_id = "duplicate_test"
        image_data = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8).tobytes()
        
        # First registration
        result1 = await avatar_registrar.register_avatar(
            file_data=image_data,
            avatar_id=avatar_id,
            file_format="jpg"
        )
        
        # Second registration with same ID
        result2 = await avatar_registrar.register_avatar(
            file_data=image_data,
            avatar_id=avatar_id,
            file_format="jpg"
        )
        
        # Assert
        assert result1["status"] == "success"
        assert result2["status"] in ["updated", "duplicate"]
    
    @pytest.mark.asyncio
    async def test_avatar_metadata_storage(self, avatar_registrar):
        """Test database operations and metadata storage"""
        # Arrange
        avatar_id = "metadata_test"
        metadata = {
            "name": "Test Metadata Avatar",
            "user_id": "metadata_user",
            "description": "Testing metadata storage",
            "tags": ["metadata", "test"]
        }
        image_data = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8).tobytes()
        
        with patch('app.core.avatar_registrar.AvatarDatabase') as mock_db:
            mock_db_instance = Mock()
            mock_db.return_value = mock_db_instance
            mock_db_instance.store_avatar_metadata = AsyncMock(return_value=True)
            
            # Act
            result = await avatar_registrar.register_avatar(
                file_data=image_data,
                avatar_id=avatar_id,
                file_format="jpg",
                metadata=metadata
            )
            
            # Assert
            mock_db_instance.store_avatar_metadata.assert_called_once()
            stored_metadata = mock_db_instance.store_avatar_metadata.call_args[0]
            assert metadata["name"] in str(stored_metadata)

class TestAvatarQualityValidation:
    """Test avatar quality assessment and validation"""
    
    @pytest.fixture
    def quality_validator(self):
        """Create quality validator instance"""
        return AvatarValidator()
    
    def test_face_quality_assessment(self, quality_validator):
        """Test face quality scoring"""
        # High quality image (clear, well-lit)
        high_quality_image = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
        
        # Low quality image (dark, blurry)
        low_quality_image = np.random.randint(0, 50, (256, 256, 3), dtype=np.uint8)
        
        # Assess quality
        high_score = quality_validator.assess_image_quality(high_quality_image)
        low_score = quality_validator.assess_image_quality(low_quality_image)
        
        # Assert quality differences
        assert high_score > low_score
        assert high_score >= 0.7  # Good quality threshold
        assert low_score <= 0.5   # Poor quality threshold
    
    def test_face_detectability_validation(self, quality_validator):
        """Test face detectability requirements"""
        with patch('app.utils.avatar_validator.Face_detect_crop') as mock_detector:
            # Mock face detection results
            mock_detector.return_value.get.return_value = [(
                np.array([50, 50, 200, 200]),  # bounding box
                0.95,  # high confidence
                np.random.rand(106, 2)  # landmarks
            )]
            
            image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            validation_result = quality_validator.validate_face_detectability(image)
            
            assert validation_result["face_detected"] == True
            assert validation_result["confidence"] >= 0.9
    
    def test_resolution_validation(self, quality_validator):
        """Test image resolution requirements"""
        # Valid resolution
        valid_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        valid_result = quality_validator.validate_resolution(valid_image)
        assert valid_result["valid"] == True
        
        # Invalid resolution (too small)
        small_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        small_result = quality_validator.validate_resolution(small_image)
        assert small_result["valid"] == False

class TestAvatarCachePerformance:
    """Test avatar cache performance and efficiency"""
    
    @pytest.fixture
    def cache_manager(self):
        """Create face cache manager for testing"""
        return FaceCacheManager()
    
    @pytest.mark.asyncio
    async def test_cache_storage_speed(self, cache_manager):
        """Test cache storage performance"""
        # Arrange
        avatar_id = "speed_test"
        face_data = {
            "avatar_id": avatar_id,
            "face_boxes": [(50, 50, 200, 200)],
            "face_landmarks": [np.random.rand(106, 2)],
            "cropped_faces": [np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)],
            "cache_timestamp": datetime.now()
        }
        
        # Act
        start_time = asyncio.get_event_loop().time()
        success = await cache_manager.store_face_cache(avatar_id, face_data)
        storage_time = asyncio.get_event_loop().time() - start_time
        
        # Assert
        assert success == True
        assert storage_time < 0.1  # Target: <100ms storage time
    
    @pytest.mark.asyncio
    async def test_cache_retrieval_speed(self, cache_manager):
        """Test cache retrieval performance"""
        # Arrange
        avatar_id = "retrieval_test"
        face_data = {
            "avatar_id": avatar_id,
            "face_boxes": [(50, 50, 200, 200)],
            "face_landmarks": [np.random.rand(106, 2)],
            "cropped_faces": [np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)],
            "cache_timestamp": datetime.now()
        }
        
        # Store first
        await cache_manager.store_face_cache(avatar_id, face_data)
        
        # Act
        start_time = asyncio.get_event_loop().time()
        retrieved_data = await cache_manager.retrieve_face_cache(avatar_id)
        retrieval_time = asyncio.get_event_loop().time() - start_time
        
        # Assert
        assert retrieved_data is not None
        assert retrieval_time < 0.01  # Target: <10ms retrieval time
        assert retrieved_data["avatar_id"] == avatar_id
    
    @pytest.mark.asyncio
    async def test_cache_compression_efficiency(self, cache_manager):
        """Test cache data compression"""
        # Arrange
        large_face_data = {
            "avatar_id": "compression_test",
            "face_boxes": [(50, 50, 200, 200)] * 100,  # Many frames
            "face_landmarks": [np.random.rand(106, 2) for _ in range(100)],
            "cropped_faces": [np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8) for _ in range(100)],
            "cache_timestamp": datetime.now()
        }
        
        # Act
        compression_result = await cache_manager.compress_and_store(large_face_data)
        
        # Assert
        assert compression_result["compression_ratio"] > 0.5  # At least 50% compression
        assert compression_result["storage_success"] == True

class TestRegistrationPerformanceMetrics:
    """Test registration performance meets target requirements"""
    
    @pytest.mark.asyncio
    async def test_registration_time_target(self):
        """Test registration completes within target time"""
        # Arrange
        registrar = Mock(spec=ColdAvatarRegistrar)
        
        async def mock_register(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate processing
            return {"status": "success", "processing_time": 0.1}
        
        registrar.register_avatar = mock_register
        
        # Act
        start_time = asyncio.get_event_loop().time()
        result = await registrar.register_avatar(
            file_data=b"test_data",
            avatar_id="perf_test",
            file_format="jpg"
        )
        total_time = asyncio.get_event_loop().time() - start_time
        
        # Assert
        assert total_time < 5.0  # Target: <5 seconds
        assert result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_concurrent_registrations(self):
        """Test multiple concurrent avatar registrations"""
        # Arrange
        registrar = Mock(spec=ColdAvatarRegistrar)
        
        async def mock_register(file_data, avatar_id, file_format):
            await asyncio.sleep(0.1)  # Simulate processing
            return {"status": "success", "avatar_id": avatar_id}
        
        registrar.register_avatar = mock_register
        
        # Act
        tasks = []
        for i in range(3):
            task = registrar.register_avatar(
                file_data=f"test_data_{i}".encode(),
                avatar_id=f"concurrent_test_{i}",
                file_format="jpg"
            )
            tasks.append(task)
        
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks)
        total_time = asyncio.get_event_loop().time() - start_time
        
        # Assert
        assert len(results) == 3
        assert all(r["status"] == "success" for r in results)
        assert total_time < 2.0  # Should be faster than sequential

# Performance benchmark fixtures
@pytest.fixture(scope="session")
def performance_metrics():
    """Track performance metrics across tests"""
    return {
        "registration_times": [],
        "cache_operations": [],
        "face_detection_times": []
    }

def pytest_runtest_teardown(item, nextitem):
    """Collect performance metrics after each test"""
    if hasattr(item, "performance_data"):
        # Log performance data for analysis
        print(f"Test {item.name} performance: {item.performance_data}") 