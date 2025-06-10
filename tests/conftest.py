"""
Pytest configuration for Avatar Streaming Service tests
Provides fixtures and test setup for model loading, avatar registration, and system testing
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
import numpy as np
from unittest.mock import Mock, MagicMock

# Test imports
from app.config.settings import Settings
from app.core.model_loader import ColdModelLoader
from app.core.avatar_registrar import ColdAvatarRegistrar  
from app.core.face_cache_manager import FaceCacheManager
from app.services.avatar_service import AvatarManagementService
from app.utils.image_utils import AvatarImageProcessor


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Test settings configuration"""
    # Create temporary directories for testing
    temp_dir = tempfile.mkdtemp(prefix="avatar_test_")
    
    settings = Settings()
    settings.avatar_storage_path = os.path.join(temp_dir, "avatars")
    settings.cache_storage_path = os.path.join(temp_dir, "cache")
    settings.model_storage_path = os.path.join(temp_dir, "models")
    settings.log_storage_path = os.path.join(temp_dir, "logs")
    settings.temp_storage_path = os.path.join(temp_dir, "temp")
    settings.database_url = f"sqlite:///{temp_dir}/test_avatars.db"
    
    # Create directories
    for path in [settings.avatar_storage_path, settings.cache_storage_path, 
                 settings.model_storage_path, settings.log_storage_path,
                 settings.temp_storage_path]:
        os.makedirs(path, exist_ok=True)
    
    return settings


@pytest.fixture
def temp_directory() -> Generator[str, None, None]:
    """Create temporary directory for test files"""
    temp_dir = tempfile.mkdtemp(prefix="avatar_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_image() -> np.ndarray:
    """Generate sample image for testing"""
    # Create a simple 256x256 RGB image
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Add a face-like pattern in the center
    center_x, center_y = 128, 128
    cv2.circle(image, (center_x, center_y - 20), 30, (255, 200, 180), -1)  # Face
    cv2.circle(image, (center_x - 15, center_y - 30), 5, (50, 50, 50), -1)  # Left eye
    cv2.circle(image, (center_x + 15, center_y - 30), 5, (50, 50, 50), -1)  # Right eye
    cv2.ellipse(image, (center_x, center_y - 10), (15, 8), 0, 0, 180, (200, 100, 100), 2)  # Mouth
    
    return image


@pytest.fixture
def sample_image_bytes(sample_image: np.ndarray) -> bytes:
    """Convert sample image to bytes"""
    import cv2
    from io import BytesIO
    from PIL import Image
    
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(sample_image)
    
    # Convert to bytes
    buffer = BytesIO()
    pil_image.save(buffer, format='JPEG', quality=90)
    
    return buffer.getvalue()


@pytest.fixture
def mock_face_detector() -> Mock:
    """Mock face detector for testing"""
    mock_detector = Mock()
    
    # Mock face detection results
    mock_detector.get_face_landmarks.return_value = [
        np.array([[100, 100], [150, 100], [125, 120], [110, 140], [140, 140]])  # Sample landmarks
    ]
    
    mock_detector.get_bbox.return_value = [(80, 80, 120, 120)]  # Sample bounding box
    
    return mock_detector


@pytest.fixture
def mock_model_loader() -> Mock:
    """Mock model loader for testing"""
    mock_loader = Mock(spec=ColdModelLoader)
    
    # Mock model instances
    mock_loader.loaded_models = {
        "wav2lip": MagicMock(),
        "wav2lip_gan": MagicMock(),
        "face_detector": MagicMock()
    }
    
    mock_loader.models_loaded = True
    mock_loader.get_model_instance.return_value = MagicMock()
    
    return mock_loader


@pytest.fixture
async def face_cache_manager(test_settings: Settings) -> FaceCacheManager:
    """Initialize face cache manager for testing"""
    cache_manager = FaceCacheManager()
    await cache_manager.initialize()
    return cache_manager


@pytest.fixture 
async def avatar_registrar(test_settings: Settings, mock_face_detector: Mock, 
                          face_cache_manager: FaceCacheManager) -> ColdAvatarRegistrar:
    """Initialize avatar registrar for testing"""
    from app.config.avatar_config import AvatarConfig
    
    avatar_config = AvatarConfig()
    avatar_config.avatar_storage_path = test_settings.avatar_storage_path
    avatar_config.cache_storage_path = test_settings.cache_storage_path
    
    registrar = ColdAvatarRegistrar(
        avatar_config=avatar_config,
        face_detector=mock_face_detector
    )
    
    return registrar


@pytest.fixture
def image_processor() -> AvatarImageProcessor:
    """Initialize image processor for testing"""
    return AvatarImageProcessor()


@pytest.fixture
async def avatar_service(avatar_registrar: ColdAvatarRegistrar, 
                        face_cache_manager: FaceCacheManager) -> AvatarManagementService:
    """Initialize avatar management service for testing"""
    service = AvatarManagementService(
        avatar_registrar=avatar_registrar,
        face_cache_manager=face_cache_manager
    )
    
    return service


@pytest.fixture
def performance_thresholds() -> Dict[str, float]:
    """Performance thresholds for testing"""
    return {
        "model_loading_time": 15.0,  # seconds
        "avatar_registration_time": 10.0,  # seconds
        "face_cache_access_time": 0.05,  # seconds (50ms)
        "chunk_processing_time": 1.0,  # seconds
        "first_chunk_latency": 2.0,  # seconds
        "memory_usage_mb": 1000,  # MB
    }


@pytest.fixture
def test_avatar_data() -> Dict[str, Any]:
    """Test avatar metadata"""
    return {
        "avatar_id": "test_avatar_001",
        "name": "Test Avatar",
        "owner_id": "test_user",
        "description": "Test avatar for unit testing",
        "file_format": "jpg"
    }


@pytest.fixture
def benchmark_data() -> Dict[str, Any]:
    """Benchmark data for performance testing"""
    return {
        "test_iterations": 10,
        "concurrent_users": 3,
        "chunk_count": 5,
        "target_fps": 25,
        "memory_threshold_mb": 2000
    }


class MockONNXSession:
    """Mock ONNX session for testing"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.inputs = [Mock(name="input", shape=[1, 3, 96, 96])]
        self.outputs = [Mock(name="output", shape=[1, 3, 96, 96])]
    
    def run(self, output_names, input_feed):
        """Mock inference run"""
        # Return dummy output with correct shape
        if self.model_name == "wav2lip":
            return [np.random.random((1, 3, 96, 96)).astype(np.float32)]
        return [np.random.random((1, 3, 96, 96)).astype(np.float32)]
    
    def get_inputs(self):
        return self.inputs
    
    def get_outputs(self):
        return self.outputs


@pytest.fixture
def mock_onnx_session() -> MockONNXSession:
    """Mock ONNX session for testing"""
    return MockONNXSession("wav2lip")


@pytest.fixture
def test_audio_chunk() -> bytes:
    """Generate test audio chunk"""
    # Generate simple sine wave audio
    import numpy as np
    
    sample_rate = 16000
    duration = 2.0  # seconds
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    
    return audio_int16.tobytes()


@pytest.fixture
def gpu_test_enabled() -> bool:
    """Check if GPU testing is enabled"""
    return os.environ.get("ENABLE_GPU_TESTS", "false").lower() == "true"


@pytest.fixture
def skip_if_no_gpu(gpu_test_enabled: bool):
    """Skip test if GPU is not available"""
    if not gpu_test_enabled:
        pytest.skip("GPU tests disabled, set ENABLE_GPU_TESTS=true to enable")


# Test markers
pytestmark = [
    pytest.mark.asyncio,
]


def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # Add unit marker to unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add performance marker to performance tests
        if "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        
        # Add GPU marker to GPU-related tests
        if "gpu" in item.name.lower() or "model" in item.name.lower():
            item.add_marker(pytest.mark.gpu)


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test"""
    yield
    
    # Cleanup any temporary files in /tmp
    import glob
    temp_files = glob.glob("/tmp/avatar_test_*")
    for temp_file in temp_files:
        try:
            if os.path.isfile(temp_file):
                os.remove(temp_file)
            elif os.path.isdir(temp_file):
                shutil.rmtree(temp_file, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors 