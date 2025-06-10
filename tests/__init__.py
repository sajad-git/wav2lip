"""
Test Suite for Avatar Streaming Service

This package contains comprehensive tests for the Avatar Streaming Service including:
- Unit tests for individual components
- Integration tests for service workflows  
- Load tests for performance validation
- Avatar registration and caching tests
"""

__version__ = "1.0.0"
__author__ = "Avatar Streaming Service Team"

# Test configuration
TEST_CONFIG = {
    "enable_gpu_tests": False,
    "enable_model_tests": False,
    "enable_avatar_tests": True,
    "test_timeout": 30,
    "max_test_avatars": 5,
}

# Test utilities and fixtures are available through conftest.py 