#!/usr/bin/env python3
"""
System Validation Script for Avatar Streaming Service
Validates all system requirements before service startup
"""

import sys
import os
import logging

# Add app to path
sys.path.append('/app')

def validate_system():
    """Validate system requirements"""
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Validating system requirements...")
    
    errors = []
    
    # Check Python version
    if sys.version_info < (3, 7):
        errors.append(f"Python 3.7+ required, found {sys.version}")
    else:
        logger.info(f"✅ Python version: {sys.version}")
    
    # Check required packages
    required_packages = [
        'numpy', 'opencv-python', 'onnxruntime', 'insightface',
        'fastapi', 'uvicorn', 'websockets', 'pydantic',
        'pillow', 'requests', 'sqlite3'
    ]
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'sqlite3':
                import sqlite3
            else:
                __import__(package)
            logger.info(f"✅ Package available: {package}")
        except ImportError:
            errors.append(f"Missing required package: {package}")
    
    # Check ONNX Runtime GPU support
    try:
        import onnxruntime as ort
        device = ort.get_device()
        providers = ort.get_available_providers()
        
        logger.info(f"✅ ONNX Runtime device: {device}")
        logger.info(f"✅ Available providers: {providers}")
        
        if 'CUDAExecutionProvider' not in providers:
            logger.warning("⚠️ CUDA provider not available - GPU acceleration disabled")
            
    except Exception as e:
        errors.append(f"ONNX Runtime error: {str(e)}")
    
    # Check environment variables
    required_env_vars = {
        'PYTHONPATH': '/app',
        'CUDA_VISIBLE_DEVICES': '0'
    }
    
    for var, expected in required_env_vars.items():
        value = os.environ.get(var)
        if value != expected:
            logger.warning(f"⚠️ Environment variable {var} = {value} (expected: {expected})")
    
    # Summary
    if errors:
        logger.error("❌ System validation failed:")
        for error in errors:
            logger.error(f"  • {error}")
        return False
    else:
        logger.info("✅ System validation passed - ready to start service")
        return True

if __name__ == "__main__":
    success = validate_system()
    sys.exit(0 if success else 1) 