#!/usr/bin/env python3
"""
Model Download Script for Avatar Streaming Service
Downloads and validates wav2lip and insightface models
"""

import os
import sys
import argparse
import hashlib
import requests
import zipfile
import shutil
from pathlib import Path
from typing import Dict, Optional
import logging

# Add app to path
sys.path.append('/app')

from app.config.settings import settings

# Model URLs and checksums
MODEL_REGISTRY = {
    "wav2lip.onnx": {
        "url": "https://github.com/OpenTalker/wav2lip-onnx/releases/download/v1.0/wav2lip.onnx",
        "checksum": "placeholder_checksum_1",  # Replace with actual checksums
        "size_mb": 45,
        "description": "Wav2Lip main model"
    },
    "wav2lip_gan.onnx": {
        "url": "https://github.com/OpenTalker/wav2lip-onnx/releases/download/v1.0/wav2lip_gan.onnx", 
        "checksum": "placeholder_checksum_2",  # Replace with actual checksums
        "size_mb": 45,
        "description": "Wav2Lip GAN model for better quality"
    },
    "antelope.zip": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "checksum": "placeholder_checksum_3",  # Replace with actual checksums
        "size_mb": 350,
        "description": "InsightFace Antelope models"
    }
}


class ModelDownloader:
    """Downloads and validates models for avatar service"""
    
    def __init__(self, verify_checksums: bool = True, gpu_optimize: bool = True):
        self.verify_checksums = verify_checksums
        self.gpu_optimize = gpu_optimize
        self.logger = self._setup_logging()
        
        # Create model directories
        self.wav2lip_dir = Path(settings.model_storage_path) / "wav2lip"
        self.insightface_dir = Path(settings.model_storage_path) / "insightface"
        self.wav2lip_dir.mkdir(parents=True, exist_ok=True)
        self.insightface_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for model downloader"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        return logging.getLogger(__name__)
    
    def download_all_models(self) -> bool:
        """Download all required models"""
        self.logger.info("📥 Starting model download process...")
        
        try:
            # Download wav2lip models
            if not self._download_wav2lip_models():
                return False
            
            # Download InsightFace models
            if not self._download_insightface_models():
                return False
            
            # Validate all models
            if not self._validate_all_models():
                return False
            
            self.logger.info("✅ All models downloaded and validated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model download failed: {str(e)}")
            return False
    
    def _download_wav2lip_models(self) -> bool:
        """Download wav2lip ONNX models"""
        self.logger.info("🎭 Downloading Wav2Lip models...")
        
        for model_name in ["wav2lip.onnx", "wav2lip_gan.onnx"]:
            model_path = self.wav2lip_dir / model_name
            
            if model_path.exists() and self._verify_model_file(model_path, model_name):
                self.logger.info(f"✅ {model_name} already exists and is valid")
                continue
            
            # Download model
            if not self._download_file(model_name, model_path):
                return False
        
        return True
    
    def _download_insightface_models(self) -> bool:
        """Download and extract InsightFace models"""
        self.logger.info("👤 Downloading InsightFace models...")
        
        # Check if antelope directory already exists
        antelope_dir = self.insightface_dir / "antelope"
        if antelope_dir.exists() and self._validate_insightface_models(antelope_dir):
            self.logger.info("✅ InsightFace models already exist and are valid")
            return True
        
        # Download zip file
        zip_path = self.insightface_dir / "antelope.zip"
        if not self._download_file("antelope.zip", zip_path):
            return False
        
        # Extract zip file
        try:
            self.logger.info("📦 Extracting InsightFace models...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.insightface_dir)
            
            # Remove zip file
            zip_path.unlink()
            
            # Rename extracted directory to antelope if needed
            extracted_dirs = [d for d in self.insightface_dir.iterdir() if d.is_dir()]
            if extracted_dirs and extracted_dirs[0].name != "antelope":
                extracted_dirs[0].rename(antelope_dir)
            
            self.logger.info("✅ InsightFace models extracted successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to extract InsightFace models: {str(e)}")
            return False
    
    def _download_file(self, model_name: str, destination: Path) -> bool:
        """Download a single model file"""
        if model_name not in MODEL_REGISTRY:
            self.logger.error(f"❌ Unknown model: {model_name}")
            return False
        
        model_info = MODEL_REGISTRY[model_name]
        url = model_info["url"]
        
        self.logger.info(f"📥 Downloading {model_name} ({model_info['size_mb']}MB)...")
        
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Progress indicator
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\r📊 Progress: {progress:.1f}%", end="", flush=True)
            
            print()  # New line after progress
            
            # Verify download
            if not self._verify_model_file(destination, model_name):
                destination.unlink()  # Remove invalid file
                return False
            
            self.logger.info(f"✅ Downloaded {model_name} successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to download {model_name}: {str(e)}")
            if destination.exists():
                destination.unlink()
            return False
    
    def _verify_model_file(self, file_path: Path, model_name: str) -> bool:
        """Verify model file integrity"""
        if not file_path.exists():
            return False
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        expected_size_mb = MODEL_REGISTRY[model_name]["size_mb"]
        
        if abs(file_size_mb - expected_size_mb) > expected_size_mb * 0.1:  # 10% tolerance
            self.logger.warning(f"⚠️ {model_name} size mismatch: {file_size_mb:.1f}MB vs expected {expected_size_mb}MB")
        
        # Verify checksum if enabled
        if self.verify_checksums:
            expected_checksum = MODEL_REGISTRY[model_name]["checksum"]
            if expected_checksum != "placeholder_checksum_1":  # Skip placeholder checksums
                actual_checksum = self._calculate_checksum(file_path)
                if actual_checksum != expected_checksum:
                    self.logger.error(f"❌ {model_name} checksum mismatch")
                    return False
        
        return True
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _validate_insightface_models(self, antelope_dir: Path) -> bool:
        """Validate InsightFace model directory"""
        required_files = ["1k3d68.onnx", "2d106det.onnx", "genderage.onnx", "scrfd_10g_bnkps.onnx"]
        
        for file_name in required_files:
            file_path = antelope_dir / file_name
            if not file_path.exists():
                self.logger.warning(f"⚠️ Missing InsightFace file: {file_name}")
                return False
        
        return True
    
    def _validate_all_models(self) -> bool:
        """Validate all downloaded models"""
        self.logger.info("🔍 Validating all models...")
        
        # Test ONNX models can be loaded
        try:
            import onnxruntime as ort
            
            # Test wav2lip models
            for model_name in ["wav2lip.onnx", "wav2lip_gan.onnx"]:
                model_path = self.wav2lip_dir / model_name
                session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
                self.logger.info(f"✅ {model_name} loads successfully")
            
            # Test InsightFace models
            antelope_dir = self.insightface_dir / "antelope"
            if self._validate_insightface_models(antelope_dir):
                self.logger.info("✅ InsightFace models validated")
            else:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model validation failed: {str(e)}")
            return False
    
    def cleanup_temp_files(self):
        """Clean up temporary download files"""
        temp_files = [
            self.insightface_dir / "antelope.zip"
        ]
        
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()
                self.logger.info(f"🧹 Cleaned up {temp_file.name}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Download models for Avatar Streaming Service")
    parser.add_argument("--verify-checksums", action="store_true", 
                       help="Verify model checksums")
    parser.add_argument("--gpu-optimize", action="store_true", 
                       help="Optimize models for GPU")
    parser.add_argument("--force-download", action="store_true",
                       help="Force re-download even if models exist")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(
        verify_checksums=args.verify_checksums,
        gpu_optimize=args.gpu_optimize
    )
    
    # Clean up any existing files if force download
    if args.force_download:
        downloader.logger.info("🔄 Force download enabled - removing existing models")
        if downloader.wav2lip_dir.exists():
            shutil.rmtree(downloader.wav2lip_dir)
            downloader.wav2lip_dir.mkdir(parents=True, exist_ok=True)
        if downloader.insightface_dir.exists():
            shutil.rmtree(downloader.insightface_dir)
            downloader.insightface_dir.mkdir(parents=True, exist_ok=True)
    
    # Download models
    success = downloader.download_all_models()
    
    # Cleanup
    downloader.cleanup_temp_files()
    
    if success:
        print("🎉 Model download completed successfully!")
        sys.exit(0)
    else:
        print("❌ Model download failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 