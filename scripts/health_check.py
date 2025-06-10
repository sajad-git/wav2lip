#!/usr/bin/env python3
"""
Health Check Script for Avatar Streaming Service
Validates system status, models, and avatars
"""

import os
import sys
import argparse
import logging
import sqlite3
from pathlib import Path

# Add app to path
sys.path.append('/app')

from app.config.settings import settings
from app.config.avatar_config import avatar_config


class HealthChecker:
    """System health checker"""
    
    def __init__(self, check_models: bool = True, check_avatars: bool = True):
        self.check_models = check_models
        self.check_avatars = check_avatars
        self.logger = self._setup_logging()
        self.errors = []
        self.warnings = []
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        return logging.getLogger(__name__)
    
    def run_health_check(self) -> bool:
        """Run complete health check"""
        self.logger.info("🔍 Starting health check...")
        
        # Check basic system requirements
        self._check_system_requirements()
        
        # Check models if requested
        if self.check_models:
            self._check_models()
        
        # Check avatars if requested  
        if self.check_avatars:
            self._check_avatars()
        
        # Check directories
        self._check_directories()
        
        # Check GPU availability
        self._check_gpu()
        
        # Summary
        self._print_summary()
        
        return len(self.errors) == 0
    
    def _check_system_requirements(self):
        """Check basic system requirements"""
        self.logger.info("🖥️ Checking system requirements...")
        
        try:
            import psutil
            
            # Check memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < 8:
                self.errors.append(f"Insufficient memory: {available_gb:.1f}GB available (8GB required)")
            else:
                self.logger.info(f"✅ Memory: {available_gb:.1f}GB available")
            
            # Check disk space
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)
            
            if free_gb < 10:
                self.warnings.append(f"Low disk space: {free_gb:.1f}GB free")
            else:
                self.logger.info(f"✅ Disk space: {free_gb:.1f}GB free")
                
        except ImportError:
            self.warnings.append("psutil not available for system checks")
    
    def _check_models(self):
        """Check model availability"""
        self.logger.info("🎭 Checking models...")
        
        # Check wav2lip models
        wav2lip_dir = Path(settings.model_storage_path) / "wav2lip"
        for model_name in ["wav2lip.onnx", "wav2lip_gan.onnx"]:
            model_path = wav2lip_dir / model_name
            if not model_path.exists():
                self.errors.append(f"Missing model: {model_path}")
            else:
                # Check file size
                size_mb = model_path.stat().st_size / (1024*1024)
                if size_mb < 10:  # Models should be at least 10MB
                    self.warnings.append(f"Model file seems small: {model_name} ({size_mb:.1f}MB)")
                else:
                    self.logger.info(f"✅ Model found: {model_name} ({size_mb:.1f}MB)")
        
        # Check InsightFace models
        insightface_dir = Path(settings.model_storage_path) / "insightface" / "antelope"
        if not insightface_dir.exists():
            self.errors.append(f"Missing InsightFace directory: {insightface_dir}")
        else:
            required_files = ["1k3d68.onnx", "2d106det.onnx", "genderage.onnx", "scrfd_10g_bnkps.onnx"]
            for file_name in required_files:
                file_path = insightface_dir / file_name
                if not file_path.exists():
                    self.errors.append(f"Missing InsightFace file: {file_name}")
                else:
                    self.logger.info(f"✅ InsightFace file found: {file_name}")
        
        # Test ONNX loading
        try:
            import onnxruntime as ort
            
            # Test one model loading
            wav2lip_path = wav2lip_dir / "wav2lip.onnx"
            if wav2lip_path.exists():
                session = ort.InferenceSession(str(wav2lip_path), providers=['CPUExecutionProvider'])
                self.logger.info("✅ ONNX Runtime working correctly")
                
        except Exception as e:
            self.errors.append(f"ONNX Runtime error: {str(e)}")
    
    def _check_avatars(self):
        """Check avatar system"""
        self.logger.info("👤 Checking avatar system...")
        
        # Check database
        if not os.path.exists(avatar_config.database_path):
            self.errors.append(f"Avatar database not found: {avatar_config.database_path}")
            return
        
        try:
            conn = sqlite3.connect(avatar_config.database_path)
            cursor = conn.cursor()
            
            # Check tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['avatars', 'face_cache']
            for table in required_tables:
                if table not in tables:
                    self.errors.append(f"Missing database table: {table}")
                else:
                    self.logger.info(f"✅ Database table found: {table}")
            
            # Check avatar count
            cursor.execute("SELECT COUNT(*) FROM avatars WHERE is_active = 1")
            active_avatars = cursor.fetchone()[0]
            
            if active_avatars == 0:
                self.warnings.append("No active avatars found")
            else:
                self.logger.info(f"✅ Found {active_avatars} active avatars")
            
            # Check default avatars
            cursor.execute("SELECT COUNT(*) FROM avatars WHERE owner_id = 'system'")
            default_avatars = cursor.fetchone()[0]
            
            if default_avatars == 0:
                self.warnings.append("No default avatars found")
            else:
                self.logger.info(f"✅ Found {default_avatars} default avatars")
            
            conn.close()
            
        except Exception as e:
            self.errors.append(f"Database error: {str(e)}")
        
        # Check cache directory
        if not os.path.exists(avatar_config.cache_storage_path):
            self.errors.append(f"Cache directory not found: {avatar_config.cache_storage_path}")
        else:
            self.logger.info(f"✅ Cache directory exists: {avatar_config.cache_storage_path}")
    
    def _check_directories(self):
        """Check required directories"""
        self.logger.info("📁 Checking directories...")
        
        required_dirs = [
            settings.model_storage_path,
            settings.avatar_storage_path,
            settings.cache_storage_path,
            settings.log_storage_path,
            settings.temp_storage_path
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                self.errors.append(f"Missing directory: {dir_path}")
            else:
                self.logger.info(f"✅ Directory exists: {dir_path}")
    
    def _check_gpu(self):
        """Check GPU availability"""
        self.logger.info("🔧 Checking GPU...")
        
        try:
            import onnxruntime as ort
            
            # Check ONNX Runtime device
            device = ort.get_device()
            if device == 'GPU':
                self.logger.info("✅ GPU available for ONNX Runtime")
                
                # Check CUDA providers
                providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in providers:
                    self.logger.info("✅ CUDA provider available")
                else:
                    self.warnings.append("CUDA provider not available")
            else:
                self.warnings.append("GPU not available - using CPU")
                
        except Exception as e:
            self.errors.append(f"GPU check error: {str(e)}")
    
    def _print_summary(self):
        """Print health check summary"""
        print("\n" + "="*50)
        print("HEALTH CHECK SUMMARY")
        print("="*50)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ ALL CHECKS PASSED - System is healthy!")
        elif not self.errors:
            print(f"\n✅ System is functional with {len(self.warnings)} warnings")
        else:
            print(f"\n❌ System has {len(self.errors)} critical errors")
        
        print("="*50)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Health check for Avatar Streaming Service")
    parser.add_argument("--check-models", action="store_true", 
                       help="Check model availability")
    parser.add_argument("--check-avatars", action="store_true",
                       help="Check avatar system")
    parser.add_argument("--all", action="store_true",
                       help="Run all checks")
    
    args = parser.parse_args()
    
    # If no specific checks requested, run basic checks
    check_models = args.check_models or args.all
    check_avatars = args.check_avatars or args.all
    
    checker = HealthChecker(
        check_models=check_models,
        check_avatars=check_avatars
    )
    
    success = checker.run_health_check()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 