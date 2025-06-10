#!/usr/bin/env python3
"""
Environment Setup Script for Avatar Streaming Service
Handles environment validation, dependency installation, and configuration setup.
"""

import os
import sys
import logging
import subprocess
import platform
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnvironmentSetup:
    """Environment setup and validation for Avatar Streaming Service"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.python_version = sys.version_info
        self.platform = platform.system()
        self.errors = []
        self.warnings = []
        
    def setup_environment(self) -> bool:
        """Run complete environment setup"""
        logger.info("🚀 Starting Avatar Service Environment Setup")
        
        setup_steps = [
            ("Validating Python version", self.validate_python_version),
            ("Checking system requirements", self.check_system_requirements),
            ("Validating GPU availability", self.validate_gpu),
            ("Setting up project directories", self.setup_directories),
            ("Installing Python dependencies", self.install_dependencies),
            ("Validating environment configuration", self.validate_environment_config),
            ("Setting up database", self.setup_database),
            ("Validating model downloads", self.validate_models),
            ("Testing avatar cache system", self.test_avatar_cache),
        ]
        
        for step_name, step_func in setup_steps:
            logger.info(f"📋 {step_name}...")
            try:
                if not step_func():
                    logger.error(f"❌ Failed: {step_name}")
                    return False
                logger.info(f"✅ Completed: {step_name}")
            except Exception as e:
                logger.error(f"❌ Error in {step_name}: {e}")
                self.errors.append(f"{step_name}: {e}")
                return False
        
        self.print_setup_summary()
        return len(self.errors) == 0
    
    def validate_python_version(self) -> bool:
        """Validate Python version compatibility"""
        required_version = (3, 7)
        current_version = self.python_version[:2]
        
        if current_version < required_version:
            self.errors.append(
                f"Python {required_version[0]}.{required_version[1]}+ required, "
                f"found {current_version[0]}.{current_version[1]}"
            )
            return False
        
        logger.info(f"✓ Python {current_version[0]}.{current_version[1]} detected")
        return True
    
    def check_system_requirements(self) -> bool:
        """Check system-level requirements"""
        requirements = {
            'git': 'Git version control',
            'ffmpeg': 'FFmpeg for audio/video processing',
        }
        
        missing = []
        for cmd, desc in requirements.items():
            if not shutil.which(cmd):
                missing.append(f"{cmd} ({desc})")
        
        if missing:
            self.errors.append(f"Missing system requirements: {', '.join(missing)}")
            return False
        
        # Check Docker if available
        if shutil.which('docker'):
            try:
                result = subprocess.run(['docker', '--version'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    logger.info(f"✓ {result.stdout.strip()}")
                else:
                    self.warnings.append("Docker not properly configured")
            except Exception:
                self.warnings.append("Docker not accessible")
        else:
            self.warnings.append("Docker not installed (optional for development)")
        
        return True
    
    def validate_gpu(self) -> bool:
        """Validate GPU availability and CUDA installation"""
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"✓ CUDA available: {gpu_count} GPU(s) - {gpu_name}")
                
                # Check CUDA version
                cuda_version = torch.version.cuda
                logger.info(f"✓ CUDA version: {cuda_version}")
                
                # Test GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                logger.info(f"✓ GPU memory: {gpu_memory_gb:.1f} GB")
                
                if gpu_memory_gb < 8:
                    self.warnings.append(f"GPU memory ({gpu_memory_gb:.1f} GB) may be insufficient for optimal performance")
                
                return True
            else:
                self.warnings.append("CUDA not available - will fallback to CPU processing")
                return True
                
        except ImportError:
            self.warnings.append("PyTorch not installed - GPU validation skipped")
            return True
        except Exception as e:
            self.warnings.append(f"GPU validation failed: {e}")
            return True
    
    def setup_directories(self) -> bool:
        """Create necessary project directories"""
        directories = [
            'logs',
            'temp',
            'data/avatar_registry',
            'data/avatar_registry/face_cache',
            'assets/avatars/registered',
            'assets/models/wav2lip',
            'assets/models/insightface/antelope',
            'assets/audio/silence_patterns',
            'assets/audio/test_samples',
        ]
        
        for dir_path in directories:
            full_path = self.project_root / dir_path
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"✓ Created directory: {dir_path}")
            except Exception as e:
                self.errors.append(f"Failed to create directory {dir_path}: {e}")
                return False
        
        # Create placeholder files
        placeholder_files = [
            'logs/.gitkeep',
            'temp/.gitkeep',
            'data/avatar_registry/face_cache/.gitkeep',
        ]
        
        for file_path in placeholder_files:
            full_path = self.project_root / file_path
            try:
                full_path.touch()
            except Exception as e:
                self.warnings.append(f"Could not create placeholder {file_path}: {e}")
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies"""
        requirements_files = [
            'docker/requirements.txt',
            'requirements.txt'
        ]
        
        # Find requirements file
        requirements_file = None
        for req_file in requirements_files:
            full_path = self.project_root / req_file
            if full_path.exists():
                requirements_file = full_path
                break
        
        if not requirements_file:
            self.warnings.append("No requirements.txt found - skipping dependency installation")
            return True
        
        try:
            logger.info(f"Installing dependencies from {requirements_file}")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self.errors.append(f"Dependency installation failed: {result.stderr}")
                return False
            
            logger.info("✓ Dependencies installed successfully")
            return True
            
        except subprocess.TimeoutExpired:
            self.errors.append("Dependency installation timed out")
            return False
        except Exception as e:
            self.errors.append(f"Dependency installation error: {e}")
            return False
    
    def validate_environment_config(self) -> bool:
        """Validate environment configuration files"""
        env_files = ['env.dev', 'env.prod', '.env.example']
        
        found_configs = []
        for env_file in env_files:
            env_path = self.project_root / env_file
            if env_path.exists():
                found_configs.append(env_file)
                logger.info(f"✓ Found environment config: {env_file}")
        
        if not found_configs:
            self.warnings.append("No environment configuration files found")
        
        # Check for required environment variables
        required_vars = [
            'OPENAI_API_KEY',
            'MCP_SERVER_URL',
            'GPU_MEMORY_LIMIT',
            'AVATAR_STORAGE_PATH'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.environ.get(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.warnings.append(f"Environment variables not set: {', '.join(missing_vars)}")
        
        return True
    
    def setup_database(self) -> bool:
        """Setup avatar registry database"""
        db_path = self.project_root / 'data' / 'avatar_registry' / 'avatars.db'
        
        try:
            import sqlite3
            
            # Create database if it doesn't exist
            if not db_path.exists():
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                # Create avatars table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS avatars (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        avatar_id TEXT UNIQUE NOT NULL,
                        name TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        cache_path TEXT,
                        owner_id TEXT,
                        file_format TEXT NOT NULL,
                        file_size INTEGER,
                        face_quality_score REAL,
                        registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_accessed TIMESTAMP,
                        access_count INTEGER DEFAULT 0,
                        is_active BOOLEAN DEFAULT 1,
                        metadata_json TEXT
                    )
                ''')
                
                conn.commit()
                conn.close()
                logger.info("✓ Avatar database created")
            else:
                logger.info("✓ Avatar database exists")
            
            return True
            
        except Exception as e:
            self.errors.append(f"Database setup failed: {e}")
            return False
    
    def validate_models(self) -> bool:
        """Validate model files and downloads"""
        model_paths = {
            'wav2lip': self.project_root / 'assets' / 'models' / 'wav2lip',
            'insightface': self.project_root / 'assets' / 'models' / 'insightface'
        }
        
        for model_name, model_path in model_paths.items():
            if not model_path.exists():
                self.warnings.append(f"{model_name} models directory not found - run download_models.py")
                continue
            
            # Check for model files
            if model_name == 'wav2lip':
                expected_files = ['wav2lip.onnx', 'wav2lip_gan.onnx']
                for file_name in expected_files:
                    file_path = model_path / file_name
                    if file_path.exists():
                        logger.info(f"✓ Found {model_name} model: {file_name}")
                    else:
                        self.warnings.append(f"Missing {model_name} model: {file_name}")
            
            elif model_name == 'insightface':
                antelope_path = model_path / 'antelope'
                if antelope_path.exists():
                    logger.info(f"✓ Found {model_name} models in antelope directory")
                else:
                    self.warnings.append(f"Missing {model_name} antelope models directory")
        
        return True
    
    def test_avatar_cache(self) -> bool:
        """Test avatar cache system functionality"""
        cache_dir = self.project_root / 'data' / 'avatar_registry' / 'face_cache'
        
        try:
            # Test cache directory accessibility
            if not cache_dir.exists():
                self.errors.append("Avatar cache directory not found")
                return False
            
            # Test write permissions
            test_file = cache_dir / 'test_cache.tmp'
            try:
                test_file.write_text('test')
                test_file.unlink()
                logger.info("✓ Avatar cache directory writable")
            except Exception as e:
                self.errors.append(f"Avatar cache directory not writable: {e}")
                return False
            
            return True
            
        except Exception as e:
            self.errors.append(f"Avatar cache test failed: {e}")
            return False
    
    def print_setup_summary(self):
        """Print setup summary"""
        logger.info("\n" + "="*60)
        logger.info("🎯 ENVIRONMENT SETUP SUMMARY")
        logger.info("="*60)
        
        if not self.errors:
            logger.info("✅ Setup completed successfully!")
        else:
            logger.error(f"❌ Setup failed with {len(self.errors)} error(s):")
            for error in self.errors:
                logger.error(f"  • {error}")
        
        if self.warnings:
            logger.warning(f"⚠️  {len(self.warnings)} warning(s):")
            for warning in self.warnings:
                logger.warning(f"  • {warning}")
        
        logger.info("\n📋 Next Steps:")
        logger.info("1. Set your OpenAI API key in environment configuration")
        logger.info("2. Configure MCP server URL if using external RAG")
        logger.info("3. Run 'python scripts/download_models.py' to download AI models")
        logger.info("4. Run 'python scripts/initialize_avatar_cache.py' to setup avatar system")
        logger.info("5. Start the service with 'python app/main.py'")
        
        logger.info("="*60)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Avatar Service Environment Setup")
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging')
    parser.add_argument('--project-root', type=Path,
                       help='Project root directory (default: auto-detect)')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup = EnvironmentSetup(project_root=args.project_root)
    success = setup.setup_environment()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 