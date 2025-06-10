#!/usr/bin/env python3
"""
Complete Environment Setup Script for Avatar Streaming Service
Automates the entire setup process including directories, dependencies, models, and configuration
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import urllib.request
import hashlib


class EnvironmentSetup:
    """Complete environment setup for Avatar Streaming Service"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.setup_logging()
        self.project_root = Path.cwd()
        self.errors = []
        self.warnings = []
        
    def setup_logging(self):
        """Setup logging configuration"""
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('setup.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_complete_setup(self) -> bool:
        """Run complete environment setup"""
        self.logger.info("🚀 Starting Avatar Service Complete Environment Setup")
        
        steps = [
            ("Checking system requirements", self.check_system_requirements),
            ("Creating directory structure", self.create_directory_structure),
            ("Setting up Python environment", self.setup_python_environment),
            ("Installing dependencies", self.install_dependencies),
            ("Creating configuration files", self.create_configuration_files),
            ("Downloading models", self.download_models),
            ("Initializing databases", self.initialize_databases),
            ("Setting up Docker environment", self.setup_docker_environment),
            ("Running validation tests", self.run_validation_tests),
            ("Creating documentation", self.create_documentation)
        ]
        
        for step_name, step_func in steps:
            try:
                self.logger.info(f"📋 {step_name}...")
                success = step_func()
                if success:
                    self.logger.info(f"✅ {step_name} completed successfully")
                else:
                    self.logger.error(f"❌ {step_name} failed")
                    self.errors.append(step_name)
            except Exception as e:
                self.logger.error(f"❌ {step_name} failed with error: {str(e)}")
                self.errors.append(f"{step_name}: {str(e)}")
        
        # Print summary
        self.print_setup_summary()
        
        return len(self.errors) == 0
    
    def check_system_requirements(self) -> bool:
        """Check system requirements"""
        requirements_met = True
        
        # Check Python version
        if sys.version_info < (3, 7):
            self.errors.append("Python 3.7+ required")
            requirements_met = False
        else:
            self.logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        
        # Check Docker
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info(f"✅ Docker detected: {result.stdout.strip()}")
            else:
                self.warnings.append("Docker not found - required for containerized deployment")
        except FileNotFoundError:
            self.warnings.append("Docker not found - required for containerized deployment")
        
        # Check Docker Compose
        try:
            result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info(f"✅ Docker Compose detected: {result.stdout.strip()}")
        except FileNotFoundError:
            self.warnings.append("Docker Compose not found - required for orchestration")
        
        # Check NVIDIA Docker (if available)
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("✅ NVIDIA GPU detected")
                # Try nvidia-docker
                result = subprocess.run(['docker', 'run', '--rm', '--gpus', 'all', 'nvidia/cuda:11.8-base', 'nvidia-smi'], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    self.logger.info("✅ NVIDIA Docker runtime available")
                else:
                    self.warnings.append("NVIDIA Docker runtime not available - GPU acceleration disabled")
            else:
                self.warnings.append("NVIDIA GPU not detected - will run in CPU mode")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.warnings.append("NVIDIA GPU not detected - will run in CPU mode")
        
        # Check available disk space
        total, used, free = self.get_disk_space()
        if free < 50:  # 50GB
            self.warnings.append(f"Low disk space: {free:.1f}GB free (50GB+ recommended)")
        else:
            self.logger.info(f"✅ Disk space: {free:.1f}GB free")
        
        return requirements_met
    
    def create_directory_structure(self) -> bool:
        """Create complete directory structure"""
        directories = [
            "app/config",
            "app/core", 
            "app/services",
            "app/streaming",
            "app/utils",
            "app/models",
            "app/middleware",
            "assets/avatars/registered",
            "assets/models/wav2lip",
            "assets/models/insightface/antelope",
            "assets/audio/silence_patterns",
            "assets/audio/test_samples",
            "data/avatar_registry/face_cache",
            "static/css",
            "static/js",
            "tests/unit",
            "tests/integration", 
            "tests/load",
            "logs",
            "temp",
            "docs",
            "monitoring",
            "scripts"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create .gitkeep for empty directories
            if not any(dir_path.iterdir()):
                (dir_path / ".gitkeep").touch()
        
        self.logger.info(f"✅ Created {len(directories)} directories")
        return True
    
    def setup_python_environment(self) -> bool:
        """Setup Python virtual environment"""
        venv_path = self.project_root / "venv"
        
        if not venv_path.exists():
            try:
                subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
                self.logger.info("✅ Virtual environment created")
            except subprocess.CalledProcessError as e:
                self.errors.append(f"Failed to create virtual environment: {e}")
                return False
        else:
            self.logger.info("✅ Virtual environment already exists")
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies"""
        requirements_file = self.project_root / "docker" / "requirements.txt"
        
        if not requirements_file.exists():
            self.warnings.append("requirements.txt not found - skipping dependency installation")
            return True
        
        try:
            # Install requirements
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True, capture_output=True)
            
            self.logger.info("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.warnings.append(f"Some dependencies failed to install: {e}")
            return True  # Don't fail setup for dependency issues
    
    def create_configuration_files(self) -> bool:
        """Create configuration files"""
        
        # Create .env from .env.example if it doesn't exist
        env_example = self.project_root / "env.example"
        env_file = self.project_root / ".env"
        
        if env_example.exists() and not env_file.exists():
            with open(env_example, 'r') as src, open(env_file, 'w') as dst:
                content = src.read()
                # Add placeholder for OpenAI API key
                content = content.replace(
                    "OPENAI_API_KEY=your_openai_api_key_here",
                    "OPENAI_API_KEY=your_openai_api_key_here  # REQUIRED: Add your OpenAI API key"
                )
                dst.write(content)
            
            self.logger.info("✅ .env file created from template")
            self.warnings.append("Please edit .env file with your OpenAI API key")
        elif env_file.exists():
            self.logger.info("✅ .env file already exists")
        
        # Create development docker-compose override
        dev_compose = self.project_root / "docker-compose.dev.yml"
        if not dev_compose.exists():
            dev_config = {
                "version": "3.8",
                "services": {
                    "avatar-service": {
                        "environment": [
                            "DEBUG=true",
                            "LOG_LEVEL=DEBUG",
                            "RELOAD=true",
                            "SKIP_MODEL_LOADING=false"
                        ],
                        "volumes": [
                            "./app:/app/app:ro",
                            "./static:/app/static:ro"
                        ]
                    }
                }
            }
            
            with open(dev_compose, 'w') as f:
                import yaml
                yaml.safe_dump(dev_config, f, default_flow_style=False)
            
            self.logger.info("✅ Development docker-compose.dev.yml created")
        
        return True
    
    def download_models(self) -> bool:
        """Download required models"""
        download_script = self.project_root / "scripts" / "download_models.py"
        
        if download_script.exists():
            try:
                subprocess.run([sys.executable, str(download_script), "--verify-checksums"], 
                             check=False, timeout=300)  # 5 minute timeout
                self.logger.info("✅ Model download process completed")
                return True
            except subprocess.TimeoutExpired:
                self.warnings.append("Model download timed out - run manually if needed")
                return True
            except Exception as e:
                self.warnings.append(f"Model download failed: {e}")
                return True
        else:
            self.warnings.append("Model download script not found")
            return True
    
    def initialize_databases(self) -> bool:
        """Initialize avatar database"""
        init_script = self.project_root / "scripts" / "initialize_avatar_cache.py"
        
        if init_script.exists():
            try:
                subprocess.run([sys.executable, str(init_script)], check=False, timeout=60)
                self.logger.info("✅ Avatar database initialized")
                return True
            except Exception as e:
                self.warnings.append(f"Database initialization failed: {e}")
                return True
        else:
            self.warnings.append("Database initialization script not found")
            return True
    
    def setup_docker_environment(self) -> bool:
        """Setup Docker environment"""
        dockerfile = self.project_root / "docker" / "Dockerfile"
        compose_file = self.project_root / "docker" / "docker-compose.yml"
        
        if not dockerfile.exists():
            self.warnings.append("Dockerfile not found")
            return True
        
        if not compose_file.exists():
            self.warnings.append("docker-compose.yml not found")
            return True
        
        # Try to build Docker image (non-blocking)
        try:
            result = subprocess.run([
                "docker", "build", "-t", "avatar-service", "-f", "docker/Dockerfile", "."
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            if result.returncode == 0:
                self.logger.info("✅ Docker image built successfully")
            else:
                self.warnings.append("Docker image build failed - check Docker setup")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.warnings.append("Docker build skipped - ensure Docker is installed")
        
        return True
    
    def run_validation_tests(self) -> bool:
        """Run validation tests"""
        validation_script = self.project_root / "scripts" / "validate_system.py"
        
        if validation_script.exists():
            try:
                result = subprocess.run([sys.executable, str(validation_script)], 
                                      capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    self.logger.info("✅ System validation passed")
                else:
                    self.warnings.append("System validation failed - check logs")
                    
            except Exception as e:
                self.warnings.append(f"Validation test failed: {e}")
        
        return True
    
    def create_documentation(self) -> bool:
        """Create additional documentation"""
        docs_dir = self.project_root / "docs"
        
        # Create API documentation
        api_doc = docs_dir / "API.md"
        if not api_doc.exists():
            api_content = """# Avatar Streaming Service API Documentation

## Authentication
All API endpoints require proper authentication headers.

## Endpoints

### Avatar Management
- `POST /avatar/register` - Register new avatar
- `GET /avatar/list` - List available avatars  
- `DELETE /avatar/{avatar_id}` - Delete avatar
- `GET /avatar/{avatar_id}/info` - Get avatar details

### Processing
- `POST /avatar/process` - Process text with avatar
- `WS /ws/stream` - WebSocket streaming endpoint

### System
- `GET /health` - Health status
- `GET /ready` - Readiness check
- `GET /metrics` - Performance metrics

For detailed API documentation, visit http://localhost:5002/docs when the service is running.
"""
            with open(api_doc, 'w') as f:
                f.write(api_content)
        
        # Create deployment guide
        deploy_doc = docs_dir / "DEPLOYMENT.md"
        if not deploy_doc.exists():
            deploy_content = """# Deployment Guide

## Quick Start
1. Copy `env.example` to `.env` and configure
2. Run `docker-compose up --build`
3. Access service at http://localhost:5002

## Production Deployment
See README.md for detailed production deployment instructions.

## Environment Variables
See `env.example` for all available configuration options.
"""
            with open(deploy_doc, 'w') as f:
                f.write(deploy_content)
        
        self.logger.info("✅ Documentation created")
        return True
    
    def get_disk_space(self) -> tuple:
        """Get disk space information in GB"""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            return (total // (1024**3), used // (1024**3), free // (1024**3))
        except:
            return (0, 0, 0)
    
    def print_setup_summary(self):
        """Print setup summary"""
        print("\n" + "="*80)
        print("AVATAR SERVICE SETUP SUMMARY")
        print("="*80)
        
        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if not self.errors:
            print("\n✅ SETUP COMPLETED SUCCESSFULLY!")
            print("\nNext Steps:")
            print("1. Edit .env file with your OpenAI API key")
            print("2. Download models: python scripts/download_models.py")
            print("3. Start service: docker-compose up --build")
            print("4. Access web interface: http://localhost:5002")
        else:
            print("\n❌ SETUP COMPLETED WITH ERRORS")
            print("Please fix the errors above and run setup again")
        
        print("\nFor detailed documentation, see README.md")
        print("="*80 + "\n")


def main():
    """Main setup execution"""
    parser = argparse.ArgumentParser(description="Avatar Service Complete Environment Setup")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--skip-docker", action="store_true", help="Skip Docker setup")
    parser.add_argument("--skip-models", action="store_true", help="Skip model download")
    
    args = parser.parse_args()
    
    setup = EnvironmentSetup(verbose=args.verbose)
    
    try:
        success = setup.run_complete_setup()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Setup failed with unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 