"""
File Utilities
Comprehensive file handling utilities for the Avatar service
"""

import os
import shutil
import hashlib
import mimetypes
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, BinaryIO
from dataclasses import dataclass
from datetime import datetime
import json

from .validation import ValidationResult


@dataclass
class FileInfo:
    """File information metadata"""
    filename: str
    filepath: str
    size: int
    mime_type: str
    extension: str
    created_at: datetime
    modified_at: datetime
    checksum: str
    is_valid: bool = True
    error_message: Optional[str] = None


@dataclass
class DirectoryInfo:
    """Directory information and statistics"""
    path: str
    total_files: int
    total_size: int
    subdirectories: List[str]
    file_types: Dict[str, int]  # extension -> count
    created_at: datetime
    modified_at: datetime


class FileManager:
    """File management utilities"""
    
    def __init__(self, base_path: str = "/app"):
        self.logger = logging.getLogger(__name__)
        self.base_path = Path(base_path)
        
        # File size limits (in bytes)
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.max_avatar_size = 50 * 1024 * 1024  # 50MB
        
        # Allowed file types
        self.allowed_image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        self.allowed_video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
        self.allowed_audio_extensions = {'.wav', '.mp3', '.aac', '.ogg', '.flac', '.m4a'}
        
        # Dangerous file types to block
        self.blocked_extensions = {'.exe', '.bat', '.cmd', '.scr', '.com', '.pif', '.js', '.vbs'}
    
    def get_file_info(self, filepath: Union[str, Path]) -> Optional[FileInfo]:
        """Get comprehensive file information"""
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                self.logger.warning(f"File does not exist: {filepath}")
                return None
            
            stat = filepath.stat()
            
            # Get file metadata
            filename = filepath.name
            size = stat.st_size
            extension = filepath.suffix.lower()
            mime_type, _ = mimetypes.guess_type(str(filepath))
            mime_type = mime_type or "application/octet-stream"
            
            # Get timestamps
            created_at = datetime.fromtimestamp(stat.st_ctime)
            modified_at = datetime.fromtimestamp(stat.st_mtime)
            
            # Calculate checksum
            checksum = self.calculate_file_checksum(filepath)
            
            return FileInfo(
                filename=filename,
                filepath=str(filepath),
                size=size,
                mime_type=mime_type,
                extension=extension,
                created_at=created_at,
                modified_at=modified_at,
                checksum=checksum
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get file info for {filepath}: {str(e)}")
            return None
    
    def calculate_file_checksum(self, filepath: Union[str, Path], 
                              algorithm: str = "sha256") -> str:
        """Calculate file checksum"""
        try:
            hash_func = getattr(hashlib, algorithm)()
            
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            
            return hash_func.hexdigest()
            
        except Exception as e:
            self.logger.error(f"Failed to calculate checksum for {filepath}: {str(e)}")
            return ""
    
    def validate_file(self, filepath: Union[str, Path], 
                     file_type: str = "general") -> ValidationResult:
        """Validate file based on type and security requirements"""
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                return ValidationResult(
                    is_valid=False,
                    error_messages=["File does not exist"],
                    security_score=0.0
                )
            
            file_info = self.get_file_info(filepath)
            if not file_info:
                return ValidationResult(
                    is_valid=False,
                    error_messages=["Could not read file information"],
                    security_score=0.0
                )
            
            errors = []
            warnings = []
            security_score = 1.0
            
            # Check file size
            max_size = self.max_avatar_size if file_type == "avatar" else self.max_file_size
            if file_info.size > max_size:
                errors.append(f"File size {file_info.size} exceeds limit {max_size}")
            
            # Check file extension
            if file_info.extension in self.blocked_extensions:
                errors.append(f"File type {file_info.extension} is not allowed")
                security_score = 0.0
            
            # Validate based on file type
            if file_type == "avatar":
                if file_info.extension not in (self.allowed_image_extensions | self.allowed_video_extensions):
                    errors.append(f"Avatar must be image or video file")
            elif file_type == "audio":
                if file_info.extension not in self.allowed_audio_extensions:
                    errors.append(f"Audio file type {file_info.extension} not supported")
            
            # Security checks
            security_issues = self._perform_security_scan(filepath)
            if security_issues:
                warnings.extend(security_issues)
                security_score *= 0.8
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                error_messages=errors,
                warnings=warnings,
                security_score=security_score
            )
            
        except Exception as e:
            self.logger.error(f"File validation failed for {filepath}: {str(e)}")
            return ValidationResult(
                is_valid=False,
                error_messages=[f"Validation error: {str(e)}"],
                security_score=0.0
            )
    
    def _perform_security_scan(self, filepath: Path) -> List[str]:
        """Perform basic security scanning"""
        warnings = []
        
        try:
            # Check for suspicious file headers
            with open(filepath, 'rb') as f:
                header = f.read(512)
            
            # Check for executable signatures
            executable_signatures = [
                b'MZ',  # Windows executable
                b'\x7fELF',  # Linux executable
                b'\xca\xfe\xba\xbe',  # Java class file
                b'PK\x03\x04'  # ZIP file (could contain executables)
            ]
            
            for sig in executable_signatures:
                if header.startswith(sig):
                    warnings.append("File contains executable signature")
                    break
            
            # Check for embedded scripts
            script_patterns = [b'<script', b'javascript:', b'vbscript:']
            for pattern in script_patterns:
                if pattern in header:
                    warnings.append("File may contain embedded scripts")
                    break
            
        except Exception as e:
            self.logger.debug(f"Security scan error for {filepath}: {str(e)}")
            warnings.append("Could not complete security scan")
        
        return warnings
    
    def create_secure_filename(self, original_filename: str, 
                             prefix: str = "", suffix: str = "") -> str:
        """Create secure filename from user input"""
        # Remove path components
        filename = os.path.basename(original_filename)
        
        # Remove or replace dangerous characters
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        filename = ''.join(c if c in safe_chars else '_' for c in filename)
        
        # Ensure filename is not empty
        if not filename or filename in ['.', '..']:
            filename = "file"
        
        # Add prefix and suffix
        name, ext = os.path.splitext(filename)
        return f"{prefix}{name}{suffix}{ext}"
    
    def create_directory(self, directory_path: Union[str, Path], 
                        mode: int = 0o755) -> bool:
        """Create directory with proper permissions"""
        try:
            directory_path = Path(directory_path)
            directory_path.mkdir(parents=True, exist_ok=True, mode=mode)
            
            self.logger.info(f"Created directory: {directory_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create directory {directory_path}: {str(e)}")
            return False
    
    def copy_file(self, source: Union[str, Path], 
                  destination: Union[str, Path],
                  preserve_metadata: bool = True) -> bool:
        """Copy file with validation and metadata preservation"""
        try:
            source = Path(source)
            destination = Path(destination)
            
            # Validate source file
            if not source.exists():
                self.logger.error(f"Source file does not exist: {source}")
                return False
            
            # Create destination directory if needed
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            if preserve_metadata:
                shutil.copy2(source, destination)
            else:
                shutil.copy(source, destination)
            
            # Verify copy
            if not destination.exists():
                self.logger.error(f"File copy verification failed: {destination}")
                return False
            
            # Compare checksums
            source_checksum = self.calculate_file_checksum(source)
            dest_checksum = self.calculate_file_checksum(destination)
            
            if source_checksum != dest_checksum:
                self.logger.error(f"Checksum mismatch after copy")
                destination.unlink()  # Remove corrupted copy
                return False
            
            self.logger.info(f"Successfully copied {source} to {destination}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to copy file {source} to {destination}: {str(e)}")
            return False
    
    def move_file(self, source: Union[str, Path], 
                  destination: Union[str, Path]) -> bool:
        """Move file with validation"""
        try:
            if self.copy_file(source, destination):
                Path(source).unlink()
                self.logger.info(f"Successfully moved {source} to {destination}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to move file {source} to {destination}: {str(e)}")
            return False
    
    def delete_file(self, filepath: Union[str, Path], 
                   secure_delete: bool = False) -> bool:
        """Delete file with optional secure deletion"""
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                self.logger.warning(f"File does not exist: {filepath}")
                return True
            
            if secure_delete:
                # Overwrite file content before deletion
                file_size = filepath.stat().st_size
                with open(filepath, 'wb') as f:
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            
            filepath.unlink()
            self.logger.info(f"Deleted file: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete file {filepath}: {str(e)}")
            return False
    
    def get_directory_info(self, directory_path: Union[str, Path]) -> Optional[DirectoryInfo]:
        """Get comprehensive directory information"""
        try:
            directory_path = Path(directory_path)
            
            if not directory_path.exists() or not directory_path.is_dir():
                return None
            
            total_files = 0
            total_size = 0
            subdirectories = []
            file_types = {}
            
            # Walk through directory
            for item in directory_path.rglob('*'):
                if item.is_file():
                    total_files += 1
                    total_size += item.stat().st_size
                    
                    # Count file types
                    ext = item.suffix.lower()
                    file_types[ext] = file_types.get(ext, 0) + 1
                    
                elif item.is_dir() and item != directory_path:
                    subdirectories.append(str(item.relative_to(directory_path)))
            
            # Get directory timestamps
            stat = directory_path.stat()
            created_at = datetime.fromtimestamp(stat.st_ctime)
            modified_at = datetime.fromtimestamp(stat.st_mtime)
            
            return DirectoryInfo(
                path=str(directory_path),
                total_files=total_files,
                total_size=total_size,
                subdirectories=subdirectories,
                file_types=file_types,
                created_at=created_at,
                modified_at=modified_at
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get directory info for {directory_path}: {str(e)}")
            return None
    
    def cleanup_old_files(self, directory_path: Union[str, Path], 
                         max_age_days: int = 30,
                         pattern: str = "*") -> int:
        """Clean up old files based on age"""
        try:
            directory_path = Path(directory_path)
            
            if not directory_path.exists():
                return 0
            
            cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)
            deleted_count = 0
            
            for filepath in directory_path.glob(pattern):
                if filepath.is_file():
                    file_mtime = filepath.stat().st_mtime
                    
                    if file_mtime < cutoff_time:
                        if self.delete_file(filepath):
                            deleted_count += 1
            
            self.logger.info(f"Cleaned up {deleted_count} old files from {directory_path}")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old files in {directory_path}: {str(e)}")
            return 0
    
    def create_temp_file(self, suffix: str = "", prefix: str = "avatar_", 
                        directory: Optional[str] = None) -> Optional[str]:
        """Create temporary file and return path"""
        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix, 
                prefix=prefix, 
                dir=directory, 
                delete=False
            ) as tmp_file:
                temp_path = tmp_file.name
            
            self.logger.debug(f"Created temporary file: {temp_path}")
            return temp_path
            
        except Exception as e:
            self.logger.error(f"Failed to create temporary file: {str(e)}")
            return None
    
    def save_uploaded_file(self, file_data: bytes, filename: str, 
                          destination_dir: Union[str, Path]) -> Optional[str]:
        """Save uploaded file with validation"""
        try:
            destination_dir = Path(destination_dir)
            destination_dir.mkdir(parents=True, exist_ok=True)
            
            # Create secure filename
            safe_filename = self.create_secure_filename(filename)
            filepath = destination_dir / safe_filename
            
            # Ensure unique filename
            counter = 1
            while filepath.exists():
                name, ext = os.path.splitext(safe_filename)
                filepath = destination_dir / f"{name}_{counter}{ext}"
                counter += 1
            
            # Write file data
            with open(filepath, 'wb') as f:
                f.write(file_data)
            
            # Validate saved file
            validation = self.validate_file(filepath)
            if not validation.is_valid:
                self.delete_file(filepath)
                self.logger.error(f"Saved file failed validation: {validation.error_messages}")
                return None
            
            self.logger.info(f"Successfully saved uploaded file: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save uploaded file {filename}: {str(e)}")
            return None
    
    def get_disk_usage(self, path: Union[str, Path]) -> Dict[str, int]:
        """Get disk usage statistics"""
        try:
            usage = shutil.disk_usage(path)
            return {
                "total": usage.total,
                "used": usage.used,
                "free": usage.free,
                "percent_used": (usage.used / usage.total) * 100
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get disk usage for {path}: {str(e)}")
            return {"total": 0, "used": 0, "free": 0, "percent_used": 0}
    
    def backup_file(self, filepath: Union[str, Path], 
                   backup_dir: Optional[Union[str, Path]] = None) -> Optional[str]:
        """Create backup copy of file"""
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                return None
            
            # Determine backup directory
            if backup_dir:
                backup_dir = Path(backup_dir)
            else:
                backup_dir = filepath.parent / "backups"
            
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name, ext = os.path.splitext(filepath.name)
            backup_filename = f"{name}_backup_{timestamp}{ext}"
            backup_path = backup_dir / backup_filename
            
            # Copy file to backup location
            if self.copy_file(filepath, backup_path):
                self.logger.info(f"Created backup: {backup_path}")
                return str(backup_path)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to backup file {filepath}: {str(e)}")
            return None


class ConfigManager:
    """Configuration file management"""
    
    def __init__(self, config_dir: str = "/app/config"):
        self.logger = logging.getLogger(__name__)
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def save_config(self, config_name: str, config_data: Dict, 
                   backup_existing: bool = True) -> bool:
        """Save configuration data to file"""
        try:
            config_path = self.config_dir / f"{config_name}.json"
            
            # Backup existing config if requested
            if backup_existing and config_path.exists():
                file_manager = FileManager()
                file_manager.backup_file(config_path)
            
            # Save new configuration
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            self.logger.info(f"Saved configuration: {config_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save config {config_name}: {str(e)}")
            return False
    
    def load_config(self, config_name: str) -> Optional[Dict]:
        """Load configuration data from file"""
        try:
            config_path = self.config_dir / f"{config_name}.json"
            
            if not config_path.exists():
                self.logger.warning(f"Configuration file not found: {config_name}")
                return None
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            self.logger.info(f"Loaded configuration: {config_name}")
            return config_data
            
        except Exception as e:
            self.logger.error(f"Failed to load config {config_name}: {str(e)}")
            return None 