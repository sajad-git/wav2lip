#!/usr/bin/env python3
"""
Resource Cleanup Script for Avatar Streaming Service
Handles cleanup of temporary files, cached data, and resource management
"""

import os
import sys
import shutil
import sqlite3
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import psutil
import gc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResourceCleanup:
    """Handles comprehensive resource cleanup for the avatar service"""
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path(__file__).parent.parent
        self.cleanup_stats = {
            'files_deleted': 0,
            'directories_deleted': 0,
            'space_freed_mb': 0,
            'errors': []
        }
    
    def cleanup_all(self, max_age_days: int = 7, dry_run: bool = False) -> Dict:
        """Perform comprehensive cleanup of all resources"""
        logger.info(f"Starting comprehensive cleanup (dry_run={dry_run})")
        
        cleanup_operations = [
            self.cleanup_temp_files,
            self.cleanup_logs,
            self.cleanup_avatar_cache,
            self.cleanup_model_cache,
            self.cleanup_processing_temp,
            self.cleanup_database_temp,
            self.cleanup_orphaned_files,
        ]
        
        for operation in cleanup_operations:
            try:
                operation(max_age_days, dry_run)
            except Exception as e:
                error_msg = f"Error in {operation.__name__}: {e}"
                logger.error(error_msg)
                self.cleanup_stats['errors'].append(error_msg)
        
        # Memory cleanup
        if not dry_run:
            self.cleanup_memory()
        
        return self.cleanup_stats
    
    def cleanup_temp_files(self, max_age_days: int, dry_run: bool) -> None:
        """Clean up temporary files and directories"""
        logger.info("Cleaning up temporary files...")
        
        temp_dirs = [
            self.base_path / 'temp',
            self.base_path / 'tmp',
            Path('/tmp/avatar_service'),  # System temp
        ]
        
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                self._cleanup_directory(temp_dir, max_age_days, dry_run, "temp files")
    
    def cleanup_logs(self, max_age_days: int, dry_run: bool) -> None:
        """Clean up old log files"""
        logger.info("Cleaning up log files...")
        
        logs_dir = self.base_path / 'logs'
        if not logs_dir.exists():
            return
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        for log_file in logs_dir.glob('*.log*'):
            try:
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_date:
                    size_mb = log_file.stat().st_size / (1024 * 1024)
                    
                    if dry_run:
                        logger.info(f"Would delete log file: {log_file} ({size_mb:.2f} MB)")
                    else:
                        log_file.unlink()
                        logger.info(f"Deleted log file: {log_file} ({size_mb:.2f} MB)")
                        self.cleanup_stats['files_deleted'] += 1
                        self.cleanup_stats['space_freed_mb'] += size_mb
            
            except Exception as e:
                error_msg = f"Error cleaning log file {log_file}: {e}"
                logger.error(error_msg)
                self.cleanup_stats['errors'].append(error_msg)
    
    def cleanup_avatar_cache(self, max_age_days: int, dry_run: bool) -> None:
        """Clean up avatar face detection cache"""
        logger.info("Cleaning up avatar cache...")
        
        cache_dir = self.base_path / 'data' / 'avatar_registry' / 'face_cache'
        if not cache_dir.exists():
            return
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Clean old cache files
        for cache_file in cache_dir.glob('*.pkl'):
            try:
                file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_time < cutoff_date:
                    size_mb = cache_file.stat().st_size / (1024 * 1024)
                    
                    if dry_run:
                        logger.info(f"Would delete cache file: {cache_file} ({size_mb:.2f} MB)")
                    else:
                        cache_file.unlink()
                        logger.info(f"Deleted cache file: {cache_file} ({size_mb:.2f} MB)")
                        self.cleanup_stats['files_deleted'] += 1
                        self.cleanup_stats['space_freed_mb'] += size_mb
            
            except Exception as e:
                error_msg = f"Error cleaning cache file {cache_file}: {e}"
                logger.error(error_msg)
                self.cleanup_stats['errors'].append(error_msg)
    
    def cleanup_model_cache(self, max_age_days: int, dry_run: bool) -> None:
        """Clean up model cache and temporary model files"""
        logger.info("Cleaning up model cache...")
        
        model_temp_dirs = [
            self.base_path / 'assets' / 'models' / '.cache',
            self.base_path / 'model_cache',
            Path.home() / '.cache' / 'onnxruntime',
            Path.home() / '.cache' / 'insightface',
        ]
        
        for cache_dir in model_temp_dirs:
            if cache_dir.exists():
                self._cleanup_directory(cache_dir, max_age_days, dry_run, "model cache")
    
    def cleanup_processing_temp(self, max_age_days: int, dry_run: bool) -> None:
        """Clean up processing temporary files"""
        logger.info("Cleaning up processing temporary files...")
        
        processing_dirs = [
            self.base_path / 'temp' / 'processing',
            self.base_path / 'temp' / 'chunks',
            self.base_path / 'temp' / 'audio',
            self.base_path / 'temp' / 'video',
        ]
        
        for proc_dir in processing_dirs:
            if proc_dir.exists():
                self._cleanup_directory(proc_dir, max_age_days, dry_run, "processing temp")
    
    def cleanup_database_temp(self, max_age_days: int, dry_run: bool) -> None:
        """Clean up database temporary files and optimize database"""
        logger.info("Cleaning up database temporary files...")
        
        db_path = self.base_path / 'data' / 'avatar_registry' / 'avatars.db'
        if not db_path.exists():
            return
        
        try:
            # Clean up database temp files
            db_temp_files = [
                db_path.with_suffix('.db-wal'),
                db_path.with_suffix('.db-shm'),
                db_path.with_suffix('.db-journal'),
            ]
            
            for temp_file in db_temp_files:
                if temp_file.exists():
                    size_mb = temp_file.stat().st_size / (1024 * 1024)
                    
                    if dry_run:
                        logger.info(f"Would delete DB temp file: {temp_file} ({size_mb:.2f} MB)")
                    else:
                        temp_file.unlink()
                        logger.info(f"Deleted DB temp file: {temp_file} ({size_mb:.2f} MB)")
                        self.cleanup_stats['files_deleted'] += 1
                        self.cleanup_stats['space_freed_mb'] += size_mb
            
            # Vacuum database if not dry run
            if not dry_run:
                self._vacuum_database(db_path)
        
        except Exception as e:
            error_msg = f"Error cleaning database temp files: {e}"
            logger.error(error_msg)
            self.cleanup_stats['errors'].append(error_msg)
    
    def cleanup_orphaned_files(self, max_age_days: int, dry_run: bool) -> None:
        """Clean up orphaned avatar files with no database entry"""
        logger.info("Cleaning up orphaned avatar files...")
        
        avatars_dir = self.base_path / 'assets' / 'avatars' / 'registered'
        db_path = self.base_path / 'data' / 'avatar_registry' / 'avatars.db'
        
        if not avatars_dir.exists() or not db_path.exists():
            return
        
        try:
            # Get list of avatar IDs from database
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT avatar_id FROM avatars WHERE is_active = 1")
                active_avatars = {row[0] for row in cursor.fetchall()}
            
            # Check for orphaned directories
            for avatar_dir in avatars_dir.iterdir():
                if avatar_dir.is_dir() and avatar_dir.name not in active_avatars:
                    # Check if directory is old enough
                    dir_time = datetime.fromtimestamp(avatar_dir.stat().st_mtime)
                    cutoff_date = datetime.now() - timedelta(days=max_age_days)
                    
                    if dir_time < cutoff_date:
                        dir_size_mb = self._get_directory_size(avatar_dir) / (1024 * 1024)
                        
                        if dry_run:
                            logger.info(f"Would delete orphaned avatar: {avatar_dir} ({dir_size_mb:.2f} MB)")
                        else:
                            shutil.rmtree(avatar_dir)
                            logger.info(f"Deleted orphaned avatar: {avatar_dir} ({dir_size_mb:.2f} MB)")
                            self.cleanup_stats['directories_deleted'] += 1
                            self.cleanup_stats['space_freed_mb'] += dir_size_mb
        
        except Exception as e:
            error_msg = f"Error cleaning orphaned files: {e}"
            logger.error(error_msg)
            self.cleanup_stats['errors'].append(error_msg)
    
    def cleanup_memory(self) -> None:
        """Perform memory cleanup and garbage collection"""
        logger.info("Performing memory cleanup...")
        
        try:
            # Force garbage collection
            collected = gc.collect()
            logger.info(f"Garbage collector freed {collected} objects")
            
            # Get memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            logger.info(f"Current memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            error_msg = f"Error during memory cleanup: {e}"
            logger.error(error_msg)
            self.cleanup_stats['errors'].append(error_msg)
    
    def _cleanup_directory(self, directory: Path, max_age_days: int, dry_run: bool, description: str) -> None:
        """Clean up files in a directory based on age"""
        if not directory.exists():
            return
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        for item in directory.rglob('*'):
            if item.is_file():
                try:
                    file_time = datetime.fromtimestamp(item.stat().st_mtime)
                    if file_time < cutoff_date:
                        size_mb = item.stat().st_size / (1024 * 1024)
                        
                        if dry_run:
                            logger.info(f"Would delete {description}: {item} ({size_mb:.2f} MB)")
                        else:
                            item.unlink()
                            logger.info(f"Deleted {description}: {item} ({size_mb:.2f} MB)")
                            self.cleanup_stats['files_deleted'] += 1
                            self.cleanup_stats['space_freed_mb'] += size_mb
                
                except Exception as e:
                    error_msg = f"Error cleaning {item}: {e}"
                    logger.error(error_msg)
                    self.cleanup_stats['errors'].append(error_msg)
    
    def _vacuum_database(self, db_path: Path) -> None:
        """Vacuum SQLite database to reclaim space"""
        try:
            with sqlite3.connect(db_path) as conn:
                logger.info("Vacuuming database...")
                conn.execute("VACUUM")
                logger.info("Database vacuum completed")
        
        except Exception as e:
            error_msg = f"Error vacuuming database: {e}"
            logger.error(error_msg)
            self.cleanup_stats['errors'].append(error_msg)
    
    def _get_directory_size(self, directory: Path) -> int:
        """Calculate total size of directory"""
        total_size = 0
        try:
            for item in directory.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
        except Exception:
            pass
        return total_size
    
    def get_disk_usage(self) -> Dict[str, float]:
        """Get disk usage statistics"""
        try:
            disk_usage = psutil.disk_usage(str(self.base_path))
            return {
                'total_gb': disk_usage.total / (1024**3),
                'used_gb': disk_usage.used / (1024**3),
                'free_gb': disk_usage.free / (1024**3),
                'percent_used': (disk_usage.used / disk_usage.total) * 100
            }
        except Exception as e:
            logger.error(f"Error getting disk usage: {e}")
            return {}
    
    def emergency_cleanup(self) -> Dict:
        """Emergency cleanup when disk space is critically low"""
        logger.warning("Performing emergency cleanup...")
        
        # More aggressive cleanup with shorter retention
        emergency_stats = self.cleanup_all(max_age_days=1, dry_run=False)
        
        # Additional emergency measures
        try:
            # Clear all temporary processing files regardless of age
            temp_dirs = [
                self.base_path / 'temp',
                self.base_path / 'tmp',
            ]
            
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    temp_dir.mkdir(exist_ok=True)
                    logger.info(f"Emergency: Cleared all temp files in {temp_dir}")
            
            # Clear model cache
            model_cache_dirs = [
                Path.home() / '.cache' / 'onnxruntime',
                Path.home() / '.cache' / 'insightface',
            ]
            
            for cache_dir in model_cache_dirs:
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                    logger.info(f"Emergency: Cleared model cache {cache_dir}")
        
        except Exception as e:
            error_msg = f"Error during emergency cleanup: {e}"
            logger.error(error_msg)
            emergency_stats['errors'].append(error_msg)
        
        return emergency_stats

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Avatar Service Resource Cleanup")
    parser.add_argument('--max-age-days', type=int, default=7,
                        help='Maximum age of files to keep (default: 7 days)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be deleted without actually deleting')
    parser.add_argument('--emergency', action='store_true',
                        help='Perform emergency cleanup for low disk space')
    parser.add_argument('--base-path', type=str,
                        help='Base path for avatar service (default: auto-detect)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize cleanup
    cleanup = ResourceCleanup(args.base_path)
    
    # Show disk usage before cleanup
    disk_usage = cleanup.get_disk_usage()
    if disk_usage:
        logger.info(f"Disk usage before cleanup: {disk_usage['used_gb']:.2f}GB / {disk_usage['total_gb']:.2f}GB ({disk_usage['percent_used']:.1f}%)")
    
    # Perform cleanup
    if args.emergency:
        stats = cleanup.emergency_cleanup()
    else:
        stats = cleanup.cleanup_all(args.max_age_days, args.dry_run)
    
    # Show results
    logger.info("Cleanup completed!")
    logger.info(f"Files deleted: {stats['files_deleted']}")
    logger.info(f"Directories deleted: {stats['directories_deleted']}")
    logger.info(f"Space freed: {stats['space_freed_mb']:.2f} MB")
    
    if stats['errors']:
        logger.warning(f"Errors encountered: {len(stats['errors'])}")
        for error in stats['errors']:
            logger.error(f"  {error}")
    
    # Show disk usage after cleanup
    disk_usage_after = cleanup.get_disk_usage()
    if disk_usage_after:
        logger.info(f"Disk usage after cleanup: {disk_usage_after['used_gb']:.2f}GB / {disk_usage_after['total_gb']:.2f}GB ({disk_usage_after['percent_used']:.1f}%)")
    
    return 0 if not stats['errors'] else 1

if __name__ == "__main__":
    sys.exit(main()) 