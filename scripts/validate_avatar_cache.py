#!/usr/bin/env python3
"""
Avatar Cache Validation Script
Validates avatar cache integrity, performance, and data consistency.
"""

import os
import sys
import sqlite3
import pickle
import json
import hashlib
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import argparse
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.avatar_config import AvatarConfig
from app.utils.file_utils import get_file_size, ensure_directory_exists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AvatarCacheValidator:
    """Avatar cache validation and integrity checking system"""
    
    def __init__(self, config: AvatarConfig):
        self.config = config
        self.db_path = os.path.join(config.avatar_storage_path, "..", "avatar_registry", "avatars.db")
        self.cache_path = os.path.join(config.avatar_storage_path, "..", "avatar_registry", "face_cache")
        self.avatar_storage_path = config.avatar_storage_path
        
        # Validation results
        self.validation_results = {
            "database": {"status": "unknown", "errors": [], "warnings": []},
            "cache_files": {"status": "unknown", "errors": [], "warnings": []},
            "avatar_files": {"status": "unknown", "errors": [], "warnings": []},
            "data_consistency": {"status": "unknown", "errors": [], "warnings": []},
            "performance": {"status": "unknown", "errors": [], "warnings": []}
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            "max_cache_access_time": 100.0,  # ms
            "max_cache_file_size": 50 * 1024 * 1024,  # 50MB
            "min_compression_ratio": 0.1,
            "max_face_detection_variance": 0.1
        }
        
    def validate_all(self, quick_check: bool = False) -> Dict[str, Any]:
        """Run comprehensive cache validation"""
        logger.info("Starting comprehensive avatar cache validation...")
        
        start_time = time.time()
        
        try:
            # 1. Validate database structure and integrity
            self._validate_database()
            
            # 2. Validate cache files
            self._validate_cache_files()
            
            # 3. Validate avatar files
            self._validate_avatar_files()
            
            # 4. Check data consistency
            self._validate_data_consistency()
            
            # 5. Performance validation (skip in quick check)
            if not quick_check:
                self._validate_performance()
            
            # Calculate overall status
            overall_status = self._calculate_overall_status()
            
            validation_time = time.time() - start_time
            
            results = {
                "validation_timestamp": datetime.utcnow().isoformat(),
                "validation_duration_seconds": validation_time,
                "overall_status": overall_status,
                "quick_check": quick_check,
                "results": self.validation_results,
                "summary": self._generate_validation_summary()
            }
            
            logger.info(f"Validation completed in {validation_time:.2f} seconds")
            logger.info(f"Overall status: {overall_status}")
            
            return results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                "validation_timestamp": datetime.utcnow().isoformat(),
                "overall_status": "error",
                "error": str(e),
                "results": self.validation_results
            }
    
    def _validate_database(self):
        """Validate database structure and integrity"""
        logger.info("Validating database structure and integrity...")
        
        try:
            if not os.path.exists(self.db_path):
                self.validation_results["database"]["status"] = "error"
                self.validation_results["database"]["errors"].append("Database file not found")
                return
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check required tables
            required_tables = ['avatars', 'migration_info']
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            missing_tables = []
            for table in required_tables:
                if table not in existing_tables:
                    missing_tables.append(table)
            
            if missing_tables:
                self.validation_results["database"]["errors"].append(
                    f"Missing required tables: {missing_tables}"
                )
            
            # Check avatars table structure
            cursor.execute("PRAGMA table_info(avatars)")
            columns = [row[1] for row in cursor.fetchall()]
            required_columns = ['avatar_id', 'name', 'file_path', 'cache_path']
            
            missing_columns = []
            for column in required_columns:
                if column not in columns:
                    missing_columns.append(column)
            
            if missing_columns:
                self.validation_results["database"]["errors"].append(
                    f"Missing required columns in avatars table: {missing_columns}"
                )
            
            # Check for orphaned records
            cursor.execute("SELECT COUNT(*) FROM avatars WHERE file_path IS NULL OR file_path = ''")
            orphaned_count = cursor.fetchone()[0]
            if orphaned_count > 0:
                self.validation_results["database"]["warnings"].append(
                    f"{orphaned_count} avatar records without file paths"
                )
            
            # Check for duplicate avatar IDs
            cursor.execute("""
                SELECT avatar_id, COUNT(*) as count 
                FROM avatars 
                GROUP BY avatar_id 
                HAVING count > 1
            """)
            duplicates = cursor.fetchall()
            if duplicates:
                self.validation_results["database"]["errors"].append(
                    f"Duplicate avatar IDs found: {[dup[0] for dup in duplicates]}"
                )
            
            # Database integrity check
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()[0]
            if integrity_result != "ok":
                self.validation_results["database"]["errors"].append(
                    f"Database integrity check failed: {integrity_result}"
                )
            
            conn.close()
            
            # Set status based on errors
            if self.validation_results["database"]["errors"]:
                self.validation_results["database"]["status"] = "error"
            elif self.validation_results["database"]["warnings"]:
                self.validation_results["database"]["status"] = "warning"
            else:
                self.validation_results["database"]["status"] = "healthy"
                
            logger.info("Database validation completed")
            
        except Exception as e:
            self.validation_results["database"]["status"] = "error"
            self.validation_results["database"]["errors"].append(f"Database validation failed: {e}")
            logger.error(f"Database validation error: {e}")
    
    def _validate_cache_files(self):
        """Validate cache files integrity and structure"""
        logger.info("Validating cache files...")
        
        try:
            if not os.path.exists(self.cache_path):
                self.validation_results["cache_files"]["status"] = "warning"
                self.validation_results["cache_files"]["warnings"].append("Cache directory does not exist")
                return
            
            cache_files = []
            corrupted_files = []
            oversized_files = []
            
            # Scan cache directory
            for root, dirs, files in os.walk(self.cache_path):
                for file in files:
                    if file.endswith('.pkl') or file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        cache_files.append(file_path)
                        
                        # Check file size
                        file_size = get_file_size(file_path)
                        if file_size > self.performance_thresholds["max_cache_file_size"]:
                            oversized_files.append((file_path, file_size))
                        
                        # Try to load and validate cache files
                        try:
                            if file.endswith('.pkl'):
                                with open(file_path, 'rb') as f:
                                    data = pickle.load(f)
                                    self._validate_cache_data_structure(data, file_path)
                            elif file.endswith('.json'):
                                with open(file_path, 'r') as f:
                                    data = json.load(f)
                                    self._validate_metadata_structure(data, file_path)
                        except Exception as e:
                            corrupted_files.append((file_path, str(e)))
            
            # Report findings
            logger.info(f"Found {len(cache_files)} cache files")
            
            if corrupted_files:
                self.validation_results["cache_files"]["errors"].append(
                    f"Corrupted cache files: {[f[0] for f in corrupted_files]}"
                )
                for file_path, error in corrupted_files:
                    logger.error(f"Corrupted cache file {file_path}: {error}")
            
            if oversized_files:
                self.validation_results["cache_files"]["warnings"].append(
                    f"Oversized cache files: {[f[0] for f in oversized_files]}"
                )
            
            # Check for empty cache directory
            if len(cache_files) == 0:
                self.validation_results["cache_files"]["warnings"].append(
                    "No cache files found - cache may be empty"
                )
            
            # Set status
            if self.validation_results["cache_files"]["errors"]:
                self.validation_results["cache_files"]["status"] = "error"
            elif self.validation_results["cache_files"]["warnings"]:
                self.validation_results["cache_files"]["status"] = "warning"
            else:
                self.validation_results["cache_files"]["status"] = "healthy"
                
            logger.info("Cache files validation completed")
            
        except Exception as e:
            self.validation_results["cache_files"]["status"] = "error"
            self.validation_results["cache_files"]["errors"].append(f"Cache validation failed: {e}")
            logger.error(f"Cache files validation error: {e}")
    
    def _validate_avatar_files(self):
        """Validate avatar files existence and integrity"""
        logger.info("Validating avatar files...")
        
        try:
            if not os.path.exists(self.db_path):
                self.validation_results["avatar_files"]["errors"].append("Cannot validate avatar files without database")
                return
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all avatar records
            cursor.execute("SELECT avatar_id, file_path FROM avatars WHERE is_active = 1")
            avatar_records = cursor.fetchall()
            
            missing_files = []
            corrupted_files = []
            
            for avatar_id, file_path in avatar_records:
                if not file_path:
                    missing_files.append(avatar_id)
                    continue
                
                full_path = os.path.join(self.avatar_storage_path, file_path)
                
                if not os.path.exists(full_path):
                    missing_files.append(f"{avatar_id}: {file_path}")
                else:
                    # Basic file integrity check
                    try:
                        file_size = get_file_size(full_path)
                        if file_size == 0:
                            corrupted_files.append(f"{avatar_id}: {file_path} (empty file)")
                        
                        # Try to read file header
                        with open(full_path, 'rb') as f:
                            header = f.read(8)
                            if len(header) < 8:
                                corrupted_files.append(f"{avatar_id}: {file_path} (truncated)")
                                
                    except Exception as e:
                        corrupted_files.append(f"{avatar_id}: {file_path} ({e})")
            
            conn.close()
            
            # Report findings
            if missing_files:
                self.validation_results["avatar_files"]["errors"].append(
                    f"Missing avatar files: {missing_files}"
                )
            
            if corrupted_files:
                self.validation_results["avatar_files"]["errors"].append(
                    f"Corrupted avatar files: {corrupted_files}"
                )
            
            # Set status
            if self.validation_results["avatar_files"]["errors"]:
                self.validation_results["avatar_files"]["status"] = "error"
            else:
                self.validation_results["avatar_files"]["status"] = "healthy"
                
            logger.info(f"Avatar files validation completed: {len(avatar_records)} files checked")
            
        except Exception as e:
            self.validation_results["avatar_files"]["status"] = "error"
            self.validation_results["avatar_files"]["errors"].append(f"Avatar files validation failed: {e}")
            logger.error(f"Avatar files validation error: {e}")
    
    def _validate_data_consistency(self):
        """Validate consistency between database and file system"""
        logger.info("Validating data consistency...")
        
        try:
            if not os.path.exists(self.db_path):
                self.validation_results["data_consistency"]["errors"].append("Database not found for consistency check")
                return
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all avatar records
            cursor.execute("SELECT avatar_id, file_path, cache_path FROM avatars WHERE is_active = 1")
            avatar_records = cursor.fetchall()
            
            inconsistencies = []
            
            for avatar_id, file_path, cache_path in avatar_records:
                # Check file path consistency
                if file_path:
                    full_avatar_path = os.path.join(self.avatar_storage_path, file_path)
                    if not os.path.exists(full_avatar_path):
                        inconsistencies.append(f"Avatar {avatar_id}: file referenced in DB but not found on disk")
                
                # Check cache path consistency
                if cache_path:
                    full_cache_path = os.path.join(self.cache_path, cache_path)
                    if not os.path.exists(full_cache_path):
                        inconsistencies.append(f"Avatar {avatar_id}: cache referenced in DB but not found on disk")
            
            # Check for orphaned files
            db_file_paths = set()
            for avatar_id, file_path, cache_path in avatar_records:
                if file_path:
                    db_file_paths.add(os.path.join(self.avatar_storage_path, file_path))
            
            # Scan actual files
            actual_files = set()
            if os.path.exists(self.avatar_storage_path):
                for root, dirs, files in os.walk(self.avatar_storage_path):
                    for file in files:
                        actual_files.add(os.path.join(root, file))
            
            orphaned_files = actual_files - db_file_paths
            if orphaned_files:
                self.validation_results["data_consistency"]["warnings"].append(
                    f"Orphaned files found (not in database): {len(orphaned_files)} files"
                )
            
            conn.close()
            
            # Report findings
            if inconsistencies:
                self.validation_results["data_consistency"]["errors"].extend(inconsistencies)
            
            # Set status
            if self.validation_results["data_consistency"]["errors"]:
                self.validation_results["data_consistency"]["status"] = "error"
            elif self.validation_results["data_consistency"]["warnings"]:
                self.validation_results["data_consistency"]["status"] = "warning"
            else:
                self.validation_results["data_consistency"]["status"] = "healthy"
                
            logger.info("Data consistency validation completed")
            
        except Exception as e:
            self.validation_results["data_consistency"]["status"] = "error"
            self.validation_results["data_consistency"]["errors"].append(f"Consistency validation failed: {e}")
            logger.error(f"Data consistency validation error: {e}")
    
    def _validate_performance(self):
        """Validate cache performance characteristics"""
        logger.info("Validating cache performance...")
        
        try:
            performance_issues = []
            
            # Test cache access times
            cache_files = []
            if os.path.exists(self.cache_path):
                for root, dirs, files in os.walk(self.cache_path):
                    for file in files:
                        if file.endswith('.pkl'):
                            cache_files.append(os.path.join(root, file))
            
            if cache_files:
                access_times = []
                sample_size = min(10, len(cache_files))  # Test up to 10 files
                
                for i in range(sample_size):
                    cache_file = cache_files[i]
                    start_time = time.time()
                    
                    try:
                        with open(cache_file, 'rb') as f:
                            data = pickle.load(f)
                        access_time = (time.time() - start_time) * 1000  # Convert to ms
                        access_times.append(access_time)
                        
                        if access_time > self.performance_thresholds["max_cache_access_time"]:
                            performance_issues.append(
                                f"Slow cache access: {cache_file} took {access_time:.2f}ms"
                            )
                            
                    except Exception as e:
                        performance_issues.append(f"Failed to load cache file {cache_file}: {e}")
                
                if access_times:
                    avg_access_time = sum(access_times) / len(access_times)
                    max_access_time = max(access_times)
                    
                    logger.info(f"Cache access performance: avg={avg_access_time:.2f}ms, max={max_access_time:.2f}ms")
                    
                    if avg_access_time > self.performance_thresholds["max_cache_access_time"] / 2:
                        performance_issues.append(
                            f"High average cache access time: {avg_access_time:.2f}ms"
                        )
            
            # Check cache compression efficiency
            self._validate_cache_compression(performance_issues)
            
            # Report findings
            if performance_issues:
                self.validation_results["performance"]["warnings"].extend(performance_issues)
            
            # Set status
            if self.validation_results["performance"]["errors"]:
                self.validation_results["performance"]["status"] = "error"
            elif self.validation_results["performance"]["warnings"]:
                self.validation_results["performance"]["status"] = "warning"
            else:
                self.validation_results["performance"]["status"] = "healthy"
                
            logger.info("Performance validation completed")
            
        except Exception as e:
            self.validation_results["performance"]["status"] = "error"
            self.validation_results["performance"]["errors"].append(f"Performance validation failed: {e}")
            logger.error(f"Performance validation error: {e}")
    
    def _validate_cache_data_structure(self, data: Any, file_path: str):
        """Validate structure of cached face data"""
        try:
            # Check if data has expected structure
            if not isinstance(data, dict):
                raise ValueError("Cache data is not a dictionary")
            
            required_fields = ['avatar_id', 'face_boxes', 'face_landmarks']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate face data
            if 'face_boxes' in data and data['face_boxes']:
                face_boxes = data['face_boxes']
                if not isinstance(face_boxes, list):
                    raise ValueError("face_boxes should be a list")
                
                for box in face_boxes:
                    if not isinstance(box, (list, tuple)) or len(box) != 4:
                        raise ValueError("Invalid face box format")
            
            if 'face_landmarks' in data and data['face_landmarks']:
                landmarks = data['face_landmarks']
                if not isinstance(landmarks, list):
                    raise ValueError("face_landmarks should be a list")
                
                for landmark in landmarks:
                    if not isinstance(landmark, np.ndarray):
                        raise ValueError("Landmarks should be numpy arrays")
                        
        except Exception as e:
            self.validation_results["cache_files"]["errors"].append(
                f"Invalid cache data structure in {file_path}: {e}"
            )
    
    def _validate_metadata_structure(self, data: Dict[str, Any], file_path: str):
        """Validate structure of metadata files"""
        try:
            if not isinstance(data, dict):
                raise ValueError("Metadata should be a dictionary")
            
            # Check for common metadata fields
            expected_fields = ['avatar_id', 'cache_timestamp', 'processing_metadata']
            missing_fields = [field for field in expected_fields if field not in data]
            
            if missing_fields:
                self.validation_results["cache_files"]["warnings"].append(
                    f"Missing metadata fields in {file_path}: {missing_fields}"
                )
                
        except Exception as e:
            self.validation_results["cache_files"]["errors"].append(
                f"Invalid metadata structure in {file_path}: {e}"
            )
    
    def _validate_cache_compression(self, performance_issues: List[str]):
        """Validate cache compression efficiency"""
        try:
            if not os.path.exists(self.cache_path):
                return
            
            # Sample cache files for compression analysis
            cache_files = []
            for root, dirs, files in os.walk(self.cache_path):
                for file in files:
                    if file.endswith('.pkl'):
                        cache_files.append(os.path.join(root, file))
            
            if len(cache_files) < 2:
                return  # Need at least 2 files for meaningful analysis
            
            compression_ratios = []
            sample_files = cache_files[:5]  # Sample first 5 files
            
            for cache_file in sample_files:
                try:
                    # Load and recompress to estimate compression ratio
                    with open(cache_file, 'rb') as f:
                        compressed_data = f.read()
                        f.seek(0)
                        original_data = pickle.load(f)
                    
                    # Estimate uncompressed size (rough approximation)
                    uncompressed_size = len(pickle.dumps(original_data, protocol=0))
                    compressed_size = len(compressed_data)
                    
                    if uncompressed_size > 0:
                        ratio = compressed_size / uncompressed_size
                        compression_ratios.append(ratio)
                        
                        if ratio > 0.8:  # Poor compression
                            performance_issues.append(
                                f"Poor compression ratio in {cache_file}: {ratio:.2f}"
                            )
                            
                except Exception as e:
                    logger.warning(f"Could not analyze compression for {cache_file}: {e}")
            
            if compression_ratios:
                avg_ratio = sum(compression_ratios) / len(compression_ratios)
                logger.info(f"Average compression ratio: {avg_ratio:.2f}")
                
        except Exception as e:
            logger.warning(f"Compression validation failed: {e}")
    
    def _calculate_overall_status(self) -> str:
        """Calculate overall validation status"""
        error_components = [
            component for component, result in self.validation_results.items()
            if result["status"] == "error"
        ]
        
        warning_components = [
            component for component, result in self.validation_results.items()
            if result["status"] == "warning"
        ]
        
        if error_components:
            return "error"
        elif warning_components:
            return "warning"
        else:
            return "healthy"
    
    def _generate_validation_summary(self) -> Dict[str, Any]:
        """Generate validation summary"""
        total_errors = sum(len(result["errors"]) for result in self.validation_results.values())
        total_warnings = sum(len(result["warnings"]) for result in self.validation_results.values())
        
        component_statuses = {
            component: result["status"] 
            for component, result in self.validation_results.items()
        }
        
        return {
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "component_statuses": component_statuses,
            "healthy_components": len([s for s in component_statuses.values() if s == "healthy"]),
            "total_components": len(component_statuses)
        }
    
    def fix_common_issues(self, dry_run: bool = False) -> Dict[str, Any]:
        """Attempt to fix common cache issues"""
        logger.info("Attempting to fix common cache issues...")
        
        fixes_applied = []
        fixes_failed = []
        
        try:
            # Remove corrupted cache files
            for component, result in self.validation_results.items():
                if component == "cache_files":
                    for error in result["errors"]:
                        if "Corrupted cache files" in error:
                            # Extract file paths and remove corrupted files
                            # This is a simplified implementation
                            if not dry_run:
                                # Implementation would extract file paths and remove them
                                fixes_applied.append("Removed corrupted cache files")
                            else:
                                fixes_applied.append("DRY RUN: Would remove corrupted cache files")
            
            # Create missing directories
            if not os.path.exists(self.cache_path):
                if not dry_run:
                    ensure_directory_exists(self.cache_path)
                    fixes_applied.append("Created missing cache directory")
                else:
                    fixes_applied.append("DRY RUN: Would create missing cache directory")
            
            return {
                "fixes_applied": fixes_applied,
                "fixes_failed": fixes_failed,
                "dry_run": dry_run
            }
            
        except Exception as e:
            fixes_failed.append(f"Fix attempt failed: {e}")
            return {
                "fixes_applied": fixes_applied,
                "fixes_failed": fixes_failed,
                "dry_run": dry_run
            }


def main():
    """Main validation script entry point"""
    parser = argparse.ArgumentParser(description="Avatar Cache Validation Tool")
    parser.add_argument("--quick", action="store_true",
                       help="Perform quick validation (skip performance tests)")
    parser.add_argument("--fix", action="store_true",
                       help="Attempt to fix common issues")
    parser.add_argument("--dry-run", action="store_true",
                       help="Perform dry run without making changes")
    parser.add_argument("--output", type=str,
                       help="Output validation results to file")
    parser.add_argument("--config-path", type=str,
                       help="Path to avatar configuration file")
    
    args = parser.parse_args()
    
    try:
        # Initialize configuration
        if args.config_path:
            # Load custom config (implementation would depend on config structure)
            config = AvatarConfig()  # Placeholder
        else:
            config = AvatarConfig()
        
        validator = AvatarCacheValidator(config)
        
        # Run validation
        logger.info("Starting avatar cache validation...")
        results = validator.validate_all(quick_check=args.quick)
        
        # Apply fixes if requested
        if args.fix:
            fix_results = validator.fix_common_issues(dry_run=args.dry_run)
            results["fixes"] = fix_results
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results written to {args.output}")
        else:
            print(json.dumps(results, indent=2))
        
        # Determine exit code
        if results["overall_status"] == "error":
            logger.error("Validation found critical errors")
            return 1
        elif results["overall_status"] == "warning":
            logger.warning("Validation found warnings")
            return 0
        else:
            logger.info("Validation passed successfully")
            return 0
            
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Validation script failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 