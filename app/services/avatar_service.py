"""
Avatar Management Service
Handles avatar registration, validation, and caching with metadata management.
"""

import os
import uuid
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from ..core.avatar_registrar import ColdAvatarRegistrar, AvatarRegistrationResult
from ..core.face_cache_manager import FaceCacheManager, WarmupReport
from ..config.avatar_config import AvatarConfig

logger = logging.getLogger(__name__)

@dataclass
class AvatarInfo:
    """Avatar information for management and display."""
    avatar_id: str
    name: str
    file_format: str
    file_size: int
    resolution: tuple[int, int]
    frame_count: int
    registration_date: datetime
    last_used: Optional[datetime]
    usage_count: int
    face_quality_score: float
    processing_ready: bool
    cache_size: int
    owner_id: str

@dataclass
class ProcessingStats:
    """Avatar usage and processing statistics."""
    total_uses: int
    last_used: Optional[datetime]
    avg_processing_time: float
    cache_hit_rate: float
    total_processing_time: float

@dataclass
class AvatarRegistrationResponse:
    """Response for avatar registration request."""
    avatar_id: str
    registration_status: str
    face_detection_results: Dict[str, Any]
    quality_assessment: Dict[str, Any]
    processing_time: float
    cache_status: str
    errors: List[str]
    warnings: List[str]

class AvatarDatabase:
    """Database operations for avatar metadata."""
    
    def __init__(self, database_path: str):
        """
        Initialize avatar database.
        
        Args:
            database_path: Path to SQLite database file
        """
        self.database_path = database_path
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Ensure database and tables exist."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Check if tables exist
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='avatars'
                """)
                
                if not cursor.fetchone():
                    # Database needs initialization - this should be handled by avatar_registrar
                    logger.warning("Avatar database not found, will be created on first registration")
                    
        except Exception as e:
            logger.error(f"Failed to check avatar database: {e}")
            raise
    
    def get_avatar_list(self, owner_id: Optional[str] = None) -> List[AvatarInfo]:
        """
        Get list of avatars, optionally filtered by owner.
        
        Args:
            owner_id: Optional owner filter
            
        Returns:
            List of avatar information
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                if owner_id:
                    cursor.execute("""
                        SELECT avatar_id, name, file_format, file_size, 
                               resolution_width, resolution_height, frame_count,
                               registration_date, last_accessed, access_count,
                               face_quality_score, owner_id, is_active
                        FROM avatars 
                        WHERE owner_id = ? AND is_active = 1
                        ORDER BY registration_date DESC
                    """, (owner_id,))
                else:
                    cursor.execute("""
                        SELECT avatar_id, name, file_format, file_size, 
                               resolution_width, resolution_height, frame_count,
                               registration_date, last_accessed, access_count,
                               face_quality_score, owner_id, is_active
                        FROM avatars 
                        WHERE is_active = 1
                        ORDER BY registration_date DESC
                    """)
                
                avatars = []
                for row in cursor.fetchall():
                    avatar_info = AvatarInfo(
                        avatar_id=row[0],
                        name=row[1],
                        file_format=row[2],
                        file_size=row[3],
                        resolution=(row[4], row[5]),
                        frame_count=row[6],
                        registration_date=datetime.fromisoformat(row[7]),
                        last_used=datetime.fromisoformat(row[8]) if row[8] else None,
                        usage_count=row[9],
                        face_quality_score=row[10],
                        processing_ready=row[10] > 0.5,  # Based on quality score
                        cache_size=0,  # Will be filled by service
                        owner_id=row[11]
                    )
                    avatars.append(avatar_info)
                
                logger.info(f"Retrieved {len(avatars)} avatars from database")
                return avatars
                
        except Exception as e:
            logger.error(f"Failed to get avatar list: {e}")
            return []
    
    def get_avatar_info(self, avatar_id: str) -> Optional[AvatarInfo]:
        """
        Get detailed information for specific avatar.
        
        Args:
            avatar_id: Avatar identifier
            
        Returns:
            Avatar information or None if not found
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT avatar_id, name, file_format, file_size, 
                           resolution_width, resolution_height, frame_count,
                           registration_date, last_accessed, access_count,
                           face_quality_score, owner_id, is_active, metadata_json
                    FROM avatars 
                    WHERE avatar_id = ? AND is_active = 1
                """, (avatar_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                avatar_info = AvatarInfo(
                    avatar_id=row[0],
                    name=row[1],
                    file_format=row[2],
                    file_size=row[3],
                    resolution=(row[4], row[5]),
                    frame_count=row[6],
                    registration_date=datetime.fromisoformat(row[7]),
                    last_used=datetime.fromisoformat(row[8]) if row[8] else None,
                    usage_count=row[9],
                    face_quality_score=row[10],
                    processing_ready=row[10] > 0.5,
                    cache_size=0,  # Will be filled by service
                    owner_id=row[11]
                )
                
                return avatar_info
                
        except Exception as e:
            logger.error(f"Failed to get avatar info for {avatar_id}: {e}")
            return None
    
    def update_avatar_usage(self, avatar_id: str):
        """Update avatar usage statistics."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE avatars 
                    SET access_count = access_count + 1,
                        last_accessed = ?
                    WHERE avatar_id = ?
                """, (datetime.now().isoformat(), avatar_id))
                
                conn.commit()
                logger.debug(f"Updated usage stats for avatar {avatar_id}")
                
        except Exception as e:
            logger.error(f"Failed to update usage for {avatar_id}: {e}")
    
    def delete_avatar(self, avatar_id: str, owner_id: str) -> bool:
        """
        Delete avatar from database (soft delete).
        
        Args:
            avatar_id: Avatar identifier
            owner_id: Owner verification
            
        Returns:
            bool: Deletion success status
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Verify ownership
                cursor.execute("""
                    SELECT owner_id FROM avatars 
                    WHERE avatar_id = ? AND is_active = 1
                """, (avatar_id,))
                
                row = cursor.fetchone()
                if not row:
                    logger.warning(f"Avatar {avatar_id} not found for deletion")
                    return False
                
                if row[0] != owner_id:
                    logger.warning(f"Ownership verification failed for avatar {avatar_id}")
                    return False
                
                # Soft delete
                cursor.execute("""
                    UPDATE avatars 
                    SET is_active = 0
                    WHERE avatar_id = ?
                """, (avatar_id,))
                
                conn.commit()
                logger.info(f"Avatar {avatar_id} deleted successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete avatar {avatar_id}: {e}")
            return False

class AvatarManagementService:
    """Main avatar management service."""
    
    def __init__(
        self,
        avatar_config: AvatarConfig,
        avatar_registrar: ColdAvatarRegistrar,
        face_cache_manager: FaceCacheManager
    ):
        """
        Initialize avatar management service.
        
        Args:
            avatar_config: Avatar configuration
            avatar_registrar: Avatar registration engine
            face_cache_manager: Face data cache manager
        """
        self.config = avatar_config
        self.avatar_registrar = avatar_registrar
        self.face_cache_manager = face_cache_manager
        
        # Initialize database
        database_path = os.path.join(avatar_config.cache_storage_path, "../avatars.db")
        self.avatar_database = AvatarDatabase(database_path)
        
        logger.info("Avatar management service initialized")
    
    async def register_new_avatar(
        self,
        file_data: bytes,
        filename: str,
        user_id: str,
        avatar_name: str,
        description: Optional[str] = None
    ) -> AvatarRegistrationResponse:
        """
        Complete avatar registration workflow.
        
        Args:
            file_data: Avatar file data
            filename: Original filename
            user_id: User identifier
            avatar_name: User-friendly name
            description: Optional description
            
        Returns:
            AvatarRegistrationResponse: Registration results
        """
        try:
            # Generate unique avatar ID
            avatar_id = self._generate_avatar_id(user_id, avatar_name)
            
            # Determine file format
            file_format = self._extract_file_format(filename)
            if not file_format:
                return AvatarRegistrationResponse(
                    avatar_id=avatar_id,
                    registration_status="failed",
                    face_detection_results={},
                    quality_assessment={},
                    processing_time=0.0,
                    cache_status="not_cached",
                    errors=["Unsupported file format"],
                    warnings=[]
                )
            
            # Validate file size
            if len(file_data) > self.config.max_file_size:
                return AvatarRegistrationResponse(
                    avatar_id=avatar_id,
                    registration_status="failed",
                    face_detection_results={},
                    quality_assessment={},
                    processing_time=0.0,
                    cache_status="not_cached",
                    errors=[f"File size exceeds limit of {self.config.max_file_size} bytes"],
                    warnings=[]
                )
            
            # Register avatar with face processing
            registration_result = await self.avatar_registrar.register_avatar(
                file_data=file_data,
                avatar_id=avatar_id,
                file_format=file_format,
                avatar_name=avatar_name,
                owner_id=user_id
            )
            
            # Convert to response format
            response = AvatarRegistrationResponse(
                avatar_id=registration_result.avatar_id,
                registration_status=registration_result.registration_status,
                face_detection_results={
                    "faces_detected": registration_result.face_detection_summary.faces_detected,
                    "primary_confidence": registration_result.face_detection_summary.primary_face_confidence,
                    "consistency_score": registration_result.face_detection_summary.face_consistency_score,
                    "quality_metrics": registration_result.face_detection_summary.quality_metrics
                },
                quality_assessment={
                    "overall_score": registration_result.quality_assessment.overall_score,
                    "processing_ready": registration_result.quality_assessment.processing_ready,
                    "recommendations": registration_result.quality_assessment.recommendations
                },
                processing_time=registration_result.processing_time,
                cache_status=registration_result.cache_status,
                errors=registration_result.errors,
                warnings=registration_result.warnings
            )
            
            logger.info(f"Avatar registration completed: {avatar_id} ({registration_result.registration_status})")
            return response
            
        except Exception as e:
            logger.error(f"Avatar registration failed: {e}")
            return AvatarRegistrationResponse(
                avatar_id=avatar_id if 'avatar_id' in locals() else "unknown",
                registration_status="failed",
                face_detection_results={},
                quality_assessment={},
                processing_time=0.0,
                cache_status="not_cached",
                errors=[f"Registration failed: {str(e)}"],
                warnings=[]
            )
    
    def get_avatar_list(self, user_id: Optional[str] = None) -> List[AvatarInfo]:
        """
        Retrieve available avatars for user.
        
        Args:
            user_id: Optional user ID for filtering
            
        Returns:
            List of avatar information
        """
        try:
            avatars = self.avatar_database.get_avatar_list(user_id)
            
            # Enhance with cache information
            for avatar in avatars:
                cache_status = self.face_cache_manager.get_cache_status(avatar.avatar_id)
                avatar.cache_size = cache_status.cache_size
            
            logger.info(f"Retrieved {len(avatars)} avatars for user {user_id}")
            return avatars
            
        except Exception as e:
            logger.error(f"Failed to get avatar list: {e}")
            return []
    
    def get_avatar_info(self, avatar_id: str) -> Optional[AvatarInfo]:
        """
        Retrieve specific avatar details.
        
        Args:
            avatar_id: Avatar identifier
            
        Returns:
            Detailed avatar information or None
        """
        try:
            avatar_info = self.avatar_database.get_avatar_info(avatar_id)
            
            if avatar_info:
                # Enhance with cache information
                cache_status = self.face_cache_manager.get_cache_status(avatar_id)
                avatar_info.cache_size = cache_status.cache_size
                
                # Update usage statistics
                self.avatar_database.update_avatar_usage(avatar_id)
            
            return avatar_info
            
        except Exception as e:
            logger.error(f"Failed to get avatar info for {avatar_id}: {e}")
            return None
    
    def delete_avatar(self, avatar_id: str, user_id: str) -> bool:
        """
        Remove avatar and associated data.
        
        Args:
            avatar_id: Avatar identifier
            user_id: User identifier for permission check
            
        Returns:
            bool: Deletion success status
        """
        try:
            # Delete from database
            db_success = self.avatar_database.delete_avatar(avatar_id, user_id)
            
            if db_success:
                # Remove cache data
                cache_status = self.face_cache_manager.get_cache_status(avatar_id)
                if cache_status.is_cached:
                    # Delete from disk cache if available
                    if hasattr(self.face_cache_manager, 'disk_cache') and self.face_cache_manager.disk_cache:
                        self.face_cache_manager.disk_cache.delete_face_data(avatar_id)
                
                # Remove avatar files
                self._cleanup_avatar_files(avatar_id)
                
                logger.info(f"Avatar {avatar_id} deleted successfully")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete avatar {avatar_id}: {e}")
            return False
    
    def warm_up_avatar_cache(self, avatar_ids: List[str]) -> WarmupReport:
        """
        Pre-load avatar caches for optimal performance.
        
        Args:
            avatar_ids: List of avatar IDs to pre-load
            
        Returns:
            WarmupReport: Warmup operation results
        """
        try:
            # Filter to only existing avatars
            existing_avatars = []
            for avatar_id in avatar_ids:
                if self.avatar_database.get_avatar_info(avatar_id):
                    existing_avatars.append(avatar_id)
            
            if not existing_avatars:
                return WarmupReport(
                    avatars_warmed=0,
                    total_avatars=len(avatar_ids),
                    warmup_time=0.0,
                    cache_size_total=0,
                    errors=[],
                    warnings=["No valid avatars found for warmup"]
                )
            
            # Perform warmup
            warmup_report = self.face_cache_manager.warm_up_avatar_cache(existing_avatars)
            
            logger.info(f"Avatar cache warmup completed: {warmup_report.avatars_warmed} avatars warmed")
            return warmup_report
            
        except Exception as e:
            logger.error(f"Avatar cache warmup failed: {e}")
            return WarmupReport(
                avatars_warmed=0,
                total_avatars=len(avatar_ids),
                warmup_time=0.0,
                cache_size_total=0,
                errors=[f"Warmup failed: {str(e)}"],
                warnings=[]
            )
    
    def get_processing_stats(self, avatar_id: str) -> Optional[ProcessingStats]:
        """
        Get processing statistics for avatar.
        
        Args:
            avatar_id: Avatar identifier
            
        Returns:
            Processing statistics or None
        """
        try:
            avatar_info = self.avatar_database.get_avatar_info(avatar_id)
            if not avatar_info:
                return None
            
            cache_metrics = self.face_cache_manager.get_cache_metrics()
            cache_hit_rate = (
                cache_metrics.cache_hits / max(cache_metrics.total_requests, 1)
                if cache_metrics.total_requests > 0 else 0.0
            )
            
            return ProcessingStats(
                total_uses=avatar_info.usage_count,
                last_used=avatar_info.last_used,
                avg_processing_time=cache_metrics.avg_access_time,
                cache_hit_rate=cache_hit_rate,
                total_processing_time=avatar_info.usage_count * cache_metrics.avg_access_time
            )
            
        except Exception as e:
            logger.error(f"Failed to get processing stats for {avatar_id}: {e}")
            return None
    
    def _generate_avatar_id(self, user_id: str, avatar_name: str) -> str:
        """Generate unique avatar identifier."""
        base_id = f"{user_id}_{avatar_name}".replace(" ", "_").lower()
        unique_suffix = str(uuid.uuid4())[:8]
        return f"{base_id}_{unique_suffix}"
    
    def _extract_file_format(self, filename: str) -> Optional[str]:
        """Extract and validate file format from filename."""
        try:
            extension = filename.lower().split('.')[-1]
            
            if f".{extension}" in self.config.supported_formats:
                return extension
            
            return None
            
        except Exception:
            return None
    
    def _cleanup_avatar_files(self, avatar_id: str):
        """Clean up avatar files after deletion."""
        try:
            avatar_dir = os.path.join(self.config.avatar_storage_path, avatar_id)
            
            if os.path.exists(avatar_dir):
                import shutil
                shutil.rmtree(avatar_dir)
                logger.info(f"Cleaned up avatar files for {avatar_id}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup avatar files for {avatar_id}: {e}")

# Convenience functions for service initialization
def create_avatar_service(
    avatar_config: AvatarConfig,
    avatar_registrar: ColdAvatarRegistrar,
    face_cache_manager: FaceCacheManager
) -> AvatarManagementService:
    """
    Create configured avatar management service.
    
    Args:
        avatar_config: Avatar configuration
        avatar_registrar: Avatar registration engine  
        face_cache_manager: Face cache manager
        
    Returns:
        AvatarManagementService: Configured service instance
    """
    return AvatarManagementService(
        avatar_config=avatar_config,
        avatar_registrar=avatar_registrar,
        face_cache_manager=face_cache_manager
    ) 