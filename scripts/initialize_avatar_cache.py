#!/usr/bin/env python3
"""
Avatar Cache Initialization Script
Sets up avatar database and registers default avatars
"""

import os
import sys
import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict
import uuid

# Add app to path
sys.path.append('/app')

from app.config.settings import settings
from app.config.avatar_config import avatar_config


class AvatarCacheInitializer:
    """Initialize avatar cache system and database"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.database_path = avatar_config.database_path
        self.default_avatars = [
            {
                "name": "Default Persian Avatar",
                "file_path": "/app/assets/avatars/default_avatar.jpg",
                "description": "Default Persian-speaking avatar"
            },
            {
                "name": "Persian Female Avatar", 
                "file_path": "/app/assets/avatars/persian_avatar.jpg",
                "description": "Persian female avatar for professional use"
            }
        ]
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        return logging.getLogger(__name__)
    
    def initialize_database(self) -> bool:
        """Initialize avatar metadata database"""
        self.logger.info("🗄️ Initializing avatar database...")
        
        try:
            # Ensure database directory exists
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
            
            # Connect to database
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create avatars table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS avatars (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    avatar_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    cache_path TEXT,
                    owner_id TEXT,
                    file_format TEXT,
                    file_size INTEGER,
                    face_quality_score REAL,
                    registration_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_accessed DATETIME,
                    access_count INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1,
                    metadata_json TEXT
                )
            """)
            
            # Create indices for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_avatar_id ON avatars(avatar_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_owner_id ON avatars(owner_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_is_active ON avatars(is_active)")
            
            # Create face cache metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    avatar_id TEXT NOT NULL,
                    cache_file_path TEXT NOT NULL,
                    cache_size INTEGER,
                    compression_ratio REAL,
                    created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_accessed DATETIME,
                    access_count INTEGER DEFAULT 0,
                    cache_version TEXT DEFAULT '1.0',
                    integrity_hash TEXT,
                    FOREIGN KEY (avatar_id) REFERENCES avatars(avatar_id)
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_face_cache_avatar_id ON face_cache(avatar_id)")
            
            conn.commit()
            conn.close()
            
            self.logger.info("✅ Avatar database initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {str(e)}")
            return False
    
    def create_default_avatars(self) -> bool:
        """Create default avatar files if they don't exist"""
        self.logger.info("🎨 Creating default avatar files...")
        
        try:
            import numpy as np
            from PIL import Image, ImageDraw, ImageFont
            
            assets_dir = Path("/app/assets/avatars")
            assets_dir.mkdir(parents=True, exist_ok=True)
            
            # Create default avatar (simple placeholder)
            default_avatar_path = assets_dir / "default_avatar.jpg"
            if not default_avatar_path.exists():
                self._create_placeholder_avatar(
                    default_avatar_path, 
                    "Default\nAvatar", 
                    (100, 150, 200)  # Blue background
                )
                self.logger.info(f"✅ Created {default_avatar_path}")
            
            # Create Persian avatar (simple placeholder)
            persian_avatar_path = assets_dir / "persian_avatar.jpg"
            if not persian_avatar_path.exists():
                self._create_placeholder_avatar(
                    persian_avatar_path, 
                    "Persian\nAvatar", 
                    (150, 100, 200)  # Purple background
                )
                self.logger.info(f"✅ Created {persian_avatar_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create default avatars: {str(e)}")
            return False
    
    def _create_placeholder_avatar(self, file_path: Path, text: str, bg_color: tuple):
        """Create a simple placeholder avatar image"""
        # Create a 256x256 image with colored background
        img = Image.new('RGB', (256, 256), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font, fallback to basic if not available
        try:
            # Try to find a reasonable font
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Draw text in center
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (256 - text_width) // 2
        y = (256 - text_height) // 2
        
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        
        # Draw a simple face outline
        face_center = (128, 100)
        face_radius = 60
        
        # Face circle
        draw.ellipse([
            face_center[0] - face_radius, face_center[1] - face_radius,
            face_center[0] + face_radius, face_center[1] + face_radius
        ], outline=(255, 255, 255), width=3)
        
        # Eyes
        eye1_pos = (face_center[0] - 20, face_center[1] - 10)
        eye2_pos = (face_center[0] + 20, face_center[1] - 10)
        draw.ellipse([eye1_pos[0] - 5, eye1_pos[1] - 5, eye1_pos[0] + 5, eye1_pos[1] + 5], 
                    fill=(255, 255, 255))
        draw.ellipse([eye2_pos[0] - 5, eye2_pos[1] - 5, eye2_pos[0] + 5, eye2_pos[1] + 5], 
                    fill=(255, 255, 255))
        
        # Mouth
        mouth_pos = (face_center[0], face_center[1] + 20)
        draw.arc([
            mouth_pos[0] - 15, mouth_pos[1] - 10,
            mouth_pos[0] + 15, mouth_pos[1] + 10
        ], start=0, end=180, fill=(255, 255, 255), width=3)
        
        # Save image
        img.save(file_path, "JPEG", quality=90)
    
    def register_default_avatars(self) -> bool:
        """Register default avatars in the database"""
        self.logger.info("👤 Registering default avatars...")
        
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            for avatar_data in self.default_avatars:
                avatar_id = str(uuid.uuid4())
                file_path = avatar_data["file_path"]
                
                # Check if file exists
                if not os.path.exists(file_path):
                    self.logger.warning(f"⚠️ Avatar file not found: {file_path}")
                    continue
                
                # Check if avatar already exists
                cursor.execute("SELECT id FROM avatars WHERE name = ?", (avatar_data["name"],))
                if cursor.fetchone():
                    self.logger.info(f"✅ Avatar '{avatar_data['name']}' already exists")
                    continue
                
                # Get file info
                file_size = os.path.getsize(file_path)
                file_format = os.path.splitext(file_path)[1].lower()
                
                # Create metadata
                metadata = {
                    "is_default": True,
                    "created_by": "system",
                    "description": avatar_data["description"],
                    "tags": ["default", "persian", "system"]
                }
                
                # Insert avatar record
                cursor.execute("""
                    INSERT INTO avatars (
                        avatar_id, name, file_path, owner_id, file_format, 
                        file_size, face_quality_score, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    avatar_id,
                    avatar_data["name"],
                    file_path,
                    "system",
                    file_format,
                    file_size,
                    0.8,  # Default quality score
                    json.dumps(metadata)
                ))
                
                self.logger.info(f"✅ Registered avatar: {avatar_data['name']} (ID: {avatar_id})")
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register default avatars: {str(e)}")
            return False
    
    def initialize_cache_directories(self) -> bool:
        """Initialize face cache directories"""
        self.logger.info("📁 Initializing cache directories...")
        
        try:
            # Create cache directory structure
            cache_dirs = [
                avatar_config.cache_storage_path,
                os.path.join(avatar_config.cache_storage_path, "compressed"),
                os.path.join(avatar_config.cache_storage_path, "temp")
            ]
            
            for cache_dir in cache_dirs:
                os.makedirs(cache_dir, exist_ok=True)
                self.logger.info(f"✅ Created cache directory: {cache_dir}")
            
            # Create cache metadata file
            cache_metadata_path = os.path.join(avatar_config.cache_storage_path, "cache_metadata.json")
            if not os.path.exists(cache_metadata_path):
                metadata = {
                    "version": "1.0",
                    "created_date": "2024-01-01T00:00:00",
                    "total_cache_size": 0,
                    "total_avatars": 0,
                    "compression_enabled": avatar_config.cache_compression
                }
                
                with open(cache_metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                self.logger.info("✅ Created cache metadata file")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize cache directories: {str(e)}")
            return False
    
    def validate_initialization(self) -> bool:
        """Validate that initialization was successful"""
        self.logger.info("🔍 Validating initialization...")
        
        try:
            # Check database
            if not os.path.exists(self.database_path):
                self.logger.error("❌ Database file not found")
                return False
            
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Check tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['avatars', 'face_cache']
            for table in required_tables:
                if table not in tables:
                    self.logger.error(f"❌ Required table missing: {table}")
                    return False
            
            # Check default avatars
            cursor.execute("SELECT COUNT(*) FROM avatars WHERE owner_id = 'system'")
            default_count = cursor.fetchone()[0]
            
            if default_count == 0:
                self.logger.warning("⚠️ No default avatars found")
            else:
                self.logger.info(f"✅ Found {default_count} default avatars")
            
            conn.close()
            
            # Check cache directories
            if not os.path.exists(avatar_config.cache_storage_path):
                self.logger.error("❌ Cache directory not found")
                return False
            
            self.logger.info("✅ Initialization validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Validation failed: {str(e)}")
            return False


def main():
    """Main initialization function"""
    initializer = AvatarCacheInitializer()
    
    print("🚀 Starting Avatar Cache Initialization...")
    
    # Step 1: Initialize database
    if not initializer.initialize_database():
        print("❌ Database initialization failed!")
        sys.exit(1)
    
    # Step 2: Create default avatar files
    if not initializer.create_default_avatars():
        print("❌ Default avatar creation failed!")
        sys.exit(1)
    
    # Step 3: Register default avatars
    if not initializer.register_default_avatars():
        print("❌ Default avatar registration failed!")
        sys.exit(1)
    
    # Step 4: Initialize cache directories
    if not initializer.initialize_cache_directories():
        print("❌ Cache directory initialization failed!")
        sys.exit(1)
    
    # Step 5: Validate initialization
    if not initializer.validate_initialization():
        print("❌ Initialization validation failed!")
        sys.exit(1)
    
    print("🎉 Avatar Cache Initialization completed successfully!")
    print(f"📍 Database: {initializer.database_path}")
    print(f"📍 Cache: {avatar_config.cache_storage_path}")
    
    sys.exit(0)


if __name__ == "__main__":
    main() 