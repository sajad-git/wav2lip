"""
Input Validation - Comprehensive validation and security measures
Validates various input types with security scanning and rate limiting
"""

import asyncio
import logging
import re
import time
import hashlib
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
from enum import Enum
import base64
import magic
import io

from app.config.settings import Settings

class ValidationStatus(Enum):
    """Validation result status"""
    VALID = "valid"
    INVALID = "invalid"
    SUSPICIOUS = "suspicious"
    REJECTED = "rejected"

@dataclass
class ValidationResult:
    """Validation result with details"""
    is_valid: bool
    status: ValidationStatus
    error_messages: List[str]
    warnings: List[str]
    sanitized_data: Optional[Any] = None
    security_score: float = 1.0
    validation_time: float = 0.0

@dataclass
class SecurityScanResult:
    """Security scan result"""
    is_safe: bool
    threat_level: str  # low, medium, high, critical
    detected_threats: List[str]
    scan_time: float
    recommendations: List[str]

class RateLimiter:
    """Rate limiting for validation requests"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting configuration
        self.max_requests_per_minute = 100
        self.max_requests_per_hour = 1000
        self.client_requests: Dict[str, List[float]] = {}
        
        # Penalty system
        self.penalty_duration = 300  # 5 minutes
        self.penalized_clients: Dict[str, float] = {}
    
    def is_rate_limited(self, client_id: str) -> tuple[bool, Optional[str]]:
        """Check if client is rate limited"""
        current_time = time.time()
        
        # Check if client is penalized
        if client_id in self.penalized_clients:
            penalty_end = self.penalized_clients[client_id]
            if current_time < penalty_end:
                remaining = int(penalty_end - current_time)
                return True, f"Rate limited for {remaining} seconds"
            else:
                del self.penalized_clients[client_id]
        
        # Initialize client tracking
        if client_id not in self.client_requests:
            self.client_requests[client_id] = []
        
        # Clean old requests
        one_minute_ago = current_time - 60
        one_hour_ago = current_time - 3600
        
        recent_requests = [
            req_time for req_time in self.client_requests[client_id]
            if req_time > one_hour_ago
        ]
        
        # Check hourly limit
        if len(recent_requests) >= self.max_requests_per_hour:
            self._penalize_client(client_id)
            return True, "Hourly rate limit exceeded"
        
        # Check minute limit
        minute_requests = [
            req_time for req_time in recent_requests
            if req_time > one_minute_ago
        ]
        
        if len(minute_requests) >= self.max_requests_per_minute:
            return True, "Rate limit exceeded, try again later"
        
        # Update request history
        self.client_requests[client_id] = recent_requests + [current_time]
        
        return False, None
    
    def _penalize_client(self, client_id: str):
        """Apply penalty to client"""
        self.penalized_clients[client_id] = time.time() + self.penalty_duration
        self.logger.warning(f"Client {client_id} penalized for rate limit violation")

class SecurityScanner:
    """Scans content for security threats"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Malicious patterns
        self.malicious_patterns = [
            # Script injection patterns
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            
            # SQL injection patterns
            r'union\s+select',
            r'drop\s+table',
            r'delete\s+from',
            r'insert\s+into',
            
            # Command injection patterns
            r'system\s*\(',
            r'exec\s*\(',
            r'eval\s*\(',
            r'\$\([^)]+\)',
            
            # Path traversal
            r'\.\./+',
            r'\\\.\\',
            
            # Binary data patterns
            r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\xFF]',
        ]
        
        # Suspicious keywords
        self.suspicious_keywords = {
            'script', 'javascript', 'vbscript', 'onload', 'onerror',
            'alert', 'confirm', 'prompt', 'document', 'window',
            'eval', 'exec', 'system', 'shell', 'cmd',
            'union', 'select', 'drop', 'delete', 'insert', 'update',
            'admin', 'root', 'password', 'passwd', 'secret'
        }
        
        # File type magic bytes
        self.allowed_audio_types = {
            b'RIFF': 'wav',
            b'ID3': 'mp3',
            b'\xff\xfb': 'mp3',
            b'OggS': 'ogg',
            b'fLaC': 'flac'
        }
        
        self.allowed_image_types = {
            b'\xff\xd8\xff': 'jpeg',
            b'\x89PNG\r\n': 'png',
            b'GIF87a': 'gif',
            b'GIF89a': 'gif',
            b'RIFF': 'webp'  # WebP in RIFF container
        }
    
    async def scan_content(self, data: bytes, data_type: str) -> SecurityScanResult:
        """Scan content for security threats"""
        start_time = time.time()
        
        threats = []
        threat_level = "low"
        
        try:
            # File type validation
            if data_type in ['audio', 'image']:
                file_threats = await self._scan_file_content(data, data_type)
                threats.extend(file_threats)
            
            # Pattern-based scanning
            text_content = self._extract_text_content(data)
            if text_content:
                pattern_threats = self._scan_text_patterns(text_content)
                threats.extend(pattern_threats)
            
            # Determine threat level
            if threats:
                if any('critical' in threat.lower() for threat in threats):
                    threat_level = "critical"
                elif any('high' in threat.lower() for threat in threats):
                    threat_level = "high"
                elif any('medium' in threat.lower() for threat in threats):
                    threat_level = "medium"
                else:
                    threat_level = "medium"
            
            # Generate recommendations
            recommendations = self._generate_security_recommendations(threats)
            
            scan_time = time.time() - start_time
            
            return SecurityScanResult(
                is_safe=len(threats) == 0,
                threat_level=threat_level,
                detected_threats=threats,
                scan_time=scan_time,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Security scan failed: {str(e)}")
            return SecurityScanResult(
                is_safe=False,
                threat_level="critical",
                detected_threats=[f"Scan error: {str(e)}"],
                scan_time=time.time() - start_time,
                recommendations=["Reject content due to scan failure"]
            )
    
    async def _scan_file_content(self, data: bytes, data_type: str) -> List[str]:
        """Scan file content for malicious patterns"""
        threats = []
        
        # Check file size
        if len(data) > 50 * 1024 * 1024:  # 50MB limit
            threats.append("File size exceeds maximum allowed (50MB)")
        
        # Validate file magic bytes
        if data_type == 'audio':
            if not self._validate_audio_file(data):
                threats.append("Invalid or suspicious audio file format")
        elif data_type == 'image':
            if not self._validate_image_file(data):
                threats.append("Invalid or suspicious image file format")
        
        # Check for embedded content
        if self._has_embedded_content(data):
            threats.append("File contains suspicious embedded content")
        
        return threats
    
    def _validate_audio_file(self, data: bytes) -> bool:
        """Validate audio file format"""
        if len(data) < 12:
            return False
        
        for magic_bytes, file_type in self.allowed_audio_types.items():
            if data.startswith(magic_bytes):
                return True
        
        return False
    
    def _validate_image_file(self, data: bytes) -> bool:
        """Validate image file format"""
        if len(data) < 12:
            return False
        
        for magic_bytes, file_type in self.allowed_image_types.items():
            if data.startswith(magic_bytes):
                return True
        
        return False
    
    def _has_embedded_content(self, data: bytes) -> bool:
        """Check for suspicious embedded content"""
        # Look for script tags, executables, etc.
        suspicious_patterns = [
            b'<script',
            b'javascript:',
            b'MZ\x90\x00',  # PE executable
            b'\x7fELF',     # ELF executable
            b'\xfe\xed\xfa', # Mach-O executable
        ]
        
        for pattern in suspicious_patterns:
            if pattern in data:
                return True
        
        return False
    
    def _extract_text_content(self, data: bytes) -> Optional[str]:
        """Extract text content from binary data"""
        try:
            # Try UTF-8 decoding
            return data.decode('utf-8', errors='ignore')
        except:
            return None
    
    def _scan_text_patterns(self, text: str) -> List[str]:
        """Scan text for malicious patterns"""
        threats = []
        text_lower = text.lower()
        
        # Check malicious patterns
        for pattern in self.malicious_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                threats.append(f"Malicious pattern detected: {pattern}")
        
        # Check suspicious keywords
        found_keywords = []
        for keyword in self.suspicious_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        if len(found_keywords) > 3:  # Multiple suspicious keywords
            threats.append(f"Multiple suspicious keywords: {', '.join(found_keywords[:5])}")
        
        return threats
    
    def _generate_security_recommendations(self, threats: List[str]) -> List[str]:
        """Generate security recommendations based on threats"""
        recommendations = []
        
        if not threats:
            recommendations.append("Content appears safe")
            return recommendations
        
        if any('script' in threat.lower() for threat in threats):
            recommendations.append("Sanitize or reject content with script injection")
        
        if any('sql' in threat.lower() for threat in threats):
            recommendations.append("Use parameterized queries to prevent SQL injection")
        
        if any('file' in threat.lower() for threat in threats):
            recommendations.append("Validate file types and scan with antivirus")
        
        if any('embedded' in threat.lower() for threat in threats):
            recommendations.append("Strip metadata and re-encode media files")
        
        if len(threats) > 3:
            recommendations.append("Consider rejecting content due to multiple threats")
        
        return recommendations

class InputValidator:
    """Main input validation class"""
    
    def __init__(self):
        self.settings = Settings()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.rate_limiter = RateLimiter()
        self.security_scanner = SecurityScanner()
        
        # Validation limits
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.max_text_length = 10000  # 10,000 characters
        self.allowed_formats = {
            'audio': {'wav', 'mp3', 'ogg', 'flac', 'm4a'},
            'image': {'jpg', 'jpeg', 'png', 'gif', 'webp'},
            'video': {'mp4', 'mov', 'avi', 'webm'}
        }
    
    async def validate_audio_input(
        self,
        audio_data: bytes,
        client_id: str
    ) -> ValidationResult:
        """Validate audio input with comprehensive checks"""
        start_time = time.time()
        result = ValidationResult(
            is_valid=True,
            status=ValidationStatus.VALID,
            error_messages=[],
            warnings=[]
        )
        
        try:
            # Rate limiting check
            is_limited, limit_msg = self.rate_limiter.is_rate_limited(client_id)
            if is_limited:
                result.is_valid = False
                result.status = ValidationStatus.REJECTED
                result.error_messages.append(limit_msg)
                return result
            
            # Size validation
            if len(audio_data) > self.max_file_size:
                result.is_valid = False
                result.error_messages.append(f"Audio file too large (max {self.max_file_size // 1024 // 1024}MB)")
            
            if len(audio_data) < 1024:  # Minimum 1KB
                result.is_valid = False
                result.error_messages.append("Audio file too small")
            
            # Security scan
            security_result = await self.security_scanner.scan_content(audio_data, 'audio')
            if not security_result.is_safe:
                if security_result.threat_level in ['critical', 'high']:
                    result.is_valid = False
                    result.status = ValidationStatus.REJECTED
                    result.error_messages.extend(security_result.detected_threats)
                else:
                    result.status = ValidationStatus.SUSPICIOUS
                    result.warnings.extend(security_result.detected_threats)
            
            # Update security score
            result.security_score = 1.0 - (len(security_result.detected_threats) * 0.1)
            
            # Validation timing
            result.validation_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Audio validation failed: {str(e)}")
            result.is_valid = False
            result.status = ValidationStatus.INVALID
            result.error_messages.append(f"Validation error: {str(e)}")
            return result
    
    async def validate_avatar_input(
        self,
        file_data: bytes,
        filename: str,
        client_id: str
    ) -> ValidationResult:
        """Validate avatar file input"""
        start_time = time.time()
        result = ValidationResult(
            is_valid=True,
            status=ValidationStatus.VALID,
            error_messages=[],
            warnings=[]
        )
        
        try:
            # Rate limiting check
            is_limited, limit_msg = self.rate_limiter.is_rate_limited(client_id)
            if is_limited:
                result.is_valid = False
                result.status = ValidationStatus.REJECTED
                result.error_messages.append(limit_msg)
                return result
            
            # File extension validation
            file_ext = filename.lower().split('.')[-1]
            if file_ext not in self.allowed_formats.get('image', set()) and \
               file_ext not in self.allowed_formats.get('video', set()):
                result.is_valid = False
                result.error_messages.append(f"Unsupported file format: {file_ext}")
            
            # Size validation
            if len(file_data) > self.max_file_size:
                result.is_valid = False
                result.error_messages.append(f"File too large (max {self.max_file_size // 1024 // 1024}MB)")
            
            if len(file_data) < 1024:  # Minimum 1KB
                result.is_valid = False
                result.error_messages.append("File too small")
            
            # Determine file type for scanning
            scan_type = 'image' if file_ext in self.allowed_formats.get('image', set()) else 'video'
            
            # Security scan
            security_result = await self.security_scanner.scan_content(file_data, scan_type)
            if not security_result.is_safe:
                if security_result.threat_level in ['critical', 'high']:
                    result.is_valid = False
                    result.status = ValidationStatus.REJECTED
                    result.error_messages.extend(security_result.detected_threats)
                else:
                    result.status = ValidationStatus.SUSPICIOUS
                    result.warnings.extend(security_result.detected_threats)
            
            # Update security score
            result.security_score = 1.0 - (len(security_result.detected_threats) * 0.1)
            
            # Validation timing
            result.validation_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Avatar validation failed: {str(e)}")
            result.is_valid = False
            result.status = ValidationStatus.INVALID
            result.error_messages.append(f"Validation error: {str(e)}")
            return result
    
    async def validate_text_input(
        self,
        text: str,
        language: str = "fa"
    ) -> ValidationResult:
        """Validate text input"""
        start_time = time.time()
        result = ValidationResult(
            is_valid=True,
            status=ValidationStatus.VALID,
            error_messages=[],
            warnings=[]
        )
        
        try:
            # Length validation
            if len(text) > self.max_text_length:
                result.is_valid = False
                result.error_messages.append(f"Text too long (max {self.max_text_length} characters)")
            
            if not text.strip():
                result.is_valid = False
                result.error_messages.append("Text cannot be empty")
            
            # Character encoding validation
            try:
                text.encode('utf-8')
            except UnicodeEncodeError:
                result.is_valid = False
                result.error_messages.append("Invalid character encoding")
            
            # Security scan
            security_result = await self.security_scanner.scan_content(text.encode('utf-8'), 'text')
            if not security_result.is_safe:
                if security_result.threat_level in ['critical', 'high']:
                    result.is_valid = False
                    result.status = ValidationStatus.REJECTED
                    result.error_messages.extend(security_result.detected_threats)
                else:
                    result.status = ValidationStatus.SUSPICIOUS
                    result.warnings.extend(security_result.detected_threats)
            
            # Sanitize text if needed
            if result.is_valid:
                sanitized_text = self._sanitize_text(text)
                result.sanitized_data = sanitized_text
            
            # Update security score
            result.security_score = 1.0 - (len(security_result.detected_threats) * 0.1)
            
            # Validation timing
            result.validation_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Text validation failed: {str(e)}")
            result.is_valid = False
            result.status = ValidationStatus.INVALID
            result.error_messages.append(f"Validation error: {str(e)}")
            return result
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitize text input"""
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', text)
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        # Limit line breaks
        sanitized = re.sub(r'\n{3,}', '\n\n', sanitized)
        
        return sanitized
    
    async def check_security_threats(
        self,
        data: bytes,
        data_type: str
    ) -> SecurityScanResult:
        """Public interface for security scanning"""
        return await self.security_scanner.scan_content(data, data_type)
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            "rate_limiter_clients": len(self.rate_limiter.client_requests),
            "penalized_clients": len(self.rate_limiter.penalized_clients),
            "max_file_size_mb": self.max_file_size // 1024 // 1024,
            "max_text_length": self.max_text_length,
            "supported_formats": self.allowed_formats
        } 