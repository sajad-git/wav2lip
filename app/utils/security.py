"""
Security utilities for Avatar Streaming Service
Provides authentication, authorization, rate limiting, and security measures
"""

import hashlib
import hmac
import secrets
import time
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Union, Any
from functools import wraps
from dataclasses import dataclass
import bcrypt
import re
import ipaddress
from cryptography.fernet import Fernet
import base64
import logging

logger = logging.getLogger(__name__)

# Security Configuration
@dataclass
class SecurityConfig:
    """Security configuration settings"""
    jwt_secret_key: str = secrets.token_urlsafe(32)
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst_size: int = 10
    allowed_file_types: Set[str] = None
    max_file_size_mb: int = 50
    encryption_key: bytes = None
    
    def __post_init__(self):
        if self.allowed_file_types is None:
            self.allowed_file_types = {'.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mov', '.wav', '.mp3'}
        if self.encryption_key is None:
            self.encryption_key = Fernet.generate_key()

# Global security configuration
security_config = SecurityConfig()

class SecurityError(Exception):
    """Base security exception"""
    pass

class AuthenticationError(SecurityError):
    """Authentication failed"""
    pass

class AuthorizationError(SecurityError):
    """Authorization failed"""
    pass

class RateLimitError(SecurityError):
    """Rate limit exceeded"""
    pass

class FileValidationError(SecurityError):
    """File validation failed"""
    pass

@dataclass
class UserSession:
    """User session information"""
    user_id: str
    username: str
    permissions: Set[str]
    created_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True

class TokenManager:
    """JWT token management"""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or security_config
    
    def generate_token(self, user_id: str, username: str, permissions: List[str] = None) -> str:
        """Generate JWT token for user"""
        try:
            payload = {
                'user_id': user_id,
                'username': username,
                'permissions': permissions or [],
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(hours=self.config.jwt_expiry_hours),
                'jti': secrets.token_urlsafe(16)  # JWT ID for token invalidation
            }
            
            token = jwt.encode(
                payload,
                self.config.jwt_secret_key,
                algorithm=self.config.jwt_algorithm
            )
            
            logger.info(f"Generated token for user {username}")
            return token
            
        except Exception as e:
            logger.error(f"Error generating token: {e}")
            raise AuthenticationError("Failed to generate authentication token")
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Check if token is expired
            if datetime.utcnow() > datetime.fromtimestamp(payload['exp']):
                raise AuthenticationError("Token expired")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            raise AuthenticationError("Invalid authentication token")
    
    def refresh_token(self, token: str) -> str:
        """Refresh an existing token"""
        try:
            # Verify current token
            payload = self.verify_token(token)
            
            # Generate new token with same permissions
            return self.generate_token(
                payload['user_id'],
                payload['username'],
                payload['permissions']
            )
            
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            raise AuthenticationError("Failed to refresh token")

class PasswordManager:
    """Password hashing and verification"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        try:
            # Generate salt and hash password
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error hashing password: {e}")
            raise SecurityError("Failed to hash password")
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'),
                hashed_password.encode('utf-8')
            )
            
        except Exception as e:
            logger.error(f"Error verifying password: {e}")
            return False
    
    @staticmethod
    def generate_secure_password(length: int = 16) -> str:
        """Generate a secure random password"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def validate_password_strength(password: str) -> Dict[str, Union[bool, str]]:
        """Validate password strength"""
        result = {
            'is_valid': True,
            'score': 0,
            'issues': []
        }
        
        # Minimum length
        if len(password) < 8:
            result['issues'].append("Password must be at least 8 characters long")
            result['is_valid'] = False
        else:
            result['score'] += 1
        
        # Character variety
        if re.search(r'[a-z]', password):
            result['score'] += 1
        else:
            result['issues'].append("Password should contain lowercase letters")
        
        if re.search(r'[A-Z]', password):
            result['score'] += 1
        else:
            result['issues'].append("Password should contain uppercase letters")
        
        if re.search(r'\d', password):
            result['score'] += 1
        else:
            result['issues'].append("Password should contain numbers")
        
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            result['score'] += 1
        else:
            result['issues'].append("Password should contain special characters")
        
        # Common patterns
        if re.search(r'(.)\1{2,}', password):
            result['issues'].append("Password should not contain repeated characters")
            result['score'] -= 1
        
        # Final validation
        if result['score'] < 3:
            result['is_valid'] = False
        
        return result

class RateLimiter:
    """Rate limiting implementation"""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or security_config
        self.requests: Dict[str, List[float]] = {}
        self.blocked_ips: Dict[str, float] = {}
    
    def is_allowed(self, identifier: str, request_type: str = "default") -> bool:
        """Check if request is allowed based on rate limits"""
        current_time = time.time()
        
        # Check if IP is temporarily blocked
        if identifier in self.blocked_ips:
            if current_time < self.blocked_ips[identifier]:
                return False
            else:
                # Unblock IP
                del self.blocked_ips[identifier]
        
        # Get request history for identifier
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        request_times = self.requests[identifier]
        
        # Remove old requests (older than 1 minute)
        cutoff_time = current_time - 60
        request_times[:] = [t for t in request_times if t > cutoff_time]
        
        # Check rate limit
        if len(request_times) >= self.config.rate_limit_requests_per_minute:
            # Block IP for escalating violations
            if len(request_times) > self.config.rate_limit_requests_per_minute * 2:
                self.blocked_ips[identifier] = current_time + (self.config.lockout_duration_minutes * 60)
            return False
        
        # Allow request and record it
        request_times.append(current_time)
        return True
    
    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for identifier"""
        current_time = time.time()
        
        if identifier not in self.requests:
            return self.config.rate_limit_requests_per_minute
        
        # Clean old requests
        cutoff_time = current_time - 60
        request_times = [t for t in self.requests[identifier] if t > cutoff_time]
        self.requests[identifier] = request_times
        
        return max(0, self.config.rate_limit_requests_per_minute - len(request_times))
    
    def reset_limits(self, identifier: str) -> None:
        """Reset rate limits for identifier"""
        if identifier in self.requests:
            del self.requests[identifier]
        if identifier in self.blocked_ips:
            del self.blocked_ips[identifier]

class FileSecurityValidator:
    """File upload security validation"""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or security_config
        self.dangerous_extensions = {
            '.exe', '.bat', '.cmd', '.scr', '.pif', '.com', '.vbs', '.js',
            '.jar', '.sh', '.py', '.php', '.asp', '.aspx', '.jsp'
        }
        self.image_magic_numbers = {
            b'\xFF\xD8\xFF': 'jpg',
            b'\x89PNG\r\n\x1a\n': 'png',
            b'GIF87a': 'gif',
            b'GIF89a': 'gif',
            b'\x00\x00\x00\x18ftypmp4': 'mp4',
            b'\x00\x00\x00\x20ftypM4V': 'mp4'
        }
    
    def validate_file(self, file_data: bytes, filename: str, expected_type: str = None) -> Dict[str, Any]:
        """Comprehensive file validation"""
        result = {
            'is_valid': True,
            'file_type': None,
            'size_mb': len(file_data) / (1024 * 1024),
            'issues': []
        }
        
        try:
            # File size validation
            if result['size_mb'] > self.config.max_file_size_mb:
                result['issues'].append(f"File size exceeds limit of {self.config.max_file_size_mb}MB")
                result['is_valid'] = False
            
            # Filename validation
            if not self._validate_filename(filename):
                result['issues'].append("Invalid filename")
                result['is_valid'] = False
            
            # Extension validation
            file_ext = self._get_file_extension(filename)
            if file_ext in self.dangerous_extensions:
                result['issues'].append("Dangerous file type detected")
                result['is_valid'] = False
            
            if file_ext not in self.config.allowed_file_types:
                result['issues'].append(f"File type {file_ext} not allowed")
                result['is_valid'] = False
            
            # Magic number validation
            detected_type = self._detect_file_type(file_data)
            result['file_type'] = detected_type
            
            if detected_type and expected_type and detected_type != expected_type:
                result['issues'].append(f"File type mismatch: expected {expected_type}, detected {detected_type}")
                result['is_valid'] = False
            
            # Content validation
            if not self._validate_file_content(file_data, detected_type):
                result['issues'].append("Suspicious file content detected")
                result['is_valid'] = False
            
        except Exception as e:
            logger.error(f"Error validating file: {e}")
            result['issues'].append("File validation error")
            result['is_valid'] = False
        
        return result
    
    def _validate_filename(self, filename: str) -> bool:
        """Validate filename for security"""
        # Check for path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return False
        
        # Check for null bytes
        if '\x00' in filename:
            return False
        
        # Check length
        if len(filename) > 255:
            return False
        
        # Check for control characters
        if any(ord(c) < 32 for c in filename if c != '\t'):
            return False
        
        return True
    
    def _get_file_extension(self, filename: str) -> str:
        """Extract file extension"""
        return '.' + filename.split('.')[-1].lower() if '.' in filename else ''
    
    def _detect_file_type(self, file_data: bytes) -> Optional[str]:
        """Detect file type by magic numbers"""
        if len(file_data) < 10:
            return None
        
        for magic_bytes, file_type in self.image_magic_numbers.items():
            if file_data.startswith(magic_bytes):
                return file_type
        
        return None
    
    def _validate_file_content(self, file_data: bytes, file_type: str) -> bool:
        """Validate file content for suspicious patterns"""
        try:
            # Check for embedded executables
            if b'MZ' in file_data[:1024]:  # DOS header
                return False
            
            # Check for script content in image files
            if file_type in ['jpg', 'png', 'gif']:
                suspicious_patterns = [
                    b'<script', b'javascript:', b'<?php', b'<%', b'eval(',
                    b'base64_decode', b'exec(', b'system('
                ]
                for pattern in suspicious_patterns:
                    if pattern in file_data:
                        return False
            
            return True
            
        except Exception:
            return False

class DataEncryption:
    """Data encryption utilities"""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or security_config
        self.cipher = Fernet(self.config.encryption_key)
    
    def encrypt_data(self, data: Union[str, bytes]) -> str:
        """Encrypt data and return base64 encoded string"""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            encrypted = self.cipher.encrypt(data)
            return base64.b64encode(encrypted).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise SecurityError("Failed to encrypt data")
    
    def decrypt_data(self, encrypted_data: str) -> bytes:
        """Decrypt base64 encoded encrypted data"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            return self.cipher.decrypt(encrypted_bytes)
            
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise SecurityError("Failed to decrypt data")
    
    def hash_data(self, data: Union[str, bytes], salt: str = None) -> str:
        """Generate secure hash of data"""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            if salt:
                data = data + salt.encode('utf-8')
            
            return hashlib.sha256(data).hexdigest()
            
        except Exception as e:
            logger.error(f"Error hashing data: {e}")
            raise SecurityError("Failed to hash data")

class IPSecurityManager:
    """IP-based security management"""
    
    def __init__(self):
        self.blocked_ips: Set[str] = set()
        self.allowed_ips: Set[str] = set()
        self.suspicious_ips: Dict[str, int] = {}
    
    def is_ip_allowed(self, ip_address: str) -> bool:
        """Check if IP address is allowed"""
        try:
            # Check if IP is blocked
            if ip_address in self.blocked_ips:
                return False
            
            # Check if IP is in allowed list (if list exists)
            if self.allowed_ips and ip_address not in self.allowed_ips:
                return False
            
            # Check for private/local IPs in production
            ip_obj = ipaddress.ip_address(ip_address)
            if ip_obj.is_private or ip_obj.is_loopback:
                return True  # Allow local IPs for development
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating IP {ip_address}: {e}")
            return False
    
    def block_ip(self, ip_address: str, reason: str = None) -> None:
        """Block IP address"""
        self.blocked_ips.add(ip_address)
        logger.warning(f"Blocked IP {ip_address}: {reason}")
    
    def unblock_ip(self, ip_address: str) -> None:
        """Unblock IP address"""
        self.blocked_ips.discard(ip_address)
        logger.info(f"Unblocked IP {ip_address}")
    
    def mark_suspicious(self, ip_address: str) -> bool:
        """Mark IP as suspicious and return if should be blocked"""
        if ip_address not in self.suspicious_ips:
            self.suspicious_ips[ip_address] = 0
        
        self.suspicious_ips[ip_address] += 1
        
        # Block after 5 suspicious activities
        if self.suspicious_ips[ip_address] >= 5:
            self.block_ip(ip_address, "Too many suspicious activities")
            return True
        
        return False

# Decorator functions for security
def require_authentication(func):
    """Decorator to require valid authentication"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # This would be implemented based on your framework
        # For FastAPI, you'd check the request headers for JWT token
        pass
    return wrapper

def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if user has required permission
            pass
        return wrapper
    return decorator

def rate_limit(requests_per_minute: int = None):
    """Decorator to apply rate limiting"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Apply rate limiting logic
            pass
        return wrapper
    return decorator

# Initialize security components
token_manager = TokenManager()
password_manager = PasswordManager()
rate_limiter = RateLimiter()
file_validator = FileSecurityValidator()
data_encryption = DataEncryption()
ip_security = IPSecurityManager()

logger.info("Security utilities initialized successfully") 