"""
Security Middleware for Avatar Streaming Service
Implements security headers, input validation, and threat detection
"""

import logging
import re
import hashlib
import time
from typing import Set, Dict, List, Optional
from dataclasses import dataclass
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config.settings import settings


@dataclass
class SecurityThreat:
    """Security threat detection record"""
    threat_type: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    client_id: str
    timestamp: float
    request_path: str


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for threat detection and protection"""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = logging.getLogger(__name__)
        
        # Security threat patterns
        self.sql_injection_patterns = [
            r"(\bUNION\b.*\bSELECT\b)",
            r"(\bSELECT\b.*\bFROM\b.*\bWHERE\b)",
            r"(\bINSERT\b.*\bINTO\b)",
            r"(\bDELETE\b.*\bFROM\b)",
            r"(\bDROP\b.*\bTABLE\b)",
            r"(\bALTER\b.*\bTABLE\b)",
            r"(\bEXEC\b.*\b\()",
            r"(\bSCRIPT\b.*\>)",
            r"(\'\s*OR\s*\'\d+\'\s*=\s*\'\d+)",
            r"(\'\s*OR\s*\d+\s*=\s*\d+)",
        ]
        
        self.xss_patterns = [
            r"(<script[^>]*>.*?</script>)",
            r"(<iframe[^>]*>.*?</iframe>)",
            r"(<object[^>]*>.*?</object>)",
            r"(<embed[^>]*>)",
            r"(<link[^>]*>)",
            r"(<meta[^>]*>)",
            r"(javascript:)",
            r"(vbscript:)",
            r"(onload\s*=)",
            r"(onerror\s*=)",
            r"(onclick\s*=)",
            r"(onmouseover\s*=)",
        ]
        
        self.command_injection_patterns = [
            r"(;|\||&|\$\(|\`)",
            r"(\.\./)",
            r"(/etc/passwd)",
            r"(/bin/sh)",
            r"(wget\s+)",
            r"(curl\s+)",
            r"(nc\s+-)",
            r"(telnet\s+)",
        ]
        
        # File upload security patterns
        self.malicious_file_patterns = [
            r"\.php$", r"\.asp$", r"\.aspx$", r"\.jsp$", r"\.jspx$",
            r"\.exe$", r"\.bat$", r"\.cmd$", r"\.sh$", r"\.ps1$",
            r"\.scr$", r"\.com$", r"\.pif$", r"\.vbs$", r"\.js$"
        ]
        
        # Blocked user agents and suspicious patterns
        self.blocked_user_agents = {
            "masscan", "nmap", "sqlmap", "nikto", "dirb", "gobuster",
            "wfuzz", "burpsuite", "owasp", "hydra", "medusa"
        }
        
        # Threat tracking
        self.threat_records: Dict[str, List[SecurityThreat]] = {}
        self.blocked_ips: Set[str] = set()
        
        # Security headers
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
                "https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: blob:; "
                "connect-src 'self' ws: wss:; "
                "media-src 'self' blob:; "
                "worker-src 'self' blob:;"
            ),
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": (
                "camera=(), microphone=(), geolocation=(), "
                "gyroscope=(), magnetometer=(), usb=()"
            )
        }
    
    def get_client_identifier(self, request: Request) -> str:
        """Get client identifier for tracking"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def detect_sql_injection(self, text: str) -> Optional[SecurityThreat]:
        """Detect SQL injection attempts"""
        text_lower = text.lower()
        
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return SecurityThreat(
                    threat_type="sql_injection",
                    severity="high",
                    description=f"SQL injection pattern detected: {pattern}",
                    client_id="",
                    timestamp=time.time(),
                    request_path=""
                )
        return None
    
    def detect_xss(self, text: str) -> Optional[SecurityThreat]:
        """Detect XSS attempts"""
        text_lower = text.lower()
        
        for pattern in self.xss_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return SecurityThreat(
                    threat_type="xss",
                    severity="high",
                    description=f"XSS pattern detected: {pattern}",
                    client_id="",
                    timestamp=time.time(),
                    request_path=""
                )
        return None
    
    def detect_command_injection(self, text: str) -> Optional[SecurityThreat]:
        """Detect command injection attempts"""
        for pattern in self.command_injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return SecurityThreat(
                    threat_type="command_injection",
                    severity="critical",
                    description=f"Command injection pattern detected: {pattern}",
                    client_id="",
                    timestamp=time.time(),
                    request_path=""
                )
        return None
    
    def validate_filename(self, filename: str) -> Optional[SecurityThreat]:
        """Validate uploaded filename for security"""
        # Check for path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            return SecurityThreat(
                threat_type="path_traversal",
                severity="high",
                description=f"Path traversal attempt in filename: {filename}",
                client_id="",
                timestamp=time.time(),
                request_path=""
            )
        
        # Check for malicious file extensions
        for pattern in self.malicious_file_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                return SecurityThreat(
                    threat_type="malicious_file",
                    severity="high",
                    description=f"Malicious file extension detected: {filename}",
                    client_id="",
                    timestamp=time.time(),
                    request_path=""
                )
        
        return None
    
    def check_user_agent(self, user_agent: str) -> Optional[SecurityThreat]:
        """Check for suspicious user agents"""
        if not user_agent:
            return SecurityThreat(
                threat_type="missing_user_agent",
                severity="low",
                description="Missing User-Agent header",
                client_id="",
                timestamp=time.time(),
                request_path=""
            )
        
        user_agent_lower = user_agent.lower()
        for blocked_agent in self.blocked_user_agents:
            if blocked_agent in user_agent_lower:
                return SecurityThreat(
                    threat_type="suspicious_user_agent",
                    severity="high",
                    description=f"Suspicious User-Agent: {user_agent}",
                    client_id="",
                    timestamp=time.time(),
                    request_path=""
                )
        
        return None
    
    def validate_content_type(self, request: Request) -> Optional[SecurityThreat]:
        """Validate content type for requests with body"""
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "").lower()
            
            # Check for suspicious content types
            suspicious_types = [
                "text/x-php", "application/x-php", "application/x-httpd-php",
                "text/x-perl", "application/x-perl", "text/x-python"
            ]
            
            for suspicious_type in suspicious_types:
                if suspicious_type in content_type:
                    return SecurityThreat(
                        threat_type="suspicious_content_type",
                        severity="medium",
                        description=f"Suspicious content type: {content_type}",
                        client_id="",
                        timestamp=time.time(),
                        request_path=""
                    )
        
        return None
    
    def record_threat(self, threat: SecurityThreat, client_id: str, request_path: str):
        """Record security threat"""
        threat.client_id = client_id
        threat.request_path = request_path
        
        if client_id not in self.threat_records:
            self.threat_records[client_id] = []
        
        self.threat_records[client_id].append(threat)
        
        # Auto-block clients with multiple high/critical threats
        high_threats = [t for t in self.threat_records[client_id] 
                       if t.severity in ["high", "critical"] and 
                       time.time() - t.timestamp < 3600]  # Last hour
        
        if len(high_threats) >= 3:
            self.blocked_ips.add(client_id)
            self.logger.critical(
                f"Auto-blocked client {client_id} due to {len(high_threats)} "
                f"high/critical threats in the last hour"
            )
        
        # Log threat
        self.logger.warning(
            f"Security threat detected: {threat.threat_type} "
            f"(severity: {threat.severity}) from {client_id} "
            f"on {request_path}: {threat.description}"
        )
    
    async def validate_request_body(self, request: Request) -> Optional[SecurityThreat]:
        """Validate request body for threats"""
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Read body (this consumes the stream, so we need to be careful)
                body = await request.body()
                if body:
                    body_text = body.decode('utf-8', errors='ignore')
                    
                    # Check for injection attempts in body
                    threat = self.detect_sql_injection(body_text)
                    if threat:
                        return threat
                    
                    threat = self.detect_xss(body_text)
                    if threat:
                        return threat
                    
                    threat = self.detect_command_injection(body_text)
                    if threat:
                        return threat
                
            except Exception as e:
                self.logger.debug(f"Error reading request body for security check: {e}")
        
        return None
    
    async def dispatch(self, request: Request, call_next):
        """Process request with security checks"""
        client_id = self.get_client_identifier(request)
        
        # Check if client is blocked
        if client_id in self.blocked_ips:
            self.logger.warning(f"Blocked request from {client_id}")
            return JSONResponse(
                status_code=403,
                content={
                    "error": "access_denied",
                    "message": "Access denied due to security violations"
                }
            )
        
        # Security checks
        threats = []
        
        # Check user agent
        user_agent = request.headers.get("user-agent", "")
        threat = self.check_user_agent(user_agent)
        if threat:
            threats.append(threat)
        
        # Check content type
        threat = self.validate_content_type(request)
        if threat:
            threats.append(threat)
        
        # Check query parameters
        for key, value in request.query_params.items():
            for check_func in [self.detect_sql_injection, self.detect_xss, self.detect_command_injection]:
                threat = check_func(f"{key}={value}")
                if threat:
                    threats.append(threat)
                    break
        
        # Check path parameters
        threat = self.detect_command_injection(str(request.url.path))
        if threat:
            threats.append(threat)
        
        # Validate filename if present
        if "filename" in request.query_params:
            threat = self.validate_filename(request.query_params["filename"])
            if threat:
                threats.append(threat)
        
        # Process any detected threats
        for threat in threats:
            self.record_threat(threat, client_id, str(request.url.path))
            
            # Block critical threats immediately
            if threat.severity == "critical":
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "security_violation",
                        "message": "Request blocked due to security policy violation"
                    }
                )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Add security headers
            for header, value in self.security_headers.items():
                response.headers[header] = value
            
            # Add custom security headers
            response.headers["X-Security-Policy"] = "strict"
            response.headers["X-Threat-Level"] = "monitored"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Security middleware error: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_error",
                    "message": "Internal security processing error"
                }
            )
    
    def get_threat_summary(self, client_id: str = None) -> Dict:
        """Get threat summary for monitoring"""
        if client_id:
            threats = self.threat_records.get(client_id, [])
            return {
                "client_id": client_id,
                "total_threats": len(threats),
                "threat_types": list(set(t.threat_type for t in threats)),
                "severity_counts": {
                    "low": len([t for t in threats if t.severity == "low"]),
                    "medium": len([t for t in threats if t.severity == "medium"]),
                    "high": len([t for t in threats if t.severity == "high"]),
                    "critical": len([t for t in threats if t.severity == "critical"])
                }
            }
        else:
            all_threats = [t for threats in self.threat_records.values() for t in threats]
            return {
                "total_clients": len(self.threat_records),
                "blocked_clients": len(self.blocked_ips),
                "total_threats": len(all_threats),
                "threat_types": list(set(t.threat_type for t in all_threats)),
                "severity_counts": {
                    "low": len([t for t in all_threats if t.severity == "low"]),
                    "medium": len([t for t in all_threats if t.severity == "medium"]),
                    "high": len([t for t in all_threats if t.severity == "high"]),
                    "critical": len([t for t in all_threats if t.severity == "critical"])
                }
            }
    
    def unblock_client(self, client_id: str) -> bool:
        """Manually unblock a client"""
        if client_id in self.blocked_ips:
            self.blocked_ips.remove(client_id)
            self.logger.info(f"Manually unblocked client {client_id}")
            return True
        return False 