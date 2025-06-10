"""
Rate Limiting Middleware for Avatar Streaming Service
Implements rate limiting for API endpoints and WebSocket connections
"""

import time
import asyncio
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config.settings import settings


@dataclass
class RateLimitRecord:
    """Rate limit record for a client"""
    requests: int = 0
    window_start: float = field(default_factory=time.time)
    last_request: float = field(default_factory=time.time)
    blocked_until: Optional[float] = None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with different limits for different endpoints"""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = logging.getLogger(__name__)
        
        # Rate limit storage: client_id -> endpoint -> RateLimitRecord
        self.rate_limits: Dict[str, Dict[str, RateLimitRecord]] = defaultdict(
            lambda: defaultdict(RateLimitRecord)
        )
        
        # Rate limit configurations for different endpoints
        self.endpoint_limits = {
            # Avatar upload endpoints - stricter limits
            "/avatar/register": {"requests": 5, "window": 300},  # 5 uploads per 5 minutes
            "/avatar/list": {"requests": 30, "window": 60},      # 30 requests per minute
            "/avatar/delete": {"requests": 10, "window": 300},   # 10 deletions per 5 minutes
            
            # Processing endpoints - moderate limits
            "/ws/stream": {"requests": 100, "window": 60},       # 100 messages per minute
            "/health": {"requests": 120, "window": 60},          # 120 health checks per minute
            
            # General API - default limits
            "default": {"requests": settings.rate_limit_requests, "window": settings.rate_limit_window}
        }
        
        # Cleanup task
        self.cleanup_task = None
        self.start_cleanup_task()
    
    def start_cleanup_task(self):
        """Start background cleanup task"""
        async def cleanup_expired_records():
            while True:
                try:
                    await asyncio.sleep(300)  # Cleanup every 5 minutes
                    await self.cleanup_expired_limits()
                except Exception as e:
                    self.logger.error(f"Rate limit cleanup error: {e}")
        
        self.cleanup_task = asyncio.create_task(cleanup_expired_records())
    
    async def cleanup_expired_limits(self):
        """Remove expired rate limit records"""
        current_time = time.time()
        expired_clients = []
        
        for client_id, endpoints in self.rate_limits.items():
            expired_endpoints = []
            
            for endpoint, record in endpoints.items():
                # Remove records older than 1 hour
                if current_time - record.last_request > 3600:
                    expired_endpoints.append(endpoint)
            
            for endpoint in expired_endpoints:
                del endpoints[endpoint]
            
            if not endpoints:
                expired_clients.append(client_id)
        
        for client_id in expired_clients:
            del self.rate_limits[client_id]
        
        if expired_clients or sum(len(eps) for eps in self.rate_limits.values()) > 0:
            self.logger.info(f"Cleaned up {len(expired_clients)} expired clients")
    
    def get_client_identifier(self, request: Request) -> str:
        """Extract client identifier from request"""
        # Try to get client ID from headers first
        client_id = request.headers.get("X-Client-ID")
        if client_id:
            return f"client_{client_id}"
        
        # Fall back to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        return f"ip_{client_ip}"
    
    def get_endpoint_key(self, path: str) -> str:
        """Get rate limit key for endpoint"""
        # Check for exact matches first
        if path in self.endpoint_limits:
            return path
        
        # Check for pattern matches
        if path.startswith("/avatar/"):
            if "/register" in path:
                return "/avatar/register"
            elif "/list" in path:
                return "/avatar/list"
            elif "/delete" in path or path.count("/") >= 3:
                return "/avatar/delete"
        
        return "default"
    
    def check_rate_limit(self, client_id: str, endpoint_key: str) -> Tuple[bool, Dict]:
        """Check if request is within rate limits"""
        current_time = time.time()
        record = self.rate_limits[client_id][endpoint_key]
        config = self.endpoint_limits[endpoint_key]
        
        # Check if client is currently blocked
        if record.blocked_until and current_time < record.blocked_until:
            remaining_block_time = record.blocked_until - current_time
            return False, {
                "error": "rate_limit_exceeded",
                "message": "Rate limit exceeded. Please try again later.",
                "blocked_until": record.blocked_until,
                "remaining_seconds": int(remaining_block_time)
            }
        
        # Reset window if expired
        if current_time - record.window_start >= config["window"]:
            record.requests = 0
            record.window_start = current_time
            record.blocked_until = None
        
        # Check rate limit
        if record.requests >= config["requests"]:
            # Block client for progressively longer periods based on violations
            block_duration = min(300, 60 * (record.requests - config["requests"] + 1))
            record.blocked_until = current_time + block_duration
            
            self.logger.warning(
                f"Rate limit exceeded for client {client_id} on {endpoint_key}. "
                f"Blocked for {block_duration} seconds."
            )
            
            return False, {
                "error": "rate_limit_exceeded", 
                "message": f"Rate limit exceeded. Blocked for {block_duration} seconds.",
                "blocked_until": record.blocked_until,
                "remaining_seconds": block_duration
            }
        
        # Update counters
        record.requests += 1
        record.last_request = current_time
        
        return True, {
            "requests_remaining": config["requests"] - record.requests,
            "window_reset": record.window_start + config["window"],
            "window_seconds": config["window"]
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting"""
        # Skip rate limiting for health checks from localhost
        if (request.url.path == "/health" and 
            request.client and 
            request.client.host in ["127.0.0.1", "localhost"]):
            return await call_next(request)
        
        # Get client identifier and endpoint
        client_id = self.get_client_identifier(request)
        endpoint_key = self.get_endpoint_key(request.url.path)
        
        # Check rate limit
        allowed, rate_info = self.check_rate_limit(client_id, endpoint_key)
        
        if not allowed:
            # Log rate limit violation
            self.logger.warning(
                f"Rate limit exceeded: {client_id} -> {request.url.path} "
                f"({request.method})"
            )
            
            # Return rate limit error
            return JSONResponse(
                status_code=429,
                content=rate_info,
                headers={
                    "Retry-After": str(int(rate_info.get("remaining_seconds", 60))),
                    "X-RateLimit-Limit": str(self.endpoint_limits[endpoint_key]["requests"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(rate_info.get("blocked_until", time.time() + 60)))
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        if hasattr(response, "headers"):
            config = self.endpoint_limits[endpoint_key]
            response.headers["X-RateLimit-Limit"] = str(config["requests"])
            response.headers["X-RateLimit-Remaining"] = str(rate_info["requests_remaining"])
            response.headers["X-RateLimit-Reset"] = str(int(rate_info["window_reset"]))
        
        return response


class WebSocketRateLimiter:
    """Rate limiter specifically for WebSocket connections"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.connections: Dict[str, RateLimitRecord] = {}
        
        # WebSocket specific limits
        self.limits = {
            "messages_per_minute": 60,
            "audio_uploads_per_minute": 10,
            "avatar_selections_per_minute": 5
        }
    
    def check_message_rate_limit(self, client_id: str, message_type: str = "general") -> bool:
        """Check WebSocket message rate limits"""
        current_time = time.time()
        
        if client_id not in self.connections:
            self.connections[client_id] = RateLimitRecord()
        
        record = self.connections[client_id]
        
        # Reset window if expired
        if current_time - record.window_start >= 60:  # 1 minute window
            record.requests = 0
            record.window_start = current_time
        
        # Get limit based on message type
        if message_type == "audio":
            limit = self.limits["audio_uploads_per_minute"]
        elif message_type == "avatar_selection":
            limit = self.limits["avatar_selections_per_minute"]
        else:
            limit = self.limits["messages_per_minute"]
        
        # Check limit
        if record.requests >= limit:
            self.logger.warning(
                f"WebSocket rate limit exceeded for {client_id}: "
                f"{message_type} messages"
            )
            return False
        
        record.requests += 1
        record.last_request = current_time
        return True
    
    def cleanup_connection(self, client_id: str):
        """Clean up connection record"""
        if client_id in self.connections:
            del self.connections[client_id]


# Global WebSocket rate limiter instance
websocket_rate_limiter = WebSocketRateLimiter() 