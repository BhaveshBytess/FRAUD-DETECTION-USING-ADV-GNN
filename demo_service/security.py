"""
Security middleware and utilities for the hHGTN Demo Service
"""
import time
import hashlib
import os
from typing import Dict, Any, Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Rate limiting storage (in production, use Redis)
rate_limit_storage = defaultdict(list)
rate_limit_blocked = set()

class RateLimitMiddleware:
    """Simple in-memory rate limiting middleware"""
    
    def __init__(
        self, 
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        block_duration: int = 300  # 5 minutes
    ):
        self.app = app
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.block_duration = block_duration
    
    def get_client_id(self, scope) -> str:
        """Get client identifier from ASGI scope"""
        client_info = scope.get("client", ("unknown", 0))
        client_ip = client_info[0] if client_info else "unknown"
        
        # Get user agent from headers
        headers = dict(scope.get("headers", []))
        user_agent = headers.get(b"user-agent", b"").decode("latin1", errors="ignore")
        
        client_hash = hashlib.md5(f"{client_ip}:{user_agent}".encode()).hexdigest()[:16]
        return f"{client_ip}_{client_hash}"
    
    async def __call__(self, scope, receive, send):
        # Check if rate limiting is disabled for tests
        if os.getenv("DISABLE_RATE_LIMITING") == "true":
            # Still add headers for testing but skip rate limiting logic
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers", []))
                    headers[b"x-ratelimit-limit"] = b"1000"
                    headers[b"x-ratelimit-remaining"] = b"999"
                    headers[b"x-ratelimit-reset"] = str(int(time.time()) + 3600).encode()
                    message["headers"] = list(headers.items())
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
            return
            
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        client_id = self.get_client_id(scope)
        current_time = datetime.now()
        
        # Check if client is blocked
        if client_id in rate_limit_blocked:
            logger.warning(f"Blocked client attempted request: {client_id}")
            response = JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later.",
                    "retry_after": self.block_duration
                }
            )
            await response(scope, receive, send)
            return
        
        # Get request history for this client
        client_requests = rate_limit_storage[client_id]
        
        # Clean old requests (older than 1 hour)
        cutoff_time = current_time - timedelta(hours=1)
        client_requests[:] = [req_time for req_time in client_requests if req_time > cutoff_time]
        
        # Check hourly limit
        if len(client_requests) >= self.requests_per_hour:
            logger.warning(f"Client exceeded hourly limit: {client_id}")
            rate_limit_blocked.add(client_id)
            # Schedule unblock
            asyncio.create_task(self._unblock_client(client_id))
            
            response = JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": "Hourly request limit exceeded",
                    "retry_after": self.block_duration
                }
            )
            await response(scope, receive, send)
            return
        
        # Check per-minute limit
        minute_cutoff = current_time - timedelta(minutes=1)
        recent_requests = [req_time for req_time in client_requests if req_time > minute_cutoff]
        
        if len(recent_requests) >= self.requests_per_minute:
            logger.warning(f"Client exceeded per-minute limit: {client_id}")
            response = JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded", 
                    "message": "Too many requests per minute",
                    "retry_after": 60
                }
            )
            await response(scope, receive, send)
            return
        
        # Record this request
        client_requests.append(current_time)
        
        # Add rate limit headers to response
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                
                rate_headers = [
                    (b"x-ratelimit-limit", str(self.requests_per_minute).encode()),
                    (b"x-ratelimit-remaining", str(max(0, self.requests_per_minute - len(recent_requests) - 1)).encode()),
                    (b"x-ratelimit-reset", str(int((minute_cutoff + timedelta(minutes=1)).timestamp())).encode()),
                ]
                
                headers.extend(rate_headers)
                message["headers"] = headers
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)
    
    async def _unblock_client(self, client_id: str):
        """Unblock client after block duration"""
        await asyncio.sleep(self.block_duration)
        rate_limit_blocked.discard(client_id)
        logger.info(f"Client unblocked: {client_id}")

class SecurityHeadersMiddleware:
    """Add security headers to responses"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Add security headers
                headers = list(message.get("headers", []))
                
                security_headers = [
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"), 
                    (b"x-xss-protection", b"1; mode=block"),
                    (b"referrer-policy", b"strict-origin-when-cross-origin"),
                    (b"content-security-policy", 
                     b"default-src 'self'; script-src 'self' 'unsafe-inline' https://d3js.org; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'"),
                ]
                
                # Add security headers
                for header in security_headers:
                    headers.append(header)
                
                # Remove server header if present
                headers = [h for h in headers if h[0].lower() != b"server"]
                
                message["headers"] = headers
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)

def validate_transaction_limits(transaction: Dict[str, Any]) -> Optional[str]:
    """Validate transaction data for suspicious patterns"""
    amount = transaction.get("amount", 0)
    
    # Check for suspicious amounts
    if amount <= 0:
        return "Transaction amount must be positive"
    
    if amount > 1_000_000:  # $1M limit for demo
        return "Transaction amount exceeds maximum limit"
    
    # Check for suspicious patterns
    user_id = transaction.get("user_id", "")
    merchant_id = transaction.get("merchant_id", "")
    device_id = transaction.get("device_id", "")
    
    # Basic validation patterns
    suspicious_patterns = ["test", "admin", "root", "null", "undefined", "<script", "SELECT", "DROP"]
    
    for field, value in [("user_id", user_id), ("merchant_id", merchant_id), ("device_id", device_id)]:
        if any(pattern.lower() in str(value).lower() for pattern in suspicious_patterns):
            logger.warning(f"Suspicious pattern detected in {field}: {value}")
            return f"Invalid {field} format"
    
    return None

def log_security_event(event_type: str, client_id: str, details: Dict[str, Any]):
    """Log security-related events"""
    logger.warning(f"SECURITY_EVENT: {event_type} | Client: {client_id} | Details: {details}")

# Input sanitization utilities
def sanitize_string(value: str, max_length: int = 255) -> str:
    """Sanitize string input"""
    if not isinstance(value, str):
        return str(value)[:max_length]
    
    # Remove potentially dangerous characters
    cleaned = value.replace('<', '').replace('>', '').replace('"', '').replace("'", '')
    return cleaned[:max_length]

def validate_ip_address(ip: str) -> bool:
    """Validate IP address format"""
    try:
        import ipaddress
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False
