"""
Comprehensive test suite for hHGTN Demo Service security features
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import time

from demo_service.app import app
from demo_service.security import (
    RateLimitMiddleware, 
    SecurityHeadersMiddleware,
    validate_transaction_limits,
    validate_ip_address
)

client = TestClient(app)

class TestSecurityMiddleware:
    """Test security middleware functionality"""
    
    def test_security_headers_added(self):
        """Test that security headers are properly added"""
        response = client.get("/health")
        
        # Check security headers
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        assert "X-XSS-Protection" in response.headers
        assert "Content-Security-Policy" in response.headers
        
    def test_server_header_removed(self):
        """Test that server header is removed for security"""
        response = client.get("/health")
        assert "server" not in response.headers

class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def test_normal_request_rate(self):
        """Test that normal request rates are allowed"""
        # Make a few requests - should all succeed
        for i in range(5):
            response = client.get("/health")
            assert response.status_code == 200
            
        # Check rate limit headers are present
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers
    
    def test_rate_limit_headers_present(self):
        """Test that rate limit headers are included"""
        response = client.get("/health")
        
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers
        
        # Check header values are reasonable
        limit = int(response.headers["X-RateLimit-Limit"])
        remaining = int(response.headers["X-RateLimit-Remaining"])
        assert limit > 0
        assert 0 <= remaining <= limit

class TestInputValidation:
    """Test input validation and sanitization"""
    
    def test_transaction_amount_validation(self):
        """Test transaction amount validation"""
        # Test negative amount
        transaction = {"amount": -100, "user_id": "user123", "merchant_id": "merch456", "device_id": "device789"}
        error = validate_transaction_limits(transaction)
        assert error == "Transaction amount must be positive"
        
        # Test zero amount
        transaction["amount"] = 0
        error = validate_transaction_limits(transaction)
        assert error == "Transaction amount must be positive"
        
        # Test excessive amount
        transaction["amount"] = 2_000_000  # Above $1M limit
        error = validate_transaction_limits(transaction)
        assert error == "Transaction amount exceeds maximum limit"
        
        # Test valid amount
        transaction["amount"] = 100.50
        error = validate_transaction_limits(transaction)
        assert error is None
    
    def test_suspicious_pattern_detection(self):
        """Test detection of suspicious patterns in input"""
        # Test SQL injection patterns
        transaction = {
            "amount": 100,
            "user_id": "'; DROP TABLE users; --",
            "merchant_id": "merchant123",
            "device_id": "device456"
        }
        error = validate_transaction_limits(transaction)
        assert "Invalid user_id format" in error
        
        # Test XSS patterns
        transaction = {
            "amount": 100,
            "user_id": "user123",
            "merchant_id": "<script>alert('xss')</script>",
            "device_id": "device456"
        }
        error = validate_transaction_limits(transaction)
        assert "Invalid merchant_id format" in error
        
        # Test admin patterns
        transaction = {
            "amount": 100,
            "user_id": "test",
            "merchant_id": "test",
            "device_id": "admin_root_test"
        }
        error = validate_transaction_limits(transaction)
        assert "Invalid device_id format" in error
    
    def test_ip_address_validation(self):
        """Test IP address validation"""
        # Valid IPv4
        assert validate_ip_address("192.168.1.1") == True
        assert validate_ip_address("10.0.0.1") == True
        assert validate_ip_address("127.0.0.1") == True
        
        # Valid IPv6
        assert validate_ip_address("::1") == True
        assert validate_ip_address("2001:db8::1") == True
        
        # Invalid IPs
        assert validate_ip_address("256.256.256.256") == False
        assert validate_ip_address("not.an.ip.address") == False
        assert validate_ip_address("") == False

class TestAPIEndpoints:
    """Test API endpoints with security considerations"""
    
    def test_predict_endpoint_input_validation(self):
        """Test prediction endpoint input validation"""
        # Valid transaction
        valid_request = {
            "transaction": {
                "user_id": "user_test_123",
                "merchant_id": "merchant_test_456", 
                "device_id": "device_test_789",
                "ip_address": "192.168.1.100",
                "timestamp": datetime.now().isoformat(),
                "amount": 100.0,
                "currency": "USD"
            },
            "explain_config": {
                "top_k_nodes": 10,
                "top_k_edges": 15
            }
        }
        
        response = client.post("/predict", json=valid_request)
        assert response.status_code == 200
        
        # Invalid amount (negative)
        invalid_request = valid_request.copy()
        invalid_request["transaction"]["amount"] = -50.0
        
        response = client.post("/predict", json=invalid_request)
        assert response.status_code == 400
        assert "validation failed" in response.json()["detail"].lower()
        
        # Invalid IP address
        invalid_request = valid_request.copy()
        invalid_request["transaction"]["ip_address"] = "not.an.ip"
        
        response = client.post("/predict", json=invalid_request)
        assert response.status_code == 422  # Pydantic validation error
    
    def test_predict_endpoint_suspicious_input(self):
        """Test prediction endpoint with suspicious input"""
        suspicious_request = {
            "transaction": {
                "user_id": "'; DROP TABLE transactions; --",
                "merchant_id": "merchant_test_456", 
                "device_id": "device_test_789",
                "ip_address": "192.168.1.100",
                "timestamp": datetime.now().isoformat(),
                "amount": 100.0,
                "currency": "USD"
            },
            "explain_config": {
                "top_k_nodes": 10,
                "top_k_edges": 15
            }
        }
        
        response = client.post("/predict", json=suspicious_request)
        assert response.status_code == 400
        assert "validation failed" in response.json()["detail"].lower()
    
    def test_health_endpoint_security(self):
        """Test health endpoint security"""
        response = client.get("/health")
        assert response.status_code == 200
        
        # Check that sensitive information is not exposed
        health_data = response.json()
        assert "status" in health_data
        assert "uptime_seconds" in health_data
        
        # Should not expose internal paths or credentials
        health_json = response.text
        assert "password" not in health_json.lower()
        assert "secret" not in health_json.lower()
        assert "token" not in health_json.lower()
    
    def test_metrics_endpoint_security(self):
        """Test metrics endpoint security"""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        metrics_data = response.json()
        assert "total_predictions" in metrics_data
        assert "uptime_seconds" in metrics_data
        
        # Should not expose sensitive system information
        metrics_json = response.text
        assert "password" not in metrics_json.lower()
        assert "secret" not in metrics_json.lower()

class TestErrorHandling:
    """Test comprehensive error handling"""
    
    def test_malformed_json_request(self):
        """Test handling of malformed JSON"""
        response = client.post(
            "/predict",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields"""
        incomplete_request = {
            "transaction": {
                "user_id": "test",
                # Missing required fields
            }
        }
        
        response = client.post("/predict", json=incomplete_request)
        assert response.status_code == 422
    
    def test_invalid_data_types(self):
        """Test handling of invalid data types"""
        invalid_request = {
            "transaction": {
                "user_id": "user_test_123",
                "merchant_id": "merchant_test_456", 
                "device_id": "device_test_789",
                "ip_address": "192.168.1.100",
                "timestamp": "not-a-timestamp",
                "amount": "not-a-number",  # Should be float
                "currency": "USD"
            },
            "explain_config": {
                "top_k_nodes": "not-a-number",  # Should be int
                "top_k_edges": 15
            }
        }
        
        response = client.post("/predict", json=invalid_request)
        assert response.status_code == 422

class TestPerformanceAndLimits:
    """Test performance and system limits"""
    
    def test_large_request_handling(self):
        """Test handling of unusually large requests"""
        # Test with very long strings
        large_request = {
            "transaction": {
                "user_id": "x" * 1000,  # Very long user ID
                "merchant_id": "merchant_test_456", 
                "device_id": "device_test_789",
                "ip_address": "192.168.1.100",
                "timestamp": datetime.now().isoformat(),
                "amount": 100.0,
                "currency": "USD"
            },
            "explain_config": {
                "top_k_nodes": 10,
                "top_k_edges": 15
            }
        }
        
        response = client.post("/predict", json=large_request)
        # Should either process or reject gracefully
        assert response.status_code in [200, 400, 422]
    
    def test_extreme_config_values(self):
        """Test handling of extreme configuration values"""
        extreme_request = {
            "transaction": {
                "user_id": "user_test_123",
                "merchant_id": "merchant_test_456", 
                "device_id": "device_test_789",
                "ip_address": "192.168.1.100",
                "timestamp": datetime.now().isoformat(),
                "amount": 100.0,
                "currency": "USD"
            },
            "explain_config": {
                "top_k_nodes": 10000,  # Very large value
                "top_k_edges": 50000   # Very large value
            }
        }
        
        response = client.post("/predict", json=extreme_request)
        # Should be limited by Pydantic validators
        assert response.status_code in [200, 422]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
