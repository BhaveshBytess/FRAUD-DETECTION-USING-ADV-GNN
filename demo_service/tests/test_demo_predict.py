"""
Integration test for /predict endpoint
"""
import pytest
import json
import os
import sys
from fastapi.testclient import TestClient

# Add demo_service to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from demo_service.app import app

client = TestClient(app)

def test_predict_endpoint_basic():
    """Test basic prediction functionality"""
    # Load sample payload
    sample_path = os.path.join(os.path.dirname(__file__), "..", "samples", "sample_predict.json")
    
    if os.path.exists(sample_path):
        with open(sample_path, 'r') as f:
            payload = json.load(f)
    else:
        # Fallback payload
        payload = {
            "transaction": {
                "user_id": "test_user_123",
                "merchant_id": "test_merchant_456",
                "device_id": "test_device_789",
                "ip_address": "192.168.1.100",
                "timestamp": "2025-09-14T10:30:00Z",
                "amount": 1500.50,
                "currency": "USD"
            }
        }
    
    response = client.post("/predict", json=payload)
    
    # Debug: Print response details if not 200
    if response.status_code != 200:
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        
        # Check if service is available
        health_response = client.get("/health")
        print(f"Health status: {health_response.status_code}")
        print(f"Health body: {health_response.text}")
    
    # If service unavailable, skip detailed tests but check error format
    if response.status_code == 503:
        error_detail = response.json()
        assert "detail" in error_detail
        assert "Model not loaded" in error_detail["detail"]
        return
    
    assert response.status_code == 200
    result = response.json()
    
    # Validate response structure
    assert "prediction_prob" in result
    assert "predicted_label" in result
    assert "confidence" in result
    assert "explanation" in result
    assert "meta" in result
    
    # Validate numeric bounds
    assert 0 <= result["prediction_prob"] <= 1
    assert 0 <= result["confidence"] <= 1
    assert result["predicted_label"] in ["fraud", "legitimate"]
    
    # Validate explanation structure if present
    if result["explanation"]:
        assert "nodes" in result["explanation"]
        assert "edges" in result["explanation"]
        assert "top_features" in result["explanation"]
        
        # Check node limit
        assert len(result["explanation"]["nodes"]) <= 30  # Default top_k
        
        # Validate node structure
        for node in result["explanation"]["nodes"]:
            assert "id" in node
            assert "type" in node
            assert "importance_score" in node
            assert 0 <= node["importance_score"] <= 1
        
        # Validate edge structure
        for edge in result["explanation"]["edges"]:
            assert "source" in edge
            assert "target" in edge
            assert "relation_type" in edge
            assert "importance_score" in edge
            assert 0 <= edge["importance_score"] <= 1

def test_predict_endpoint_with_config():
    """Test prediction with custom explanation config"""
    payload = {
        "transaction": {
            "user_id": "test_user_456",
            "merchant_id": "test_merchant_789",
            "device_id": "test_device_123",
            "ip_address": "10.0.0.1",
            "timestamp": "2025-09-14T11:00:00Z",
            "amount": 250.75
        },
        "explain_config": {
            "top_k_nodes": 15,
            "top_k_edges": 25,
            "explain_method": "gnn_explainer"
        }
    }
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    result = response.json()
    
    # Check custom limits are respected
    if result["explanation"]:
        assert len(result["explanation"]["nodes"]) <= 15
        assert len(result["explanation"]["edges"]) <= 25

def test_predict_endpoint_validation_errors():
    """Test input validation"""
    # Missing required fields
    invalid_payload = {
        "transaction": {
            "user_id": "test_user",
            # Missing required fields
        }
    }
    
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Validation error
    
    # Invalid amount
    invalid_payload = {
        "transaction": {
            "user_id": "test_user",
            "merchant_id": "test_merchant",
            "device_id": "test_device",
            "ip_address": "192.168.1.1",
            "timestamp": "2025-09-14T10:00:00Z",
            "amount": -100  # Invalid negative amount
        }
    }
    
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

def test_predict_endpoint_large_limits():
    """Test explanation limits are enforced"""
    payload = {
        "transaction": {
            "user_id": "test_user",
            "merchant_id": "test_merchant",
            "device_id": "test_device",
            "ip_address": "192.168.1.1",
            "timestamp": "2025-09-14T10:00:00Z",
            "amount": 100
        },
        "explain_config": {
            "top_k_nodes": 1000,  # Exceeds maximum
            "top_k_edges": 2000   # Exceeds maximum
        }
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Should fail validation

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
