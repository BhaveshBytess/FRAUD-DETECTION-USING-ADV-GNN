#!/usr/bin/env python3
"""
Quick demo script to test the hHGTN fraud detection API
"""
import requests
import json
import time

def test_demo_api():
    base_url = "http://127.0.0.1:8006"
    
    print("🚀 Testing hHGTN Fraud Detection Demo API")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing Health Endpoint:")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Service is healthy!")
            print(f"   Status: {health_data.get('status')}")
            print(f"   Timestamp: {health_data.get('timestamp')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test 2: Sample fraud prediction
    print("\n2. Testing Fraud Prediction:")
    sample_transaction = {
        "transaction": {
            "user_id": "user_suspicious_001",
            "merchant_id": "merchant_high_risk",
            "device_id": "device_unknown_999", 
            "ip_address": "10.0.0.1",
            "timestamp": "2025-09-14T13:30:00Z",
            "amount": 9999.99,  # Suspiciously high amount
            "currency": "USD",
            "location": "Unknown",
            "context": {
                "user_history_length": 1,  # New user
                "merchant_category": "crypto",
                "is_weekend": False
            }
        },
        "explain_config": {
            "top_k_nodes": 10,
            "top_k_edges": 15,
            "explain_method": "gnn_explainer"
        }
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict", 
            json=sample_transaction,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful!")
            print(f"   Fraud Probability: {result.get('fraud_probability', 0):.1%}")
            print(f"   Predicted Label: {result.get('predicted_label', 'unknown').upper()}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            
            # Show explanation if available
            explanation = result.get('explanation', {})
            if explanation:
                print(f"   Risk Factors: {len(explanation.get('risk_factors', []))} identified")
                print(f"   Important Nodes: {len(explanation.get('important_nodes', []))} in subgraph")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
    
    # Test 3: Legitimate transaction
    print("\n3. Testing Legitimate Transaction:")
    legitimate_transaction = {
        "transaction": {
            "user_id": "user_trusted_123",
            "merchant_id": "merchant_verified",
            "device_id": "device_known_456",
            "ip_address": "192.168.1.100",
            "timestamp": "2025-09-14T13:30:00Z", 
            "amount": 25.99,  # Normal amount
            "currency": "USD",
            "location": "New York, NY",
            "context": {
                "user_history_length": 150,  # Established user
                "merchant_category": "grocery",
                "is_weekend": False
            }
        },
        "explain_config": {
            "top_k_nodes": 5,
            "top_k_edges": 8,
            "explain_method": "gnn_explainer"
        }
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=legitimate_transaction,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful!")
            print(f"   Fraud Probability: {result.get('fraud_probability', 0):.1%}")
            print(f"   Predicted Label: {result.get('predicted_label', 'unknown').upper()}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
    
    # Test 4: Metrics endpoint
    print("\n4. Testing Metrics Endpoint:")
    try:
        response = requests.get(f"{base_url}/metrics", timeout=5)
        if response.status_code == 200:
            metrics = response.json()
            print(f"✅ Metrics retrieved!")
            print(f"   Total Predictions: {metrics.get('total_predictions', 0)}")
            print(f"   Fraud Detected: {metrics.get('fraud_predictions', 0)}")
            print(f"   Average Response Time: {metrics.get('avg_response_time', 0):.3f}s")
        else:
            print(f"❌ Metrics failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Metrics error: {e}")
    
    print("\n🎉 Demo testing complete!")
    print(f"💻 Web Interface: {base_url}")
    print(f"📚 API Documentation: {base_url}/docs")
    print(f"🔍 Interactive Demo: {base_url}/static/index.html")

if __name__ == "__main__":
    test_demo_api()
