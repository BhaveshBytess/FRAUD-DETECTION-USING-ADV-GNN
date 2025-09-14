#!/usr/bin/env python3
"""
Quick test script to verify the fraud detection demo service is working properly.
Run this after Docker installation to test both the application and container setup.
"""

import requests
import time
import json
from typing import Dict, Any

def test_local_service():
    """Test the service running locally without Docker"""
    print("🧪 Testing Local Service...")
    
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Local service not running: {e}")
        return False
    
    # Test prediction endpoint
    test_data = {
        "user_id": "user123",
        "merchant_id": "merchant456",
        "amount": 100.50,
        "transaction_type": "purchase",
        "timestamp": "2024-01-15T10:30:00Z"
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=test_data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction endpoint working")
            print(f"   Fraud probability: {result['fraud_probability']}")
        else:
            print(f"❌ Prediction endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction test failed: {e}")
        return False
    
    return True

def test_docker_service():
    """Test the service running in Docker container"""
    print("\n🐳 Testing Docker Service...")
    
    import subprocess
    
    # Check if Docker is installed
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker installed: {result.stdout.strip()}")
        else:
            print("❌ Docker not installed or not working")
            return False
    except FileNotFoundError:
        print("❌ Docker not found in PATH")
        return False
    
    # Check if our image exists
    try:
        result = subprocess.run(["docker", "images", "fraud-detection-demo"], capture_output=True, text=True)
        if "fraud-detection-demo" in result.stdout:
            print("✅ Docker image found")
        else:
            print("❌ Docker image not built yet")
            print("   Run: docker build -t fraud-detection-demo ./demo_service")
            return False
    except Exception as e:
        print(f"❌ Error checking Docker images: {e}")
        return False
    
    # Test container service (assuming it's running on port 8001 to avoid conflicts)
    base_url = "http://localhost:8001"
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Docker container service working")
        else:
            print(f"❌ Docker container not responding on port 8001")
    except requests.exceptions.RequestException:
        print("❌ Docker container not running on port 8001")
        print("   Start with: docker run -p 8001:8000 fraud-detection-demo")
        return False
    
    return True

def test_security_features():
    """Test security middleware"""
    print("\n🔒 Testing Security Features...")
    
    base_url = "http://localhost:8000"
    
    # Test security headers
    try:
        response = requests.get(f"{base_url}/health")
        headers = response.headers
        
        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "X-XSS-Protection",
            "Content-Security-Policy"
        ]
        
        for header in security_headers:
            if header in headers:
                print(f"✅ Security header present: {header}")
            else:
                print(f"❌ Missing security header: {header}")
                
    except Exception as e:
        print(f"❌ Security test failed: {e}")
        return False
    
    # Test input validation
    malicious_data = {
        "user_id": "'; DROP TABLE users; --",
        "merchant_id": "merchant123",
        "amount": 100.50,
        "transaction_type": "purchase",
        "timestamp": "2024-01-15T10:30:00Z"
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=malicious_data)
        if response.status_code == 400:
            print("✅ Input validation working - blocked malicious input")
        else:
            print(f"❌ Input validation may be bypassed: {response.status_code}")
    except Exception as e:
        print(f"❌ Input validation test failed: {e}")
        
    return True

def run_performance_test():
    """Basic performance test"""
    print("\n⚡ Running Performance Test...")
    
    base_url = "http://localhost:8000"
    
    # Test multiple requests
    start_time = time.time()
    success_count = 0
    total_requests = 10
    
    test_data = {
        "user_id": "user123",
        "merchant_id": "merchant456",
        "amount": 100.50,
        "transaction_type": "purchase",
        "timestamp": "2024-01-15T10:30:00Z"
    }
    
    for i in range(total_requests):
        try:
            response = requests.post(f"{base_url}/predict", json=test_data, timeout=2)
            if response.status_code == 200:
                success_count += 1
        except:
            pass
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"✅ Performance test results:")
    print(f"   Requests: {total_requests}")
    print(f"   Successful: {success_count}")
    print(f"   Success rate: {(success_count/total_requests)*100:.1f}%")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Avg response time: {(duration/total_requests)*1000:.1f}ms")

def main():
    """Run all tests"""
    print("🚀 Fraud Detection Demo Service Test Suite")
    print("=" * 50)
    
    # Test local service
    local_working = test_local_service()
    
    # Test Docker service
    docker_working = test_docker_service()
    
    # Test security features (if local service is working)
    if local_working:
        test_security_features()
        run_performance_test()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   Local Service: {'✅ Working' if local_working else '❌ Not Working'}")
    print(f"   Docker Setup: {'✅ Working' if docker_working else '❌ Not Working'}")
    
    if local_working and docker_working:
        print("\n🎉 All systems operational! Ready for demo.")
    elif local_working:
        print("\n⚠️  Local service working. Install Docker for containerization.")
    else:
        print("\n❌ Service not running. Start with: uvicorn demo_service.app:app --reload")

if __name__ == "__main__":
    main()
