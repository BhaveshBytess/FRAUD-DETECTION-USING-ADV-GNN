"""
Performance and load testing for hHGTN Demo Service
"""
import pytest
import asyncio
import time
import concurrent.futures
from fastapi.testclient import TestClient
from datetime import datetime
import statistics

from demo_service.app import app

client = TestClient(app)

class TestPerformance:
    """Performance testing for the demo service"""
    
    def test_health_endpoint_performance(self):
        """Test health endpoint response time"""
        times = []
        
        for _ in range(10):
            start = time.time()
            response = client.get("/health")
            end = time.time()
            
            assert response.status_code == 200
            times.append((end - start) * 1000)  # Convert to ms
        
        avg_time = statistics.mean(times)
        max_time = max(times)
        
        print(f"Health endpoint - Avg: {avg_time:.2f}ms, Max: {max_time:.2f}ms")
        
        # Health endpoint should be very fast
        assert avg_time < 50  # Less than 50ms average
        assert max_time < 200  # Less than 200ms maximum
    
    def test_prediction_endpoint_performance(self):
        """Test prediction endpoint response time"""
        test_request = {
            "transaction": {
                "user_id": "perf_test_user",
                "merchant_id": "perf_test_merchant", 
                "device_id": "perf_test_device",
                "ip_address": "192.168.1.100",
                "timestamp": datetime.now().isoformat(),
                "amount": 100.0,
                "currency": "USD"
            },
            "explain_config": {
                "top_k_nodes": 15,
                "top_k_edges": 20
            }
        }
        
        times = []
        
        for _ in range(5):  # Fewer iterations for prediction tests
            start = time.time()
            response = client.post("/predict", json=test_request)
            end = time.time()
            
            assert response.status_code == 200
            times.append((end - start) * 1000)  # Convert to ms
        
        avg_time = statistics.mean(times)
        max_time = max(times)
        
        print(f"Prediction endpoint - Avg: {avg_time:.2f}ms, Max: {max_time:.2f}ms")
        
        # Prediction should be reasonably fast for demo
        assert avg_time < 500   # Less than 500ms average
        assert max_time < 1000  # Less than 1 second maximum
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        def make_request():
            response = client.get("/health")
            return response.status_code == 200
        
        # Test with 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # All requests should succeed
        assert all(results)
        print(f"Concurrent requests: {len(results)} succeeded")
    
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable under load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Make many requests
        for _ in range(50):
            response = client.get("/health")
            assert response.status_code == 200
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage - Initial: {initial_memory/1024/1024:.1f}MB, "
              f"Final: {final_memory/1024/1024:.1f}MB, "
              f"Increase: {memory_increase/1024/1024:.1f}MB")
        
        # Memory increase should be reasonable (less than 50MB for 50 requests)
        assert memory_increase < 50 * 1024 * 1024

class TestLoadTesting:
    """Load testing scenarios"""
    
    def test_sustained_load(self):
        """Test service under sustained load"""
        success_count = 0
        error_count = 0
        times = []
        
        # Run for 30 seconds with continuous requests
        start_time = time.time()
        while time.time() - start_time < 30:
            request_start = time.time()
            try:
                response = client.get("/health")
                request_end = time.time()
                
                if response.status_code == 200:
                    success_count += 1
                    times.append((request_end - request_start) * 1000)
                else:
                    error_count += 1
                    
            except Exception:
                error_count += 1
            
            # Small delay to prevent overwhelming
            time.sleep(0.1)
        
        total_requests = success_count + error_count
        success_rate = success_count / total_requests if total_requests > 0 else 0
        avg_time = statistics.mean(times) if times else 0
        
        print(f"Sustained load - Requests: {total_requests}, "
              f"Success rate: {success_rate:.2%}, "
              f"Avg response time: {avg_time:.2f}ms")
        
        # Should maintain high success rate under load
        assert success_rate > 0.95  # 95% success rate
        assert avg_time < 100       # Average response time under 100ms
    
    def test_burst_load(self):
        """Test service under burst load"""
        def make_burst_requests():
            results = []
            for _ in range(20):  # 20 requests in rapid succession
                try:
                    response = client.get("/health")
                    results.append(response.status_code == 200)
                except Exception:
                    results.append(False)
            return results
        
        # Execute 5 concurrent bursts (100 total requests)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_burst_requests) for _ in range(5)]
            all_results = []
            for future in futures:
                all_results.extend(future.result())
        
        success_count = sum(all_results)
        success_rate = success_count / len(all_results)
        
        print(f"Burst load - {len(all_results)} requests, "
              f"Success rate: {success_rate:.2%}")
        
        # Should handle burst load reasonably well
        assert success_rate > 0.80  # 80% success rate under burst

class TestStressScenarios:
    """Stress testing with edge cases"""
    
    def test_malformed_request_flood(self):
        """Test handling of many malformed requests"""
        error_responses = []
        
        for _ in range(20):
            # Send various types of malformed requests
            malformed_requests = [
                "",  # Empty body
                "not json",  # Invalid JSON
                {"invalid": "structure"},  # Wrong structure
                None  # None body
            ]
            
            for bad_request in malformed_requests:
                try:
                    if bad_request is None:
                        response = client.post("/predict")
                    else:
                        response = client.post(
                            "/predict", 
                            data=str(bad_request) if bad_request else "",
                            headers={"Content-Type": "application/json"}
                        )
                    error_responses.append(response.status_code)
                except Exception:
                    error_responses.append(500)  # Assume server error
        
        # All should return proper error codes (not 500)
        server_errors = [code for code in error_responses if code == 500]
        error_rate = len(server_errors) / len(error_responses)
        
        print(f"Malformed request handling - Server error rate: {error_rate:.2%}")
        
        # Should handle malformed requests gracefully
        assert error_rate < 0.1  # Less than 10% server errors
    
    def test_large_payload_handling(self):
        """Test handling of unusually large payloads"""
        # Create a large but valid request
        large_request = {
            "transaction": {
                "user_id": "x" * 500,  # 500 character user ID
                "merchant_id": "y" * 500,
                "device_id": "z" * 500,
                "ip_address": "192.168.1.100",
                "timestamp": datetime.now().isoformat(),
                "amount": 100.0,
                "currency": "USD"
            },
            "explain_config": {
                "top_k_nodes": 100,
                "top_k_edges": 200
            }
        }
        
        start_time = time.time()
        response = client.post("/predict", json=large_request)
        response_time = (time.time() - start_time) * 1000
        
        print(f"Large payload handling - Status: {response.status_code}, "
              f"Time: {response_time:.2f}ms")
        
        # Should either process successfully or reject gracefully
        assert response.status_code in [200, 400, 422, 413]  # Valid response codes
        assert response_time < 2000  # Should respond within 2 seconds

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to see print statements
