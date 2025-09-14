"""
pytest configuration for demo service tests.
Provides test fixtures and configurations.
"""
import pytest
import os
from demo_service.security import rate_limit_storage, rate_limit_blocked


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment with disabled rate limiting."""
    # Disable rate limiting for tests
    os.environ["DISABLE_RATE_LIMITING"] = "true"
    yield
    # Clean up
    if "DISABLE_RATE_LIMITING" in os.environ:
        del os.environ["DISABLE_RATE_LIMITING"]


@pytest.fixture(scope="function", autouse=True)
def reset_rate_limiter():
    """Reset rate limiter state before each test."""
    # Clear rate limiter storage
    rate_limit_storage.clear()
    rate_limit_blocked.clear()
    yield

@pytest.fixture
def sample_transaction_data():
    """Sample transaction data for testing."""
    return {
        "user_id": "user123",
        "merchant_id": "merchant456", 
        "amount": 100.50,
        "transaction_type": "purchase",
        "timestamp": "2024-01-15T10:30:00Z"
    }


@pytest.fixture
def large_transaction_data():
    """Large transaction data for stress testing."""
    return {
        "user_id": "user" + "x" * 100,
        "merchant_id": "merchant" + "y" * 100,
        "amount": 999999.99,
        "transaction_type": "large_purchase",
        "timestamp": "2024-01-15T10:30:00Z",
        "metadata": {"description": "x" * 1000}
    }
