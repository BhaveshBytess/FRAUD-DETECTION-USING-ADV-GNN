# hHGTN Demo Service API Specification

## Overview

Lightweight demo service for cryptocurrency fraud detection using heterogeneous graph transformer networks (hHGTN). Provides real-time fraud prediction with explainable AI through subgraph analysis.

**Base URL**: `http://localhost:8000` (development)  
**Framework**: FastAPI with auto-generated docs at `/docs`  
**Model**: hHGTN checkpoint with explainability support  

## Endpoints

### 1. POST /predict

**Purpose**: Predict fraud probability for a transaction with explanations

**Request Schema**:
```json
{
  "transaction": {
    "user_id": "string|int",
    "merchant_id": "string|int", 
    "device_id": "string|int",
    "ip_address": "string",
    "timestamp": "string (ISO 8601)",
    "amount": "float",
    "currency": "string (optional, default USD)",
    "location": "string (optional)",
    "context": {
      "user_history_length": "int (optional)",
      "merchant_category": "string (optional)",
      "is_weekend": "bool (optional)"
    }
  },
  "explain_config": {
    "top_k_nodes": "int (optional, default 30, max 500)",
    "top_k_edges": "int (optional, default 50, max 1000)",
    "explain_method": "string (optional, default 'gnn_explainer')"
  }
}
```

**Response Schema**:
```json
{
  "prediction_prob": "float [0,1]",
  "predicted_label": "string ('fraud'|'legitimate')",
  "confidence": "float [0,1]",
  "explanation": {
    "nodes": [
      {
        "id": "string|int",
        "type": "string ('user'|'merchant'|'device'|'ip')",
        "importance_score": "float [0,1]",
        "features": {
          "risk_level": "string (optional)",
          "activity_count": "int (optional)"
        }
      }
    ],
    "edges": [
      {
        "source": "string|int",
        "target": "string|int", 
        "relation_type": "string ('transaction'|'device_link'|'location_link')",
        "importance_score": "float [0,1]",
        "weight": "float (optional)"
      }
    ],
    "top_features": [
      {
        "feature_name": "string",
        "importance_score": "float [0,1]",
        "value": "string|float (optional)"
      }
    ]
  },
  "meta": {
    "subgraph_nodes": "int",
    "subgraph_edges": "int", 
    "explain_time_ms": "int",
    "explain_timed_out": "bool",
    "explain_error": "string (null if success)",
    "model_version": "string",
    "request_id": "string"
  }
}
```

**Example Request**:
```json
{
  "transaction": {
    "user_id": "user_12345",
    "merchant_id": "merchant_789",
    "device_id": "device_abc123",
    "ip_address": "192.168.1.100",
    "timestamp": "2025-09-14T10:30:00Z",
    "amount": 1500.50,
    "currency": "USD",
    "location": "New York, NY",
    "context": {
      "user_history_length": 45,
      "merchant_category": "electronics",
      "is_weekend": false
    }
  },
  "explain_config": {
    "top_k_nodes": 25,
    "top_k_edges": 40
  }
}
```

**Example Response**:
```json
{
  "prediction_prob": 0.87,
  "predicted_label": "fraud",
  "confidence": 0.92,
  "explanation": {
    "nodes": [
      {
        "id": "user_12345",
        "type": "user",
        "importance_score": 0.95,
        "features": {
          "risk_level": "high",
          "activity_count": 12
        }
      },
      {
        "id": "device_abc123", 
        "type": "device",
        "importance_score": 0.78,
        "features": {
          "risk_level": "medium",
          "activity_count": 8
        }
      }
    ],
    "edges": [
      {
        "source": "user_12345",
        "target": "merchant_789",
        "relation_type": "transaction",
        "importance_score": 0.89,
        "weight": 1500.50
      },
      {
        "source": "user_12345",
        "target": "device_abc123", 
        "relation_type": "device_link",
        "importance_score": 0.72
      }
    ],
    "top_features": [
      {
        "feature_name": "transaction_amount",
        "importance_score": 0.85,
        "value": 1500.50
      },
      {
        "feature_name": "user_velocity",
        "importance_score": 0.73,
        "value": "high"
      }
    ]
  },
  "meta": {
    "subgraph_nodes": 25,
    "subgraph_edges": 40,
    "explain_time_ms": 342,
    "explain_timed_out": false,
    "explain_error": null,
    "model_version": "hHGTN-v1.0.0",
    "request_id": "req_20250914_103000_abc123"
  }
}
```

**Error Responses**:
- `400 Bad Request`: Invalid input payload or missing required fields
- `422 Unprocessable Entity`: Validation errors in request schema  
- `500 Internal Server Error`: Model inference or explainability failure
- `503 Service Unavailable`: Model not loaded or service overloaded

---

### 2. GET /health

**Purpose**: Service health check and model status

**Response Schema**:
```json
{
  "status": "string ('ok'|'degraded'|'error')",
  "model_loaded": "bool",
  "model_version": "string",
  "uptime_seconds": "int",
  "last_prediction_time": "string (ISO 8601, nullable)",
  "memory_usage_mb": "float (optional)",
  "gpu_available": "bool (optional)"
}
```

**Example Response**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "hHGTN-v1.0.0",
  "uptime_seconds": 3600,
  "last_prediction_time": "2025-09-14T10:29:58Z",
  "memory_usage_mb": 2048.5,
  "gpu_available": false
}
```

---

### 3. GET /metrics (Optional)

**Purpose**: Simple runtime metrics for monitoring

**Response Schema**:
```json
{
  "request_count": "int",
  "prediction_count": "int", 
  "avg_latency_ms": "float",
  "avg_explain_time_ms": "float",
  "error_count": "int",
  "uptime_seconds": "int",
  "last_reset_time": "string (ISO 8601)"
}
```

**Example Response**:
```json
{
  "request_count": 245,
  "prediction_count": 238,
  "avg_latency_ms": 425.8,
  "avg_explain_time_ms": 312.4,
  "error_count": 7,
  "uptime_seconds": 7200,
  "last_reset_time": "2025-09-14T08:00:00Z"
}
```

## Configuration

### Environment Variables

- `HHGTN_CKPT`: Path to model checkpoint (default: `./experiments/demo/checkpoint_lite.ckpt`)
- `DEVICE`: Computation device (default: `cpu`, options: `cuda`, `mps`)
- `TOP_K_NODES`: Default explanation node limit (default: `30`, max: `500`)
- `TOP_K_EDGES`: Default explanation edge limit (default: `50`, max: `1000`)
- `EXPLAIN_TIMEOUT`: Explanation timeout in seconds (default: `5`)
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `MAX_PAYLOAD_SIZE`: Request size limit in bytes (default: `102400` = 100KB)

### Model Requirements

- **Checkpoint Format**: PyTorch state dict compatible with hHGTN architecture
- **Required Files**: 
  - `experiments/demo/checkpoint_lite.ckpt` (model weights)
  - `experiments/demo/demo_mappings.json` (entity ID mappings)
  - `experiments/demo/config.yaml` (model hyperparameters)

## Performance Specifications

### Latency Targets
- **Prediction Only**: ≤ 200ms (95th percentile)
- **Prediction + Explanation**: ≤ 2000ms (95th percentile)
- **Health Check**: ≤ 50ms

### Throughput Limits
- **Demo Mode**: 10 requests/minute per client
- **Load Testing**: Up to 100 concurrent requests

### Resource Limits
- **Memory**: ≤ 4GB RAM for model + service
- **CPU**: 2-4 cores recommended
- **Storage**: ≤ 1GB for checkpoint + mappings

## Security & Privacy

### Input Validation
- Maximum payload size: 100KB
- String sanitization for all text inputs
- Numeric bounds checking for amounts and scores
- IP address format validation

### PII Handling
- **Masking**: User IDs, device IDs, IP addresses masked in logs
- **Retention**: No request data stored beyond current session
- **Logging**: Only aggregate metrics and sanitized error messages

### Rate Limiting (Production Recommendation)
- 10 requests/minute per IP in demo mode
- 100 requests/minute per authenticated user
- Circuit breaker for service protection

## Testing & Validation

### Integration Tests
- `tests/test_demo_predict.py`: End-to-end prediction testing
- `tests/test_health.py`: Health endpoint validation  
- `tests/test_edge_cases.py`: Error handling and edge cases

### Sample Payloads
- `demo_service/samples/sample_predict.json`: Valid prediction request
- `demo_service/samples/sample_large.json`: Maximum size payload
- `demo_service/samples/sample_invalid.json`: Invalid request for error testing

## Usage Examples

### cURL Commands

**Health Check**:
```bash
curl -X GET http://localhost:8000/health
```

**Fraud Prediction**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @demo_service/samples/sample_predict.json
```

**Metrics**:
```bash
curl -X GET http://localhost:8000/metrics
```

### Python Client Example

```python
import requests
import json

# Load sample transaction
with open('demo_service/samples/sample_predict.json', 'r') as f:
    payload = json.load(f)

# Make prediction request
response = requests.post(
    'http://localhost:8000/predict',
    json=payload,
    headers={'Content-Type': 'application/json'}
)

result = response.json()
print(f"Fraud Probability: {result['prediction_prob']:.2f}")
print(f"Prediction: {result['predicted_label']}")
print(f"Explanation Nodes: {len(result['explanation']['nodes'])}")
```

## Auto-Generated Documentation

FastAPI automatically generates interactive documentation:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`
