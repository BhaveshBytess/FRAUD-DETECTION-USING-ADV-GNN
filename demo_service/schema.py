"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List, Any, Union
from datetime import datetime

class TransactionContext(BaseModel):
    user_history_length: Optional[int] = Field(None, ge=0, description="Number of previous transactions")
    merchant_category: Optional[str] = Field(None, max_length=50, description="Merchant business category")
    is_weekend: Optional[bool] = Field(None, description="Whether transaction occurs on weekend")

class Transaction(BaseModel):
    user_id: Union[str, int] = Field(..., description="User identifier")
    merchant_id: Union[str, int] = Field(..., description="Merchant identifier") 
    device_id: Union[str, int] = Field(..., description="Device identifier")
    ip_address: str = Field(..., max_length=45, description="IP address (IPv4/IPv6)")
    timestamp: str = Field(..., description="Transaction timestamp (ISO 8601)")
    amount: float = Field(..., ge=0, le=1000000, description="Transaction amount")
    currency: Optional[str] = Field("USD", max_length=3, description="Currency code")
    location: Optional[str] = Field(None, max_length=100, description="Transaction location")
    context: Optional[TransactionContext] = None

    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError('Invalid timestamp format - use ISO 8601')
        return v

    @field_validator('ip_address')
    @classmethod
    def validate_ip_address(cls, v):
        import ipaddress
        try:
            ipaddress.ip_address(v)
        except ValueError:
            raise ValueError('Invalid IP address format')
        return v

class ExplainConfig(BaseModel):
    top_k_nodes: Optional[int] = Field(30, ge=1, le=500, description="Maximum explanation nodes")
    top_k_edges: Optional[int] = Field(50, ge=1, le=1000, description="Maximum explanation edges") 
    explain_method: Optional[str] = Field("gnn_explainer", description="Explanation method")

class PredictRequest(BaseModel):
    transaction: Transaction
    explain_config: Optional[ExplainConfig] = None

class ExplanationNode(BaseModel):
    id: Union[str, int] = Field(..., description="Node identifier")
    type: str = Field(..., description="Node type (user/merchant/device/ip)")
    importance_score: float = Field(..., ge=0, le=1, description="Node importance [0,1]")
    features: Optional[Dict[str, Any]] = Field(None, description="Additional node features")

class ExplanationEdge(BaseModel):
    source: Union[str, int] = Field(..., description="Source node ID")
    target: Union[str, int] = Field(..., description="Target node ID")
    relation_type: str = Field(..., description="Edge type (transaction/device_link/location_link)")
    importance_score: float = Field(..., ge=0, le=1, description="Edge importance [0,1]")
    weight: Optional[float] = Field(None, description="Edge weight (optional)")

class TopFeature(BaseModel):
    feature_name: str = Field(..., description="Feature name")
    importance_score: float = Field(..., ge=0, le=1, description="Feature importance [0,1]")
    value: Optional[Union[str, float]] = Field(None, description="Feature value")

class Explanation(BaseModel):
    nodes: List[ExplanationNode] = Field(..., description="Important nodes in explanation")
    edges: List[ExplanationEdge] = Field(..., description="Important edges in explanation")
    top_features: List[TopFeature] = Field(..., description="Top contributing features")

class PredictMeta(BaseModel):
    subgraph_nodes: int = Field(..., ge=0, description="Number of nodes in subgraph")
    subgraph_edges: int = Field(..., ge=0, description="Number of edges in subgraph")
    explain_time_ms: int = Field(..., ge=0, description="Explanation computation time")
    explain_timed_out: bool = Field(..., description="Whether explanation timed out")
    explain_error: Optional[str] = Field(None, description="Explanation error message")
    model_version: str = Field(..., description="Model version")
    request_id: str = Field(..., description="Unique request identifier")

class PredictResponse(BaseModel):
    prediction_prob: float = Field(..., ge=0, le=1, description="Fraud probability [0,1]")
    predicted_label: str = Field(..., description="Prediction label (fraud/legitimate)")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence [0,1]")
    explanation: Optional[Explanation] = Field(None, description="Explanation subgraph")
    meta: PredictMeta = Field(..., description="Request metadata")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status (ok/degraded/error)")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str = Field(..., description="Model version")
    uptime_seconds: int = Field(..., ge=0, description="Service uptime in seconds")
    last_prediction_time: Optional[str] = Field(None, description="Last prediction timestamp")
    memory_usage_mb: Optional[float] = Field(None, ge=0, description="Memory usage in MB")
    gpu_available: Optional[bool] = Field(None, description="GPU availability")

class MetricsResponse(BaseModel):
    request_count: int = Field(..., ge=0, description="Total request count")
    prediction_count: int = Field(..., ge=0, description="Total prediction count")
    avg_latency_ms: float = Field(..., ge=0, description="Average request latency")
    avg_explain_time_ms: float = Field(..., ge=0, description="Average explanation time")
    error_count: int = Field(..., ge=0, description="Total error count")
    uptime_seconds: int = Field(..., ge=0, description="Service uptime")
    last_reset_time: Optional[str] = Field(None, description="Last metrics reset time")
