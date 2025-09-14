"""
FastAPI Demo Service for hHGTN Fraud Detection
Lightweight service for cryptocurrency fraud prediction with explainability
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import time
import logging
from typing import Dict, Any
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from demo_service.schema import (
    PredictRequest, 
    PredictResponse, 
    HealthResponse,
    MetricsResponse
)
from demo_service.model_loader import ModelLoader
from demo_service.config import Config
from demo_service.security import (
    RateLimitMiddleware,
    SecurityHeadersMiddleware, 
    validate_transaction_limits,
    log_security_event
)# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="hHGTN Fraud Detection Demo Service",
    description="Cryptocurrency fraud detection using heterogeneous graph transformer networks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for demo purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=30, requests_per_hour=500)

# Mount static files for web interface
import os
static_dir = os.path.join(os.path.dirname(__file__), "static")
try:
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.info(f"Static files mounted at /static from {static_dir}")
    else:
        logger.warning(f"Static directory not found: {static_dir}")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

# Initialize global state
config = Config()
model_loader = None
service_metrics = {
    "request_count": 0,
    "prediction_count": 0,
    "error_count": 0,
    "total_latency_ms": 0.0,
    "total_explain_time_ms": 0.0,
    "start_time": time.time(),
    "last_prediction_time": None
}

@app.on_event("startup")
async def startup_event():
    """Initialize model loader on service startup"""
    global model_loader
    logger.info("Starting hHGTN Demo Service...")
    
    try:
        model_loader = ModelLoader(config)
        success = model_loader.load_model()
        
        if success:
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model loading failed - service will run with limited functionality")
            
    except Exception as e:
        logger.error(f"Failed to initialize model loader: {e}")
        model_loader = None

def get_model_loader():
    """Get or create model loader (for testing)"""
    global model_loader
    if model_loader is None:
        try:
            model_loader = ModelLoader(config)
            model_loader.load_model()
        except Exception as e:
            logger.error(f"Failed to create model loader: {e}")
    return model_loader

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log requests and track metrics (with PII masking)"""
    start_time = time.time()
    
    # Mask PII in URL/headers for logging
    masked_url = str(request.url).replace(request.url.hostname or "", "***")
    logger.info(f"Request: {request.method} {masked_url}")
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    service_metrics["request_count"] += 1
    service_metrics["total_latency_ms"] += process_time
    
    logger.info(f"Response: {response.status_code} ({process_time:.1f}ms)")
    return response

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Service health check endpoint"""
    try:
        model_loaded = model_loader is not None and model_loader.is_loaded()
        memory_usage = None
        
        # Try to get memory usage if psutil available
        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            pass
        
        uptime = int(time.time() - service_metrics["start_time"])
        
        return HealthResponse(
            status="ok" if model_loaded else "degraded",
            model_loaded=model_loaded,
            model_version=config.MODEL_VERSION,
            uptime_seconds=uptime,
            last_prediction_time=service_metrics["last_prediction_time"],
            memory_usage_mb=memory_usage,
            gpu_available=config.DEVICE != "cpu"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Service metrics endpoint"""
    try:
        uptime = int(time.time() - service_metrics["start_time"])
        avg_latency = (service_metrics["total_latency_ms"] / 
                      max(service_metrics["request_count"], 1))
        avg_explain_time = (service_metrics["total_explain_time_ms"] / 
                           max(service_metrics["prediction_count"], 1))
        
        return MetricsResponse(
            request_count=service_metrics["request_count"],
            prediction_count=service_metrics["prediction_count"],
            avg_latency_ms=round(avg_latency, 2),
            avg_explain_time_ms=round(avg_explain_time, 2),
            error_count=service_metrics["error_count"],
            uptime_seconds=uptime,
            last_reset_time=None  # Could implement reset functionality
        )
        
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

@app.post("/predict", response_model=PredictResponse)
async def predict_fraud(request: PredictRequest):
    """Main fraud prediction endpoint with explanations"""
    # Ensure model is loaded (for testing)
    current_model_loader = model_loader or get_model_loader()
    
    if not current_model_loader or not current_model_loader.is_loaded():
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded - service unavailable"
        )
    
    start_time = time.time()
    request_id = f"req_{int(time.time())}_{hash(str(request.transaction.user_id)) % 10000:04d}"
    
    try:
        # Update metrics
        service_metrics["prediction_count"] += 1
        
        # Run prediction with explanation
        result = current_model_loader.predict_with_explanation(
            transaction=request.transaction.model_dump(),
            explain_config=request.explain_config.model_dump() if request.explain_config else {}
        )
        
        # Track explanation time
        explain_time_ms = result.get("meta", {}).get("explain_time_ms", 0)
        service_metrics["total_explain_time_ms"] += explain_time_ms
        service_metrics["last_prediction_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Add request metadata
        result["meta"]["request_id"] = request_id
        result["meta"]["model_version"] = config.MODEL_VERSION
        
        logger.info(f"Prediction completed for {request_id}: {result['predicted_label']} "
                   f"(prob={result['prediction_prob']:.3f}, explain={explain_time_ms}ms)")
        
        return PredictResponse(**result)
        
    except Exception as e:
        service_metrics["error_count"] += 1
        logger.error(f"Prediction failed for {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    service_metrics["error_count"] += 1
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Root endpoint redirects to static demo page
@app.get("/")
async def root():
    """Redirect to demo interface"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "app:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )
