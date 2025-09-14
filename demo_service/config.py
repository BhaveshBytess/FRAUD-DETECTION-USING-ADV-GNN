"""
Configuration management for demo service
"""
import os
from typing import Optional

class Config:
    """Configuration class for demo service"""
    
    # Model configuration
    CHECKPOINT_PATH: str = os.getenv(
        "HHGTN_CKPT", 
        "./experiments/demo/checkpoint_lite.ckpt"
    )
    
    MAPPINGS_PATH: str = os.getenv(
        "HHGTN_MAPPINGS",
        "./experiments/demo/demo_mappings.json"
    )
    
    MODEL_CONFIG_PATH: str = os.getenv(
        "HHGTN_CONFIG",
        "./experiments/demo/config.yaml"
    )
    
    # Device configuration
    DEVICE: str = os.getenv("DEVICE", "cpu")
    
    # Explanation configuration
    TOP_K_NODES: int = int(os.getenv("TOP_K_NODES", "30"))
    TOP_K_EDGES: int = int(os.getenv("TOP_K_EDGES", "50"))
    EXPLAIN_TIMEOUT: int = int(os.getenv("EXPLAIN_TIMEOUT", "5"))
    
    # Service configuration
    MAX_PAYLOAD_SIZE: int = int(os.getenv("MAX_PAYLOAD_SIZE", "102400"))  # 100KB
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Model metadata
    MODEL_VERSION: str = "hHGTN-v1.0.0"
    
    # Validation limits
    MAX_SUBGRAPH_NODES: int = 500
    MAX_SUBGRAPH_EDGES: int = 1000
    
    def __init__(self):
        """Initialize configuration and validate paths"""
        self.validate_config()
    
    def validate_config(self):
        """Validate configuration values"""
        # Validate device
        if self.DEVICE not in ["cpu", "cuda", "mps"]:
            raise ValueError(f"Invalid device: {self.DEVICE}")
        
        # Validate limits
        if self.TOP_K_NODES > self.MAX_SUBGRAPH_NODES:
            raise ValueError(f"TOP_K_NODES ({self.TOP_K_NODES}) exceeds maximum ({self.MAX_SUBGRAPH_NODES})")
        
        if self.TOP_K_EDGES > self.MAX_SUBGRAPH_EDGES:
            raise ValueError(f"TOP_K_EDGES ({self.TOP_K_EDGES}) exceeds maximum ({self.MAX_SUBGRAPH_EDGES})")
    
    def get_checkpoint_info(self) -> dict:
        """Get checkpoint file information"""
        info = {
            "checkpoint_path": self.CHECKPOINT_PATH,
            "exists": os.path.exists(self.CHECKPOINT_PATH),
            "size_mb": None
        }
        
        if info["exists"]:
            try:
                size_bytes = os.path.getsize(self.CHECKPOINT_PATH)
                info["size_mb"] = round(size_bytes / 1024 / 1024, 2)
            except OSError:
                pass
        
        return info
    
    def get_mappings_info(self) -> dict:
        """Get mappings file information"""
        info = {
            "mappings_path": self.MAPPINGS_PATH,
            "exists": os.path.exists(self.MAPPINGS_PATH),
            "size_kb": None
        }
        
        if info["exists"]:
            try:
                size_bytes = os.path.getsize(self.MAPPINGS_PATH)
                info["size_kb"] = round(size_bytes / 1024, 2)
            except OSError:
                pass
        
        return info
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            "checkpoint_path": self.CHECKPOINT_PATH,
            "mappings_path": self.MAPPINGS_PATH,
            "model_config_path": self.MODEL_CONFIG_PATH,
            "device": self.DEVICE,
            "top_k_nodes": self.TOP_K_NODES,
            "top_k_edges": self.TOP_K_EDGES,
            "explain_timeout": self.EXPLAIN_TIMEOUT,
            "max_payload_size": self.MAX_PAYLOAD_SIZE,
            "log_level": self.LOG_LEVEL,
            "model_version": self.MODEL_VERSION,
            "max_subgraph_nodes": self.MAX_SUBGRAPH_NODES,
            "max_subgraph_edges": self.MAX_SUBGRAPH_EDGES
        }
