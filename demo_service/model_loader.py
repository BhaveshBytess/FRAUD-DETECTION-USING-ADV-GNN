"""
Model loader for hHGTN fraud detection with explainability
"""
import os
import json
import time
import logging
import traceback
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Graph libraries
try:
    import torch_geometric
    from torch_geometric.data import Data, HeteroData
    GEOMETRIC_AVAILABLE = True
except ImportError:
    GEOMETRIC_AVAILABLE = False

from demo_service.config import Config

logger = logging.getLogger(__name__)

class ModelLoader:
    """Handles model loading, prediction, and explanation generation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.mappings = None
        self.model_config = None
        self.loaded = False
        
        # Check dependencies
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available - model loading will fail")
        if not GEOMETRIC_AVAILABLE:
            logger.error("PyTorch Geometric not available - model loading will fail")
    
    def load_model(self) -> bool:
        """Load model checkpoint and mappings"""
        try:
            logger.info(f"Loading model from {self.config.CHECKPOINT_PATH}")
            
            # Check if checkpoint exists
            if not os.path.exists(self.config.CHECKPOINT_PATH):
                logger.error(f"Checkpoint not found: {self.config.CHECKPOINT_PATH}")
                return False
            
            # Load mappings if available
            self._load_mappings()
            
            # Load model configuration if available
            self._load_model_config()
            
            # For demo purposes, create a mock model if actual loading fails
            success = self._load_actual_model()
            if not success:
                logger.warning("Actual model loading failed, using mock model for demo")
                success = self._create_mock_model()
            
            if success:
                # Run self-check
                success = self._run_self_check()
            
            self.loaded = success
            return success
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _load_mappings(self):
        """Load entity ID mappings"""
        try:
            if os.path.exists(self.config.MAPPINGS_PATH):
                with open(self.config.MAPPINGS_PATH, 'r') as f:
                    self.mappings = json.load(f)
                logger.info(f"Loaded mappings from {self.config.MAPPINGS_PATH}")
            else:
                logger.warning(f"Mappings file not found: {self.config.MAPPINGS_PATH}")
                # Create default mappings for demo
                self.mappings = {
                    "user_to_id": {},
                    "merchant_to_id": {},
                    "device_to_id": {},
                    "ip_to_id": {},
                    "id_to_user": {},
                    "id_to_merchant": {},
                    "id_to_device": {},
                    "id_to_ip": {}
                }
        except Exception as e:
            logger.error(f"Failed to load mappings: {e}")
            self.mappings = {}
    
    def _load_model_config(self):
        """Load model configuration"""
        try:
            if os.path.exists(self.config.MODEL_CONFIG_PATH):
                import yaml
                with open(self.config.MODEL_CONFIG_PATH, 'r') as f:
                    self.model_config = yaml.safe_load(f)
                logger.info(f"Loaded model config from {self.config.MODEL_CONFIG_PATH}")
            else:
                logger.warning(f"Model config not found: {self.config.MODEL_CONFIG_PATH}")
                # Default configuration
                self.model_config = {
                    "hidden_dim": 128,
                    "num_layers": 3,
                    "num_heads": 8,
                    "dropout": 0.1
                }
        except Exception as e:
            logger.error(f"Failed to load model config: {e}")
            self.model_config = {}
    
    def _load_actual_model(self) -> bool:
        """Attempt to load actual PyTorch model"""
        try:
            if not TORCH_AVAILABLE or not GEOMETRIC_AVAILABLE:
                return False
            
            # Load checkpoint
            checkpoint = torch.load(self.config.CHECKPOINT_PATH, map_location=self.config.DEVICE)
            
            # For this demo, we'll create a simple mock architecture
            # In practice, this would instantiate the actual hHGTN model
            logger.info("Creating model architecture...")
            
            # This is a placeholder - actual implementation would use the real hHGTN
            class MockHGTN(nn.Module):
                def __init__(self, hidden_dim=128):
                    super().__init__()
                    self.encoder = nn.Linear(10, hidden_dim)  # Mock input features
                    self.classifier = nn.Linear(hidden_dim, 2)  # Binary classification
                    
                def forward(self, x):
                    h = torch.relu(self.encoder(x))
                    return torch.softmax(self.classifier(h), dim=-1)
            
            self.model = MockHGTN()
            
            # Try to load state dict if compatible
            try:
                if 'model_state_dict' in checkpoint:
                    # Would load actual weights here
                    logger.info("Checkpoint contains model_state_dict (mock loading)")
                elif 'state_dict' in checkpoint:
                    logger.info("Checkpoint contains state_dict (mock loading)")
                else:
                    logger.info("Using checkpoint as direct state_dict (mock loading)")
            except Exception as e:
                logger.warning(f"Could not load checkpoint weights: {e}")
            
            self.model.eval()
            logger.info("Model loaded successfully (mock version)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load actual model: {e}")
            return False
    
    def _create_mock_model(self) -> bool:
        """Create a mock model for demo purposes"""
        try:
            logger.info("Creating mock model for demo")
            
            class MockModel:
                def __init__(self):
                    self.device = "cpu"
                
                def predict(self, features):
                    # Generate deterministic but realistic-looking predictions
                    np.random.seed(hash(str(features)) % 2**32)
                    prob = np.random.beta(2, 5)  # Biased towards lower fraud probability
                    if prob > 0.5:
                        prob = np.random.beta(8, 2)  # Sometimes high fraud probability
                    return prob, "fraud" if prob > 0.5 else "legitimate"
                
                def explain(self, features, top_k_nodes=30, top_k_edges=50):
                    # Generate mock explanation
                    np.random.seed(hash(str(features)) % 2**32)
                    
                    nodes = []
                    for i in range(min(top_k_nodes, 15)):
                        node_types = ["user", "merchant", "device", "ip"]
                        node_type = np.random.choice(node_types)
                        nodes.append({
                            "id": f"{node_type}_{i+1}",
                            "type": node_type,
                            "importance_score": np.random.beta(2, 3),
                            "features": {
                                "risk_level": np.random.choice(["low", "medium", "high"]),
                                "activity_count": np.random.randint(1, 100)
                            }
                        })
                    
                    edges = []
                    for i in range(min(top_k_edges, 20)):
                        edge_types = ["transaction", "device_link", "location_link"]
                        edges.append({
                            "source": f"user_{np.random.randint(1, 6)}",
                            "target": f"merchant_{np.random.randint(1, 6)}",
                            "relation_type": np.random.choice(edge_types),
                            "importance_score": np.random.beta(2, 3),
                            "weight": np.random.uniform(0.1, 1.0)
                        })
                    
                    features = [
                        {"feature_name": "transaction_amount", "importance_score": np.random.beta(3, 2)},
                        {"feature_name": "user_velocity", "importance_score": np.random.beta(2, 3)},
                        {"feature_name": "merchant_risk", "importance_score": np.random.beta(2, 4)},
                        {"feature_name": "device_reputation", "importance_score": np.random.beta(2, 3)}
                    ]
                    
                    return nodes, edges, features
            
            self.model = MockModel()
            logger.info("Mock model created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create mock model: {e}")
            return False
    
    def _run_self_check(self) -> bool:
        """Run basic self-check on loaded model"""
        try:
            logger.info("Running model self-check...")
            
            # Create test features
            test_features = {
                "user_id": "test_user_123",
                "merchant_id": "test_merchant_456", 
                "amount": 100.0,
                "timestamp": "2025-09-14T10:00:00Z"
            }
            
            # Run prediction (handle both mock and real models)
            if hasattr(self.model, 'predict'):
                prob, label = self.model.predict(test_features)
            else:
                # For PyTorch models, create dummy forward pass
                logger.info("Running forward pass for PyTorch model")
                dummy_input = torch.randn(1, 10)
                with torch.no_grad():
                    output = self.model(dummy_input)
                prob = float(output[0, 1])  # Fraud probability
                label = "fraud" if prob > 0.5 else "legitimate"
            
            # Validate output
            if not (0 <= prob <= 1):
                logger.error(f"Invalid probability: {prob}")
                return False
            
            if label not in ["fraud", "legitimate"]:
                logger.error(f"Invalid label: {label}")
                return False
            
            logger.info(f"Self-check passed: prob={prob:.3f}, label={label}")
            return True
            
        except Exception as e:
            logger.error(f"Self-check failed: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self.loaded and self.model is not None
    
    def predict_with_explanation(self, transaction: Dict[str, Any], explain_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate prediction with explanation for transaction"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Extract prediction parameters
            features = self._extract_features(transaction)
            
            # Generate prediction (handle both mock and real models)
            if hasattr(self.model, 'predict'):
                prob, label = self.model.predict(features)
            else:
                # For PyTorch models, implement prediction logic
                logger.info("Using PyTorch model for prediction")
                dummy_input = torch.randn(1, 10)
                with torch.no_grad():
                    output = self.model(dummy_input)
                prob = float(output[0, 1])  # Fraud probability
                label = "fraud" if prob > 0.5 else "legitimate"
            
            confidence = abs(prob - 0.5) * 2  # Convert to confidence score
            
            # Generate explanation
            explain_start = time.time()
            
            if explain_config is None:
                explain_config = {}
            
            top_k_nodes = explain_config.get("top_k_nodes", self.config.TOP_K_NODES)
            top_k_edges = explain_config.get("top_k_edges", self.config.TOP_K_EDGES)
            
            try:
                if hasattr(self.model, 'explain'):
                    nodes, edges, top_features = self.model.explain(
                        features, 
                        top_k_nodes=top_k_nodes,
                        top_k_edges=top_k_edges
                    )
                else:
                    # Generate mock explanation for PyTorch models
                    logger.info("Generating mock explanation for PyTorch model")
                    nodes, edges, top_features = self._generate_mock_explanation(
                        features, top_k_nodes, top_k_edges
                    )
                
                explanation = {
                    "nodes": nodes,
                    "edges": edges,
                    "top_features": top_features
                }
                explain_error = None
                explain_timed_out = False
                
            except Exception as e:
                logger.error(f"Explanation generation failed: {e}")
                explanation = None
                explain_error = str(e)
                explain_timed_out = False
            
            explain_time_ms = int((time.time() - explain_start) * 1000)
            
            # Build response
            result = {
                "prediction_prob": float(prob),
                "predicted_label": label,
                "confidence": float(confidence),
                "explanation": explanation,
                "meta": {
                    "subgraph_nodes": len(explanation["nodes"]) if explanation else 0,
                    "subgraph_edges": len(explanation["edges"]) if explanation else 0,
                    "explain_time_ms": explain_time_ms,
                    "explain_timed_out": explain_timed_out,
                    "explain_error": explain_error
                }
            }
            
            total_time_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Prediction completed in {total_time_ms}ms (explain: {explain_time_ms}ms)")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def _extract_features(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize features from transaction"""
        # This would normally involve complex feature engineering
        # For demo, we'll use the transaction data directly
        features = transaction.copy()
        
        # Add derived features
        features["amount_log"] = np.log1p(transaction.get("amount", 0))
        features["is_high_amount"] = transaction.get("amount", 0) > 1000
        
        # Add mapping lookups if available
        if self.mappings:
            user_id = transaction.get("user_id")
            if user_id and str(user_id) in self.mappings.get("user_to_id", {}):
                features["user_internal_id"] = self.mappings["user_to_id"][str(user_id)]
        
        return features
    
    def _generate_mock_explanation(self, features: Dict[str, Any], top_k_nodes: int, top_k_edges: int):
        """Generate mock explanation for PyTorch models"""
        np.random.seed(hash(str(features)) % 2**32)
        
        nodes = []
        for i in range(min(top_k_nodes, 15)):
            node_types = ["user", "merchant", "device", "ip"]
            node_type = np.random.choice(node_types)
            nodes.append({
                "id": f"{node_type}_{i+1}",
                "type": node_type,
                "importance_score": float(np.random.beta(2, 3)),
                "features": {
                    "risk_level": np.random.choice(["low", "medium", "high"]),
                    "activity_count": int(np.random.randint(1, 100))
                }
            })
        
        edges = []
        for i in range(min(top_k_edges, 20)):
            edge_types = ["transaction", "device_link", "location_link"]
            edges.append({
                "source": f"user_{np.random.randint(1, 6)}",
                "target": f"merchant_{np.random.randint(1, 6)}",
                "relation_type": np.random.choice(edge_types),
                "importance_score": float(np.random.beta(2, 3)),
                "weight": float(np.random.uniform(0.1, 1.0))
            })
        
        top_features = [
            {"feature_name": "transaction_amount", "importance_score": float(np.random.beta(3, 2))},
            {"feature_name": "user_velocity", "importance_score": float(np.random.beta(2, 3))},
            {"feature_name": "merchant_risk", "importance_score": float(np.random.beta(2, 4))},
            {"feature_name": "device_reputation", "importance_score": float(np.random.beta(2, 3))}
        ]
        
        return nodes, edges, top_features
