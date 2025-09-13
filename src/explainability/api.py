"""
HTTP REST API for hHGTN Explainability.

Provides web-based endpoints for generating explanations.
This module complements the CLI interface with REST API endpoints.

Endpoints:
- POST /explain - Explain a single node
- POST /explain/batch - Explain multiple nodes
- POST /explain/auto - Auto-detect and explain suspicious nodes
- GET /health - Health check
- GET /config - Get current configuration

Author: GitHub Copilot (Stage 10 Implementation)
"""

try:
    from flask import Flask, request, jsonify, send_file
    from flask_cors import CORS
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    Flask = None

import os
import sys
import json
import tempfile
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import traceback

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.explainability.integration import (
    explain_instance, ExplainabilityPipeline, ExplainabilityConfig
)


class ExplainabilityAPI:
    """REST API for explainability services."""
    
    def __init__(self, model=None, data=None, config: ExplainabilityConfig = None, device: str = 'cpu'):
        """
        Initialize API with model and data.
        
        Args:
            model: Trained hHGTN model
            data: Graph data
            config: Explainability configuration
            device: Device for computation
        """
        if not HAS_FLASK:
            raise ImportError("Flask is required for API functionality. Install with: pip install flask flask-cors")
        
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for web clients
        
        self.model = model
        self.data = data
        self.config = config or ExplainabilityConfig()
        self.device = device
        self.pipeline = None
        
        if self.model is not None and self.data is not None:
            self.pipeline = ExplainabilityPipeline(self.model, self.config, self.device)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register API routes."""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            status = {
                'status': 'healthy',
                'model_loaded': self.model is not None,
                'data_loaded': self.data is not None,
                'pipeline_ready': self.pipeline is not None
            }
            return jsonify(status), 200
        
        @self.app.route('/config', methods=['GET'])
        def get_config():
            """Get current configuration."""
            config_dict = {
                'explainer_type': self.config.explainer_type,
                'k_hops': self.config.k_hops,
                'max_nodes': self.config.max_nodes,
                'edge_mask_threshold': self.config.edge_mask_threshold,
                'feature_threshold': self.config.feature_threshold,
                'top_k_features': self.config.top_k_features,
                'visualization': self.config.visualization,
                'save_reports': self.config.save_reports,
                'seed': self.config.seed
            }
            return jsonify(config_dict), 200
        
        @self.app.route('/config', methods=['POST'])
        def update_config():
            """Update configuration."""
            try:
                new_config = request.json
                
                # Update configuration
                for key, value in new_config.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                
                # Recreate pipeline with new config
                if self.model is not None and self.data is not None:
                    self.pipeline = ExplainabilityPipeline(self.model, self.config, self.device)
                
                return jsonify({'status': 'Configuration updated'}), 200
                
            except Exception as e:
                self.logger.error(f"Failed to update config: {e}")
                return jsonify({'error': str(e)}), 400
        
        @self.app.route('/explain', methods=['POST'])
        def explain_node():
            """Explain a single node."""
            try:
                if self.pipeline is None:
                    return jsonify({'error': 'Model and data not loaded'}), 400
                
                data = request.json
                node_id = data.get('node_id')
                
                if node_id is None:
                    return jsonify({'error': 'node_id is required'}), 400
                
                # Override config if provided
                temp_config = self.config
                if 'config' in data:
                    temp_config = ExplainabilityConfig(**data['config'])
                
                result = explain_instance(
                    model=self.model,
                    data=self.data,
                    node_id=node_id,
                    config=temp_config,
                    device=self.device
                )
                
                # Convert numpy arrays to lists for JSON serialization
                result = self._serialize_result(result)
                
                return jsonify(result), 200
                
            except Exception as e:
                self.logger.error(f"Failed to explain node: {e}")
                return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500
        
        @self.app.route('/explain/batch', methods=['POST'])
        def explain_batch():
            """Explain multiple nodes."""
            try:
                if self.pipeline is None:
                    return jsonify({'error': 'Model and data not loaded'}), 400
                
                data = request.json
                node_ids = data.get('node_ids', [])
                
                if not node_ids:
                    return jsonify({'error': 'node_ids list is required'}), 400
                
                # Override config if provided
                if 'config' in data:
                    temp_config = ExplainabilityConfig(**data['config'])
                    temp_pipeline = ExplainabilityPipeline(self.model, temp_config, self.device)
                else:
                    temp_pipeline = self.pipeline
                
                results = temp_pipeline.explain_nodes(self.data, node_ids)
                
                # Serialize results
                results = [self._serialize_result(result) for result in results]
                
                return jsonify({'results': results}), 200
                
            except Exception as e:
                self.logger.error(f"Failed to explain batch: {e}")
                return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500
        
        @self.app.route('/explain/auto', methods=['POST'])
        def explain_auto():
            """Auto-detect and explain suspicious nodes."""
            try:
                if self.pipeline is None:
                    return jsonify({'error': 'Model and data not loaded'}), 400
                
                data = request.json or {}
                threshold = data.get('threshold', 0.5)
                max_nodes = data.get('max_nodes', 100)
                
                # Override config if provided
                if 'config' in data:
                    temp_config = ExplainabilityConfig(**data['config'])
                    temp_pipeline = ExplainabilityPipeline(self.model, temp_config, self.device)
                else:
                    temp_pipeline = self.pipeline
                
                results = temp_pipeline.explain_suspicious_nodes(
                    self.data, threshold=threshold, max_nodes=max_nodes
                )
                
                # Serialize results
                results = [self._serialize_result(result) for result in results]
                
                return jsonify({'results': results, 'threshold': threshold, 'max_nodes': max_nodes}), 200
                
            except Exception as e:
                self.logger.error(f"Failed to auto-explain: {e}")
                return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500
        
        @self.app.route('/report/<int:node_id>', methods=['GET'])
        def get_report(node_id):
            """Get HTML report for a node."""
            try:
                report_path = self.config.output_dir / f"node_{node_id}_report.html"
                
                if not report_path.exists():
                    return jsonify({'error': f'Report for node {node_id} not found'}), 404
                
                return send_file(str(report_path), mimetype='text/html')
                
            except Exception as e:
                self.logger.error(f"Failed to get report: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/visualization/<int:node_id>/<viz_type>', methods=['GET'])
        def get_visualization(node_id, viz_type):
            """Get visualization file for a node."""
            try:
                # Map visualization types to file extensions
                file_map = {
                    'static': '.png',
                    'interactive': '.html',
                    'features': '_features.png'
                }
                
                if viz_type not in file_map:
                    return jsonify({'error': f'Invalid visualization type: {viz_type}'}), 400
                
                if viz_type == 'features':
                    viz_path = self.config.output_dir / f"node_{node_id}_features.png"
                elif viz_type == 'static':
                    viz_path = self.config.output_dir / f"node_{node_id}_visualization.png"
                elif viz_type == 'interactive':
                    viz_path = self.config.output_dir / f"node_{node_id}_visualization.html"
                
                if not viz_path.exists():
                    return jsonify({'error': f'Visualization for node {node_id} not found'}), 404
                
                # Determine MIME type
                if viz_type == 'interactive':
                    mimetype = 'text/html'
                else:
                    mimetype = 'image/png'
                
                return send_file(str(viz_path), mimetype=mimetype)
                
            except Exception as e:
                self.logger.error(f"Failed to get visualization: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Endpoint not found'}), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({'error': 'Internal server error'}), 500
    
    def _serialize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize result for JSON response."""
        import numpy as np
        
        serialized = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_result(value)
            elif isinstance(value, (np.integer, np.floating)):
                serialized[key] = value.item()
            else:
                serialized[key] = value
        
        return serialized
    
    def load_model_and_data(self, model_path: str, data_path: str):
        """Load model and data from paths."""
        try:
            import torch
            
            self.logger.info(f"Loading model from {model_path}")
            self.model = torch.load(model_path, map_location=self.device)
            
            self.logger.info(f"Loading data from {data_path}")
            self.data = torch.load(data_path, map_location=self.device)
            
            # Create pipeline
            self.pipeline = ExplainabilityPipeline(self.model, self.config, self.device)
            
            self.logger.info("Model and data loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model and data: {e}")
            raise
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Run the API server."""
        self.logger.info(f"Starting API server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def create_api_from_cli():
    """Create API instance from command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hHGTN Explainability API')
    parser.add_argument('--model_path', type=str, help='Path to trained model')
    parser.add_argument('--data_path', type=str, help='Path to graph data')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--device', type=str, default='cpu', help='Device for computation')
    parser.add_argument('--output_dir', type=str, default='api_explanations', help='Output directory')
    
    args = parser.parse_args()
    
    # Create configuration
    config = ExplainabilityConfig(output_dir=args.output_dir)
    
    # Create API
    api = ExplainabilityAPI(config=config, device=args.device)
    
    # Load model and data if provided
    if args.model_path and args.data_path:
        api.load_model_and_data(args.model_path, args.data_path)
    
    return api, args


def main_api():
    """Main API entry point."""
    if not HAS_FLASK:
        print("Flask is required for API functionality. Install with: pip install flask flask-cors")
        sys.exit(1)
    
    api, args = create_api_from_cli()
    api.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main_api()
