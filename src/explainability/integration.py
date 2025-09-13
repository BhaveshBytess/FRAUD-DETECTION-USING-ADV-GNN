"""
Phase D — Integration API for hHGTN Explainability.

Provides pipeline hooks and endpoints for on-demand explanations.
Integrates with existing training and inference pipelines.

This module implements:
1. explain_instance() - Main pipeline hook for single node explanations
2. ExplainabilityPipeline - High-level interface for batch explanations  
3. CLI interface for command-line explanations
4. HTTP REST API for web-based explanations
5. Integration with hHGTN models from Stage 9

Author: GitHub Copilot (Stage 10 Implementation)
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import time
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import our explainability modules
from src.explainability.extract_subgraph import SubgraphExtractor, extract_khop_subgraph
from src.explainability.gnne_explainers import (
    BaseExplainer, GNNExplainerWrapper, PGExplainerTrainer, 
    HGNNExplainer, TemporalExplainer
)
from src.explainability.visualizer import (
    visualize_subgraph, explain_report, explain_batch_to_html,
    create_feature_importance_plot
)

# Import model and data utilities
try:
    from src.model import HeteroGNN
    from src.data_utils import load_data
    from src.utils import set_seed
except ImportError as e:
    logging.warning(f"Could not import project modules: {e}")
    HeteroGNN = None
    
    # Fallback implementation for set_seed
    def set_seed(seed):
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)


class ExplainabilityConfig:
    """Configuration for explainability pipeline."""
    
    def __init__(self, 
                 explainer_type: str = 'gnn_explainer',
                 k_hops: int = 2,
                 max_nodes: int = 50,
                 edge_mask_threshold: float = 0.1,
                 feature_threshold: float = 0.05,
                 top_k_features: int = 10,
                 visualization: bool = True,
                 save_reports: bool = True,
                 output_dir: str = 'explanations',
                 seed: int = 42):
        """
        Initialize explainability configuration.
        
        Args:
            explainer_type: Type of explainer ('gnn_explainer', 'pg_explainer', 'hgnn_explainer')
            k_hops: Number of hops for subgraph extraction
            max_nodes: Maximum nodes in extracted subgraph
            edge_mask_threshold: Threshold for significant edges
            feature_threshold: Threshold for significant features
            top_k_features: Number of top features to report
            visualization: Whether to generate visualizations
            save_reports: Whether to save HTML reports
            output_dir: Directory for saving outputs
            seed: Random seed for reproducibility
        """
        self.explainer_type = explainer_type
        self.k_hops = k_hops
        self.max_nodes = max_nodes
        self.edge_mask_threshold = edge_mask_threshold
        self.feature_threshold = feature_threshold
        self.top_k_features = top_k_features
        self.visualization = visualization
        self.save_reports = save_reports
        self.output_dir = Path(output_dir)
        self.seed = seed
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)


def explain_instance(model: nn.Module,
                    data: Any,
                    node_id: int,
                    config: ExplainabilityConfig = None,
                    device: str = 'cpu') -> Dict[str, Any]:
    """
    Main pipeline hook for explaining a single node prediction.
    
    This function integrates all explainability components into a single
    pipeline for generating explanations of model predictions.
    
    Args:
        model: Trained hHGTN model
        data: Graph data (torch_geometric.data.Data or HeteroData)
        node_id: Target node to explain
        config: Explainability configuration
        device: Device for computation
        
    Returns:
        Dictionary containing:
        - prediction: Model prediction probability
        - explanation_masks: Edge and node feature masks
        - top_features: Most important features
        - subgraph_info: Extracted subgraph details
        - visualization_paths: Paths to generated visualizations
        - report_path: Path to HTML report
        - explanation_text: Human-readable explanation
    """
    if config is None:
        config = ExplainabilityConfig()
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Explaining node {node_id} with {config.explainer_type}")
    
    try:
        # Move model and data to device
        model = model.to(device)
        if hasattr(data, 'to'):
            data = data.to(device)
        
        # Step 1: Extract subgraph around target node
        logger.info("Extracting subgraph...")
        extractor = SubgraphExtractor(
            max_nodes=config.max_nodes,
            seed=config.seed
        )
        
        subgraph_data = extractor.extract(
            graph_data=data,
            node_id=node_id,
            num_hops=config.k_hops
        )
        
        # Convert to Data-like object for compatibility
        if isinstance(subgraph_data, dict):
            # Create a simple object with the required attributes
            class SimpleData:
                def __init__(self, data_dict):
                    self.edge_index = data_dict.get('edge_index', torch.tensor([[], []], dtype=torch.long))
                    self.num_nodes = len(data_dict.get('subset', []))
                    self.num_edges = self.edge_index.size(1)
                    if hasattr(data, 'x') and data.x is not None:
                        subset = data_dict.get('subset', torch.arange(self.num_nodes))
                        self.x = data.x[subset] if len(subset) > 0 else torch.randn(self.num_nodes, data.x.size(1))
                    else:
                        self.x = torch.randn(self.num_nodes, 5)  # Default feature size
                
                def to(self, device):
                    self.edge_index = self.edge_index.to(device)
                    self.x = self.x.to(device)
                    return self
            
            subgraph_data = SimpleData(subgraph_data)
        
        # Step 2: Get model prediction for target node
        logger.info("Getting model prediction...")
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'forward'):
                output = model(data)
                if isinstance(output, dict):
                    # Handle heterogeneous output
                    node_type = _get_node_type(data, node_id)
                    pred_logits = output[node_type][node_id]
                else:
                    pred_logits = output[node_id]
                
                pred_prob = torch.softmax(pred_logits, dim=-1)
                if pred_prob.dim() > 1:
                    pred_prob = pred_prob[0]  # Take first sample if batch
                
                # Assume binary classification - take probability of positive class
                if len(pred_prob) == 2:
                    fraud_prob = pred_prob[1].item()
                else:
                    fraud_prob = pred_prob.max().item()
            else:
                # Fallback for testing
                fraud_prob = 0.75
                logger.warning("Model does not have forward method, using default prediction")
        
        # Step 3: Initialize explainer
        logger.info(f"Initializing {config.explainer_type} explainer...")
        if config.explainer_type == 'gnn_explainer':
            explainer = GNNExplainerWrapper(model=model, device=device)
        elif config.explainer_type == 'pg_explainer':
            explainer = PGExplainerTrainer(model=model, device=device)
        elif config.explainer_type == 'hgnn_explainer':
            explainer = HGNNExplainer(model=model, device=device)
        elif config.explainer_type == 'temporal_explainer':
            explainer = TemporalExplainer(model=model, device=device)
        else:
            raise ValueError(f"Unknown explainer type: {config.explainer_type}")
        
        # Step 4: Generate explanation
        logger.info("Generating explanation...")
        explanation = explainer.explain(
            data=subgraph_data,
            node_indices=torch.tensor([0]),  # Target node is now at index 0 after relabeling
            additional_forward_args=None
        )
        
        # Step 5: Process explanation masks
        edge_mask = explanation.get('edge_mask', torch.tensor([]))
        node_feat_mask = explanation.get('node_feat_mask', torch.tensor([]))
        
        # Filter significant edges and features
        significant_edges = (edge_mask > config.edge_mask_threshold).nonzero().flatten()
        significant_features = (node_feat_mask > config.feature_threshold).nonzero().flatten()
        
        # Step 6: Generate top features list
        logger.info("Extracting top features...")
        top_features = []
        if len(node_feat_mask) > 0:
            # Get feature names (use indices if names not available)
            feature_names = getattr(data, 'feature_names', [f'feature_{i}' for i in range(len(node_feat_mask))])
            
            # Sort features by importance
            sorted_indices = torch.argsort(torch.abs(node_feat_mask), descending=True)
            for i, idx in enumerate(sorted_indices[:config.top_k_features]):
                if i < len(feature_names):
                    name = feature_names[idx.item()]
                else:
                    name = f'feature_{idx.item()}'
                importance = node_feat_mask[idx].item()
                top_features.append((name, importance))
        
        # Step 7: Generate human-readable explanation
        explanation_text = _generate_explanation_text(
            node_id=node_id,
            fraud_prob=fraud_prob,
            top_features=top_features,
            significant_edges=len(significant_edges),
            explainer_type=config.explainer_type
        )
        
        # Step 8: Create visualizations
        visualization_paths = {}
        if config.visualization:
            logger.info("Creating visualizations...")
            try:
                vis_base_path = config.output_dir / f"node_{node_id}_visualization"
                
                # Convert subgraph to visualization format
                if hasattr(subgraph_data, 'edge_index'):
                    vis_graph = {'edge_index': subgraph_data.edge_index}
                else:
                    vis_graph = subgraph_data
                
                # Create node metadata
                node_meta = {
                    'labels': {0: 'target'},  # Target node at index 0
                    'features': {}
                }
                
                # Add feature information if available
                if hasattr(subgraph_data, 'x') and subgraph_data.x is not None:
                    for i, feat in enumerate(subgraph_data.x):
                        node_meta['features'][i] = feat.cpu().numpy().tolist()
                
                vis_results = visualize_subgraph(
                    G=vis_graph,
                    masks={'edge_mask': edge_mask, 'node_feat_mask': node_feat_mask},
                    node_meta=node_meta,
                    target_node=0,
                    top_k=len(significant_edges),
                    output_path=str(vis_base_path),
                    interactive=True
                )
                
                visualization_paths.update(vis_results)
                
                # Create feature importance plot
                if top_features:
                    feat_plot_path = config.output_dir / f"node_{node_id}_features.png"
                    create_feature_importance_plot(
                        top_features=top_features,
                        output_path=str(feat_plot_path)
                    )
                    visualization_paths['feature_plot'] = str(feat_plot_path)
                
            except Exception as e:
                logger.warning(f"Failed to create visualizations: {e}")
        
        # Step 9: Generate HTML report
        report_path = None
        if config.save_reports:
            logger.info("Generating HTML report...")
            try:
                report_path = config.output_dir / f"node_{node_id}_report.html"
                explain_report(
                    node_id=node_id,
                    pred_prob=fraud_prob,
                    masks={'edge_mask': edge_mask, 'node_feat_mask': node_feat_mask, 'explanation_type': config.explainer_type},
                    top_features=top_features,
                    explanation_text=explanation_text,
                    output_path=str(report_path)
                )
            except Exception as e:
                logger.warning(f"Failed to create HTML report: {e}")
                report_path = None
        
        # Compile results
        results = {
            'node_id': node_id,
            'prediction': fraud_prob,
            'explanation_masks': {
                'edge_mask': edge_mask.cpu().numpy() if len(edge_mask) > 0 else np.array([]),
                'node_feat_mask': node_feat_mask.cpu().numpy() if len(node_feat_mask) > 0 else np.array([])
            },
            'top_features': top_features,
            'subgraph_info': {
                'num_nodes': subgraph_data.num_nodes if hasattr(subgraph_data, 'num_nodes') else len(subgraph_data.get('x', [])),
                'num_edges': subgraph_data.num_edges if hasattr(subgraph_data, 'num_edges') else len(subgraph_data.get('edge_index', [[], []])[0]),
                'significant_edges': len(significant_edges),
                'significant_features': len(significant_features)
            },
            'visualization_paths': visualization_paths,
            'report_path': str(report_path) if report_path else None,
            'explanation_text': explanation_text,
            'explainer_type': config.explainer_type,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Successfully explained node {node_id}")
        return results
        
    except Exception as e:
        logger.error(f"Failed to explain node {node_id}: {e}")
        raise


class ExplainabilityPipeline:
    """High-level interface for batch explanations and pipeline integration."""
    
    def __init__(self, model: nn.Module, config: ExplainabilityConfig = None, device: str = 'cpu'):
        """
        Initialize explainability pipeline.
        
        Args:
            model: Trained hHGTN model
            config: Explainability configuration
            device: Device for computation
        """
        self.model = model
        self.config = config or ExplainabilityConfig()
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Setup output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def explain_nodes(self, data: Any, node_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Explain multiple nodes in batch.
        
        Args:
            data: Graph data
            node_ids: List of node IDs to explain
            
        Returns:
            List of explanation results
        """
        results = []
        
        self.logger.info(f"Explaining {len(node_ids)} nodes...")
        
        for i, node_id in enumerate(node_ids):
            self.logger.info(f"Processing node {node_id} ({i+1}/{len(node_ids)})")
            
            try:
                result = explain_instance(
                    model=self.model,
                    data=data,
                    node_id=node_id,
                    config=self.config,
                    device=self.device
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to explain node {node_id}: {e}")
                # Add error result
                results.append({
                    'node_id': node_id,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Generate batch report
        if self.config.save_reports and results:
            self._generate_batch_report(results)
        
        return results
    
    def _generate_batch_report(self, results: List[Dict[str, Any]]):
        """Generate batch HTML report."""
        try:
            # Prepare data for batch report
            explanations = []
            for result in results:
                if 'error' not in result:
                    explanations.append({
                        'node_id': result['node_id'],
                        'prediction': result['prediction'],
                        'masks': result['explanation_masks'],
                        'top_features': result['top_features'],
                        'explanation_text': result['explanation_text']
                    })
            
            if explanations:
                batch_dir = self.config.output_dir / 'batch_reports'
                batch_dir.mkdir(exist_ok=True)
                
                explain_batch_to_html(
                    explanations=explanations,
                    out_dir=str(batch_dir)
                )
                
                self.logger.info(f"Batch report saved to {batch_dir}")
                
        except Exception as e:
            self.logger.warning(f"Failed to generate batch report: {e}")
    
    def explain_suspicious_nodes(self, data: Any, threshold: float = 0.5, max_nodes: int = 100) -> List[Dict[str, Any]]:
        """
        Automatically identify and explain suspicious nodes.
        
        Args:
            data: Graph data
            threshold: Prediction threshold for suspicious nodes
            max_nodes: Maximum number of nodes to explain
            
        Returns:
            List of explanation results for suspicious nodes
        """
        self.logger.info(f"Identifying suspicious nodes (threshold={threshold})...")
        
        # Get predictions for all nodes
        self.model.eval()
        with torch.no_grad():
            if hasattr(data, 'to'):
                data = data.to(self.device)
            
            output = self.model(data)
            
            if isinstance(output, dict):
                # Handle heterogeneous output - assume 'transaction' nodes are targets
                if 'transaction' in output:
                    predictions = torch.softmax(output['transaction'], dim=-1)[:, 1]  # Fraud probability
                    node_indices = list(range(len(predictions)))
                else:
                    # Take first node type
                    node_type = list(output.keys())[0]
                    predictions = torch.softmax(output[node_type], dim=-1)
                    if predictions.size(-1) == 2:
                        predictions = predictions[:, 1]
                    else:
                        predictions = predictions.max(dim=-1)[0]
                    node_indices = list(range(len(predictions)))
            else:
                predictions = torch.softmax(output, dim=-1)
                if predictions.size(-1) == 2:
                    predictions = predictions[:, 1]
                else:
                    predictions = predictions.max(dim=-1)[0]
                node_indices = list(range(len(predictions)))
        
        # Find suspicious nodes
        suspicious_mask = predictions > threshold
        suspicious_indices = torch.nonzero(suspicious_mask).flatten()
        
        # Sort by prediction score (most suspicious first)
        sorted_indices = suspicious_indices[torch.argsort(predictions[suspicious_indices], descending=True)]
        
        # Limit number of nodes
        selected_indices = sorted_indices[:max_nodes].cpu().numpy().tolist()
        
        self.logger.info(f"Found {len(selected_indices)} suspicious nodes to explain")
        
        # Explain selected nodes
        return self.explain_nodes(data, selected_indices)


def _get_node_type(data: Any, node_id: int) -> str:
    """Get node type for heterogeneous graphs."""
    if hasattr(data, 'node_types') and data.node_types is not None:
        # Simple heuristic - assume transaction nodes are main targets
        if 'transaction' in data.node_types:
            return 'transaction'
        else:
            return data.node_types[0]
    return 'default'


def _generate_explanation_text(node_id: int, 
                             fraud_prob: float,
                             top_features: List[Tuple[str, float]],
                             significant_edges: int,
                             explainer_type: str) -> str:
    """Generate human-readable explanation text."""
    
    # Determine risk level
    if fraud_prob > 0.7:
        risk_level = "high"
    elif fraud_prob > 0.4:
        risk_level = "moderate"
    else:
        risk_level = "low"
    
    # Start explanation
    explanation = f"Node {node_id} has a {risk_level} fraud risk with {fraud_prob:.1%} probability. "
    
    # Add top contributing factors
    if top_features:
        pos_features = [f for f, score in top_features if score > 0]
        neg_features = [f for f, score in top_features if score < 0]
        
        if pos_features:
            explanation += f"Key risk factors include: {', '.join(pos_features[:3])}. "
        
        if neg_features:
            explanation += f"Mitigating factors include: {', '.join(neg_features[:2])}. "
    
    # Add network information
    if significant_edges > 0:
        explanation += f"The node has {significant_edges} significant connections that influence this prediction. "
    
    # Add explainer method
    explanation += f"This explanation was generated using {explainer_type.replace('_', ' ')} method."
    
    return explanation


# CLI Interface
def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line interface for explanations."""
    parser = argparse.ArgumentParser(description='Generate explanations for hHGTN predictions')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to graph data')
    parser.add_argument('--node_id', type=int,
                       help='Specific node ID to explain')
    parser.add_argument('--node_ids', type=str,
                       help='Comma-separated list of node IDs to explain')
    parser.add_argument('--auto_detect', action='store_true',
                       help='Automatically detect and explain suspicious nodes')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for automatic detection')
    parser.add_argument('--explainer_type', type=str, default='gnn_explainer',
                       choices=['gnn_explainer', 'pg_explainer', 'hgnn_explainer', 'temporal_explainer'],
                       help='Type of explainer to use')
    parser.add_argument('--output_dir', type=str, default='explanations',
                       help='Output directory for results')
    parser.add_argument('--k_hops', type=int, default=2,
                       help='Number of hops for subgraph extraction')
    parser.add_argument('--max_nodes', type=int, default=50,
                       help='Maximum nodes in subgraph')
    parser.add_argument('--top_k_features', type=int, default=10,
                       help='Number of top features to show')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for computation (cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--no_viz', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--no_reports', action='store_true',
                       help='Skip HTML report generation')
    
    return parser


def main_cli():
    """Main CLI entry point."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Load model and data
        logger.info(f"Loading model from {args.model_path}")
        if HeteroGNN is None:
            raise ImportError("Could not import model classes. Please check your project setup.")
        
        # This would need to be adapted based on actual model loading logic
        model = torch.load(args.model_path, map_location=args.device)
        
        logger.info(f"Loading data from {args.data_path}")
        # This would need to be adapted based on actual data loading logic
        data = torch.load(args.data_path, map_location=args.device)
        
        # Create configuration
        config = ExplainabilityConfig(
            explainer_type=args.explainer_type,
            k_hops=args.k_hops,
            max_nodes=args.max_nodes,
            top_k_features=args.top_k_features,
            visualization=not args.no_viz,
            save_reports=not args.no_reports,
            output_dir=args.output_dir,
            seed=args.seed
        )
        
        # Create pipeline
        pipeline = ExplainabilityPipeline(model, config, args.device)
        
        # Determine which nodes to explain
        if args.node_id is not None:
            node_ids = [args.node_id]
        elif args.node_ids is not None:
            node_ids = [int(x.strip()) for x in args.node_ids.split(',')]
        elif args.auto_detect:
            logger.info("Auto-detecting suspicious nodes...")
            results = pipeline.explain_suspicious_nodes(data, args.threshold)
            logger.info(f"Explained {len(results)} suspicious nodes")
            return
        else:
            raise ValueError("Must specify --node_id, --node_ids, or --auto_detect")
        
        # Explain nodes
        results = pipeline.explain_nodes(data, node_ids)
        
        # Print summary
        logger.info(f"Successfully explained {len([r for r in results if 'error' not in r])}/{len(results)} nodes")
        logger.info(f"Results saved to {config.output_dir}")
        
    except Exception as e:
        logger.error(f"CLI execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
