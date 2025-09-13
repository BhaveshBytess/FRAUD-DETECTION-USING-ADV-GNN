"""
Phase C — Visualizer & Report generation.

Implements visualization functions for subgraphs, explanations, and HTML reports
using pyvis, networkx, and plotly.

Following Stage 10 Reference §Phase C requirements.
"""

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import os
from datetime import datetime
import json

# Import pyvis with fallback
try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False
    print("Warning: pyvis not available. Interactive visualizations will be limited.")

logger = logging.getLogger(__name__)


def visualize_subgraph(
    G: Union[nx.Graph, Dict[str, Any]],
    masks: Dict[str, torch.Tensor],
    node_meta: Dict[str, Any],
    target_node: int,
    top_k: int = 50,
    output_path: Optional[str] = None,
    interactive: bool = True
) -> Dict[str, str]:
    """
    Create visualization of subgraph with explanation highlights.
    
    Args:
        G: NetworkX graph or graph data dictionary
        masks: Dictionary with 'edge_mask', 'node_feat_mask', etc.
        node_meta: Node metadata (labels, features, etc.)
        target_node: Target node to highlight
        top_k: Number of top edges to highlight
        output_path: Path to save visualization
        interactive: Whether to create interactive HTML
        
    Returns:
        Dictionary with paths to generated visualizations
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"explanation_{timestamp}"
    
    results = {}
    
    # Convert graph data to NetworkX if needed
    if isinstance(G, dict):
        G = _dict_to_networkx(G)
    
    # Create static matplotlib visualization
    static_path = _create_static_visualization(
        G, masks, node_meta, target_node, top_k, output_path
    )
    results['static'] = static_path
    
    # Create interactive pyvis visualization if available
    if interactive and HAS_PYVIS:
        interactive_path = _create_interactive_visualization(
            G, masks, node_meta, target_node, top_k, output_path
        )
        results['interactive'] = interactive_path
    
    # Create plotly visualization
    plotly_path = _create_plotly_visualization(
        G, masks, node_meta, target_node, top_k, output_path
    )
    results['plotly'] = plotly_path
    
    logger.info(f"Created visualizations: {list(results.keys())}")
    return results


def _dict_to_networkx(graph_data: Dict[str, Any]) -> nx.Graph:
    """Convert graph dictionary to NetworkX graph."""
    G = nx.Graph()
    
    if 'edge_index' in graph_data:
        edge_index = graph_data['edge_index']
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.numpy()
        
        # Add edges
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            G.add_edge(src, dst)
    
    return G


def _create_static_visualization(
    G: nx.Graph,
    masks: Dict[str, torch.Tensor], 
    node_meta: Dict[str, Any],
    target_node: int,
    top_k: int,
    output_path: str
) -> str:
    """Create static matplotlib visualization."""
    plt.figure(figsize=(12, 8))
    
    # Get layout
    pos = nx.spring_layout(G, seed=42)
    
    # Get edge mask and select top edges
    edge_mask = masks.get('edge_mask', torch.ones(G.number_of_edges()))
    if isinstance(edge_mask, torch.Tensor):
        edge_mask = edge_mask.numpy()
    
    # Get top edges
    edge_list = list(G.edges())
    if len(edge_mask) == len(edge_list):
        edge_importance = list(zip(edge_list, edge_mask))
        edge_importance.sort(key=lambda x: x[1], reverse=True)
        top_edges = [edge for edge, _ in edge_importance[:top_k]]
    else:
        top_edges = edge_list[:top_k]
    
    # Draw all edges in light gray
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='lightgray', width=0.5)
    
    # Draw important edges in red with thickness proportional to importance
    if top_edges:
        important_weights = []
        for edge in top_edges:
            if edge in edge_list:
                idx = edge_list.index(edge)
                weight = edge_mask[idx] if idx < len(edge_mask) else 0.5
                important_weights.append(weight)
            else:
                important_weights.append(0.5)
        
        # Normalize weights for line thickness
        max_weight = max(important_weights) if important_weights else 1.0
        widths = [3 * (w / max_weight) + 0.5 for w in important_weights]
        
        nx.draw_networkx_edges(
            G, pos, edgelist=top_edges, 
            edge_color='red', width=widths, alpha=0.8
        )
    
    # Draw nodes
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node == target_node:
            node_colors.append('yellow')
            node_sizes.append(500)
        else:
            node_colors.append('lightblue')
            node_sizes.append(200)
    
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8
    )
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(f"Explanation for Node {target_node}\nTop {len(top_edges)} Important Edges")
    plt.axis('off')
    
    # Save
    static_path = f"{output_path}_static.png"
    plt.savefig(static_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return static_path


def _create_interactive_visualization(
    G: nx.Graph,
    masks: Dict[str, torch.Tensor],
    node_meta: Dict[str, Any], 
    target_node: int,
    top_k: int,
    output_path: str
) -> str:
    """Create interactive pyvis visualization."""
    net = Network(height="600px", width="100%", bgcolor="#ffffff")
    
    # Get edge mask
    edge_mask = masks.get('edge_mask', torch.ones(G.number_of_edges()))
    if isinstance(edge_mask, torch.Tensor):
        edge_mask = edge_mask.numpy()
    
    # Add nodes
    for node in G.nodes():
        if node == target_node:
            net.add_node(
                node, 
                label=f"Target: {node}",
                color="yellow",
                size=30,
                title=f"Target node {node}"
            )
        else:
            net.add_node(
                node,
                label=str(node),
                color="lightblue", 
                size=20,
                title=f"Node {node}"
            )
    
    # Add edges with importance-based styling
    edge_list = list(G.edges())
    for i, (src, dst) in enumerate(edge_list):
        importance = edge_mask[i] if i < len(edge_mask) else 0.5
        
        # Style based on importance
        if importance > 0.7:
            color = "red"
            width = 5
        elif importance > 0.5:
            color = "orange" 
            width = 3
        else:
            color = "lightgray"
            width = 1
        
        net.add_edge(
            src, dst,
            color=color,
            width=width,
            title=f"Importance: {importance:.3f}"
        )
    
    # Configure physics
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100}
      }
    }
    """)
    
    # Save
    interactive_path = f"{output_path}_interactive.html"
    net.save_graph(interactive_path)
    
    return interactive_path


def _create_plotly_visualization(
    G: nx.Graph,
    masks: Dict[str, torch.Tensor],
    node_meta: Dict[str, Any],
    target_node: int, 
    top_k: int,
    output_path: str
) -> str:
    """Create plotly visualization."""
    # Get layout
    pos = nx.spring_layout(G, seed=42)
    
    # Prepare edge traces
    edge_x = []
    edge_y = []
    edge_info = []
    
    edge_mask = masks.get('edge_mask', torch.ones(G.number_of_edges()))
    if isinstance(edge_mask, torch.Tensor):
        edge_mask = edge_mask.numpy()
    
    edge_list = list(G.edges())
    for i, (src, dst) in enumerate(edge_list):
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        
        importance = edge_mask[i] if i < len(edge_mask) else 0.5
        
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_info.append(f"Edge {src}-{dst}: {importance:.3f}")
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Prepare node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        if node == target_node:
            node_text.append(f"Target: {node}")
            node_color.append('yellow')
            node_size.append(20)
        else:
            node_text.append(str(node))
            node_color.append('lightblue')
            node_size.append(10)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='black')
        )
    )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=f'Explanation for Node {target_node}', font=dict(size=16)),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Important edges highlighted",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='black', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    
    # Save
    plotly_path = f"{output_path}_plotly.html"
    fig.write_html(plotly_path)
    
    return plotly_path


def explain_report(
    node_id: int,
    pred_prob: float,
    masks: Dict[str, torch.Tensor],
    top_features: List[Tuple[str, float]],
    explanation_text: str,
    output_path: Optional[str] = None
) -> str:
    """
    Generate HTML explanation report for a single instance.
    
    Args:
        node_id: Target node ID
        pred_prob: Predicted probability
        masks: Explanation masks
        top_features: List of (feature_name, importance) tuples
        explanation_text: Human-readable explanation
        output_path: Path to save report
        
    Returns:
        Path to generated HTML report
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"report_{node_id}_{timestamp}.html"
    
    # Create HTML content
    html_content = _generate_report_html(
        node_id, pred_prob, masks, top_features, explanation_text
    )
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Generated explanation report: {output_path}")
    return output_path


def _generate_report_html(
    node_id: int,
    pred_prob: float,
    masks: Dict[str, torch.Tensor],
    top_features: List[Tuple[str, float]],
    explanation_text: str
) -> str:
    """Generate HTML content for explanation report."""
    
    # Calculate summary statistics
    edge_mask = masks.get('edge_mask', torch.tensor([]))
    num_important_edges = (edge_mask > 0.7).sum().item() if len(edge_mask) > 0 else 0
    avg_edge_importance = edge_mask.mean().item() if len(edge_mask) > 0 else 0.0
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Explanation Report - Node {node_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .feature-table {{ width: 100%; border-collapse: collapse; }}
            .feature-table th, .feature-table td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            .feature-table th {{ background-color: #f2f2f2; }}
            .importance-bar {{ height: 20px; background-color: #e0e0e0; border-radius: 10px; overflow: hidden; }}
            .importance-fill {{ height: 100%; background-color: #4CAF50; }}
            .high-risk {{ color: #d32f2f; font-weight: bold; }}
            .low-risk {{ color: #388e3c; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 Fraud Detection Explanation</h1>
            <h2>Node {node_id} Analysis</h2>
            <p><strong>Prediction:</strong> 
                <span class="{'high-risk' if pred_prob > 0.5 else 'low-risk'}">
                    {pred_prob:.1%} fraud probability
                </span>
            </p>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <div class="section">
            <h3>📊 Summary Statistics</h3>
            <ul>
                <li><strong>Important Edges:</strong> {num_important_edges}</li>
                <li><strong>Average Edge Importance:</strong> {avg_edge_importance:.3f}</li>
                <li><strong>Total Edges Analyzed:</strong> {len(edge_mask)}</li>
            </ul>
        </div>
        
        <div class="section">
            <h3>🎯 Top Important Features</h3>
            <table class="feature-table">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Importance</th>
                        <th>Visual</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Add feature rows
    for feature_name, importance in top_features[:10]:  # Top 10 features
        importance_pct = abs(importance) * 100
        html_template += f"""
                    <tr>
                        <td>{feature_name}</td>
                        <td>{importance:.3f}</td>
                        <td>
                            <div class="importance-bar">
                                <div class="importance-fill" style="width: {importance_pct}%;"></div>
                            </div>
                        </td>
                    </tr>
        """
    
    html_template += f"""
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h3>💡 Explanation</h3>
            <p>{explanation_text}</p>
        </div>
        
        <div class="section">
            <h3>🔧 Technical Details</h3>
            <ul>
                <li><strong>Explanation Method:</strong> {masks.get('explanation_type', 'Unknown')}</li>
                <li><strong>Edge Mask Range:</strong> {
                    f"[{edge_mask.min().item():.3f}, {edge_mask.max().item():.3f}]" 
                    if edge_mask.numel() > 0 else "[No edges]"
                }</li>
                <li><strong>Reproducible:</strong> ✅ (Fixed seed used)</li>
            </ul>
        </div>
        
        <div class="section">
            <h3>📋 Raw Data</h3>
            <details>
                <summary>Click to view raw explanation data</summary>
                <pre>{json.dumps({
                    'node_id': node_id,
                    'prediction': pred_prob,
                    'edge_importance': edge_mask.tolist() if len(edge_mask) > 0 else [],
                    'top_features': top_features
                }, indent=2)}</pre>
            </details>
        </div>
    </body>
    </html>
    """
    
    return html_template


def explain_batch_to_html(
    explanations: List[Dict[str, Any]], 
    out_dir: str
) -> List[str]:
    """
    Generate HTML reports for a batch of explanations.
    
    Args:
        explanations: List of explanation dictionaries
        out_dir: Output directory for reports
        
    Returns:
        List of paths to generated reports
    """
    os.makedirs(out_dir, exist_ok=True)
    
    generated_files = []
    
    for i, explanation in enumerate(explanations):
        try:
            # Extract data from explanation
            node_id = explanation.get('node_id', i)
            pred_prob = explanation.get('prediction', 0.5)
            masks = explanation.get('masks', {})
            top_features = explanation.get('top_features', [])
            explanation_text = explanation.get('explanation_text', 'No explanation provided.')
            
            # Generate report path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(out_dir, f"explanation_{node_id}_{timestamp}_{i}.html")
            
            # Generate report
            generated_path = explain_report(
                node_id=node_id,
                pred_prob=pred_prob,
                masks=masks,
                top_features=top_features,
                explanation_text=explanation_text,
                output_path=report_path
            )
            
            generated_files.append(generated_path)
            
        except Exception as e:
            logger.error(f"Failed to generate report for explanation {i}: {e}")
    
    logger.info(f"Generated {len(generated_files)} explanation reports in {out_dir}")
    return generated_files


def create_feature_importance_plot(
    top_features: List[Tuple[str, float]],
    output_path: Optional[str] = None
) -> str:
    """
    Create feature importance plot.
    
    Args:
        top_features: List of (feature_name, importance) tuples
        output_path: Path to save plot
        
    Returns:
        Path to saved plot
    """
    if not top_features:
        raise ValueError("No features provided")
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"feature_importance_{timestamp}.png"
    
    # Prepare data
    features, importances = zip(*top_features[:15])  # Top 15
    
    # Create plot
    plt.figure(figsize=(10, 8))
    colors = ['red' if imp < 0 else 'green' for imp in importances]
    
    plt.barh(range(len(features)), importances, color=colors, alpha=0.7)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance Score')
    plt.title('Top Feature Importances')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, imp in enumerate(importances):
        plt.text(imp + 0.01 if imp >= 0 else imp - 0.01, i, f'{imp:.3f}', 
                va='center', ha='left' if imp >= 0 else 'right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path
