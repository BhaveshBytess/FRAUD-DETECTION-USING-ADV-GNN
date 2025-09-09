"""
Memory State Visualization for Temporal Graph Neural Networks

This module provides visualization tools for memory states in TGN models,
helping to understand how node memories evolve over time and interactions.

Key features:
- Memory evolution tracking
- Temporal memory patterns
- Memory distribution analysis
- Interactive visualizations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


class MemoryVisualizer:
    """
    Visualizes memory states and evolution in temporal graph neural networks.
    """
    
    def __init__(self, save_path: str = "visualizations"):
        self.save_path = save_path
        self.memory_history = []
        self.interaction_history = []
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def track_memory_state(
        self, 
        memory_state: torch.Tensor, 
        timestamp: float,
        node_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Track memory state at a given timestamp.
        
        Args:
            memory_state: Memory tensor (num_nodes, memory_dim)
            timestamp: Current timestamp
            node_ids: Node IDs (if subset)
            labels: Node labels for coloring
        """
        memory_snapshot = {
            'timestamp': timestamp,
            'memory': memory_state.detach().cpu().numpy(),
            'node_ids': node_ids.cpu().numpy() if node_ids is not None else np.arange(len(memory_state)),
            'labels': labels.cpu().numpy() if labels is not None else None
        }
        
        self.memory_history.append(memory_snapshot)
    
    def track_interaction(
        self,
        source_node: int,
        target_node: int,
        timestamp: float,
        memory_before: torch.Tensor,
        memory_after: torch.Tensor
    ):
        """Track memory changes due to interactions."""
        interaction = {
            'timestamp': timestamp,
            'source': source_node,
            'target': target_node,
            'memory_before': memory_before.detach().cpu().numpy(),
            'memory_after': memory_after.detach().cpu().numpy()
        }
        
        self.interaction_history.append(interaction)
    
    def plot_memory_evolution(
        self, 
        node_ids: List[int], 
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        Plot memory evolution over time for specific nodes.
        
        Args:
            node_ids: List of node IDs to visualize
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.memory_history:
            raise ValueError("No memory history tracked")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Memory State Evolution Over Time', fontsize=16)
        
        # Prepare data
        timestamps = [snapshot['timestamp'] for snapshot in self.memory_history]
        
        for i, node_id in enumerate(node_ids[:4]):  # Max 4 nodes
            ax = axes[i // 2, i % 2]
            
            # Get memory values for this node over time
            memory_values = []
            for snapshot in self.memory_history:
                node_idx = np.where(snapshot['node_ids'] == node_id)[0]
                if len(node_idx) > 0:
                    # Use L2 norm of memory vector
                    memory_norm = np.linalg.norm(snapshot['memory'][node_idx[0]])
                    memory_values.append(memory_norm)
                else:
                    memory_values.append(0.0)
            
            ax.plot(timestamps, memory_values, marker='o', linewidth=2, markersize=4)
            ax.set_title(f'Node {node_id} Memory Evolution')
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Memory L2 Norm')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'{self.save_path}/memory_evolution.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_memory_distribution(
        self, 
        timestamp_idx: int = -1,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot memory distribution at a specific timestamp.
        
        Args:
            timestamp_idx: Index of timestamp to visualize (-1 for latest)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.memory_history:
            raise ValueError("No memory history tracked")
        
        snapshot = self.memory_history[timestamp_idx]
        memory_data = snapshot['memory']
        labels = snapshot['labels']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Memory Distribution at t={snapshot["timestamp"]:.2f}', fontsize=16)
        
        # 1. Memory norm distribution
        memory_norms = np.linalg.norm(memory_data, axis=1)
        axes[0, 0].hist(memory_norms, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Memory L2 Norm Distribution')
        axes[0, 0].set_xlabel('L2 Norm')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Memory dimension variance
        memory_var = np.var(memory_data, axis=0)
        axes[0, 1].plot(memory_var, marker='o')
        axes[0, 1].set_title('Variance Across Memory Dimensions')
        axes[0, 1].set_xlabel('Memory Dimension')
        axes[0, 1].set_ylabel('Variance')
        
        # 3. PCA visualization
        if memory_data.shape[0] > 2:
            pca = PCA(n_components=2)
            memory_pca = pca.fit_transform(memory_data)
            
            if labels is not None:
                scatter = axes[1, 0].scatter(
                    memory_pca[:, 0], memory_pca[:, 1], 
                    c=labels, cmap='viridis', alpha=0.6
                )
                plt.colorbar(scatter, ax=axes[1, 0])
            else:
                axes[1, 0].scatter(memory_pca[:, 0], memory_pca[:, 1], alpha=0.6)
            
            axes[1, 0].set_title('Memory PCA (2D)')
            axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        
        # 4. Memory activation heatmap (sample)
        sample_indices = np.random.choice(len(memory_data), min(50, len(memory_data)), replace=False)
        sample_memory = memory_data[sample_indices]
        
        sns.heatmap(
            sample_memory[:20, :20].T,  # Show subset for visibility
            ax=axes[1, 1],
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Memory Value'}
        )
        axes[1, 1].set_title('Memory Activation Heatmap (Sample)')
        axes[1, 1].set_xlabel('Node Sample')
        axes[1, 1].set_ylabel('Memory Dimension')
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'{self.save_path}/memory_distribution.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_interaction_impact(
        self, 
        max_interactions: int = 20,
        figsize: Tuple[int, int] = (15, 8)
    ) -> plt.Figure:
        """
        Plot impact of interactions on memory states.
        
        Args:
            max_interactions: Maximum number of interactions to show
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.interaction_history:
            raise ValueError("No interaction history tracked")
        
        # Sample interactions
        interactions = self.interaction_history[:max_interactions]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Memory Changes Due to Interactions', fontsize=16)
        
        # 1. Memory change magnitude over time
        timestamps = [inter['timestamp'] for inter in interactions]
        memory_changes = []
        
        for inter in interactions:
            change = np.linalg.norm(inter['memory_after'] - inter['memory_before'])
            memory_changes.append(change)
        
        axes[0, 0].plot(timestamps, memory_changes, marker='o', linewidth=2)
        axes[0, 0].set_title('Memory Change Magnitude Over Time')
        axes[0, 0].set_xlabel('Timestamp')
        axes[0, 0].set_ylabel('L2 Norm of Change')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribution of memory changes
        axes[0, 1].hist(memory_changes, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Distribution of Memory Changes')
        axes[0, 1].set_xlabel('Change Magnitude')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Memory change per dimension
        if interactions:
            change_per_dim = []
            for inter in interactions:
                diff = inter['memory_after'] - inter['memory_before']
                change_per_dim.append(np.abs(diff))
            
            change_per_dim = np.array(change_per_dim)
            mean_change = np.mean(change_per_dim, axis=0)
            
            axes[1, 0].plot(mean_change, marker='o')
            axes[1, 0].set_title('Average Change per Memory Dimension')
            axes[1, 0].set_xlabel('Memory Dimension')
            axes[1, 0].set_ylabel('Average Absolute Change')
        
        # 4. Before vs After memory comparison (sample)
        if interactions:
            sample_inter = interactions[0]
            mem_before = sample_inter['memory_before']
            mem_after = sample_inter['memory_after']
            
            x = np.arange(len(mem_before))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, mem_before, width, label='Before', alpha=0.7)
            axes[1, 1].bar(x + width/2, mem_after, width, label='After', alpha=0.7)
            axes[1, 1].set_title('Memory Before vs After (Sample Interaction)')
            axes[1, 1].set_xlabel('Memory Dimension')
            axes[1, 1].set_ylabel('Memory Value')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'{self.save_path}/interaction_impact.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_memory_plot(
        self, 
        use_tsne: bool = True,
        perplexity: int = 30
    ) -> go.Figure:
        """
        Create interactive 3D visualization of memory evolution.
        
        Args:
            use_tsne: Whether to use t-SNE for dimensionality reduction
            perplexity: t-SNE perplexity parameter
            
        Returns:
            Plotly figure
        """
        if not self.memory_history:
            raise ValueError("No memory history tracked")
        
        # Prepare data
        all_memories = []
        all_timestamps = []
        all_node_ids = []
        all_labels = []
        
        for snapshot in self.memory_history:
            memory_data = snapshot['memory']
            timestamp = snapshot['timestamp']
            node_ids = snapshot['node_ids']
            labels = snapshot['labels']
            
            # Apply dimensionality reduction
            if use_tsne and memory_data.shape[1] > 3:
                if len(memory_data) > 50:  # Only if enough samples
                    reducer = TSNE(n_components=3, perplexity=min(perplexity, len(memory_data)-1), random_state=42)
                    memory_reduced = reducer.fit_transform(memory_data)
                else:
                    pca = PCA(n_components=3)
                    memory_reduced = pca.fit_transform(memory_data)
            else:
                # Use first 3 dimensions or PCA
                if memory_data.shape[1] >= 3:
                    memory_reduced = memory_data[:, :3]
                else:
                    # Pad with zeros if less than 3 dimensions
                    padding = np.zeros((memory_data.shape[0], 3 - memory_data.shape[1]))
                    memory_reduced = np.hstack([memory_data, padding])
            
            all_memories.append(memory_reduced)
            all_timestamps.extend([timestamp] * len(memory_reduced))
            all_node_ids.extend(node_ids)
            all_labels.extend(labels if labels is not None else [0] * len(memory_reduced))
        
        # Combine all data
        all_memories = np.vstack(all_memories)
        
        # Create DataFrame
        df = pd.DataFrame({
            'x': all_memories[:, 0],
            'y': all_memories[:, 1],
            'z': all_memories[:, 2],
            'timestamp': all_timestamps,
            'node_id': all_node_ids,
            'label': all_labels
        })
        
        # Create animated 3D scatter plot
        fig = px.scatter_3d(
            df, 
            x='x', y='y', z='z',
            color='label',
            animation_frame='timestamp',
            hover_data=['node_id'],
            title='Memory State Evolution in 3D Space',
            labels={'label': 'Node Type'}
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Memory Dim 1',
                yaxis_title='Memory Dim 2',
                zaxis_title='Memory Dim 3'
            ),
            width=800,
            height=600
        )
        
        # Save interactive plot
        fig.write_html(f'{self.save_path}/memory_evolution_3d.html')
        
        return fig
    
    def generate_memory_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive memory analysis report.
        
        Returns:
            Dictionary with analysis results
        """
        if not self.memory_history:
            return {"error": "No memory history available"}
        
        report = {}
        
        # Basic statistics
        final_snapshot = self.memory_history[-1]
        memory_data = final_snapshot['memory']
        
        report['basic_stats'] = {
            'num_nodes': len(memory_data),
            'memory_dim': memory_data.shape[1],
            'num_snapshots': len(self.memory_history),
            'time_span': self.memory_history[-1]['timestamp'] - self.memory_history[0]['timestamp']
        }
        
        # Memory statistics
        memory_norms = np.linalg.norm(memory_data, axis=1)
        report['memory_stats'] = {
            'mean_memory_norm': float(np.mean(memory_norms)),
            'std_memory_norm': float(np.std(memory_norms)),
            'min_memory_norm': float(np.min(memory_norms)),
            'max_memory_norm': float(np.max(memory_norms)),
            'memory_sparsity': float(np.mean(memory_data == 0))
        }
        
        # Interaction statistics
        if self.interaction_history:
            memory_changes = []
            for inter in self.interaction_history:
                change = np.linalg.norm(inter['memory_after'] - inter['memory_before'])
                memory_changes.append(change)
            
            report['interaction_stats'] = {
                'num_interactions': len(self.interaction_history),
                'mean_memory_change': float(np.mean(memory_changes)),
                'std_memory_change': float(np.std(memory_changes)),
                'max_memory_change': float(np.max(memory_changes))
            }
        
        return report
    
    def save_memory_states(self, filename: str = "memory_states.npz"):
        """Save memory states for later analysis."""
        if not self.memory_history:
            print("No memory history to save")
            return
        
        # Prepare data for saving
        save_data = {
            'timestamps': [s['timestamp'] for s in self.memory_history],
            'memory_states': [s['memory'] for s in self.memory_history],
            'node_ids': [s['node_ids'] for s in self.memory_history],
            'labels': [s['labels'] for s in self.memory_history]
        }
        
        np.savez_compressed(f'{self.save_path}/{filename}', **save_data)
        print(f"Memory states saved to {self.save_path}/{filename}")


def create_memory_visualizer(save_path: str = "visualizations") -> MemoryVisualizer:
    """Create and return a memory visualizer instance."""
    import os
    os.makedirs(save_path, exist_ok=True)
    return MemoryVisualizer(save_path)


if __name__ == "__main__":
    # Test memory visualization
    print("Testing memory visualization...")
    
    # Create sample memory data
    num_nodes = 100
    memory_dim = 64
    num_snapshots = 10
    
    visualizer = create_memory_visualizer("test_visualizations")
    
    # Generate sample memory evolution
    for t in range(num_snapshots):
        # Create evolving memory state
        base_memory = torch.randn(num_nodes, memory_dim) * (1 + 0.1 * t)
        node_ids = torch.arange(num_nodes)
        labels = torch.randint(0, 2, (num_nodes,))
        
        visualizer.track_memory_state(base_memory, float(t), node_ids, labels)
        
        # Add some interactions
        if t > 0:
            for _ in range(5):
                src, tgt = np.random.choice(num_nodes, 2, replace=False)
                mem_before = base_memory[tgt:tgt+1]
                mem_after = mem_before + torch.randn_like(mem_before) * 0.1
                
                visualizer.track_interaction(src, tgt, float(t), mem_before, mem_after)
    
    # Generate visualizations
    print("\n1. Plotting memory evolution...")
    fig1 = visualizer.plot_memory_evolution([0, 1, 2, 3])
    plt.show()
    
    print("\n2. Plotting memory distribution...")
    fig2 = visualizer.plot_memory_distribution()
    plt.show()
    
    print("\n3. Plotting interaction impact...")
    fig3 = visualizer.plot_interaction_impact()
    plt.show()
    
    print("\n4. Creating interactive 3D plot...")
    fig4 = visualizer.create_interactive_memory_plot()
    print("Interactive plot saved as HTML")
    
    print("\n5. Generating memory report...")
    report = visualizer.generate_memory_report()
    print("Memory Report:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Memory visualization working correctly!")
