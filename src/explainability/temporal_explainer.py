"""
Temporal explainer helpers and wrappers for TGN-like models.

Provides simplified temporal explainer interface compatible with the main
explainability pipeline.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import logging

from .gnne_explainers import BaseExplainer

logger = logging.getLogger(__name__)


class SimplifiedTemporalExplainer(BaseExplainer):
    """
    Simplified temporal explainer that works with any temporal model.
    
    Uses event masking approach to identify important temporal patterns.
    """
    
    def __init__(self, model, window_size: int = 10, device: str = 'cpu', seed: int = 0):
        super().__init__(model, device, seed)
        self.window_size = window_size
        
        logger.info(f"Initialized SimplifiedTemporalExplainer with window_size={window_size}")
    
    def explain_node(self, node_id: int, x: torch.Tensor, edge_index: torch.Tensor, 
                    label: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Base explain_node interface for compatibility.
        For temporal explanation, use explain_temporal_prediction instead.
        """
        # Create dummy temporal data for interface compatibility
        edge_time = torch.arange(edge_index.size(1), dtype=torch.float)
        
        return {
            'edge_mask': torch.ones(edge_index.size(1)) * 0.5,
            'node_feat_mask': None,
            'important_subgraph': {
                'edge_index': edge_index,
                'edge_weights': torch.ones(edge_index.size(1)) * 0.5
            },
            'explanation_type': 'temporal_simplified'
        }
    
    def explain_temporal_prediction(self, node_id: int, event_sequence: List[Dict], 
                                  target_time: float) -> Dict[str, Any]:
        """
        Explain temporal prediction by identifying important events/time windows.
        
        Args:
            node_id: Target node ID
            event_sequence: List of events with 'time', 'src', 'dst', 'features'
            target_time: Time of prediction
            
        Returns:
            Dictionary with temporal importance and event masks
        """
        self._set_seed()
        
        # Filter events to those before target time (causality)
        relevant_events = [e for e in event_sequence if e['time'] <= target_time]
        
        if len(relevant_events) == 0:
            return {
                'event_importance': [],
                'time_window_importance': {},
                'explanation_type': 'temporal_simplified'
            }
        
        # Compute importance for each event (simplified heuristic)
        event_importance = self._compute_event_importance(relevant_events, node_id, target_time)
        
        # Create time windows and their importance
        time_windows = self._create_time_windows(relevant_events, event_importance)
        
        return {
            'event_importance': event_importance,
            'time_window_importance': time_windows,
            'relevant_events': relevant_events,
            'explanation_type': 'temporal_simplified'
        }
    
    def _compute_event_importance(self, events: List[Dict], node_id: int, target_time: float) -> List[float]:
        """Compute importance score for each event."""
        importance_scores = []
        
        for event in events:
            # Simple heuristics for event importance
            score = 0.0
            
            # 1. Temporal proximity (closer to target time = more important)
            time_diff = abs(target_time - event['time'])
            max_time_diff = max(abs(target_time - e['time']) for e in events)
            temporal_score = 1.0 - (time_diff / (max_time_diff + 1e-8))
            score += temporal_score * 0.4
            
            # 2. Node involvement (events involving target node are more important)
            if event.get('src') == node_id or event.get('dst') == node_id:
                score += 0.6
            
            # 3. Event frequency (rare events might be more important)
            # This is simplified - in practice you'd compute actual frequencies
            score += 0.1  # Base frequency score
            
            # Ensure score stays in [0,1] range
            score = min(1.0, max(0.0, score))
            
            importance_scores.append(score)
        
        return importance_scores
    
    def _create_time_windows(self, events: List[Dict], importance_scores: List[float]) -> Dict[str, Any]:
        """Create time windows with aggregated importance."""
        if len(events) == 0:
            return {}
        
        # Get time range
        times = [e['time'] for e in events]
        min_time, max_time = min(times), max(times)
        
        # Create time bins
        window_size = (max_time - min_time) / self.window_size
        windows = {}
        
        for i in range(self.window_size):
            window_start = min_time + i * window_size
            window_end = min_time + (i + 1) * window_size
            
            # Find events in this window
            window_events = []
            window_importance = []
            
            for j, event in enumerate(events):
                if window_start <= event['time'] < window_end:
                    window_events.append(event)
                    window_importance.append(importance_scores[j])
            
            if window_events:
                windows[f"window_{i}"] = {
                    'time_range': (window_start, window_end),
                    'num_events': len(window_events),
                    'avg_importance': sum(window_importance) / len(window_importance),
                    'max_importance': max(window_importance),
                    'events': window_events
                }
        
        return windows


# Compatibility wrapper for the temporal explainer interface
def create_temporal_explainer(model, explainer_type: str = 'simplified', **kwargs):
    """
    Create temporal explainer based on type.
    
    Args:
        model: Temporal model to explain
        explainer_type: Type of temporal explainer
        **kwargs: Additional arguments
        
    Returns:
        Temporal explainer instance
    """
    if explainer_type == 'simplified':
        return SimplifiedTemporalExplainer(model, **kwargs)
    else:
        raise ValueError(f"Unknown temporal explainer type: {explainer_type}")
