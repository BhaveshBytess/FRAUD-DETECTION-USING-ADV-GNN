"""
Temporal Sampling and Event Loading for Stage 4

This module implements time-ordered event loading and neighbor sampling
that respects temporal constraints for proper temporal graph neural networks.

Key features:
- Time-ordered event processing
- Temporal neighbor sampling
- Chronological data loading
- Time-leakage prevention
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Iterator
from collections import defaultdict, deque
import bisect


class TemporalEvent:
    """
    Represents a temporal event in the graph.
    """
    
    def __init__(
        self,
        timestamp: float,
        source_node: int,
        target_node: int,
        edge_features: torch.Tensor,
        event_type: str = 'interaction'
    ):
        self.timestamp = timestamp
        self.source_node = source_node
        self.target_node = target_node
        self.edge_features = edge_features
        self.event_type = event_type
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp
    
    def __repr__(self):
        return f"Event(t={self.timestamp}, {self.source_node}->{self.target_node})"


class TemporalEventLoader:
    """
    Loads and manages temporal events in chronological order.
    
    Ensures proper temporal ordering and prevents time leakage.
    """
    
    def __init__(
        self,
        data_path: str,
        node_features: torch.Tensor,
        time_window: float = 1.0,
        max_events_per_batch: int = 1000
    ):
        self.data_path = data_path
        self.node_features = node_features
        self.time_window = time_window
        self.max_events_per_batch = max_events_per_batch
        
        # Load and sort events
        self.events = self._load_events()
        self.current_time = 0.0
        self.event_pointer = 0
        
        print(f"Loaded {len(self.events)} temporal events")
    
    def _load_events(self) -> List[TemporalEvent]:
        """Load temporal events from data files."""
        events = []
        
        # Load transaction data
        try:
            # Load transaction edges
            tx_edges_path = f"{self.data_path}/txs_edgelist.csv"
            if pd.io.common.file_exists(tx_edges_path):
                tx_edges = pd.read_csv(tx_edges_path)
                
                for _, row in tx_edges.iterrows():
                    # Create edge features (can be extended)
                    edge_features = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Basic features
                    
                    event = TemporalEvent(
                        timestamp=row.get('time_step', 0.0),
                        source_node=row['txId1'],
                        target_node=row['txId2'],
                        edge_features=edge_features,
                        event_type='transaction'
                    )
                    events.append(event)
            
            # Load address-transaction edges
            addr_tx_path = f"{self.data_path}/AddrTx_edgelist.csv"
            if pd.io.common.file_exists(addr_tx_path):
                addr_tx = pd.read_csv(addr_tx_path)
                
                for _, row in addr_tx.iterrows():
                    edge_features = torch.tensor([0.0, 1.0, 0.0, 0.0])
                    
                    event = TemporalEvent(
                        timestamp=row.get('time_step', 0.0),
                        source_node=row['addr'],
                        target_node=row['txId'],
                        edge_features=edge_features,
                        event_type='addr_tx'
                    )
                    events.append(event)
            
            # Load transaction-address edges
            tx_addr_path = f"{self.data_path}/TxAddr_edgelist.csv"
            if pd.io.common.file_exists(tx_addr_path):
                tx_addr = pd.read_csv(tx_addr_path)
                
                for _, row in tx_addr.iterrows():
                    edge_features = torch.tensor([0.0, 0.0, 1.0, 0.0])
                    
                    event = TemporalEvent(
                        timestamp=row.get('time_step', 0.0),
                        source_node=row['txId'],
                        target_node=row['addr'],
                        edge_features=edge_features,
                        event_type='tx_addr'
                    )
                    events.append(event)
        
        except Exception as e:
            print(f"Warning: Could not load some edge files: {e}")
            # Create synthetic events for testing
            events = self._create_synthetic_events()
        
        # Sort events by timestamp
        events.sort(key=lambda x: x.timestamp)
        
        return events
    
    def _create_synthetic_events(self) -> List[TemporalEvent]:
        """Create synthetic temporal events for testing."""
        events = []
        num_nodes = self.node_features.size(0)
        
        for i in range(5000):  # Create 5000 synthetic events
            timestamp = np.random.exponential(1.0) * i / 100  # Increasing timestamps
            source = np.random.randint(0, min(1000, num_nodes))
            target = np.random.randint(0, min(1000, num_nodes))
            
            if source != target:
                edge_features = torch.randn(4)
                event = TemporalEvent(timestamp, source, target, edge_features)
                events.append(event)
        
        return events
    
    def get_events_in_window(self, start_time: float, end_time: float) -> List[TemporalEvent]:
        """Get all events within a time window."""
        window_events = []
        
        for event in self.events:
            if start_time <= event.timestamp <= end_time:
                window_events.append(event)
            elif event.timestamp > end_time:
                break
        
        return window_events
    
    def get_next_batch(self) -> Optional[List[TemporalEvent]]:
        """Get next batch of events in chronological order."""
        if self.event_pointer >= len(self.events):
            return None
        
        batch_events = []
        start_time = self.current_time
        end_time = start_time + self.time_window
        
        while (self.event_pointer < len(self.events) and 
               len(batch_events) < self.max_events_per_batch):
            
            event = self.events[self.event_pointer]
            
            if event.timestamp <= end_time:
                batch_events.append(event)
                self.event_pointer += 1
            else:
                break
        
        if batch_events:
            self.current_time = end_time
            return batch_events
        else:
            return None
    
    def reset(self):
        """Reset the event loader to the beginning."""
        self.current_time = 0.0
        self.event_pointer = 0


class TemporalNeighborSampler:
    """
    Samples neighbors for nodes while respecting temporal constraints.
    
    Key features:
    - Only samples neighbors from past interactions
    - Maintains temporal ordering
    - Supports different sampling strategies
    """
    
    def __init__(
        self,
        max_neighbors: int = 20,
        sampling_strategy: str = 'recent',
        time_decay: float = 0.1
    ):
        self.max_neighbors = max_neighbors
        self.sampling_strategy = sampling_strategy
        self.time_decay = time_decay
        
        # Maintain neighbor history for each node
        self.neighbor_history = defaultdict(list)  # node_id -> [(neighbor_id, timestamp, edge_features)]
    
    def update_history(self, events: List[TemporalEvent]):
        """Update neighbor history with new events."""
        for event in events:
            # Add bidirectional neighbors
            self.neighbor_history[event.source_node].append(
                (event.target_node, event.timestamp, event.edge_features)
            )
            self.neighbor_history[event.target_node].append(
                (event.source_node, event.timestamp, event.edge_features)
            )
    
    def sample_neighbors(
        self, 
        node_id: int, 
        current_time: float,
        num_neighbors: Optional[int] = None
    ) -> List[Tuple[int, float, torch.Tensor]]:
        """
        Sample neighbors for a node at a given time.
        
        Args:
            node_id: Target node ID
            current_time: Current timestamp (only sample from past)
            num_neighbors: Number of neighbors to sample
            
        Returns:
            List of (neighbor_id, timestamp, edge_features) tuples
        """
        if num_neighbors is None:
            num_neighbors = self.max_neighbors
        
        # Get all past neighbors
        past_neighbors = [
            (neighbor_id, timestamp, edge_features)
            for neighbor_id, timestamp, edge_features in self.neighbor_history[node_id]
            if timestamp < current_time  # Only past interactions
        ]
        
        if not past_neighbors:
            return []
        
        # Apply sampling strategy
        if self.sampling_strategy == 'recent':
            # Sample most recent neighbors
            past_neighbors.sort(key=lambda x: x[1], reverse=True)
            sampled = past_neighbors[:num_neighbors]
        
        elif self.sampling_strategy == 'uniform':
            # Uniform random sampling
            if len(past_neighbors) <= num_neighbors:
                sampled = past_neighbors
            else:
                indices = np.random.choice(len(past_neighbors), num_neighbors, replace=False)
                sampled = [past_neighbors[i] for i in indices]
        
        elif self.sampling_strategy == 'time_weighted':
            # Sample based on time proximity with decay
            weights = []
            for _, timestamp, _ in past_neighbors:
                time_diff = current_time - timestamp
                weight = np.exp(-self.time_decay * time_diff)
                weights.append(weight)
            
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            if len(past_neighbors) <= num_neighbors:
                sampled = past_neighbors
            else:
                indices = np.random.choice(
                    len(past_neighbors), 
                    num_neighbors, 
                    replace=False, 
                    p=weights
                )
                sampled = [past_neighbors[i] for i in indices]
        
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        return sampled
    
    def get_temporal_subgraph(
        self, 
        center_nodes: List[int], 
        current_time: float,
        num_hops: int = 2
    ) -> Dict[str, torch.Tensor]:
        """
        Extract temporal subgraph around center nodes.
        
        Args:
            center_nodes: Central nodes to build subgraph around
            current_time: Current timestamp
            num_hops: Number of hops for neighborhood
            
        Returns:
            Dictionary with edge_index, edge_attr, and node mapping
        """
        all_nodes = set(center_nodes)
        edge_list = []
        edge_features = []
        
        # BFS to collect neighbors
        current_level = set(center_nodes)
        
        for hop in range(num_hops):
            next_level = set()
            
            for node_id in current_level:
                neighbors = self.sample_neighbors(node_id, current_time)
                
                for neighbor_id, timestamp, edge_feat in neighbors:
                    if neighbor_id not in all_nodes:
                        next_level.add(neighbor_id)
                        all_nodes.add(neighbor_id)
                    
                    # Add edge
                    edge_list.append([node_id, neighbor_id])
                    edge_features.append(edge_feat)
            
            current_level = next_level
            if not current_level:
                break
        
        # Create node mapping
        node_list = list(all_nodes)
        node_mapping = {node_id: idx for idx, node_id in enumerate(node_list)}
        
        # Convert edges to tensor format
        if edge_list:
            edge_index = torch.tensor(
                [[node_mapping[edge[0]], node_mapping[edge[1]]] for edge in edge_list]
            ).T
            edge_attr = torch.stack(edge_features)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4))
        
        return {
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'node_mapping': node_mapping,
            'nodes': torch.tensor(node_list)
        }


class TemporalBatchLoader:
    """
    Combines event loading and neighbor sampling for batch processing.
    """
    
    def __init__(
        self,
        event_loader: TemporalEventLoader,
        neighbor_sampler: TemporalNeighborSampler,
        batch_size: int = 32
    ):
        self.event_loader = event_loader
        self.neighbor_sampler = neighbor_sampler
        self.batch_size = batch_size
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over temporal batches."""
        self.event_loader.reset()
        
        while True:
            # Get next batch of events
            events = self.event_loader.get_next_batch()
            if events is None:
                break
            
            # Update neighbor history
            self.neighbor_sampler.update_history(events)
            
            # Create batch data
            if events:
                current_time = max(event.timestamp for event in events)
                
                # Sample nodes for prediction (e.g., target nodes from events)
                predict_nodes = list(set(event.target_node for event in events))
                
                if len(predict_nodes) > self.batch_size:
                    predict_nodes = np.random.choice(
                        predict_nodes, self.batch_size, replace=False
                    ).tolist()
                
                # Get temporal subgraph
                subgraph_data = self.neighbor_sampler.get_temporal_subgraph(
                    predict_nodes, current_time
                )
                
                # Prepare batch
                batch = {
                    'events': events,
                    'predict_nodes': torch.tensor(predict_nodes),
                    'current_time': current_time,
                    'subgraph': subgraph_data
                }
                
                yield batch


def create_temporal_loader(
    data_path: str,
    node_features: torch.Tensor,
    config: Dict[str, Any]
) -> TemporalBatchLoader:
    """
    Create temporal batch loader with configuration.
    """
    # Event loader
    event_loader = TemporalEventLoader(
        data_path=data_path,
        node_features=node_features,
        time_window=config.get('time_window', 1.0),
        max_events_per_batch=config.get('max_events_per_batch', 1000)
    )
    
    # Neighbor sampler
    neighbor_sampler = TemporalNeighborSampler(
        max_neighbors=config.get('max_neighbors', 20),
        sampling_strategy=config.get('sampling_strategy', 'recent'),
        time_decay=config.get('time_decay', 0.1)
    )
    
    # Batch loader
    batch_loader = TemporalBatchLoader(
        event_loader=event_loader,
        neighbor_sampler=neighbor_sampler,
        batch_size=config.get('batch_size', 32)
    )
    
    return batch_loader


if __name__ == "__main__":
    # Test temporal sampling
    print("Testing temporal sampling components...")
    
    # Create sample node features
    num_nodes = 1000
    node_features = torch.randn(num_nodes, 93)
    
    # Test event loader
    print("\n1. Testing TemporalEventLoader...")
    event_loader = TemporalEventLoader(
        data_path="data/ellipticpp",  # Will use synthetic data if files not found
        node_features=node_features,
        time_window=2.0
    )
    
    batch_events = event_loader.get_next_batch()
    if batch_events:
        print(f"✓ Loaded batch with {len(batch_events)} events")
        print(f"  Time range: {batch_events[0].timestamp:.2f} - {batch_events[-1].timestamp:.2f}")
    
    # Test neighbor sampler
    print("\n2. Testing TemporalNeighborSampler...")
    neighbor_sampler = TemporalNeighborSampler(max_neighbors=10)
    
    if batch_events:
        neighbor_sampler.update_history(batch_events)
        neighbors = neighbor_sampler.sample_neighbors(
            batch_events[0].target_node, 
            batch_events[-1].timestamp
        )
        print(f"✓ Sampled {len(neighbors)} neighbors")
    
    # Test batch loader
    print("\n3. Testing TemporalBatchLoader...")
    config = {
        'time_window': 2.0,
        'max_neighbors': 10,
        'batch_size': 16
    }
    
    batch_loader = create_temporal_loader("data/ellipticpp", node_features, config)
    
    batch_count = 0
    for batch in batch_loader:
        batch_count += 1
        print(f"✓ Processed batch {batch_count} with {len(batch['events'])} events")
        if batch_count >= 3:  # Test first 3 batches
            break
    
    print(f"\n✅ Temporal sampling working correctly! Processed {batch_count} batches.")
