"""
Smart Configuration Selector for hHGTN
Automatically selects optimal component configuration based on dataset characteristics
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class DatasetCharacteristics:
    """Analyze dataset characteristics to determine optimal configuration."""
    
    def __init__(self):
        self.num_nodes = 0
        self.num_edges = 0
        self.num_node_types = 0
        self.num_edge_types = 0
        self.has_temporal = False
        self.has_hyperedges = False
        self.fraud_ratio = 0.0
        self.is_synthetic = False
        self.is_adversarial = False
        self.graph_type = "unknown"  # homogeneous, heterogeneous, hypergraph
        
    def analyze_from_data(self, data):
        """Analyze dataset characteristics from loaded data."""
        
        try:
            # Analyze node characteristics
            if hasattr(data, 'x_dict'):
                self.num_node_types = len(data.x_dict)
                self.num_nodes = sum(x.size(0) for x in data.x_dict.values())
                self.graph_type = "heterogeneous" if self.num_node_types > 1 else "homogeneous"
            elif hasattr(data, 'x'):
                self.num_nodes = data.x.size(0)
                self.num_node_types = 1
                self.graph_type = "homogeneous"
            
            # Analyze edge characteristics
            if hasattr(data, 'edge_index_dict'):
                self.num_edge_types = len(data.edge_index_dict)
                self.num_edges = sum(edge_idx.size(1) for edge_idx in data.edge_index_dict.values())
            elif hasattr(data, 'edge_index'):
                self.num_edges = data.edge_index.size(1)
                self.num_edge_types = 1
            
            # Check for temporal information
            self.has_temporal = (
                hasattr(data, 'edge_time') or 
                hasattr(data, 'timestamp') or
                hasattr(data, 'time_dict')
            )
            
            # Check for hyperedges (3+ nodes per edge)
            self.has_hyperedges = self._detect_hyperedges(data)
            
            # Analyze fraud ratio if labels available
            if hasattr(data, 'y'):
                labels = data.y
                if labels.dim() > 0 and len(labels) > 0:
                    fraud_count = (labels == 1).sum().item()
                    total_count = len(labels)
                    self.fraud_ratio = fraud_count / total_count if total_count > 0 else 0.0
            
            logger.info(f"Dataset analysis: {self.num_nodes} nodes, {self.num_edges} edges, "
                       f"{self.num_node_types} node types, fraud ratio: {self.fraud_ratio:.3f}")
                       
        except Exception as e:
            logger.warning(f"Error analyzing dataset: {e}. Using default characteristics.")
    
    def _detect_hyperedges(self, data) -> bool:
        """Detect if dataset has hyperedges (simplified heuristic)."""
        
        # Simple heuristic: if we have >2 node types and complex edge patterns
        if self.num_node_types > 2 and self.num_edge_types > 2:
            return True
            
        # Could add more sophisticated hyperedge detection here
        return False
    
    def get_complexity_score(self) -> float:
        """Calculate dataset complexity score (0-1)."""
        
        complexity = 0.0
        
        # Size complexity
        if self.num_nodes > 10000:
            complexity += 0.3
        elif self.num_nodes > 1000:
            complexity += 0.1
            
        # Type complexity
        if self.num_node_types > 3:
            complexity += 0.3
        elif self.num_node_types > 1:
            complexity += 0.1
            
        # Temporal complexity
        if self.has_temporal:
            complexity += 0.2
            
        # Hypergraph complexity
        if self.has_hyperedges:
            complexity += 0.2
            
        return min(complexity, 1.0)


class SmartConfigSelector:
    """Smart configuration selector for hHGTN based on dataset characteristics."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.dataset_configs = self._load_dataset_configs()
        
    def _load_dataset_configs(self) -> Dict[str, Any]:
        """Load dataset-specific configurations."""
        
        config_path = self.config_dir / "dataset_configs.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Dataset configs not found at {config_path}")
            return {}
    
    def select_config(self, 
                     dataset_name: Optional[str] = None,
                     dataset_characteristics: Optional[DatasetCharacteristics] = None,
                     data = None,
                     mode: str = "auto") -> Dict[str, Any]:
        """
        Select optimal configuration for dataset.
        
        Args:
            dataset_name: Known dataset name (e.g., "ellipticpp", "tabular")
            dataset_characteristics: Pre-analyzed characteristics
            data: Raw data for analysis
            mode: Selection mode ("auto", "conservative", "aggressive", "benchmark")
            
        Returns:
            Optimal configuration dictionary
        """
        
        # 1. If specific dataset name provided and exists in configs
        if dataset_name and dataset_name in self.dataset_configs:
            logger.info(f"Using predefined config for dataset: {dataset_name}")
            return self.dataset_configs[dataset_name]
        
        # 2. Analyze dataset characteristics if not provided
        if dataset_characteristics is None and data is not None:
            dataset_characteristics = DatasetCharacteristics()
            dataset_characteristics.analyze_from_data(data)
        
        # 3. Auto-select based on characteristics
        if dataset_characteristics:
            return self._auto_select_config(dataset_characteristics, mode)
        
        # 4. Fallback to development config
        logger.warning("No dataset information available, using development config")
        return self.dataset_configs.get("development", self._get_default_config())
    
    def _auto_select_config(self, 
                           characteristics: DatasetCharacteristics, 
                           mode: str) -> Dict[str, Any]:
        """Automatically select configuration based on dataset characteristics."""
        
        logger.info(f"Auto-selecting config for {characteristics.graph_type} graph with "
                   f"{characteristics.num_nodes} nodes, {characteristics.num_node_types} node types")
        
        # Start with base configuration
        config = self._get_base_config()
        
        # Apply mode-specific settings
        if mode == "conservative":
            config = self._apply_conservative_settings(config)
        elif mode == "aggressive":
            config = self._apply_aggressive_settings(config)
        elif mode == "benchmark":
            return self.dataset_configs.get("benchmark", config)
        
        # Apply dataset-specific rules
        config = self._apply_characteristic_rules(config, characteristics)
        
        return config
    
    def _get_base_config(self) -> Dict[str, Any]:
        """Get base configuration template."""
        
        return {
            'model': {
                'name': 'hHGTN_AutoConfig',
                'use_hypergraph': False,
                'use_hetero': True,
                'use_memory': False,
                'use_cusp': False,
                'use_tdgnn': False,
                'use_gsampler': False,
                'use_spottarget': False,
                'use_robustness': False,
                'hidden_dim': 64,
                'num_layers': 2,
                'dropout': 0.1
            },
            'training': {
                'mode': 'lite',
                'batch_size': 256,
                'learning_rate': 0.001,
                'epochs': 50,
                'loss_type': 'bce'
            }
        }
    
    def _apply_characteristic_rules(self, 
                                  config: Dict[str, Any], 
                                  characteristics: DatasetCharacteristics) -> Dict[str, Any]:
        """Apply rules based on dataset characteristics."""
        
        model_config = config['model']
        training_config = config['training']
        
        # Rule 1: Heterogeneous data
        if characteristics.num_node_types > 1:
            model_config['use_hetero'] = True
            logger.info("✓ Enabled heterogeneous layers (multiple node types)")
        
        # Rule 2: Complex heterogeneous (>2 types) -> enable hypergraph
        if characteristics.num_node_types > 2:
            model_config['use_hypergraph'] = True
            model_config['hidden_dim'] = max(model_config['hidden_dim'], 96)
            logger.info("✓ Enabled hypergraph layers (>2 node types)")
        
        # Rule 3: Temporal data
        if characteristics.has_temporal:
            model_config['use_memory'] = True
            model_config['use_tdgnn'] = True
            model_config['hidden_dim'] = max(model_config['hidden_dim'], 128)
            logger.info("✓ Enabled temporal components (temporal data detected)")
        
        # Rule 4: Large graphs (>10K nodes)
        if characteristics.num_nodes > 10000:
            model_config['use_gsampler'] = True
            model_config['use_memory'] = False  # Memory expensive for large graphs
            training_config['batch_size'] = min(training_config['batch_size'], 128)
            logger.info("✓ Enabled sampling, disabled memory (large graph)")
        
        # Rule 5: Scale-free networks (high node count + heterogeneous)
        if characteristics.num_nodes > 1000 and characteristics.num_node_types > 1:
            model_config['use_cusp'] = True
            logger.info("✓ Enabled CUSP embeddings (scale-free characteristics)")
        
        # Rule 6: Imbalanced fraud data
        if characteristics.fraud_ratio < 0.1 and characteristics.fraud_ratio > 0:
            training_config['loss_type'] = 'focal'
            training_config['focal_gamma'] = 2.0
            model_config['use_spottarget'] = True
            logger.info(f"✓ Enabled focal loss and SpotTarget (fraud ratio: {characteristics.fraud_ratio:.3f})")
        
        # Rule 7: Complex datasets need robustness
        complexity = characteristics.get_complexity_score()
        if complexity > 0.5:
            model_config['use_robustness'] = True
            model_config['num_layers'] = min(model_config['num_layers'] + 1, 4)
            logger.info(f"✓ Enabled robustness (complexity score: {complexity:.2f})")
        
        # Rule 8: Adjust hidden dimensions based on complexity
        if complexity > 0.7:
            model_config['hidden_dim'] = 128
        elif complexity > 0.4:
            model_config['hidden_dim'] = 96
        
        return config
    
    def _apply_conservative_settings(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply conservative settings (faster, simpler)."""
        
        model_config = config['model']
        
        # Disable expensive components
        model_config['use_memory'] = False
        model_config['use_cusp'] = False
        model_config['use_tdgnn'] = False
        
        # Smaller model
        model_config['hidden_dim'] = min(model_config['hidden_dim'], 64)
        model_config['num_layers'] = min(model_config['num_layers'], 2)
        
        # Faster training
        config['training']['epochs'] = min(config['training']['epochs'], 30)
        config['training']['learning_rate'] = 0.003
        
        logger.info("Applied conservative settings (faster training)")
        return config
    
    def _apply_aggressive_settings(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply aggressive settings (more components, larger model)."""
        
        model_config = config['model']
        
        # Enable most components
        model_config['use_hypergraph'] = True
        model_config['use_memory'] = True
        model_config['use_cusp'] = True
        model_config['use_robustness'] = True
        
        # Larger model
        model_config['hidden_dim'] = max(model_config['hidden_dim'], 128)
        model_config['num_layers'] = max(model_config['num_layers'], 3)
        
        # More training
        config['training']['epochs'] = max(config['training']['epochs'], 100)
        config['training']['mode'] = 'full'
        
        logger.info("Applied aggressive settings (full feature set)")
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get failsafe default configuration."""
        
        return {
            'model': {
                'name': 'hHGTN_Default',
                'use_hypergraph': False,
                'use_hetero': True,
                'use_memory': False,
                'use_cusp': False,
                'use_tdgnn': False,
                'use_gsampler': False,
                'use_spottarget': False,
                'use_robustness': False,
                'hidden_dim': 64,
                'num_layers': 2
            },
            'training': {
                'mode': 'lite',
                'batch_size': 64,
                'learning_rate': 0.001,
                'epochs': 20
            }
        }
    
    def print_config_summary(self, config: Dict[str, Any]):
        """Print human-readable configuration summary."""
        
        model_config = config.get('model', {})
        
        print("\n" + "="*60)
        print("🎯 hHGTN CONFIGURATION SUMMARY")
        print("="*60)
        
        # Active components
        components = []
        if model_config.get('use_hypergraph'): components.append('Hypergraph')
        if model_config.get('use_hetero'): components.append('Heterogeneous')
        if model_config.get('use_memory'): components.append('Memory')
        if model_config.get('use_cusp'): components.append('CUSP')
        if model_config.get('use_tdgnn'): components.append('TDGNN')
        if model_config.get('use_gsampler'): components.append('GSampler')
        if model_config.get('use_spottarget'): components.append('SpotTarget')
        if model_config.get('use_robustness'): components.append('Robustness')
        
        print(f"📦 Active Components ({len(components)}): {', '.join(components)}")
        print(f"🏗️  Architecture: {model_config.get('hidden_dim', 64)}D hidden, "
              f"{model_config.get('num_layers', 2)} layers")
        
        training_config = config.get('training', {})
        print(f"🚀 Training: {training_config.get('mode', 'lite')} mode, "
              f"{training_config.get('epochs', 50)} epochs")
        
        print("="*60)


def test_smart_selector():
    """Test the smart configuration selector."""
    
    print("🧪 Testing Smart Configuration Selector...")
    
    selector = SmartConfigSelector()
    
    # Test 1: Known dataset
    print("\n1. Testing known dataset (ellipticpp):")
    config1 = selector.select_config(dataset_name="ellipticpp")
    selector.print_config_summary(config1)
    
    # Test 2: Mock characteristics for large heterogeneous temporal dataset
    print("\n2. Testing large heterogeneous temporal dataset:")
    chars = DatasetCharacteristics()
    chars.num_nodes = 15000
    chars.num_node_types = 3
    chars.has_temporal = True
    chars.fraud_ratio = 0.05
    chars.graph_type = "heterogeneous"
    
    config2 = selector.select_config(dataset_characteristics=chars)
    selector.print_config_summary(config2)
    
    # Test 3: Small homogeneous dataset
    print("\n3. Testing small homogeneous dataset:")
    chars3 = DatasetCharacteristics()
    chars3.num_nodes = 500
    chars3.num_node_types = 1
    chars3.has_temporal = False
    chars3.fraud_ratio = 0.2
    chars3.graph_type = "homogeneous"
    
    config3 = selector.select_config(dataset_characteristics=chars3, mode="conservative")
    selector.print_config_summary(config3)
    
    print("\n✅ Smart selector tests completed!")


if __name__ == "__main__":
    test_smart_selector()
