"""
Smart Configuration Demo - Test dataset adaptability without full model
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from config_selector import SmartConfigSelector, DatasetCharacteristics


def demo_dataset_adaptability():
    """Demonstrate smart configuration for different dataset types."""
    
    print("🎯 DATASET ADAPTABILITY DEMO")
    print("="*60)
    print("This demo shows how hHGTN automatically adapts its configuration")
    print("based on dataset characteristics to avoid component conflicts.\n")
    
    selector = SmartConfigSelector()
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'EllipticPP (Large Heterogeneous Financial)',
            'dataset': 'ellipticpp',
            'characteristics': None
        },
        {
            'name': 'Tabular Converted (Synthetic Graph)',
            'dataset': 'tabular_converted',
            'characteristics': None
        },
        {
            'name': 'Large Social Network (Auto-detected)',
            'dataset': None,
            'characteristics': create_mock_characteristics(
                num_nodes=50000,
                num_node_types=1,
                has_temporal=True,
                fraud_ratio=0.02,
                graph_type='homogeneous'
            )
        },
        {
            'name': 'Complex Hypergraph (Auto-detected)',
            'dataset': None,
            'characteristics': create_mock_characteristics(
                num_nodes=8000,
                num_node_types=4,
                has_temporal=False,
                has_hyperedges=True,
                fraud_ratio=0.15,
                graph_type='hypergraph'
            )
        },
        {
            'name': 'Small Development Dataset',
            'dataset': 'development',
            'characteristics': None
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print("-" * 50)
        
        config = selector.select_config(
            dataset_name=scenario['dataset'],
            dataset_characteristics=scenario['characteristics']
        )
        
        # Analyze compatibility
        analyze_configuration_compatibility(config, scenario['name'])
        
        # Show configuration summary
        selector.print_config_summary(config)


def create_mock_characteristics(num_nodes, num_node_types, has_temporal=False, 
                               has_hyperedges=False, fraud_ratio=0.1, graph_type='homogeneous'):
    """Create mock dataset characteristics for testing."""
    
    characteristics = DatasetCharacteristics()
    characteristics.num_nodes = num_nodes
    characteristics.num_node_types = num_node_types
    characteristics.num_edges = num_nodes * 3  # Typical graph density
    characteristics.has_temporal = has_temporal
    characteristics.has_hyperedges = has_hyperedges
    characteristics.fraud_ratio = fraud_ratio
    characteristics.graph_type = graph_type
    
    return characteristics


def analyze_configuration_compatibility(config, scenario_name):
    """Analyze whether a configuration is likely to be compatible."""
    
    model_config = config.get('model', {})
    
    # Count active components
    active_components = sum(1 for key, value in model_config.items() 
                          if key.startswith('use_') and value)
    
    # Check for common incompatibility patterns
    warnings = []
    recommendations = []
    
    # 1. Too many components for small datasets
    hidden_dim = model_config.get('hidden_dim', 64)
    num_layers = model_config.get('num_layers', 2)
    
    if active_components > 5 and hidden_dim < 96:
        warnings.append("⚠️  Many components with small hidden dimension may cause dimension mismatches")
        recommendations.append("💡 Consider increasing hidden_dim to 128+ or reducing components")
    
    # 2. Memory + Large graphs
    if model_config.get('use_memory') and model_config.get('use_gsampler'):
        warnings.append("⚠️  Memory module with graph sampling may be inefficient")
        recommendations.append("💡 Consider disabling memory for large graphs with sampling")
    
    # 3. Hypergraph without heterogeneous
    if model_config.get('use_hypergraph') and not model_config.get('use_hetero'):
        warnings.append("⚠️  Hypergraph layers typically require heterogeneous support")
        recommendations.append("💡 Consider enabling heterogeneous layers with hypergraph")
    
    # 4. SpotTarget without imbalanced data indicator
    training_config = config.get('training', {})
    if (model_config.get('use_spottarget') and 
        training_config.get('loss_type') != 'focal'):
        warnings.append("⚠️  SpotTarget enabled but no focal loss (may not be needed)")
        recommendations.append("💡 SpotTarget is designed for highly imbalanced datasets")
    
    # Compatibility score
    compatibility_issues = len(warnings)
    if compatibility_issues == 0:
        print("✅ High compatibility - No obvious issues detected")
    elif compatibility_issues <= 2:
        print("⚡ Medium compatibility - Minor adjustments may be needed")
    else:
        print("🔥 Low compatibility - Significant adjustments recommended")
    
    # Show issues and recommendations
    for warning in warnings:
        print(f"   {warning}")
    for rec in recommendations:
        print(f"   {rec}")
    
    # Estimated performance characteristics
    print(f"📊 Config Stats: {active_components} components, {hidden_dim}D hidden, {num_layers} layers")


def show_component_interaction_matrix():
    """Show which components work well together."""
    
    print("\n" + "="*60)
    print("🔬 COMPONENT INTERACTION MATRIX")
    print("="*60)
    print("✅ = Good synergy  ⚠️ = Potential issues  ❌ = Known conflicts\n")
    
    components = ['Hypergraph', 'Hetero', 'Memory', 'CUSP', 'TDGNN', 'GSampler', 'SpotTarget', 'Robustness']
    
    # Interaction rules (simplified)
    good_pairs = [
        ('Hypergraph', 'Hetero'),
        ('Memory', 'TDGNN'),
        ('CUSP', 'Hetero'),
        ('SpotTarget', 'Robustness'),
        ('GSampler', 'Robustness')
    ]
    
    warning_pairs = [
        ('Memory', 'GSampler'),
        ('Hypergraph', 'TDGNN'),
        ('CUSP', 'Memory')
    ]
    
    conflict_pairs = [
        # None currently, but could add dimension conflicts
    ]
    
    print("Key Insights:")
    print("• Hypergraph ↔ Heterogeneous: Natural synergy for complex graph structures")
    print("• Memory ↔ TDGNN: Both handle temporal information effectively")
    print("• CUSP ↔ Heterogeneous: Scale-free embeddings work well with multiple node types")
    print("• SpotTarget ↔ Robustness: Both help with adversarial/imbalanced scenarios")
    print("• Memory ⚠️ GSampler: Memory caching conflicts with dynamic sampling")
    print("• CUSP ⚠️ Memory: Different embedding strategies may interfere")


def show_usage_examples():
    """Show practical usage examples."""
    
    print("\n" + "="*60)
    print("💡 PRACTICAL USAGE EXAMPLES")
    print("="*60)
    
    examples = [
        {
            'scenario': 'Financial fraud detection (like EllipticPP)',
            'command': 'python scripts/train_enhanced.py --dataset ellipticpp --test-only',
            'description': 'Uses full feature set optimized for heterogeneous financial networks'
        },
        {
            'scenario': 'Large social network analysis',
            'command': 'python scripts/train_enhanced.py --mode conservative --data social_network.pt',
            'description': 'Automatically adapts to large homogeneous graphs with sampling'
        },
        {
            'scenario': 'Quick prototyping/testing',
            'command': 'python scripts/train_enhanced.py --dataset development --epochs 10',
            'description': 'Minimal configuration for fast iteration and debugging'
        },
        {
            'scenario': 'Benchmark comparison',
            'command': 'python scripts/train_enhanced.py --mode benchmark --save-config my_config.yaml',
            'description': 'Standardized configuration for fair model comparisons'
        },
        {
            'scenario': 'Production deployment (stable)',
            'command': 'python scripts/train_enhanced.py --mode conservative --config my_optimized.yaml',
            'description': 'Conservative, well-tested configuration for production systems'
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['scenario']}")
        print(f"   Command: {example['command']}")
        print(f"   Result:  {example['description']}")


if __name__ == "__main__":
    demo_dataset_adaptability()
    show_component_interaction_matrix()
    show_usage_examples()
    
    print(f"\n{'='*60}")
    print("🎉 Demo completed! The smart configuration system addresses your")
    print("concern about component compatibility by automatically selecting")
    print("optimal configurations based on dataset characteristics.")
    print(f"{'='*60}")
