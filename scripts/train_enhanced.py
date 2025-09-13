"""
Enhanced hHGTN Training Script with Smart Configuration Selection
Automatically adapts component configuration based on dataset characteristics
"""

import torch
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config_selector import SmartConfigSelector, DatasetCharacteristics
from models.hhgt import hHGTN


class EnhancedHHGTNTrainer:
    """Enhanced trainer with smart configuration selection."""
    
    def __init__(self, 
                 data_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 dataset_name: Optional[str] = None,
                 mode: str = "auto"):
        
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.mode = mode
        
        # Initialize smart selector
        config_dir = Path(__file__).parent.parent / "configs"
        self.config_selector = SmartConfigSelector(str(config_dir))
        
        # Load or generate configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            self.config = self._generate_smart_config()
        
        # Initialize model and training components
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def _generate_smart_config(self) -> Dict[str, Any]:
        """Generate smart configuration based on dataset."""
        
        logger.info("🧠 Generating smart configuration...")
        
        # Try to load and analyze data
        data = None
        characteristics = None
        
        if self.data_path:
            try:
                data = self._load_dataset()
                characteristics = DatasetCharacteristics()
                characteristics.analyze_from_data(data)
                logger.info("✅ Dataset analyzed successfully")
            except Exception as e:
                logger.warning(f"Could not analyze dataset: {e}")
        
        # Use smart selector
        config = self.config_selector.select_config(
            dataset_name=self.dataset_name,
            dataset_characteristics=characteristics,
            data=data,
            mode=self.mode
        )
        
        # Print configuration summary
        self.config_selector.print_config_summary(config)
        
        return config
    
    def _load_dataset(self):
        """Load dataset for analysis (mock implementation)."""
        
        logger.info(f"Loading dataset from {self.data_path}")
        
        # Mock dataset with realistic characteristics
        class MockData:
            def __init__(self):
                # Simulate different dataset types
                if "ellipticpp" in str(self.data_path).lower():
                    # EllipticPP characteristics
                    self.x = torch.randn(203769, 166)  # ~200K nodes
                    self.edge_index = torch.randint(0, 203769, (2, 1000000))  # 1M edges
                    self.y = torch.randint(0, 2, (203769,))
                    
                    # Heterogeneous structure
                    self.x_dict = {
                        'transaction': torch.randn(150000, 166),
                        'address': torch.randn(53769, 8)
                    }
                    self.edge_index_dict = {
                        ('transaction', 'connects', 'address'): torch.randint(0, 50000, (2, 500000)),
                        ('address', 'transacts', 'transaction'): torch.randint(0, 50000, (2, 500000))
                    }
                    
                elif "tabular" in str(self.data_path).lower():
                    # Converted tabular data
                    self.x = torch.randn(10000, 50)
                    self.edge_index = torch.randint(0, 10000, (2, 30000))
                    self.y = torch.randint(0, 2, (10000,))
                    
                else:
                    # Generic dataset
                    self.x = torch.randn(5000, 32)
                    self.edge_index = torch.randint(0, 5000, (2, 20000))
                    self.y = torch.randint(0, 2, (5000,))
        
        return MockData()
    
    def initialize_model(self):
        """Initialize hHGTN model with smart configuration."""
        
        model_config = self.config['model']
        
        logger.info("🏗️  Initializing hHGTN model...")
        
        # Build model with configuration
        # Set default node and edge types based on heterogeneous setting
        if model_config['use_hetero']:
            node_types = {
                'transaction': model_config.get('input_dim', 166),
                'address': model_config.get('input_dim', 166) // 2
            }
            edge_types = [('transaction', 'connects', 'address'), ('address', 'transacts', 'transaction')]
        else:
            node_types = {
                'node': model_config.get('input_dim', 166)
            }
            edge_types = [('node', 'connects', 'node')]
        
        self.model = hHGTN(
            input_dim=model_config.get('input_dim', 166),
            hidden_dim=model_config['hidden_dim'],
            output_dim=model_config.get('output_dim', 2),
            node_types=node_types,
            edge_types=edge_types,
            num_layers=model_config['num_layers'],
            dropout=model_config.get('dropout', 0.1),
            
            # Component toggles
            use_hypergraph=model_config['use_hypergraph'],
            use_hetero=model_config['use_hetero'],
            use_memory=model_config['use_memory'],
            use_cusp=model_config['use_cusp'],
            use_tdgnn=model_config['use_tdgnn'],
            use_gsampler=model_config['use_gsampler'],
            use_spottarget=model_config['use_spottarget'],
            use_robustness=model_config['use_robustness']
        ).to(self.device)
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"📊 Model initialized: {trainable_params:,} trainable parameters")
        
        # Log active components
        active_components = []
        for key, value in model_config.items():
            if key.startswith('use_') and value:
                component_name = key.replace('use_', '').upper()
                active_components.append(component_name)
        
        logger.info(f"🔧 Active components: {', '.join(active_components)}")
    
    def run_compatibility_test(self):
        """Run compatibility test with current configuration."""
        
        logger.info("🧪 Running compatibility test...")
        
        if self.model is None:
            self.initialize_model()
        
        try:
            # Create mock batch
            batch_size = 128
            mock_data = self._create_mock_batch(batch_size)
            
            # Test forward pass
            self.model.eval()
            with torch.no_grad():
                output = self.model(mock_data)
            
            logger.info(f"✅ Compatibility test passed! Output shape: {output.shape}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Compatibility test failed: {e}")
            return False
    
    def _create_mock_batch(self, batch_size: int):
        """Create mock data batch for testing."""
        
        model_config = self.config['model']
        input_dim = model_config.get('input_dim', 166)
        
        # Basic mock data structure
        class MockBatch:
            def __init__(self, device):
                self.device = device
                self.x = torch.randn(batch_size, input_dim, device=device)
                self.edge_index = torch.randint(0, batch_size, (2, batch_size * 3), device=device)
                self.batch = torch.zeros(batch_size, dtype=torch.long, device=device)
                
                # Add heterogeneous structure if needed
                if model_config['use_hetero']:
                    self.x_dict = {
                        'transaction': torch.randn(batch_size//2, input_dim, device=device),
                        'address': torch.randn(batch_size//2, input_dim//2, device=device)
                    }
                    self.edge_index_dict = {
                        ('transaction', 'connects', 'address'): torch.randint(
                            0, batch_size//2, (2, batch_size), device=device
                        )
                    }
                
                # Add temporal info if needed
                if model_config['use_tdgnn'] or model_config['use_memory']:
                    self.edge_time = torch.randint(1, 100, (batch_size * 3,), device=device)
        
        return MockBatch(self.device)
    
    def run_training(self, epochs: Optional[int] = None):
        """Run training with smart configuration."""
        
        if self.model is None:
            self.initialize_model()
        
        training_config = self.config['training']
        epochs = epochs or training_config['epochs']
        
        logger.info(f"🚀 Starting training for {epochs} epochs...")
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=training_config['learning_rate']
        )
        
        # Setup loss function
        loss_type = training_config.get('loss_type', 'bce')
        if loss_type == 'focal':
            criterion = self._create_focal_loss(training_config.get('focal_gamma', 2.0))
        else:
            criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            
            # Create mock batch
            batch = self._create_mock_batch(training_config['batch_size'])
            mock_labels = torch.randint(0, 2, (training_config['batch_size'],), device=self.device)
            
            # Forward pass
            optimizer.zero_grad()
            output = self.model(batch)
            loss = criterion(output, mock_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}")
        
        logger.info("✅ Training completed successfully!")
    
    def _create_focal_loss(self, gamma: float = 2.0):
        """Create focal loss for imbalanced datasets."""
        
        class FocalLoss(torch.nn.Module):
            def __init__(self, gamma=2.0):
                super().__init__()
                self.gamma = gamma
                self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
            
            def forward(self, pred, target):
                ce_loss = self.ce_loss(pred, target)
                pt = torch.exp(-ce_loss)
                focal_loss = (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
        
        return FocalLoss(gamma)
    
    def save_config(self, output_path: str):
        """Save the generated configuration."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        logger.info(f"💾 Configuration saved to {output_path}")
    
    def run_ablation_preview(self):
        """Preview what an ablation study would look like."""
        
        logger.info("🔬 Ablation Study Preview:")
        
        model_config = self.config['model']
        active_components = [key for key, value in model_config.items() 
                           if key.startswith('use_') and value]
        
        print(f"\n📋 Would test {2**len(active_components)} configurations:")
        print(f"   - Baseline (no components)")
        for component in active_components:
            comp_name = component.replace('use_', '').upper()
            print(f"   - {comp_name} ablation")
        print(f"   - All combinations ({len(active_components)} components)")
        
        # Estimate computational cost
        base_params = 64 * 64 * 2  # Basic MLP
        component_costs = {
            'use_hypergraph': 50000,
            'use_hetero': 30000,
            'use_memory': 40000,
            'use_cusp': 20000,
            'use_tdgnn': 35000,
            'use_gsampler': 10000,
            'use_spottarget': 15000,
            'use_robustness': 25000
        }
        
        total_extra_params = sum(component_costs.get(comp, 0) for comp in active_components)
        estimated_params = base_params + total_extra_params
        
        print(f"\n📊 Estimated model size: ~{estimated_params:,} parameters")


def main():
    """Main training function."""
    
    parser = argparse.ArgumentParser(description='Enhanced hHGTN Training')
    parser.add_argument('--data', type=str, help='Path to dataset')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--dataset', type=str, help='Known dataset name')
    parser.add_argument('--mode', type=str, default='auto', 
                       choices=['auto', 'conservative', 'aggressive', 'benchmark'],
                       help='Configuration mode')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--test-only', action='store_true', help='Only run compatibility test')
    parser.add_argument('--save-config', type=str, help='Save generated config to path')
    parser.add_argument('--preview-ablation', action='store_true', help='Preview ablation study')
    
    args = parser.parse_args()
    
    print("🎯 Enhanced hHGTN Training with Smart Configuration")
    print("="*60)
    
    # Initialize trainer
    trainer = EnhancedHHGTNTrainer(
        data_path=args.data,
        config_path=args.config,
        dataset_name=args.dataset,
        mode=args.mode
    )
    
    # Save configuration if requested
    if args.save_config:
        trainer.save_config(args.save_config)
    
    # Run compatibility test
    if trainer.run_compatibility_test():
        logger.info("✅ Configuration is compatible!")
        
        if args.preview_ablation:
            trainer.run_ablation_preview()
        
        if not args.test_only:
            trainer.run_training(args.epochs)
    else:
        logger.error("❌ Configuration compatibility failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
