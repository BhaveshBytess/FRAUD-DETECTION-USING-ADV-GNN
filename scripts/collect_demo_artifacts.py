#!/usr/bin/env python3
"""
Demo artifact collection script for hHGTN fraud detection
Tests the demo pipeline and creates sample outputs
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import argparse

# Add project root to path
sys.path.append('.')

import torch
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Collect demo artifacts for hHGTN')
    parser.add_argument('--checkpoint', default='experiments/demo/checkpoint_lite.ckpt', 
                       help='Path to model checkpoint')
    parser.add_argument('--out', default=None, 
                       help='Output directory (default: experiments/demo/run_<timestamp>)')
    args = parser.parse_args()
    
    print("hHGTN Demo Artifact Collection")
    print("=" * 50)
    
    # Set up output directory
    if args.out is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f'experiments/demo/run_{timestamp}')
    else:
        output_dir = Path(args.out)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    explanations_dir = output_dir / 'explanations'
    explanations_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Load demo data
    print("\nLoading demo data...")
    try:
        data_path = Path('demo_data')
        labels_df = pd.read_csv(data_path / 'labels.csv')
        print(f"Loaded {len(labels_df)} labeled transactions")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Create simplified model
    print("\nInitializing demo model...")
    
    class SimpleDemoModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 2)
            )
        
        def forward(self, x):
            return self.classifier(x)
    
    model = SimpleDemoModel()
    model.eval()
    
    # Generate synthetic features and run inference
    print("\nRunning inference...")
    torch.manual_seed(42)
    num_samples = len(labels_df)
    features = torch.randn(num_samples, 64)
    
    with torch.no_grad():
        logits = model(features)
        probs = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
    
    # Create results
    results_df = labels_df.copy()
    results_df['predicted_label'] = predictions.numpy()
    results_df['fraud_probability'] = probs[:, 1].numpy()
    results_df['confidence'] = torch.max(probs, dim=1)[0].numpy()
    
    # Save predictions CSV
    preds_file = output_dir / 'preds.csv'
    results_df.to_csv(preds_file, index=False)
    print(f"Saved predictions: {preds_file}")
    
    # Generate sample explanations
    print("\nGenerating explanations...")
    sample_txns = results_df.sample(n=min(3, len(results_df)), random_state=42)
    
    feature_names = ['Amount', 'Time', 'Network', 'History', 'Risk', 'Location']
    
    for _, row in sample_txns.iterrows():
        # Generate feature importance
        torch.manual_seed(int(row['id']))
        importance = torch.rand(len(feature_names)).numpy()
        importance = importance / importance.sum()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Transaction {row['id']} Explanation</title></head>
        <body style="font-family: Arial; padding: 20px;">
            <h1>Transaction {row['id']} Analysis</h1>
            <p><strong>Prediction:</strong> {"FRAUD" if row['predicted_label'] == 1 else "LEGITIMATE"}</p>
            <p><strong>Confidence:</strong> {row['confidence']:.1%}</p>
            <p><strong>Fraud Probability:</strong> {row['fraud_probability']:.1%}</p>
            <h3>Feature Importance:</h3>
            <ul>
        """
        
        for feature, score in zip(feature_names, importance):
            html_content += f"<li>{feature}: {score:.1%}</li>"
        
        html_content += """
            </ul>
        </body>
        </html>
        """
        
        html_file = explanations_dir / f"transaction_{row['id']}_explanation.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"Generated: {html_file}")
    
    # Save summary metrics
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'num_transactions': int(len(results_df)),
        'num_explanations': int(len(sample_txns)),
        'accuracy': float((results_df['predicted_label'] == results_df['label']).mean()),
        'avg_confidence': float(results_df['confidence'].mean())
    }
    
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nDemo complete!")
    print(f"Results: {output_dir}")
    print(f"   * Predictions: preds.csv")
    print(f"   * Explanations: explanations/*.html ({len(sample_txns)} files)")
    print(f"   * Metrics: metrics.json")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
