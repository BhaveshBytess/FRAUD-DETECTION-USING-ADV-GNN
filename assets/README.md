# Assets Directory

This directory contains portfolio-ready visual assets, documentation snapshots, and marketing materials for the fraud detection project.

## Directory Structure

```
assets/
├── README.md                 # This file
├── screenshots/              # Project screenshots and demos
│   ├── architecture_diagram.png
│   ├── performance_chart.png  
│   ├── attention_visualization.png
│   ├── colab_demo.png
│   └── confusion_matrix.png
├── logos/                    # Project branding
│   ├── project_logo.png
│   ├── hhgtn_icon.svg
│   └── favicon.ico
├── diagrams/                 # Technical architecture
│   ├── system_architecture.png
│   ├── data_flow.png
│   ├── model_architecture.png
│   └── deployment_diagram.png
├── presentations/            # Slide decks and demos
│   ├── project_overview.pdf
│   ├── technical_deep_dive.pdf
│   └── demo_slides.pdf
└── social/                   # Social media assets
    ├── linkedin_banner.png
    ├── twitter_card.png
    ├── github_banner.png
    └── portfolio_thumbnail.png
```

## Asset Usage Guidelines

### Screenshots
- **architecture_diagram.png**: Use for technical documentation and presentations
- **performance_chart.png**: Include in resume bullets and portfolio cards
- **attention_visualization.png**: Highlight model interpretability in interviews
- **colab_demo.png**: Showcase interactive capabilities
- **confusion_matrix.png**: Demonstrate evaluation rigor

### Recommended Resolutions
- Portfolio thumbnails: 400x300px
- LinkedIn banners: 1200x627px  
- GitHub repository images: 1280x640px
- Presentation slides: 1920x1080px
- Documentation diagrams: 800x600px

### File Naming Convention
- Use lowercase with underscores
- Include descriptive keywords
- Add version numbers for iterations
- Example: `hhgtn_performance_v2.png`

## Creating Assets

### Performance Visualizations
```python
# Generate performance comparison chart
import matplotlib.pyplot as plt
import numpy as np

models = ['hHGTN', 'GAT', 'GraphSAGE', 'Random Forest']
auc_scores = [0.89, 0.83, 0.78, 0.72]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, auc_scores, color=['#2E86C1', '#48C9B0', '#F4D03F', '#EC7063'])
plt.title('Model Performance Comparison - AUC Scores', fontsize=16, fontweight='bold')
plt.ylabel('AUC Score', fontsize=12)
plt.ylim(0.6, 1.0)

# Add value labels on bars
for bar, score in zip(bars, auc_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('assets/screenshots/performance_chart.png', dpi=300, bbox_inches='tight')
```

### Architecture Diagram
```python
# Create system architecture visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Define components
components = [
    {'name': 'Data Input\n(EllipticPP)', 'pos': (1, 7), 'color': '#3498DB'},
    {'name': 'Graph Construction', 'pos': (3, 7), 'color': '#E74C3C'},
    {'name': 'hHGTN Model', 'pos': (5, 7), 'color': '#2ECC71'},
    {'name': 'Attention Mechanism', 'pos': (7, 7), 'color': '#F39C12'},
    {'name': 'Fraud Prediction', 'pos': (9, 7), 'color': '#9B59B6'}
]

# Draw components
for comp in components:
    rect = patches.FancyBboxPatch(
        (comp['pos'][0]-0.4, comp['pos'][1]-0.3), 0.8, 0.6,
        boxstyle="round,pad=0.1", facecolor=comp['color'], alpha=0.7
    )
    ax.add_patch(rect)
    ax.text(comp['pos'][0], comp['pos'][1], comp['name'], 
            ha='center', va='center', fontweight='bold', fontsize=10)

# Draw connections
for i in range(len(components)-1):
    ax.arrow(components[i]['pos'][0]+0.4, components[i]['pos'][1], 
             1.2, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

ax.set_xlim(0, 10)
ax.set_ylim(6, 8)
ax.set_title('hHGTN System Architecture', fontsize=16, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('assets/diagrams/system_architecture.png', dpi=300, bbox_inches='tight')
```

## Social Media Templates

### LinkedIn Post Template
```
🚀 Excited to share my latest project: Advanced Fraud Detection using Graph Transformers!

Key achievements:
✅ 89% AUC on 203K cryptocurrency transactions
✅ 11% improvement over baseline methods  
✅ Novel heterogeneous graph attention mechanisms
✅ Production-ready deployment with Docker

Technologies: PyTorch Geometric, Graph Neural Networks, MLOps

The complete project includes interactive Colab demos, comprehensive documentation, and reproducible experiments.

#MachineLearning #FraudDetection #GraphNeuralNetworks #PyTorch #MLOps #Cryptocurrency

[Link to project] [Live demo]
```

### Twitter Thread Starter
```
Thread: Built an advanced fraud detection system using graph transformers 🧵

1/5 The challenge: Detecting fraudulent cryptocurrency transactions in complex networks with 200K+ transactions and 800K+ addresses

Used heterogeneous graph transformer networks (hHGTN) to model transaction-address relationships...
```

## Portfolio Integration

### Project Card Format
```html
<div class="project-card">
  <img src="assets/screenshots/performance_chart.png" alt="Performance Results">
  <h3>Cryptocurrency Fraud Detection</h3>
  <p>Advanced ML system using graph transformers - 89% AUC</p>
  <div class="tech-stack">
    <span>PyTorch</span>
    <span>Graph Neural Networks</span>
    <span>MLOps</span>
  </div>
  <div class="project-links">
    <a href="#">View Code</a>
    <a href="#">Live Demo</a>
    <a href="#">Report</a>
  </div>
</div>
```

## Brand Colors

- **Primary Blue**: #2E86C1 (Technical elements)
- **Success Green**: #2ECC71 (Performance highlights)  
- **Warning Orange**: #F39C12 (Attention/alerts)
- **Danger Red**: #E74C3C (Fraud indicators)
- **Purple**: #9B59B6 (Innovation/advanced features)

## Usage Rights

All assets in this directory are created for this project and may be used for:
- Portfolio websites and presentations
- Resume and cover letter materials  
- Social media promotion
- Academic and professional presentations
- GitHub repository documentation

For commercial use beyond personal portfolio purposes, please ensure proper attribution.
