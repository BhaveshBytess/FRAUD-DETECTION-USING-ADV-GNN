# Git Branch Management Strategy

## 🌳 Current Branch Structure

### Main Branch
- **Branch**: `main` 
- **Status**: ✅ Stage 6 Complete (TDGNN + G-SAMPLER)
- **Tag**: `v6.0.0`
- **Description**: Production-ready code with complete Stage 6 implementation

### Feature Branches

#### Stage 6 (Retroactive Documentation)
- **Branch**: `feature/stage-6-tdgnn-gsampler`
- **Status**: ✅ Complete and Merged to Main
- **Purpose**: Contains all Stage 6 TDGNN + G-SAMPLER work
- **Key Features**:
  - Timestamped Directed Graph Neural Networks
  - G-SAMPLER framework with GPU/CPU hybrid
  - Temporal neighbor sampling with binary search
  - Integration with Stage 5 hypergraph models

#### Stage 7 (Active Development)
- **Branch**: `feature/stage-7-ensemble-methods`
- **Status**: 🚧 Ready for Development
- **Purpose**: Next stage development - Ensemble Methods & Model Fusion
- **Planned Features**:
  - Advanced ensemble methods combining TDGNN with transformers
  - Model fusion techniques (temporal-spatial-structural)
  - Adaptive learning with dynamic architecture selection
  - Cross-temporal validation frameworks

---

## 🔄 Branching Workflow

### Development Process
1. **Feature Development**: Work on `feature/stage-X-description` branches
2. **Testing & Validation**: Complete experimental validation on feature branch
3. **Documentation**: Add comprehensive docs and release notes
4. **Merge to Main**: After complete validation and testing
5. **Tagging**: Create version tags for major releases

### Branch Naming Convention
```
feature/stage-X-description
├── feature/stage-7-ensemble-methods
├── feature/stage-8-self-supervised
├── feature/stage-9-production-optimization
└── ...
```

### Current Repository State
```
Repository: FRAUD-DETECTION-USING-ADV-GNN
├── main (v6.0.0) - Stage 6 Complete
├── feature/stage-6-tdgnn-gsampler - Stage 6 work (merged)
└── feature/stage-7-ensemble-methods - Ready for Stage 7
```

---

## 📋 Branch Management Commands

### Switch to Stage 7 Development
```bash
git checkout feature/stage-7-ensemble-methods
```

### Return to Main (Stable)
```bash
git checkout main
```

### Create New Feature Branch
```bash
git checkout main
git checkout -b feature/stage-X-description
git push -u origin feature/stage-X-description
```

### Merge Feature to Main (After Completion)
```bash
git checkout main
git merge feature/stage-X-description
git tag -a vX.0.0 -m "Stage X Release: Description"
git push origin main
git push origin vX.0.0
```

---

## 🎯 Current Development Status

### ✅ Completed Stages
- **Stage 0-6**: All complete and on main branch
- **Current Release**: v6.0.0 (TDGNN + G-SAMPLER)

### 🚧 Active Development
- **Stage 7**: Ready on `feature/stage-7-ensemble-methods`
- **Focus**: Ensemble Methods & Model Fusion

### 📋 Future Stages
- **Stage 8**: Self-supervised Learning & Advanced Training
- **Stage 9**: Production Optimization & Deployment
- **Stage 10+**: Real-time Systems & Monitoring

---

## 🔗 GitHub Repository Links

### Main Repository
- **URL**: https://github.com/BhaveshBytess/FRAUD-DETECTION-USING-ADV-GNN
- **Main Branch**: https://github.com/BhaveshBytess/FRAUD-DETECTION-USING-ADV-GNN/tree/main
- **Stage 6 Branch**: https://github.com/BhaveshBytess/FRAUD-DETECTION-USING-ADV-GNN/tree/feature/stage-6-tdgnn-gsampler
- **Stage 7 Branch**: https://github.com/BhaveshBytess/FRAUD-DETECTION-USING-ADV-GNN/tree/feature/stage-7-ensemble-methods

### Pull Request Templates
When Stage 7 is complete, create PR from `feature/stage-7-ensemble-methods` → `main`

---

**✅ Branch Management Setup Complete!**
- Main branch reflects current stable state (Stage 6 complete)
- Stage 6 work properly documented in feature branch
- Stage 7 development branch ready for future work
