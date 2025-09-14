# 🤝 Contributing to hHGTN Fraud Detection

Thank you for your interest in contributing to the hHGTN fraud detection project! This document provides guidelines for contributing to our research and development efforts.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric
- Docker (for deployment testing)

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/BhaveshBytess/FRAUD-DETECTION-USING-ADV-GNN.git
cd FRAUD-DETECTION-USING-ADV-GNN

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r demo_service/requirements.txt

# Run tests to verify setup
python -m pytest demo_service/tests/ -v
```

## 📋 How to Contribute

### 🐛 Bug Reports
When reporting bugs, please include:
- Python version and OS
- Complete error traceback
- Minimal reproducible example
- Expected vs actual behavior

### ✨ Feature Requests
For new features, please:
- Check existing issues first
- Describe the use case and motivation
- Provide implementation details if possible
- Consider backward compatibility

### 🔬 Research Contributions
We welcome research contributions including:
- Novel GNN architectures
- Improved explainability methods
- Advanced temporal modeling
- Fraud detection techniques
- Benchmarking and evaluation

## 🏗️ Development Guidelines

### Code Style
- Follow PEP 8 Python style guidelines
- Use type hints for function signatures
- Include docstrings for public functions/classes
- Keep functions focused and well-named

### Testing
- Add unit tests for new functionality
- Ensure existing tests continue to pass
- Include integration tests for major features
- Test both "lite" and "full" modes

```bash
# Run test suite
python -m pytest demo_service/tests/ -v

# Run specific test files
python -m pytest demo_service/tests/test_demo_predict.py -v
```

### Documentation
- Update README.md for user-facing changes
- Add docstrings with examples
- Update API documentation
- Include configuration examples

## 🎯 Contribution Areas

### High Priority
1. **Performance Optimization**
   - Memory usage reduction
   - Inference speed improvements
   - GPU utilization optimization

2. **Security Enhancements**
   - Additional input validation
   - Advanced rate limiting
   - Audit logging improvements

3. **Model Improvements**
   - Novel GNN architectures
   - Better explainability methods
   - Temporal modeling advances

### Medium Priority
1. **Production Features**
   - Kubernetes deployment
   - Monitoring and alerting
   - A/B testing framework

2. **Research Extensions**
   - Self-supervised learning
   - Domain adaptation
   - Multi-dataset evaluation

3. **Documentation**
   - Tutorial notebooks
   - Video demonstrations
   - Academic paper preparation

## 📝 Pull Request Process

### Before Submitting
1. **Create an Issue**: Discuss major changes first
2. **Fork the Repository**: Work on your own fork
3. **Create Feature Branch**: `git checkout -b feature/your-feature-name`
4. **Test Thoroughly**: Ensure all tests pass
5. **Document Changes**: Update relevant documentation

### PR Requirements
- [ ] All tests pass (`pytest demo_service/tests/`)
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] PR description explains changes

### Review Process
1. Automated tests run on PR creation
2. Code review by maintainers
3. Feedback incorporation
4. Final approval and merge

## 🧪 Testing Framework

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **Performance Tests**: Load and scalability testing
- **Security Tests**: Input validation and security testing

### Running Tests
```bash
# All tests
python -m pytest demo_service/tests/ -v

# Specific categories
python -m pytest demo_service/tests/test_security.py -v
python -m pytest demo_service/tests/test_performance.py -v

# With coverage
python -m pytest --cov=demo_service demo_service/tests/
```

## 🏷️ Release Process

### Versioning
We use semantic versioning (SemVer):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version bumped
- [ ] Changelog updated
- [ ] Docker images built
- [ ] GitHub release created

## 📞 Getting Help

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Documentation**: Complete guides in `docs/` directory

### Maintainer Response Time
- **Bug Reports**: 1-2 business days
- **Feature Requests**: 3-5 business days
- **Pull Reviews**: 2-3 business days

## 🎓 Research Collaboration

### Academic Contributions
We welcome academic collaborations including:
- Novel algorithm implementations
- Comprehensive evaluations
- Theoretical analysis
- Benchmark contributions

### Citation
If you use this work in academic research, please cite:
```bibtex
@software{hhgtn_fraud_detection,
  title={hHGTN: Heterogeneous Hypergraph Transformer Networks for Fraud Detection},
  author={Your Name and Contributors},
  year={2025},
  url={https://github.com/BhaveshBytess/FRAUD-DETECTION-USING-ADV-GNN}
}
```

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors are recognized in:
- GitHub contributors list
- README acknowledgments
- Release notes
- Academic publications (where applicable)

Thank you for helping make fraud detection more effective and accessible! 🚀
