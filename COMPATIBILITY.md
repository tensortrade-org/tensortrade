# TensorTrade Compatibility Matrix

This document provides tested version combinations for TensorTrade and its dependencies to ensure a working environment.

## Tested Version Combinations

### Recommended Configuration (Latest)
- **Python**: 3.11.9+
- **TensorTrade**: 1.0.4-dev1
- **Ray**: 2.37.0
- **TensorFlow**: 2.15.1
- **NumPy**: 1.26.4
- **Pandas**: 2.2.3
- **Gymnasium**: 0.28.1
- **PyYAML**: 6.0.2

### Alternative Configurations

#### For CUDA Support
- **Python**: 3.11.9+
- **TensorTrade**: 1.0.4-dev1
- **Ray**: 2.37.0
- **TensorFlow**: 2.15.1 (with CUDA 12.0+)
- **NumPy**: 1.26.4
- **Pandas**: 2.2.3
- **Gymnasium**: 0.28.1

#### For Google Colab
- **Python**: 3.11.9+ (Colab default)
- **TensorTrade**: 1.0.4-dev1
- **Ray**: 2.37.0
- **TensorFlow**: 2.15.1
- **NumPy**: 1.26.4
- **Pandas**: 2.2.3
- **Gymnasium**: 0.28.1

## Platform Support

### Windows 10/11
- ✅ **Supported**: Python 3.11.9+, all dependencies
- ⚠️ **Note**: May require Visual Studio Build Tools for some packages

### Linux (Ubuntu 20.04+)
- ✅ **Supported**: Python 3.11.9+, all dependencies
- ✅ **Recommended**: For production deployments

### macOS (12.0+)
- ✅ **Supported**: Python 3.11.9+, all dependencies
- ⚠️ **Note**: May require Xcode Command Line Tools

### Google Colab
- ✅ **Supported**: With specific setup instructions
- ⚠️ **Note**: Some features may be limited due to Colab restrictions

## Breaking Changes

### Ray 2.x Migration
- **Old API**: `tune.run()` → **New API**: `tune.Tuner()`
- **Checkpoint handling**: Updated checkpoint save/load methods
- **RLlib configuration**: Some configuration parameters changed

### TensorFlow 2.15+
- **NumPy compatibility**: Requires NumPy < 2.0
- **CUDA support**: Updated CUDA requirements

## Troubleshooting

### Common Issues

#### 1. Ray Installation Issues
```bash
# If Ray installation fails, try:
pip install --upgrade pip
pip install ray[default,tune,rllib,serve]==2.37.0
```

#### 2. TensorFlow CUDA Issues
```bash
# For CUDA support:
pip install tensorflow[and-cuda]==2.15.1
```

#### 3. NumPy Compatibility
```bash
# Ensure NumPy < 2.0 for TensorFlow compatibility:
pip install "numpy>=1.26.4,<2.0"
```

#### 4. Importlib-metadata Issues
```bash
# If you encounter EntryPoints errors:
pip install "importlib-metadata>=4.13.0,<6.0"
```

### Environment Setup

#### Quick Setup
```bash
# Create virtual environment
python -m venv tensortrade-env
source tensortrade-env/bin/activate  # On Windows: tensortrade-env\Scripts\activate

# Install TensorTrade
pip install --upgrade pip
pip install -r requirements.txt
pip install -r examples/requirements.txt
pip install -e .
```

#### Google Colab Setup
```python
# Run in Colab cell:
!pip install --upgrade pip
!pip install -r requirements.txt
!pip install -r examples/requirements.txt
!pip install -e .
```

## Version History

### v1.0.4-dev1 (Current)
- Updated Ray to 2.37.0
- Updated TensorFlow to 2.15.1
- Updated NumPy to 1.26.4
- Updated Pandas to 2.2.3
- Fixed stream selector issues
- Fixed GPU compatibility

### v1.0.3 (Previous)
- Ray 1.9.2
- TensorFlow 2.7.0
- NumPy 1.17.0
- Pandas 0.25.0

## Support

If you encounter compatibility issues:

1. Check this compatibility matrix first
2. Verify your Python version (3.11.9+ required)
3. Try the recommended configuration
4. Check the troubleshooting section
5. Open an issue on GitHub with your environment details

## Contributing

When adding new dependencies or updating versions:

1. Test on multiple platforms
2. Update this compatibility matrix
3. Add test cases for new versions
4. Document any breaking changes
