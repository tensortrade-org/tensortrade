# TensorTrade Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.4] - 2025-02-04

### Breaking Changes

- **Ray 2.x Migration**: Updated from Ray 1.9.2 to Ray 2.37.0
  - `tune.run()` API replaced with `tune.Tuner()` API
  - Checkpoint handling updated
  - RLlib configuration format changed
  - See [Migration Guide](MIGRATION_GUIDE.md) for details

- **Python Version**: Now requires Python >= 3.11.9 (previously 3.7+)

### Added

- **GPU Compatibility**: Added device management utilities to ensure observations are returned as numpy arrays
  - New `device` parameter in environment creation
  - `_ensure_numpy()` method to handle tensor conversions
  - Fixes Issue #382 (GPU device placement errors)

- **Comprehensive Test Suite**:
  - Unit tests for stream selector functionality (`tests/tensortrade/unit/env/default/test_stream_selector.py`)
  - Unit tests for exchange stream naming (`tests/tensortrade/unit/oms/exchanges/test_exchange_streams.py`)
  - Unit tests for GPU compatibility (`tests/tensortrade/unit/env/test_gpu_compatibility.py`)
  - Integration tests for end-to-end workflows (`tests/tensortrade/integration/test_end_to_end.py`)

- **Documentation**:
  - New comprehensive [Environment Setup Guide](docs/ENVIRONMENT_SETUP.md)
  - New [Compatibility Matrix](COMPATIBILITY.md) with tested version combinations
  - New [Examples README](examples/README.md) with detailed example descriptions
  - Updated main README with quick start and troubleshooting sections

- **Example Notebooks**:
  - Added Ray 2.x API examples in `use_lstm_rllib.ipynb`
  - Added fixed `generate_features_fixed()` function in `train_and_evaluate.ipynb`
  - Added fixed `generate_all_default_quantstats_features_fixed()` function

### Changed

- **Dependencies Updated**:
  - Ray: 1.9.2 → 2.37.0
  - TensorFlow: >=2.7.0 → >=2.15.1
  - NumPy: >=1.17.0 → >=1.26.4,<2.0
  - Pandas: >=0.25.0 → >=2.2.3
  - Gymnasium: Already at >=0.28.1 (no change)

- **Stream Selector Logic** (`tensortrade/env/default/observers.py`):
  - Fixed to handle multiple naming conventions (`:symbol`, `-symbol`, `symbol`)
  - Resolves Issue #470 ("No stream satisfies selector condition")

- **Exchange Stream Naming** (`tensortrade/oms/exchanges/exchange.py`):
  - Standardized on `exchange_name:/base-quote` format
  - Added comments for clarity
  - Improved consistency across codebase

### Fixed

- **Issue #470**: Stream selector error ("No stream satisfies selector condition")
  - Updated observer to handle multiple stream naming conventions
  - Added comprehensive tests

- **Issue #477**: Environment setup difficulties
  - Created detailed setup guide
  - Documented working environment configurations
  - Added troubleshooting section

- **Issue #459**: Gym/Gymnasium compatibility
  - Already using Gymnasium >=0.28.1
  - Updated examples to use Ray 2.x which is compatible

- **Issue #466, #443**: Technical analysis library issues
  - Fixed `df.ta.study()` usage in examples
  - Fixed `treynor_ratio()` benchmark parameter issue
  - Provided fixed functions in notebooks

- **Issue #382**: GPU compatibility
  - Added device management utilities
  - Ensured observations are numpy arrays
  - Added configuration option for device placement

- **Issue #452**: importlib-metadata compatibility
  - Updated dependencies to compatible versions
  - Added version constraints

### Security

- Updated all dependencies to latest secure versions
- TensorFlow 2.15.1 includes security patches
- NumPy < 2.0 for TensorFlow compatibility

### Deprecated

- Old Ray 1.x API (`tune.run()`) - Use Ray 2.x API (`tune.Tuner()`)
- Python < 3.11.9 - Update to Python 3.11.9 or higher

### Removed

- None

## [1.0.3] - Previous Release

### Added

- Initial stable release
- Core trading environment
- OMS (Order Management System)
- Data feed system
- Agent framework
- Example notebooks

### Known Issues

- Ray 1.9.2 compatibility issues with newer Python versions
- Stream selector errors with certain naming conventions
- GPU device placement errors
- Technical analysis library compatibility issues

## Migration Notes

### From 1.0.3 to 1.0.4

1. **Update Python**: Upgrade to Python 3.11.9 or higher
2. **Update Dependencies**: Run `pip install -r requirements.txt --upgrade`
3. **Update Ray Code**: Replace `tune.run()` with `tune.Tuner()` (see Migration Guide)
4. **Test Your Code**: Run tests to ensure compatibility
5. **Update Examples**: Use new example notebooks as reference

See [Migration Guide](MIGRATION_GUIDE.md) for detailed instructions.

## Testing

All changes have been tested with:
- Python 3.11.9+
- Windows 10/11
- Ray 2.37.0
- TensorFlow 2.15.1
- NumPy 1.26.4
- Pandas 2.2.3

## Contributors

- TensorTrade Team
- Community Contributors

## Links

- [GitHub Repository](https://github.com/tensortrade-org/tensortrade)
- [Documentation](https://www.tensortrade.org/)
- [Discord](https://discord.gg/ZZ7BGWh)
- [Issues](https://github.com/tensortrade-org/tensortrade/issues)

