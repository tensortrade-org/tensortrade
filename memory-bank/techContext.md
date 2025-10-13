# TensorTrade Technical Context

## Technology Stack

### Core Dependencies (Updated January 2025)
- **Python**: >=3.11.9 (strict requirement for modern features)
- **NumPy**: >=1.26.4,<2.0 (TensorFlow compatibility)
- **Pandas**: >=2.2.3 (modern data manipulation)
- **Gymnasium**: >=0.28.1 (RL environment interface)
- **TensorFlow**: >=2.15.1 (security and compatibility improvements)
- **PyYAML**: >=5.1.2 (configuration files)
- **Stochastic**: >=0.6.0 (financial stochastic processes)
- **Ray**: 2.37.0 (distributed computing - major upgrade from 1.9.2)

### Visualization & UI
- **Matplotlib**: >=3.1.1 (plotting)
- **Plotly**: >=4.5.0 (interactive charts)
- **IPython**: >=7.12.0 (notebook support)
- **IPyWidgets**: >=7.0.0 (interactive widgets for rendering)

### Development & Testing
- **Pytest**: >=5.1.1 (testing framework)
- **TA**: >=0.4.7 (technical analysis indicators)
- **Deprecated**: >=1.2.13 (deprecation warnings)

### Documentation
- **Sphinx**: Documentation generation
- **Sphinx RTD Theme**: Read the Docs theme
- **NBSphinx**: Jupyter notebook integration
- **Recommonmark**: Markdown support

## Architecture Components

### 1. Core System (`tensortrade/core/`)
- **Base Classes**: `Identifiable`, `TimeIndexed`, `Observable`
- **Component System**: `Component` with context injection
- **Context Management**: `TradingContext` for configuration
- **Clock System**: Global time management
- **Registry**: Component discovery and registration

### 2. Order Management System (`tensortrade/oms/`)
- **Exchanges**: Trading venue simulation
- **Instruments**: Currency and asset definitions
- **Orders**: Order types and execution
- **Wallets**: Balance and position management
- **Services**: Execution, slippage, risk management

### 3. Environment System (`tensortrade/env/`)
- **Generic Environment**: Base trading environment
- **Default Components**: Pre-built action schemes, observers, rewards
- **Component Interfaces**: Standardized component contracts

### 4. Data Pipeline (`tensortrade/feed/`)
- **Stream Processing**: Functional data pipeline
- **DataFeed**: Stream orchestration
- **API Layer**: Stream operations and transformations
- **Core**: Base stream classes and utilities

### 5. Agent Framework (`tensortrade/agents/`)
- **Base Agent**: Abstract agent interface
- **Built-in Agents**: DQN, A2C implementations (deprecated)
- **Parallel Agents**: Multi-process agent support
- **Replay Memory**: Experience replay for DQN

### 6. Stochastic Processes (`tensortrade/stochastic/`)
- **Financial Models**: GBM, Heston, Merton, etc.
- **Utilities**: Parameter estimation and helpers
- **Process Integration**: Integration with data pipeline

## Development Environment

### Build System
- **Setuptools**: Package building and distribution
- **Makefile**: Common development tasks
- **Docker**: Containerized development environment

### Testing Infrastructure (Enhanced January 2025)
- **Unit Tests**: Comprehensive test coverage (19 test cases across 4 files)
- **Integration Tests**: End-to-end testing with workflow validation
- **Test Data**: Sample datasets and configurations
- **Mock Objects**: Testing utilities
- **Issue-Specific Tests**: Dedicated tests for each resolved GitHub issue
- **Cross-Platform Tests**: Platform-specific testing considerations

### Documentation System (Enhanced January 2025)
- **Sphinx**: Automated documentation generation
- **API Documentation**: Auto-generated from docstrings
- **Tutorials**: Jupyter notebook examples
- **Examples**: Working code samples
- **Setup Guide**: Comprehensive installation and troubleshooting guide
- **Migration Guide**: Step-by-step migration instructions
- **Compatibility Matrix**: Tested version combinations
- **Testing Report**: Comprehensive testing results and validation

## Configuration Management

### Configuration Formats
- **JSON**: Primary configuration format
- **YAML**: Alternative configuration format
- **Python**: Programmatic configuration

### Configuration Structure
```json
{
  "base_instrument": "USD",
  "instruments": ["BTC", "ETH"],
  "actions": {
    "n_actions": 24,
    "action_type": "discrete"
  },
  "exchanges": {
    "name": "bitfinex",
    "credentials": {...}
  }
}
```

## Data Handling

### Data Sources
- **Historical Data**: CSV, API feeds
- **Real-time Data**: WebSocket connections
- **Synthetic Data**: Stochastic process generation
- **External APIs**: Crypto data download utilities

### Data Processing
- **Stream Processing**: Real-time data transformation
- **Technical Analysis**: Built-in indicators
- **Feature Engineering**: Custom feature creation
- **Data Validation**: Input validation and cleaning

## Performance Considerations

### Memory Management
- **Stream Processing**: Lazy evaluation to minimize memory usage
- **Data Windows**: Configurable observation windows
- **Garbage Collection**: Proper cleanup of resources

### Computational Efficiency
- **Vectorized Operations**: NumPy-based computations
- **Batch Processing**: Efficient batch operations
- **Caching**: Strategic caching of expensive operations

### Scalability
- **Parallel Processing**: Multi-process agent support
- **Distributed Computing**: Ray integration for scaling
- **Modular Design**: Easy to scale individual components

## Security & Risk Management

### Financial Safety
- **Precision Handling**: Decimal arithmetic for financial calculations
- **Balance Validation**: Insufficient funds protection
- **Order Validation**: Order size and price validation
- **Risk Limits**: Configurable risk management

### Code Safety
- **Type Hints**: Comprehensive type annotations
- **Input Validation**: Parameter validation
- **Error Handling**: Graceful error handling
- **Deprecation Warnings**: Clear migration paths

## Integration Points

### External Libraries (Updated January 2025)
- **Ray**: 2.37.0 (distributed computing framework - major API upgrade)
- **Stable-Baselines3**: RL algorithm implementations
- **TensorFlow**: 2.15.1+ (deep learning backend with security improvements)
- **Gymnasium**: >=0.28.1 (RL environment standard)
- **pandas_ta**: Technical analysis library (replaces problematic ta library)
- **quantstats**: Performance analytics (with fixed treynor_ratio usage)

### APIs & Services
- **Exchange APIs**: Live trading integration
- **Data Providers**: Market data feeds
- **Cloud Services**: Deployment platforms
- **Monitoring**: Performance monitoring tools

## Deployment Considerations

### Production Readiness (Enhanced January 2025)
- **Beta Status**: Framework marked as beta for production use
- **Error Handling**: Comprehensive error handling with critical bug fixes
- **Logging**: Configurable logging system
- **Monitoring**: Performance and error monitoring
- **GPU Compatibility**: Proper device management and tensor handling
- **Stream Robustness**: Fixed stream selector for multiple naming conventions
- **Dependency Stability**: Modern, compatible dependency versions
- **Migration Support**: Complete migration path for existing users

### Deployment Options
- **Local Development**: Direct Python installation
- **Docker**: Containerized deployment
- **Cloud Platforms**: AWS, GCP, Azure support
- **Jupyter**: Interactive development environment
