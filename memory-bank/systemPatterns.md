# TensorTrade System Patterns

## Core Architecture Patterns

### 1. Component-Based Architecture
- **Base Class**: `Component` - All major system components inherit from this
- **Context System**: `TradingContext` - Dependency injection and configuration management
- **Registry Pattern**: Automatic registration of components for discovery
- **Metaclass Magic**: `InitContextMeta` handles context injection during instantiation

### 2. Stream Processing Pattern
- **Functional Data Pipeline**: Data flows through streams with functional transformations
- **Lazy Evaluation**: Streams compute values on-demand
- **Composition**: Complex data flows built by composing simple stream operations
- **Namespace Management**: Hierarchical naming for stream organization

### 3. Observer Pattern
- **Observable Base**: Core objects can be observed for state changes
- **Event-Driven**: Components communicate through events and listeners
- **Decoupled Design**: Loose coupling between components through event system
- **Stream Selector**: Enhanced to handle multiple naming conventions (Issue #470 fix)

### 4. Strategy Pattern
- **Action Schemes**: Pluggable strategies for agent actions
- **Reward Schemes**: Configurable reward calculation methods
- **Execution Services**: Swappable order execution strategies

## Data Flow Patterns

### 1. Feed-Stream Architecture
```
Data Sources → Streams → DataFeed → Observer → Agent
     ↓           ↓         ↓         ↓        ↓
  External    Transform  Compile  Observe  Action
   Data       Pipeline   Order   State    Decision
```

### 2. Order Execution Flow
```
Agent Action → Action Scheme → Order Creation → Broker → Exchange → Execution Service → Trade → Portfolio Update
```

### 3. Observation Pipeline
```
Market Data → Stream Processing → Feature Engineering → Observation Window → Agent State
```

## Design Patterns

### 1. Factory Pattern
- **Component Creation**: Components created through registry and context system
- **Configuration-Driven**: Components instantiated based on configuration files
- **Polymorphic Creation**: Same interface for different component types

### 2. Builder Pattern
- **Environment Construction**: Complex environments built step-by-step
- **Fluent Interface**: Method chaining for configuration
- **Validation**: Built-in validation during construction

### 3. Template Method Pattern
- **Agent Interface**: Abstract methods for training, action selection, etc.
- **Environment Steps**: Standardized step/reset cycle
- **Component Lifecycle**: Consistent initialization and cleanup

### 4. Decorator Pattern
- **Stream Operations**: Functional transformations as decorators
- **Order Criteria**: Composable order execution conditions
- **Reward Modifiers**: Stackable reward calculation components

## Integration Patterns

### 1. Adapter Pattern
- **External Libraries**: Adapters for different RL frameworks
- **Data Sources**: Adapters for various market data providers
- **Exchange APIs**: Adapters for different exchange interfaces

### 2. Facade Pattern
- **Environment Interface**: Simplified interface to complex trading system
- **Portfolio API**: High-level portfolio management interface
- **Configuration Management**: Simple configuration interface

### 3. Proxy Pattern
- **Live Trading**: Proxy for live exchange connections
- **Data Caching**: Proxy for expensive data operations
- **Risk Management**: Proxy for order validation
- **Device Management**: GPU/CPU device handling with automatic tensor conversion (Issue #382 fix)

## Error Handling Patterns

### 1. Exception Hierarchy
- **Domain-Specific Exceptions**: `InsufficientFunds`, `InvalidOrderQuantity`
- **Graceful Degradation**: System continues with warnings for non-critical errors
- **Validation**: Input validation at component boundaries

### 2. Circuit Breaker Pattern
- **Exchange Failures**: Automatic fallback for exchange connectivity issues
- **Data Source Failures**: Graceful handling of data feed interruptions
- **Agent Failures**: Recovery mechanisms for agent errors

## Performance Patterns

### 1. Lazy Loading
- **Stream Evaluation**: Values computed only when needed
- **Component Initialization**: Components created on-demand
- **Data Loading**: Historical data loaded incrementally

### 2. Caching
- **Price Data**: Cached market data for performance
- **Computed Features**: Cached technical indicators
- **Portfolio State**: Cached portfolio calculations

### 3. Batch Processing
- **Order Execution**: Batch processing of multiple orders
- **Data Updates**: Batch updates to portfolio state
- **Feature Computation**: Batch computation of technical indicators

## Testing Patterns

### 1. Mock Objects
- **Exchange Mocks**: Simulated exchanges for testing
- **Data Mocks**: Synthetic data for reproducible tests
- **Agent Mocks**: Simple agents for environment testing

### 2. Test Fixtures
- **Environment Setup**: Standardized test environment creation
- **Data Fixtures**: Predefined test datasets
- **Component Fixtures**: Reusable component configurations

### 3. Property-Based Testing
- **Portfolio Invariants**: Properties that must always hold
- **Order Validation**: Properties of valid orders
- **Stream Properties**: Mathematical properties of stream operations

## Recent Pattern Enhancements (January 2025)

### 1. Stream Selector Robustness
- **Multi-Convention Support**: Handles `:symbol`, `-symbol`, and plain `symbol` naming
- **Error Prevention**: Eliminates "No stream satisfies selector condition" errors
- **Backward Compatibility**: Maintains compatibility with existing naming schemes

### 2. Device Management Pattern
- **Automatic Tensor Conversion**: Ensures observations are numpy arrays for GPU compatibility
- **Device Configuration**: Explicit device parameter for environment creation
- **Cross-Platform Support**: Works with both CPU and GPU configurations

### 3. Dependency Management Pattern
- **Version Constraints**: Strict version constraints to prevent compatibility issues
- **Modern Dependencies**: Updated to latest stable versions (Ray 2.37.0, TensorFlow 2.15.1)
- **Compatibility Matrix**: Documented tested version combinations

### 4. Testing Pattern Enhancements
- **Comprehensive Coverage**: 19 test cases across 4 test files
- **Issue-Specific Tests**: Dedicated tests for each resolved GitHub issue
- **Integration Testing**: End-to-end workflow validation
- **Cross-Platform Validation**: Platform-specific testing considerations
