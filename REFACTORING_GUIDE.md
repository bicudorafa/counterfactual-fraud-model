# Counterfactual Fraud Model - Refactoring Guide

## Overview

This document describes the comprehensive refactoring of the counterfactual fraud model codebase to address modularity, coupling, and maintainability issues.

## Problems with the Original Architecture

### 1. **Tight Coupling**
- `SyntheticRetrainingPipeline` directly instantiated `SyntheticOffPolicyEvaluationPipeline`
- Components created their own dependencies instead of receiving them
- Hard-coded parameter passing throughout the codebase

### 2. **Parameter Explosion**
- Classes had massive parameter lists (20+ parameters in constructors)
- Same parameters repeated across multiple classes
- No clear separation between configuration and runtime parameters

### 3. **Code Duplication**
- Model creation logic duplicated across multiple classes
- Validation logic repeated everywhere
- Similar parameter handling patterns throughout

### 4. **Mixed Responsibilities** 
- `SyntheticDataGenerator` did both data generation AND model training
- No clear separation of concerns
- Monolithic classes doing too many things

### 5. **Lack of Abstractions**
- No interfaces/protocols defined
- Hard to test components in isolation
- Difficult to extend with new functionality

## New Architecture

### 1. **Pydantic Configuration Models**
```python
from counterfactual_fraud_model import SyntheticDataConfig, ModelConfig

# Type-safe, validated configuration
data_config = SyntheticDataConfig(
    n_samples=10000,
    n_features=20,
    n_informative=10,
    random_state=42
)

model_config = ModelConfig(
    model_type=ModelType.LIGHTGBM,
    model_params={"n_estimators": 50},
    random_state=42
)
```

### 2. **Protocol-Based Interfaces**
```python
from counterfactual_fraud_model import SyntheticDataGeneratorProtocol

# Clear contracts for dependency injection
class CustomDataGenerator(SyntheticDataGeneratorProtocol):
    def generate_data(self) -> pd.DataFrame: ...
    def get_model_performance(self) -> Dict[str, float]: ...
```

### 3. **Dependency Injection**
```python
# Components receive dependencies instead of creating them
pipeline = SyntheticOffPolicyEvaluationPipeline(
    config=config,
    synthetic_data_generator=custom_generator,  # Injected dependency
    logging_policy_generator=custom_policy      # Injected dependency
)
```

### 4. **Factory Pattern**
```python
from counterfactual_fraud_model import ModelFactory, ModelConfig

# Centralized model creation
factory = ModelFactory()
model = factory.create_model(model_config)
```

### 5. **Single Responsibility**
- `SyntheticDataGenerator`: Only generates data
- `ModelTrainer`: Only trains models and calculates performance
- `LoggingPolicyGenerator`: Only applies policies
- `CounterfactualEstimator`: Only estimates metrics

### 6. **Composition over Inheritance**
```python
# Old: SyntheticRetrainingPipeline inherited from SyntheticOffPolicyEvaluationPipeline
# New: SyntheticRetrainingPipeline composes a base pipeline
class SyntheticRetrainingPipeline:
    def __init__(self, config, base_pipeline=None):
        self.base_pipeline = base_pipeline or SyntheticOffPolicyEvaluationPipeline(config.base_config)
```

## Directory Structure

```
src/counterfactual_fraud_model/
├── __init__.py                 # Main exports (new + legacy)
├── config.py                   # Pydantic configuration models  
├── protocols.py                # Protocol definitions for DI
├── core/                       # Core components
│   ├── __init__.py
│   └── model_trainer.py        # Model training logic
├── factories/                  # Factory components
│   ├── __init__.py
│   └── model_factory.py        # Centralized model creation
├── generators/                 # Data and policy generators
│   ├── __init__.py
│   ├── data_generator.py       # Basic data generation
│   ├── synthetic_data_generator.py  # Synthetic data with features
│   └── logging_policy_generator.py  # Policy application
├── estimators/                 # Counterfactual estimation
│   ├── __init__.py
│   └── counterfactual_estimator.py  # Refactored estimator
├── pipelines/                  # Orchestration components
│   ├── __init__.py
│   ├── off_policy_evaluation_pipeline.py
│   ├── synthetic_off_policy_evaluation_pipeline.py
│   └── synthetic_retraining_pipeline.py
└── legacy/                     # Original files (for compatibility)
    ├── data_generator.py       # Original DataGenerator
    ├── synthetic_data_generator.py
    ├── logging_policy.py
    ├── counterfactual_estimator.py
    ├── pipeline.py
    ├── synthetic_pipeline.py
    └── synthetic_retraining_pipeline.py
```

## Migration Guide

### Before (Old Architecture)
```python
# Massive parameter lists, tight coupling
pipeline = SyntheticRetrainingPipeline(
    n_samples=100_000,
    n_features=30,
    n_informative=15,
    n_redundant=5,
    n_repeated=0,
    n_clusters_per_class=2,
    weights=[0.985, 0.015],
    flip_y=0.01,
    class_sep=1.0,
    model_type="lightgbm",
    model_params={},
    test_size=0.3,
    n_bootstrap=5000,
    retrain_test_size=0.3,
    classification_threshold=0.05,
    retrain_model_type="lightgbm",
    retrain_model_params={},
    random_state=42
)

results = pipeline.run_pipeline(cutoff=0.05, exploration_rate=0.05)
```

### After (New Architecture)
```python
# Configuration-based, loosely coupled
from counterfactual_fraud_model import (
    SyntheticRetrainingConfig,
    SyntheticOffPolicyEvaluationConfig,
    SyntheticDataConfig,
    ModelConfig,
    RetrainingConfig,
    ModelType,
    SyntheticRetrainingPipeline
)

# 1. Create configurations
config = SyntheticRetrainingConfig(
    base_config=SyntheticOffPolicyEvaluationConfig(
        synthetic_data=SyntheticDataConfig(
            n_samples=100_000,
            n_features=30,
            n_informative=15,
            random_state=42
        ),
        model=ModelConfig(
            model_type=ModelType.LIGHTGBM,
            random_state=42
        )
    ),
    retraining=RetrainingConfig(
        classification_threshold=0.05,
        retrain_model=ModelConfig(
            model_type=ModelType.LIGHTGBM,
            random_state=42
        )
    )
)

# 2. Create pipeline (dependencies injected automatically)
pipeline = SyntheticRetrainingPipeline(config)

# 3. Run with optional overrides
results = pipeline.run_pipeline(cutoff=0.05, exploration_rate=0.05)
```

## Benefits of the New Architecture

### 1. **Type Safety & Validation**
- Pydantic models provide automatic validation
- Clear error messages when configuration is invalid
- IDE support with autocomplete and type checking

### 2. **Testability**
```python
# Easy to mock dependencies for testing
mock_generator = Mock(spec=SyntheticDataGeneratorProtocol)
pipeline = SyntheticOffPolicyEvaluationPipeline(
    config=config,
    synthetic_data_generator=mock_generator
)
```

### 3. **Extensibility**
```python
# Easy to add new model types
class XGBoostFactory(ModelFactoryProtocol):
    def create_model(self, config: ModelConfig):
        import xgboost as xgb
        return xgb.XGBClassifier(**config.model_params)

# Use custom factory
trainer = ModelTrainer(XGBoostFactory())
```

### 4. **Configuration Management**
```python
# Save/load configurations
config = SyntheticOffPolicyEvaluationConfig(...)
config_dict = config.model_dump()
# Save to file, load later
loaded_config = SyntheticOffPolicyEvaluationConfig(**config_dict)
```

### 5. **Clear Separation of Concerns**
- Each class has a single, well-defined responsibility
- Easy to understand and modify individual components
- Reduced cognitive load when working on specific features

## Backward Compatibility

The refactored codebase maintains backward compatibility:

```python
# Old imports still work
from counterfactual_fraud_model import (
    CounterfactualValuesEstimator,  # Original estimator
    LegacySyntheticRetrainingPipeline  # Original pipeline
)

# New imports for refactored components
from counterfactual_fraud_model import (
    CounterfactualEstimator,  # Refactored estimator
    SyntheticRetrainingPipeline  # Refactored pipeline
)
```

## Usage Examples

See `refactoring_example.py` for comprehensive examples demonstrating:
- Basic usage with configuration models
- Pipeline usage with dependency injection  
- Retraining pipeline with composition
- Custom dependency injection
- Comparison of old vs new approaches

## Key Principles Applied

1. **SOLID Principles**
   - Single Responsibility: Each class has one reason to change
   - Open/Closed: Open for extension, closed for modification
   - Liskov Substitution: Protocols enable substitutability
   - Interface Segregation: Small, focused protocols
   - Dependency Inversion: Depend on abstractions, not concretions

2. **Composition over Inheritance**
   - Retraining pipeline composes base pipeline instead of inheriting

3. **Dependency Injection**
   - Components receive dependencies rather than creating them
   - Enables loose coupling and easy testing

4. **Factory Pattern**
   - Centralized creation of complex objects (models)
   - Easy to extend with new model types

5. **Configuration as Code**
   - Pydantic models for type-safe, validated configuration
   - Clear separation of configuration and runtime behavior

This refactoring significantly improves the codebase's maintainability, testability, and extensibility while preserving all existing functionality. 