# Modular Structure Guide

## Overview

The counterfactual fraud model package has been reorganized into a clean modular structure that improves readability, maintainability, and follows Python packaging best practices. This reorganization maintains complete backward compatibility while providing better code organization.

## Directory Structure

```
src/counterfactual_fraud_model/
├── __init__.py                    # Main package interface (backward compatible)
├── core/                          # Abstract interfaces and base classes
│   ├── __init__.py               
│   └── interfaces.py             # DataGenerator, ModelTrainer, etc.
├── strategies/                    # Data generation strategies
│   ├── __init__.py               
│   └── generation.py             # Probabilistic & sklearn strategies
├── factories/                     # Factory classes for object creation
│   ├── __init__.py               
│   └── model_trainers.py         # ModelTrainerFactory, concrete trainers
├── generators/                    # Concrete data generator implementations  
│   ├── __init__.py               
│   └── implementations.py        # ProbabilisticDataGenerator, MLModelDataGenerator
├── legacy/                        # Backward compatibility adapters
│   ├── __init__.py               
│   └── adapters.py               # LegacyDataGenerator, LegacySyntheticDataGenerator
├── data_generator.py             # Original DataGenerator (unchanged)
├── synthetic_data_generator.py   # Original SyntheticDataGenerator (unchanged)
├── logging_policy.py             # LoggingPolicyGenerator (unchanged)
├── counterfactual_estimator.py   # CounterfactualValuesEstimator (unchanged)
├── pipeline.py                   # OffPolicyEvaluationPipeline (unchanged)
└── synthetic_pipeline.py         # SyntheticOffPolicyEvaluationPipeline (unchanged)
```

## Submodule Descriptions

### 🔧 `core/` - Abstract Interfaces
Contains the abstract base classes and interfaces that define contracts for the system:

- **`DataGenerator`**: Abstract base class for all data generators
- **`DataGenerationConfig`**: Base configuration with Pydantic validation
- **`ModelTrainer`**: Abstract interface for model training strategies
- **`DataGenerationStrategy`**: Abstract interface for data generation algorithms

**Purpose**: Provides the foundation for dependency inversion and ensures all components follow consistent interfaces.

### 🎯 `strategies/` - Generation Strategies
Contains concrete implementations of data generation strategies:

- **`ProbabilisticGenerationStrategy`**: Beta distribution + noise approach
- **`SklearnGenerationStrategy`**: Scikit-learn make_classification approach  
- **`ProbabilisticGenerationConfig`**: Pydantic config for probabilistic generation
- **`SklearnGenerationConfig`**: Pydantic config for sklearn generation

**Purpose**: Implements the Strategy pattern, allowing different data generation algorithms to be swapped easily.

### 🏭 `factories/` - Object Creation
Contains factory classes that implement the Factory pattern:

- **`ModelTrainerFactory`**: Creates model trainers (LightGBM, RandomForest, Logistic)
- **`LightGBMTrainer`**: Concrete trainer for LightGBM models
- **`RandomForestTrainer`**: Concrete trainer for Random Forest models
- **`LogisticRegressionTrainer`**: Concrete trainer for Logistic Regression models
- **`ModelEvaluator`**: Utility for evaluating trained models

**Purpose**: Centralizes object creation and enables easy extension without modifying existing code.

### 🎲 `generators/` - Concrete Implementations
Contains the main data generator implementations:

- **`ProbabilisticDataGenerator`**: Uses probabilistic strategies for data generation
- **`MLModelDataGenerator`**: Uses ML models for realistic data generation
- **`DataGeneratorFactory`**: Factory for creating data generators

**Purpose**: Provides the main functionality using composition of strategies and factories.

### 🔄 `legacy/` - Backward Compatibility
Contains adapters that maintain the original API:

- **`LegacyDataGenerator`**: Adapter for original DataGenerator API
- **`LegacySyntheticDataGenerator`**: Adapter for original SyntheticDataGenerator API

**Purpose**: Ensures all existing code continues to work without modification.

## Usage Patterns

### Importing from Main Package (Recommended)
```python
# Public API - recommended approach
from src.counterfactual_fraud_model import (
    DataGenerator,                    # Original API
    SyntheticDataGenerator,          # Original API
    ProbabilisticGenerationConfig,   # New Pydantic configs
    SklearnGenerationConfig,
    DataGeneratorFactory,            # New factory
    ModelTrainerFactory,
    ProbabilisticDataGenerator,      # New generators
    MLModelDataGenerator
)
```

### Importing from Submodules (Advanced)
```python
# Direct submodule access for advanced use cases
from src.counterfactual_fraud_model.core import DataGenerator, ModelTrainer
from src.counterfactual_fraud_model.strategies import ProbabilisticGenerationStrategy
from src.counterfactual_fraud_model.factories import ModelEvaluator
from src.counterfactual_fraud_model.generators import DataGeneratorFactory
from src.counterfactual_fraud_model.legacy import LegacyDataGenerator
```

### Creating Custom Extensions
```python
# Easy to extend with new strategies
from src.counterfactual_fraud_model.core import DataGenerationStrategy
from src.counterfactual_fraud_model.factories import ModelTrainerFactory

class CustomGenerationStrategy(DataGenerationStrategy):
    def generate_features_and_labels(self, config):
        # Your custom logic here
        return features, labels
    
    def get_strategy_info(self):
        return {'strategy_type': 'custom'}

# Register new model trainer
class XGBoostTrainer(ModelTrainer):
    # Implementation...
    pass

ModelTrainerFactory.register_trainer('xgboost', XGBoostTrainer)
```

## Benefits of Modular Structure

### 📁 **Better Organization**
- Related functionality grouped together
- Clear separation of concerns
- Easier to find and understand code
- Follows Python packaging conventions

### 🔍 **Improved Readability**
- Each module has a single, clear purpose
- Dependencies are explicit and minimal
- Easier to understand the overall architecture
- Self-documenting through structure

### 🧪 **Enhanced Testability**
- Individual modules can be tested in isolation
- Easy to mock dependencies
- Clear boundaries for unit tests
- Integration tests are more focused

### 🚀 **Better Extensibility**
- New strategies can be added to `strategies/`
- New model trainers can be added to `factories/`
- New generators can be added to `generators/`
- Extensions don't affect existing code

### 🔄 **Maintained Compatibility**
- All existing imports continue to work
- No breaking changes to public API
- Gradual migration path available
- Legacy code fully supported

### 📦 **Scalable Architecture**
- Easy to add new submodules as needed
- Clear patterns for where new code belongs
- Supports large team development
- Follows industry best practices

## Migration Strategy

### For Existing Users
**No action required!** All your existing code will continue to work exactly as before:

```python
# This still works unchanged
from src.counterfactual_fraud_model import DataGenerator, SyntheticDataGenerator
generator = SyntheticDataGenerator(n_samples=10000, model_type="lightgbm")
data = generator.generate_data()
```

### For New Development
Consider using the new organized API for better maintainability:

```python
# New approach with better organization
from src.counterfactual_fraud_model import (
    SklearnGenerationConfig,
    DataGeneratorFactory
)

config = SklearnGenerationConfig(n_samples=10000, n_features=20, random_state=42)
generator = DataGeneratorFactory.create_ml_model_generator(
    generation_config=config,
    model_type="lightgbm"
)
data = generator.generate_data()
```

### Gradual Adoption
You can adopt the new structure incrementally:

1. **Start with configs**: Use new Pydantic configuration objects
2. **Use factories**: Leverage factories for new model types
3. **Direct imports**: Import from submodules when needed
4. **Custom extensions**: Add your own strategies and trainers

## Design Principles Applied

### SOLID Principles
- **Single Responsibility**: Each module has one clear purpose
- **Open/Closed**: Easy to extend without modifying existing code
- **Liskov Substitution**: All implementations follow consistent interfaces
- **Interface Segregation**: Small, focused interfaces
- **Dependency Inversion**: Depends on abstractions, not concretions

### Design Patterns
- **Strategy Pattern**: Implemented in `strategies/`
- **Factory Pattern**: Implemented in `factories/`
- **Adapter Pattern**: Implemented in `legacy/`
- **Composition over Inheritance**: Throughout the architecture

### Python Best Practices
- **Clear module structure**: Each directory has a clear purpose
- **Explicit imports**: Dependencies are clear and minimal
- **Namespace organization**: Related functionality grouped together
- **Backward compatibility**: Existing code continues to work

## Conclusion

The new modular structure provides significant improvements in code organization, readability, and maintainability while preserving complete backward compatibility. The architecture is now much more scalable and follows established software engineering best practices.

Whether you're maintaining existing code or building new features, the modular structure provides clear guidance on where code belongs and how to extend the system safely and efficiently. 