# Data Generator Refactoring Guide

## Overview

The data generator classes in the counterfactual fraud model have been refactored to follow SOLID principles and implement proven design patterns. This refactoring addresses code duplication, improves maintainability, and enhances extensibility while maintaining complete backward compatibility.

## Problems Addressed

### Original Issues

1. **Single Responsibility Principle (SRP) Violations**
   - `SyntheticDataGenerator` handled data generation, model training, evaluation, and validation
   - Mixed concerns made the class difficult to test and maintain

2. **Open/Closed Principle (OCP) Violations**
   - Adding new model types required modifying the `_create_model()` method
   - No clean way to extend functionality without changing existing code

3. **Dependency Inversion Principle (DIP) Violations**
   - No abstractions - everything depended on concrete implementations
   - Tight coupling between components

4. **Code Duplication**
   - Both generators shared similar patterns for parameter management
   - Repeated validation logic and data output formatting

5. **High Complexity**
   - Large constructor signatures
   - Mixed responsibilities in single classes

## Refactoring Solution

### Design Patterns Implemented

#### 1. Strategy Pattern
```python
# Different data generation algorithms can be swapped easily
from src.counterfactual_fraud_model import (
    ProbabilisticGenerationStrategy,
    SklearnGenerationStrategy
)

# Use different strategies with the same interface
strategy1 = ProbabilisticGenerationStrategy(config)
strategy2 = SklearnGenerationStrategy(config)
```

#### 2. Factory Pattern
```python
# Centralized, extensible object creation
from src.counterfactual_fraud_model import ModelTrainerFactory

# Create different model types without knowing implementation details
trainer = ModelTrainerFactory.create_trainer("lightgbm", params)

# Easy to extend without modifying existing code
ModelTrainerFactory.register_trainer("xgboost", XGBoostTrainer)
```

#### 3. Composition over Inheritance
```python
# Complex functionality built by composing simple components
generator = MLModelDataGenerator(
    generation_config=config,
    model_type="lightgbm"
)
# Internally uses strategy + trainer + evaluator through composition
```

#### 4. Configuration Object Pattern with Pydantic
```python
# Complex parameters encapsulated in Pydantic configuration objects
# Includes validation, type checking, and JSON serialization
config = SklearnGenerationConfig(
    n_samples=100_000,
    n_features=30,
    n_informative=15,
    weights=[0.985, 0.015],
    random_state=42
)

# Automatic validation
try:
    invalid_config = SklearnGenerationConfig(n_features=10, n_informative=15)
except ValidationError as e:
    print("Validation caught: n_informative cannot exceed n_features")

# JSON serialization/deserialization
config_json = config.model_dump_json()
reconstructed = SklearnGenerationConfig.model_validate_json(config_json)
```

### SOLID Principles Compliance

#### ✅ Single Responsibility Principle (SRP)
- **Data Generation Strategies**: Only responsible for generating data
- **Model Trainers**: Only responsible for training models
- **Model Evaluator**: Only responsible for evaluation
- **Configuration Objects**: Only responsible for parameter management

#### ✅ Open/Closed Principle (OCP)
- New model types can be added without modifying existing code
- New data generation strategies can be added independently
- Factory pattern enables extension without modification

#### ✅ Liskov Substitution Principle (LSP)
- All data generators implement the same interface
- Components are truly interchangeable
- Polymorphic behavior works correctly

#### ✅ Interface Segregation Principle (ISP)
- Clients depend only on interfaces they use
- Clear separation between data generation and model training
- Focused, minimal interfaces

#### ✅ Dependency Inversion Principle (DIP)
- High-level modules depend on abstractions
- Factory pattern provides abstraction over concrete implementations
- Strategy pattern abstracts algorithms

## Usage Guide

### Backward Compatibility

**All existing code continues to work unchanged:**

```python
# Original API still works exactly the same
from src.counterfactual_fraud_model import DataGenerator, SyntheticDataGenerator

# Your existing code doesn't need to change
generator = SyntheticDataGenerator(n_samples=10000, model_type="lightgbm")
data = generator.generate_data()
```

### New Refactored API

#### Using the Factory Pattern

```python
from src.counterfactual_fraud_model import DataGeneratorFactory

# Create probabilistic generator
prob_gen = DataGeneratorFactory.create_probabilistic_generator(
    alpha=0.1,
    beta_param=2.0,
    sample_size=100_000,
    random_state=42
)

# Create ML model generator
ml_gen = DataGeneratorFactory.create_ml_model_generator(
    n_samples=100_000,
    n_features=30,
    model_type="lightgbm",
    random_state=42
)
```

#### Using Configuration Objects

```python
from src.counterfactual_fraud_model import (
    ProbabilisticGenerationConfig,
    SklearnGenerationConfig,
    ProbabilisticDataGenerator,
    MLModelDataGenerator
)

# Probabilistic approach with clean configuration
config = ProbabilisticGenerationConfig(
    alpha=0.1,
    beta_param=2.0,
    mean=0,
    sd=0.5,
    sample_size=100_000,
    random_state=42
)
generator = ProbabilisticDataGenerator(config)

# ML approach with clean configuration
config = SklearnGenerationConfig(
    n_samples=100_000,
    n_features=30,
    n_informative=15,
    weights=[0.985, 0.015],
    random_state=42
)
generator = MLModelDataGenerator(config, model_type="lightgbm")
```

#### Extending with Custom Components

```python
# Add a custom model trainer
class XGBoostTrainer(ModelTrainer):
    def train_model(self, X, y):
        import xgboost as xgb
        model = xgb.XGBClassifier(**self.final_params)
        model.fit(X, y)
        return model
    
    def get_model_info(self):
        return {'model_type': 'xgboost', ...}

# Register with factory
ModelTrainerFactory.register_trainer('xgboost', XGBoostTrainer)

# Now you can use it
trainer = ModelTrainerFactory.create_trainer('xgboost', params)
```

#### Enhanced Configuration with Pydantic Features

```python
from pydantic import ValidationError

# Configuration with automatic validation
try:
    config = SklearnGenerationConfig(
        n_samples=10000,
        n_features="20",  # Automatic string → int conversion
        n_informative=15,
        weights=["0.98", "0.02"],  # Automatic string → float conversion
        flip_y=0.01
    )
    print(f"Created config with {config.n_features} features")
except ValidationError as e:
    print(f"Configuration error: {e}")

# JSON serialization for reproducible experiments
config_json = config.model_dump_json(indent=2)
with open("experiment_config.json", "w") as f:
    f.write(config_json)

# Load configuration from JSON
with open("experiment_config.json", "r") as f:
    loaded_config = SklearnGenerationConfig.model_validate_json(f.read())

# Schema generation for documentation
schema = config.model_json_schema()
print("Configuration schema:", schema)

# Easy configuration inspection
print("Config as dict:", config.model_dump())
print("Config fields:", list(config.model_fields.keys()))
```

## Benefits

### 1. **Improved Maintainability**
- Clear separation of concerns
- Reduced complexity in individual classes
- Easier to understand and modify

### 2. **Enhanced Testability**
- Components can be tested in isolation
- Easy to inject mock objects
- Configuration objects simplify test setup

### 3. **Better Extensibility**
- Add new model types without modifying existing code
- Create new data generation strategies independently
- Factory pattern enables clean extension points

### 4. **Reduced Code Duplication**
- Shared functionality extracted into reusable components
- Common patterns implemented once and reused
- Parameter management centralized in configuration objects

### 5. **Improved Type Safety**
- Clear interfaces define expected behavior
- Configuration objects provide validation
- Better IDE support and error detection

### 6. **Performance Benefits**
- Lazy initialization reduces memory usage
- Composition is more memory-efficient than inheritance
- Clear separation allows for optimization opportunities

### 7. **Enhanced Configuration Management with Pydantic**
- **Runtime Validation**: Automatic type checking and constraint validation
- **Clear Error Messages**: Descriptive validation errors with field-specific context
- **JSON Serialization**: Easy save/load of configurations for reproducibility
- **Schema Generation**: Automatic documentation and API schema generation
- **Type Coercion**: Safe automatic type conversion (e.g., "100" → 100)
- **Field Descriptions**: Self-documenting configurations with built-in help

## Migration Guide

### For Existing Users

**No migration required!** Your existing code will continue to work exactly as before. The original classes (`DataGenerator` and `SyntheticDataGenerator`) are still available and unchanged.

### For New Projects

Consider using the new refactored API for better maintainability and extensibility:

```python
# Instead of this (old way)
generator = SyntheticDataGenerator(
    n_samples=100_000,
    n_features=30,
    model_type="lightgbm",
    random_state=42
)

# Use this (new way)
generator = DataGeneratorFactory.create_ml_model_generator(
    n_samples=100_000,
    n_features=30,
    model_type="lightgbm",
    random_state=42
)
```

### Gradual Adoption

You can gradually adopt the new API in your codebase:

1. Start using new configuration objects for complex parameter sets
2. Use factories for creating new model types
3. Leverage the improved testing capabilities
4. Extend with custom strategies when needed

## New Modular Structure

The refactored code has been organized into a clean modular structure for better readability and maintainability:

```
src/counterfactual_fraud_model/
├── core/                    # Abstract interfaces and base classes
├── strategies/              # Data generation strategies  
├── factories/               # Factory classes for object creation
├── generators/              # Concrete data generator implementations
├── legacy/                  # Backward compatibility adapters
└── [original files]         # Unchanged original components
```

**Benefits:**
- 📁 **Better Organization**: Related functionality grouped together
- 🔍 **Improved Readability**: Each module has a single, clear purpose  
- 🧪 **Enhanced Testability**: Individual modules can be tested in isolation
- 🚀 **Better Extensibility**: Clear patterns for where new code belongs
- 🔄 **Maintained Compatibility**: All existing imports continue to work

See [`MODULAR_STRUCTURE_GUIDE.md`](MODULAR_STRUCTURE_GUIDE.md) for detailed information about the new organization.

## Testing the Refactored Code

Run the demonstration scripts to see the improvements in action:

```bash
# Show SOLID principles and design patterns
python refactoring_demo.py

# Show Pydantic validation and configuration features
python pydantic_validation_demo.py
```

These will show:
- Backward compatibility verification
- SOLID principles in action
- Design patterns implementation
- Extensibility examples
- Testing improvements
- Enhanced validation with Pydantic
- JSON serialization capabilities
- Type coercion and error handling
- New modular structure benefits

## Conclusion

This refactoring maintains complete backward compatibility while providing a much more maintainable, extensible, and testable codebase. The new architecture follows established software engineering principles and patterns, making it easier to add new features and maintain existing functionality.

### Key Improvements Summary

1. **SOLID Principles Compliance**: Clean separation of concerns and dependency management
2. **Design Patterns Implementation**: Strategy, Factory, and Composition patterns for flexibility
3. **Enhanced Configuration Management**: Pydantic models with validation, serialization, and documentation
4. **Improved Testability**: Individual components can be tested in isolation
5. **Better Error Handling**: Clear, descriptive validation errors and type checking
6. **JSON Serialization**: Easy configuration persistence and sharing
7. **Schema Generation**: Self-documenting APIs and configurations
8. **Type Safety**: Runtime type checking and automatic conversion
9. **Extensibility**: Easy to add new models and generation strategies
10. **Backward Compatibility**: All existing code continues to work without changes

The refactored code is ready for production use and provides a solid foundation for future enhancements to the counterfactual fraud model simulator. The Pydantic integration adds enterprise-level configuration management capabilities while maintaining the simplicity of the original API. 