# Counterfactual Fraud Model

A Python package for counterfactual evaluation of fraud detection models, implementing data generation, logging policies, counterfactual estimation, and comprehensive simulation capabilities.

## ✨ Features

- **SOLID Principles Architecture**: Clean, maintainable, and extensible codebase
- **Design Patterns Implementation**: Strategy, Factory, and Adapter patterns
- **Enhanced Configuration**: Pydantic models with validation and JSON serialization
- **Modular Structure**: Organized submodules for better code organization
- **Backward Compatibility**: All existing code continues to work unchanged
- **Type Safety**: Runtime validation and type checking
- **Easy Extension**: Plugin-like architecture for adding new features

## 🚀 Quick Start

### Installation

1. Install [uv](https://github.com/astral-sh/uv):
   ```sh
   pip install uv
   ```

2. Install dependencies:
   ```sh
   uv pip install -r requirements.txt
   ```

### Basic Usage

```python
from src.counterfactual_fraud_model import DataGeneratorFactory, SklearnGenerationConfig

# Create configuration with validation
config = SklearnGenerationConfig(
    n_samples=10000,
    n_features=20,
    n_informative=10,
    random_state=42
)

# Generate data using factory
generator = DataGeneratorFactory.create_ml_model_generator(
    generation_config=config,
    model_type="lightgbm"
)

data = generator.generate_data()
print(f"Generated {data.shape[0]} samples with fraud rate: {data['is_fraud'].mean():.3f}")
```

## 📚 Documentation

**Comprehensive documentation is available in the [`docs/`](docs/) directory:**

- **[Architecture Overview](docs/REFACTORING_GUIDE.md)** - Complete guide to SOLID principles implementation
- **[Modular Structure](docs/MODULAR_STRUCTURE_GUIDE.md)** - Detailed explanation of the new organization  
- **[Refactoring Summary](docs/COMPREHENSIVE_REFACTORING_SUMMARY.md)** - Executive summary of all improvements
- **[Project Context](docs/CONTEXT.md)** - Background and project goals
- **[Original Instructions](docs/INSTRUCTIONS.md)** - Initial requirements

See [`docs/README.md`](docs/README.md) for a complete documentation index and reading recommendations.

## 🧪 Testing & Demos

Run the demonstration scripts to see the improvements in action:

```bash
# Show SOLID principles and design patterns
python refactoring_demo.py

# Show Pydantic validation and configuration features  
python pydantic_validation_demo.py
```

Run tests:
```sh
make test
```

## 🛠️ Development

- **Format code**: `make format`
- **Lint code**: `make lint`  
- **Type check**: `make typecheck`
- **Build docs**: `make docs`

## 🏗️ Architecture

The project follows a clean modular architecture:

```
src/counterfactual_fraud_model/
├── core/                    # Abstract interfaces & base classes
├── strategies/              # Data generation strategies
├── factories/               # Factory classes for object creation
├── generators/              # Concrete implementations
├── legacy/                  # Backward compatibility adapters
└── [original components]    # Unchanged pipeline components
```

## 📋 Version History

- **v0.1.0**: Original implementation
- **v0.2.0**: SOLID principles and design patterns  
- **v0.3.0**: Pydantic integration and modular structure 