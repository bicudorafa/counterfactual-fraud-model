# Counterfactual Fraud Model

A comprehensive Python framework for **off-policy evaluation** of fraud detection models using counterfactual estimation techniques.

## 🎯 What is this project?

This framework enables data scientists and researchers to evaluate fraud detection policies without running risky A/B tests in production. Using **counterfactual estimation** and **importance sampling**, you can:

- **Simulate "what-if" scenarios**: Estimate how different fraud detection thresholds would perform using historical data
- **Test exploration strategies**: Allow a controlled percentage of normally-blocked transactions to pass through, then analyze the results
- **Compare model performance**: Evaluate different machine learning models and retraining strategies
- **Optimize policies safely**: Find optimal fraud detection parameters without exposing your business to actual fraud

### Key Concepts

- **Logging Policy**: Your current fraud detection system that blocks/allows transactions
- **Target Policy**: The new policy you want to evaluate
- **Exploration Rate**: Percentage of blocked transactions allowed through for learning
- **Counterfactual Estimation**: Statistical techniques to estimate target policy performance from logged data

### Use Cases

- **Risk Management**: Safely test new fraud detection thresholds
- **Model Comparison**: Compare different ML algorithms without production deployment
- **Policy Optimization**: Find optimal cutoff values and exploration rates
- **Research**: Study bias in fraud detection systems and mitigation strategies

## Setup

### Prerequisites

- [uv](https://github.com/astral-sh/uv) (Python package and project manager)

*Note: This project requires Python 3.9+, but uv can install and manage Python versions for you.*

### Installation

1. **Install uv** (if not already installed):
   
   On macOS and Linux:
   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   
   On Windows:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   
   Alternatively, install via pip:
   ```sh
   pip install uv
   ```

2. **Install dependencies**:
   ```sh
   # Install production dependencies
   uv sync
   ```

4. **Install development dependencies** (optional, for contributors):
   ```sh
   # Install with development dependencies
   uv sync --group dev
   ```

5. **Activate the virtual environment**:
   ```sh
   # Activate the virtual environment created by uv
   source .venv/bin/activate
   ```

### Dependency Management

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. The `uv.lock` file is committed to ensure reproducible builds across environments.

- **Add new dependencies**: `uv add <package-name>`
- **Add development dependencies**: `uv add --group dev <package-name>`
- **Update dependencies**: `uv sync`
- **Remove dependencies**: `uv remove <package-name>`

### Development Setup

For development work, install the full development environment:

```sh
# Install with all development tools
uv sync --group dev

# Install pre-commit hooks (recommended)
uv run pre-commit install
```

This includes tools for:
- Code formatting (Black, isort)
- Linting (flake8)
- Type checking (mypy)
- Testing (pytest, pytest-cov, pytest-mock)
- Documentation (Sphinx)
- Jupyter notebooks (ipykernel)

## 📁 Repository Structure

### 🎯 Getting Started (Choose Your Path)

**For beginners**: Start with [`notebooks/getting_started.ipynb`](./notebooks/getting_started.ipynb) - Interactive tutorial with step-by-step examples

**For adv usage**: Explore [`notebooks/counterfactual_simulations.ipynb`](./notebooks/counterfactual_simulations.ipynb) - Advanced simulation scenarios

### 🏗️ Core Framework (`counterfactual_fraud_model/`)

- **`generators/`**: Data generation, policy simulation, and retraining preprocessors
- **`estimators/`**: Counterfactual evaluation with confidence intervals (optimized implementations)
- **`pipelines/`**: Three main workflows - basic OPE, synthetic ML, and retraining comparison
- **`core/`**: Model training supporting LightGBM, Random Forest, and Logistic Regression
- **`factories/`**: Centralized model creation
- **`config.py`**: Type-safe Pydantic configuration classes
- **`protocols.py`**: Interface definitions for dependency injection

### 📓 Examples & Documentation

- **`notebooks/`**: Interactive Jupyter tutorials and advanced simulation analysis
- **`aux_scripts/`**: Command-line tools including comprehensive getting started guide and simulation runner
- **`documents/`**: Technical documentation covering specifications, optimization, and architecture


## 🔧 Advanced Features

- **Performance Optimization**: Vectorized implementations with 44% average speedup
- **Multiple Model Support**: LightGBM, Random Forest, Logistic Regression
- **Retraining Strategies**: Filtering, inverse propensity weighting, fraud injection
- **Comprehensive Analysis**: Built-in visualization and statistical analysis tools
- **Type Safety**: Pydantic-based configuration with automatic validation
- **Extensible Architecture**: Protocol-based interfaces for easy customization