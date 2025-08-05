# 📓 Notebooks Directory

Welcome to the notebooks directory of the Counterfactual Fraud Model project! This directory contains Jupyter notebooks and analysis utilities that demonstrate the capabilities of the framework through practical examples and comprehensive simulations.

## 📋 Directory Contents

### 📚 Main Notebooks

#### [`getting_started.ipynb`](./getting_started.ipynb)
**🎯 Perfect starting point for newcomers!**

A comprehensive tutorial that introduces you to the counterfactual fraud model project. This notebook provides:

- **Configuration System Overview**: Understanding all configuration classes and enums
- **Core Concepts**: Data generation, model training, logging policies, and counterfactual estimation
- **Hands-on Examples**: Working code samples you can run immediately
- **Pipeline Demonstrations**: Complete examples of all three main pipeline types
- **Best Practices**: How to configure and use the framework effectively

**What you'll learn:**
- How to configure `DataGeneratorConfig`, `SyntheticDataConfig`, `ModelConfig`, etc.
- Understanding `ModelType`, `PropensityType`, and `RetrainingStrategy` enums
- Using concrete classes like `DataGenerator`, `SyntheticDataGenerator`, `LoggingPolicyGenerator`
- Running `OffPolicyEvaluationPipeline`, `SyntheticOffPolicyEvaluationPipeline`, and `SyntheticRetrainingPipeline`
- Interpreting results and performance metrics

#### [`counterfactual_simulations.ipynb`](./counterfactual_simulations.ipynb)
**🔬 Advanced simulation scenarios and analysis**

This notebook contains three comprehensive simulation studies that demonstrate different aspects of the framework:

1. **Basic OPE Simulations**
   - Uses `OffPolicyEvaluationPipeline` with distributional parameters
   - Varies exploration rates to study policy performance
   - Generates model score histograms, calibration curves, and precision-recall analysis

2. **Synthetic OPE Simulations**
   - Uses `SyntheticOffPolicyEvaluationPipeline` with realistic ML datasets
   - Compares different model types (LightGBM, Random Forest, Logistic Regression)
   - Evaluates the impact of exploration rates on synthetic data

3. **Synthetic Retraining Simulations**
   - Uses `SyntheticRetrainingPipeline` to study model retraining strategies
   - Compares filtering, weighting, and fraud injection approaches
   - Analyzes the effectiveness of different retraining strategies across exploration rates

### 🛠️ Utilities (`utils/` directory)

The `utils/` directory contains specialized analysis modules that support the simulation notebooks:

#### [`common_analysis.py`](./utils/common_analysis.py)
**📊 Shared visualization and analysis functions**

Contains reusable plotting and analysis functions used across all simulation scripts:

- **Visualization Functions**:
  - `plot_model_scores_histogram()`: Distribution of model scores
  - `plot_calibration_curve()`: Model calibration analysis
  - `plot_precision_recall_curve()`: Precision-recall performance
  - `plot_ope_metrics()`: Off-policy evaluation metrics comparison

- **Summary Tables**:
  - `create_standard_data_summary_table()`: Basic data statistics
  - `create_synthetic_data_summary_table()`: Synthetic dataset overview
  - `create_policy_metrics_table()`: Policy performance comparison

#### [`simulation_analysis.py`](./utils/simulation_analysis.py)
**🔄 Basic OPE simulation utilities**

Specialized functions for running multiple `OffPolicyEvaluationPipeline` simulations:

- `run_simulations()`: Execute multiple simulations with different exploration rates
- Generates comprehensive analysis for distributional parameter-based simulations
- Supports parameter sweeps and comparative analysis

#### [`synthetic_simulation_analysis.py`](./utils/synthetic_simulation_analysis.py)
**🧬 Synthetic data simulation utilities**

Advanced functions for `SyntheticOffPolicyEvaluationPipeline` analysis:

- `run_exploration_rate_simulations()`: Multi-parameter simulations with synthetic data
- Supports different model types and exploration rate combinations
- Includes model performance comparisons and synthetic data analysis

#### [`retraining_simulation_analysis.py`](./utils/retraining_simulation_analysis.py)
**🔄 Retraining strategy analysis**

Comprehensive utilities for studying model retraining approaches:

- `run_retraining_simulations()`: Execute simulations across multiple retraining strategies
- `RetrainingSimulationAnalyzer` class: Advanced analysis and visualization
- Supports comparison of filtering, weighting, and fraud injection strategies
- Generates dot charts and summary tables for strategy effectiveness

## 🚀 Getting Started

### For Newcomers
1. **Start with [`getting_started.ipynb`](./getting_started.ipynb)**
   - Run all cells to understand the framework basics
   - Experiment with different configuration parameters
   - Try modifying the examples to see how they behave

2. **Explore [`counterfactual_simulations.ipynb`](./counterfactual_simulations.ipynb)**
   - Review the three simulation scenarios
   - Understand how different parameters affect results
   - Use this as a template for your own experiments

### For Advanced Users
1. **Leverage the `utils/` modules**
   - Import analysis functions in your own notebooks
   - Extend existing functions for custom analysis
   - Use the simulation runners for parameter studies

2. **Customize simulations**
   - Modify exploration rates, sample sizes, and other parameters
   - Add new model types or retraining strategies
   - Create your own analysis pipelines

## 📊 Key Analysis Capabilities

The notebooks and utilities provide comprehensive analysis including:

- **Model Performance**: ROC-AUC, precision, recall, F1-score, average precision
- **Policy Evaluation**: Allow/block rates, fraud rates, exploration effectiveness
- **Counterfactual Metrics**: Off-policy evaluation with confidence intervals
- **Calibration Analysis**: Model reliability and confidence assessment
- **Strategy Comparison**: Systematic comparison of different approaches

## 💡 Usage Tips

### Running Simulations
```python
# Basic OPE simulation
from notebooks.utils.simulation_analysis import run_simulations
results, data = run_simulations(
    exploration_rates=np.linspace(0.01, 0.2, 5),
    cutoff=0.05,
    sample_size=50_000
)

# Synthetic retraining simulation
from notebooks.utils.retraining_simulation_analysis import run_retraining_simulations
results, data = run_retraining_simulations(
    exploration_rates=np.linspace(0.01, 0.1, 5),
    strategies=[RetrainingStrategy.FILTERING, RetrainingStrategy.WEIGHTING]
)
```

### Creating Visualizations
```python
from notebooks.utils.common_analysis import (
    plot_ope_metrics, 
    create_policy_metrics_table,
    plot_calibration_curve
)

# Generate comprehensive plots
plot_ope_metrics(results)
table = create_policy_metrics_table(results)
plot_calibration_curve(reference_data)
```

## 🔧 Dependencies

All notebooks require the main counterfactual fraud model package to be installed and importable. Make sure you're running from the project root or have the project path in your Python path:

```python
import sys
from pathlib import Path
project_root = Path().resolve().parent
sys.path.insert(0, str(project_root))
```

## 📈 Performance Considerations

- **Sample Sizes**: Start with smaller sample sizes (5K-10K) for experimentation, scale up for final analysis
- **Bootstrap Iterations**: Use 1K iterations for quick tests, 5K+ for publication-quality confidence intervals
- **Simulation Counts**: The retraining simulations can be computationally intensive with large parameter grids

## 🤝 Contributing

When adding new notebooks or utilities:

1. Follow the established naming conventions
2. Add comprehensive docstrings and comments
3. Use the common analysis functions where possible
4. Update this README with new capabilities
5. Include practical examples in your notebooks

Happy analyzing! 🎉