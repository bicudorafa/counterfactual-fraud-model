# Auxiliary Scripts Directory

This directory contains auxiliary scripts that were primarily used by Cursor Agent for testing new implementations, validating refactoring changes, and exploring different approaches during the development of the counterfactual fraud model framework.

## Overview

These scripts serve as:
- **Development testing tools** for validating new features and refactoring efforts
- **Performance analysis utilities** for benchmarking different implementations
- **Usage examples** demonstrating various framework capabilities
- **Integration tests** ensuring system reliability across different scenarios

## Scripts Description

### 🚀 `getting_started_guide.py` (801 lines)
**Purpose**: Comprehensive tutorial and demonstration script

**What it does**:
- Provides a complete getting started guide for newcomers to the counterfactual fraud model framework
- Demonstrates all major configuration classes (DataGeneratorConfig, SyntheticDataConfig, ModelConfig, etc.)
- Shows concrete implementation examples of generators, estimators, and trainers
- Includes runnable examples for all three pipeline types:
  - Basic Off-Policy Evaluation Pipeline
  - Synthetic Off-Policy Evaluation Pipeline  
  - Synthetic Retraining Pipeline
- Explains results interpretation and common use cases
- Acts as both documentation and working code examples

**Likely usage**: Used by Cursor Agent to create comprehensive documentation and validate that all framework components work together correctly through practical examples.

---

### ⚡ `performance_stress_test.py` (495 lines)
**Purpose**: Performance optimization analysis and benchmarking

**What it does**:
- Compares original loop-based vs vectorized implementations of CounterfactualEstimator
- Tests various scenarios with different dataset sizes, bootstrap counts, and thread configurations
- Monitors memory usage patterns and execution times
- Validates statistical similarity between implementations
- Provides performance visualizations and recommendations
- Tests BLAS threading configuration impact

**Likely usage**: Used by Cursor Agent to validate performance optimizations during CounterfactualEstimator refactoring, ensuring vectorized implementations provide speedup without sacrificing accuracy.

---

### 🔄 `simulations.py` (130 lines)
**Purpose**: Centralized simulation runner

**What it does**:
- Orchestrates execution of all available simulation analyses from notebooks/utils
- Runs three comprehensive analyses in sequence:
  1. Standard Off-Policy Evaluation Pipeline Analysis
  2. Synthetic Off-Policy Evaluation Pipeline Analysis
  3. Synthetic Retraining Pipeline Analysis
- Provides command-line interface for running specific analyses or all at once
- Consolidates simulation outputs and provides summary reporting

**Likely usage**: Used by Cursor Agent as a quick way to run comprehensive testing across all simulation scenarios to validate framework behavior across different configurations.

---

### 📋 `synthetic_retraining_pipeline_example.py` (369 lines)
**Purpose**: Comprehensive usage demonstration for SyntheticRetrainingPipeline

**What it does**:
- Demonstrates basic usage with default configurations
- Shows custom configuration examples with different models and parameters
- Illustrates runtime parameter overrides for experimentation
- Provides comprehensive analysis examples with data access
- Includes error handling and edge case demonstrations
- Tests various retraining strategies and model types

**Likely usage**: Used by Cursor Agent to develop and test the SyntheticRetrainingPipeline configuration system, ensuring it handles various scenarios and provides flexible experimentation capabilities.

---

### 🧪 `test_counterfactual_estimator_refactoring.py` (740 lines)
**Purpose**: Comprehensive refactoring validation test suite

**What it does**:
- Validates functional correctness of all CounterfactualEstimator public methods
- Tests statistical similarity between different implementations
- Validates input validation and error handling
- Tests edge cases and boundary conditions
- Compares performance characteristics of different implementations
- Analyzes memory efficiency patterns
- Validates helper method functionality
- Ensures API backward compatibility
- Tests warning systems

**Likely usage**: Used by Cursor Agent during major refactoring of the CounterfactualEstimator class to ensure no regressions were introduced while improving code quality and performance.

---

### ✅ `test_simplified_results.py` (605 lines)
**Purpose**: Validation of simplified results structure

**What it does**:
- Tests the refactored SyntheticRetrainingPipeline run_pipeline method
- Validates new simplified results output structure containing:
  - `original_results`: Results from the base pipeline
  - `retrained_model_performance`: Performance metrics of retrained model
  - `ope_metrics`: Counterfactual estimation results
- Tests separate method functionality (generate_logging_policy_data, run_retrain_pipeline)
- Compares results across different configurations
- Tests edge cases for robustness
- Validates structure consistency and expected data types

**Likely usage**: Used by Cursor Agent to validate and test a major refactoring that simplified the SyntheticRetrainingPipeline output structure, ensuring the new design works correctly while maintaining functionality.

## Common Patterns

### Testing Philosophy
These scripts demonstrate thorough testing practices including:
- **Functional correctness** - Ensuring methods work as expected
- **Statistical validation** - Comparing implementations for numerical accuracy
- **Performance analysis** - Benchmarking different approaches
- **Edge case handling** - Testing boundary conditions
- **API compatibility** - Ensuring backward compatibility during refactoring

### Development Workflow
The scripts show a development pattern of:
1. **Implementation** - Creating new features or refactoring existing ones
2. **Validation** - Comprehensive testing of changes
3. **Benchmarking** - Performance analysis and optimization
4. **Documentation** - Creating usage examples and guides

### Framework Exploration
Many scripts explore different aspects of the framework:
- Configuration flexibility and customization
- Pipeline composition and workflow
- Model comparison and evaluation
- Policy analysis and optimization

## Usage Notes

- These scripts are primarily development and testing tools
- They provide valuable examples for understanding framework capabilities
- Some scripts generate visualizations and performance reports
- All scripts are designed to be run independently
- They serve as regression tests for major framework changes

## Dependencies

All scripts depend on the main counterfactual fraud model framework and typically require:
- pandas, numpy for data manipulation
- matplotlib, seaborn for visualizations
- psutil for system monitoring (performance tests)
- sklearn for machine learning components

---

*These auxiliary scripts represent the iterative development and testing process used to build and refine the counterfactual fraud model framework, ensuring robustness, performance, and usability.*