# Documentation Directory

This directory contains comprehensive documentation generated during the development and optimization of the counterfactual fraud model project. These documents were primarily created by the Cursor agent during iterative development sessions, optimization experiments, and architectural refactoring efforts.

## Overview

The documentation covers four main areas:
1. **Project Specifications** - Original requirements and feature definitions
2. **Performance Optimization** - Detailed optimization analysis and usage guides  
3. **Architecture Refactoring** - Comprehensive codebase restructuring documentation

## Document Summaries

### 📋 [INSTRUCTIONS.md](./INSTRUCTIONS.md)
**Purpose**: Original project specification and requirements document

**Content Summary**:
- **Goal**: Create a simulator for testing counterfactual evaluation scenarios of a synthetic fraud model
- **Core Concept**: Allow a random share of transactions that would be blocked by the model to pass through, then use this data to estimate model performance if all transactions were allowed
- **Key Components**:
  - **Data Generator**: Creates synthetic fraud data using beta distributions for model scores and normal distributions for model error
  - **Logging Policy Generator**: Simulates fraud prevention policies with configurable cutoff values and exploration rates
  - **Counterfactual Values Estimator**: Applies counterfactual concepts using bootstrap methods and importance sampling
  - **Off Policy Evaluation Pipeline**: Orchestrates the complete evaluation workflow
  - **Off Policy Evaluation Simulator**: Runs multiple scenarios and generates visualization plots

**Key Parameters**: Beta distribution parameters (α=0.5, β=10), normal distribution (μ=0, σ=0.1), sample sizes (10K default), exploration rates, and bootstrap repetitions.

---

### ⚡ [OPTIMIZATION_USAGE_GUIDE.md](./OPTIMIZATION_USAGE_GUIDE.md)
**Purpose**: Comprehensive guide for using optimized CounterfactualEstimator methods

**Content Summary**:
- **Two Implementation Methods**:
  - `estimate_ope_metrics()`: Loop-based, memory-efficient (default)
  - `estimate_ope_metrics_vectorized()`: Matrix-based, faster for large datasets
- **Performance Results**: 1.44x average speedup with vectorized method, 1.67x best case
- **Memory Trade-off**: 32x more memory usage for vectorized approach
- **Usage Recommendations**:
  - **Loop-based**: Small datasets (<1K observations), memory constraints, production environments
  - **Vectorized**: Large datasets (>1K observations), high bootstrap counts (>100), performance-critical scenarios
- **Intelligent Warnings**: Automatic guidance to help choose optimal method based on dataset size and memory usage
- **Memory Estimation**: Guidelines for estimating memory requirements (~8MB per 1K obs × 1K bootstrap)

**Key Features**: Automatic method selection warnings, memory usage estimation, system requirements for BLAS optimization, troubleshooting guide.

---

### 📊 [PERFORMANCE_ANALYSIS_SUMMARY.md](./PERFORMANCE_ANALYSIS_SUMMARY.md)
**Purpose**: Detailed technical analysis of performance optimization results

**Content Summary**:
- **Optimization Confirmation**: Vectorized approach provides significant performance improvements
- **Performance Metrics**:
  - Average speedup: 1.44x (44% faster)
  - Best case: 1.67x speedup
  - Scalability: Performance advantage increases with problem size
- **Technical Analysis**:
  - **BLAS Configuration**: Apple Accelerate Framework with automatic multithreading
  - **Memory Trade-off**: 32.22x average memory increase
  - **Peak Memory**: Up to 153MB for large problems (10K observations, 1K bootstrap)
  - **Statistical Accuracy**: Both methods produce statistically equivalent results
- **Why It Works**: BLAS threading, vectorization eliminating overhead, better cache efficiency, optimized broadcasting
- **Recommendations**: 
  - Use vectorized for n_observed > 1,000 and sufficient memory
  - Use loop-based for memory-constrained environments or small datasets

**Technical Details**: Comprehensive benchmark results across 10 test scenarios on macOS ARM64 with Apple Accelerate Framework.

---

### 🏗️ [REFACTORING_GUIDE.md](./REFACTORING_GUIDE.md)
**Purpose**: Comprehensive documentation of major architectural refactoring

**Content Summary**:
- **Problems Addressed**:
  - **Tight Coupling**: Components directly instantiated dependencies
  - **Parameter Explosion**: Classes with 20+ constructor parameters
  - **Code Duplication**: Repeated logic across multiple classes
  - **Mixed Responsibilities**: Single classes doing too many things
  - **Lack of Abstractions**: No interfaces/protocols for testing and extension

- **New Architecture Features**:
  - **Pydantic Configuration Models**: Type-safe, validated configuration
  - **Protocol-Based Interfaces**: Clear contracts for dependency injection
  - **Dependency Injection**: Components receive dependencies instead of creating them
  - **Factory Pattern**: Centralized model creation
  - **Single Responsibility**: Each class has one clear purpose
  - **Composition over Inheritance**: Better modularity and flexibility

- **Benefits**:
  - **Type Safety**: Automatic validation and IDE support
  - **Testability**: Easy mocking of dependencies
  - **Extensibility**: Simple to add new model types and components
  - **Configuration Management**: Save/load configurations as code
  - **Maintainability**: Clear separation of concerns

- **Backward Compatibility**: Legacy imports still work alongside new refactored components

**Migration Guide**: Detailed before/after examples showing the transformation from massive parameter lists to clean configuration-based architecture.

---

## Usage Notes

- All documents were generated during active development and optimization sessions
- **INSTRUCTIONS.md** should be referenced for understanding the original project scope
- **OPTIMIZATION_USAGE_GUIDE.md** is essential for developers using the CounterfactualEstimator
- **PERFORMANCE_ANALYSIS_SUMMARY.md** provides the technical justification for optimization decisions
- **REFACTORING_GUIDE.md** is crucial for understanding the new architecture and migration path

## Related Files

These documents reference and complement other parts of the codebase:
- Source code in `counterfactual_fraud_model/`
- Example scripts in `aux_scripts/`
- Jupyter notebooks in `notebooks/`
- Test files in `tests/`

---

*This documentation represents the evolution of the counterfactual fraud model from initial concept through performance optimization to architectural maturity.*