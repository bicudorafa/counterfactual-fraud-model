# CounterfactualEstimator Optimization Usage Guide

## Overview

The `CounterfactualEstimator` now provides two methods for off-policy evaluation, each optimized for different scenarios:

1. **`estimate_ope_metrics()`** - Loop-based (default, memory-efficient)
2. **`estimate_ope_metrics_vectorized()`** - Matrix-based (faster for large datasets)

## Performance Summary

Based on comprehensive benchmarking:
- **Average speedup**: 1.44x with vectorized method
- **Best case**: 1.67x speedup for large problems
- **Memory trade-off**: 32x more memory usage for vectorized
- **Statistical accuracy**: Both methods produce equivalent results

## Method Selection Guide

### Use `estimate_ope_metrics()` (Loop-based) When:
✅ **Small datasets**: `n_observed < 1,000`  
✅ **Memory constraints**: Limited available RAM  
✅ **Default choice**: Safe for all scenarios  
✅ **Production environments**: When memory usage is critical  

### Use `estimate_ope_metrics_vectorized()` When:
✅ **Large datasets**: `n_observed > 1,000`  
✅ **High bootstrap counts**: `n_bootstrap > 100`  
✅ **Performance critical**: Speed is more important than memory  
✅ **Sufficient memory**: Can handle ~8MB per 1K obs × 1K bootstrap  

## Automatic Warnings

Both methods include intelligent warnings to help you choose optimally:

### Vectorized Method Warnings:
- **Small dataset warning**: When `n_observed < 1,000`
- **Moderate memory warning**: When estimated memory > 5GB
- **High memory warning**: When estimated memory > 8GB (approaching 10GB limit)
- **Very large matrix warning**: When > 1.25B elements (~10GB)

### Loop-based Method Warnings:
- **Performance warning**: When vectorized could provide significant speedup
- **Memory consideration**: Includes estimated memory usage for comparison

## Usage Examples

### Basic Usage (Default)
```python
from counterfactual_fraud_model.config import CounterfactualEstimatorConfig
from counterfactual_fraud_model.estimators.counterfactual_estimator import CounterfactualEstimator

# Standard configuration
config = CounterfactualEstimatorConfig(n_bootstrap=1000, random_state=42)
estimator = CounterfactualEstimator(config, data)

# Default method (automatically chooses loop-based)
results = estimator.estimate_ope_metrics(new_actions, new_actions_proba)
```

### Optimized for Large Datasets
```python
# For large datasets with sufficient memory
if estimator.n_observed > 1000:
    # Use vectorized method for better performance
    results = estimator.estimate_ope_metrics_vectorized(new_actions, new_actions_proba)
else:
    # Use standard method for small datasets
    results = estimator.estimate_ope_metrics(new_actions, new_actions_proba)
```

### Memory-Conscious Configuration
```python
# For memory-constrained environments
config = CounterfactualEstimatorConfig(
    n_bootstrap=500,  # Reduce bootstrap count
    random_state=42
)

# Always use loop-based method
results = estimator.estimate_ope_metrics(new_actions, new_actions_proba)
```

### Performance-Optimized Configuration
```python
# For maximum performance (when memory allows)
config = CounterfactualEstimatorConfig(
    n_bootstrap=1000,
    random_state=42
)

# Use vectorized method for large problems
if estimator.n_observed > 1000:
    results = estimator.estimate_ope_metrics_vectorized(new_actions, new_actions_proba)
else:
    results = estimator.estimate_ope_metrics(new_actions, new_actions_proba)
```

## Memory Usage Estimation

To estimate memory usage for the vectorized method:

```python
# Rough memory estimation (in MB and GB)
estimated_memory_mb = n_observed * n_bootstrap * 8 / (1024 * 1024)
estimated_memory_gb = estimated_memory_mb / 1024

# Examples:
# 5,000 observations × 1,000 bootstrap = ~38MB
# 10,000 observations × 10,000 bootstrap = ~763MB
# 50,000 observations × 10,000 bootstrap = ~3.8GB
# 100,000 observations × 10,000 bootstrap = ~7.5GB (approaching limit)
```

## Warning Management

To suppress warnings if you know what you're doing:

```python
import warnings

# Suppress specific warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Using vectorized method with small dataset")
    results = estimator.estimate_ope_metrics_vectorized(new_actions, new_actions_proba)
```

## System Requirements

### BLAS Configuration
The optimization leverages numpy's BLAS backend:
- **Apple Silicon (M1/M2)**: Uses Accelerate Framework (excellent performance)
- **Intel/AMD**: Works with OpenBLAS, MKL, or ATLAS
- **Threading**: Automatic when supported by BLAS library

### Check Your Configuration
```python
import numpy as np
print(np.__config__.show())  # Shows BLAS configuration
```

## Troubleshooting

### Poor Vectorized Performance
If vectorized method is slower than expected:
1. Check BLAS configuration (`np.__config__.show()`)
2. Ensure threading is enabled (check environment variables)
3. Consider dataset size - very small datasets may not benefit

### Memory Issues
If running out of memory:
1. Reduce `n_bootstrap` in configuration
2. Use loop-based method instead
3. Process data in smaller chunks

### Warning Fatigue
If warnings are too frequent:
1. Review your usage patterns
2. Consider switching to the recommended method
3. Adjust warning thresholds if needed (advanced)

## Conclusion

The optimization provides meaningful performance improvements while maintaining backward compatibility and statistical accuracy. The intelligent warnings help ensure you're always using the optimal method for your specific use case.

**Recommendation**: Start with the default `estimate_ope_metrics()` and switch to `estimate_ope_metrics_vectorized()` when the warnings suggest it would be beneficial.