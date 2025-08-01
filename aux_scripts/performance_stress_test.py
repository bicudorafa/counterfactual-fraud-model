#!/usr/bin/env python3
"""
Comprehensive performance stress test for CounterfactualEstimator optimizations.

This script compares the original loop-based implementation with the vectorized
matrix-based implementation to assess:
1. Performance improvements/degradation
2. Memory usage differences  
3. Correctness (statistical similarity)
4. Impact of BLAS threading

The script tests various scenarios:
- Different dataset sizes (n_observed)
- Different bootstrap counts
- Different number of threads
- Different metric configurations
"""

import time
import psutil
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import our estimator
from counterfactual_fraud_model.config import CounterfactualEstimatorConfig
from counterfactual_fraud_model.estimators.counterfactual_estimator import CounterfactualEstimator


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    method: str
    n_observed: int
    n_bootstrap: int
    n_threads: int
    execution_time: float
    memory_peak_mb: float
    memory_increase_mb: float
    results_dict: Dict
    

class MemoryMonitor:
    """Monitor memory usage during execution."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.peak_memory = 0
        self.initial_memory = 0
        self.monitoring = False
        self.monitor_thread = None
        
    def start(self):
        """Start monitoring memory usage."""
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        """Stop monitoring and return peak memory usage."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        return self.peak_memory, self.peak_memory - self.initial_memory
        
    def _monitor(self):
        """Internal monitoring loop."""
        while self.monitoring:
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = max(self.peak_memory, current_memory)
            time.sleep(0.01)  # Monitor every 10ms


def create_synthetic_data(n_observed: int, fraud_rate: float = 0.05) -> pd.DataFrame:
    """Create synthetic data for testing."""
    np.random.seed(42)  # For reproducibility
    
    # Generate synthetic fraud data
    is_fraud = np.random.binomial(1, fraud_rate, n_observed)
    
    # Model scores (higher for fraud)
    model_scores = np.random.beta(2, 5, n_observed)
    model_scores[is_fraud == 1] += np.random.beta(3, 2, np.sum(is_fraud))
    model_scores = np.clip(model_scores, 0, 1)
    
    # Policy actions (allow all for simplicity)
    policy_action = ['allow'] * n_observed
    
    # Model actions based on scores
    model_action = ['block' if score > 0.5 else 'allow' for score in model_scores]
    
    # Propensity scores (all allowed, so propensity = 1)
    propensity_score = np.ones(n_observed)
    
    return pd.DataFrame({
        'is_fraud': is_fraud,
        'model_scores': model_scores,
        'propensity_score': propensity_score,
        'model_action': model_action,
        'policy_action': policy_action
    })


def run_benchmark(estimator: CounterfactualEstimator, 
                 method: str,
                 new_actions: np.ndarray,
                 new_actions_proba: np.ndarray,
                 n_bootstrap: int,
                 n_threads: int) -> BenchmarkResult:
    """Run a single benchmark test."""
    
    # Set number of threads (for BLAS libraries that support it)
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.environ['MKL_NUM_THREADS'] = str(n_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)
    
    # Update estimator config
    original_n_bootstrap = estimator.config.n_bootstrap
    estimator.config.n_bootstrap = n_bootstrap
    
    # Initialize memory monitor
    memory_monitor = MemoryMonitor()
    
    try:
        # Start monitoring
        memory_monitor.start()
        
        # Choose method and measure execution time
        start_time = time.perf_counter()
        
        if method == 'loop':
            results = estimator._estimate_ope_metrics_loop_based(new_actions, new_actions_proba)
        elif method == 'vectorized':
            results = estimator.estimate_ope_metrics_vectorized(new_actions, new_actions_proba)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        end_time = time.perf_counter()
        
        # Stop monitoring
        peak_memory, memory_increase = memory_monitor.stop()
        
        execution_time = end_time - start_time
        
        return BenchmarkResult(
            method=method,
            n_observed=len(new_actions),
            n_bootstrap=n_bootstrap,
            n_threads=n_threads,
            execution_time=execution_time,
            memory_peak_mb=peak_memory,
            memory_increase_mb=memory_increase,
            results_dict=results
        )
        
    finally:
        # Restore original config
        estimator.config.n_bootstrap = original_n_bootstrap


def check_statistical_similarity(result1: Dict, result2: Dict, tolerance: float = 0.05) -> bool:
    """Check if two results are statistically similar."""
    
    for metric_name in result1.keys():
        if metric_name not in result2:
            return False
            
        # Compare means
        mean1 = result1[metric_name]['mean']
        mean2 = result2[metric_name]['mean']
        
        if abs(mean1 - mean2) > tolerance:
            return False
            
        # Compare confidence intervals overlap
        ci1_low = result1[metric_name]['p025']
        ci1_high = result1[metric_name]['p975']
        ci2_low = result2[metric_name]['p025']
        ci2_high = result2[metric_name]['p975']
        
        # Check if confidence intervals overlap
        if ci1_high < ci2_low or ci2_high < ci1_low:
            return False
    
    return True


def run_comprehensive_benchmark() -> List[BenchmarkResult]:
    """Run comprehensive benchmark across different scenarios."""
    
    print("Starting comprehensive performance benchmark...")
    print("=" * 60)
    
    # Test scenarios
    test_scenarios = [
        # (n_observed, n_bootstrap, n_threads)
        (1000, 100, 1),      # Small problem, single thread
        (1000, 1000, 1),     # Small problem, many bootstrap, single thread  
        (5000, 100, 1),      # Medium problem, single thread
        (5000, 1000, 1),     # Medium problem, many bootstrap, single thread
        (10000, 100, 1),     # Large problem, single thread
        (10000, 1000, 1),    # Large problem, many bootstrap, single thread
        (1000, 100, 4),      # Small problem, multi-thread
        (5000, 1000, 4),     # Medium problem, many bootstrap, multi-thread  
        (10000, 1000, 4),    # Large problem, many bootstrap, multi-thread
        (50000, 100, 1),     # Very large problem, single thread (if memory allows)
    ]
    
    results = []
    
    for i, (n_observed, n_bootstrap, n_threads) in enumerate(test_scenarios):
        print(f"\nTest {i+1}/{len(test_scenarios)}: n_observed={n_observed}, n_bootstrap={n_bootstrap}, n_threads={n_threads}")
        
        try:
            # Create synthetic data
            data = create_synthetic_data(n_observed)
            config = CounterfactualEstimatorConfig(n_bootstrap=n_bootstrap, random_state=42)
            estimator = CounterfactualEstimator(config, data)
            
            # Create new actions for testing
            np.random.seed(42)
            new_actions = np.random.binomial(1, 0.3, n_observed)  # 30% block rate
            new_actions_proba = np.random.beta(2, 5, n_observed)
            
            # Test both methods
            for method in ['loop', 'vectorized']:
                print(f"  Running {method} method...", end=' ', flush=True)
                
                try:
                    result = run_benchmark(
                        estimator, method, new_actions, new_actions_proba, 
                        n_bootstrap, n_threads
                    )
                    results.append(result)
                    print(f"✓ {result.execution_time:.3f}s, {result.memory_increase_mb:.1f}MB")
                    
                except Exception as e:
                    print(f"✗ Error: {e}")
                    continue
            
            # Check correctness
            loop_results = [r for r in results if r.method == 'loop' and 
                          r.n_observed == n_observed and r.n_bootstrap == n_bootstrap and r.n_threads == n_threads]
            vec_results = [r for r in results if r.method == 'vectorized' and 
                         r.n_observed == n_observed and r.n_bootstrap == n_bootstrap and r.n_threads == n_threads]
            
            if loop_results and vec_results:
                similar = check_statistical_similarity(
                    loop_results[-1].results_dict, 
                    vec_results[-1].results_dict
                )
                print(f"  Statistical similarity: {'✓' if similar else '✗'}")
                
        except Exception as e:
            print(f"  ✗ Scenario failed: {e}")
            continue
    
    return results


def analyze_results(results: List[BenchmarkResult]) -> None:
    """Analyze and visualize benchmark results."""
    
    if not results:
        print("No results to analyze!")
        return
        
    print("\n" + "=" * 60)
    print("BENCHMARK ANALYSIS")
    print("=" * 60)
    
    # Convert to DataFrame for easier analysis
    df_data = []
    for r in results:
        df_data.append({
            'method': r.method,
            'n_observed': r.n_observed,
            'n_bootstrap': r.n_bootstrap,
            'n_threads': r.n_threads,
            'execution_time': r.execution_time,
            'memory_increase_mb': r.memory_increase_mb
        })
    
    df = pd.DataFrame(df_data)
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 30)
    
    summary = df.groupby('method').agg({
        'execution_time': ['mean', 'std', 'min', 'max'],
        'memory_increase_mb': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print(summary)
    
    # Performance comparison
    print("\nPERFORMANCE COMPARISON:")
    print("-" * 30)
    
    # Group by test configuration to compare methods
    comparison_data = []
    
    for (n_obs, n_boot, n_threads), group in df.groupby(['n_observed', 'n_bootstrap', 'n_threads']):
        if len(group) == 2:  # Both methods present
            loop_time = group[group['method'] == 'loop']['execution_time'].iloc[0]
            vec_time = group[group['method'] == 'vectorized']['execution_time'].iloc[0]
            
            loop_mem = group[group['method'] == 'loop']['memory_increase_mb'].iloc[0]
            vec_mem = group[group['method'] == 'vectorized']['memory_increase_mb'].iloc[0]
            
            speedup = loop_time / vec_time
            memory_ratio = vec_mem / loop_mem if loop_mem > 0 else float('inf')
            
            comparison_data.append({
                'n_observed': n_obs,
                'n_bootstrap': n_boot,
                'n_threads': n_threads,
                'speedup': speedup,
                'memory_ratio': memory_ratio,
                'loop_time': loop_time,
                'vec_time': vec_time
            })
    
    comp_df = pd.DataFrame(comparison_data)
    
    if not comp_df.empty:
        print(f"Average speedup (vectorized/loop): {comp_df['speedup'].mean():.2f}x")
        print(f"Best speedup: {comp_df['speedup'].max():.2f}x")
        print(f"Worst speedup: {comp_df['speedup'].min():.2f}x")
        print(f"Average memory increase ratio: {comp_df['memory_ratio'].mean():.2f}x")
        
        print("\nDETAILED COMPARISON:")
        print(comp_df.to_string(index=False))
    
    # Create visualizations
    create_visualizations(df, comp_df)


def create_visualizations(df: pd.DataFrame, comp_df: pd.DataFrame) -> None:
    """Create performance visualization plots."""
    
    if df.empty:
        return
        
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CounterfactualEstimator Performance Analysis', fontsize=16)
    
    # 1. Execution time comparison
    ax1 = axes[0, 0]
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        ax1.scatter(method_data['n_observed'], method_data['execution_time'], 
                   label=method, alpha=0.7, s=60)
    ax1.set_xlabel('Number of Observations')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Execution Time vs Problem Size')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 2. Memory usage comparison
    ax2 = axes[0, 1]
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        ax2.scatter(method_data['n_observed'], method_data['memory_increase_mb'],
                   label=method, alpha=0.7, s=60)
    ax2.set_xlabel('Number of Observations')
    ax2.set_ylabel('Memory Increase (MB)')
    ax2.set_title('Memory Usage vs Problem Size')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 3. Speedup analysis
    if not comp_df.empty:
        ax3 = axes[1, 0]
        scatter = ax3.scatter(comp_df['n_observed'], comp_df['speedup'], 
                            c=comp_df['n_bootstrap'], s=80, alpha=0.7, cmap='viridis')
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
        ax3.set_xlabel('Number of Observations')
        ax3.set_ylabel('Speedup (vectorized/loop)')
        ax3.set_title('Speedup vs Problem Size')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Bootstrap Count')
    
    # 4. Bootstrap scaling
    ax4 = axes[1, 1]
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        ax4.scatter(method_data['n_bootstrap'], method_data['execution_time'],
                   label=method, alpha=0.7, s=60)
    ax4.set_xlabel('Number of Bootstrap Samples')
    ax4.set_ylabel('Execution Time (seconds)')
    ax4.set_title('Execution Time vs Bootstrap Count')
    ax4.legend()
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'performance_analysis.png'")


def check_numpy_blas_info():
    """Check numpy BLAS configuration."""
    print("\nNUMPY BLAS CONFIGURATION:")
    print("-" * 30)
    
    try:
        import numpy as np
        config = np.__config__.show()
        print("NumPy configuration:")
        print(config)
    except Exception as e:
        print(f"Could not get NumPy config: {e}")
    
    # Check thread settings
    print("\nTHREAD ENVIRONMENT VARIABLES:")
    thread_vars = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']
    for var in thread_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")


def main():
    """Main benchmark execution."""
    
    print("COUNTERFACTUAL ESTIMATOR PERFORMANCE OPTIMIZATION ANALYSIS")
    print("=" * 70)
    
    # Check system info
    check_numpy_blas_info()
    
    # Run benchmark
    results = run_comprehensive_benchmark()
    
    # Analyze results
    analyze_results(results)
    
    # Conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    if results:
        # Calculate overall statistics
        df = pd.DataFrame([{
            'method': r.method,
            'execution_time': r.execution_time,
            'memory_increase_mb': r.memory_increase_mb
        } for r in results])
        
        loop_results = df[df['method'] == 'loop']
        vec_results = df[df['method'] == 'vectorized']
        
        if not loop_results.empty and not vec_results.empty:
            avg_loop_time = loop_results['execution_time'].mean()
            avg_vec_time = vec_results['execution_time'].mean()
            avg_speedup = avg_loop_time / avg_vec_time
            
            avg_loop_mem = loop_results['memory_increase_mb'].mean()
            avg_vec_mem = vec_results['memory_increase_mb'].mean()
            
            print(f"Average performance improvement: {avg_speedup:.2f}x")
            print(f"Average memory usage change: {avg_vec_mem/avg_loop_mem:.2f}x")
            
            if avg_speedup > 1.1:
                print("✓ RECOMMENDATION: Use vectorized implementation for better performance")
            elif avg_speedup < 0.9:
                print("✗ RECOMMENDATION: Stick with loop-based implementation")
            else:
                print("~ RECOMMENDATION: Performance difference is marginal")
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()