#!/usr/bin/env python3
"""
Multi-parameter simulation analysis for RefactoredSyntheticOffPolicyEvaluationPipeline.

This script runs multiple simulations with different exploration rates and model types,
generating comprehensive analysis including plots and summary tables.

Updated to use the new refactored pipeline implementation with SOLID principles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from sklearn.metrics import precision_recall_curve, auc
from sklearn.calibration import calibration_curve

from src.counterfactual_fraud_model import (
    PipelineFactory,
    SyntheticPipelineConfig,
    RefactoredSyntheticOffPolicyEvaluationPipeline
)


def run_exploration_rate_simulations(
    exploration_rates: np.ndarray, 
    cutoff: float = 0.05,
    model_type: str = "lightgbm",
    random_state: int = 42
) -> tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Run multiple simulations with different exploration rates using synthetic data.
    
    Args:
        exploration_rates: Array of exploration rate values to test
        cutoff: Fixed cutoff value for all simulations
        model_type: Model type to use for all simulations
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (list of simulation results, reference data for plots)
    """
    print(f"Running {len(exploration_rates)} simulations with {model_type} model using refactored pipeline...")
    
    # Create configuration using Pydantic
    config = SyntheticPipelineConfig(
        n_samples=100_000,
        n_features=30,
        n_informative=15,
        n_redundant=5,
        n_repeated=0,
        n_clusters_per_class=2,
        weights=[0.985, 0.015],
        flip_y=0.01,
        class_sep=1.0,
        model_type=model_type,
        model_params={},
        test_size=0.3,
        n_bootstrap=5000,
        random_state=random_state
    )
    
    # Initialize pipeline using improved factory pattern - pass entire config!
    pipeline = PipelineFactory.create_synthetic_pipeline(config)
    
    # Get reference data (same for all simulations since we use same data generation params)
    reference_data = pipeline.generated_data
    
    results = []
    
    for i, exploration_rate in enumerate(exploration_rates):
        print(f"Running simulation {i+1}/{len(exploration_rates)} with exploration_rate={exploration_rate:.3f}")
        
        # Run pipeline with current exploration rate
        # The new pipeline uses Pydantic validation automatically
        result = pipeline.run_pipeline(
            cutoff=cutoff,
            exploration_rate=exploration_rate,
            include_data=False  # Don't include data to save memory
        )
        
        # Add exploration_rate to result for easy tracking
        result['parameters']['exploration_rate'] = exploration_rate
        results.append(result)
    
    return results, reference_data


def run_model_comparison_simulations(
    model_types: List[str],
    cutoff: float = 0.05,
    exploration_rate: float = 0.05,
    random_state: int = 42
) -> tuple[List[Dict[str, Any]], Dict[str, pd.DataFrame]]:
    """
    Run simulations comparing different model types.
    
    Args:
        model_types: List of model types to compare
        cutoff: Fixed cutoff value for all simulations
        exploration_rate: Fixed exploration rate for all simulations
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (list of simulation results, dict of reference data by model type)
    """
    print(f"Running model comparison with {len(model_types)} models using refactored pipeline...")
    
    results = []
    reference_data = {}
    
    for i, model_type in enumerate(model_types):
        print(f"Running simulation {i+1}/{len(model_types)} with {model_type} model...")
        
        # Create configuration using Pydantic
        config = SyntheticPipelineConfig(
            n_samples=100_000,
            n_features=30,
            n_informative=15,
            n_redundant=5,
            model_type=model_type,
            test_size=0.3,
            random_state=random_state
        )
        
        # Initialize pipeline using improved factory pattern - pass entire config!
        pipeline = PipelineFactory.create_synthetic_pipeline(config)
        
        # Get reference data for this model
        reference_data[model_type] = pipeline.generated_data
        
        # Run pipeline
        result = pipeline.run_pipeline(
            cutoff=cutoff,
            exploration_rate=exploration_rate,
            include_data=False
        )
        
        # Add model_type to result for easy tracking
        result['parameters']['model_type'] = model_type
        results.append(result)
    
    return results, reference_data


def plot_model_scores_histogram(data: pd.DataFrame, title_suffix: str = "") -> None:
    """Plot histogram of model scores."""
    plt.figure(figsize=(10, 6))
    plt.hist(data['model_scores'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Model Scores')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Model Scores{title_suffix} (Refactored Pipeline)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_model_scores_comparison(reference_data: Dict[str, pd.DataFrame]) -> None:
    """Plot comparison of model scores across different models."""
    plt.figure(figsize=(15, 5))
    
    n_models = len(reference_data)
    
    for i, (model_type, data) in enumerate(reference_data.items()):
        plt.subplot(1, n_models, i+1)
        plt.hist(data['model_scores'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Model Scores')
        plt.ylabel('Frequency')
        plt.title(f'{model_type.replace("_", " ").title()} Model\n(Refactored Pipeline)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def create_synthetic_data_summary_table(data: pd.DataFrame, pipeline_params: Dict[str, Any]) -> pd.DataFrame:
    """Create summary table with synthetic data statistics and generation parameters."""
    total_transactions = len(data)
    true_fraud_rate = data['is_fraud'].mean()
    
    summary_data = {
        'Metric': [
            'Total Transactions', 'True Fraud Rate', 'N Samples', 'N Features', 
            'N Informative', 'N Redundant', 'Model Type', 'Test Size', 'Class Sep', 'Pipeline Type'
        ],
        'Value': [
            total_transactions,
            f"{true_fraud_rate:.4f}",
            pipeline_params['n_samples'],
            pipeline_params['n_features'], 
            pipeline_params['n_informative'],
            pipeline_params['n_redundant'],
            pipeline_params['model_type'],
            pipeline_params['test_size'],
            pipeline_params['class_sep'],
            'Refactored (SOLID Principles)'
        ]
    }
    
    return pd.DataFrame(summary_data)


def plot_calibration_curve(data: pd.DataFrame, title_suffix: str = "") -> None:
    """Plot calibration curve for model scores vs fraud values."""
    plt.figure(figsize=(8, 8))
    
    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        data['is_fraud'], data['model_scores'], n_bins=50
    )
    
    # Plot calibration curve
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model", linewidth=2)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Plot{title_suffix} (Refactored Pipeline)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(data: pd.DataFrame, title_suffix: str = "") -> None:
    """Plot precision-recall curve for model scores vs fraud values."""
    plt.figure(figsize=(8, 6))
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(data['is_fraud'], data['model_scores'])
    auc_score = auc(recall, precision)
    
    # Plot precision-recall curve
    plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {auc_score:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve{title_suffix} (Refactored Pipeline)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_policy_metrics_table(results: List[Dict[str, Any]], comparison_key: str = 'exploration_rate') -> pd.DataFrame:
    """Create table with policy metrics for each comparison parameter."""
    table_data = []
    
    for result in results:
        comparison_value = result['parameters'][comparison_key]
        stats = result['statistics']
        
        row_data = {
            comparison_key: comparison_value,
            'policy_approval_rate': stats['policy_approval_rate'],
            'policy_fraud_rate': stats['policy_fraud_rate'],
            'approval_rate_increase': stats['approval_rate_increase'],
            'fraud_rate_increase': stats['fraud_rate_increase']
        }
        
        # Add model performance metrics if available
        if 'model_performance' in result:
            model_perf = result['model_performance']
            row_data.update({
                'model_roc_auc': model_perf['roc_auc'],
                'model_precision': model_perf['precision'],
                'model_recall': model_perf['recall'],
                'model_f1': model_perf['f1_score']
            })
        
        table_data.append(row_data)
    
    return pd.DataFrame(table_data)


def plot_ope_metrics(results: List[Dict[str, Any]], comparison_key: str = 'exploration_rate') -> None:
    """Plot OPE metrics vs comparison parameter with confidence intervals."""
    # Extract comparison values
    comparison_values = [result['parameters'][comparison_key] for result in results]
    
    # Get all available metrics from first result
    metrics = list(results[0]['ope_metrics'].keys())
    
    # Create subplots
    n_metrics = len(metrics)
    cols = 2
    rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Extract metric values
        means = [result['ope_metrics'][metric]['mean'] for result in results]
        p025 = [result['ope_metrics'][metric]['p025'] for result in results]
        p975 = [result['ope_metrics'][metric]['p975'] for result in results]
        
        # Plot mean line
        ax.plot(comparison_values, means, 'o-', linewidth=2, markersize=6, label='Mean')
        
        # Plot confidence interval
        ax.fill_between(comparison_values, p025, p975, alpha=0.3, label='95% CI')
        
        ax.set_xlabel(comparison_key.replace('_', ' ').title())
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'OPE Metric: {metric.replace("_", " ").title()} (Refactored)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots if any
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_model_performance_comparison(results: List[Dict[str, Any]]) -> None:
    """Plot model performance metrics comparison."""
    if not results or 'model_performance' not in results[0]:
        print("No model performance data available for plotting.")
        return
    
    # Extract data for plotting
    model_types = [result['parameters']['model_type'] for result in results]
    
    metrics = ['roc_auc', 'precision', 'recall', 'f1_score']
    metric_data = {}
    
    for metric in metrics:
        metric_data[metric] = [result['model_performance'][metric] for result in results]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(model_types, metric_data[metric])
        ax.set_title(f'Model {metric.replace("_", " ").title()} (Refactored Pipeline)')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_xlabel('Model Type')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_data[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demonstrate_pydantic_features(config: SyntheticPipelineConfig) -> None:
    """Demonstrate Pydantic configuration features for synthetic pipeline."""
    print("\n" + "=" * 60)
    print("PYDANTIC CONFIGURATION FEATURES")
    print("=" * 60)
    
    # Create pipeline using improved factory pattern
    pipeline = PipelineFactory.create_synthetic_pipeline(config)
    
    # Get configuration as dictionary
    config_dict = pipeline.get_params()
    print(f"✅ Configuration extracted: {len(config_dict)} parameters")
    
    # Show JSON serialization
    import json
    config_json = json.dumps(config_dict, indent=2)
    print(f"✅ JSON serialization available ({len(config_json)} characters)")
    print("  Sample JSON:")
    print("  " + config_json[:300] + "...")
    
    # Show configuration validation worked
    print(f"✅ Validation passed for:")
    print(f"  - N samples: {config_dict['n_samples']} (must be > 0)")
    print(f"  - N features: {config_dict['n_features']} (must be > 0)")
    print(f"  - N informative: {config_dict['n_informative']} (must be <= n_features)")
    print(f"  - Model type: {config_dict['model_type']} (must be valid)")
    print(f"  - Test size: {config_dict['test_size']} (must be 0-1)")
    
    # Show model performance features
    try:
        model_perf = pipeline.get_model_performance()
        print(f"✅ Model performance available: {len(model_perf)} metrics")
        print(f"  - ROC AUC: {model_perf.get('roc_auc', 'N/A'):.3f}")
        print(f"  - F1 Score: {model_perf.get('f1_score', 'N/A'):.3f}")
    except:
        print("⚠️  Model performance not yet available (data not generated)")
    
    # Demonstrate config object benefits
    print(f"✅ Configuration object benefits:")
    print(f"  - Direct config → factory: PipelineFactory.create_synthetic_pipeline(config)")
    print(f"  - No parameter duplication or unpacking needed")
    print(f"  - Type safety and validation at config creation time")
    print(f"  - Easy to extend with new parameters")


def run_exploration_rate_analysis(exploration_rates: np.ndarray, model_type: str = "lightgbm"):
    """Run complete exploration rate analysis."""
    print(f"\nExploration Rate Analysis with {model_type} model")
    print("Using REFACTORED Synthetic Pipeline")
    print("=" * 60)
    
    # Run simulations
    results, reference_data = run_exploration_rate_simulations(exploration_rates, model_type=model_type)
    
    # Get pipeline parameters for summary table
    pipeline_params = results[0]['parameters']
    
    # Demonstrate configuration features using config object directly
    config = SyntheticPipelineConfig(
        n_samples=pipeline_params['n_samples'],
        n_features=pipeline_params['n_features'],
        n_informative=pipeline_params['n_informative'],
        n_redundant=pipeline_params['n_redundant'],
        n_repeated=pipeline_params['n_repeated'],
        n_clusters_per_class=pipeline_params['n_clusters_per_class'],
        weights=pipeline_params['weights'],
        flip_y=pipeline_params['flip_y'],
        class_sep=pipeline_params['class_sep'],
        model_type=pipeline_params['model_type'],
        model_params=pipeline_params['model_params'],
        test_size=pipeline_params['test_size'],
        n_bootstrap=pipeline_params['n_bootstrap'],
        random_state=pipeline_params.get('random_state')
    )
    demonstrate_pydantic_features(config)
    
    print("\nGenerating visualizations and tables...")
    print("=" * 60)
    
    # 1. Plot model scores histogram
    print("1. Generating model scores histogram...")
    plot_model_scores_histogram(reference_data, f" ({model_type.replace('_', ' ').title()})")
    
    # 2. Create and display data summary table
    print("2. Creating data summary table...")
    summary_table = create_synthetic_data_summary_table(reference_data, pipeline_params)
    print("\nSynthetic Data Summary Table:")
    print(summary_table.to_string(index=False))
    
    # 3. Plot calibration curve
    print("\n3. Generating calibration plot...")
    plot_calibration_curve(reference_data, f" ({model_type.replace('_', ' ').title()})")
    
    # 4. Plot precision-recall curve
    print("4. Generating precision-recall curve...")
    plot_precision_recall_curve(reference_data, f" ({model_type.replace('_', ' ').title()})")
    
    # 5. Create and display policy metrics table
    print("5. Creating policy metrics table...")
    policy_table = create_policy_metrics_table(results, 'exploration_rate')
    print("\nPolicy Metrics Table:")
    print(policy_table.round(4).to_string(index=False))
    
    # 6. Plot OPE metrics
    print("\n6. Generating OPE metrics plots...")
    plot_ope_metrics(results, 'exploration_rate')


def run_model_comparison_analysis(model_types: List[str]):
    """Run complete model comparison analysis."""
    print(f"\nModel Comparison Analysis")
    print("Using REFACTORED Synthetic Pipeline")
    print("=" * 60)
    
    # Run simulations
    results, reference_data = run_model_comparison_simulations(model_types)
    
    print("\nGenerating visualizations and tables...")
    print("=" * 60)
    
    # 1. Plot model scores comparison
    print("1. Generating model scores comparison...")
    plot_model_scores_comparison(reference_data)
    
    # 2. Create and display model comparison table
    print("2. Creating model comparison table...")
    comparison_table = create_policy_metrics_table(results, 'model_type')
    print("\nModel Comparison Table:")
    print(comparison_table.round(4).to_string(index=False))
    
    # 3. Plot model performance comparison
    print("\n3. Generating model performance comparison...")
    plot_model_performance_comparison(results)
    
    # 4. Plot calibration curves for each model
    print("4. Generating calibration plots for each model...")
    for model_type, data in reference_data.items():
        plot_calibration_curve(data, f" ({model_type.replace('_', ' ').title()} Model)")
    
    # 5. Plot precision-recall curves for each model
    print("5. Generating precision-recall curves for each model...")
    for model_type, data in reference_data.items():
        plot_precision_recall_curve(data, f" ({model_type.replace('_', ' ').title()} Model)")


def main():
    """Main function to run simulations and generate all plots and tables."""
    print("Starting Synthetic Multi-Parameter Simulation Analysis")
    print("Using REFACTORED Pipeline with SOLID Principles")
    print("=" * 60)
    
    # Configuration for exploration rate analysis
    exploration_rates = np.linspace(0.01, 0.1, 10)
    
    # Configuration for model comparison
    model_types = ["lightgbm", "random_forest", "logistic"]
    
    print(f"\nConfiguration:")
    print(f"  - Exploration rates: {len(exploration_rates)} values from {exploration_rates[0]:.3f} to {exploration_rates[-1]:.3f}")
    print(f"  - Model types: {', '.join(model_types)}")
    print(f"  - Using: RefactoredSyntheticOffPolicyEvaluationPipeline")
    print(f"  - Factory pattern: PipelineFactory")
    print(f"  - Configuration: SyntheticPipelineConfig (Pydantic)")
    
    # Run exploration rate analysis with LightGBM
    run_exploration_rate_analysis(exploration_rates, model_type="lightgbm")
    
    # Run model comparison analysis
    run_model_comparison_analysis(model_types)
    
    print("\nAll refactored synthetic pipeline analyses complete!")
    print("=" * 60)
    print("\n🎯 Benefits Demonstrated:")
    print("  ✅ SOLID principles compliance")
    print("  ✅ Pydantic validation and serialization for synthetic data")
    print("  ✅ Factory pattern for pipeline creation")
    print("  ✅ Template Method pattern for consistent workflow")
    print("  ✅ Strategy pattern for ML model data generation")
    print("  ✅ Model performance tracking and comparison")
    print("  ✅ Enhanced configuration management")
    print("  ✅ Improved maintainability and extensibility")


if __name__ == "__main__":
    main() 