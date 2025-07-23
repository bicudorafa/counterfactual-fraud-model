#!/usr/bin/env python3
"""
Multi-parameter simulation analysis for OffPolicyEvaluationPipeline.

This script runs multiple simulations with different exploration_rate values
and generates comprehensive analysis including plots and summary tables.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from sklearn.metrics import precision_recall_curve, auc
from sklearn.calibration import calibration_curve

from counterfactual_fraud_model import (
    OffPolicyEvaluationPipeline,
    OffPolicyEvaluationConfig,
    DataGeneratorConfig,
    LoggingPolicyConfig,
    CounterfactualEstimatorConfig,
    PipelineConfig
)


def run_simulations(exploration_rates: np.ndarray, 
                   cutoff: float = 0.05,
                   random_state: int = 42) -> tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Run multiple OffPolicyEvaluationPipeline simulations with different exploration rates.
    
    Args:
        exploration_rates: Array of exploration rate values to test
        cutoff: Fixed cutoff value for all simulations
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (list of simulation results, reference data for plots)
    """
    print(f"Running {len(exploration_rates)} simulations...")
    
    # Create configuration for the pipeline
    config = OffPolicyEvaluationConfig(
        data_generator=DataGeneratorConfig(
            # HACK: erase later
            sample_size=20_000,
            random_state=random_state
        ),
        logging_policy=LoggingPolicyConfig(
            cutoff=cutoff,
            exploration_rate=0.05,  # Default, will be overridden in loop
            random_state=random_state
        ),
        counterfactual_estimator=CounterfactualEstimatorConfig(random_state=random_state),
        pipeline=PipelineConfig(include_data=False)
    )
    
    # Initialize pipeline with configuration
    pipeline = OffPolicyEvaluationPipeline(config)
    
    # Get reference data (same for all simulations since we use same data generation params)
    reference_data = pipeline._get_or_generate_data()
    
    results = []
    
    for i, exploration_rate in enumerate(exploration_rates):
        print(f"Running simulation {i+1}/{len(exploration_rates)} with exploration_rate={exploration_rate:.3f}")
        
        # Run pipeline with current exploration rate (overrides config)
        result = pipeline.run_pipeline(
            # TODO: review this funcionality. It doesn't make sense to have a cutoff and exploration rate as config if they can be changed in the run_pipeline method.
            cutoff=cutoff,
            exploration_rate=exploration_rate,
            include_data=False  # Don't include data to save memory
        )
        
        # Add exploration_rate to result for easy tracking
        result['parameters']['exploration_rate'] = exploration_rate
        results.append(result)
    
    return results, reference_data


def plot_model_scores_histogram(data: pd.DataFrame) -> None:
    """Plot histogram of model scores."""
    plt.figure(figsize=(10, 6))
    plt.hist(data['model_scores'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Model Scores')
    plt.ylabel('Frequency')
    plt.title('Distribution of Model Scores')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_data_summary_table(data: pd.DataFrame, pipeline_params: Dict[str, Any]) -> pd.DataFrame:
    """Create summary table with data statistics and generation parameters."""
    total_transactions = len(data)
    true_fraud_rate = data['is_fraud'].mean()
    
    # Extract data generator parameters from nested structure
    data_gen_params = pipeline_params['data_generator']
    
    summary_data = {
        'Metric': ['Total Transactions', 'True Fraud Rate', 'Alpha', 'Beta', 'Mean', 'SD', 'Sample Size'],
        'Value': [
            total_transactions,
            f"{true_fraud_rate:.4f}",
            data_gen_params['alpha'],
            data_gen_params['beta_param'], 
            data_gen_params['mean'],
            data_gen_params['sd'],
            data_gen_params['sample_size']
        ]
    }
    
    return pd.DataFrame(summary_data)


def plot_calibration_curve(data: pd.DataFrame) -> None:
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
    plt.title('Calibration Plot (Reliability Diagram)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(data: pd.DataFrame) -> None:
    """Plot precision-recall curve for model scores vs fraud values."""
    plt.figure(figsize=(8, 6))
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(data['is_fraud'], data['model_scores'])
    auc_score = auc(recall, precision)
    
    # Plot precision-recall curve
    plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {auc_score:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_policy_metrics_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create table with policy metrics for each exploration rate."""
    table_data = []
    
    for result in results:
        exploration_rate = result['parameters']['exploration_rate']
        stats = result['statistics']
        
        table_data.append({
            'exploration_rate': exploration_rate,
            'allow_rate': stats['allow_rate'],
            'block_rate': stats['block_rate'],
            'fraud_rate_overall': stats['fraud_rate_overall'],
            'fraud_rate_allowed': stats['fraud_rate_allowed'],
            'total_transactions': stats['total_transactions'],
            'allowed_transactions': stats['allowed_transactions'],
            'blocked_transactions': stats['blocked_transactions']
        })
    
    return pd.DataFrame(table_data)


def plot_ope_metrics(results: List[Dict[str, Any]]) -> None:
    """Plot OPE metrics vs exploration rate with confidence intervals."""
    # Extract exploration rates
    exploration_rates = [result['parameters']['exploration_rate'] for result in results]
    
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
        
        # Extract metric values (updated for new structure)
        estimates = [result['ope_metrics'][metric]['mean'] for result in results]
        ci_lower = [result['ope_metrics'][metric]['p025'] for result in results]
        ci_upper = [result['ope_metrics'][metric]['p975'] for result in results]
        
        # Plot estimate line
        ax.plot(exploration_rates, estimates, 'o-', linewidth=2, markersize=6, label='Estimate')
        
        # Plot confidence interval
        ax.fill_between(exploration_rates, ci_lower, ci_upper, alpha=0.3, label='95% CI')
        
        ax.set_xlabel('Exploration Rate')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'OPE Metric: {metric.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots if any
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run simulations and generate all plots and tables."""
    print("Starting Multi-Parameter Simulation Analysis")
    print("=" * 50)
    
    # Configuration
    exploration_rates = np.linspace(0.01, 0.1, 10)
    cutoff = 0.05
    random_state = 42
    
    # Run simulations
    results, reference_data = run_simulations(exploration_rates, cutoff, random_state)
    
    # Get pipeline parameters for summary table
    pipeline_params = results[0]['parameters']
    
    print("\nGenerating visualizations and tables...")
    print("=" * 50)
    
    # 1. Plot model scores histogram
    print("1. Generating model scores histogram...")
    plot_model_scores_histogram(reference_data)
    
    # 2. Create and display data summary table
    print("2. Creating data summary table...")
    summary_table = create_data_summary_table(reference_data, pipeline_params)
    print("\nData Summary Table:")
    print(summary_table.to_string(index=False))
    
    # 3. Plot calibration curve
    print("\n3. Generating calibration plot...")
    plot_calibration_curve(reference_data)
    
    # 4. Plot precision-recall curve
    print("4. Generating precision-recall curve...")
    plot_precision_recall_curve(reference_data)
    
    # 5. Create and display policy metrics table
    print("5. Creating policy metrics table...")
    policy_table = create_policy_metrics_table(results)
    print("\nPolicy Metrics Table:")
    print(policy_table.round(4).to_string(index=False))
    
    # 6. Plot OPE metrics
    print("\n6. Generating OPE metrics plots...")
    plot_ope_metrics(results)
    
    print("\nAnalysis complete!")
    print("=" * 50)


if __name__ == "__main__":
    main() 