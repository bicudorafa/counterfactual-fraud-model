#!/usr/bin/env python3
"""
Common analysis functions shared across different simulation scripts.

This module contains plotting and analysis functions that are used by multiple
simulation analysis scripts, avoiding code duplication while maintaining
flexibility for different use cases.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import precision_recall_curve, auc
from sklearn.calibration import calibration_curve


def plot_model_scores_histogram(
    data: pd.DataFrame, 
    title_suffix: str = "",
    figsize: Tuple[int, int] = (10, 6),
    bins: int = 50,
    alpha: float = 0.7
) -> None:
    """
    Plot histogram of model scores.
    
    Args:
        data: DataFrame containing 'model_scores' column
        title_suffix: Optional suffix to add to the title
        figsize: Figure size tuple
        bins: Number of histogram bins
        alpha: Transparency level
    """
    plt.figure(figsize=figsize)
    plt.hist(data['model_scores'], bins=bins, alpha=alpha, edgecolor='black')
    plt.xlabel('Model Scores')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Model Scores{title_suffix}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_calibration_curve(
    data: pd.DataFrame, 
    title_suffix: str = "",
    figsize: Tuple[int, int] = (8, 8),
    n_bins: int = 50
) -> None:
    """
    Plot calibration curve for model scores vs fraud values.
    
    Args:
        data: DataFrame containing 'model_scores' and 'is_fraud' columns
        title_suffix: Optional suffix to add to the title
        figsize: Figure size tuple
        n_bins: Number of bins for calibration curve
    """
    plt.figure(figsize=figsize)
    
    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        data['is_fraud'], data['model_scores'], n_bins=n_bins
    )
    
    # Plot calibration curve
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model", linewidth=2)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Plot{title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(
    data: pd.DataFrame, 
    title_suffix: str = "",
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot precision-recall curve for model scores vs fraud values.
    
    Args:
        data: DataFrame containing 'model_scores' and 'is_fraud' columns
        title_suffix: Optional suffix to add to the title
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(data['is_fraud'], data['model_scores'])
    auc_score = auc(recall, precision)
    
    # Plot precision-recall curve
    plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {auc_score:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve{title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_policy_metrics_table(
    results: List[Dict[str, Any]], 
    comparison_key: str = 'exploration_rate'
) -> pd.DataFrame:
    """
    Create table with policy metrics for each comparison parameter.
    
    Args:
        results: List of simulation results
        comparison_key: Key to use for comparison (e.g., 'exploration_rate', 'model_type')
        
    Returns:
        DataFrame with policy metrics
    """
    table_data = []
    
    for result in results:
        comparison_value = result['parameters'][comparison_key]
        stats = result['statistics']
        
        row_data = {
            comparison_key: comparison_value,
            'allow_rate': stats['allow_rate'],
            'block_rate': stats['block_rate'],
            'fraud_rate_overall': stats['fraud_rate_overall'],
            'fraud_rate_allowed': stats['fraud_rate_allowed'],
            'total_transactions': stats['total_transactions'],
            'allowed_transactions': stats['allowed_transactions'],
            'blocked_transactions': stats['blocked_transactions']
        }
        
        # Add model performance metrics if available
        if 'model_performance' in result:
            model_perf = result['model_performance']
            row_data.update({
                'model_roc_auc': model_perf['roc_auc'],
                'model_precision': model_perf['precision'],
                'model_recall': model_perf['recall'],
                'model_f1': model_perf['f1']
            })
        
        table_data.append(row_data)
    
    return pd.DataFrame(table_data)


def plot_ope_metrics(
    results: List[Dict[str, Any]], 
    comparison_key: str = 'exploration_rate',
    figsize: Optional[Tuple[int, int]] = None,
    include_model_baseline: bool = True
) -> None:
    """
    Plot OPE metrics vs comparison parameter with confidence intervals.
    
    Args:
        results: List of simulation results
        comparison_key: Key to use for x-axis comparison
        figsize: Figure size tuple (auto-calculated if None)
        include_model_baseline: Whether to include model baseline if available
    """
    # Extract comparison values
    comparison_values = [result['parameters'][comparison_key] for result in results]
    
    # Get all available metrics from first result
    metrics = list(results[0]['ope_metrics'].keys())
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        n_metrics = len(metrics)
        cols = 2
        rows = (n_metrics + 1) // 2
        figsize = (15, 5 * rows)
    
    # Create subplots
    n_metrics = len(metrics)
    cols = 2
    rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Extract metric values
        estimates = [result['ope_metrics'][metric]['mean'] for result in results]
        ci_lower = [result['ope_metrics'][metric]['p025'] for result in results]
        ci_upper = [result['ope_metrics'][metric]['p975'] for result in results]
        
        # Plot estimate line
        ax.plot(comparison_values, estimates, 'o-', linewidth=2, markersize=6, label='OPE Estimate')
        
        # Plot confidence interval
        ax.fill_between(comparison_values, ci_lower, ci_upper, alpha=0.3, label='95% CI')
        
        # Plot model baseline if available and requested
        if include_model_baseline and 'model' in results[0]:
            model_metrics = results[0]['model']
            # Add fraud rate from statistics if not present in model metrics
            if 'fraud_rate' not in model_metrics and 'statistics' in results[0]:
                model_metrics = model_metrics.copy()
                model_metrics['fraud_rate'] = results[0]['statistics']['fraud_rate_allowed']
            
            if metric in model_metrics:
                model_value = model_metrics[metric]
                ax.axhline(y=model_value, color='red', linestyle='--', linewidth=2, label='Real Value')
        
        ax.set_xlabel(comparison_key.replace('_', ' ').title())
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'OPE Metric: {metric.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots if any
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_model_scores_comparison(reference_data: Dict[str, pd.DataFrame]) -> None:
    """
    Plot comparison of model scores across different models.
    
    Args:
        reference_data: Dictionary mapping model names to DataFrames
    """
    plt.figure(figsize=(15, 5))
    
    n_models = len(reference_data)
    
    for i, (model_type, data) in enumerate(reference_data.items()):
        plt.subplot(1, n_models, i+1)
        plt.hist(data['model_scores'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Model Scores')
        plt.ylabel('Frequency')
        plt.title(f'{model_type.replace("_", " ").title()} Model')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_model_performance_comparison(results: List[Dict[str, Any]]) -> None:
    """
    Plot model performance metrics comparison.
    
    Args:
        results: List of simulation results with model_performance data
    """
    if not results or 'model_performance' not in results[0]:
        print("No model performance data available for plotting.")
        return
    
    # Extract data for plotting
    model_types = [result['parameters']['model_type'] for result in results]
    
    metrics = ['roc_auc', 'precision', 'recall', 'f1']
    metric_data = {}
    
    for metric in metrics:
        metric_data[metric] = [result['model_performance'][metric] for result in results]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(model_types, metric_data[metric])
        ax.set_title(f'Model {metric.replace("_", " ").title()}')
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


def create_standard_data_summary_table(data: pd.DataFrame, pipeline_params: Dict[str, Any]) -> pd.DataFrame:
    """
    Create summary table with standard data statistics and generation parameters.
    
    Args:
        data: DataFrame with transaction data
        pipeline_params: Pipeline configuration parameters
        
    Returns:
        DataFrame with summary statistics
    """
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


def create_synthetic_data_summary_table(data: pd.DataFrame, pipeline_params: Dict[str, Any]) -> pd.DataFrame:
    """
    Create summary table with synthetic data statistics and generation parameters.
    
    Args:
        data: DataFrame with transaction data
        pipeline_params: Pipeline configuration parameters
        
    Returns:
        DataFrame with summary statistics
    """
    total_transactions = len(data)
    true_fraud_rate = data['is_fraud'].mean()
    
    # Extract parameters from nested structure
    synthetic_data_params = pipeline_params['synthetic_data']
    model_params = pipeline_params['model']
    
    summary_data = {
        'Metric': [
            'Total Transactions', 'True Fraud Rate', 'N Samples', 'N Features', 
            'N Informative', 'N Redundant', 'Model Type', 'Test Size', 'Class Sep'
        ],
        'Value': [
            total_transactions,
            f"{true_fraud_rate:.4f}",
            synthetic_data_params['n_samples'],
            synthetic_data_params['n_features'], 
            synthetic_data_params['n_informative'],
            synthetic_data_params['n_redundant'],
            model_params['model_type'],
            synthetic_data_params['test_size'],
            synthetic_data_params['class_sep']
        ]
    }
    
    return pd.DataFrame(summary_data)