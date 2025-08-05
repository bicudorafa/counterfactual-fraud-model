#!/usr/bin/env python3
"""
Multi-parameter simulation analysis for SyntheticOffPolicyEvaluationPipeline.

This script runs multiple simulations with different exploration rates and model types,
generating comprehensive analysis including plots and summary tables.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional

# Add the project root to Python path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from counterfactual_fraud_model import (
    SyntheticOffPolicyEvaluationPipeline,
    SyntheticOffPolicyEvaluationConfig,
    SyntheticDataConfig,
    ModelConfig,
    LoggingPolicyConfig,
    CounterfactualEstimatorConfig,
    PipelineConfig,
    ModelType
)

# Import common analysis functions
from .common_analysis import (
    plot_model_scores_histogram,
    plot_calibration_curve,
    plot_precision_recall_curve,
    create_policy_metrics_table,
    plot_ope_metrics,
    plot_model_scores_comparison,
    plot_model_performance_comparison,
    create_synthetic_data_summary_table
)


def run_exploration_rate_simulations(
    exploration_rates: np.ndarray, 
    cutoff: float = 0.05,
    model_type: str = "lightgbm",
    n_samples: int = 50_000,
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
    print(f"Running {len(exploration_rates)} simulations with {model_type} model...")
    
    # Convert string model type to enum
    if model_type == "lightgbm":
        model_type_enum = ModelType.LIGHTGBM
    elif model_type == "random_forest":
        model_type_enum = ModelType.RANDOM_FOREST
    elif model_type == "logistic":
        model_type_enum = ModelType.LOGISTIC
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create configuration for the pipeline
    config = SyntheticOffPolicyEvaluationConfig(
        synthetic_data=SyntheticDataConfig(n_samples=n_samples, random_state=random_state),
        model=ModelConfig(model_type=model_type_enum, random_state=random_state),
        logging_policy=LoggingPolicyConfig(
            cutoff=cutoff,
            exploration_rate=0.05,  # Default, will be overridden in loop
            random_state=random_state
        ),
        counterfactual_estimator=CounterfactualEstimatorConfig(random_state=random_state),
        pipeline=PipelineConfig(include_data=False)
    )
    
    # Initialize pipeline with configuration
    pipeline = SyntheticOffPolicyEvaluationPipeline(config)
    
    # TODO: improve this in the future. We generate the data twice (here and below). Create method similar to retraining pipeline
    reference_data = pipeline._get_or_generate_data()
    
    results = []
    
    for i, exploration_rate in enumerate(exploration_rates):
        print(f"Running simulation {i+1}/{len(exploration_rates)} with exploration_rate={exploration_rate:.3f}")
        
        # Run pipeline with current exploration rate (overrides config)
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
    print(f"Running model comparison with {len(model_types)} models...")
    
    results = []
    reference_data = {}
    
    for i, model_type in enumerate(model_types):
        print(f"Running simulation {i+1}/{len(model_types)} with {model_type} model...")
        
        # Convert string model type to enum
        if model_type == "lightgbm":
            model_type_enum = ModelType.LIGHTGBM
        elif model_type == "random_forest":
            model_type_enum = ModelType.RANDOM_FOREST
        elif model_type == "logistic":
            model_type_enum = ModelType.LOGISTIC
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create configuration for current model type
        config = SyntheticOffPolicyEvaluationConfig(
            synthetic_data=SyntheticDataConfig(random_state=random_state),
            model=ModelConfig(model_type=model_type_enum, random_state=random_state),
            logging_policy=LoggingPolicyConfig(
                cutoff=cutoff,
                exploration_rate=exploration_rate,
                random_state=random_state
            ),
            counterfactual_estimator=CounterfactualEstimatorConfig(random_state=random_state),
            pipeline=PipelineConfig(include_data=False)
        )
        
        # Initialize pipeline with configuration
        pipeline = SyntheticOffPolicyEvaluationPipeline(config)
        
        # Get reference data for this model
        reference_data[model_type] = pipeline._get_or_generate_data()
        
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


def run_exploration_rate_analysis(exploration_rates: np.ndarray, model_type: str = "lightgbm"):
    """Run complete exploration rate analysis."""
    print(f"\nExploration Rate Analysis with {model_type} model")
    print("=" * 60)
    
    # Run simulations
    results, reference_data = run_exploration_rate_simulations(exploration_rates, model_type=model_type)
    
    # Get pipeline parameters for summary table
    pipeline_params = results[0]['parameters']
    
    print("\nGenerating visualizations and tables...")
    print("=" * 50)
    
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
    policy_table = create_policy_metrics_table(results, comparison_key='exploration_rate')
    print("\nPolicy Metrics Table:")
    print(policy_table.round(4).to_string(index=False))
    
    # 6. Plot OPE metrics
    print("\n6. Generating OPE metrics plots...")
    plot_ope_metrics(results, comparison_key='exploration_rate', include_model_baseline=False)


def run_model_comparison_analysis(model_types: List[str]):
    """Run complete model comparison analysis."""
    print(f"\nModel Comparison Analysis")
    print("=" * 60)
    
    # Run simulations
    results, reference_data = run_model_comparison_simulations(model_types)
    
    print("\nGenerating visualizations and tables...")
    print("=" * 50)
    
    # 1. Plot model scores comparison
    print("1. Generating model scores comparison...")
    plot_model_scores_comparison(reference_data)
    
    # 2. Create and display model comparison table
    print("2. Creating model comparison table...")
    comparison_table = create_policy_metrics_table(results, comparison_key='model_type')
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
    print("=" * 60)
    
    # Configuration for exploration rate analysis
    exploration_rates = np.linspace(0.01, 0.1, 10)
    
    # Configuration for model comparison
    model_types = ["lightgbm", "random_forest", "logistic"]
    
    # Run exploration rate analysis with LightGBM
    run_exploration_rate_analysis(exploration_rates, model_type="lightgbm")
    
    # Run model comparison analysis
    run_model_comparison_analysis(model_types)
    
    print("\nAll analyses complete!")
    print("=" * 60)


if __name__ == "__main__":
    main() 