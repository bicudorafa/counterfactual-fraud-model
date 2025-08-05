#!/usr/bin/env python3
"""
Multi-parameter simulation analysis for OffPolicyEvaluationPipeline.

This script runs multiple simulations with different exploration_rate values
and generates comprehensive analysis including plots and summary tables.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from counterfactual_fraud_model import (
    OffPolicyEvaluationPipeline,
    OffPolicyEvaluationConfig,
    DataGeneratorConfig,
    LoggingPolicyConfig,
    CounterfactualEstimatorConfig,
    PipelineConfig
)

# Import common analysis functions
from .common_analysis import (
    plot_model_scores_histogram,
    plot_calibration_curve,
    plot_precision_recall_curve,
    create_policy_metrics_table,
    plot_ope_metrics,
    create_standard_data_summary_table
)


def run_simulations(exploration_rates: np.ndarray, 
                   cutoff: float = 0.05,
                   sample_size: int = 20_000,
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
            sample_size=sample_size,
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
    summary_table = create_standard_data_summary_table(reference_data, pipeline_params)
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
    policy_table = create_policy_metrics_table(results, comparison_key='exploration_rate')
    print("\nPolicy Metrics Table:")
    print(policy_table.round(4).to_string(index=False))
    
    # 6. Plot OPE metrics
    print("\n6. Generating OPE metrics plots...")
    plot_ope_metrics(results, comparison_key='exploration_rate')
    
    print("\nAnalysis complete!")
    print("=" * 50)


if __name__ == "__main__":
    main() 