"""
Comprehensive Usage Example for SyntheticRetrainingPipeline

This example demonstrates how to use the SyntheticRetrainingPipeline with the new 
configuration-based architecture for counterfactual fraud model simulation.

The pipeline performs the following steps:
1. Generates synthetic data and trains an initial model
2. Applies a logging policy to create transaction decisions
3. Filters to only allowed transactions for retraining
4. Trains a new model on the filtered data
5. Evaluates the new model using counterfactual estimation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

# Import configuration classes
from counterfactual_fraud_model.config import (
    SyntheticRetrainingConfig,
    SyntheticOffPolicyEvaluationConfig,
    SyntheticDataConfig,
    ModelConfig,
    LoggingPolicyConfig,
    CounterfactualEstimatorConfig,
    RetrainingConfig,
    PipelineConfig,
    ModelType
)

# Import the pipeline
from counterfactual_fraud_model.pipelines import SyntheticRetrainingPipeline


def basic_usage_example():
    """Demonstrate basic usage with default configuration."""
    print("=" * 60)
    print("BASIC USAGE EXAMPLE")
    print("=" * 60)
    
    # Create pipeline with default configuration
    config = SyntheticRetrainingConfig()
    pipeline = SyntheticRetrainingPipeline(config)
    
    print("Pipeline created with default configuration")
    print(f"Base dataset size: {config.base_config.synthetic_data.n_samples}")
    print(f"Number of features: {config.base_config.synthetic_data.n_features}")
    print(f"Model type: {config.base_config.model.model_type}")
    print(f"Classification threshold: {config.retraining.classification_threshold}")
    print()
    
    # Run the pipeline
    print("Running pipeline...")
    results = pipeline.run_pipeline(include_data=False)
    
    # Display key results
    print("Pipeline completed successfully!")
    print()
    display_results_summary(results)
    
    return results


def custom_configuration_example():
    """Demonstrate usage with custom configuration."""
    print("=" * 60)
    print("CUSTOM CONFIGURATION EXAMPLE")
    print("=" * 60)
    
    # Create custom configuration with specific parameters
    config = SyntheticRetrainingConfig(
        base_config=SyntheticOffPolicyEvaluationConfig(
            synthetic_data=SyntheticDataConfig(
                n_samples=50_000,        # Smaller dataset for faster execution
                n_features=20,           # More features for complexity
                n_informative=12,        # More informative features
                n_redundant=3,
                weights=[0.98, 0.02],    # More imbalanced classes
                random_state=42
            ),
            model=ModelConfig(
                model_type=ModelType.RANDOM_FOREST,  # Use Random Forest instead of LightGBM
                model_params={
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5
                },
                random_state=42
            ),
            logging_policy=LoggingPolicyConfig(
                cutoff=0.03,            # More conservative cutoff
                exploration_rate=0.1,   # Higher exploration rate
                random_state=42
            ),
            counterfactual_estimator=CounterfactualEstimatorConfig(
                n_bootstrap=1000,       # Fewer bootstrap samples for speed
                random_state=42
            ),
            pipeline=PipelineConfig(
                include_data=False
            )
        ),
        retraining=RetrainingConfig(
            retrain_test_size=0.25,     # Use 25% for testing retrained model
            classification_threshold=0.04,  # Different threshold for new model
            retrain_model=ModelConfig(
                model_type=ModelType.LOGISTIC,  # Use different model type for retraining
                model_params={
                    "C": 1.0,
                    "max_iter": 1000
                },
                random_state=42
            )
        )
    )
    
    print("Custom configuration created:")
    print(f"  Dataset: {config.base_config.synthetic_data.n_samples:,} samples, {config.base_config.synthetic_data.n_features} features")
    print(f"  Original model: {config.base_config.model.model_type}")
    print(f"  Retrained model: {config.retraining.retrain_model.model_type}")
    print(f"  Cutoff: {config.base_config.logging_policy.cutoff}")
    print(f"  Classification threshold: {config.retraining.classification_threshold}")
    print()
    
    # Create pipeline with custom configuration
    pipeline = SyntheticRetrainingPipeline(config)
    
    print("Running pipeline with custom configuration...")
    results = pipeline.run_pipeline()
    
    print("Pipeline completed successfully!")
    print()
    display_results_summary(results)
    
    return results


def runtime_override_example():
    """Demonstrate runtime parameter overrides."""
    print("=" * 60)
    print("RUNTIME OVERRIDE EXAMPLE")
    print("=" * 60)
    
    # Create pipeline with standard configuration
    config = SyntheticRetrainingConfig(
        base_config=SyntheticOffPolicyEvaluationConfig(
            synthetic_data=SyntheticDataConfig(
                n_samples=30_000,
                random_state=42
            ),
            model=ModelConfig(
                model_type=ModelType.LIGHTGBM,
                random_state=42
            )
        )
    )
    
    pipeline = SyntheticRetrainingPipeline(config)
    
    print("Running multiple experiments with different parameters...")
    print()
    
    # Experiment 1: Conservative policy
    print("Experiment 1: Conservative policy (low cutoff, low exploration)")
    results1 = pipeline.run_pipeline(
        cutoff=0.02,
        exploration_rate=0.03,
        include_data=False
    )
    print(f"  Original approval rate: {results1['statistics']['original_approval_rate']:.3f}")
    print(f"  New model fraud rate: {results1['statistics']['new_model_fraud_rate']:.4f}")
    print()
    
    # Experiment 2: Aggressive policy  
    print("Experiment 2: Aggressive policy (high cutoff, high exploration)")
    results2 = pipeline.run_pipeline(
        cutoff=0.08,
        exploration_rate=0.15,
        include_data=False
    )
    print(f"  Original approval rate: {results2['statistics']['original_approval_rate']:.3f}")
    print(f"  New model fraud rate: {results2['statistics']['new_model_fraud_rate']:.4f}")
    print()
    
    # Compare results
    print("Comparison:")
    print(f"  Conservative -> Aggressive approval rate change: "
          f"{results1['statistics']['original_approval_rate']:.3f} -> "
          f"{results2['statistics']['original_approval_rate']:.3f}")
    print(f"  Conservative -> Aggressive fraud rate change: "
          f"{results1['statistics']['new_model_fraud_rate']:.4f} -> "
          f"{results2['statistics']['new_model_fraud_rate']:.4f}")
    
    return results1, results2


def comprehensive_analysis_example():
    """Demonstrate comprehensive analysis with data access."""
    print("=" * 60)
    print("COMPREHENSIVE ANALYSIS EXAMPLE")
    print("=" * 60)
    
    # Create configuration that includes data in results
    config = SyntheticRetrainingConfig(
        base_config=SyntheticOffPolicyEvaluationConfig(
            synthetic_data=SyntheticDataConfig(
                n_samples=20_000,  # Smaller dataset since we're including data
                random_state=42
            ),
            pipeline=PipelineConfig(
                include_data=True  # Include data for analysis
            )
        )
    )
    
    pipeline = SyntheticRetrainingPipeline(config)
    
    print("Running pipeline with data inclusion for detailed analysis...")
    results = pipeline.run_pipeline()
    
    print("Pipeline completed! Performing detailed analysis...")
    print()
    
    # Access the data for analysis
    original_data = results['original_data']
    retrained_data = results['retrained_data']
    new_scores = results['new_scores']
    binary_predictions = results['binary_predictions']
    
    print("Dataset Analysis:")
    print(f"  Original dataset size: {len(original_data):,}")
    print(f"  Allowed transactions for retraining: {len(retrained_data):,}")
    print(f"  Percentage allowed: {len(retrained_data)/len(original_data)*100:.1f}%")
    print()
    
    # Analyze score distributions
    print("Score Analysis:")
    print(f"  New model scores - Min: {new_scores.min():.4f}, Max: {new_scores.max():.4f}")
    print(f"  New model scores - Mean: {new_scores.mean():.4f}, Std: {new_scores.std():.4f}")
    print(f"  Prediction distribution: {binary_predictions.mean():.3f} blocked, {1-binary_predictions.mean():.3f} allowed")
    print()
    
    # Model performance comparison
    print("Model Performance Comparison:")
    orig_perf = results['original_model_performance']
    retrain_perf = results['retrained_model_performance']
    
    print(f"  Original Model:")
    for metric, value in orig_perf.items():
        if isinstance(value, (int, float)):
            print(f"    {metric}: {value:.4f}")
    
    print(f"  Retrained Model:")
    for metric, value in retrain_perf.items():
        if isinstance(value, (int, float)):
            print(f"    {metric}: {value:.4f}")
    
    print()
    
    # OPE Metrics
    print("Off-Policy Evaluation Metrics:")
    ope_metrics = results['ope_metrics']
    for metric, value in ope_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
    
    return results


def display_results_summary(results: Dict[str, Any]):
    """Display a summary of pipeline results."""
    stats = results['statistics']
    ope = results['ope_metrics']
    
    print("RESULTS SUMMARY")
    print("-" * 40)
    
    print("Dataset Statistics:")
    print(f"  Total transactions: {stats['total_transactions']:,}")
    print(f"  Allowed for retraining: {stats['total_allowed_for_retraining']:,}")
    print(f"  Original approval rate: {stats['original_approval_rate']:.3f}")
    print(f"  Original fraud rate: {stats['original_fraud_rate']:.4f}")
    print()
    
    print("New Model Performance:")
    print(f"  Block rate: {stats['new_model_block_rate']:.3f}")
    print(f"  Allow rate: {stats['new_model_allow_rate']:.3f}")
    print(f"  Fraud rate in allowed: {stats['new_model_fraud_rate']:.4f}")
    print()
    
    print("Key OPE Metrics:")
    print(ope)
    # print(f"  Policy value: {ope.get('policy_value', 'N/A')}")
    # print(f"  Expected reward: {ope.get('expected_reward', 'N/A')}")
    # print()


def error_handling_example():
    """Demonstrate error handling and edge cases."""
    print("=" * 60)
    print("ERROR HANDLING EXAMPLE")
    print("=" * 60)
    
    # Create configuration that might lead to edge cases
    config = SyntheticRetrainingConfig(
        base_config=SyntheticOffPolicyEvaluationConfig(
            synthetic_data=SyntheticDataConfig(
                n_samples=1000,  # Very small dataset
                random_state=42
            ),
            logging_policy=LoggingPolicyConfig(
                cutoff=0.001,  # Very low cutoff - might allow almost everything
                exploration_rate=0.0,  # No exploration
                random_state=42
            )
        )
    )
    
    pipeline = SyntheticRetrainingPipeline(config)
    
    try:
        print("Running pipeline with edge case configuration...")
        results = pipeline.run_pipeline()
        print("Pipeline completed successfully despite edge case configuration!")
        display_results_summary(results)
        
    except ValueError as e:
        print(f"Expected error occurred: {e}")
        print("This demonstrates the pipeline's error handling for edge cases.")
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Consider adjusting the configuration parameters.")


def main():
    """Run all examples."""
    print("SyntheticRetrainingPipeline Usage Examples")
    print("=" * 60)
    print()
    
    try:
        # Run all examples
        basic_usage_example()
        print("\n" + "="*60 + "\n")
        
        custom_configuration_example()
        print("\n" + "="*60 + "\n")
        
        runtime_override_example()
        print("\n" + "="*60 + "\n")
        
        comprehensive_analysis_example()
        print("\n" + "="*60 + "\n")
        
        error_handling_example()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Please ensure all dependencies are installed and the environment is set up correctly.")


if __name__ == "__main__":
    main() 