import numpy as np
from typing import List, Dict, Any, Tuple
import pandas as pd
from counterfactual_fraud_model.config import (
    SyntheticRetrainingConfig,
    SyntheticOffPolicyEvaluationConfig,
    SyntheticDataConfig,
    ModelConfig,
    LoggingPolicyConfig,
    CounterfactualEstimatorConfig,
    RetrainingConfig,
    RetrainingModelConfig,
    ModelType,
    RetrainingStrategy
)
from counterfactual_fraud_model.pipelines import SyntheticRetrainingPipeline


def run_retraining_simulations(
    exploration_rates: np.ndarray,
    strategies: List[RetrainingStrategy],
    cutoff: float = 0.05,
    sample_size: int = 10_000,
    classification_threshold: float = 0.1,
    random_state: int = 42
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Run multiple SyntheticRetrainingPipeline simulations with different exploration rates and strategies.
    
    Args:
        exploration_rates: Array of exploration rate values to test
        strategies: List of retraining strategies to compare
        cutoff: Fixed cutoff value for all simulations
        sample_size: Number of samples in synthetic dataset
        classification_threshold: Threshold for binary classification
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (list of simulation results, reference data for plots)
    """
    # Calculate total number of simulations
    total_sims = len(exploration_rates) * len(strategies)
    print(f"Running {total_sims} simulations...")
    print(f"  - Exploration rates: {len(exploration_rates)} values")
    print(f"  - Retraining strategies: {len(strategies)} strategies")
    
    results = []
    reference_data = None  # Will store base data for plotting

    
    # Run simulations for each combination
    for exploration_rate in exploration_rates:
        print(f"Running simulations for exploration_rate={exploration_rate:.3f}")

        # Base config for most simulations
        config = SyntheticRetrainingConfig(
            base_config=SyntheticOffPolicyEvaluationConfig(
                synthetic_data=SyntheticDataConfig(
                    n_samples=sample_size,
                    n_features=15,
                    n_informative=10,
                    n_redundant=3,
                    n_repeated=0,
                    random_state=random_state  # Same seed for fair comparison
                ),
                model=ModelConfig(
                    model_type=ModelType.LIGHTGBM,
                    random_state=random_state
                ),
                logging_policy=LoggingPolicyConfig(
                    cutoff=cutoff,
                    exploration_rate=exploration_rate,
                    random_state=random_state
                ),
                counterfactual_estimator=CounterfactualEstimatorConfig(
                    n_bootstrap=1000,  # Reasonable for analysis
                    random_state=random_state
                )
            )
        )
        
        # Initialize and run pipeline
        pipeline = SyntheticRetrainingPipeline(config)

        pipeline.generate_logging_policy_data()

        for strategy in strategies:
            print(f"Strategy={strategy.value} simulation")

            retraining_config = RetrainingConfig(
                retrain_test_size=0.3,
                retrain_model=RetrainingModelConfig(
                    base_model=ModelConfig(
                        model_type=ModelType.LIGHTGBM,
                        random_state=random_state
                    ),
                    strategy=strategy, # it doesn't matter which strategy we use, it'll change dinamically through the simulation
                    classification_threshold=0.1
                )
            )
            
            result = pipeline.run_retrain_pipeline(retraining_config=retraining_config)  # Removed include_data parameter
        
            # Add metadata for easy tracking
            result['simulation_metadata'] = {
                'exploration_rate': exploration_rate,
                'strategy': strategy.value,
                'result': result
            }
            
            results.append(result)
            
            # Store reference data from first simulation for plotting
            if reference_data is None:
                # Get the base data for plotting (same across all simulations with same random_state)
                reference_data = pipeline.get_test_policy_data()
    
    return results, reference_data

def main():
    exploration_rates = [0.05, 0.1, 0.2]
    strategies = [RetrainingStrategy.FILTERING, RetrainingStrategy.WEIGHTING]
    results, reference_data = run_retraining_simulations(exploration_rates, strategies)
    print(results)

if __name__ == "__main__":
    main()