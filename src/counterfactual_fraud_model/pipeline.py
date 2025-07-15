"""Off Policy Evaluation Pipeline for Counterfactual Fraud Model Simulation."""

import pandas as pd
from typing import Dict, List, Optional, Any
from .data_generator import DataGenerator
from .logging_policy import LoggingPolicyGenerator
from .counterfactual_estimator import CounterfactualValuesEstimator


class OffPolicyEvaluationPipeline:
    """
    Orchestrates the complete off-policy evaluation workflow.
    
    Combines data generation, logging policy simulation, and counterfactual
    estimation to evaluate fraud detection policies.
    """
    
    def __init__(
        self,
        # Data generator parameters
        alpha: float = 0.5,
        beta_param: float = 10.0,
        mean: float = 0.0,
        sd: float = 0.1,
        sample_size: int = 10_000,
        # Logging policy parameters
        cutoff: float = 0.05,
        exploration_rate: float = 0.05,
        propensity_type: str = "uniform",
        # Counterfactual estimator parameters
        n_bootstrap: int = 5000,
        random_state: Optional[int] = None
    ):
        """
        Initialize OffPolicyEvaluationPipeline.
        
        Args:
            alpha: Alpha parameter for beta distribution
            beta_param: Beta parameter for beta distribution
            mean: Mean for normal error distribution
            sd: Standard deviation for normal error distribution
            sample_size: Number of samples to generate
            cutoff: Score threshold for blocking transactions
            exploration_rate: Base exploration rate for blocked transactions
            propensity_type: Type of propensity function ("uniform" or "linear")
            n_bootstrap: Number of bootstrap repetitions
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        
        # Initialize components
        self.data_generator = DataGenerator(
            alpha=alpha,
            beta_param=beta_param,
            mean=mean,
            sd=sd,
            sample_size=sample_size,
            random_state=random_state
        )
        
        self.logging_policy = LoggingPolicyGenerator(
            cutoff=cutoff,
            exploration_rate=exploration_rate,
            propensity_type=propensity_type,
            random_state=random_state
        )
        
        # Store estimator parameters for later use
        self.estimator_params = {
            'n_bootstrap': n_bootstrap,
            'random_state': random_state
        }
        
        # Store parameters for easy access
        self.params = {
            'alpha': alpha,
            'beta_param': beta_param,
            'mean': mean,
            'sd': sd,
            'sample_size': sample_size,
            'cutoff': cutoff,
            'exploration_rate': exploration_rate,
            'propensity_type': propensity_type,
            'n_bootstrap': n_bootstrap,
            'random_state': random_state
        }
    
    def run_pipeline(self, policy_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute the complete off-policy evaluation pipeline.
        
        Args:
            policy_threshold: Threshold for policy evaluation (defaults to cutoff)
            
        Returns:
            Dictionary containing data, metrics, and pipeline information
        """
        if policy_threshold is None:
            policy_threshold = self.params['cutoff']
        
        # Step 1: Generate synthetic data
        data = self.data_generator.generate_data()
        
        # Step 2: Apply logging policy
        policy_data = self.logging_policy.generate_policy(data)
        
        # Step 3: Create estimator with data and estimate counterfactual metrics
        estimator = CounterfactualValuesEstimator(
            data=policy_data,
            **self.estimator_params
        )
        
        metrics_results = estimator.estimate_threshold_metrics(policy_threshold)
        
        # Calculate some basic statistics for reporting
        total_transactions = len(policy_data)
        observed_transactions = (policy_data['policy_action'] == 'allow').sum()
        observation_rate = observed_transactions / total_transactions
        
        fraud_rate_observed = policy_data[policy_data['policy_action'] == 'allow']['is_fraud'].mean()
        fraud_rate_all = policy_data['is_fraud'].mean()
        
        results = {
            'data': policy_data,
            'metrics': metrics_results,
            'statistics': {
                'total_transactions': total_transactions,
                'observed_transactions': observed_transactions,
                'observation_rate': observation_rate,
                'fraud_rate_observed': fraud_rate_observed,
                'fraud_rate_all': fraud_rate_all,
                'policy_threshold': policy_threshold
            },
            'parameters': self.params.copy()
        }
        
        return results
    
    def evaluate_multiple_thresholds(
        self, 
        thresholds: List[float]
    ) -> Dict[str, Any]:
        """
        Evaluate the pipeline across multiple policy thresholds.
        
        Args:
            thresholds: List of policy thresholds to evaluate
            
        Returns:
            Dictionary with results for each threshold
        """
        # Generate data once
        data = self.data_generator.generate_data()
        policy_data = self.logging_policy.generate_policy(data)
        
        # Create estimator with data once
        estimator = CounterfactualValuesEstimator(
            data=policy_data,
            **self.estimator_params
        )
        
        results = {}
        for threshold in thresholds:
            metrics_results = estimator.estimate_threshold_metrics(threshold)
            
            results[threshold] = {
                'metrics': metrics_results,
                'threshold': threshold
            }
        
        # Add common data and parameters
        results['data'] = policy_data
        results['parameters'] = self.params.copy()
        results['thresholds'] = thresholds
        
        return results
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the pipeline data.
        
        Args:
            data: DataFrame from pipeline execution
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_samples': len(data),
            'fraud_rate': data['is_fraud'].mean(),
            'mean_model_score': data['model_scores'].mean(),
            'propensity_score_stats': {
                'mean': data['propensity_score'].mean(),
                'min': data['propensity_score'].min(),
                'max': data['propensity_score'].max(),
                'std': data['propensity_score'].std()
            },
            'observation_rate': (data['policy_action'] == 'allow').mean(),
            'block_rate': (data['policy_action'] == 'block').mean()
        }
        
        return summary
    
    def update_exploration_rate(self, new_exploration_rate: float) -> None:
        """Update the exploration rate and reinitialize logging policy."""
        self.params['exploration_rate'] = new_exploration_rate
        self.logging_policy = LoggingPolicyGenerator(
            cutoff=self.params['cutoff'],
            exploration_rate=new_exploration_rate,
            propensity_type=self.params['propensity_type'],
            random_state=self.random_state
        )
    
    def get_params(self) -> dict:
        """Return the current parameters."""
        return self.params.copy() 