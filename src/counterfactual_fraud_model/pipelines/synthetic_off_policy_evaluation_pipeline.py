"""Refactored Synthetic Off Policy Evaluation Pipeline for Counterfactual Fraud Model Simulation.

This pipeline orchestrates the complete off-policy evaluation workflow using 
synthetic datasets with dependency injection and composition for loose coupling.
"""

import pandas as pd
from typing import Dict, Any

from ..config import SyntheticOffPolicyEvaluationConfig, LoggingPolicyConfig
from ..protocols import (
    SyntheticDataGeneratorProtocol,
    LoggingPolicyGeneratorProtocol,
    CounterfactualEstimatorProtocol,
    PipelineProtocol
)
from ..generators import SyntheticDataGenerator, LoggingPolicyGenerator
from ..estimators import CounterfactualEstimator


class SyntheticOffPolicyEvaluationPipeline(PipelineProtocol):
    """
    Orchestrates the complete off-policy evaluation workflow using synthetic datasets.
    
    Combines synthetic data generation with model training, logging policy simulation, 
    and counterfactual estimation to evaluate fraud detection policies using dependency injection.
    """
    
    def __init__(
        self,
        config: SyntheticOffPolicyEvaluationConfig,
        synthetic_data_generator: SyntheticDataGeneratorProtocol = None,
        logging_policy_generator: LoggingPolicyGeneratorProtocol = None
    ):
        """
        Initialize SyntheticOffPolicyEvaluationPipeline with configuration and dependencies.
        
        Args:
            config: Complete configuration for the pipeline
            synthetic_data_generator: Synthetic data generator (injected dependency, optional)
            logging_policy_generator: Policy generator (injected dependency, optional)
        """
        self.config = config
        
        # Use provided dependencies or create default ones
        self.synthetic_data_generator = synthetic_data_generator or SyntheticDataGenerator(
            config.synthetic_data, config.model
        )
        self.logging_policy_generator = logging_policy_generator or LoggingPolicyGenerator(config.logging_policy)
        
        # Cache for generated data
        self._generated_data: pd.DataFrame = None
    
    def run_pipeline(
        self,
        cutoff: float = None,
        exploration_rate: float = None,
        include_data: bool = None
    ) -> Dict[str, Any]:
        """
        Execute the complete synthetic off-policy evaluation pipeline.
        
        Args:
            cutoff: Score threshold for the logging policy (overrides config if provided)
            exploration_rate: Rate of exploration for blocked transactions (overrides config if provided)
            include_data: Whether to include the full dataset in results (overrides config if provided)
            
        Returns:
            Dictionary containing statistics, metrics, parameters, model performance, 
            dataset info, and optionally data
        """
        # Use provided parameters or fall back to config
        policy_config = self._build_policy_config(cutoff, exploration_rate)
        include_data_flag = include_data if include_data is not None else self.config.pipeline.include_data
        
        # Step 1: Generate synthetic data with trained model scores
        data = self._get_or_generate_data()
        
        # Step 2: Apply logging policy (create new generator if params changed)
        if policy_config != self.logging_policy_generator.get_config():
            policy_generator = LoggingPolicyGenerator(policy_config)
        else:
            policy_generator = self.logging_policy_generator
        
        policy_data = policy_generator.generate_policy(data)
        
        # Step 3: Create counterfactual estimator and estimate metrics
        estimator = CounterfactualEstimator(self.config.counterfactual_estimator, policy_data)
        ope_metrics_results = estimator.estimate_policy_metrics()
        
        # Step 4: Calculate summary statistics
        summary_results = self._calculate_summary_statistics(policy_data)
        
        # Step 5: Get additional information about model and dataset
        model_performance = self.get_model_performance()
        dataset_info = self.get_dataset_info()
        
        # Step 6: Compile parameters used in this run
        run_parameters = {
            'synthetic_data': self.synthetic_data_generator.get_config().model_dump(),
            'model': self.config.model.model_dump(),
            'logging_policy': policy_config.model_dump(),
            'counterfactual_estimator': self.config.counterfactual_estimator.model_dump(),
            'pipeline': self.config.pipeline.model_dump()
        }
        
        # Step 7: Compile results
        results = {
            **summary_results,
            'ope_metrics': ope_metrics_results,
            'model_performance': model_performance,
            'dataset_info': dataset_info,
            'parameters': run_parameters
        }
        
        if include_data_flag:
            results['data'] = policy_data
            
        return results
    
    def get_config(self) -> SyntheticOffPolicyEvaluationConfig:
        """Get the current configuration."""
        return self.config
    
    def get_model_performance(self) -> Dict[str, float]:
        """Get performance metrics of the trained model."""
        return self.synthetic_data_generator.get_model_performance()
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the generated dataset."""
        return self.synthetic_data_generator.get_dataset_info()
    
    def _get_or_generate_data(self) -> pd.DataFrame:
        """Get cached data or generate new data."""
        if self._generated_data is None:
            self._generated_data = self.synthetic_data_generator.generate_data()
        return self._generated_data.copy()
    
    def _build_policy_config(self, cutoff: float = None, exploration_rate: float = None) -> LoggingPolicyConfig:
        """Build logging policy configuration with optional overrides."""
        config_dict = self.config.logging_policy.model_dump()
        
        if cutoff is not None:
            config_dict['cutoff'] = cutoff
        if exploration_rate is not None:
            config_dict['exploration_rate'] = exploration_rate
            
        return LoggingPolicyConfig(**config_dict)
    
    def _calculate_summary_statistics(self, policy_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for the pipeline results."""
        total_transactions = len(policy_data)
        allowed_transactions = (policy_data['policy_action'] == 'allow').sum()
        blocked_transactions = (policy_data['policy_action'] == 'block').sum()
        
        # Calculate fraud rates
        if allowed_transactions > 0:
            allowed_data = policy_data[policy_data['policy_action'] == 'allow']
            fraud_rate_allowed = allowed_data['is_fraud'].mean()
        else:
            fraud_rate_allowed = 0.0
        
        overall_fraud_rate = policy_data['is_fraud'].mean()
        
        return {
            'statistics': {
                'total_transactions': total_transactions,
                'allowed_transactions': allowed_transactions,
                'blocked_transactions': blocked_transactions,
                'allow_rate': allowed_transactions / total_transactions if total_transactions > 0 else 0.0,
                'block_rate': blocked_transactions / total_transactions if total_transactions > 0 else 0.0,
                'fraud_rate_overall': overall_fraud_rate,
                'fraud_rate_allowed': fraud_rate_allowed
            }
        } 