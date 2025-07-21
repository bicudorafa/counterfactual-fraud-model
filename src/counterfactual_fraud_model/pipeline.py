"""Off Policy Evaluation Pipeline for Counterfactual Fraud Model Simulation."""

import pandas as pd
from typing import Dict, Optional, Any
from .generators.implementations import DataGeneratorFactory
from .strategies.generation import ProbabilisticGenerationConfig
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
        alpha: float = 0.1,
        beta_param: float = 2.0,
        mean: float = -0.5, # simulate the possible bias from training with selection bias
        sd: float = 0.5,
        sample_size: int = 100_000,
        # Counterfactual estimator parameters
        n_bootstrap: int = 5000,
        random_state: Optional[int] = None,
    ):
        """
        Initialize OffPolicyEvaluationPipeline.
        
        Args:
            alpha: Alpha parameter for beta distribution in data generation
            beta_param: Beta parameter for beta distribution in data generation
            mean: Mean for normal error distribution in data generation
            sd: Standard deviation for normal error distribution in data generation
            sample_size: Number of samples to generate
            n_bootstrap: Number of bootstrap repetitions for counterfactual estimation
            random_state: Random seed for reproducibility across all components
        """
        # Validate inputs
        self._validate_initialization_params(
            alpha, beta_param, mean, sd, sample_size, n_bootstrap
        )
        
        # Store parameters for component initialization
        self._alpha = alpha
        self._beta_param = beta_param
        self._mean = mean
        self._sd = sd
        self._sample_size = sample_size
        self._n_bootstrap = n_bootstrap
        self._random_state = random_state
        
        # Initialize data generator (but don't generate data yet - lazy initialization)
        self._data_generator = None
        self._generated_data = None
        
    @property
    def params(self) -> Dict[str, Any]:
        """Return the current parameters."""
        return {
            'alpha': self._alpha,
            'beta_param': self._beta_param,
            'mean': self._mean,
            'sd': self._sd,
            'sample_size': self._sample_size,
            'n_bootstrap': self._n_bootstrap,
            'random_state': self._random_state
        }
    
    def _validate_initialization_params(
        self, 
        alpha: float, 
        beta_param: float, 
        mean: float, 
        sd: float, 
        sample_size: int, 
        n_bootstrap: int
    ) -> None:
        """Validate initialization parameters."""
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if beta_param <= 0:
            raise ValueError("beta_param must be positive")
        if sd <= 0:
            raise ValueError("sd must be positive")
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be positive")
    
    def _validate_pipeline_params(self, cutoff: float, exploration_rate: float) -> None:
        """Validate pipeline execution parameters."""
        if not 0 <= cutoff <= 1:
            raise ValueError("cutoff must be between 0 and 1")
        if not 0 <= exploration_rate <= 1:
            raise ValueError("exploration_rate must be between 0 and 1")
    
    @property
    def generated_data(self) -> pd.DataFrame:
        """
        Lazy property to get generated data.
        
        Returns:
            Generated synthetic fraud data
        """
        if self._generated_data is None:
            self._ensure_data_generator()
            self._generated_data = self._data_generator.generate_data()
        return self._generated_data.copy()
    
    def _ensure_data_generator(self) -> None:
        """Ensure data generator is initialized using the new factory pattern."""
        if self._data_generator is None:
            self._data_generator = DataGeneratorFactory.create_probabilistic_generator(
                alpha=self._alpha,
                beta_param=self._beta_param,
                mean=self._mean,
                sd=self._sd,
                sample_size=self._sample_size,
                random_state=self._random_state
            )
    
    def _create_logging_policy_generator(
        self, 
        cutoff: float, 
        exploration_rate: float
    ) -> LoggingPolicyGenerator:
        """Create and return a logging policy generator with specified parameters."""
        return LoggingPolicyGenerator(
            cutoff=cutoff,
            exploration_rate=exploration_rate,
            random_state=self._random_state
        )
    
    def _create_counterfactual_estimator(self, policy_data: pd.DataFrame) -> CounterfactualValuesEstimator:
        """Create and return a counterfactual estimator with policy data."""
        return CounterfactualValuesEstimator(
            data=policy_data,
            n_bootstrap=self._n_bootstrap,
            random_state=self._random_state,
        )
    
    def _calculate_summary_statistics(self, policy_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive summary statistics for the pipeline data.
        
        Args:
            policy_data: DataFrame with policy actions and model predictions
            
        Returns:
            Dictionary containing various statistical summaries
        """
        total_transactions = len(policy_data)
        
        # Original model performance
        original_approval_rate = (policy_data['model_action'] == 'allow').mean()
        original_allowed_data = policy_data[policy_data['model_action'] == 'allow']
        original_fraud_rate = original_allowed_data['is_fraud'].mean() if len(original_allowed_data) > 0 else 0.0
        
        # Policy performance
        policy_approval_rate = (policy_data['policy_action'] == 'allow').mean()
        policy_allowed_data = policy_data[policy_data['policy_action'] == 'allow']
        policy_fraud_rate = policy_allowed_data['is_fraud'].mean() if len(policy_allowed_data) > 0 else 0.0
        
        # Overall statistics
        true_fraud_rate = policy_data['is_fraud'].mean()
        
        # Comparative metrics
        approval_rate_increase = policy_approval_rate/original_approval_rate - 1
        fraud_rate_increase = policy_fraud_rate/original_fraud_rate - 1
        
        return {
            'statistics': {
                'total_transactions': total_transactions,
                'original_approval_rate': original_approval_rate,
                'original_fraud_rate': original_fraud_rate,
                'policy_approval_rate': policy_approval_rate,
                'policy_fraud_rate': policy_fraud_rate,
                'true_fraud_rate': true_fraud_rate,
                'approval_rate_increase': approval_rate_increase,
                'fraud_rate_increase': fraud_rate_increase
            }
        }
    
    def run_pipeline(
        self,
        cutoff: float = 0.05,
        exploration_rate: float = 0.05,
        include_data: bool = True
    ) -> Dict[str, Any]:
        """
        Execute the complete off-policy evaluation pipeline.
        
        Args:
            cutoff: Score threshold for the logging policy (0-1)
            exploration_rate: Rate of exploration for blocked transactions (0-1)
            include_data: Whether to include the full dataset in results
            
        Returns:
            Dictionary containing statistics, metrics, parameters, and optionally data
        """
        # Validate parameters
        self._validate_pipeline_params(cutoff, exploration_rate)
        
        # Step 1: Get synthetic data
        data = self.generated_data
        
        # Step 2: Apply logging policy
        logging_policy_generator = self._create_logging_policy_generator(cutoff, exploration_rate)
        policy_data = logging_policy_generator.generate_policy(data)
        
        # Step 3: Estimate counterfactual metrics
        estimator = self._create_counterfactual_estimator(policy_data)
        ope_metrics_results = estimator.estimate_policy_metrics()
        
        # Step 4: Calculate summary statistics
        summary_results = self._calculate_summary_statistics(policy_data)
        
        # Compile parameters used in this run
        run_parameters = self.params.copy()
        run_parameters.update({
            'cutoff': cutoff,
            'exploration_rate': exploration_rate
        })
        
        # Compile results
        results = {
            **summary_results,
            'ope_metrics': ope_metrics_results,
            'parameters': run_parameters
        }
        
        if include_data:
            results['data'] = policy_data
            
        return results
    
    def get_params(self) -> Dict[str, Any]:
        """Return the current parameters."""
        return self.params
    
    def regenerate_data(self) -> None:
        """Force regeneration of synthetic data with current parameters."""
        self._generated_data = None
        # Data will be regenerated on next access to generated_data property 