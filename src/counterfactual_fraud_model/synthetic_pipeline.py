"""Synthetic Off Policy Evaluation Pipeline for Counterfactual Fraud Model Simulation."""

import pandas as pd
from typing import Dict, Optional, Any, List
from .generators.implementations import DataGeneratorFactory
from .strategies.generation import SklearnGenerationConfig
from .logging_policy import LoggingPolicyGenerator
from .counterfactual_estimator import CounterfactualValuesEstimator


class SyntheticOffPolicyEvaluationPipeline:
    """
    Orchestrates the complete off-policy evaluation workflow using synthetic datasets.
    
    Combines synthetic data generation with model training, logging policy simulation, 
    and counterfactual estimation to evaluate fraud detection policies.
    """
    
    def __init__(
        self,
        # Synthetic data generator parameters
        n_samples: int = 100_000,
        n_features: int = 30,
        n_informative: int = 15,
        n_redundant: int = 5,
        n_repeated: int = 0,
        n_clusters_per_class: int = 2,
        weights: Optional[List[float]] = None,
        flip_y: float = 0.01,
        class_sep: float = 1.0,
        # Model parameters
        model_type: str = "lightgbm",
        model_params: Optional[Dict[str, Any]] = None,
        test_size: float = 0.3,
        # Counterfactual estimator parameters
        n_bootstrap: int = 5000,
        random_state: Optional[int] = None,
    ):
        """
        Initialize SyntheticOffPolicyEvaluationPipeline.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Total number of features  
            n_informative: Number of informative features
            n_redundant: Number of redundant features
            n_repeated: Number of duplicated features
            n_clusters_per_class: Number of clusters per class
            weights: Class balance (e.g., [0.99, 0.01] for imbalanced fraud data)
            flip_y: Fraction of samples whose class is flipped (label noise)
            class_sep: Factor multiplying the hypercube size
            model_type: Type of model to train ("lightgbm", "random_forest", "logistic")
            model_params: Additional parameters for the model
            test_size: Fraction of data to use for testing model
            n_bootstrap: Number of bootstrap repetitions for counterfactual estimation
            random_state: Random seed for reproducibility across all components
        """
        # Validate inputs
        self._validate_initialization_params(
            n_samples, n_features, n_informative, n_redundant, 
            test_size, n_bootstrap
        )
        
        # Store parameters for component initialization
        self._n_samples = n_samples
        self._n_features = n_features
        self._n_informative = n_informative
        self._n_redundant = n_redundant
        self._n_repeated = n_repeated
        self._n_clusters_per_class = n_clusters_per_class
        self._weights = weights or [0.985, 0.015]
        self._flip_y = flip_y
        self._class_sep = class_sep
        self._model_type = model_type
        self._model_params = model_params or {}
        self._test_size = test_size
        self._n_bootstrap = n_bootstrap
        self._random_state = random_state
        
        # Initialize synthetic data generator (but don't generate data yet - lazy initialization)
        self._synthetic_data_generator = None
        self._generated_data = None
        
    @property
    def params(self) -> Dict[str, Any]:
        """Return the current parameters."""
        return {
            'n_samples': self._n_samples,
            'n_features': self._n_features,
            'n_informative': self._n_informative,
            'n_redundant': self._n_redundant,
            'n_repeated': self._n_repeated,
            'n_clusters_per_class': self._n_clusters_per_class,
            'weights': self._weights,
            'flip_y': self._flip_y,
            'class_sep': self._class_sep,
            'model_type': self._model_type,
            'model_params': self._model_params,
            'test_size': self._test_size,
            'n_bootstrap': self._n_bootstrap,
            'random_state': self._random_state
        }
    
    def _validate_initialization_params(
        self, 
        n_samples: int,
        n_features: int,
        n_informative: int,
        n_redundant: int,
        test_size: float,
        n_bootstrap: int
    ) -> None:
        """Validate initialization parameters."""
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if n_informative <= 0:
            raise ValueError("n_informative must be positive")
        if n_informative > n_features:
            raise ValueError("n_informative cannot exceed n_features")
        if not 0 <= test_size <= 1:
            raise ValueError("test_size must be between 0 and 1")
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
        Lazy property to get generated synthetic data.
        
        Returns:
            Generated synthetic fraud data with model scores
        """
        if self._generated_data is None:
            self._ensure_synthetic_data_generator()
            self._generated_data = self._synthetic_data_generator.generate_data()
        return self._generated_data.copy()
    
    def _ensure_synthetic_data_generator(self) -> None:
        """Ensure synthetic data generator is initialized using the new factory pattern."""
        if self._synthetic_data_generator is None:
            self._synthetic_data_generator = DataGeneratorFactory.create_ml_model_generator(
                n_samples=self._n_samples,
                n_features=self._n_features,
                n_informative=self._n_informative,
                n_redundant=self._n_redundant,
                n_repeated=self._n_repeated,
                n_clusters_per_class=self._n_clusters_per_class,
                weights=self._weights,
                flip_y=self._flip_y,
                class_sep=self._class_sep,
                model_type=self._model_type,
                model_params=self._model_params,
                test_size=self._test_size,
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
        approval_rate_increase = policy_approval_rate/original_approval_rate - 1 if original_approval_rate > 0 else 0
        fraud_rate_increase = policy_fraud_rate/original_fraud_rate - 1 if original_fraud_rate > 0 else 0
        
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
    
    def get_model_performance(self) -> Dict[str, float]:
        """
        Get performance metrics of the trained model.
        
        Returns:
            Dictionary with performance metrics
        """
        self._ensure_synthetic_data_generator()
        if self._generated_data is None:
            # Trigger data generation to train the model
            _ = self.generated_data
        
        return self._synthetic_data_generator.get_model_performance()
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the generated synthetic dataset.
        
        Returns:
            Dictionary with dataset information
        """
        self._ensure_synthetic_data_generator()
        if self._generated_data is None:
            # Trigger data generation
            _ = self.generated_data
        
        return self._synthetic_data_generator.get_dataset_info()
    
    def run_pipeline(
        self,
        cutoff: float = 0.05,
        exploration_rate: float = 0.05,
        include_data: bool = True
    ) -> Dict[str, Any]:
        """
        Execute the complete synthetic off-policy evaluation pipeline.
        
        Args:
            cutoff: Score threshold for the logging policy (0-1)
            exploration_rate: Rate of exploration for blocked transactions (0-1)
            include_data: Whether to include the full dataset in results
            
        Returns:
            Dictionary containing statistics, metrics, parameters, model performance, 
            dataset info, and optionally data
        """
        # Validate parameters
        self._validate_pipeline_params(cutoff, exploration_rate)
        
        # Step 1: Get synthetic data with trained model scores
        data = self.generated_data
        
        # Step 2: Apply logging policy
        logging_policy_generator = self._create_logging_policy_generator(cutoff, exploration_rate)
        policy_data = logging_policy_generator.generate_policy(data)
        
        # Step 3: Estimate counterfactual metrics
        estimator = self._create_counterfactual_estimator(policy_data)
        ope_metrics_results = estimator.estimate_policy_metrics()
        
        # Step 4: Calculate summary statistics
        summary_results = self._calculate_summary_statistics(policy_data)
        
        # Step 5: Get additional information about model and dataset
        model_performance = self.get_model_performance()
        dataset_info = self.get_dataset_info()
        
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
            'model_performance': model_performance,
            'dataset_info': dataset_info,
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
        if self._synthetic_data_generator is not None:
            self._synthetic_data_generator.regenerate_data()
        self._generated_data = None
        # Data will be regenerated on next access to generated_data property 