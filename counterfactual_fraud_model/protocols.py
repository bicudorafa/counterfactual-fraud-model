"""Protocol definitions for counterfactual fraud model components.

This module defines abstract interfaces (protocols) for all major component types
to enable dependency injection, testing, and loose coupling.
"""

import pandas as pd
import numpy as np
from typing import Protocol, Dict, Any, runtime_checkable, Tuple, Optional
from sklearn.base import BaseEstimator

from .config import (
    DataGeneratorConfig,
    SyntheticDataConfig,
    ModelConfig,
    LoggingPolicyConfig,
    CounterfactualEstimatorConfig,
    RetrainingModelConfig
)


@runtime_checkable
class DataGeneratorProtocol(Protocol):
    """Protocol for data generators that create basic fraud detection datasets."""
    
    def generate_data(self) -> pd.DataFrame:
        """Generate synthetic fraud data.
        
        Returns:
            DataFrame with columns: model_scores, is_fraud
        """
        ...
    
    def get_config(self) -> DataGeneratorConfig:
        """Get the current configuration."""
        ...


@runtime_checkable  
class SyntheticDataGeneratorProtocol(Protocol):
    """Protocol for synthetic data generators that create datasets with features."""
    
    def generate_data(self) -> pd.DataFrame:
        """Generate synthetic fraud data with features.
        
        Returns:
            DataFrame with feature columns, is_fraud, and model_scores
        """
        ...
    
    def get_config(self) -> SyntheticDataConfig:
        """Get the current configuration."""
        ...
    
    def get_model(self) -> BaseEstimator:
        """Get the trained model."""
        ...
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the generated dataset."""
        ...


@runtime_checkable
class ModelFactoryProtocol(Protocol):
    """Protocol for model factories that create ML models."""
    
    def create_model(self, config: ModelConfig) -> BaseEstimator:
        """Create a model instance based on configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Configured scikit-learn compatible model instance
        """
        ...


@runtime_checkable
class LoggingPolicyGeneratorProtocol(Protocol):
    """Protocol for logging policy generators."""
    
    def generate_policy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate logging policy with propensity scores and actions.
        
        Args:
            data: DataFrame with model_scores column
            
        Returns:
            DataFrame with added columns: propensity_score, model_action, policy_action
        """
        ...
    
    def get_config(self) -> LoggingPolicyConfig:
        """Get the current configuration."""
        ...


@runtime_checkable
class CounterfactualEstimatorProtocol(Protocol):
    """Protocol for counterfactual estimators."""
    
    def estimate_policy_metrics(self, is_vectorized: bool = True) -> Dict[str, Any]:
        """Estimate counterfactual policy metrics.
        
        Args:
            is_vectorized: If True, uses the vectorized implementation for better performance
                         with large datasets. If False, uses the loop-based implementation.
                         Default True for optimal performance.
        
        Returns:
            Dictionary with estimated metrics and confidence intervals
        """
        ...
    
    def estimate_ope_metrics(self, new_actions: np.ndarray, new_actions_proba: np.ndarray) -> Dict[str, Any]:
        """Estimate off-policy evaluation metrics using loop-based implementation.
        
        Args:
            new_actions: Array of new policy actions (0=allow, 1=block)
            new_actions_proba: Array of new policy action probabilities/scores
            
        Returns:
            Dictionary with estimated metrics and confidence intervals
        """
        ...
    
    def estimate_ope_metrics_vectorized(self, new_actions: np.ndarray, new_actions_proba: np.ndarray) -> Dict[str, Any]:
        """Estimate off-policy evaluation metrics using vectorized implementation.
        
        Args:
            new_actions: Array of new policy actions (0=allow, 1=block)
            new_actions_proba: Array of new policy action probabilities/scores
            
        Returns:
            Dictionary with estimated metrics and confidence intervals
        """
        ...
    
    def get_config(self) -> CounterfactualEstimatorConfig:
        """Get the current configuration."""
        ...


@runtime_checkable
class PipelineProtocol(Protocol):
    """Protocol for evaluation pipelines."""
    
    def run_pipeline(self, **kwargs) -> Dict[str, Any]:
        """Execute the pipeline.
        
        Returns:
            Dictionary with results including metrics, statistics, and optionally data
        """
        ...
    
    def get_config(self) -> Any:
        """Get the current configuration."""
        ...


@runtime_checkable
class ModelTrainerProtocol(Protocol):
    """Protocol for model training components."""
    
    def train_model(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        config: ModelConfig, 
        sample_weight: Optional[np.ndarray] = None
    ) -> BaseEstimator:
        """Train a model on the provided data.
        
        Args:
            X: Feature matrix
            y: Target vector
            config: Model configuration
            sample_weight: Optional sample weights for training
            
        Returns:
            Trained model instance
        """
        ...
    
    def calculate_performance(
        self, 
        model: BaseEstimator, 
        X: pd.DataFrame, 
        y: pd.Series, 
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """Calculate performance metrics for a model.
        
        Args:
            model: Trained model
            X: Feature matrix for evaluation
            y: Target vector for evaluation
            threshold: Optional threshold for binary classification. If provided, 
                      uses probabilities >= threshold as positive predictions.
                      If None, uses model's default predict() method.
            
        Returns:
            Dictionary with performance metrics
        """
        ...


@runtime_checkable  
class RetrainingDataPreprocessorProtocol(Protocol):
    """Protocol for retraining data preprocessing strategies."""
    
    def prepare_training_data(
        self, 
        policy_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[np.ndarray]]:
        """Prepare training data from policy data using specific strategy.
        
        Args:
            policy_data: Full policy data with features, actions, and propensity scores
            
        Returns:
            Tuple of (features_df, target_series, sample_weights_or_none)
            - features_df: DataFrame with feature columns ready for training
            - target_series: Target variable (is_fraud)
            - sample_weights_or_none: Optional sample weights for training (None for filtering strategies)
        """
        ...
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the preprocessing strategy used.
        
        Returns:
            Dictionary with strategy details for logging/debugging
        """
        ... 