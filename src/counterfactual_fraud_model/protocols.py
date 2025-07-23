"""Protocol definitions for counterfactual fraud model components.

This module defines abstract interfaces (protocols) for all major component types
to enable dependency injection, testing, and loose coupling.
"""

import pandas as pd
import numpy as np
from typing import Protocol, Dict, Any, runtime_checkable
from sklearn.base import BaseEstimator

from .config import (
    DataGeneratorConfig,
    SyntheticDataConfig,
    ModelConfig,
    LoggingPolicyConfig,
    CounterfactualEstimatorConfig
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
    
    def get_model_performance(self) -> Dict[str, float]:
        """Get performance metrics of the trained model."""
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
    
    def estimate_policy_metrics(self) -> Dict[str, Any]:
        """Estimate counterfactual policy metrics.
        
        Returns:
            Dictionary with estimated metrics and confidence intervals
        """
        ...
    
    def estimate_ope_metrics(self, new_actions: np.ndarray) -> Dict[str, Any]:
        """Estimate off-policy evaluation metrics for new actions.
        
        Args:
            new_actions: Array of new policy actions (0=allow, 1=block)
            
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
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, config: ModelConfig) -> BaseEstimator:
        """Train a model on the provided data.
        
        Args:
            X: Feature matrix
            y: Target vector
            config: Model configuration
            
        Returns:
            Trained model instance
        """
        ...
    
    def calculate_performance(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics for a model.
        
        Args:
            model: Trained model
            X: Feature matrix for evaluation
            y: Target vector for evaluation
            
        Returns:
            Dictionary with performance metrics
        """
        ... 