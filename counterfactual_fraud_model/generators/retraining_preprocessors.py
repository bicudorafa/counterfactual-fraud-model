"""Retraining data preprocessing strategies for counterfactual fraud model.

This module provides different strategies for preprocessing policy data
before retraining models, supporting both filtering and weighting approaches.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

from ..protocols import RetrainingDataPreprocessorProtocol
from ..config import RetrainingStrategy


class FilteringDataPreprocessor(RetrainingDataPreprocessorProtocol):
    """Data preprocessor that filters to only allowed transactions (current implementation)."""
    
    def __init__(self, strategy_params: Dict[str, Any] = None):
        """
        Initialize the filtering preprocessor.
        
        Args:
            strategy_params: Additional parameters for the filtering strategy
        """
        self.strategy_params = strategy_params or {}
        self._last_preprocessing_info = None
    
    def prepare_training_data(
        self, 
        policy_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[np.ndarray]]:
        """
        Filter policy data to only allowed transactions and extract features/targets.
        
        Args:
            policy_data: Full policy data with features, actions, and propensity scores
            
        Returns:
            Tuple of (features_df, target_series, None) - no sample weights for filtering
        """
        # Filter to only allowed transactions by the model (model_action == 'allow'), not by the policy (policy_action == 'allow'). The goal is to mimic what the policy would do.
        allowed_data = policy_data[policy_data['model_action'] == 'allow'].copy()
        
        if len(allowed_data) == 0:
            raise ValueError("No transactions with model_action == 'allow' found. Cannot retrain model.")
        
        # Extract feature columns
        feature_columns = [col for col in allowed_data.columns 
                          if col.startswith('feature_') or col.startswith('x')]
        if len(feature_columns) == 0:
            raise ValueError("No feature columns found in data. Expected columns starting with 'feature_' or 'x'")
        
        X = allowed_data[feature_columns]
        y = allowed_data['is_fraud']
        
        # Store info for logging
        self._last_preprocessing_info = {
            'strategy': 'filtering',
            'original_samples': len(policy_data),
            'filtered_samples': len(allowed_data),
            'filter_ratio': len(allowed_data) / len(policy_data),
            'feature_count': len(feature_columns),
            'fraud_rate': y.mean(),
            'strategy_params': self.strategy_params
        }
        
        return X, y, None  # No sample weights for filtering strategy
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the last preprocessing operation."""
        if self._last_preprocessing_info is None:
            raise ValueError("No preprocessing has been performed yet")
        return self._last_preprocessing_info.copy()


class WeightingDataPreprocessor(RetrainingDataPreprocessorProtocol):
    """Data preprocessor that filters to allowed transactions and applies inverse propensity weighting."""
    
    def __init__(self, strategy_params: Dict[str, Any] = None):
        """
        Initialize the weighting preprocessor.
        
        Args:
            strategy_params: Parameters for weighting strategy. Supported:
                - min_weight: Minimum weight value to prevent extreme weights (default: 0.01)
                - max_weight: Maximum weight value to prevent extreme weights (default: 100.0)
        """
        self.strategy_params = strategy_params or {}
        self.min_weight = self.strategy_params.get('min_weight', 0.01)
        self.max_weight = self.strategy_params.get('max_weight', 100.0)
        self._last_preprocessing_info = None
    
    def prepare_training_data(
        self, 
        policy_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[np.ndarray]]:
        """
        Filter to allowed transactions and apply inverse propensity weighting.
        
        Args:
            policy_data: Full policy data with features, actions, and propensity scores
            
        Returns:
            Tuple of (features_df, target_series, weights)
        """
        # Filter to only allowed transactions by the policy (policy_action == 'allow')
        allowed_data = policy_data[policy_data['policy_action'] == 'allow'].copy()
        
        if len(allowed_data) == 0:
            raise ValueError("No transactions with model_action == 'allow' found. Cannot retrain model.")

        # Extract feature columns
        feature_columns = [col for col in allowed_data.columns 
                          if col.startswith('feature_') or col.startswith('x')]
        if len(feature_columns) == 0:
            raise ValueError("No feature columns found in data. Expected columns starting with 'feature_' or 'x'")
        
        # Check for required columns
        if 'propensity_score' not in allowed_data.columns:
            raise ValueError("propensity_score column required for weighting strategy")
        if 'model_action' not in allowed_data.columns:
            raise ValueError("model_action column required for weighting strategy")
        
        X = allowed_data[feature_columns]
        y = allowed_data['is_fraud']
        
        # Calculate sample weights based on inverse propensity scores
        weights = 1.0 / allowed_data['propensity_score']
        # Clip weights to prevent extreme values
        weights = np.clip(weights, self.min_weight, self.max_weight)
        
        # Store info for logging
        self._last_preprocessing_info = {
            'strategy': 'weighting',
            'original_samples': len(policy_data),
            'filtered_samples': len(allowed_data),
            'filter_ratio': len(allowed_data) / len(policy_data),
            'feature_count': len(feature_columns),
            'fraud_rate': y.mean(),
            'weight_stats': {
                'min_weight': float(weights.min()),
                'max_weight': float(weights.max()),
                'mean_weight': float(weights.mean()),
                'std_weight': float(weights.std())
            },
            'strategy_params': self.strategy_params
        }
        
        return X, y, weights
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the last preprocessing operation."""
        if self._last_preprocessing_info is None:
            raise ValueError("No preprocessing has been performed yet")
        return self._last_preprocessing_info.copy()


def create_preprocessor(
    strategy: RetrainingStrategy,
    strategy_params: Dict[str, Any] = None
) -> RetrainingDataPreprocessorProtocol:
    """
    Factory function to create appropriate preprocessor based on strategy.
    
    Args:
        strategy: The retraining strategy to use
        strategy_params: Additional parameters for the strategy
        
    Returns:
        Configured preprocessor instance
    """
    if strategy == RetrainingStrategy.FILTERING:
        return FilteringDataPreprocessor(strategy_params)
    elif strategy == RetrainingStrategy.WEIGHTING:
        return WeightingDataPreprocessor(strategy_params)
    else:
        raise ValueError(f"Unsupported retraining strategy: {strategy}") 