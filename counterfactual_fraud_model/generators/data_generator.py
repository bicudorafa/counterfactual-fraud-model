"""Refactored Data Generator for Counterfactual Fraud Model Simulation.

This generator creates basic fraud detection datasets using beta distributions
and noise generation. Configured via Pydantic models for type safety.
"""

import numpy as np
import pandas as pd
from scipy.stats import beta
from typing import Dict, Optional
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score

from ..config import DataGeneratorConfig
from ..protocols import DataGeneratorProtocol


class DataGenerator(DataGeneratorProtocol):
    """
    Generates synthetic fraud model data for counterfactual evaluation.
    
    Creates model scores from beta distribution, adds noise, and generates
    fraud labels based on the resulting scores.
    """
    
    def __init__(self, config: DataGeneratorConfig):
        """
        Initialize DataGenerator with configuration.
        
        Args:
            config: Configuration object containing all generation parameters
        """
        self.config = config
        self._model_performance: Optional[Dict[str, float]] = None
        
        # Set random seed if provided
        if config.random_state is not None:
            np.random.seed(config.random_state)
    
    def generate_data(self) -> pd.DataFrame:
        """
        Generate synthetic fraud model data.
        
        Returns:
            DataFrame with columns: model_scores, is_fraud
        """
        # Generate model scores from beta distribution
        model_scores = beta.rvs(
            a=self.config.alpha, 
            b=self.config.beta_param, 
            size=self.config.sample_size,
            random_state=self.config.random_state
        )
        
        # Generate model error from normal distribution
        model_error = np.random.normal(
            self.config.mean, self.config.sd, self.config.sample_size
        )
        
        # Sum model error and model score, clip to [0, 1]
        fraud_probabilities = np.clip(model_scores + model_error, 0, 1)
        
        # Generate fraud labels based on adjusted scores using binomial sampling
        # Higher adjusted scores should have higher probability of fraud
        fraud = np.random.binomial(1, fraud_probabilities, self.config.sample_size)
        
        # Create DataFrame
        data = pd.DataFrame({
            'model_scores': model_scores,
            'is_fraud': fraud
        })
        
        # Calculate and store performance metrics
        self._model_performance = self._calculate_performance(data)
        
        return data
    
    def get_config(self) -> DataGeneratorConfig:
        """Get the current configuration."""
        return self.config
    
    def get_model_performance(self) -> Dict[str, float]:
        """Get performance metrics of the generated model scores."""
        if self._model_performance is None:
            raise ValueError("Data has not been generated yet. Call generate_data() first.")
        return self._model_performance.copy()
    
    def _calculate_performance(self, data: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
        """Calculate performance metrics from generated data.
        
        Args:
            data: DataFrame with model_scores and is_fraud columns
            threshold: Threshold for converting probabilities to binary predictions
            
        Returns:
            Dictionary with performance metrics
        """
        y_true = data['is_fraud']
        y_pred_proba = data['model_scores']
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics with zero_division handling
        performance = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'average_precision': average_precision_score(y_true, y_pred_proba)
        }
        
        return performance 