"""Model trainer component for training ML models.

Separates model training concerns from data generation to follow
single responsibility principle.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score

from ..config import ModelConfig
from ..protocols import ModelTrainerProtocol, ModelFactoryProtocol
from ..factories import default_model_factory


class ModelTrainer(ModelTrainerProtocol):
    """Component responsible for training ML models and calculating performance metrics."""
    
    def __init__(self, model_factory: ModelFactoryProtocol = default_model_factory):
        """Initialize ModelTrainer.
        
        Args:
            model_factory: Factory for creating model instances
        """
        self.model_factory = model_factory
    
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
        model = self.model_factory.create_model(config)
        
        # Check if model supports sample weights
        if sample_weight is not None:
            try:
                model.fit(X, y, sample_weight=sample_weight)
            except TypeError:
                # Model doesn't support sample weights, train without them
                print(f"Warning: {type(model).__name__} does not support sample weights. Training without weights.")
                model.fit(X, y)
        else:
            model.fit(X, y)
        
        return model
    
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
        # Get probabilities
        y_pred_proba = model.predict_proba(X)[:, 1]  # Probability of positive class
        
        # Get binary predictions based on threshold or model default
        if threshold is not None:
            y_pred = (y_pred_proba >= threshold).astype(int)
        else:
            y_pred = model.predict(X)
        
        # Calculate metrics with zero_division handling
        performance = {
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba),  # ROC-AUC uses probabilities, not binary predictions
            'average_precision': average_precision_score(y, y_pred_proba)  # Average Precision uses probabilities
        }
        
        return performance


# Default trainer instance
default_model_trainer = ModelTrainer() 