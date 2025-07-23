"""Model trainer component for training ML models.

Separates model training concerns from data generation to follow
single responsibility principle.
"""

import pandas as pd
from typing import Dict
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

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
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, config: ModelConfig) -> BaseEstimator:
        """Train a model on the provided data.
        
        Args:
            X: Feature matrix
            y: Target vector
            config: Model configuration
            
        Returns:
            Trained model instance
        """
        model = self.model_factory.create_model(config)
        model.fit(X, y)
        return model
    
    def calculate_performance(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics for a model.
        
        Args:
            model: Trained model
            X: Feature matrix for evaluation
            y: Target vector for evaluation
            
        Returns:
            Dictionary with performance metrics
        """
        # Get predictions and probabilities
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]  # Probability of positive class
        
        # Calculate metrics with zero_division handling
        performance = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        return performance


# Default trainer instance
default_model_trainer = ModelTrainer() 