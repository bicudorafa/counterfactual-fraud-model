"""Factory classes for creating model trainers and other components."""

from typing import Dict, Any, Optional
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np

from ..core.interfaces import ModelTrainer


class LightGBMTrainer(ModelTrainer):
    """Model trainer for LightGBM classifier."""
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None, random_state: Optional[int] = None):
        """Initialize LightGBM trainer with parameters."""
        from lightgbm import LGBMClassifier
        
        self.random_state = random_state
        self.model_params = model_params or {}
        self._model = None
        self._training_data_size = None
        
        # Set default parameters
        default_params = {
            'random_state': random_state,
            'verbosity': -1,
            'objective': 'binary',
            'metric': 'binary_logloss'
        }
        default_params.update(self.model_params)
        self.final_params = default_params
    
    def train_model(self, X, y) -> BaseEstimator:
        """Train and return a LightGBM model."""
        from lightgbm import LGBMClassifier
        
        self._model = LGBMClassifier(**self.final_params)
        self._model.fit(X, y)
        self._training_data_size = len(X)
        return self._model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained LightGBM model."""
        if self._model is None:
            raise ValueError("Model must be trained first")
        
        return {
            'model_type': 'lightgbm',
            'parameters': self.final_params,
            'training_samples': self._training_data_size,
            'n_features': self._model.n_features_in_ if hasattr(self._model, 'n_features_in_') else None
        }


class RandomForestTrainer(ModelTrainer):
    """Model trainer for Random Forest classifier."""
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None, random_state: Optional[int] = None):
        """Initialize Random Forest trainer with parameters."""
        from sklearn.ensemble import RandomForestClassifier
        
        self.random_state = random_state
        self.model_params = model_params or {}
        self._model = None
        self._training_data_size = None
        
        # Set default parameters
        default_params = {
            'random_state': random_state,
            'n_estimators': 100
        }
        default_params.update(self.model_params)
        self.final_params = default_params
    
    def train_model(self, X, y) -> BaseEstimator:
        """Train and return a Random Forest model."""
        from sklearn.ensemble import RandomForestClassifier
        
        self._model = RandomForestClassifier(**self.final_params)
        self._model.fit(X, y)
        self._training_data_size = len(X)
        return self._model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained Random Forest model."""
        if self._model is None:
            raise ValueError("Model must be trained first")
        
        return {
            'model_type': 'random_forest',
            'parameters': self.final_params,
            'training_samples': self._training_data_size,
            'n_features': self._model.n_features_in_
        }


class LogisticRegressionTrainer(ModelTrainer):
    """Model trainer for Logistic Regression classifier."""
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None, random_state: Optional[int] = None):
        """Initialize Logistic Regression trainer with parameters."""
        from sklearn.linear_model import LogisticRegression
        
        self.random_state = random_state
        self.model_params = model_params or {}
        self._model = None
        self._training_data_size = None
        
        # Set default parameters
        default_params = {
            'random_state': random_state,
            'max_iter': 1000
        }
        default_params.update(self.model_params)
        self.final_params = default_params
    
    def train_model(self, X, y) -> BaseEstimator:
        """Train and return a Logistic Regression model."""
        from sklearn.linear_model import LogisticRegression
        
        self._model = LogisticRegression(**self.final_params)
        self._model.fit(X, y)
        self._training_data_size = len(X)
        return self._model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained Logistic Regression model."""
        if self._model is None:
            raise ValueError("Model must be trained first")
        
        return {
            'model_type': 'logistic_regression',
            'parameters': self.final_params,
            'training_samples': self._training_data_size,
            'n_features': self._model.n_features_in_
        }


class ModelTrainerFactory:
    """
    Factory for creating model trainers following the Factory pattern.
    
    This design allows adding new model types without modifying existing code,
    following the Open/Closed Principle.
    """
    
    _trainers = {
        'lightgbm': LightGBMTrainer,
        'random_forest': RandomForestTrainer,
        'logistic': LogisticRegressionTrainer,
    }
    
    @classmethod
    def create_trainer(
        self, 
        model_type: str, 
        model_params: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None
    ) -> ModelTrainer:
        """
        Create a model trainer instance.
        
        Args:
            model_type: Type of model trainer to create
            model_params: Additional parameters for the model
            random_state: Random seed for reproducibility
            
        Returns:
            ModelTrainer instance
            
        Raises:
            ValueError: If model_type is not supported
        """
        trainer_class = self._trainers.get(model_type.lower())
        if trainer_class is None:
            available_types = list(self._trainers.keys())
            raise ValueError(f"Unsupported model_type: {model_type}. Available types: {available_types}")
        
        return trainer_class(model_params=model_params, random_state=random_state)
    
    @classmethod
    def register_trainer(cls, model_type: str, trainer_class: type):
        """
        Register a new model trainer type.
        
        This allows extending the factory without modifying existing code.
        
        Args:
            model_type: Name of the model type
            trainer_class: ModelTrainer subclass
        """
        if not issubclass(trainer_class, ModelTrainer):
            raise ValueError("trainer_class must be a subclass of ModelTrainer")
        
        cls._trainers[model_type.lower()] = trainer_class
    
    @classmethod
    def get_available_types(cls) -> list:
        """Get list of available model types."""
        return list(cls._trainers.keys())


class ModelEvaluator:
    """
    Utility class for evaluating trained models.
    
    Separated from model training to follow Single Responsibility Principle.
    """
    
    @staticmethod
    def evaluate_model(model: BaseEstimator, X, y, test_size: float = 0.3, random_state: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate model performance using train/test split.
        
        Args:
            model: Trained scikit-learn model
            X: Features
            y: Labels
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with performance metrics
        """
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_prob),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        return metrics 