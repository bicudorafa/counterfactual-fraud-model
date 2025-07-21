"""Synthetic Data Generator for Counterfactual Fraud Model Simulation."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from lightgbm import LGBMClassifier


class SyntheticDataGenerator:
    """
    Generates synthetic fraud datasets using sklearn and trains models to produce fraud scores.
    
    Creates realistic synthetic classification datasets, trains machine learning models,
    and extracts model scores for counterfactual evaluation.
    """
    
    def __init__(
        self,
        # Dataset generation parameters
        n_samples: int = 100_000,
        n_features: int = 30,
        n_informative: int = 15,
        n_redundant: int = 5,
        n_repeated: int = 0,
        n_classes: int = 2,
        n_clusters_per_class: int = 2,
        weights: list = None,
        flip_y: float = 0.01,
        class_sep: float = 1.0,
        # Model parameters
        model_type: str = "lightgbm",
        model_params: Optional[Dict[str, Any]] = None,
        test_size: float = 0.3,
        random_state: Optional[int] = None,
    ):
        """
        Initialize SyntheticDataGenerator.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Total number of features
            n_informative: Number of informative features
            n_redundant: Number of redundant features
            n_repeated: Number of duplicated features
            n_classes: Number of classes (should be 2 for fraud detection)
            n_clusters_per_class: Number of clusters per class
            weights: Class balance (e.g., [0.99, 0.01] for imbalanced fraud data)
            flip_y: Fraction of samples whose class is flipped (label noise)
            class_sep: Factor multiplying the hypercube size
            model_type: Type of model to train ("lightgbm", "random_forest", "logistic")
            model_params: Additional parameters for the model
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        # Dataset parameters
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_redundant = n_redundant
        self.n_repeated = n_repeated
        self.n_classes = n_classes
        self.n_clusters_per_class = n_clusters_per_class
        self.weights = weights or [0.985, 0.015]  # Default: imbalanced fraud dataset
        self.flip_y = flip_y
        self.class_sep = class_sep
        
        # Model parameters
        self.model_type = model_type
        self.model_params = model_params or {}
        self.test_size = test_size
        self.random_state = random_state
        
        # Validation
        self._validate_params()
        
        # Internal state
        self._X = None
        self._y = None
        self._model = None
        self._generated_data = None
        
    def _validate_params(self) -> None:
        """Validate initialization parameters."""
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.n_features <= 0:
            raise ValueError("n_features must be positive")
        if self.n_informative > self.n_features:
            raise ValueError("n_informative cannot exceed n_features")
        if self.n_classes != 2:
            raise ValueError("n_classes must be 2 for fraud detection")
        if not 0 <= self.test_size <= 1:
            raise ValueError("test_size must be between 0 and 1")
        if not 0 <= self.flip_y <= 1:
            raise ValueError("flip_y must be between 0 and 1")
        if len(self.weights) != 2:
            raise ValueError("weights must have exactly 2 elements for binary classification")
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("weights must sum to 1")
    
    def _create_model(self) -> BaseEstimator:
        """Create and return a model instance based on model_type."""
        if self.model_type.lower() == "lightgbm":
            default_params = {
                'random_state': self.random_state,
                'verbosity': -1,
                'objective': 'binary',
                'metric': 'binary_logloss'
            }
            default_params.update(self.model_params)
            return LGBMClassifier(**default_params)
            
        elif self.model_type.lower() == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            default_params = {
                'random_state': self.random_state,
                'n_estimators': 100
            }
            default_params.update(self.model_params)
            return RandomForestClassifier(**default_params)
            
        elif self.model_type.lower() == "logistic":
            from sklearn.linear_model import LogisticRegression
            default_params = {
                'random_state': self.random_state,
                'max_iter': 1000
            }
            default_params.update(self.model_params)
            return LogisticRegression(**default_params)
            
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
    
    def _generate_synthetic_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic classification dataset."""
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_redundant=self.n_redundant,
            n_repeated=self.n_repeated,
            n_classes=self.n_classes,
            n_clusters_per_class=self.n_clusters_per_class,
            weights=self.weights,
            flip_y=self.flip_y,
            class_sep=self.class_sep,
            scale=1.0,
            shuffle=True,
            random_state=self.random_state
        )
        return X, y
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> BaseEstimator:
        """Train the model on the training data."""
        model = self._create_model()
        model.fit(X_train, y_train)
        return model
    
    def generate_data(self) -> pd.DataFrame:
        """
        Generate synthetic fraud data with trained model scores.
        
        Returns:
            DataFrame with columns: model_scores, is_fraud
            Compatible with LoggingPolicyGenerator and existing pipeline
        """
        if self._generated_data is not None:
            return self._generated_data.copy()
        
        # Generate synthetic dataset
        X, y = self._generate_synthetic_dataset()
        self._X, self._y = X, y
        
        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state
        )
        
        # Train model
        self._model = self._train_model(X_train, y_train)
        
        # Get predictions on the full dataset (both train and test)
        # This simulates having model scores for all transactions
        model_scores = self._model.predict_proba(X)[:, 1]  # Probability of fraud class
        
        # Create DataFrame with required format
        self._generated_data = pd.DataFrame({
            'model_scores': model_scores,
            'is_fraud': y
        })
        
        return self._generated_data.copy()
    
    def get_model_performance(self) -> Dict[str, float]:
        """
        Get basic performance metrics of the trained model.
        
        Returns:
            Dictionary with performance metrics
        """
        if self._model is None or self._X is None or self._y is None:
            raise ValueError("Must call generate_data() first to train model")
        
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
        
        # Split data again for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            self._X, self._y,
            test_size=self.test_size,
            stratify=self._y,
            random_state=self.random_state
        )
        
        # Get predictions
        y_pred = self._model.predict(X_test)
        y_prob = self._model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_prob),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        return metrics
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the generated dataset.
        
        Returns:
            Dictionary with dataset information
        """
        if self._y is None:
            raise ValueError("Must call generate_data() first")
        
        fraud_rate = np.mean(self._y)
        
        return {
            'total_samples': len(self._y),
            'fraud_rate': fraud_rate,
            'fraud_count': np.sum(self._y),
            'legitimate_count': len(self._y) - np.sum(self._y),
            'n_features': self.n_features,
            'model_type': self.model_type
        }
    
    def get_params(self) -> Dict[str, Any]:
        """Return the current parameters."""
        return {
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'n_informative': self.n_informative,
            'n_redundant': self.n_redundant,
            'n_repeated': self.n_repeated,
            'n_classes': self.n_classes,
            'n_clusters_per_class': self.n_clusters_per_class,
            'weights': self.weights,
            'flip_y': self.flip_y,
            'class_sep': self.class_sep,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'test_size': self.test_size,
            'random_state': self.random_state
        }
    
    def regenerate_data(self) -> None:
        """Force regeneration of synthetic data with current parameters."""
        self._generated_data = None
        self._X = None
        self._y = None
        self._model = None
        # Data will be regenerated on next call to generate_data() 