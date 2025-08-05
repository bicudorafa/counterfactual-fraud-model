"""Refactored Synthetic Data Generator for Counterfactual Fraud Model Simulation.

This generator creates synthetic fraud datasets using sklearn and manages
the complete dataset generation workflow including model training through
dependency injection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

from ..config import SyntheticDataConfig, ModelConfig
from ..protocols import SyntheticDataGeneratorProtocol, ModelTrainerProtocol
from ..core import default_model_trainer


class SyntheticDataGenerator(SyntheticDataGeneratorProtocol):
    """
    Generates synthetic fraud datasets using sklearn and trains models to produce fraud scores.
    
    Creates realistic synthetic classification datasets, trains machine learning models
    through dependency injection, and extracts model scores for counterfactual evaluation.
    """
    
    def __init__(
        self,
        data_config: SyntheticDataConfig,
        model_config: ModelConfig,
        model_trainer: ModelTrainerProtocol = default_model_trainer
    ):
        """
        Initialize SyntheticDataGenerator with configurations and dependencies.
        
        Args:
            data_config: Configuration for dataset generation
            model_config: Configuration for model training
            model_trainer: Trainer for model creation and training (dependency injection)
        """
        self.data_config = data_config
        self.model_config = model_config
        self.model_trainer = model_trainer
        
        # Internal state
        self._model: Optional[BaseEstimator] = None
        self._train_data: Optional[pd.DataFrame] = None
        self._test_data: Optional[pd.DataFrame] = None
        self._dataset_info: Optional[Dict[str, Any]] = None
    
    def generate_data(self) -> pd.DataFrame:
        """
        Generate synthetic fraud data with trained model scores.
        
        Returns:
            DataFrame with entire test dataset: all features + is_fraud + model_scores
        """
        # Check if test data with scores already exists
        if self._test_data is not None and self._model is not None:
            # Return cached result with entire test dataset plus model scores
            test_features = self._test_data.drop('is_fraud', axis=1)
            test_model_scores = self._model.predict_proba(test_features)[:, 1]
            
            # Create result with all features + is_fraud + model_scores
            result = self._test_data.copy()
            result['model_scores'] = test_model_scores
            return result
        
        # Generate synthetic dataset
        X, y = self._generate_synthetic_dataset()
        
        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.data_config.test_size,
            stratify=y,
            random_state=self.data_config.random_state
        )
        
        # Store train and test data with DataFrame structure
        # Each feature as a separate column
        train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
        train_df['is_fraud'] = y_train
        self._train_data = train_df
        
        test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
        test_df['is_fraud'] = y_test
        self._test_data = test_df
        
        # Train model using injected trainer
        feature_columns = [col for col in train_df.columns if col.startswith('feature_')]
        X_train_df = train_df[feature_columns]
        y_train_series = train_df['is_fraud']
        
        self._model = self.model_trainer.train_model(X_train_df, y_train_series, self.model_config)
        
        # Calculate dataset info
        self._dataset_info = self._calculate_dataset_info(y)
        
        # Get predictions on TEST dataset only
        test_model_scores = self._model.predict_proba(test_df[feature_columns])[:, 1]  # Probability of fraud class
        
        # Return entire test dataset with all features + is_fraud + model_scores
        result = self._test_data.copy()
        result['model_scores'] = test_model_scores
        return result
    
    def get_config(self) -> SyntheticDataConfig:
        """Get the current data configuration."""
        return self.data_config
    
    def get_model(self) -> BaseEstimator:
        """Get the trained model.
        
        Returns:
            The trained machine learning model
            
        Raises:
            ValueError: If model has not been trained yet
        """
        if self._model is None:
            raise ValueError("Model has not been trained yet. Call generate_data() first.")
        return self._model
    

    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the generated dataset."""
        if self._dataset_info is None:
            raise ValueError("Dataset has not been generated yet. Call generate_data() first.")
        return self._dataset_info.copy()
    
    def _generate_synthetic_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic classification dataset using sklearn."""
        X, y = make_classification(
            n_samples=self.data_config.n_samples,
            n_features=self.data_config.n_features,
            n_informative=self.data_config.n_informative,
            n_redundant=self.data_config.n_redundant,
            n_repeated=self.data_config.n_repeated,
            n_classes=2,  # Binary classification for fraud detection
            n_clusters_per_class=self.data_config.n_clusters_per_class,
            weights=self.data_config.weights,
            flip_y=self.data_config.flip_y,
            class_sep=self.data_config.class_sep,
            scale=1.0,
            shuffle=True,
            random_state=self.data_config.random_state
        )
        return X, y
    
    def _calculate_dataset_info(self, y: np.ndarray) -> Dict[str, Any]:
        """Calculate information about the generated dataset."""
        return {
            'n_samples': len(y),
            'n_features': self.data_config.n_features,
            'n_informative': self.data_config.n_informative,
            'fraud_rate': y.mean(),
            'n_fraud': y.sum(),
            'n_legitimate': (y == 0).sum(),
            'class_balance': self.data_config.weights
        } 