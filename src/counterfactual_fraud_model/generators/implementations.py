"""Refactored data generators following SOLID principles and design patterns."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split

from ..core.interfaces import DataGenerator, DataGenerationConfig
from ..strategies.generation import (
    ProbabilisticGenerationStrategy, 
    SklearnGenerationStrategy,
    ProbabilisticGenerationConfig,
    SklearnGenerationConfig
)
from ..factories.model_trainers import ModelTrainerFactory, ModelEvaluator


class ProbabilisticDataGenerator(DataGenerator):
    """
    Refactored probabilistic data generator following SOLID principles.
    
    Uses composition and strategy pattern for clean separation of concerns.
    """
    
    def __init__(self, config: ProbabilisticGenerationConfig):
        """
        Initialize with probabilistic generation configuration.
        
        Args:
            config: Configuration object for probabilistic generation
        """
        super().__init__(config)
        self._strategy = ProbabilisticGenerationStrategy(config)
        self._generated_data = None
    
    def generate_data(self) -> pd.DataFrame:
        """
        Generate synthetic fraud data using probabilistic approach.
        
        Returns:
            DataFrame with columns: model_scores, is_fraud
        """
        if self._generated_data is not None:
            return self._generated_data.copy()
        
        # Use strategy to generate raw data
        model_scores, fraud_labels = self._strategy.generate_features_and_labels(self.config)
        
        # Create DataFrame with required format
        self._generated_data = pd.DataFrame({
            'model_scores': model_scores,
            'is_fraud': fraud_labels
        })
        
        return self._generated_data.copy()
    
    def get_generation_info(self) -> Dict[str, Any]:
        """Get information about the probabilistic data generation."""
        if self._generated_data is None:
            raise ValueError("Must call generate_data() first")
        
        strategy_info = self._strategy.get_strategy_info()
        fraud_rate = np.mean(self._generated_data['is_fraud'])
        
        return {
            **strategy_info,
            'total_samples': len(self._generated_data),
            'fraud_rate': fraud_rate,
            'fraud_count': np.sum(self._generated_data['is_fraud']),
            'legitimate_count': len(self._generated_data) - np.sum(self._generated_data['is_fraud'])
        }
    
    def regenerate_data(self) -> None:
        """Force regeneration of data with current parameters."""
        self._generated_data = None


class MLModelDataGenerator(DataGenerator):
    """
    Refactored ML model-based data generator following SOLID principles.
    
    Uses composition to separate data generation, model training, and evaluation
    responsibilities into distinct components.
    """
    
    def __init__(
        self, 
        generation_config: SklearnGenerationConfig,
        model_type: str = "lightgbm",
        model_params: Optional[Dict[str, Any]] = None,
        test_size: float = 0.3
    ):
        """
        Initialize with sklearn generation configuration and model parameters.
        
        Args:
            generation_config: Configuration object for sklearn data generation
            model_type: Type of model to train for generating scores
            model_params: Additional parameters for the model
            test_size: Fraction of data to use for model testing
        """
        super().__init__(generation_config)
        self._generation_strategy = SklearnGenerationStrategy(generation_config)
        self._model_trainer = ModelTrainerFactory.create_trainer(
            model_type=model_type,
            model_params=model_params,
            random_state=generation_config.random_state
        )
        self._model_evaluator = ModelEvaluator()
        self._test_size = test_size
        
        # Internal state
        self._raw_features = None
        self._raw_labels = None
        self._trained_model = None
        self._generated_data = None
    
    def generate_data(self) -> pd.DataFrame:
        """
        Generate synthetic fraud data with trained model scores.
        
        Returns:
            DataFrame with columns: model_scores, is_fraud
        """
        if self._generated_data is not None:
            return self._generated_data.copy()
        
        # Step 1: Generate raw dataset using strategy
        features, labels = self._generation_strategy.generate_features_and_labels(self.config)
        self._raw_features, self._raw_labels = features, labels
        
        # Step 2: Split data for training
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels,
            test_size=self._test_size,
            stratify=labels,
            random_state=self.config.random_state
        )
        
        # Step 3: Train model using trainer
        self._trained_model = self._model_trainer.train_model(X_train, y_train)
        
        # Step 4: Get model scores for full dataset
        model_scores = self._trained_model.predict_proba(features)[:, 1]
        
        # Step 5: Create DataFrame with required format
        self._generated_data = pd.DataFrame({
            'model_scores': model_scores,
            'is_fraud': labels
        })
        
        return self._generated_data.copy()
    
    def get_generation_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the ML data generation."""
        if self._generated_data is None:
            raise ValueError("Must call generate_data() first")
        
        # Get strategy information
        strategy_info = self._generation_strategy.get_strategy_info()
        
        # Get model information
        model_info = self._model_trainer.get_model_info()
        
        # Calculate dataset statistics
        fraud_rate = np.mean(self._generated_data['is_fraud'])
        dataset_info = {
            'total_samples': len(self._generated_data),
            'fraud_rate': fraud_rate,
            'fraud_count': np.sum(self._generated_data['is_fraud']),
            'legitimate_count': len(self._generated_data) - np.sum(self._generated_data['is_fraud'])
        }
        
        return {
            'generation_strategy': strategy_info,
            'model_info': model_info,
            'dataset_info': dataset_info
        }
    
    def get_model_performance(self) -> Dict[str, float]:
        """
        Get model performance metrics using separate evaluator.
        
        Returns:
            Dictionary with performance metrics
        """
        if self._trained_model is None or self._raw_features is None:
            raise ValueError("Must call generate_data() first to train model")
        
        return self._model_evaluator.evaluate_model(
            model=self._trained_model,
            X=self._raw_features,
            y=self._raw_labels,
            test_size=self._test_size,
            random_state=self.config.random_state
        )
    
    def regenerate_data(self) -> None:
        """Force regeneration of data with current parameters."""
        self._generated_data = None
        self._raw_features = None
        self._raw_labels = None
        self._trained_model = None


class DataGeneratorFactory:
    """
    Factory for creating data generators.
    
    Provides a consistent interface for creating different types of data generators
    while hiding the implementation details.
    """
    
    @staticmethod
    def create_probabilistic_generator(
        alpha: float = 0.1,
        beta_param: float = 2.0,
        mean: float = 0,
        sd: float = 0.5,
        sample_size: int = 100_000,
        random_state: Optional[int] = None
    ) -> ProbabilisticDataGenerator:
        """
        Create a probabilistic data generator with specified parameters.
        
        Args:
            alpha: Alpha parameter for beta distribution
            beta_param: Beta parameter for beta distribution  
            mean: Mean for normal error distribution
            sd: Standard deviation for normal error distribution
            sample_size: Number of samples to generate
            random_state: Random seed for reproducibility
            
        Returns:
            ProbabilisticDataGenerator instance
        """
        config = ProbabilisticGenerationConfig(
            alpha=alpha,
            beta_param=beta_param,
            mean=mean,
            sd=sd,
            sample_size=sample_size,
            random_state=random_state
        )
        return ProbabilisticDataGenerator(config)
    
    @staticmethod
    def create_ml_model_generator(
        n_samples: int = 100_000,
        n_features: int = 30,
        n_informative: int = 15,
        n_redundant: int = 5,
        model_type: str = "lightgbm",
        model_params: Optional[Dict[str, Any]] = None,
        test_size: float = 0.3,
        random_state: Optional[int] = None,
        **sklearn_params
    ) -> MLModelDataGenerator:
        """
        Create an ML model-based data generator with specified parameters.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Total number of features
            n_informative: Number of informative features
            n_redundant: Number of redundant features
            model_type: Type of model to train
            model_params: Additional parameters for the model
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            **sklearn_params: Additional sklearn make_classification parameters
            
        Returns:
            MLModelDataGenerator instance
        """
        # Set up sklearn generation config
        sklearn_config_params = {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_informative': n_informative,
            'n_redundant': n_redundant,
            'random_state': random_state,
            **sklearn_params
        }
        
        generation_config = SklearnGenerationConfig(**sklearn_config_params)
        
        return MLModelDataGenerator(
            generation_config=generation_config,
            model_type=model_type,
            model_params=model_params,
            test_size=test_size
        ) 