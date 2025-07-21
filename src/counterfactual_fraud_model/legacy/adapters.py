"""Backward compatibility adapters for existing API."""

import pandas as pd
from typing import Dict, Any, Optional

from ..generators.implementations import DataGeneratorFactory
from ..strategies.generation import ProbabilisticGenerationConfig, SklearnGenerationConfig


class LegacyDataGenerator:
    """
    Backward compatibility adapter for the original DataGenerator.
    
    Maintains the exact same API as the original while using the new
    refactored structure internally.
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        beta_param: float = 2.0,
        mean: float = 0,
        sd: float = 0.5,
        sample_size: int = 100_000,
        random_state: Optional[int] = None
    ):
        """
        Initialize LegacyDataGenerator with same parameters as original.
        
        Args:
            alpha: Alpha parameter for beta distribution
            beta_param: Beta parameter for beta distribution  
            mean: Mean for normal error distribution
            sd: Standard deviation for normal error distribution
            sample_size: Number of samples to generate
            random_state: Random seed for reproducibility
        """
        # Store original parameters for compatibility
        self.alpha = alpha
        self.beta_param = beta_param
        self.mean = mean
        self.sd = sd
        self.sample_size = sample_size
        self.random_state = random_state
        
        # Create new generator internally
        self._generator = DataGeneratorFactory.create_probabilistic_generator(
            alpha=alpha,
            beta_param=beta_param,
            mean=mean,
            sd=sd,
            sample_size=sample_size,
            random_state=random_state
        )
    
    def generate_data(self) -> pd.DataFrame:
        """
        Generate synthetic fraud model data.
        
        Returns:
            DataFrame with columns: model_scores, is_fraud
            (Same format as original DataGenerator)
        """
        return self._generator.generate_data()
    
    def get_params(self) -> dict:
        """Return the current parameters (original API)."""
        return {
            'alpha': self.alpha,
            'beta_param': self.beta_param,
            'mean': self.mean,
            'sd': self.sd,
            'sample_size': self.sample_size,
            'random_state': self.random_state
        }


class LegacySyntheticDataGenerator:
    """
    Backward compatibility adapter for the original SyntheticDataGenerator.
    
    Maintains the exact same API as the original while using the new
    refactored structure internally.
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
        Initialize LegacySyntheticDataGenerator with same parameters as original.
        
        Args:
            (Same as original SyntheticDataGenerator)
        """
        # Store original parameters for compatibility
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_redundant = n_redundant
        self.n_repeated = n_repeated
        self.n_classes = n_classes
        self.n_clusters_per_class = n_clusters_per_class
        self.weights = weights or [0.985, 0.015]
        self.flip_y = flip_y
        self.class_sep = class_sep
        self.model_type = model_type
        self.model_params = model_params or {}
        self.test_size = test_size
        self.random_state = random_state
        
        # Validate like original
        self._validate_params()
        
        # Create new generator internally
        self._generator = DataGeneratorFactory.create_ml_model_generator(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=n_repeated,
            n_clusters_per_class=n_clusters_per_class,
            weights=self.weights,
            flip_y=flip_y,
            class_sep=class_sep,
            model_type=model_type,
            model_params=model_params,
            test_size=test_size,
            random_state=random_state
        )
    
    def _validate_params(self) -> None:
        """Validate initialization parameters (same as original)."""
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
        if not abs(sum(self.weights) - 1.0) < 1e-6:
            raise ValueError("weights must sum to 1")
    
    def generate_data(self) -> pd.DataFrame:
        """
        Generate synthetic fraud data with trained model scores.
        
        Returns:
            DataFrame with columns: model_scores, is_fraud
            (Same format as original SyntheticDataGenerator)
        """
        return self._generator.generate_data()
    
    def get_model_performance(self) -> Dict[str, float]:
        """
        Get basic performance metrics of the trained model.
        
        Returns:
            Dictionary with performance metrics (same as original)
        """
        return self._generator.get_model_performance()
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the generated dataset.
        
        Returns:
            Dictionary with dataset information (same format as original)
        """
        generation_info = self._generator.get_generation_info()
        dataset_info = generation_info['dataset_info']
        
        # Transform to match original format exactly
        return {
            'total_samples': dataset_info['total_samples'],
            'fraud_rate': dataset_info['fraud_rate'],
            'fraud_count': dataset_info['fraud_count'],
            'legitimate_count': dataset_info['legitimate_count'],
            'n_features': self.n_features,
            'model_type': self.model_type
        }
    
    def get_params(self) -> Dict[str, Any]:
        """Return the current parameters (same as original)."""
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
        self._generator.regenerate_data() 