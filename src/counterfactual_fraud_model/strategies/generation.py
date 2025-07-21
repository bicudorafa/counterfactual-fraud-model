"""Concrete data generation strategies implementing the Strategy pattern."""

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.datasets import make_classification
from typing import Dict, Any, Optional, List, Tuple
from pydantic import Field, field_validator, model_validator

from ..core.interfaces import DataGenerationStrategy, DataGenerationConfig


class ProbabilisticGenerationConfig(DataGenerationConfig):
    """Configuration for probabilistic data generation strategy."""
    
    alpha: float = Field(default=0.1, gt=0, description="Alpha parameter for beta distribution")
    beta_param: float = Field(default=2.0, gt=0, description="Beta parameter for beta distribution")
    mean: float = Field(default=0, description="Mean for normal error distribution")
    sd: float = Field(default=0.5, gt=0, description="Standard deviation for normal error distribution")
    
    @field_validator('alpha', 'beta_param', 'sd')
    @classmethod
    def validate_positive_params(cls, v, info):
        """Validate that alpha, beta_param, and sd are positive."""
        field_name = info.field_name
        if v <= 0:
            raise ValueError(f"{field_name} must be positive")
        return v


class SklearnGenerationConfig(DataGenerationConfig):
    """Configuration for sklearn-based data generation strategy."""
    
    n_features: int = Field(default=30, gt=0, description="Total number of features")
    n_informative: int = Field(default=15, gt=0, description="Number of informative features")
    n_redundant: int = Field(default=5, ge=0, description="Number of redundant features")
    n_repeated: int = Field(default=0, ge=0, description="Number of duplicated features")
    n_clusters_per_class: int = Field(default=2, gt=0, description="Number of clusters per class")
    weights: Optional[List[float]] = Field(default=None, description="Class balance weights")
    flip_y: float = Field(default=0.01, ge=0, le=1, description="Fraction of samples with flipped labels")
    class_sep: float = Field(default=1.0, gt=0, description="Factor multiplying the hypercube size")
    
    @field_validator('n_informative')
    @classmethod
    def validate_informative_features(cls, v, info):
        """Validate that n_informative doesn't exceed n_features."""
        # Note: We'll check this in model_validator since we need access to multiple fields
        return v
    
    @field_validator('weights')
    @classmethod
    def validate_weights(cls, v):
        """Set default weights and validate format."""
        if v is None:
            return [0.985, 0.015]
        
        if len(v) != 2:
            raise ValueError("weights must have exactly 2 elements for binary classification")
        
        if not np.isclose(sum(v), 1.0, rtol=1e-5):
            raise ValueError("weights must sum to 1")
        
        return v
    
    @model_validator(mode='after')
    def validate_feature_constraints(self):
        """Validate constraints that require multiple fields."""
        if self.n_informative > self.n_features:
            raise ValueError("n_informative cannot exceed n_features")
        
        return self


class ProbabilisticGenerationStrategy(DataGenerationStrategy):
    """
    Probabilistic data generation using beta distribution and normal noise.
    
    This strategy generates model scores from a beta distribution, adds normal
    noise, and creates fraud labels based on the adjusted probabilities.
    """
    
    def __init__(self, config: ProbabilisticGenerationConfig):
        """Initialize with probabilistic generation configuration."""
        self.config = config
        if config.random_state is not None:
            np.random.seed(config.random_state)
    
    def generate_features_and_labels(self, config: DataGenerationConfig = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate features (model scores) and labels using probabilistic approach.
        
        Args:
            config: Optional config override (uses instance config if None)
            
        Returns:
            Tuple of (model_scores, fraud_labels)
        """
        config = config or self.config
        
        # Generate model scores from beta distribution
        model_scores = beta.rvs(
            a=self.config.alpha,
            b=self.config.beta_param,
            size=config.sample_size,
            random_state=config.random_state
        )
        
        # Generate model error from normal distribution
        model_error = np.random.normal(
            self.config.mean, self.config.sd, config.sample_size
        )
        
        # Sum model error and model score, clip to [0, 1]
        fraud_probabilities = np.clip(model_scores + model_error, 0, 1)
        
        # Generate fraud labels based on adjusted scores using binomial sampling
        fraud_labels = np.random.binomial(1, fraud_probabilities, config.sample_size)
        
        return model_scores, fraud_labels
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the probabilistic generation strategy."""
        return {
            'strategy_type': 'probabilistic',
            'alpha': self.config.alpha,
            'beta_param': self.config.beta_param,
            'noise_mean': self.config.mean,
            'noise_sd': self.config.sd,
            'description': 'Beta distribution + normal noise approach'
        }


class SklearnGenerationStrategy(DataGenerationStrategy):
    """
    Sklearn-based data generation using make_classification.
    
    This strategy creates realistic synthetic classification datasets with
    multiple features and controlled characteristics.
    """
    
    def __init__(self, config: SklearnGenerationConfig):
        """Initialize with sklearn generation configuration."""
        self.config = config
    
    def generate_features_and_labels(self, config: DataGenerationConfig = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate features and labels using sklearn's make_classification.
        
        Args:
            config: Optional config override (uses instance config if None)
            
        Returns:
            Tuple of (features_matrix, fraud_labels)
        """
        config = config or self.config
        
        X, y = make_classification(
            n_samples=config.sample_size,
            n_features=self.config.n_features,
            n_informative=self.config.n_informative,
            n_redundant=self.config.n_redundant,
            n_repeated=self.config.n_repeated,
            n_classes=2,
            n_clusters_per_class=self.config.n_clusters_per_class,
            weights=self.config.weights,
            flip_y=self.config.flip_y,
            class_sep=self.config.class_sep,
            scale=1.0,
            shuffle=True,
            random_state=config.random_state
        )
        
        return X, y
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the sklearn generation strategy."""
        return {
            'strategy_type': 'sklearn_classification',
            'n_features': self.config.n_features,
            'n_informative': self.config.n_informative,
            'n_redundant': self.config.n_redundant,
            'class_weights': self.config.weights,
            'class_separation': self.config.class_sep,
            'label_noise': self.config.flip_y,
            'description': 'Sklearn make_classification approach'
        } 