"""Concrete data generation strategies implementing the Strategy pattern."""

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.datasets import make_classification
from typing import Dict, Any, Optional, List, Tuple
from pydantic import Field, field_validator, model_validator
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from ..core.interfaces import PipelineConfig

from ..core.interfaces import DataGenerationStrategy, DataGenerationConfig, PipelineStrategy


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


class ProbabilisticPipelineStrategy(PipelineStrategy):
    """Pipeline strategy for probabilistic data generation."""
    
    def __init__(
        self,
        alpha: float = 0.1,
        beta_param: float = 2.0,
        mean: float = -0.5,
        sd: float = 0.5,
        sample_size: int = 100_000,
        random_state: Optional[int] = None
    ):
        self._alpha = alpha
        self._beta_param = beta_param
        self._mean = mean
        self._sd = sd
        self._sample_size = sample_size
        self._random_state = random_state
        self._data_generator = None
        self._generated_data = None
    
    def generate_data(self) -> pd.DataFrame:
        """Generate data using probabilistic approach."""
        if self._generated_data is None:
            self._ensure_data_generator()
            self._generated_data = self._data_generator.generate_data()
        return self._generated_data.copy()
    
    def _ensure_data_generator(self) -> None:
        """Ensure data generator is initialized."""
        if self._data_generator is None:
            from ..generators.implementations import DataGeneratorFactory
            self._data_generator = DataGeneratorFactory.create_probabilistic_generator(
                alpha=self._alpha,
                beta_param=self._beta_param,
                mean=self._mean,
                sd=self._sd,
                sample_size=self._sample_size,
                random_state=self._random_state
            )
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy information."""
        return {
            'strategy_type': 'probabilistic',
            'alpha': self._alpha,
            'beta_param': self._beta_param,
            'mean': self._mean,
            'sd': self._sd,
            'sample_size': self._sample_size,
            'random_state': self._random_state
        }


class SyntheticPipelineStrategy(PipelineStrategy):
    """Pipeline strategy for ML model-based synthetic data generation."""
    
    def __init__(
        self,
        n_samples: int = 100_000,
        n_features: int = 30,
        n_informative: int = 15,
        n_redundant: int = 5,
        n_repeated: int = 0,
        n_clusters_per_class: int = 2,
        weights: Optional[List[float]] = None,
        flip_y: float = 0.01,
        class_sep: float = 1.0,
        model_type: str = "lightgbm",
        model_params: Optional[Dict[str, Any]] = None,
        test_size: float = 0.3,
        random_state: Optional[int] = None
    ):
        self._n_samples = n_samples
        self._n_features = n_features
        self._n_informative = n_informative
        self._n_redundant = n_redundant
        self._n_repeated = n_repeated
        self._n_clusters_per_class = n_clusters_per_class
        self._weights = weights or [0.985, 0.015]
        self._flip_y = flip_y
        self._class_sep = class_sep
        self._model_type = model_type
        self._model_params = model_params or {}
        self._test_size = test_size
        self._random_state = random_state
        self._synthetic_data_generator = None
        self._generated_data = None
    
    def generate_data(self) -> pd.DataFrame:
        """Generate synthetic data with ML model."""
        if self._generated_data is None:
            self._ensure_synthetic_data_generator()
            self._generated_data = self._synthetic_data_generator.generate_data()
        return self._generated_data.copy()
    
    def _ensure_synthetic_data_generator(self) -> None:
        """Ensure synthetic data generator is initialized."""
        if self._synthetic_data_generator is None:
            from ..generators.implementations import DataGeneratorFactory
            self._synthetic_data_generator = DataGeneratorFactory.create_ml_model_generator(
                n_samples=self._n_samples,
                n_features=self._n_features,
                n_informative=self._n_informative,
                n_redundant=self._n_redundant,
                n_repeated=self._n_repeated,
                n_clusters_per_class=self._n_clusters_per_class,
                weights=self._weights,
                flip_y=self._flip_y,
                class_sep=self._class_sep,
                model_type=self._model_type,
                model_params=self._model_params,
                test_size=self._test_size,
                random_state=self._random_state
            )
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy information."""
        return {
            'strategy_type': 'synthetic_ml',
            'n_samples': self._n_samples,
            'n_features': self._n_features,
            'n_informative': self._n_informative,
            'n_redundant': self._n_redundant,
            'n_repeated': self._n_repeated,
            'n_clusters_per_class': self._n_clusters_per_class,
            'weights': self._weights,
            'flip_y': self._flip_y,
            'class_sep': self._class_sep,
            'model_type': self._model_type,
            'model_params': self._model_params,
            'test_size': self._test_size,
            'random_state': self._random_state
        }
    
    def get_model_performance(self) -> Dict[str, float]:
        """Get performance metrics of the trained model."""
        self._ensure_synthetic_data_generator()
        if self._generated_data is None:
            _ = self.generate_data()  # Trigger data generation
        return self._synthetic_data_generator.get_model_performance()
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the generated synthetic dataset."""
        self._ensure_synthetic_data_generator()
        if self._generated_data is None:
            _ = self.generate_data()  # Trigger data generation
        return self._synthetic_data_generator.get_dataset_info()
    
    def regenerate_data(self) -> None:
        """Force regeneration of synthetic data."""
        if self._synthetic_data_generator is not None:
            self._synthetic_data_generator.regenerate_data()
        self._generated_data = None 


class ProbabilisticPipelineConfig(PipelineConfig):
    """
    Pydantic configuration for probabilistic pipeline parameters.
    
    Provides validation, type safety, and JSON serialization capabilities.
    """
    
    # Data generation parameters
    alpha: float = Field(
        default=0.1, 
        gt=0, 
        description="Alpha parameter for beta distribution in data generation"
    )
    beta_param: float = Field(
        default=2.0, 
        gt=0, 
        description="Beta parameter for beta distribution in data generation"
    )
    mean: float = Field(
        default=-0.5, 
        description="Mean for normal error distribution in data generation"
    )
    sd: float = Field(
        default=0.5, 
        gt=0, 
        description="Standard deviation for normal error distribution in data generation"
    )
    sample_size: int = Field(
        default=100_000, 
        gt=0, 
        description="Number of samples to generate"
    )
    
    class Config:
        extra = "forbid"
        json_schema_extra = {
            "example": {
                "alpha": 0.1,
                "beta_param": 2.0,
                "mean": -0.5,
                "sd": 0.5,
                "sample_size": 100_000,
                "n_bootstrap": 5000,
                "random_state": 42
            }
        }


class SyntheticPipelineConfig(PipelineConfig):
    """
    Pydantic configuration for synthetic ML pipeline parameters.
    
    Provides validation, type safety, and JSON serialization capabilities.
    """
    
    # Synthetic data generation parameters
    n_samples: int = Field(
        default=100_000, 
        gt=0, 
        description="Number of samples to generate"
    )
    n_features: int = Field(
        default=30, 
        gt=0, 
        description="Total number of features"
    )
    n_informative: int = Field(
        default=15, 
        gt=0, 
        description="Number of informative features"
    )
    n_redundant: int = Field(
        default=5, 
        ge=0, 
        description="Number of redundant features"
    )
    n_repeated: int = Field(
        default=0, 
        ge=0, 
        description="Number of duplicated features"
    )
    n_clusters_per_class: int = Field(
        default=2, 
        gt=0, 
        description="Number of clusters per class"
    )
    weights: Optional[List[float]] = Field(
        default=[0.985, 0.015], 
        description="Class balance (e.g., [0.99, 0.01] for imbalanced fraud data)"
    )
    flip_y: float = Field(
        default=0.01, 
        ge=0, 
        le=1, 
        description="Fraction of samples whose class is flipped (label noise)"
    )
    class_sep: float = Field(
        default=1.0, 
        gt=0, 
        description="Factor multiplying the hypercube size"
    )
    
    # Model parameters
    model_type: str = Field(
        default="lightgbm", 
        description="Type of model to train ('lightgbm', 'random_forest', 'logistic')"
    )
    model_params: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Additional parameters for the model"
    )
    test_size: float = Field(
        default=0.3, 
        gt=0, 
        lt=1, 
        description="Fraction of data to use for testing model"
    )
    
    @model_validator(mode='after')
    def validate_informative_features(self):
        """Ensure n_informative doesn't exceed n_features."""
        if self.n_informative > self.n_features:
            raise ValueError("n_informative cannot exceed n_features")
        return self
    
    @field_validator('weights')
    @classmethod 
    def validate_weights(cls, v):
        """Ensure weights sum to 1 and are positive."""
        if v is not None:
            if len(v) != 2:
                raise ValueError("weights must contain exactly 2 values for binary classification")
            if any(w <= 0 for w in v):
                raise ValueError("all weights must be positive")
            if not abs(sum(v) - 1.0) < 1e-6:
                raise ValueError("weights must sum to 1.0")
        return v
    
    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v):
        """Ensure model_type is supported."""
        allowed_types = ["lightgbm", "random_forest", "logistic"]
        if v not in allowed_types:
            raise ValueError(f"model_type must be one of {allowed_types}")
        return v
    
    class Config:
        extra = "forbid"
        json_schema_extra = {
            "example": {
                "n_samples": 100_000,
                "n_features": 30,
                "n_informative": 15,
                "n_redundant": 5,
                "n_repeated": 0,
                "n_clusters_per_class": 2,
                "weights": [0.985, 0.015],
                "flip_y": 0.01,
                "class_sep": 1.0,
                "model_type": "lightgbm",
                "model_params": {},
                "test_size": 0.3,
                "n_bootstrap": 5000,
                "random_state": 42
            }
        } 


class PipelineExecutionConfig(BaseModel):
    """
    Pydantic configuration for pipeline execution parameters.
    
    Replaces ValidationUtils with proper Pydantic validation.
    """
    
    cutoff: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Score threshold for the logging policy (0-1)"
    )
    exploration_rate: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Rate of exploration for blocked transactions (0-1)"
    )
    include_data: bool = Field(
        default=True,
        description="Whether to include the full dataset in results"
    )
    
    class Config:
        extra = "forbid"
        json_schema_extra = {
            "example": {
                "cutoff": 0.05,
                "exploration_rate": 0.05,
                "include_data": True
            }
        } 