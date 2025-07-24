"""Configuration models for counterfactual fraud model components.

This module defines Pydantic models for type-safe configuration of all
components in the counterfactual fraud model simulator.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class ModelType(str, Enum):
    """Supported machine learning model types."""
    
    LIGHTGBM = "lightgbm"
    RANDOM_FOREST = "random_forest"
    LOGISTIC = "logistic"


class PropensityType(str, Enum):
    """Supported propensity score calculation types."""
    
    UNIFORM = "uniform"
    # Future: ADAPTIVE, BAYESIAN, etc.


class DataGeneratorConfig(BaseModel):
    """Configuration for basic data generation."""
    
    alpha: float = Field(default=0.1, gt=0, description="Alpha parameter for beta distribution")
    beta_param: float = Field(default=2.0, gt=0, description="Beta parameter for beta distribution") 
    mean: float = Field(default=-0.5, description="Mean for normal error distribution")
    sd: float = Field(default=0.5, gt=0, description="Standard deviation for normal error distribution")
    sample_size: int = Field(default=10_000, gt=0, description="Number of samples to generate")
    random_state: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class SyntheticDataConfig(BaseModel):
    """Configuration for synthetic dataset generation."""
    
    n_samples: int = Field(default=10_000, gt=0, description="Number of samples to generate")
    n_features: int = Field(default=30, gt=0, description="Total number of features")
    n_informative: int = Field(default=15, gt=0, description="Number of informative features")
    n_redundant: int = Field(default=5, ge=0, description="Number of redundant features")
    n_repeated: int = Field(default=0, ge=0, description="Number of duplicated features")
    n_clusters_per_class: int = Field(default=2, gt=0, description="Number of clusters per class")
    weights: List[float] = Field(default=[0.985, 0.015], description="Class balance weights")
    flip_y: float = Field(default=0.01, ge=0, le=1, description="Fraction of samples with flipped class")
    class_sep: float = Field(default=1.0, gt=0, description="Factor multiplying the hypercube size")
    test_size: float = Field(default=0.5, gt=0, lt=1, description="Fraction of data for testing")
    random_state: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    
    @field_validator('n_informative')
    @classmethod
    def validate_informative_features(cls, v, info):
        """Ensure n_informative does not exceed n_features."""
        if 'n_features' in info.data and v > info.data['n_features']:
            raise ValueError("n_informative cannot exceed n_features")
        return v
    
    @field_validator('weights')
    @classmethod
    def validate_weights(cls, v):
        """Ensure weights sum to 1 and have exactly 2 elements."""
        if len(v) != 2:
            raise ValueError("weights must have exactly 2 elements for binary classification")
        if not abs(sum(v) - 1.0) < 1e-6:
            raise ValueError("weights must sum to 1")
        return v


class ModelConfig(BaseModel):
    """Configuration for machine learning models."""
    
    model_type: ModelType = Field(default=ModelType.LIGHTGBM, description="Type of model to use")
    model_params: Dict[str, Any] = Field(default_factory=dict, description="Additional model parameters")
    random_state: Optional[int] = Field(default=None, description="Random seed for model reproducibility")


class LoggingPolicyConfig(BaseModel):
    """Configuration for logging policy generation."""
    
    cutoff: float = Field(default=0.1, ge=0, le=1, description="Score threshold for blocking transactions")
    exploration_rate: float = Field(default=0.05, ge=0, le=1, description="Rate of exploration for blocked transactions")
    propensity_type: PropensityType = Field(default=PropensityType.UNIFORM, description="Type of propensity function")
    random_state: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class CounterfactualEstimatorConfig(BaseModel):
    """Configuration for counterfactual estimation."""
    
    n_bootstrap: int = Field(default=5000, gt=0, description="Number of bootstrap repetitions")
    random_state: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class PipelineConfig(BaseModel):
    """Configuration for pipeline execution."""
    
    include_data: bool = Field(default=True, description="Whether to include full dataset in results")


class OffPolicyEvaluationConfig(BaseModel):
    """Complete configuration for off-policy evaluation pipeline."""
    
    data_generator: DataGeneratorConfig = Field(default_factory=DataGeneratorConfig)
    logging_policy: LoggingPolicyConfig = Field(default_factory=LoggingPolicyConfig)
    counterfactual_estimator: CounterfactualEstimatorConfig = Field(default_factory=CounterfactualEstimatorConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


class SyntheticOffPolicyEvaluationConfig(BaseModel):
    """Complete configuration for synthetic off-policy evaluation pipeline."""
    
    synthetic_data: SyntheticDataConfig = Field(default_factory=SyntheticDataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    logging_policy: LoggingPolicyConfig = Field(default_factory=LoggingPolicyConfig)
    counterfactual_estimator: CounterfactualEstimatorConfig = Field(default_factory=CounterfactualEstimatorConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


class RetrainingConfig(BaseModel):
    """Configuration for model retraining."""
    
    retrain_test_size: float = Field(default=0.5, gt=0, lt=1, description="Fraction of allowed data for testing retrained model")
    classification_threshold: float = Field(default=0.1, ge=0, le=1, description="Threshold for converting scores to binary predictions")
    retrain_model: ModelConfig = Field(default_factory=ModelConfig, description="Configuration for retrained model")


class SyntheticRetrainingConfig(BaseModel):
    """Complete configuration for synthetic retraining pipeline."""
    
    base_config: SyntheticOffPolicyEvaluationConfig = Field(default_factory=SyntheticOffPolicyEvaluationConfig)
    retraining: RetrainingConfig = Field(default_factory=RetrainingConfig) 