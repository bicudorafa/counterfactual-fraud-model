"""Counterfactual Fraud Model Simulator Package.

A Python package for counterfactual evaluation of fraud detection models.
Implements data generation, logging policies, counterfactual estimation,
and comprehensive simulation capabilities.

This refactored version provides:
- Pydantic configuration models for type safety
- Dependency injection for loose coupling
- Protocol-based interfaces for testability
- Separation of concerns for maintainability
"""

# Configuration models
from .config import (
    ModelType,
    PropensityType,
    DataGeneratorConfig,
    SyntheticDataConfig,
    ModelConfig,
    LoggingPolicyConfig,
    CounterfactualEstimatorConfig,
    PipelineConfig,
    OffPolicyEvaluationConfig,
    SyntheticOffPolicyEvaluationConfig,
    RetrainingConfig,
    SyntheticRetrainingConfig
)

# Protocols for dependency injection
from .protocols import (
    DataGeneratorProtocol,
    SyntheticDataGeneratorProtocol,
    ModelFactoryProtocol,
    LoggingPolicyGeneratorProtocol,
    CounterfactualEstimatorProtocol,
    PipelineProtocol,
    ModelTrainerProtocol
)

# Core components
from .core import ModelTrainer, default_model_trainer
from .factories import ModelFactory, default_model_factory

# Generators
from .generators import (
    DataGenerator,
    SyntheticDataGenerator,
    LoggingPolicyGenerator
)

# Estimators  
from .estimators import CounterfactualEstimator

# Pipelines
from .pipelines import (
    OffPolicyEvaluationPipeline,
    SyntheticOffPolicyEvaluationPipeline,
    SyntheticRetrainingPipeline
)

# Legacy imports for backward compatibility
from .legacy.data_generator import DataGenerator as LegacyDataGenerator
from .legacy.synthetic_data_generator import SyntheticDataGenerator as LegacySyntheticDataGenerator
from .legacy.logging_policy import LoggingPolicyGenerator as LegacyLoggingPolicyGenerator
from .legacy.counterfactual_estimator import CounterfactualValuesEstimator
from .legacy.pipeline import OffPolicyEvaluationPipeline as LegacyOffPolicyEvaluationPipeline
from .legacy.synthetic_pipeline import SyntheticOffPolicyEvaluationPipeline as LegacySyntheticOffPolicyEvaluationPipeline
from .legacy.synthetic_retraining_pipeline import SyntheticRetrainingPipeline as LegacySyntheticRetrainingPipeline

__version__ = "0.2.0"
__author__ = "Counterfactual Fraud Model Team"

# Main exports (new architecture)
__all__ = [
    # Configuration
    "ModelType",
    "PropensityType", 
    "DataGeneratorConfig",
    "SyntheticDataConfig",
    "ModelConfig",
    "LoggingPolicyConfig",
    "CounterfactualEstimatorConfig",
    "PipelineConfig",
    "OffPolicyEvaluationConfig",
    "SyntheticOffPolicyEvaluationConfig",
    "RetrainingConfig",
    "SyntheticRetrainingConfig",
    
    # Protocols
    "DataGeneratorProtocol",
    "SyntheticDataGeneratorProtocol",
    "ModelFactoryProtocol",
    "LoggingPolicyGeneratorProtocol", 
    "CounterfactualEstimatorProtocol",
    "PipelineProtocol",
    "ModelTrainerProtocol",
    
    # Core components
    "ModelTrainer",
    "default_model_trainer",
    "ModelFactory",
    "default_model_factory",
    
    # Generators
    "DataGenerator",
    "SyntheticDataGenerator",
    "LoggingPolicyGenerator",
    
    # Estimators
    "CounterfactualEstimator",
    
    # Pipelines
    "OffPolicyEvaluationPipeline",
    "SyntheticOffPolicyEvaluationPipeline",
    "SyntheticRetrainingPipeline",
    
    # Legacy compatibility
    "CounterfactualValuesEstimator",
    "LegacyDataGenerator",
    "LegacySyntheticDataGenerator", 
    "LegacyLoggingPolicyGenerator",
    "LegacyOffPolicyEvaluationPipeline",
    "LegacySyntheticOffPolicyEvaluationPipeline",
    "LegacySyntheticRetrainingPipeline",
]



