"""Counterfactual Fraud Model Simulator Package.

A Python package for counterfactual evaluation of fraud detection models.
Implements data generation, logging policies, counterfactual estimation,
and comprehensive simulation capabilities.

This package includes refactored components following SOLID principles:
- Base classes and interfaces for extensibility (core/)
- Strategy pattern for different data generation approaches (strategies/)
- Factory pattern for model creation (factories/)
- Composition over inheritance for better modularity (generators/)
- Backward compatibility adapters to maintain existing API (legacy/)

The new modular structure improves readability and maintainability while
preserving complete backward compatibility.
"""

# Original API (maintained for backward compatibility)
from .data_generator import DataGenerator
from .synthetic_data_generator import SyntheticDataGenerator

# Backward compatibility adapters
from .legacy import LegacyDataGenerator, LegacySyntheticDataGenerator

# New refactored API - organized by submodules
from .core import DataGenerator as BaseDataGenerator, DataGenerationConfig
from .strategies import (
    ProbabilisticGenerationConfig, 
    SklearnGenerationConfig,
    ProbabilisticGenerationStrategy,
    SklearnGenerationStrategy
)
from .factories import ModelTrainerFactory, ModelEvaluator
from .generators import (
    ProbabilisticDataGenerator,
    MLModelDataGenerator, 
    DataGeneratorFactory
)

# Pipeline and other components remain the same
from .logging_policy import LoggingPolicyGenerator
from .counterfactual_estimator import CounterfactualValuesEstimator
from .pipeline import OffPolicyEvaluationPipeline
from .synthetic_pipeline import SyntheticOffPolicyEvaluationPipeline

# New refactored pipeline implementations with SOLID principles
from .generators.pipeline_implementations import (
    RefactoredOffPolicyEvaluationPipeline,
    RefactoredSyntheticOffPolicyEvaluationPipeline,
    PipelineFactory
)

# Pipeline configuration objects
from .strategies.generation import (
    ProbabilisticPipelineConfig,
    SyntheticPipelineConfig,
    PipelineExecutionConfig
)

# Pipeline factories and utilities
from .factories.pipeline_components import (
    StatisticsCalculatorFactory,
    PipelineStrategyFactory
)

__version__ = "0.3.0"
__author__ = "Counterfactual Fraud Model Team"

# Backward compatibility exports
__all__ = [
    # Original backward-compatible API
    "DataGenerator",
    "SyntheticDataGenerator", 
    "OffPolicyEvaluationPipeline",
    "SyntheticOffPolicyEvaluationPipeline",
    "LoggingPolicyGenerator",
    "CounterfactualValuesEstimator",
    
    # Refactored modular components
    "ProbabilisticGenerationConfig",
    "SklearnGenerationConfig", 
    "ProbabilisticDataGenerator",
    "MLModelDataGenerator",
    "DataGeneratorFactory",
    "ModelTrainerFactory",
    "LightGBMTrainer",
    "RandomForestTrainer", 
    "LogisticRegressionTrainer",
    "ModelEvaluator",
    
    # New refactored pipeline implementations
    "RefactoredOffPolicyEvaluationPipeline",
    "RefactoredSyntheticOffPolicyEvaluationPipeline",
    "PipelineFactory",
    "ProbabilisticPipelineConfig",
    "SyntheticPipelineConfig",
    "PipelineExecutionConfig",
    "StatisticsCalculatorFactory",
    "PipelineStrategyFactory",
    
    # Legacy adapters for backward compatibility
    "LegacyDataGenerator",
    "LegacySyntheticDataGenerator"
]



