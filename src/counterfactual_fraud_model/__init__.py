"""Counterfactual Fraud Model Simulator Package.

A Python package for counterfactual evaluation of fraud detection models.
Implements data generation, logging policies, counterfactual estimation,
and comprehensive simulation capabilities.
"""

from .data_generator import DataGenerator
from .logging_policy import LoggingPolicyGenerator
from .counterfactual_estimator import CounterfactualValuesEstimator
from .pipeline import OffPolicyEvaluationPipeline
from .simulator import OffPolicyEvaluationSimulator

__version__ = "0.1.0"
__author__ = "Counterfactual Fraud Model Team"

__all__ = [
    "DataGenerator",
    "LoggingPolicyGenerator", 
    "CounterfactualValuesEstimator",
    "OffPolicyEvaluationPipeline",
    "OffPolicyEvaluationSimulator",
]



