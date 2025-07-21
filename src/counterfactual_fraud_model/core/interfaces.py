"""Base classes and interfaces for counterfactual fraud model components."""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class DataGenerationConfig(BaseModel):
    """Configuration object for data generation parameters."""
    
    sample_size: int = Field(default=100_000, gt=0, description="Number of samples to generate")
    random_state: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    
    @field_validator('sample_size')
    @classmethod
    def validate_sample_size(cls, v):
        """Validate that sample_size is positive."""
        if v <= 0:
            raise ValueError("sample_size must be positive")
        return v


class DataGenerator(ABC):
    """
    Abstract base class for data generators in counterfactual fraud evaluation.
    
    Defines the contract that all data generators must follow, ensuring
    consistent interface and output format for downstream components.
    """
    
    def __init__(self, config: DataGenerationConfig):
        """
        Initialize data generator with configuration.
        
        Args:
            config: Configuration object containing generation parameters
        """
        self.config = config
    
    @abstractmethod
    def generate_data(self) -> pd.DataFrame:
        """
        Generate synthetic fraud data.
        
        Returns:
            DataFrame with columns: model_scores, is_fraud
            - model_scores: Fraud probability scores (0-1)
            - is_fraud: Binary fraud labels (0/1)
        """
        pass
    
    @abstractmethod
    def get_generation_info(self) -> Dict[str, Any]:
        """
        Get information about the data generation process.
        
        Returns:
            Dictionary containing generation metadata
        """
        pass
    
    def get_config(self) -> DataGenerationConfig:
        """Return the current configuration."""
        return self.config


class ModelTrainer(ABC):
    """Abstract base class for model training strategies."""
    
    @abstractmethod
    def train_model(self, X, y):
        """Train and return a model."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        pass


class DataGenerationStrategy(ABC):
    """Abstract base class for data generation strategies."""
    
    @abstractmethod
    def generate_features_and_labels(self, config: DataGenerationConfig):
        """Generate raw features and labels."""
        pass
    
    @abstractmethod
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the generation strategy."""
        pass 