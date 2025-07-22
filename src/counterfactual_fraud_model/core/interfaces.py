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


class PipelineConfig(BaseModel):
    """Base configuration for all pipeline types."""
    n_bootstrap: int = 5000
    random_state: Optional[int] = None
    
    class Config:
        extra = "forbid"


class PipelineStrategy(ABC):
    """Abstract base class for pipeline data generation strategies."""
    
    @abstractmethod
    def generate_data(self) -> pd.DataFrame:
        """Generate data using the specific strategy."""
        pass
    
    @abstractmethod
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return information about the strategy."""
        pass


class StatisticsCalculator(ABC):
    """Abstract base class for calculating pipeline statistics."""
    
    @abstractmethod
    def calculate_statistics(self, policy_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive statistics for policy data."""
        pass


class BaseOffPolicyEvaluationPipeline(ABC):
    """
    Abstract base class for all off-policy evaluation pipelines.
    
    Implements the Template Method pattern for the pipeline workflow.
    """
    
    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        self._strategy: Optional[PipelineStrategy] = None
        self._statistics_calculator: Optional[StatisticsCalculator] = None
    
    @abstractmethod
    def _create_strategy(self) -> PipelineStrategy:
        """Create the data generation strategy (Template Method step)."""
        pass
    
    def _create_logging_policy_generator(self, cutoff: float, exploration_rate: float):
        """Create logging policy generator (shared implementation)."""
        from ..logging_policy import LoggingPolicyGenerator
        return LoggingPolicyGenerator(
            cutoff=cutoff,
            exploration_rate=exploration_rate,
            random_state=self.config.random_state
        )
    
    def _create_counterfactual_estimator(self, policy_data: pd.DataFrame):
        """Create counterfactual estimator (shared implementation)."""
        from ..counterfactual_estimator import CounterfactualValuesEstimator
        return CounterfactualValuesEstimator(
            data=policy_data,
            n_bootstrap=self.config.n_bootstrap,
            random_state=self.config.random_state,
        )
    
    def _ensure_strategy(self) -> None:
        """Ensure strategy is initialized (lazy initialization)."""
        if self._strategy is None:
            self._strategy = self._create_strategy()
    
    def _ensure_statistics_calculator(self) -> None:
        """Ensure statistics calculator is initialized."""
        if self._statistics_calculator is None:
            from ..factories.pipeline_components import StatisticsCalculatorFactory
            self._statistics_calculator = StatisticsCalculatorFactory.create_calculator()
    
    @property
    def generated_data(self) -> pd.DataFrame:
        """Get generated data (lazy loading)."""
        self._ensure_strategy()
        return self._strategy.generate_data()
    
    def run_pipeline(
        self,
        cutoff: float = 0.05,
        exploration_rate: float = 0.05,
        include_data: bool = True
    ) -> Dict[str, Any]:
        """
        Execute the complete off-policy evaluation pipeline (Template Method).
        
        This method defines the algorithm skeleton that concrete classes follow.
        """
        # Use Pydantic for validation instead of ValidationUtils
        from ..strategies.generation import PipelineExecutionConfig
        exec_config = PipelineExecutionConfig(
            cutoff=cutoff,
            exploration_rate=exploration_rate,
            include_data=include_data
        )
        
        # Template Method: Step 1 - Get data using strategy
        data = self.generated_data
        
        # Template Method: Step 2 - Apply logging policy
        logging_policy_generator = self._create_logging_policy_generator(exec_config.cutoff, exec_config.exploration_rate)
        policy_data = logging_policy_generator.generate_policy(data)
        
        # Template Method: Step 3 - Estimate counterfactual metrics
        estimator = self._create_counterfactual_estimator(policy_data)
        ope_metrics_results = estimator.estimate_policy_metrics()
        
        # Template Method: Step 4 - Calculate summary statistics
        self._ensure_statistics_calculator()
        summary_results = self._statistics_calculator.calculate_statistics(policy_data)
        
        # Template Method: Step 5 - Compile results
        return self._compile_results(
            summary_results, ope_metrics_results, exec_config.cutoff, exec_config.exploration_rate, 
            policy_data if exec_config.include_data else None
        )
    
    @abstractmethod
    def _compile_results(
        self, 
        summary_results: Dict[str, Any], 
        ope_metrics_results: Dict[str, Any],
        cutoff: float,
        exploration_rate: float,
        policy_data: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Compile final results (Template Method step - allows customization)."""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return current parameters."""
        pass
    
    def regenerate_data(self) -> None:
        """Force regeneration of data."""
        if self._strategy is not None:
            # Reset strategy to force regeneration
            self._strategy = None 