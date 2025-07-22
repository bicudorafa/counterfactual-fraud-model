"""Factory classes for pipeline components following Factory pattern."""

import pandas as pd
from typing import Dict, Any, Type, Optional
from ..core.interfaces import StatisticsCalculator, PipelineStrategy
from ..strategies.generation import ProbabilisticPipelineStrategy, SyntheticPipelineStrategy


class DefaultStatisticsCalculator(StatisticsCalculator):
    """Default implementation of statistics calculator for pipeline data."""
    
    def calculate_statistics(self, policy_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive summary statistics for the pipeline data.
        
        Args:
            policy_data: DataFrame with policy actions and model predictions
            
        Returns:
            Dictionary containing various statistical summaries
        """
        total_transactions = len(policy_data)
        
        # Original model performance
        original_approval_rate = (policy_data['model_action'] == 'allow').mean()
        original_allowed_data = policy_data[policy_data['model_action'] == 'allow']
        original_fraud_rate = original_allowed_data['is_fraud'].mean() if len(original_allowed_data) > 0 else 0.0
        
        # Policy performance
        policy_approval_rate = (policy_data['policy_action'] == 'allow').mean()
        policy_allowed_data = policy_data[policy_data['policy_action'] == 'allow']
        policy_fraud_rate = policy_allowed_data['is_fraud'].mean() if len(policy_allowed_data) > 0 else 0.0
        
        # Overall statistics
        true_fraud_rate = policy_data['is_fraud'].mean()
        
        # Comparative metrics
        approval_rate_increase = policy_approval_rate/original_approval_rate - 1 if original_approval_rate > 0 else 0
        fraud_rate_increase = policy_fraud_rate/original_fraud_rate - 1 if original_fraud_rate > 0 else 0
        
        return {
            'statistics': {
                'total_transactions': total_transactions,
                'original_approval_rate': original_approval_rate,
                'original_fraud_rate': original_fraud_rate,
                'policy_approval_rate': policy_approval_rate,
                'policy_fraud_rate': policy_fraud_rate,
                'true_fraud_rate': true_fraud_rate,
                'approval_rate_increase': approval_rate_increase,
                'fraud_rate_increase': fraud_rate_increase
            }
        }


class StatisticsCalculatorFactory:
    """Factory for creating statistics calculators."""
    
    _calculators: Dict[str, Type[StatisticsCalculator]] = {
        'default': DefaultStatisticsCalculator,
    }
    
    @classmethod
    def create_calculator(
        self, 
        calculator_type: str = 'default'
    ) -> StatisticsCalculator:
        """
        Create a statistics calculator of the specified type.
        
        Args:
            calculator_type: Type of calculator to create
            
        Returns:
            StatisticsCalculator instance
            
        Raises:
            ValueError: If calculator_type is not registered
        """
        if calculator_type not in self._calculators:
            available = list(self._calculators.keys())
            raise ValueError(f"Unknown calculator type '{calculator_type}'. Available: {available}")
        
        calculator_class = self._calculators[calculator_type]
        return calculator_class()
    
    @classmethod
    def register_calculator(
        cls, 
        calculator_type: str, 
        calculator_class: Type[StatisticsCalculator]
    ) -> None:
        """
        Register a new statistics calculator type.
        
        Args:
            calculator_type: Name for the calculator type
            calculator_class: Calculator class to register
        """
        cls._calculators[calculator_type] = calculator_class
    
    @classmethod
    def get_available_calculators(cls) -> list[str]:
        """Return list of available calculator types."""
        return list(cls._calculators.keys())


class PipelineStrategyFactory:
    """Factory for creating pipeline strategies."""
    
    @classmethod
    def create_probabilistic_strategy(
        cls,
        alpha: float = 0.1,
        beta_param: float = 2.0,
        mean: float = -0.5,
        sd: float = 0.5,
        sample_size: int = 100_000,
        random_state: Optional[int] = None
    ) -> ProbabilisticPipelineStrategy:
        """Create a probabilistic pipeline strategy."""
        return ProbabilisticPipelineStrategy(
            alpha=alpha,
            beta_param=beta_param,
            mean=mean,
            sd=sd,
            sample_size=sample_size,
            random_state=random_state
        )
    
    @classmethod
    def create_synthetic_strategy(
        cls,
        n_samples: int = 100_000,
        n_features: int = 30,
        n_informative: int = 15,
        n_redundant: int = 5,
        n_repeated: int = 0,
        n_clusters_per_class: int = 2,
        weights: Optional[list[float]] = None,
        flip_y: float = 0.01,
        class_sep: float = 1.0,
        model_type: str = "lightgbm",
        model_params: Optional[Dict[str, Any]] = None,
        test_size: float = 0.3,
        random_state: Optional[int] = None
    ) -> SyntheticPipelineStrategy:
        """Create a synthetic ML pipeline strategy."""
        return SyntheticPipelineStrategy(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=n_repeated,
            n_clusters_per_class=n_clusters_per_class,
            weights=weights,
            flip_y=flip_y,
            class_sep=class_sep,
            model_type=model_type,
            model_params=model_params,
            test_size=test_size,
            random_state=random_state
        ) 