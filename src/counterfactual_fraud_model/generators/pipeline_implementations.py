"""Refactored pipeline implementations following SOLID principles and design patterns."""

import pandas as pd
from typing import Dict, Any, Optional, List
from ..core.interfaces import BaseOffPolicyEvaluationPipeline, PipelineStrategy
from ..strategies.generation import (
    ProbabilisticPipelineConfig, 
    SyntheticPipelineConfig,
    ProbabilisticPipelineStrategy,
    SyntheticPipelineStrategy
)


class RefactoredOffPolicyEvaluationPipeline(BaseOffPolicyEvaluationPipeline):
    """
    Refactored off-policy evaluation pipeline using probabilistic data generation.
    
    Follows SOLID principles and implements Template Method pattern.
    """
    
    def __init__(self, config: ProbabilisticPipelineConfig):
        """
        Initialize pipeline with probabilistic configuration.
        
        Args:
            config: Pydantic configuration object with validation
        """
        super().__init__(config)
        self._probabilistic_config = config
    
    def _create_strategy(self) -> PipelineStrategy:
        """Create probabilistic data generation strategy."""
        return ProbabilisticPipelineStrategy(
            alpha=self._probabilistic_config.alpha,
            beta_param=self._probabilistic_config.beta_param,
            mean=self._probabilistic_config.mean,
            sd=self._probabilistic_config.sd,
            sample_size=self._probabilistic_config.sample_size,
            random_state=self._probabilistic_config.random_state
        )
    
    def _compile_results(
        self, 
        summary_results: Dict[str, Any], 
        ope_metrics_results: Dict[str, Any],
        cutoff: float,
        exploration_rate: float,
        policy_data: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Compile final results for probabilistic pipeline."""
        # Get strategy info for additional context
        strategy_info = self._strategy.get_strategy_info() if self._strategy else {}
        
        # Compile parameters used in this run
        run_parameters = self._probabilistic_config.model_dump()
        run_parameters.update({
            'cutoff': cutoff,
            'exploration_rate': exploration_rate
        })
        
        # Compile results
        results = {
            **summary_results,
            'ope_metrics': ope_metrics_results,
            'strategy_info': strategy_info,
            'parameters': run_parameters
        }
        
        if policy_data is not None:
            results['data'] = policy_data
            
        return results
    
    def get_params(self) -> Dict[str, Any]:
        """Return current parameters."""
        return self._probabilistic_config.model_dump()


class RefactoredSyntheticOffPolicyEvaluationPipeline(BaseOffPolicyEvaluationPipeline):
    """
    Refactored synthetic off-policy evaluation pipeline using ML model-based data generation.
    
    Follows SOLID principles and implements Template Method pattern.
    """
    
    def __init__(self, config: SyntheticPipelineConfig):
        """
        Initialize pipeline with synthetic configuration.
        
        Args:
            config: Pydantic configuration object with validation
        """
        super().__init__(config)
        self._synthetic_config = config
    
    def _create_strategy(self) -> PipelineStrategy:
        """Create synthetic ML data generation strategy."""
        return SyntheticPipelineStrategy(
            n_samples=self._synthetic_config.n_samples,
            n_features=self._synthetic_config.n_features,
            n_informative=self._synthetic_config.n_informative,
            n_redundant=self._synthetic_config.n_redundant,
            n_repeated=self._synthetic_config.n_repeated,
            n_clusters_per_class=self._synthetic_config.n_clusters_per_class,
            weights=self._synthetic_config.weights,
            flip_y=self._synthetic_config.flip_y,
            class_sep=self._synthetic_config.class_sep,
            model_type=self._synthetic_config.model_type,
            model_params=self._synthetic_config.model_params,
            test_size=self._synthetic_config.test_size,
            random_state=self._synthetic_config.random_state
        )
    
    def _compile_results(
        self, 
        summary_results: Dict[str, Any], 
        ope_metrics_results: Dict[str, Any],
        cutoff: float,
        exploration_rate: float,
        policy_data: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Compile final results for synthetic pipeline."""
        # Get strategy info and additional ML model information
        strategy_info = self._strategy.get_strategy_info() if self._strategy else {}
        
        # Get model performance and dataset info if available
        model_performance = {}
        dataset_info = {}
        if isinstance(self._strategy, SyntheticPipelineStrategy):
            try:
                model_performance = self._strategy.get_model_performance()
                dataset_info = self._strategy.get_dataset_info()
            except Exception:
                # Handle cases where model info is not available yet
                pass
        
        # Compile parameters used in this run
        run_parameters = self._synthetic_config.model_dump()
        run_parameters.update({
            'cutoff': cutoff,
            'exploration_rate': exploration_rate
        })
        
        # Compile results
        results = {
            **summary_results,
            'ope_metrics': ope_metrics_results,
            'strategy_info': strategy_info,
            'model_performance': model_performance,
            'dataset_info': dataset_info,
            'parameters': run_parameters
        }
        
        if policy_data is not None:
            results['data'] = policy_data
            
        return results
    
    def get_params(self) -> Dict[str, Any]:
        """Return current parameters."""
        return self._synthetic_config.model_dump()
    
    def get_model_performance(self) -> Dict[str, float]:
        """Get performance metrics of the trained model."""
        self._ensure_strategy()
        if isinstance(self._strategy, SyntheticPipelineStrategy):
            return self._strategy.get_model_performance()
        return {}
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the generated synthetic dataset."""
        self._ensure_strategy()
        if isinstance(self._strategy, SyntheticPipelineStrategy):
            return self._strategy.get_dataset_info()
        return {}


class PipelineFactory:
    """
    Factory for creating different types of pipelines.
    
    Implements Factory pattern for centralized pipeline creation.
    """
    
    @classmethod
    def create_probabilistic_pipeline(
        cls,
        config: ProbabilisticPipelineConfig
    ) -> RefactoredOffPolicyEvaluationPipeline:
        """
        Create a probabilistic off-policy evaluation pipeline from configuration.
        
        Args:
            config: Pydantic configuration object with all parameters
            
        Returns:
            Configured probabilistic pipeline
        """
        return RefactoredOffPolicyEvaluationPipeline(config)
    
    @classmethod
    def create_synthetic_pipeline(
        cls,
        config: SyntheticPipelineConfig
    ) -> RefactoredSyntheticOffPolicyEvaluationPipeline:
        """
        Create a synthetic ML off-policy evaluation pipeline from configuration.
        
        Args:
            config: Pydantic configuration object with all parameters
            
        Returns:
            Configured synthetic pipeline
        """
        return RefactoredSyntheticOffPolicyEvaluationPipeline(config)
    
    # Convenience methods for backward compatibility and simple creation
    @classmethod
    def create_probabilistic_pipeline_simple(
        cls,
        alpha: float = 0.1,
        beta_param: float = 2.0,
        mean: float = -0.5,
        sd: float = 0.5,
        sample_size: int = 100_000,
        n_bootstrap: int = 5000,
        random_state: Optional[int] = None,
    ) -> RefactoredOffPolicyEvaluationPipeline:
        """
        Create a probabilistic pipeline with individual parameters (convenience method).
        
        For simple use cases where you don't want to create a config object first.
        """
        config = ProbabilisticPipelineConfig(
            alpha=alpha,
            beta_param=beta_param,
            mean=mean,
            sd=sd,
            sample_size=sample_size,
            n_bootstrap=n_bootstrap,
            random_state=random_state
        )
        return cls.create_probabilistic_pipeline(config)
    
    @classmethod
    def create_synthetic_pipeline_simple(
        cls,
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
        n_bootstrap: int = 5000,
        random_state: Optional[int] = None,
    ) -> RefactoredSyntheticOffPolicyEvaluationPipeline:
        """
        Create a synthetic pipeline with individual parameters (convenience method).
        
        For simple use cases where you don't want to create a config object first.
        """
        config = SyntheticPipelineConfig(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=n_repeated,
            n_clusters_per_class=n_clusters_per_class,
            weights=weights or [0.985, 0.015],
            flip_y=flip_y,
            class_sep=class_sep,
            model_type=model_type,
            model_params=model_params or {},
            test_size=test_size,
            n_bootstrap=n_bootstrap,
            random_state=random_state
        )
        return cls.create_synthetic_pipeline(config)
    
    @classmethod
    def create_from_config(
        cls, 
        config: Dict[str, Any], 
        pipeline_type: str = "auto"
    ) -> BaseOffPolicyEvaluationPipeline:
        """
        Create pipeline from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            pipeline_type: Type of pipeline ("probabilistic", "synthetic", "auto")
            
        Returns:
            Configured pipeline
        """
        if pipeline_type == "auto":
            # Auto-detect based on presence of specific parameters
            if "alpha" in config or "beta_param" in config:
                pipeline_type = "probabilistic"
            elif "n_features" in config or "model_type" in config:
                pipeline_type = "synthetic"
            else:
                raise ValueError("Cannot auto-detect pipeline type from config")
        
        if pipeline_type == "probabilistic":
            validated_config = ProbabilisticPipelineConfig(**config)
            return cls.create_probabilistic_pipeline(validated_config)
        elif pipeline_type == "synthetic":
            validated_config = SyntheticPipelineConfig(**config)
            return cls.create_synthetic_pipeline(validated_config)
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}") 