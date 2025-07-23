"""
Example demonstrating the refactored counterfactual fraud model architecture.

This example shows:
1. How to use the new configuration-based approach
2. Benefits of dependency injection and loose coupling
3. Comparison with the old tightly coupled approach
"""

import pandas as pd
from typing import Dict, Any

# Import the new refactored components
from src.counterfactual_fraud_model import (
    # Configuration models
    DataGeneratorConfig,
    SyntheticDataConfig,
    ModelConfig,
    LoggingPolicyConfig,
    CounterfactualEstimatorConfig,
    PipelineConfig,
    SyntheticOffPolicyEvaluationConfig,
    SyntheticRetrainingConfig,
    RetrainingConfig,
    ModelType,
    
    # Components
    DataGenerator,
    SyntheticDataGenerator,
    LoggingPolicyGenerator,
    CounterfactualEstimator,
    ModelTrainer,
    ModelFactory,
    
    # Pipelines
    SyntheticOffPolicyEvaluationPipeline,
    SyntheticRetrainingPipeline,
    
    # Default instances
    default_model_trainer,
    default_model_factory
)


def example_new_architecture_basic():
    """Example using the new configuration-based architecture."""
    print("=== New Architecture: Basic Usage ===")
    
    # 1. Create configurations (type-safe with validation)
    data_config = SyntheticDataConfig(
        n_samples=10000,
        n_features=20,
        n_informative=10,
        random_state=42
    )
    
    model_config = ModelConfig(
        model_type=ModelType.LIGHTGBM,
        model_params={"n_estimators": 50},
        random_state=42
    )
    
    policy_config = LoggingPolicyConfig(
        cutoff=0.05,
        exploration_rate=0.1,
        random_state=42
    )
    
    estimator_config = CounterfactualEstimatorConfig(
        n_bootstrap=1000,
        random_state=42
    )
    
    # 2. Create components with dependency injection
    data_generator = SyntheticDataGenerator(data_config, model_config)
    policy_generator = LoggingPolicyGenerator(policy_config)
    
    # 3. Generate data and apply policy
    data = data_generator.generate_data()
    policy_data = policy_generator.generate_policy(data)
    
    # 4. Create estimator and evaluate
    estimator = CounterfactualEstimator(estimator_config, policy_data)
    results = estimator.estimate_policy_metrics()
    
    print(f"Data shape: {data.shape}")
    print(f"Model performance: {data_generator.get_model_performance()}")
    print(f"Precision estimate: {results['precision']['estimate']:.4f}")
    print()


def example_new_architecture_pipeline():
    """Example using the new pipeline architecture with dependency injection."""
    print("=== New Architecture: Pipeline with DI ===")
    
    # 1. Create complete configuration
    config = SyntheticOffPolicyEvaluationConfig(
        synthetic_data=SyntheticDataConfig(
            n_samples=10000,
            n_features=15,
            n_informative=8,
            random_state=42
        ),
        model=ModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            model_params={"n_estimators": 100},
            random_state=42
        ),
        logging_policy=LoggingPolicyConfig(
            cutoff=0.03,
            exploration_rate=0.15,
            random_state=42
        ),
        counterfactual_estimator=CounterfactualEstimatorConfig(
            n_bootstrap=1000,
            random_state=42
        )
    )
    
    # 2. Create pipeline (dependencies are injected automatically)
    pipeline = SyntheticOffPolicyEvaluationPipeline(config)
    
    # 3. Run pipeline with optional parameter overrides
    results = pipeline.run_pipeline(
        cutoff=0.04,  # Override config value
        exploration_rate=0.12,  # Override config value
        include_data=False
    )
    
    print(f"Pipeline results available: {list(results.keys())}")
    print(f"Allow rate: {results['statistics']['allow_rate']:.4f}")
    print(f"Fraud rate in allowed: {results['statistics']['fraud_rate_allowed']:.4f}")
    print()


def example_new_architecture_retraining():
    """Example using the new retraining pipeline with composition."""
    print("=== New Architecture: Retraining with Composition ===")
    
    # 1. Create configuration for retraining
    base_config = SyntheticOffPolicyEvaluationConfig(
        synthetic_data=SyntheticDataConfig(
            n_samples=5000,
            n_features=12,
            n_informative=6,
            random_state=42
        ),
        model=ModelConfig(
            model_type=ModelType.LIGHTGBM,
            random_state=42
        )
    )
    
    retraining_config = SyntheticRetrainingConfig(
        base_config=base_config,
        retraining=RetrainingConfig(
            retrain_test_size=0.3,
            classification_threshold=0.06,
            retrain_model=ModelConfig(
                model_type=ModelType.LOGISTIC,
                random_state=42
            )
        )
    )
    
    # 2. Create pipeline with custom model trainer if needed
    custom_trainer = ModelTrainer(default_model_factory)
    pipeline = SyntheticRetrainingPipeline(
        config=retraining_config,
        model_trainer=custom_trainer
    )
    
    # 3. Run retraining pipeline
    results = pipeline.run_pipeline(include_data=False)
    
    print(f"Original model performance: {results['original_model_performance']}")
    print(f"Retrained model performance: {results['retrained_model_performance']}")
    print(f"New model allows {results['statistics']['new_model_allow_rate']:.2%} of transactions")
    print()


def example_dependency_injection():
    """Example showing the benefits of dependency injection."""
    print("=== Dependency Injection Benefits ===")
    
    # Create a custom model factory that only uses logistic regression
    class LogisticOnlyModelFactory:
        def create_model(self, config):
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(random_state=config.random_state, max_iter=1000)
    
    # Create a custom trainer that uses this factory
    custom_factory = LogisticOnlyModelFactory()
    custom_trainer = ModelTrainer(custom_factory)
    
    # Use the custom trainer in a pipeline
    config = SyntheticOffPolicyEvaluationConfig()
    
    # Inject custom synthetic data generator
    custom_data_generator = SyntheticDataGenerator(
        config.synthetic_data,
        ModelConfig(model_type=ModelType.LOGISTIC),  # This will be overridden by our factory
        custom_trainer
    )
    
    pipeline = SyntheticOffPolicyEvaluationPipeline(
        config=config,
        synthetic_data_generator=custom_data_generator
    )
    
    results = pipeline.run_pipeline(include_data=False)
    print(f"Custom pipeline completed with model performance: {results['model_performance']}")
    print()


def compare_old_vs_new():
    """Compare the old tightly coupled approach vs new modular approach."""
    print("=== Comparison: Old vs New Architecture ===")
    
    print("OLD APPROACH ISSUES:")
    print("- Massive parameter lists (20+ parameters in constructors)")
    print("- Tight coupling (SyntheticRetrainingPipeline creates SyntheticOffPolicyEvaluationPipeline)")
    print("- Code duplication (model creation logic repeated)")
    print("- Hard to test (can't inject mock dependencies)")
    print("- Mixed responsibilities (SyntheticDataGenerator does data generation AND model training)")
    print("- No validation (parameters could be invalid)")
    print("- Difficult to extend (adding new model types requires changing multiple files)")
    print()
    
    print("NEW APPROACH BENEFITS:")
    print("- Configuration objects (type-safe, validated, clear structure)")
    print("- Dependency injection (loose coupling, easy testing)")
    print("- Single responsibility (each class has one clear purpose)")
    print("- Composition over inheritance (SyntheticRetrainingPipeline composes base pipeline)")
    print("- Factory pattern (centralized model creation)")
    print("- Protocol-based interfaces (clear contracts, easy mocking)")
    print("- Easy to extend (add new model types in one place)")
    print("- Better error messages (Pydantic validation)")
    print()


if __name__ == "__main__":
    print("Demonstrating Refactored Counterfactual Fraud Model Architecture\n")
    
    example_new_architecture_basic()
    example_new_architecture_pipeline()
    example_new_architecture_retraining()
    example_dependency_injection()
    compare_old_vs_new()
    
    print("Examples completed successfully! 🎉")
    print("\nKey improvements:")
    print("✅ Modular, loosely coupled components")
    print("✅ Type-safe configuration with Pydantic")
    print("✅ Dependency injection for testability")
    print("✅ Single responsibility principle")
    print("✅ Factory pattern for model creation")
    print("✅ Composition over inheritance")
    print("✅ Protocol-based interfaces")
    print("✅ Centralized validation")
    print("✅ Easier to extend and maintain") 