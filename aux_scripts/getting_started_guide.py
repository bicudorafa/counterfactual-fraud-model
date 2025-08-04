#!/usr/bin/env python3
"""
🚀 COUNTERFACTUAL FRAUD MODEL - GETTING STARTED GUIDE
====================================================

This comprehensive guide introduces newcomers to the counterfactual fraud model project.
It covers the main concepts, configuration classes, and pipelines with practical examples.

The framework is designed for off-policy evaluation of fraud detection models, allowing you to:
- Generate synthetic fraud data with realistic characteristics
- Apply logging policies (e.g., block high-risk transactions)  
- Estimate counterfactual metrics using different models and strategies
- Compare model performance and retraining approaches

🎯 What you'll learn:
1. Configuration system and main classes
2. Data generation approaches (basic vs synthetic)
3. Pipeline types and their use cases
4. Practical examples you can run immediately
5. How to analyze and interpret results
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import all the main components
from counterfactual_fraud_model import (
    # Enums for configuration
    ModelType, PropensityType,
    
    # Configuration classes
    DataGeneratorConfig, SyntheticDataConfig, ModelConfig, 
    LoggingPolicyConfig, CounterfactualEstimatorConfig, PipelineConfig,
    OffPolicyEvaluationConfig, SyntheticOffPolicyEvaluationConfig, 
    SyntheticRetrainingConfig, RetrainingConfig,
    
    # Main pipeline classes
    OffPolicyEvaluationPipeline,
    SyntheticOffPolicyEvaluationPipeline, 
    SyntheticRetrainingPipeline
)

# Import additional config classes not in main __init__.py
from counterfactual_fraud_model.config import RetrainingStrategy, RetrainingModelConfig

# Import individual component classes for examples
from counterfactual_fraud_model.generators import DataGenerator, SyntheticDataGenerator, LoggingPolicyGenerator
from counterfactual_fraud_model.estimators import CounterfactualEstimator
from counterfactual_fraud_model.core import ModelTrainer


def print_section_header(title: str, emoji: str = "📋") -> None:
    """Print a formatted section header."""
    print(f"\n{emoji} {title}")
    print("=" * (len(title) + 4))


def print_subsection(title: str, emoji: str = "🔸") -> None:
    """Print a formatted subsection header."""
    print(f"\n{emoji} {title}")
    print("-" * (len(title) + 4))


def show_basic_configs():
    """Demonstrate the basic configuration classes."""
    print_section_header("1. BASIC CONFIGURATION CLASSES", "⚙️")
    
    print("""
The project uses Pydantic models for type-safe configuration.
All configurations have sensible defaults and validation.
""")
    
    print_subsection("ModelType Enum")
    print("Available model types:")
    for model_type in ModelType:
        print(f"  • {model_type.value}")
    
    print_subsection("Basic Data Generator Config")
    data_config = DataGeneratorConfig(
        alpha=0.1,           # Beta distribution parameter
        beta_param=2.0,      # Beta distribution parameter  
        mean=-0.5,           # Normal error mean
        sd=0.5,              # Normal error std
        sample_size=10_000,  # Number of samples
        random_state=42      # For reproducibility
    )
    print("DataGeneratorConfig example:")
    print(f"  • Sample size: {data_config.sample_size:,}")
    print(f"  • Alpha/Beta: {data_config.alpha}/{data_config.beta_param}")
    print(f"  • Random state: {data_config.random_state}")
    
    print_subsection("Synthetic Data Config") 
    synthetic_config = SyntheticDataConfig(
        n_samples=50_000,      # Total samples
        n_features=20,         # Total features
        n_informative=15,      # Informative features
        n_redundant=3,         # Redundant features
        weights=[0.985, 0.015], # Class imbalance (98.5% legitimate, 1.5% fraud)
        class_sep=1.2,         # How separated classes are
        random_state=42
    )
    print("SyntheticDataConfig example:")
    print(f"  • Samples: {synthetic_config.n_samples:,}")
    print(f"  • Features: {synthetic_config.n_features} ({synthetic_config.n_informative} informative)")
    print(f"  • Class balance: {synthetic_config.weights}")
    
    print_subsection("Model Configuration")
    model_config = ModelConfig(
        model_type=ModelType.LIGHTGBM,
        model_params={
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1
        },
        random_state=42
    )
    print("ModelConfig example:")
    print(f"  • Type: {model_config.model_type.value}")
    print(f"  • Custom params: {model_config.model_params}")
    
    print_subsection("Logging Policy Configuration")
    logging_config = LoggingPolicyConfig(
        cutoff=0.05,              # Block transactions with score > 0.05
        exploration_rate=0.1,     # Allow 10% of blocked transactions for exploration
        propensity_type=PropensityType.UNIFORM,
        random_state=42
    )
    print("LoggingPolicyConfig example:")
    print(f"  • Cutoff threshold: {logging_config.cutoff}")
    print(f"  • Exploration rate: {logging_config.exploration_rate}")
    print(f"  • Propensity type: {logging_config.propensity_type.value}")
    
    print_subsection("Counterfactual Estimator Configuration")
    estimator_config = CounterfactualEstimatorConfig(
        n_bootstrap=5000,        # Number of bootstrap samples for confidence intervals
        random_state=42          # For reproducible results
    )
    print("CounterfactualEstimatorConfig example:")
    print(f"  • Bootstrap samples: {estimator_config.n_bootstrap:,}")
    print(f"  • Random state: {estimator_config.random_state}")
    print("  • Purpose: Controls the precision of off-policy evaluation confidence intervals")


def show_concrete_implementations():
    """Demonstrate the concrete implementation classes used by pipelines."""
    print_section_header("2. CONCRETE IMPLEMENTATION CLASSES", "🔧")
    
    print("""
The pipelines use concrete implementation classes that handle specific tasks.
Understanding these classes helps you customize behavior and debug issues.
""")
    
    print_subsection("DataGenerator - Basic Data Creation")
    data_gen = DataGenerator(DataGeneratorConfig(
        alpha=0.1,
        beta_param=2.0, 
        sample_size=1000,
        random_state=42
    ))
    sample_data = data_gen.generate_data()
    
    print("DataGenerator creates basic fraud datasets:")
    print(f"  • Uses beta distribution (alpha={data_gen.config.alpha}, beta={data_gen.config.beta_param})")
    print(f"  • Adds normal noise (mean={data_gen.config.mean}, sd={data_gen.config.sd})")
    print(f"  • Generated {len(sample_data)} samples")
    print(f"  • Fraud rate: {sample_data['is_fraud'].mean():.3f}")
    print(f"  • Score range: [{sample_data['model_scores'].min():.3f}, {sample_data['model_scores'].max():.3f}]")
    
    print_subsection("SyntheticDataGenerator - ML Dataset Creation")
    synthetic_gen = SyntheticDataGenerator(
        SyntheticDataConfig(
            n_samples=2000,
            n_features=10,
            n_informative=6,
            n_redundant=2,
            random_state=42
        ),
        ModelConfig(model_type=ModelType.LIGHTGBM, random_state=42)
    )
    synthetic_data = synthetic_gen.generate_data()
    
    print("SyntheticDataGenerator creates ML-ready datasets:")
    print(f"  • Uses sklearn.make_classification for realistic features")
    print(f"  • Trains {synthetic_gen.model_config.model_type.value} model automatically")
    print(f"  • Generated {len(synthetic_data)} samples with {len([c for c in synthetic_data.columns if c.startswith('feature_')])} features")
    print(f"  • Model performance available via get_model_performance()")
    if hasattr(synthetic_gen, '_model_performance') and synthetic_gen._model_performance:
        perf = synthetic_gen.get_model_performance()
        print(f"  • Model ROC-AUC: {perf['roc_auc']:.3f}")
    
    print_subsection("LoggingPolicyGenerator - Policy Simulation")
    policy_gen = LoggingPolicyGenerator(LoggingPolicyConfig(
        cutoff=0.05,
        exploration_rate=0.1,
        random_state=42
    ))
    
    # Use the sample data from DataGenerator
    policy_data = policy_gen.generate_policy(sample_data.head(100))  # Small sample for demo
    
    print("LoggingPolicyGenerator simulates fraud prevention policies:")
    print(f"  • Cutoff threshold: {policy_gen.config.cutoff} (block if score > threshold)")
    print(f"  • Exploration rate: {policy_gen.config.exploration_rate} (% of blocked transactions to allow)")
    print(f"  • Actions generated: {policy_data['model_action'].value_counts().to_dict()}")
    print(f"  • Policy actions: {policy_data['policy_action'].value_counts().to_dict()}")
    print(f"  • Propensity scores: [{policy_data['propensity_score'].min():.3f}, {policy_data['propensity_score'].max():.3f}]")
    
    print_subsection("CounterfactualEstimator - Off-Policy Evaluation")
    estimator = CounterfactualEstimator(
        CounterfactualEstimatorConfig(n_bootstrap=100, random_state=42),  # Small bootstrap for demo
        policy_data
    )
    
    print("CounterfactualEstimator performs off-policy evaluation:")
    print(f"  • Uses importance sampling with propensity scores")
    print(f"  • Bootstrap samples: {estimator.config.n_bootstrap} for confidence intervals")
    print(f"  • Input data: {len(policy_data)} transactions")
    print(f"  • Observed data: {len(estimator.observed_data)} allowed transactions")
    
    # Demonstrate actual off-policy evaluation
    print("\n  🎯 Example: Evaluating a random policy")
    
    # Create a simple random policy for allowed transactions
    observed_data = estimator.observed_data
    if len(observed_data) > 0:
        # Random binary decisions and prediction probabilities
        random_policy = np.random.RandomState(42).randint(0, 2, len(observed_data))
        random_policy_proba = np.random.RandomState(42).uniform(0, 1, len(observed_data))
        
        # Estimate OPE metrics for this random policy
        ope_results = estimator.estimate_ope_metrics(random_policy, random_policy_proba)
        
        print(f"  • Random policy results:")
        for metric_name, metric_data in ope_results.items():
            mean_val = metric_data.get('mean', 'N/A')
            ci_lower = metric_data.get('p025', 'N/A')
            ci_upper = metric_data.get('p975', 'N/A')
            print(f"    - {metric_name}: {mean_val:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    else:
        print("  • No observed data available for demonstration")
    
    print(f"  • This shows how to evaluate ANY new policy using historical data!")
    
    print_subsection("ModelTrainer - ML Model Training") 
    trainer = ModelTrainer()
    print("ModelTrainer handles ML model training:")
    print(f"  • Supports {', '.join([t.value for t in ModelType])} models")
    print(f"  • Calculates performance metrics (ROC-AUC, precision, recall, F1)")
    print(f"  • Handles sample weights for biased training data")
    print(f"  • Used by SyntheticDataGenerator and retraining pipelines")
    
    # Demonstrate concrete training using the synthetic data from above
    print("\n  🎯 Example: Training models on synthetic data")
    
    # Extract features and target from the synthetic data generated above
    feature_columns = [col for col in synthetic_data.columns if col.startswith('feature_')]
    X = synthetic_data[feature_columns]
    y = synthetic_data['is_fraud']
    
    # Split into train/test sets for proper evaluation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"  • Using {len(feature_columns)} features from synthetic dataset")
    print(f"  • Train set: {len(X_train)} samples (fraud rate: {y_train.mean():.3f})")
    print(f"  • Test set: {len(X_test)} samples (fraud rate: {y_test.mean():.3f})")
    
    # Train and compare different model types
    model_performance = {}
    for model_type in [ModelType.LIGHTGBM, ModelType.RANDOM_FOREST, ModelType.LOGISTIC]:
        print(f"\n  📊 Training {model_type.value} model...")
        
        # Create model configuration
        model_config = ModelConfig(
            model_type=model_type,
            model_params={"random_state": 42, "n_estimators": 50} if model_type != ModelType.LOGISTIC else {"random_state": 42},
            random_state=42
        )
        
        # Train the model
        trained_model = trainer.train_model(X_train, y_train, model_config)
        
        # Evaluate on test set
        test_performance = trainer.calculate_performance(trained_model, X_test, y_test)
        model_performance[model_type.value] = test_performance
        
        print(f"    • ROC-AUC: {test_performance['roc_auc']:.4f}")
        print(f"    • Precision: {test_performance['precision']:.4f}")
        print(f"    • Recall: {test_performance['recall']:.4f}")
        print(f"    • F1-Score: {test_performance['f1']:.4f}")
    
    # Show best performing model
    best_model = max(model_performance.items(), key=lambda x: x[1]['roc_auc'])
    print(f"\n  🏆 Best performing model: {best_model[0]} (ROC-AUC: {best_model[1]['roc_auc']:.4f})")
    print("  • This demonstrates ModelTrainer's capability to train and evaluate different models!")
    
    print_subsection("Retraining Preprocessors - Data Bias Correction")
    
    print("Three strategies for handling biased logging policy data:")
    print(f"  • FilteringDataPreprocessor: Uses only allowed transactions")
    print(f"    - Simple but may lose valuable information from blocked transactions")
    print(f"    - Fast and straightforward implementation")
    
    print(f"  • WeightingDataPreprocessor: Reweights by inverse propensity scores")
    print(f"    - Corrects for selection bias in logging policy")
    print(f"    - May create extreme weights requiring clipping")
    
    print(f"  • FraudInjectionDataPreprocessor: Augments with synthetic fraud")
    print(f"    - Combines allowed data with fraud-enriched blocked data")
    print(f"    - Balances dataset while preserving real transaction patterns")


def show_pipeline_configs():
    """Demonstrate complete pipeline configurations."""
    print_section_header("3. COMPLETE PIPELINE CONFIGURATIONS", "🔧")
    
    print("""
Pipeline configurations combine multiple component configs for end-to-end runs.
Each pipeline type serves different use cases.
""")
    
    print_subsection("Off-Policy Evaluation Config")
    ope_config = OffPolicyEvaluationConfig(
        data_generator=DataGeneratorConfig(sample_size=20_000, random_state=42),
        logging_policy=LoggingPolicyConfig(cutoff=0.05, exploration_rate=0.1),
        counterfactual_estimator=CounterfactualEstimatorConfig(n_bootstrap=1000),
        pipeline=PipelineConfig(include_data=False)
    )
    print("OffPolicyEvaluationConfig - for basic data generation:")
    print(f"  • Data samples: {ope_config.data_generator.sample_size:,}")
    print(f"  • Bootstrap iterations: {ope_config.counterfactual_estimator.n_bootstrap:,}")
    
    print_subsection("Synthetic Off-Policy Evaluation Config")  
    synthetic_ope_config = SyntheticOffPolicyEvaluationConfig(
        synthetic_data=SyntheticDataConfig(n_samples=50_000, n_features=20),
        model=ModelConfig(model_type=ModelType.LIGHTGBM),
        logging_policy=LoggingPolicyConfig(cutoff=0.05, exploration_rate=0.05),
        counterfactual_estimator=CounterfactualEstimatorConfig(n_bootstrap=5000),
        pipeline=PipelineConfig(include_data=False)
    )
    print("SyntheticOffPolicyEvaluationConfig - for ML model training:")
    print(f"  • Synthetic samples: {synthetic_ope_config.synthetic_data.n_samples:,}")
    print(f"  • Model type: {synthetic_ope_config.model.model_type.value}")
    print(f"  • Features: {synthetic_ope_config.synthetic_data.n_features}")
    
    print_subsection("Retraining Configuration")
    print("RetrainingStrategy options:")
    for strategy in RetrainingStrategy:
        print(f"  • {strategy.value}")
    
    retraining_config = SyntheticRetrainingConfig(
        base_config=synthetic_ope_config,
        retraining=RetrainingConfig(
            retrain_test_size=0.3,
            retrain_model=RetrainingModelConfig(
                base_model=ModelConfig(model_type=ModelType.LIGHTGBM),
                strategy=RetrainingStrategy.FILTERING,
                classification_threshold=0.1
            )
        )
    )
    print("SyntheticRetrainingConfig - for comparing retraining strategies:")
    print(f"  • Retraining strategy: {retraining_config.retraining.retrain_model.strategy.value}")
    print(f"  • Test size: {retraining_config.retraining.retrain_test_size}")


def example_basic_pipeline():
    """Run a simple off-policy evaluation example."""
    print_section_header("4. BASIC PIPELINE EXAMPLE", "🎯")
    
    print("""
Let's run a simple off-policy evaluation to see how the framework works.
This example uses basic data generation (no ML training required).
""")
    
    # Create a simple configuration
    config = OffPolicyEvaluationConfig(
        data_generator=DataGeneratorConfig(
            sample_size=5_000,  # Small dataset for quick demo
            alpha=0.1,
            beta_param=2.0,
            random_state=42
        ),
        logging_policy=LoggingPolicyConfig(
            cutoff=0.05,
            exploration_rate=0.1,
            random_state=42
        ),
        counterfactual_estimator=CounterfactualEstimatorConfig(
            n_bootstrap=1000,  # Fewer iterations for speed
            random_state=42
        ),
        pipeline=PipelineConfig(include_data=False)
    )
    
    print("📊 Configuration:")
    print(f"  • Sample size: {config.data_generator.sample_size:,}")
    print(f"  • Cutoff threshold: {config.logging_policy.cutoff}")
    print(f"  • Exploration rate: {config.logging_policy.exploration_rate}")
    print(f"  • Bootstrap samples: {config.counterfactual_estimator.n_bootstrap:,}")
    
    # Initialize and run pipeline
    print("\n🚀 Running pipeline...")
    pipeline = OffPolicyEvaluationPipeline(config)
    
    # Run with default parameters (can also override here)
    result = pipeline.run_pipeline(include_data=False)
    
    print("\n✅ Results:")
    print_result_summary(result)


def example_synthetic_pipeline():
    """Run a synthetic off-policy evaluation example."""
    print_section_header("5. SYNTHETIC PIPELINE EXAMPLE", "🤖")
    
    print("""
Now let's try the synthetic pipeline which trains actual ML models.
This is more realistic and allows comparison of different model types.
""")
    
    # Create synthetic configuration
    config = SyntheticOffPolicyEvaluationConfig(
        synthetic_data=SyntheticDataConfig(
            n_samples=10_000,     # Moderate size for demo
            n_features=15,
            n_informative=10,
            n_redundant=2,
            weights=[0.98, 0.02], # 2% fraud rate
            class_sep=1.0,
            random_state=42
        ),
        model=ModelConfig(
            model_type=ModelType.LIGHTGBM,
            model_params={
                "n_estimators": 50,  # Fewer trees for speed
                "max_depth": 4
            },
            random_state=42
        ),
        logging_policy=LoggingPolicyConfig(
            cutoff=0.05,
            exploration_rate=0.05,
            random_state=42
        ),
        counterfactual_estimator=CounterfactualEstimatorConfig(
            n_bootstrap=1000,
            random_state=42
        ),
        pipeline=PipelineConfig(include_data=False)
    )
    
    print("📊 Configuration:")
    print(f"  • Synthetic samples: {config.synthetic_data.n_samples:,}")
    print(f"  • Features: {config.synthetic_data.n_features} ({config.synthetic_data.n_informative} informative)")
    print(f"  • Model: {config.model.model_type.value}")
    print(f"  • Fraud rate: {config.synthetic_data.weights[1]:.1%}")
    
    # Initialize and run pipeline
    print("\n🚀 Running synthetic pipeline...")
    pipeline = SyntheticOffPolicyEvaluationPipeline(config)
    
    result = pipeline.run_pipeline(include_data=False)
    
    print("\n✅ Results:")
    print_result_summary(result)
    
    # Show model performance if available
    if 'model_performance' in result:
        print_model_performance(result['model_performance'])


def example_retraining_pipeline():
    """Run a retraining comparison example.""" 
    print_section_header("6. RETRAINING PIPELINE EXAMPLE", "🔄")
    
    print("""
The retraining pipeline allows you to compare different strategies for
handling biased logging policy data when retraining models.
""")
    
    # Create retraining configuration
    base_config = SyntheticOffPolicyEvaluationConfig(
        synthetic_data=SyntheticDataConfig(
            n_samples=15_000,
            n_features=15,
            n_informative=10,
            n_redundant=2,  # Ensure sum is < n_features
            weights=[0.985, 0.015],
            random_state=42
        ),
        model=ModelConfig(model_type=ModelType.LIGHTGBM, random_state=42),
        logging_policy=LoggingPolicyConfig(
            cutoff=0.05,
            exploration_rate=0.1,
            random_state=42
        ),
        counterfactual_estimator=CounterfactualEstimatorConfig(
            n_bootstrap=1000,
            random_state=42
        )
    )
    
    config = SyntheticRetrainingConfig(
        base_config=base_config,
        retraining=RetrainingConfig(
            retrain_test_size=0.3,
            retrain_model=RetrainingModelConfig(
                base_model=ModelConfig(model_type=ModelType.LIGHTGBM, random_state=42),
                strategy=RetrainingStrategy.FILTERING,  # Will try different strategies
                classification_threshold=0.1
            )
        )
    )
    
    print("📊 Configuration:")
    print(f"  • Base samples: {config.base_config.synthetic_data.n_samples:,}")
    print(f"  • Features: {config.base_config.synthetic_data.n_features} ({config.base_config.synthetic_data.n_informative} informative)")
    print(f"  • Retraining test size: {config.retraining.retrain_test_size}")
    print(f"  • Default strategy: {config.retraining.retrain_model.strategy.value}")
    
    # Initialize pipeline
    pipeline = SyntheticRetrainingPipeline(config)
    
    # Generate the logging policy data first
    print("\n🔄 Generating logging policy data...")
    pipeline.generate_logging_policy_data()
    
    # Try different retraining strategies
    strategies_to_test = [RetrainingStrategy.FILTERING, RetrainingStrategy.WEIGHTING]
    
    for strategy in strategies_to_test:
        print(f"\n🎯 Testing {strategy.value} strategy...")
        
        # Create retraining config for this strategy
        retraining_config = RetrainingConfig(
            retrain_test_size=0.3,
            retrain_model=RetrainingModelConfig(
                base_model=ModelConfig(model_type=ModelType.LIGHTGBM, random_state=42),
                strategy=strategy,
                classification_threshold=0.1
            )
        )
        
        result = pipeline.run_retrain_pipeline(retraining_config=retraining_config)
        
        print(f"\n✅ Results for {strategy.value}:")
        print_result_summary(result, strategy.value)


def print_result_summary(result: Dict[str, Any], strategy_name: str = None) -> None:
    """Print a formatted summary of pipeline results."""
    prefix = f"[{strategy_name}] " if strategy_name else ""
    
    # Print key statistics
    if 'statistics' in result:
        stats = result['statistics']
        print(f"  {prefix}📈 Key Statistics:")
        print(f"    • Total transactions: {stats.get('total_transactions', 'N/A'):,}")
        print(f"    • Allow rate: {stats.get('allow_rate', 'N/A'):.3f}")
        print(f"    • Block rate: {stats.get('block_rate', 'N/A'):.3f}")
        print(f"    • Overall fraud rate: {stats.get('fraud_rate_overall', 'N/A'):.4f}")
        print(f"    • Fraud rate in allowed: {stats.get('fraud_rate_allowed', 'N/A'):.4f}")
    
    # Print OPE metrics
    if 'ope_metrics' in result:
        print(f"  {prefix}📊 Off-Policy Evaluation Metrics:")
        for metric_name, metric_data in result['ope_metrics'].items():
            mean_val = metric_data.get('mean', 'N/A')
            ci_lower = metric_data.get('p025', 'N/A')
            ci_upper = metric_data.get('p975', 'N/A')
            print(f"    • {metric_name}: {mean_val:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")


def print_model_performance(model_perf: Dict[str, Any]) -> None:
    """Print model performance metrics."""
    print("  🎯 Model Performance:")
    for metric, value in model_perf.items():
        print(f"    • {metric}: {value:.4f}")


def show_advanced_usage_patterns():
    """Show advanced usage patterns and tips."""
    print_section_header("7. ADVANCED USAGE PATTERNS", "🎓")
    
    print("""
Here are some advanced patterns you can use for more sophisticated analyses:
""")
    
    print_subsection("Parameter Sweeps")
    print("""
Run multiple simulations with different parameters:

# Example: Test different exploration rates
exploration_rates = np.linspace(0.01, 0.2, 10)
for rate in exploration_rates:
    config.logging_policy.exploration_rate = rate
    result = pipeline.run_pipeline()
    # Store and analyze results
""")
    
    print_subsection("Model Comparison")
    print("""
Compare different model types:

model_types = [ModelType.LIGHTGBM, ModelType.RANDOM_FOREST, ModelType.LOGISTIC]
for model_type in model_types:
    config.model.model_type = model_type
    result = pipeline.run_pipeline()
    # Compare performance metrics
""")
    
    print_subsection("Custom Model Parameters")
    print("""
Fine-tune model hyperparameters:

config.model.model_params = {
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8
}
""")
    
    print_subsection("Data Analysis")
    print("""
Access intermediate data for custom analysis:

# Get the generated dataset
data = pipeline._get_or_generate_data()

# Get policy decisions
policy_data = pipeline.logging_policy_generator.generate_policy(data)

# Analyze feature distributions, correlations, etc.
""")


def show_results_interpretation():
    """Explain how to interpret the results."""
    print_section_header("8. INTERPRETING RESULTS", "📈")
    
    print("""
Understanding the output metrics and what they tell you:
""")
    
    print_subsection("Key Statistics")
    print("""
• total_transactions: Total number of transactions processed
• allow_rate: Fraction of transactions allowed through (not blocked)
• block_rate: Fraction of transactions blocked by the policy  
• fraud_rate_overall: Overall fraud rate in the dataset
• fraud_rate_allowed: Fraud rate among transactions that were allowed
""")
    
    print_subsection("Off-Policy Evaluation Metrics")
    print("""
These estimate what would happen with a different policy:

• fraud_rate: Estimated fraud rate under the evaluated policy
• precision: Estimated precision of fraud detection
• recall: Estimated recall of fraud detection  
• f1_score: Harmonic mean of precision and recall

Each metric includes:
- mean: Point estimate
- p025/p975: 95% confidence interval bounds
""")
    
    print_subsection("Model Performance (Synthetic Pipeline)")
    print("""
When training ML models, you also get:

• roc_auc: Area under the ROC curve
• precision/recall/f1: Standard classification metrics
• These help assess the base model quality before policy evaluation
""")
    
    print_subsection("Retraining Strategies")
    print("""
When comparing retraining strategies:

• FILTERING: Uses only allowed transactions (removes blocked ones)
• WEIGHTING: Reweights training samples by inverse propensity scores
• FRAUD_INJECTION: Augments training data with synthetic fraud cases

Compare OPE metrics across strategies to see which works best.
""")


def show_common_use_cases():
    """Show common use cases and when to use each pipeline."""
    print_section_header("9. COMMON USE CASES", "💡")
    
    print_subsection("Basic Off-Policy Evaluation")
    print("""
Use OffPolicyEvaluationPipeline when:
• You have existing model scores
• You want to quickly test different policy parameters
• You don't need to train new models
• You want to understand baseline counterfactual estimation
""")
    
    print_subsection("Model Development & Comparison")
    print("""
Use SyntheticOffPolicyEvaluationPipeline when:
• You want to train and compare different ML models
• You need realistic synthetic fraud data
• You want to test feature engineering approaches
• You're developing new model architectures
""")
    
    print_subsection("Retraining Strategy Research") 
    print("""
Use SyntheticRetrainingPipeline when:
• You're studying bias in logged data
• You want to compare retraining approaches
• You're researching domain adaptation techniques
• You need to understand exploration vs exploitation tradeoffs
""")
    
    print_subsection("Production Policy Evaluation")
    print("""
For production use cases:
• Use higher sample sizes (100k+ transactions)
• Increase bootstrap iterations (5k+ for stable CIs)
• Test multiple exploration rates and cutoffs
• Validate on held-out historical data
""")


def main():
    """Run the complete getting started guide."""
    print("""
🚀 WELCOME TO THE COUNTERFACTUAL FRAUD MODEL PROJECT!
====================================================

This framework helps you evaluate fraud detection policies using counterfactual estimation.
Whether you're a researcher, data scientist, or ML engineer, this guide will get you started.

Let's dive in step by step...
""")
    
    # 1. Show basic configuration classes
    show_basic_configs()
    
    # 2. Show concrete implementation classes
    show_concrete_implementations()
    
    # 3. Show complete pipeline configurations  
    show_pipeline_configs()
    
    # 4. Run basic pipeline example
    example_basic_pipeline()
    
    # 5. Run synthetic pipeline example
    example_synthetic_pipeline()
    
    # 6. Run retraining pipeline example  
    example_retraining_pipeline()
    
    # 7. Show advanced usage patterns
    show_advanced_usage_patterns()
    
    # 8. Explain results interpretation
    show_results_interpretation()
    
    # 9. Show common use cases
    show_common_use_cases()
    
    print_section_header("🎉 CONGRATULATIONS!", "🎊")
    print("""
You've completed the getting started guide! 

Next steps:
• Explore the simulation_utils/ examples for more advanced analyses
• Check out the documentation in documents/ folder
• Try modifying the examples with your own parameters
• Run the notebooks/ for interactive exploration

The framework is highly configurable - experiment with different:
• Model types and hyperparameters
• Data generation parameters  
• Logging policy settings
• Retraining strategies
• Bootstrap sample sizes

Happy experimenting! 🔬
""")


if __name__ == "__main__":
    main()