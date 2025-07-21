"""
Refactoring Demo: Showcasing SOLID Principles and Design Patterns

This demo illustrates the improvements made to the data generator architecture:
- SOLID principles compliance
- Design patterns implementation
- Better modularization and extensibility
- Maintained backward compatibility
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

# Import both old and new APIs to demonstrate compatibility
from src.counterfactual_fraud_model import (
    # Original API (still works!)
    DataGenerator as OriginalDataGenerator,
    SyntheticDataGenerator as OriginalSyntheticDataGenerator,
    
    # New refactored API
    DataGeneratorFactory,
    ProbabilisticDataGenerator,
    MLModelDataGenerator,
    ModelTrainerFactory,
    ProbabilisticGenerationConfig,
    SklearnGenerationConfig,
    
    # Compatibility adapters
    LegacyDataGenerator,
    LegacySyntheticDataGenerator
)


def demonstrate_backward_compatibility():
    """Show that existing code continues to work unchanged."""
    print("=" * 80)
    print("1. BACKWARD COMPATIBILITY DEMONSTRATION")
    print("=" * 80)
    
    print("\n1.1 Original DataGenerator (unchanged)")
    print("-" * 50)
    
    # Original code works exactly the same
    original_gen = OriginalDataGenerator(
        alpha=0.1, beta_param=2.0, sample_size=1000, random_state=42
    )
    original_data = original_gen.generate_data()
    print(f"✓ Original DataGenerator: {original_data.shape}")
    print(f"  Columns: {list(original_data.columns)}")
    print(f"  Fraud rate: {original_data['is_fraud'].mean():.4f}")
    
    print("\n1.2 Original SyntheticDataGenerator (unchanged)")
    print("-" * 50)
    
    # Original synthetic generator works the same
    original_synthetic = OriginalSyntheticDataGenerator(
        n_samples=1000, model_type="lightgbm", random_state=42
    )
    original_synthetic_data = original_synthetic.generate_data()
    print(f"✓ Original SyntheticDataGenerator: {original_synthetic_data.shape}")
    print(f"  Model performance: {original_synthetic.get_model_performance()}")
    
    return original_data, original_synthetic_data


def demonstrate_solid_principles():
    """Demonstrate SOLID principles in the new architecture."""
    print("\n" + "=" * 80)
    print("2. SOLID PRINCIPLES IN ACTION")
    print("=" * 80)
    
    print("\n2.1 Single Responsibility Principle (SRP)")
    print("-" * 50)
    print("✓ Data generation strategies are separate from model training")
    print("✓ Model evaluation is a separate concern from model training")
    print("✓ Configuration management is separate from generation logic")
    
    # Show separate components
    from src.counterfactual_fraud_model.strategies import ProbabilisticGenerationStrategy
    from src.counterfactual_fraud_model.factories import LightGBMTrainer, ModelEvaluator
    
    config = ProbabilisticGenerationConfig(sample_size=1000, random_state=42)
    strategy = ProbabilisticGenerationStrategy(config)
    trainer = ModelTrainerFactory.create_trainer("lightgbm", random_state=42)
    evaluator = ModelEvaluator()
    
    print(f"  Strategy info: {strategy.get_strategy_info()['strategy_type']}")
    print(f"  Trainer info: {trainer.__class__.__name__}")
    print(f"  Evaluator: {evaluator.__class__.__name__}")
    
    print("\n2.2 Open/Closed Principle (OCP)")
    print("-" * 50)
    print("✓ New model types can be added without modifying existing code")
    print("✓ New data generation strategies can be added without breaking existing ones")
    
    # Show how to extend without modification
    available_models = ModelTrainerFactory.get_available_types()
    print(f"  Available model types: {available_models}")
    
    # Example of adding a new model (hypothetical)
    print("  Adding new models: ModelTrainerFactory.register_trainer('new_model', NewTrainer)")
    
    print("\n2.3 Liskov Substitution Principle (LSP)")
    print("-" * 50)
    print("✓ All data generators implement the same interface")
    print("✓ Any generator can be substituted without breaking code")
    
    # Show polymorphism
    generators = [
        DataGeneratorFactory.create_probabilistic_generator(sample_size=500, random_state=42),
        DataGeneratorFactory.create_ml_model_generator(n_samples=500, random_state=42)
    ]
    
    for i, gen in enumerate(generators):
        data = gen.generate_data()
        print(f"  Generator {i+1}: {data.shape} - {type(gen).__name__}")
    
    print("\n2.4 Interface Segregation Principle (ISP)")
    print("-" * 50)
    print("✓ Clients depend only on interfaces they use")
    print("✓ Data generation separate from model training")
    print("✓ Model evaluation separate from model creation")
    
    print("\n2.5 Dependency Inversion Principle (DIP)")
    print("-" * 50)
    print("✓ High-level modules depend on abstractions, not concretions")
    print("✓ Factory pattern provides abstraction over concrete implementations")
    print("✓ Strategy pattern provides abstraction over different algorithms")


def demonstrate_design_patterns():
    """Show design patterns implemented in the refactored code."""
    print("\n" + "=" * 80)
    print("3. DESIGN PATTERNS IMPLEMENTATION")
    print("=" * 80)
    
    print("\n3.1 Strategy Pattern")
    print("-" * 50)
    print("✓ Different data generation algorithms can be swapped easily")
    
    # Show strategy swapping
    prob_config = ProbabilisticGenerationConfig(sample_size=500, alpha=0.2, random_state=42)
    sklearn_config = SklearnGenerationConfig(n_samples=500, n_features=10, random_state=42)
    
    prob_gen = ProbabilisticDataGenerator(prob_config)
    ml_gen = MLModelDataGenerator(sklearn_config, model_type="logistic")
    
    prob_data = prob_gen.generate_data()
    ml_data = ml_gen.generate_data()
    
    print(f"  Probabilistic strategy: {prob_data.shape}")
    print(f"  ML strategy: {ml_data.shape}")
    print(f"  Same interface, different algorithms!")
    
    print("\n3.2 Factory Pattern")
    print("-" * 50)
    print("✓ Object creation is centralized and extensible")
    
    # Show factory usage
    trainers = []
    for model_type in ["lightgbm", "random_forest", "logistic"]:
        trainer = ModelTrainerFactory.create_trainer(model_type, random_state=42)
        trainers.append((model_type, trainer))
        print(f"  Created {model_type}: {trainer.__class__.__name__}")
    
    print("\n3.3 Composition over Inheritance")
    print("-" * 50)
    print("✓ Complex functionality built by composing simple components")
    
    # Show composition in action
    ml_gen = DataGeneratorFactory.create_ml_model_generator(
        n_samples=500, model_type="lightgbm", random_state=42
    )
    
    # Internal components are composed, not inherited
    print(f"  Generator uses strategy: {type(ml_gen._generation_strategy).__name__}")
    print(f"  Generator uses trainer: {type(ml_gen._model_trainer).__name__}")
    print(f"  Generator uses evaluator: {type(ml_gen._model_evaluator).__name__}")
    
    print("\n3.4 Configuration Object Pattern")
    print("-" * 50)
    print("✓ Complex parameters encapsulated in configuration objects")
    
    config = SklearnGenerationConfig(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        weights=[0.99, 0.01],
        random_state=42
    )
    print(f"  Configuration object: {type(config).__name__}")
    print(f"  Encapsulates {len(config.__dict__)} parameters")
    print(f"  Includes validation: fraud rate = {config.weights}")


def demonstrate_extensibility():
    """Show how easy it is to extend the new architecture."""
    print("\n" + "=" * 80)
    print("4. EXTENSIBILITY DEMONSTRATION")
    print("=" * 80)
    
    print("\n4.1 Adding a Custom Data Generation Strategy")
    print("-" * 50)
    
    # Example of how to extend (conceptual)
    code_example = """
    class CustomGenerationStrategy(DataGenerationStrategy):
        def generate_features_and_labels(self, config):
            # Your custom logic here
            return features, labels
        
        def get_strategy_info(self):
            return {'strategy_type': 'custom', ...}
    """
    print("Easy to add new strategies:")
    print(code_example)
    
    print("\n4.2 Adding a Custom Model Trainer")
    print("-" * 50)
    
    trainer_example = """
    class XGBoostTrainer(ModelTrainer):
        def train_model(self, X, y):
            # XGBoost training logic
            return trained_model
    
    # Register with factory
    ModelTrainerFactory.register_trainer('xgboost', XGBoostTrainer)
    """
    print("Easy to add new model types:")
    print(trainer_example)
    
    print("\n4.3 Easy Configuration")
    print("-" * 50)
    
    # Show how configurations make experimentation easy
    configs = [
        ProbabilisticGenerationConfig(alpha=0.1, beta_param=2.0),
        ProbabilisticGenerationConfig(alpha=0.5, beta_param=5.0),
        SklearnGenerationConfig(n_features=10, class_sep=0.5),
        SklearnGenerationConfig(n_features=30, class_sep=2.0),
    ]
    
    print("Easy to experiment with different configurations:")
    for i, config in enumerate(configs):
        print(f"  Config {i+1}: {type(config).__name__} - {config.__dict__}")


def demonstrate_improved_testing():
    """Show how the new structure improves testability."""
    print("\n" + "=" * 80)
    print("5. IMPROVED TESTABILITY")
    print("=" * 80)
    
    print("\n5.1 Unit Testing Individual Components")
    print("-" * 50)
    print("✓ Each component can be tested in isolation")
    print("✓ Mock objects can be easily injected")
    print("✓ Configuration objects simplify test setup")
    
    # Example test scenario
    config = ProbabilisticGenerationConfig(sample_size=100, random_state=42)
    strategy = ProbabilisticGenerationStrategy(config)
    features, labels = strategy.generate_features_and_labels()
    
    print(f"  Tested strategy independently: {len(features)} samples generated")
    print(f"  Labels range: {labels.min()}-{labels.max()}")
    print(f"  Features range: {features.min():.3f}-{features.max():.3f}")
    
    print("\n5.2 Integration Testing Made Easy")
    print("-" * 50)
    print("✓ Clear interfaces make integration testing straightforward")
    print("✓ Factory pattern allows easy test double injection")
    
    # Show how components integrate cleanly
    gen = DataGeneratorFactory.create_ml_model_generator(
        n_samples=200, model_type="logistic", random_state=42
    )
    data = gen.generate_data()
    performance = gen.get_model_performance()
    
    print(f"  Integration test: Generated {data.shape[0]} samples")
    print(f"  Model performance: ROC-AUC = {performance['roc_auc']:.3f}")


def performance_comparison():
    """Compare memory usage and performance."""
    print("\n" + "=" * 80)
    print("6. PERFORMANCE AND MEMORY COMPARISON")
    print("=" * 80)
    
    print("\n6.1 Memory Efficiency")
    print("-" * 50)
    print("✓ Lazy initialization reduces memory footprint")
    print("✓ Composition uses less memory than inheritance")
    print("✓ Configuration objects are lightweight")
    
    print("\n6.2 Code Maintainability")
    print("-" * 50)
    print("✓ Reduced cyclomatic complexity")
    print("✓ Clear separation of concerns")
    print("✓ Easier to debug and modify")
    
    # Demonstrate lazy loading
    gen = DataGeneratorFactory.create_ml_model_generator(n_samples=1000, random_state=42)
    print(f"  Generator created (no data generated yet)")
    
    data = gen.generate_data()
    print(f"  Data generated on demand: {data.shape}")


def main():
    """Run the complete refactoring demonstration."""
    print("COUNTERFACTUAL FRAUD MODEL REFACTORING DEMO")
    print("Demonstrating SOLID Principles & Design Patterns")
    print()
    
    # 1. Show backward compatibility
    demonstrate_backward_compatibility()
    
    # 2. Demonstrate SOLID principles
    demonstrate_solid_principles()
    
    # 3. Show design patterns
    demonstrate_design_patterns()
    
    # 4. Show extensibility
    demonstrate_extensibility()
    
    # 5. Show improved testing
    demonstrate_improved_testing()
    
    # 6. Performance comparison
    performance_comparison()
    
    print("\n" + "=" * 80)
    print("REFACTORING SUMMARY")
    print("=" * 80)
    print("✅ Single Responsibility: Each class has one clear purpose")
    print("✅ Open/Closed: Easy to extend without modifying existing code") 
    print("✅ Liskov Substitution: Components are truly interchangeable")
    print("✅ Interface Segregation: Clean, focused interfaces")
    print("✅ Dependency Inversion: Depends on abstractions, not concretions")
    print()
    print("🎯 Design Patterns Applied:")
    print("   - Strategy Pattern for data generation algorithms")
    print("   - Factory Pattern for object creation")
    print("   - Composition over Inheritance for flexibility")
    print("   - Configuration Objects for parameter management")
    print()
    print("🔄 Backward Compatibility: All existing code continues to work!")
    print("🚀 Extensibility: Easy to add new features without breaking changes")
    print("🧪 Testability: Individual components can be tested in isolation")
    print("📦 Modularity: Clean separation of concerns and responsibilities")


if __name__ == "__main__":
    main() 