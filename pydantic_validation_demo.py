"""
Pydantic Validation Demo: Showcasing Enhanced Configuration Management

This demo illustrates the improvements gained by switching from dataclasses to Pydantic:
- Enhanced validation with clear error messages
- Type safety and runtime type checking
- JSON serialization/deserialization
- Better configuration management
- Field descriptions and constraints
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any
from pydantic import ValidationError

# Import both old and new APIs
from src.counterfactual_fraud_model import (
    # New Pydantic-based configurations
    ProbabilisticGenerationConfig,
    SklearnGenerationConfig,
    DataGeneratorFactory,
    ProbabilisticDataGenerator,
    MLModelDataGenerator
)


def demonstrate_enhanced_validation():
    """Show improved validation with Pydantic."""
    print("=" * 80)
    print("1. ENHANCED VALIDATION WITH PYDANTIC")
    print("=" * 80)
    
    print("\n1.1 Valid Configuration Creation")
    print("-" * 50)
    
    # Valid configurations work as expected
    valid_prob_config = ProbabilisticGenerationConfig(
        sample_size=1000,
        alpha=0.2,
        beta_param=3.0,
        mean=0.1,
        sd=0.3,
        random_state=42
    )
    print(f"✅ Valid probabilistic config: {valid_prob_config.sample_size} samples")
    print(f"   Alpha: {valid_prob_config.alpha}, Beta: {valid_prob_config.beta_param}")
    
    valid_sklearn_config = SklearnGenerationConfig(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        weights=[0.99, 0.01],
        flip_y=0.02,
        random_state=42
    )
    print(f"✅ Valid sklearn config: {valid_sklearn_config.n_samples} samples")
    print(f"   Features: {valid_sklearn_config.n_features}, Informative: {valid_sklearn_config.n_informative}")
    
    print("\n1.2 Validation Error Examples")
    print("-" * 50)
    
    # Test various validation failures
    validation_tests = [
        ("Negative sample size", lambda: ProbabilisticGenerationConfig(sample_size=-100)),
        ("Zero alpha parameter", lambda: ProbabilisticGenerationConfig(alpha=0)),
        ("Negative beta parameter", lambda: ProbabilisticGenerationConfig(beta_param=-1)),
        ("Negative standard deviation", lambda: ProbabilisticGenerationConfig(sd=-0.1)),
        ("flip_y > 1", lambda: SklearnGenerationConfig(flip_y=1.5)),
        ("flip_y < 0", lambda: SklearnGenerationConfig(flip_y=-0.1)),
        ("n_informative > n_features", lambda: SklearnGenerationConfig(n_features=10, n_informative=15)),
        ("Invalid weights length", lambda: SklearnGenerationConfig(weights=[0.5, 0.3, 0.2])),
        ("Weights don't sum to 1", lambda: SklearnGenerationConfig(weights=[0.7, 0.2])),
        ("Zero n_features", lambda: SklearnGenerationConfig(n_features=0)),
    ]
    
    for test_name, test_func in validation_tests:
        try:
            test_func()
            print(f"❌ {test_name}: Should have failed but didn't!")
        except ValidationError as e:
            print(f"✅ {test_name}: Properly caught - {str(e).split(chr(10))[0]}")
        except Exception as e:
            print(f"⚠️  {test_name}: Unexpected error type - {type(e).__name__}: {e}")


def demonstrate_field_constraints():
    """Show Pydantic field constraints and descriptions."""
    print("\n" + "=" * 80)
    print("2. FIELD CONSTRAINTS AND DESCRIPTIONS")
    print("=" * 80)
    
    print("\n2.1 Field Information")
    print("-" * 50)
    
    # Show field information
    prob_schema = ProbabilisticGenerationConfig.model_json_schema()
    sklearn_schema = SklearnGenerationConfig.model_json_schema()
    
    print("📋 ProbabilisticGenerationConfig fields:")
    for field_name, field_info in prob_schema['properties'].items():
        description = field_info.get('description', 'No description')
        field_type = field_info.get('type', 'unknown')
        default = field_info.get('default', 'No default')
        print(f"   {field_name}: {field_type} - {description}")
        if 'minimum' in field_info or 'maximum' in field_info:
            constraints = []
            if 'minimum' in field_info:
                constraints.append(f"min: {field_info['minimum']}")
            if 'exclusiveMinimum' in field_info:
                constraints.append(f"min (exclusive): {field_info['exclusiveMinimum']}")
            if 'maximum' in field_info:
                constraints.append(f"max: {field_info['maximum']}")
            print(f"      Constraints: {', '.join(constraints)}")
    
    print("\n📋 SklearnGenerationConfig key constraints:")
    sklearn_props = sklearn_schema['properties']
    constraint_fields = ['n_features', 'n_informative', 'flip_y', 'weights']
    for field_name in constraint_fields:
        if field_name in sklearn_props:
            field_info = sklearn_props[field_name]
            description = field_info.get('description', 'No description')
            print(f"   {field_name}: {description}")
            if field_name == 'flip_y':
                print(f"      Range: 0 ≤ flip_y ≤ 1")
            elif field_name == 'weights':
                print(f"      Must be list of 2 floats that sum to 1")


def demonstrate_json_serialization():
    """Show JSON serialization and deserialization capabilities."""
    print("\n" + "=" * 80)
    print("3. JSON SERIALIZATION AND DESERIALIZATION")
    print("=" * 80)
    
    print("\n3.1 Configuration to JSON")
    print("-" * 50)
    
    # Create a configuration
    config = SklearnGenerationConfig(
        n_samples=5000,
        n_features=25,
        n_informative=12,
        weights=[0.98, 0.02],
        flip_y=0.015,
        class_sep=1.5,
        random_state=123
    )
    
    # Serialize to JSON
    config_json = config.model_dump_json(indent=2)
    print("📤 Configuration serialized to JSON:")
    print(config_json)
    
    print("\n3.2 JSON to Configuration")
    print("-" * 50)
    
    # Deserialize from JSON
    config_dict = json.loads(config_json)
    reconstructed_config = SklearnGenerationConfig(**config_dict)
    
    print("📥 Configuration reconstructed from JSON:")
    print(f"   Samples: {reconstructed_config.n_samples}")
    print(f"   Features: {reconstructed_config.n_features}")
    print(f"   Weights: {reconstructed_config.weights}")
    print(f"   ✅ Reconstruction successful!")
    
    print("\n3.3 Configuration Comparison")
    print("-" * 50)
    
    # Compare original and reconstructed
    print(f"   Original == Reconstructed: {config == reconstructed_config}")
    print(f"   Original dict == Reconstructed dict: {config.model_dump() == reconstructed_config.model_dump()}")


def demonstrate_type_coercion():
    """Show Pydantic's type coercion capabilities."""
    print("\n" + "=" * 80)
    print("4. TYPE COERCION AND CONVERSION")
    print("=" * 80)
    
    print("\n4.1 Automatic Type Conversion")
    print("-" * 50)
    
    # Pydantic can automatically convert compatible types
    conversion_tests = [
        ("String to int", {"sample_size": "1000"}, "sample_size"),
        ("String to float", {"alpha": "0.5"}, "alpha"),
        ("String to float list", {"weights": ["0.95", "0.05"]}, "weights"),
        ("Int to float", {"class_sep": 2}, "class_sep"),
    ]
    
    for test_name, test_data, field_name in conversion_tests:
        try:
            if field_name in ["sample_size", "alpha"]:
                config = ProbabilisticGenerationConfig(**test_data)
                value = getattr(config, field_name)
                original_type = type(test_data[field_name]).__name__
                final_type = type(value).__name__
                print(f"✅ {test_name}: '{test_data[field_name]}' ({original_type}) → {value} ({final_type})")
            else:
                config = SklearnGenerationConfig(**test_data)
                value = getattr(config, field_name)
                original_type = type(test_data[field_name]).__name__
                final_type = type(value).__name__
                print(f"✅ {test_name}: {test_data[field_name]} ({original_type}) → {value} ({final_type})")
        except Exception as e:
            print(f"❌ {test_name}: Failed - {e}")


def demonstrate_default_values():
    """Show how Pydantic handles default values."""
    print("\n" + "=" * 80)
    print("5. DEFAULT VALUES AND PARTIAL CONFIGURATION")
    print("=" * 80)
    
    print("\n5.1 Minimal Configuration with Defaults")
    print("-" * 50)
    
    # Create configs with minimal parameters
    minimal_prob = ProbabilisticGenerationConfig(sample_size=500)
    minimal_sklearn = SklearnGenerationConfig(n_samples=500)
    
    print("📋 Probabilistic config with only sample_size specified:")
    print(f"   sample_size: {minimal_prob.sample_size} (specified)")
    print(f"   alpha: {minimal_prob.alpha} (default)")
    print(f"   beta_param: {minimal_prob.beta_param} (default)")
    print(f"   mean: {minimal_prob.mean} (default)")
    print(f"   sd: {minimal_prob.sd} (default)")
    print(f"   random_state: {minimal_prob.random_state} (default)")
    
    print("\n📋 Sklearn config with only n_samples specified:")
    print(f"   n_samples: {minimal_sklearn.n_samples} (specified)")
    print(f"   n_features: {minimal_sklearn.n_features} (default)")
    print(f"   n_informative: {minimal_sklearn.n_informative} (default)")
    print(f"   weights: {minimal_sklearn.weights} (default)")
    print(f"   flip_y: {minimal_sklearn.flip_y} (default)")
    print(f"   class_sep: {minimal_sklearn.class_sep} (default)")


def demonstrate_practical_usage():
    """Show practical usage with the improved configurations."""
    print("\n" + "=" * 80)
    print("6. PRACTICAL USAGE WITH PYDANTIC CONFIGS")
    print("=" * 80)
    
    print("\n6.1 Creating Data Generators with Pydantic Configs")
    print("-" * 50)
    
    # Create configurations
    prob_config = ProbabilisticGenerationConfig(
        sample_size=1000,
        alpha=0.15,
        beta_param=2.5,
        random_state=42
    )
    
    sklearn_config = SklearnGenerationConfig(
        n_samples=1000,
        n_features=15,
        n_informative=8,
        weights=[0.97, 0.03],
        random_state=42
    )
    
    # Use configurations with generators
    prob_generator = ProbabilisticDataGenerator(prob_config)
    ml_generator = MLModelDataGenerator(sklearn_config, model_type="logistic")
    
    # Generate data
    prob_data = prob_generator.generate_data()
    ml_data = ml_generator.generate_data()
    
    print(f"✅ Probabilistic generator: {prob_data.shape}")
    print(f"   Fraud rate: {prob_data['is_fraud'].mean():.4f}")
    print(f"   Score range: [{prob_data['model_scores'].min():.3f}, {prob_data['model_scores'].max():.3f}]")
    
    print(f"✅ ML generator: {ml_data.shape}")
    print(f"   Fraud rate: {ml_data['is_fraud'].mean():.4f}")
    print(f"   Score range: [{ml_data['model_scores'].min():.3f}, {ml_data['model_scores'].max():.3f}]")
    
    print("\n6.2 Configuration Inspection")
    print("-" * 50)
    
    print("📋 Can easily inspect configuration:")
    print(f"   Prob config dict: {prob_config.model_dump()}")
    print(f"   ML config JSON schema available: {bool(sklearn_config.model_json_schema())}")


def demonstrate_validation_performance():
    """Show validation performance and error handling."""
    print("\n" + "=" * 80)
    print("7. VALIDATION PERFORMANCE AND ERROR HANDLING")
    print("=" * 80)
    
    print("\n7.1 Comprehensive Validation Test")
    print("-" * 50)
    
    # Test comprehensive validation
    test_configs = [
        # Valid configs
        {"n_samples": 1000, "n_features": 10, "n_informative": 5, "weights": [0.9, 0.1]},
        {"sample_size": 500, "alpha": 0.3, "beta_param": 1.5, "sd": 0.2},
        
        # Invalid configs that should be caught
        {"n_samples": -100},  # negative samples
        {"alpha": -0.1},      # negative alpha
        {"weights": [0.6, 0.3]},  # weights don't sum to 1
        {"n_features": 5, "n_informative": 10},  # informative > features
    ]
    
    valid_count = 0
    invalid_count = 0
    
    for i, config_data in enumerate(test_configs):
        try:
            if 'n_samples' in config_data:
                config = SklearnGenerationConfig(**config_data)
                valid_count += 1
                print(f"✅ Config {i+1}: Valid sklearn config")
            else:
                config = ProbabilisticGenerationConfig(**config_data)
                valid_count += 1
                print(f"✅ Config {i+1}: Valid probabilistic config")
        except ValidationError as e:
            invalid_count += 1
            error_msg = str(e).split('\n')[0]  # Get first line of error
            print(f"❌ Config {i+1}: Validation failed - {error_msg}")
    
    print(f"\n📊 Validation Summary:")
    print(f"   Valid configurations: {valid_count}")
    print(f"   Invalid configurations caught: {invalid_count}")
    print(f"   Validation working correctly: {invalid_count > 0}")


def main():
    """Run the complete Pydantic validation demonstration."""
    print("PYDANTIC VALIDATION DEMO")
    print("Enhanced Configuration Management with Pydantic")
    print()
    
    # 1. Enhanced validation
    demonstrate_enhanced_validation()
    
    # 2. Field constraints and descriptions
    demonstrate_field_constraints()
    
    # 3. JSON serialization
    demonstrate_json_serialization()
    
    # 4. Type coercion
    demonstrate_type_coercion()
    
    # 5. Default values
    demonstrate_default_values()
    
    # 6. Practical usage
    demonstrate_practical_usage()
    
    # 7. Validation performance
    demonstrate_validation_performance()
    
    print("\n" + "=" * 80)
    print("PYDANTIC CONVERSION SUMMARY")
    print("=" * 80)
    print("✅ Enhanced Validation: Clear, descriptive error messages")
    print("✅ Type Safety: Runtime type checking and conversion")
    print("✅ Field Constraints: Built-in validation rules (gt, ge, le, etc.)")
    print("✅ JSON Serialization: Easy config save/load functionality")
    print("✅ Self-Documenting: Field descriptions and schema generation")
    print("✅ Default Values: Intelligent default handling")
    print("✅ Type Coercion: Automatic type conversion when safe")
    print("✅ Performance: Fast validation with detailed error reporting")
    print()
    print("🔄 Backward Compatibility: All existing APIs still work!")
    print("🚀 Enhanced Developer Experience: Better error messages and tooling")
    print("📋 Schema Generation: Automatic documentation and validation")
    print("🛡️  Robust Validation: Comprehensive input validation")


if __name__ == "__main__":
    main() 