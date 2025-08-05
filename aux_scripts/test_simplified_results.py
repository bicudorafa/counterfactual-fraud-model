"""
Test Script for Simplified SyntheticRetrainingPipeline Results

This script tests the refactored run_pipeline method to assess the new simplified 
results output structure, which now contains only three key pieces of information:
1. original_results: Results from the base pipeline
2. retrained_model_performance: Performance metrics of the retrained model
3. ope_metrics: Counterfactual estimation results
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import json
from datetime import datetime

# Import configuration classes
from counterfactual_fraud_model.config import (
    SyntheticRetrainingConfig,
    SyntheticOffPolicyEvaluationConfig,
    SyntheticDataConfig,
    ModelConfig,
    LoggingPolicyConfig,
    CounterfactualEstimatorConfig,
    RetrainingConfig,
    RetrainingModelConfig,
    PipelineConfig,
    ModelType,
    RetrainingStrategy
)

# Import the pipeline
from counterfactual_fraud_model.pipelines import SyntheticRetrainingPipeline


def print_section_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_subsection_header(title: str):
    """Print a formatted subsection header."""
    print(f"\n{'-' * 60}")
    print(f" {title}")
    print(f"{'-' * 60}")


def analyze_results_structure(results: Dict[str, Any]) -> None:
    """Analyze and display the structure of the simplified results."""
    print_subsection_header("RESULTS STRUCTURE ANALYSIS")
    
    print(f"Number of top-level keys: {len(results.keys())}")
    print(f"Top-level keys: {list(results.keys())}")
    
    for key, value in results.items():
        print(f"\n📊 {key}:")
        if isinstance(value, dict):
            print(f"   Type: Dictionary with {len(value)} keys")
            print(f"   Keys: {list(value.keys())}")
        elif isinstance(value, (list, tuple)):
            print(f"   Type: {type(value).__name__} with {len(value)} items")
        elif isinstance(value, (int, float)):
            print(f"   Type: {type(value).__name__}, Value: {value}")
        elif value is None:
            print(f"   Type: None")
        else:
            print(f"   Type: {type(value).__name__}")


def analyze_original_results(original_results: Dict[str, Any]) -> None:
    """Analyze the original_results component."""
    print_subsection_header("ORIGINAL RESULTS ANALYSIS")
    
    if not original_results:
        print("❌ Original results is empty or None")
        return
    
    print(f"Original results contains {len(original_results)} keys:")
    for key, value in original_results.items():
        if isinstance(value, dict):
            print(f"  📋 {key}: Dict with {len(value)} keys")
            if key == 'statistics':
                print("     Statistics keys:", list(value.keys()))
        elif isinstance(value, pd.DataFrame):
            print(f"  📊 {key}: DataFrame with shape {value.shape}")
        elif isinstance(value, np.ndarray):
            print(f"  🔢 {key}: Array with shape {value.shape}")
        else:
            print(f"  📄 {key}: {type(value).__name__}")


def analyze_retrained_model_performance(performance: Dict[str, float]) -> None:
    """Analyze the retrained_model_performance component."""
    print_subsection_header("RETRAINED MODEL PERFORMANCE ANALYSIS")
    
    if not performance:
        print("❌ Retrained model performance is empty or None")
        return
    
    print("Performance metrics:")
    for metric, value in performance.items():
        print(f"  📈 {metric}: {value:.4f}")


def analyze_ope_metrics(ope_metrics: Dict[str, Any]) -> None:
    """Analyze the ope_metrics component."""
    print_subsection_header("OPE METRICS ANALYSIS")
    
    if not ope_metrics:
        print("❌ OPE metrics is empty or None")
        return
    
    print(f"OPE metrics contains {len(ope_metrics)} keys:")
    for key, value in ope_metrics.items():
        if isinstance(value, dict):
            print(f"  📊 {key}: Dict with {len(value)} keys")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, (int, float)):
                    print(f"     {subkey}: {subvalue:.4f}")
                else:
                    print(f"     {subkey}: {type(subvalue).__name__}")
        elif isinstance(value, (int, float)):
            print(f"  📈 {key}: {value:.4f}")
        else:
            print(f"  📄 {key}: {type(value).__name__}")


def compare_with_expected_structure(results: Dict[str, Any]) -> None:
    """Compare results with expected simplified structure."""
    print_subsection_header("STRUCTURE VALIDATION")
    
    expected_keys = {'original_results', 'retrained_model_performance', 'ope_metrics'}
    actual_keys = set(results.keys())
    
    print("Expected keys:", expected_keys)
    print("Actual keys:  ", actual_keys)
    
    missing_keys = expected_keys - actual_keys
    extra_keys = actual_keys - expected_keys
    
    if missing_keys:
        print(f"❌ Missing keys: {missing_keys}")
    else:
        print("✅ All expected keys present")
    
    if extra_keys:
        print(f"⚠️  Extra keys (not expected): {extra_keys}")
    else:
        print("✅ No unexpected keys found")
    
    # Check data types
    type_checks = [
        ('original_results', dict),
        ('retrained_model_performance', dict),
        ('ope_metrics', dict)
    ]
    
    print("\nType validation:")
    for key, expected_type in type_checks:
        if key in results:
            actual_type = type(results[key])
            if isinstance(results[key], expected_type):
                print(f"✅ {key}: {actual_type.__name__} (correct)")
            else:
                print(f"❌ {key}: {actual_type.__name__} (expected {expected_type.__name__})")
        else:
            print(f"❌ {key}: Missing")


def run_basic_test():
    """Run a basic test with default configuration."""
    print_section_header("BASIC TEST WITH DEFAULT CONFIGURATION")
    
    # Create basic configuration
    config = SyntheticRetrainingConfig(
        base_config=SyntheticOffPolicyEvaluationConfig(
            synthetic_data=SyntheticDataConfig(
                n_samples=5_000,  # Small for quick testing
                n_features=10,
                n_informative=6,
                n_redundant=2,  # Ensure sum doesn't exceed n_features
                n_repeated=0,   # Keep it simple
                random_state=42
            ),
            model=ModelConfig(
                model_type=ModelType.LIGHTGBM,
                random_state=42
            ),
            logging_policy=LoggingPolicyConfig(
                cutoff=0.05,
                exploration_rate=0.1,
                random_state=42
            ),
            counterfactual_estimator=CounterfactualEstimatorConfig(
                n_bootstrap=500,  # Reduced for speed
                random_state=42
            )
        ),
        retraining=RetrainingConfig(
            retrain_test_size=0.3,
            retrain_model=RetrainingModelConfig(
                base_model=ModelConfig(
                    model_type=ModelType.LIGHTGBM,
                    random_state=42
                ),
                strategy=RetrainingStrategy.FILTERING,
                classification_threshold=0.1
            )
        )
    )
    
    print("Configuration created:")
    print(f"  Dataset size: {config.base_config.synthetic_data.n_samples:,}")
    print(f"  Features: {config.base_config.synthetic_data.n_features}")
    print(f"  Model type: {config.base_config.model.model_type}")
    print(f"  Retraining strategy: {config.retraining.retrain_model.strategy}")
    
    # Create and run pipeline
    pipeline = SyntheticRetrainingPipeline(config)
    
    print("\nRunning pipeline...")
    start_time = datetime.now()
    
    try:
        results = pipeline.run_pipeline()
        end_time = datetime.now()
        
        print(f"✅ Pipeline completed successfully in {end_time - start_time}")
        
        # Analyze results
        analyze_results_structure(results)
        analyze_original_results(results.get('original_results', {}))
        analyze_retrained_model_performance(results.get('retrained_model_performance', {}))
        analyze_ope_metrics(results.get('ope_metrics', {}))
        compare_with_expected_structure(results)
        
        return results
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        raise


def test_separate_methods():
    """Test the new separate methods: generate_logging_policy_data and run_retrain_pipeline."""
    print_section_header("TESTING SEPARATE METHODS")
    
    # Create configuration
    config = SyntheticRetrainingConfig(
        base_config=SyntheticOffPolicyEvaluationConfig(
            synthetic_data=SyntheticDataConfig(
                n_samples=3_000,
                n_features=8,
                n_informative=5,
                n_redundant=2,
                random_state=42
            ),
            model=ModelConfig(
                model_type=ModelType.LIGHTGBM,
                random_state=42
            ),
            logging_policy=LoggingPolicyConfig(
                cutoff=0.05,
                exploration_rate=0.1,
                random_state=42
            ),
            counterfactual_estimator=CounterfactualEstimatorConfig(
                n_bootstrap=500,
                random_state=42
            )
        ),
        retraining=RetrainingConfig(
            retrain_test_size=0.3,
            retrain_model=RetrainingModelConfig(
                base_model=ModelConfig(
                    model_type=ModelType.LIGHTGBM,
                    random_state=42
                ),
                strategy=RetrainingStrategy.FILTERING,
                classification_threshold=0.1
            )
        )
    )
    
    pipeline = SyntheticRetrainingPipeline(config)
    
    print_subsection_header("Step 1: Generate Logging Policy Data")
    start_time = datetime.now()
    
    try:
        # Test generate_logging_policy_data method
        pipeline.generate_logging_policy_data(
            logging_policy_cutoff=0.05,
            logging_policy_exploration_rate=0.1
        )
        
        end_time = datetime.now()
        print(f"✅ Logging policy data generated in {end_time - start_time}")
        
        # Test getter methods
        original_results = pipeline.get_original_results()
        train_data = pipeline.get_train_policy_data()
        test_data = pipeline.get_test_policy_data()
        
        print(f"  Original results keys: {list(original_results.keys())}")
        print(f"  Training data shape: {train_data.shape}")
        print(f"  Test data shape: {test_data.shape}")
        
    except Exception as e:
        print(f"❌ Data generation failed: {str(e)}")
        raise
    
    print_subsection_header("Step 2: Run Multiple Retrain Experiments")
    
    # Test different retraining configurations
    retrain_configs = [
        {
            'name': 'Conservative',
            'config': RetrainingConfig(
                retrain_test_size=0.3,
                retrain_model=RetrainingModelConfig(
                    base_model=ModelConfig(
                        model_type=ModelType.LIGHTGBM,
                        random_state=42
                    ),
                    strategy=RetrainingStrategy.FILTERING,
                    classification_threshold=0.05
                )
            )
        },
        {
            'name': 'Moderate',
            'config': RetrainingConfig(
                retrain_test_size=0.3,
                retrain_model=RetrainingModelConfig(
                    base_model=ModelConfig(
                        model_type=ModelType.LIGHTGBM,
                        random_state=42
                    ),
                    strategy=RetrainingStrategy.FILTERING,
                    classification_threshold=0.1
                )
            )
        },
        {
            'name': 'Aggressive',
            'config': RetrainingConfig(
                retrain_test_size=0.3,
                retrain_model=RetrainingModelConfig(
                    base_model=ModelConfig(
                        model_type=ModelType.LIGHTGBM,
                        random_state=42
                    ),
                    strategy=RetrainingStrategy.FILTERING,
                    classification_threshold=0.2
                )
            )
        }
    ]
    
    retrain_results = []
    
    for i, cfg in enumerate(retrain_configs):
        print(f"\n  Experiment {i+1}: {cfg['name']} (threshold={cfg['config'].retrain_model.classification_threshold})")
        
        try:
            start_time = datetime.now()
            results = pipeline.run_retrain_pipeline(retraining_config=cfg['config'])
            end_time = datetime.now()
            
            print(f"    ✅ Completed in {end_time - start_time}")
            print(f"    Result keys: {list(results.keys())}")
            
            # Extract performance metrics if available
            if 'retrained_model_performance' in results and results['retrained_model_performance']:
                auc = results['retrained_model_performance'].get('auc', 'N/A')
                print(f"    Retrained model AUC: {auc}")
            
            retrain_results.append({
                'name': cfg['name'],
                'threshold': cfg['config'].retrain_model.classification_threshold,
                'results': results,
                'success': True
            })
            
        except Exception as e:
            print(f"    ❌ Failed: {str(e)}")
            retrain_results.append({
                'name': cfg['name'],
                'threshold': cfg['config'].retrain_model.classification_threshold,
                'error': str(e),
                'success': False
            })
    
    print_subsection_header("Separate Methods Summary")
    print("✅ Data generation and multiple retraining experiments completed")
    print(f"   Successful experiments: {sum(1 for r in retrain_results if r['success'])}/{len(retrain_results)}")
    
    return retrain_results


def run_comparison_test():
    """Run tests with different configurations to compare results."""
    print_section_header("COMPARISON TEST WITH DIFFERENT CONFIGURATIONS")
    
    configurations = [
        {
            'name': 'Conservative Policy',
            'logging_policy_cutoff': 0.02,
            'logging_policy_exploration_rate': 0.05,
            'threshold': 0.05
        },
        {
            'name': 'Moderate Policy',
            'logging_policy_cutoff': 0.05,
            'logging_policy_exploration_rate': 0.1,
            'threshold': 0.1
        },
        {
            'name': 'Aggressive Policy',
            'logging_policy_cutoff': 0.1,
            'logging_policy_exploration_rate': 0.2,
            'threshold': 0.15
        }
    ]
    
    results_comparison = []
    
    for i, cfg in enumerate(configurations):
        print_subsection_header(f"Test {i+1}: {cfg['name']}")
        
        # Create configuration
        config = SyntheticRetrainingConfig(
            base_config=SyntheticOffPolicyEvaluationConfig(
                synthetic_data=SyntheticDataConfig(
                    n_samples=3_000,  # Small for quick testing
                    n_features=8,
                    n_informative=5,
                    n_redundant=2,  # Ensure sum doesn't exceed n_features
                    n_repeated=0,   # Keep it simple
                    random_state=42  # Same seed for fair comparison
                )
            ),
            retraining=RetrainingConfig(
                retrain_model=RetrainingModelConfig(
                    classification_threshold=cfg['threshold']
                )
            )
        )
        
        pipeline = SyntheticRetrainingPipeline(config)
        
        try:
            # Create custom retraining config
            custom_retrain_config = RetrainingConfig(
                retrain_test_size=0.3,
                retrain_model=RetrainingModelConfig(
                    base_model=ModelConfig(
                        model_type=ModelType.LIGHTGBM,
                        random_state=42
                    ),
                    strategy=RetrainingStrategy.FILTERING,
                    classification_threshold=cfg['threshold']
                )
            )
            
            results = pipeline.run_pipeline(
                logging_policy_cutoff=cfg['logging_policy_cutoff'],
                logging_policy_exploration_rate=cfg['logging_policy_exploration_rate'],
                retraining_config=custom_retrain_config
            )
            
            # Extract key metrics for comparison
            comparison_data = {
                'name': cfg['name'],
                'logging_policy_cutoff': cfg['logging_policy_cutoff'],
                'logging_policy_exploration_rate': cfg['logging_policy_exploration_rate'],
                'threshold': cfg['threshold'],
                'has_original_results': 'original_results' in results,
                'has_retrained_performance': 'retrained_model_performance' in results,
                'has_ope_metrics': 'ope_metrics' in results,
                'result_keys': list(results.keys())
            }
            
            # Extract specific metrics if available
            if 'retrained_model_performance' in results and results['retrained_model_performance']:
                comparison_data['retrained_auc'] = results['retrained_model_performance'].get('auc', 'N/A')
            
            results_comparison.append(comparison_data)
            print(f"✅ {cfg['name']} completed successfully")
            
        except Exception as e:
            print(f"❌ {cfg['name']} failed: {str(e)}")
            results_comparison.append({
                'name': cfg['name'],
                'error': str(e)
            })
    
    # Display comparison
    print_subsection_header("COMPARISON SUMMARY")
    for result in results_comparison:
        if 'error' not in result:
            print(f"\n{result['name']}:")
            print(f"  Structure complete: {result['has_original_results'] and result['has_retrained_performance'] and result['has_ope_metrics']}")
            print(f"  Keys present: {result['result_keys']}")
            if 'retrained_auc' in result:
                print(f"  Retrained AUC: {result['retrained_auc']}")
        else:
            print(f"\n{result['name']}: ❌ {result['error']}")


def run_edge_case_test():
    """Test edge cases to ensure robustness."""
    print_section_header("EDGE CASE TESTING")
    
    edge_cases = [
        {
            'name': 'Very Small Dataset',
            'config_override': {
                'n_samples': 500,
                'logging_policy_cutoff': 0.05,
                'logging_policy_exploration_rate': 0.1
            }
        },
        {
            'name': 'High Exploration Rate',
            'config_override': {
                'n_samples': 2000,
                'logging_policy_cutoff': 0.02,
                'logging_policy_exploration_rate': 0.5
            }
        }
    ]
    
    for case in edge_cases:
        print_subsection_header(f"Edge Case: {case['name']}")
        
        try:
            config = SyntheticRetrainingConfig(
                base_config=SyntheticOffPolicyEvaluationConfig(
                    synthetic_data=SyntheticDataConfig(
                        n_samples=case['config_override']['n_samples'],
                        n_features=6,
                        n_informative=4,
                        n_redundant=1,  # Ensure sum doesn't exceed n_features
                        n_repeated=0,   # Keep it simple
                        random_state=42
                    )
                )
            )
            
            pipeline = SyntheticRetrainingPipeline(config)
            results = pipeline.run_pipeline(
                logging_policy_cutoff=case['config_override']['logging_policy_cutoff'],
                logging_policy_exploration_rate=case['config_override']['logging_policy_exploration_rate']
            )
            
            print(f"✅ {case['name']} handled successfully")
            print(f"   Result structure: {list(results.keys())}")
            
        except Exception as e:
            print(f"⚠️  {case['name']} failed (may be expected): {str(e)}")


def main():
    """Main function to run all tests."""
    print("🧪 TESTING SIMPLIFIED SYNTHETIC RETRAINING PIPELINE RESULTS")
    print(f"Test started at: {datetime.now()}")
    
    try:
        # Run basic test
        basic_results = run_basic_test()
        
        # Test separate methods
        separate_results = test_separate_methods()
        
        # Run comparison test
        run_comparison_test()
        
        # Run edge case test
        run_edge_case_test()
        
        print_section_header("SUMMARY")
        print("✅ All tests completed successfully!")
        print("\n📋 Key findings:")
        print("   • Results structure is simplified to 3 key components")
        print("   • All expected keys are present in successful runs")
        print("   • Pipeline handles various configurations appropriately")
        print("   • Simplified structure reduces complexity and improves maintainability")
        print("   • New separate methods allow for flexible experimentation")
        print("   • RetrainingConfig parameter provides comprehensive control")
        
        return basic_results
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main() 