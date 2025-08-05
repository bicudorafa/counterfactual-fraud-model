#!/usr/bin/env python3
"""
Comprehensive test suite for CounterfactualEstimator refactoring validation.

This script validates that all refactoring changes to the CounterfactualEstimator
class maintain correctness, performance, and API compatibility. It tests:

1. Functional correctness of all public methods
2. Statistical similarity between different implementations
3. Edge case handling and error conditions
4. Performance characteristics
5. Memory usage patterns
6. API backward compatibility
7. Helper method functionality

The tests are designed to catch any regressions introduced during the refactoring
while validating that the code improvements work as expected.
"""

import warnings
import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json
import traceback

# Import the estimator and configuration
from counterfactual_fraud_model.config import CounterfactualEstimatorConfig
from counterfactual_fraud_model.estimators.counterfactual_estimator import CounterfactualEstimator

# Suppress warnings for cleaner output during testing
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class TestResult:
    """Results from a single test."""
    test_name: str
    passed: bool
    execution_time: float
    memory_usage_mb: float
    error_message: str = ""
    additional_info: Dict[str, Any] = None


class TestSuite:
    """Main test suite for CounterfactualEstimator refactoring validation."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.process = psutil.Process()
        
    def print_header(self, title: str):
        """Print a formatted test section header."""
        print("\n" + "=" * 80)
        print(f" {title}")
        print("=" * 80)
    
    def print_subheader(self, title: str):
        """Print a formatted test subsection header."""
        print(f"\n{'-' * 60}")
        print(f" {title}")
        print(f"{'-' * 60}")
    
    def create_test_data(self, n_samples: int = 1000, fraud_rate: float = 0.05, 
                        seed: int = 42) -> pd.DataFrame:
        """Create synthetic test data."""
        np.random.seed(seed)
        
        # Generate fraud labels
        is_fraud = np.random.binomial(1, fraud_rate, n_samples)
        
        # Generate model scores (higher for fraud)
        model_scores = np.random.beta(2, 5, n_samples)
        model_scores[is_fraud == 1] += np.random.beta(3, 2, np.sum(is_fraud))
        model_scores = np.clip(model_scores, 0, 1)
        
        # Policy action (all allowed for simplicity)
        policy_action = ['allow'] * n_samples
        
        # Model actions based on threshold
        model_action = ['block' if score > 0.5 else 'allow' for score in model_scores]
        
        # Propensity scores (uniform for simplicity)
        propensity_score = np.ones(n_samples)
        
        return pd.DataFrame({
            'is_fraud': is_fraud,
            'model_scores': model_scores,
            'propensity_score': propensity_score,
            'model_action': model_action,
            'policy_action': policy_action
        })
    
    def run_test(self, test_name: str, test_func) -> TestResult:
        """Run a single test and record results."""
        print(f"  Running: {test_name}...", end=' ', flush=True)
        
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        start_time = time.perf_counter()
        
        try:
            additional_info = test_func()
            end_time = time.perf_counter()
            final_memory = self.process.memory_info().rss / 1024 / 1024
            
            result = TestResult(
                test_name=test_name,
                passed=True,
                execution_time=end_time - start_time,
                memory_usage_mb=final_memory - initial_memory,
                additional_info=additional_info or {}
            )
            print("✅ PASSED")
            
        except Exception as e:
            end_time = time.perf_counter()
            final_memory = self.process.memory_info().rss / 1024 / 1024
            
            result = TestResult(
                test_name=test_name,
                passed=False,
                execution_time=end_time - start_time,
                memory_usage_mb=final_memory - initial_memory,
                error_message=str(e)
            )
            print("❌ FAILED")
            print(f"     Error: {str(e)}")
        
        self.results.append(result)
        return result

    def test_basic_functionality(self):
        """Test basic functionality of all public methods."""
        
        def test_basic_methods():
            """Test all basic public methods work."""
            data = self.create_test_data(1000)
            config = CounterfactualEstimatorConfig(n_bootstrap=100, random_state=42)
            estimator = CounterfactualEstimator(config, data)
            
            # Test basic policy metrics
            policy_results = estimator.estimate_policy_metrics()
            assert isinstance(policy_results, dict), "Policy metrics should return dict"
            assert len(policy_results) > 0, "Policy metrics should not be empty"
            
            # Test OPE metrics with custom actions
            n_obs = len(data[data['policy_action'] == 'allow'])
            new_actions = np.random.binomial(1, 0.3, n_obs)
            new_actions_proba = np.random.beta(2, 5, n_obs)
            
            ope_results = estimator.estimate_ope_metrics(new_actions, new_actions_proba)
            assert isinstance(ope_results, dict), "OPE metrics should return dict"
            assert len(ope_results) > 0, "OPE metrics should not be empty"
            
            # Test vectorized version
            vec_results = estimator.estimate_ope_metrics_vectorized(new_actions, new_actions_proba)
            assert isinstance(vec_results, dict), "Vectorized metrics should return dict"
            assert len(vec_results) > 0, "Vectorized metrics should not be empty"
            
            return {
                'policy_metrics_count': len(policy_results),
                'ope_metrics_count': len(ope_results),
                'vectorized_metrics_count': len(vec_results)
            }
        
        return self.run_test("Basic Functionality", test_basic_methods)

    def test_statistical_correctness(self):
        """Test that refactored methods produce statistically similar results."""
        
        def test_statistical_similarity():
            """Compare loop-based and vectorized implementations."""
            data = self.create_test_data(2000, seed=42)
            config = CounterfactualEstimatorConfig(n_bootstrap=500, random_state=42)
            estimator = CounterfactualEstimator(config, data)
            
            n_obs = len(data[data['policy_action'] == 'allow'])
            np.random.seed(42)  # Ensure same random actions
            new_actions = np.random.binomial(1, 0.3, n_obs)
            new_actions_proba = np.random.beta(2, 5, n_obs)
            
            # Run both implementations with same seed
            estimator.config.random_state = 42
            np.random.seed(42)
            loop_results = estimator._estimate_ope_metrics_loop_based(new_actions, new_actions_proba)
            
            estimator.config.random_state = 42
            np.random.seed(42)
            vec_results = estimator.estimate_ope_metrics_vectorized(new_actions, new_actions_proba)
            
            # Compare results
            differences = {}
            tolerance = 0.05  # 5% tolerance for means
            
            for metric in loop_results.keys():
                if metric in vec_results:
                    loop_mean = loop_results[metric]['mean']
                    vec_mean = vec_results[metric]['mean']
                    diff = abs(loop_mean - vec_mean)
                    rel_diff = diff / abs(loop_mean) if loop_mean != 0 else diff
                    differences[metric] = rel_diff
                    
                    assert rel_diff < tolerance, f"Results differ too much for {metric}: {rel_diff:.4f}"
            
            return {
                'compared_metrics': list(differences.keys()),
                'max_relative_difference': max(differences.values()) if differences else 0,
                'average_relative_difference': np.mean(list(differences.values())) if differences else 0
            }
        
        return self.run_test("Statistical Correctness", test_statistical_similarity)

    def test_metric_selection(self):
        """Test selective metric calculation functionality."""
        
        def test_selective_metrics():
            """Test that metric selection works correctly."""
            data = self.create_test_data(1000)
            config = CounterfactualEstimatorConfig(n_bootstrap=100, random_state=42)
            estimator = CounterfactualEstimator(config, data)
            
            n_obs = len(data[data['policy_action'] == 'allow'])
            new_actions = np.random.binomial(1, 0.3, n_obs)
            new_actions_proba = np.random.beta(2, 5, n_obs)
            
            # Test action metrics only
            action_only = estimator.estimate_ope_metrics(
                new_actions, new_actions_proba,
                action_metrics=['precision', 'recall'],
                proba_metrics=None
            )
            expected_action_keys = {'precision', 'recall'}
            assert set(action_only.keys()) == expected_action_keys, f"Expected {expected_action_keys}, got {set(action_only.keys())}"
            
            # Test proba metrics only
            proba_only = estimator.estimate_ope_metrics(
                new_actions, new_actions_proba,
                action_metrics=None,
                proba_metrics=['roc_auc', 'brier_score']
            )
            expected_proba_keys = {'roc_auc', 'brier_score'}
            assert set(proba_only.keys()) == expected_proba_keys, f"Expected {expected_proba_keys}, got {set(proba_only.keys())}"
            
            # Test mixed metrics
            mixed = estimator.estimate_ope_metrics(
                new_actions, new_actions_proba,
                action_metrics=['precision'],
                proba_metrics=['roc_auc']
            )
            expected_mixed_keys = {'precision', 'roc_auc'}
            assert set(mixed.keys()) == expected_mixed_keys, f"Expected {expected_mixed_keys}, got {set(mixed.keys())}"
            
            # Test empty selection should default to all
            all_metrics = estimator.estimate_ope_metrics(new_actions, new_actions_proba)
            assert len(all_metrics) > len(mixed), "Default should include more metrics than selective"
            
            return {
                'action_only_count': len(action_only),
                'proba_only_count': len(proba_only),
                'mixed_count': len(mixed),
                'all_metrics_count': len(all_metrics)
            }
        
        return self.run_test("Metric Selection", test_selective_metrics)

    def test_input_validation(self):
        """Test input validation and error handling."""
        
        def test_validation_errors():
            """Test various input validation scenarios."""
            data = self.create_test_data(1000)
            config = CounterfactualEstimatorConfig(n_bootstrap=100, random_state=42)
            estimator = CounterfactualEstimator(config, data)
            
            n_obs = len(data[data['policy_action'] == 'allow'])
            
            # Test wrong array sizes
            try:
                estimator.estimate_ope_metrics(
                    np.array([1, 0, 1]),  # Wrong size
                    np.random.beta(2, 5, n_obs)
                )
                assert False, "Should have raised ValueError for wrong array size"
            except ValueError:
                pass  # Expected
            
            # Test invalid metric names
            try:
                estimator.estimate_ope_metrics(
                    np.random.binomial(1, 0.3, n_obs),
                    np.random.beta(2, 5, n_obs),
                    action_metrics=['invalid_metric']
                )
                assert False, "Should have raised ValueError for invalid metric"
            except ValueError:
                pass  # Expected
            
            # Test data without required columns
            try:
                bad_data = pd.DataFrame({'wrong_column': [1, 2, 3]})
                CounterfactualEstimator(config, bad_data)
                assert False, "Should have raised ValueError for missing columns"
            except ValueError:
                pass  # Expected
            
            # Test data with no allowed transactions
            try:
                bad_data = data.copy()
                bad_data['policy_action'] = 'block'  # No allowed transactions
                CounterfactualEstimator(config, bad_data)
                assert False, "Should have raised ValueError for no allowed transactions"
            except ValueError:
                pass  # Expected
            
            return {'validation_tests_passed': 4}
        
        return self.run_test("Input Validation", test_validation_errors)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        
        def test_edge_conditions():
            """Test various edge cases."""
            edge_cases_passed = 0
            
            # Test with very small dataset
            small_data = self.create_test_data(50)
            config = CounterfactualEstimatorConfig(n_bootstrap=10, random_state=42)
            estimator = CounterfactualEstimator(config, small_data)
            
            n_obs = len(small_data[small_data['policy_action'] == 'allow'])
            new_actions = np.random.binomial(1, 0.5, n_obs)
            new_actions_proba = np.random.beta(2, 5, n_obs)
            
            # Should work with small data
            results = estimator.estimate_ope_metrics(new_actions, new_actions_proba)
            assert len(results) > 0, "Should work with small dataset"
            edge_cases_passed += 1
            
            # Test with all zeros/ones actions
            all_zeros = np.zeros(n_obs, dtype=int)
            all_ones = np.ones(n_obs, dtype=int)
            
            zero_results = estimator.estimate_ope_metrics(all_zeros, new_actions_proba)
            assert len(zero_results) > 0, "Should handle all-allow policy"
            edge_cases_passed += 1
            
            one_results = estimator.estimate_ope_metrics(all_ones, new_actions_proba)
            assert len(one_results) > 0, "Should handle all-block policy"
            edge_cases_passed += 1
            
            # Test with extreme probability values
            extreme_proba = np.array([0.0] * (n_obs//2) + [1.0] * (n_obs//2))
            extreme_results = estimator.estimate_ope_metrics(new_actions, extreme_proba)
            assert len(extreme_results) > 0, "Should handle extreme probabilities"
            edge_cases_passed += 1
            
            # Test with no fraud in data
            no_fraud_data = small_data.copy()
            no_fraud_data['is_fraud'] = 0
            no_fraud_estimator = CounterfactualEstimator(config, no_fraud_data)
            
            no_fraud_results = no_fraud_estimator.estimate_ope_metrics(new_actions, new_actions_proba)
            assert len(no_fraud_results) > 0, "Should handle data with no fraud"
            edge_cases_passed += 1
            
            return {'edge_cases_passed': edge_cases_passed}
        
        return self.run_test("Edge Cases", test_edge_conditions)

    def test_performance_characteristics(self):
        """Test performance characteristics of different implementations."""
        
        def test_performance():
            """Compare performance of different methods."""
            data = self.create_test_data(5000)
            config = CounterfactualEstimatorConfig(n_bootstrap=500, random_state=42)
            estimator = CounterfactualEstimator(config, data)
            
            n_obs = len(data[data['policy_action'] == 'allow'])
            new_actions = np.random.binomial(1, 0.3, n_obs)
            new_actions_proba = np.random.beta(2, 5, n_obs)
            
            # Time loop-based method
            start_time = time.perf_counter()
            loop_results = estimator._estimate_ope_metrics_loop_based(new_actions, new_actions_proba)
            loop_time = time.perf_counter() - start_time
            
            # Time vectorized method
            start_time = time.perf_counter()
            vec_results = estimator.estimate_ope_metrics_vectorized(new_actions, new_actions_proba)
            vec_time = time.perf_counter() - start_time
            
            speedup = loop_time / vec_time
            
            return {
                'loop_time_seconds': loop_time,
                'vectorized_time_seconds': vec_time,
                'speedup_factor': speedup,
                'vectorized_faster': speedup > 1.0
            }
        
        return self.run_test("Performance Characteristics", test_performance)

    def test_memory_efficiency(self):
        """Test memory usage patterns."""
        
        def test_memory():
            """Test memory usage of different implementations."""
            # Test with progressively larger datasets
            memory_results = {}
            
            for size in [1000, 5000]:
                data = self.create_test_data(size)
                config = CounterfactualEstimatorConfig(n_bootstrap=100, random_state=42)
                estimator = CounterfactualEstimator(config, data)
                
                n_obs = len(data[data['policy_action'] == 'allow'])
                new_actions = np.random.binomial(1, 0.3, n_obs)
                new_actions_proba = np.random.beta(2, 5, n_obs)
                
                # Measure memory for loop method
                initial_mem = self.process.memory_info().rss / 1024 / 1024
                _ = estimator._estimate_ope_metrics_loop_based(new_actions, new_actions_proba)
                loop_mem = self.process.memory_info().rss / 1024 / 1024 - initial_mem
                
                # Measure memory for vectorized method (this may allocate more)
                initial_mem = self.process.memory_info().rss / 1024 / 1024
                _ = estimator.estimate_ope_metrics_vectorized(new_actions, new_actions_proba)
                vec_mem = self.process.memory_info().rss / 1024 / 1024 - initial_mem
                
                memory_results[f'size_{size}'] = {
                    'loop_memory_mb': loop_mem,
                    'vectorized_memory_mb': vec_mem,
                    'memory_ratio': vec_mem / loop_mem if loop_mem > 0 else 1.0
                }
            
            return memory_results
        
        return self.run_test("Memory Efficiency", test_memory)

    def test_helper_methods(self):
        """Test the new helper methods work correctly."""
        
        def test_helpers():
            """Test helper method functionality."""
            data = self.create_test_data(1000)
            config = CounterfactualEstimatorConfig(n_bootstrap=100, random_state=42)
            estimator = CounterfactualEstimator(config, data)
            
            n_obs = len(data[data['policy_action'] == 'allow'])
            
            # Test input validation helper
            new_actions = np.random.binomial(1, 0.3, n_obs)
            new_actions_proba = np.random.beta(2, 5, n_obs)
            
            validated_actions, validated_proba = estimator._validate_and_convert_inputs(
                new_actions, new_actions_proba
            )
            assert isinstance(validated_actions, np.ndarray), "Should return numpy array"
            assert isinstance(validated_proba, np.ndarray), "Should return numpy array"
            assert len(validated_actions) == n_obs, "Should preserve length"
            
            # Test metric filtering helper
            action_metrics, proba_metrics = estimator._filter_metrics(
                action_metrics=['precision'], 
                proba_metrics=['roc_auc']
            )
            assert len(action_metrics) == 1, "Should filter to 1 action metric"
            assert len(proba_metrics) == 1, "Should filter to 1 proba metric"
            assert 'precision' in action_metrics, "Should contain precision"
            assert 'roc_auc' in proba_metrics, "Should contain roc_auc"
            
            # Test weight checking helper
            zero_weights = np.zeros(10)
            positive_weights = np.array([1, 2, 0, 3, 0])
            
            assert not estimator._has_positive_weights(zero_weights), "Should detect zero weights"
            assert estimator._has_positive_weights(positive_weights), "Should detect positive weights"
            
            # Test bootstrap statistics helper
            bootstrap_values = np.array([0.1, 0.2, 0.15, 0.18, 0.12])
            stats = estimator._calculate_bootstrap_statistics(bootstrap_values)
            
            expected_keys = {'mean', 'p025', 'p975', 'n_bootstrap'}
            assert set(stats.keys()) == expected_keys, f"Should have keys {expected_keys}"
            assert stats['mean'] == np.mean(bootstrap_values), "Mean should match"
            assert stats['n_bootstrap'] == len(bootstrap_values), "Count should match"
            
            return {
                'helper_methods_tested': 4,
                'all_helpers_working': True
            }
        
        return self.run_test("Helper Methods", test_helpers)

    def test_api_compatibility(self):
        """Test that public API remains unchanged."""
        
        def test_api():
            """Test API compatibility."""
            data = self.create_test_data(1000)
            config = CounterfactualEstimatorConfig(n_bootstrap=100, random_state=42)
            estimator = CounterfactualEstimator(config, data)
            
            # Test all public methods exist and work
            api_methods = [
                'estimate_policy_metrics',
                'estimate_ope_metrics', 
                'estimate_ope_metrics_vectorized',
                'get_available_action_metrics',
                'get_available_proba_metrics',
                'get_config'
            ]
            
            working_methods = 0
            
            for method_name in api_methods:
                assert hasattr(estimator, method_name), f"Missing method: {method_name}"
                method = getattr(estimator, method_name)
                assert callable(method), f"Method {method_name} is not callable"
                working_methods += 1
            
            # Test method signatures work as expected
            n_obs = len(data[data['policy_action'] == 'allow'])
            new_actions = np.random.binomial(1, 0.3, n_obs)
            new_actions_proba = np.random.beta(2, 5, n_obs)
            
            # Test estimate_policy_metrics with parameters
            policy_result1 = estimator.estimate_policy_metrics()  # Default
            policy_result2 = estimator.estimate_policy_metrics(is_vectorized=False)  # Loop
            policy_result3 = estimator.estimate_policy_metrics(
                action_metrics=['precision'], 
                proba_metrics=['roc_auc']
            )  # Selective
            
            assert all(isinstance(r, dict) for r in [policy_result1, policy_result2, policy_result3])
            
            # Test estimate_ope_metrics with parameters
            ope_result1 = estimator.estimate_ope_metrics(new_actions, new_actions_proba)  # Default
            ope_result2 = estimator.estimate_ope_metrics(
                new_actions, new_actions_proba,
                action_metrics=['precision'], 
                proba_metrics=None
            )  # Selective
            
            assert all(isinstance(r, dict) for r in [ope_result1, ope_result2])
            
            # Test getter methods
            action_metrics = estimator.get_available_action_metrics()
            proba_metrics = estimator.get_available_proba_metrics()
            config_obj = estimator.get_config()
            
            assert isinstance(action_metrics, list), "Should return list"
            assert isinstance(proba_metrics, list), "Should return list"
            assert isinstance(config_obj, CounterfactualEstimatorConfig), "Should return config"
            
            return {
                'api_methods_working': working_methods,
                'total_api_methods': len(api_methods),
                'api_compatibility': working_methods == len(api_methods)
            }
        
        return self.run_test("API Compatibility", test_api)

    def test_warning_systems(self):
        """Test that warning systems work correctly."""
        
        def test_warnings():
            """Test warning functionality."""
            warnings_caught = 0
            
            # Capture warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # Test vectorized method with small dataset (should warn)
                small_data = self.create_test_data(500)  # Small dataset
                config = CounterfactualEstimatorConfig(n_bootstrap=100, random_state=42)
                estimator = CounterfactualEstimator(config, small_data)
                
                n_obs = len(small_data[small_data['policy_action'] == 'allow'])
                new_actions = np.random.binomial(1, 0.3, n_obs)
                new_actions_proba = np.random.beta(2, 5, n_obs)
                
                # This should trigger a warning about using vectorized with small data
                _ = estimator.estimate_ope_metrics_vectorized(new_actions, new_actions_proba)
                
                if w:
                    warnings_caught += 1
                
                # Clear warnings
                w.clear()
                
                # Test loop method with large dataset (should warn about potential speedup)
                large_data = self.create_test_data(2000)  # Large dataset
                large_config = CounterfactualEstimatorConfig(n_bootstrap=200, random_state=42)
                large_estimator = CounterfactualEstimator(large_config, large_data)
                
                n_obs_large = len(large_data[large_data['policy_action'] == 'allow'])
                large_actions = np.random.binomial(1, 0.3, n_obs_large)
                large_proba = np.random.beta(2, 5, n_obs_large)
                
                # This should trigger a warning about potential speedup
                _ = large_estimator._estimate_ope_metrics_loop_based(large_actions, large_proba)
                
                if w:
                    warnings_caught += 1
            
            return {
                'warnings_caught': warnings_caught,
                'warning_system_working': warnings_caught > 0
            }
        
        return self.run_test("Warning Systems", test_warnings)

    def run_all_tests(self):
        """Run all tests in the suite."""
        
        self.print_header("COUNTERFACTUAL ESTIMATOR REFACTORING VALIDATION")
        print(f"Test suite started at: {datetime.now()}")
        print(f"Python process ID: {self.process.pid}")
        
        # Run all test categories
        test_categories = [
            ("Basic Functionality Tests", self.test_basic_functionality),
            ("Statistical Correctness Tests", self.test_statistical_correctness),
            ("Metric Selection Tests", self.test_metric_selection),
            ("Input Validation Tests", self.test_input_validation),
            ("Edge Case Tests", self.test_edge_cases),
            ("Performance Tests", self.test_performance_characteristics),
            ("Memory Efficiency Tests", self.test_memory_efficiency),
            ("Helper Method Tests", self.test_helper_methods),
            ("API Compatibility Tests", self.test_api_compatibility),
            ("Warning System Tests", self.test_warning_systems)
        ]
        
        total_start_time = time.perf_counter()
        
        for category_name, test_method in test_categories:
            self.print_subheader(category_name)
            test_method()
        
        total_time = time.perf_counter() - total_start_time
        
        # Print summary
        self.print_summary(total_time)
        
        return self.results

    def print_summary(self, total_time: float):
        """Print test summary."""
        
        self.print_header("TEST SUMMARY")
        
        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if not r.passed]
        
        print(f"Total tests run: {len(self.results)}")
        print(f"Passed: {len(passed_tests)} ✅")
        print(f"Failed: {len(failed_tests)} ❌")
        print(f"Success rate: {len(passed_tests)/len(self.results)*100:.1f}%")
        print(f"Total execution time: {total_time:.2f} seconds")
        
        if failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"  • {test.test_name}: {test.error_message}")
        
        # Performance summary
        if passed_tests:
            avg_time = np.mean([t.execution_time for t in passed_tests])
            max_time = max([t.execution_time for t in passed_tests])
            total_memory = sum([t.memory_usage_mb for t in passed_tests])
            
            print(f"\n📊 PERFORMANCE SUMMARY:")
            print(f"  Average test time: {avg_time:.3f} seconds")
            print(f"  Slowest test time: {max_time:.3f} seconds")
            print(f"  Total memory usage: {total_memory:.1f} MB")
        
        # Key findings
        print(f"\n🔍 KEY FINDINGS:")
        
        # Look for specific test results
        performance_test = next((r for r in self.results if "Performance" in r.test_name), None)
        if performance_test and performance_test.passed:
            speedup = performance_test.additional_info.get('speedup_factor', 1.0)
            print(f"  • Vectorized implementation speedup: {speedup:.2f}x")
        
        statistical_test = next((r for r in self.results if "Statistical" in r.test_name), None)
        if statistical_test and statistical_test.passed:
            max_diff = statistical_test.additional_info.get('max_relative_difference', 0)
            print(f"  • Maximum statistical difference: {max_diff:.4f}")
        
        api_test = next((r for r in self.results if "API" in r.test_name), None)
        if api_test and api_test.passed:
            compatibility = api_test.additional_info.get('api_compatibility', False)
            print(f"  • API compatibility maintained: {'✅' if compatibility else '❌'}")
        
        # Overall verdict
        if len(failed_tests) == 0:
            print(f"\n🎉 OVERALL VERDICT: ALL TESTS PASSED!")
            print("   The refactoring successfully maintained functionality while improving code quality.")
        elif len(failed_tests) <= 2:
            print(f"\n⚠️  OVERALL VERDICT: MOSTLY SUCCESSFUL")
            print("   Most functionality is working, but some issues need attention.")
        else:
            print(f"\n❌ OVERALL VERDICT: SIGNIFICANT ISSUES DETECTED")
            print("   The refactoring may have introduced regressions that need fixing.")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if len(failed_tests) == 0:
            print("   • The refactored code is ready for production use")
            print("   • Consider running additional performance benchmarks")
            print("   • Monitor memory usage in production environments")
        else:
            print("   • Review and fix failing tests before deployment")
            print("   • Consider additional edge case testing")
            print("   • Verify statistical correctness with larger datasets")


def main():
    """Main function to run the test suite."""
    
    # Initialize and run test suite
    test_suite = TestSuite()
    results = test_suite.run_all_tests()
    
    # Return results for further analysis if needed
    return results


if __name__ == "__main__":
    results = main()