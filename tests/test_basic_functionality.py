"""Basic functionality tests for counterfactual fraud model."""

import pytest
import numpy as np
import pandas as pd
from src.counterfactual_fraud_model import (
    DataGenerator,
    LoggingPolicyGenerator,
    CounterfactualValuesEstimator,
    OffPolicyEvaluationPipeline,
    OffPolicyEvaluationSimulator
)


class TestDataGenerator:
    """Test the DataGenerator class."""
    
    def test_data_generator_initialization(self):
        """Test that DataGenerator initializes correctly."""
        generator = DataGenerator(random_state=42)
        assert generator.alpha == 0.5
        assert generator.beta_param == 10.0
        assert generator.sample_size == 10_000
    
    def test_data_generation(self):
        """Test data generation produces correct structure."""
        generator = DataGenerator(sample_size=1000, random_state=42)
        data = generator.generate_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 1000
        assert list(data.columns) == ['model_scores', 'is_fraud']
        assert data['model_scores'].between(0, 1).all()
        assert data['is_fraud'].isin([0, 1]).all()
    
    def test_data_generation_reproducible(self):
        """Test that data generation is reproducible with same random state."""
        generator1 = DataGenerator(sample_size=100, random_state=42)
        generator2 = DataGenerator(sample_size=100, random_state=42)
        
        data1 = generator1.generate_data()
        data2 = generator2.generate_data()
        
        pd.testing.assert_frame_equal(data1, data2)


class TestLoggingPolicyGenerator:
    """Test the LoggingPolicyGenerator class."""
    
    def test_logging_policy_initialization(self):
        """Test that LoggingPolicyGenerator initializes correctly."""
        policy = LoggingPolicyGenerator(random_state=42)
        assert policy.cutoff == 0.05
        assert policy.exploration_rate == 0.5
        assert policy.propensity_type == "uniform"
    
    def test_policy_generation(self):
        """Test policy generation produces correct structure."""
        # Generate test data
        generator = DataGenerator(sample_size=100, random_state=42)
        data = generator.generate_data()
        
        # Apply policy
        policy = LoggingPolicyGenerator(random_state=42)
        policy_data = policy.generate_policy(data)
        
        # Check structure
        expected_columns = ['model_scores', 'is_fraud', 'propensity_score', 'action']
        assert list(policy_data.columns) == expected_columns
        assert len(policy_data) == 100
        
        # Check data types and values
        assert policy_data['action'].isin(['allow', 'block']).all()
        assert policy_data['propensity_score'].between(0, 1).all()
    
    def test_below_cutoff_always_allowed(self):
        """Test that transactions below cutoff are always allowed."""
        # Create data with scores below cutoff
        data = pd.DataFrame({
            'model_scores': [0.01, 0.02, 0.03, 0.04],
            'is_fraud': [0, 1, 0, 1]
        })
        
        policy = LoggingPolicyGenerator(cutoff=0.05, random_state=42)
        result = policy.generate_policy(data)
        
        # All should be allowed with propensity score 1.0
        assert (result['action'] == 'allow').all()
        assert (result['propensity_score'] == 1.0).all()
    
    def test_above_cutoff_exploration(self):
        """Test that transactions above cutoff are explored based on exploration rate."""
        # Create data with scores above cutoff
        data = pd.DataFrame({
            'model_scores': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'is_fraud': [0, 1, 0, 1, 0, 1, 0, 1, 0]
        })
        
        policy = LoggingPolicyGenerator(cutoff=0.05, exploration_rate=0.5, random_state=42)
        result = policy.generate_policy(data)
        
        # Check that some are allowed and some are blocked
        allowed = result['action'] == 'allow'
        blocked = result['action'] == 'block'
        
        assert allowed.sum() > 0  # Some should be allowed
        assert blocked.sum() > 0  # Some should be blocked
        
        # Check propensity scores
        assert (result.loc[allowed, 'propensity_score'] == 0.5).all()
        assert (result.loc[blocked, 'propensity_score'] == 0.0).all()


class TestCounterfactualValuesEstimator:
    """Test the CounterfactualValuesEstimator class."""
    
    def test_estimator_initialization(self):
        """Test that CounterfactualValuesEstimator initializes correctly."""
        estimator = CounterfactualValuesEstimator(n_bootstrap=100, random_state=42)
        assert estimator.n_bootstrap == 100
        assert estimator.metrics == ['precision', 'recall']
    
    def test_metric_estimation(self):
        """Test that metric estimation works with simple data."""
        # Create simple test data
        generator = DataGenerator(sample_size=200, random_state=42)
        data = generator.generate_data()
        
        policy = LoggingPolicyGenerator(exploration_rate=0.8, random_state=42)
        policy_data = policy.generate_policy(data)
        
        estimator = CounterfactualValuesEstimator(n_bootstrap=50, random_state=42)
        results = estimator.estimate_policy_metrics(policy_data, policy_threshold=0.5)
        
        # Check structure
        assert 'precision' in results
        assert 'recall' in results
        
        for metric_name in ['precision', 'recall']:
            metric_stats = results[metric_name]
            assert 'mean' in metric_stats
            assert 'p025' in metric_stats
            assert 'p975' in metric_stats
            assert 'std' in metric_stats
            assert 'n_bootstrap' in metric_stats
            
            # Check that values are reasonable
            assert 0 <= metric_stats['mean'] <= 1
            assert 0 <= metric_stats['p025'] <= 1
            assert 0 <= metric_stats['p975'] <= 1


class TestOffPolicyEvaluationPipeline:
    """Test the OffPolicyEvaluationPipeline class."""
    
    def test_pipeline_initialization(self):
        """Test that pipeline initializes correctly."""
        pipeline = OffPolicyEvaluationPipeline(sample_size=500, random_state=42)
        assert pipeline.params['sample_size'] == 500
        assert pipeline.params['random_state'] == 42
    
    def test_pipeline_execution(self):
        """Test that pipeline executes end-to-end."""
        pipeline = OffPolicyEvaluationPipeline(
            sample_size=200,
            n_bootstrap=20,
            random_state=42
        )
        
        results = pipeline.run_pipeline()
        
        # Check structure
        assert 'data' in results
        assert 'metrics' in results
        assert 'statistics' in results
        assert 'parameters' in results
        
        # Check data
        data = results['data']
        assert len(data) == 200
        assert 'model_scores' in data.columns
        assert 'is_fraud' in data.columns
        assert 'action' in data.columns
        
        # Check metrics
        metrics = results['metrics']
        assert 'precision' in metrics
        assert 'recall' in metrics


class TestOffPolicyEvaluationSimulator:
    """Test the OffPolicyEvaluationSimulator class."""
    
    def test_simulator_initialization(self):
        """Test that simulator initializes correctly."""
        simulator = OffPolicyEvaluationSimulator(sample_size=100, random_state=42)
        assert simulator.base_params['sample_size'] == 100
        assert simulator.metrics == ['precision', 'recall']
    
    def test_exploration_rate_simulation(self):
        """Test simulation across multiple exploration rates."""
        simulator = OffPolicyEvaluationSimulator(
            sample_size=200,
            n_bootstrap=10,
            random_state=42
        )
        
        exploration_rates = [0.1, 0.3, 0.5]
        results = simulator.simulate_exploration_rates(exploration_rates)
        
        # Check structure
        assert 'exploration_rates' in results
        assert 'metrics_by_rate' in results
        assert 'data' in results
        assert 'baseline_data' in results
        
        # Check that we have results for each exploration rate
        assert len(results['metrics_by_rate']) == 3
        for rate in exploration_rates:
            assert rate in results['metrics_by_rate']
            assert 'metrics' in results['metrics_by_rate'][rate]
            assert 'statistics' in results['metrics_by_rate'][rate]


def test_integration():
    """Test that all components work together in integration."""
    # Create a complete workflow
    np.random.seed(42)
    
    # Generate data
    generator = DataGenerator(sample_size=500, random_state=42)
    data = generator.generate_data()
    
    # Apply logging policy
    policy = LoggingPolicyGenerator(exploration_rate=0.6, random_state=42)
    policy_data = policy.generate_policy(data)
    
    # Estimate metrics
    estimator = CounterfactualValuesEstimator(n_bootstrap=50, random_state=42)
    metrics = estimator.estimate_policy_metrics(policy_data)
    
    # Run full pipeline
    pipeline = OffPolicyEvaluationPipeline(sample_size=300, n_bootstrap=20, random_state=42)
    pipeline_results = pipeline.run_pipeline()
    
    # Run simulator
    simulator = OffPolicyEvaluationSimulator(sample_size=200, n_bootstrap=10, random_state=42)
    sim_results = simulator.simulate_exploration_rates([0.2, 0.4])
    
    # Basic checks that everything executed without errors
    assert len(data) == 500
    assert 'action' in policy_data.columns
    assert 'precision' in metrics
    assert 'data' in pipeline_results
    assert len(sim_results['metrics_by_rate']) == 2


if __name__ == "__main__":
    # Run some basic tests if executed directly
    test_integration()
    print("Basic integration test passed!") 