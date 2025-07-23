"""Refactored Synthetic Retraining Pipeline for Counterfactual Fraud Model Simulation.

This pipeline extends the synthetic off-policy evaluation by adding model retraining
functionality. Uses composition and dependency injection for loose coupling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

from ..config import SyntheticRetrainingConfig, LoggingPolicyConfig
from ..protocols import (
    SyntheticDataGeneratorProtocol,
    LoggingPolicyGeneratorProtocol,
    ModelTrainerProtocol,
    PipelineProtocol
)
from ..generators import SyntheticDataGenerator, LoggingPolicyGenerator
from ..estimators import CounterfactualEstimator
from ..core import default_model_trainer
from .synthetic_off_policy_evaluation_pipeline import SyntheticOffPolicyEvaluationPipeline


class SyntheticRetrainingPipeline(PipelineProtocol):
    """
    Pipeline that retrains a model on allowed transactions and evaluates its performance.
    
    This pipeline extends the synthetic off-policy evaluation by:
    1. Generating policy data using the base synthetic pipeline
    2. Filtering to only allowed transactions (model_action == 'allow') 
    3. Training a new model on this filtered data
    4. Generating new model scores and converting to binary predictions
    5. Using counterfactual estimation to evaluate the new model's performance
    """
    
    def __init__(
        self,
        config: SyntheticRetrainingConfig,
        base_pipeline: SyntheticOffPolicyEvaluationPipeline = None,
        model_trainer: ModelTrainerProtocol = default_model_trainer
    ):
        """
        Initialize SyntheticRetrainingPipeline with configuration and dependencies.
        
        Args:
            config: Complete configuration for the retraining pipeline
            base_pipeline: Base pipeline for initial data generation (composition instead of inheritance)
            model_trainer: Trainer for retraining models (dependency injection)
        """
        self.config = config
        self.model_trainer = model_trainer
        
        # Use provided base pipeline or create default one
        self.base_pipeline = base_pipeline or SyntheticOffPolicyEvaluationPipeline(config.base_config)
        
        # State for retrained model and metrics
        self._retrained_model: BaseEstimator = None
        self._retrain_performance: Dict[str, float] = None
        self._retrain_dataset_info: Dict[str, Any] = None
    
    def run_pipeline(
        self,
        cutoff: float = None,
        exploration_rate: float = None,
        include_data: bool = None
    ) -> Dict[str, Any]:
        """
        Execute the complete synthetic retraining pipeline.
        
        Args:
            cutoff: Score threshold for the original logging policy (overrides config if provided)
            exploration_rate: Rate of exploration for blocked transactions (overrides config if provided)
            include_data: Whether to include the full dataset in results (overrides config if provided)
            
        Returns:
            Dictionary containing statistics, metrics, parameters, model performance, 
            dataset info, and optionally data for both original and retrained models
        """
        # Use provided parameters or fall back to config
        include_data_flag = include_data if include_data is not None else self.config.base_config.pipeline.include_data
        
        # Step 1: Run base pipeline to get policy data
        original_results = self.base_pipeline.run_pipeline(
            cutoff=cutoff,
            exploration_rate=exploration_rate,
            include_data=True  # We need the data for retraining
        )
        policy_data = original_results['data']
        
        # Step 2: Filter to only allowed transactions (model_action == 'allow')
        allowed_data = policy_data[policy_data['model_action'] == 'allow'].copy()
        
        if len(allowed_data) == 0:
            raise ValueError("No transactions with model_action == 'allow' found. Cannot retrain model.")
        
        # Step 3: Prepare features and target for retraining
        feature_columns = [col for col in allowed_data.columns 
                          if col.startswith('feature_') or col.startswith('x')]
        if len(feature_columns) == 0:
            raise ValueError("No feature columns found in data. Expected columns starting with 'feature_' or 'x'")
        
        X = allowed_data[feature_columns]
        y = allowed_data['is_fraud']
        
        # Step 4: Split into train/test for retraining
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.retraining.retrain_test_size,
            random_state=self.config.base_config.synthetic_data.random_state,
            stratify=y if len(y.unique()) > 1 else None
        )
        
        # Step 5: Train new model using injected trainer
        self._retrained_model = self.model_trainer.train_model(
            X_train, y_train, self.config.retraining.retrain_model
        )
        
        # Step 6: Calculate retrained model performance
        self._retrain_performance = self.model_trainer.calculate_performance(
            self._retrained_model, X_test, y_test
        )
        
        # Step 7: Calculate dataset info
        self._retrain_dataset_info = self._calculate_dataset_info(allowed_data)
        
        # Step 8: Generate new model scores on test set
        new_scores = self._retrained_model.predict_proba(X_test)[:, 1]  # Probability of fraud
        
        # Step 9: Convert to binary predictions using classification threshold
        binary_predictions = (new_scores > self.config.retraining.classification_threshold).astype(int)
        
        # Step 10: Use CounterfactualEstimator with binary predictions
        # Create a subset of allowed_data corresponding to X_test for counterfactual estimation
        test_indices = X_test.index
        test_allowed_data = allowed_data.loc[test_indices].copy()
        
        estimator = CounterfactualEstimator(
            self.config.base_config.counterfactual_estimator, 
            test_allowed_data
        )
        
        # Use the binary predictions directly for counterfactual estimation
        ope_metrics_results = estimator.estimate_ope_metrics(binary_predictions)
        
        # Step 11: Calculate summary statistics
        summary_results = self._calculate_summary_statistics(
            policy_data, allowed_data, test_allowed_data, new_scores, binary_predictions
        )
        
        # Step 12: Compile parameters used in this run
        run_parameters = {
            'base_config': self.config.base_config.model_dump(),
            'retraining': self.config.retraining.model_dump(),
            'runtime_overrides': {
                'cutoff': cutoff,
                'exploration_rate': exploration_rate,
                'include_data': include_data_flag
            }
        }
        
        # Step 13: Compile results
        results = {
            **summary_results,
            'ope_metrics': ope_metrics_results,
            'original_model_performance': original_results['model_performance'],
            'retrained_model_performance': self._retrain_performance,
            'original_dataset_info': original_results['dataset_info'],
            'retrained_dataset_info': self._retrain_dataset_info,
            'parameters': run_parameters
        }
        
        if include_data_flag:
            results['original_data'] = policy_data
            results['retrained_data'] = test_allowed_data
            results['new_scores'] = new_scores
            results['binary_predictions'] = binary_predictions
            
        return results
    
    def get_config(self) -> SyntheticRetrainingConfig:
        """Get the current configuration."""
        return self.config
    
    def get_retrained_model_performance(self) -> Dict[str, float]:
        """Get performance metrics of the retrained model."""
        if self._retrain_performance is None:
            raise ValueError("Pipeline must be run before accessing retrained model performance")
        return self._retrain_performance.copy()
    
    def get_retrained_dataset_info(self) -> Dict[str, Any]:
        """Get information about the retrained dataset."""
        if self._retrain_dataset_info is None:
            raise ValueError("Pipeline must be run before accessing retrained dataset info")
        return self._retrain_dataset_info.copy()
    
    def _calculate_dataset_info(self, allowed_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate information about the retrained dataset."""
        return {
            'n_samples': len(allowed_data),
            'n_features': len([col for col in allowed_data.columns 
                               if col.startswith('feature_') or col.startswith('x')]),
            'fraud_rate': allowed_data['is_fraud'].mean(),
            'n_fraud': allowed_data['is_fraud'].sum(),
            'n_legitimate': (allowed_data['is_fraud'] == 0).sum()
        }
    
    def _calculate_summary_statistics(
        self, 
        policy_data: pd.DataFrame,
        allowed_data: pd.DataFrame,
        test_allowed_data: pd.DataFrame,
        new_scores: np.ndarray,
        binary_predictions: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics for the retraining pipeline."""
        total_transactions = len(policy_data)
        total_allowed = len(allowed_data)
        
        # Original model performance on policy data
        original_approval_rate = (policy_data['model_action'] == 'allow').mean()
        original_allowed_data = policy_data[policy_data['model_action'] == 'allow']
        original_fraud_rate = original_allowed_data['is_fraud'].mean() if len(original_allowed_data) > 0 else 0.0
        
        # New model predictions statistics
        new_block_rate = binary_predictions.mean()
        new_allow_rate = 1 - new_block_rate
        
        # Calculate fraud rate for new model's allowed transactions
        # Use test_allowed_data which corresponds to binary_predictions
        allowed_by_new_model = test_allowed_data[binary_predictions == 0]  # 0 means allow
        new_model_fraud_rate = allowed_by_new_model['is_fraud'].mean() if len(allowed_by_new_model) > 0 else 0.0
        
        return {
            'statistics': {
                'total_transactions': total_transactions,
                'total_allowed_for_retraining': total_allowed,
                'original_approval_rate': original_approval_rate,
                'original_fraud_rate': original_fraud_rate,
                'new_model_block_rate': new_block_rate,
                'new_model_allow_rate': new_allow_rate,
                'new_model_fraud_rate': new_model_fraud_rate,
                'classification_threshold': self.config.retraining.classification_threshold
            }
        } 