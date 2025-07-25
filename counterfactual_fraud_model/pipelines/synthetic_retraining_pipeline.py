"""Refactored Synthetic Retraining Pipeline for Counterfactual Fraud Model Simulation.

This pipeline extends the synthetic off-policy evaluation by adding model retraining
functionality. Uses composition and dependency injection for loose coupling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

from ..config import SyntheticRetrainingConfig, LoggingPolicyConfig
from ..protocols import (
    SyntheticDataGeneratorProtocol,
    LoggingPolicyGeneratorProtocol,
    ModelTrainerProtocol,
    RetrainingDataPreprocessorProtocol,
    PipelineProtocol
)
from ..generators import SyntheticDataGenerator, LoggingPolicyGenerator, create_preprocessor
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
        model_trainer: ModelTrainerProtocol = default_model_trainer,
        data_preprocessor: RetrainingDataPreprocessorProtocol = None
    ):
        """
        Initialize SyntheticRetrainingPipeline with configuration and dependencies.
        
        Args:
            config: Complete configuration for the retraining pipeline
            base_pipeline: Base pipeline for initial data generation (composition instead of inheritance)
            model_trainer: Trainer for retraining models (dependency injection)
            data_preprocessor: Preprocessor for retraining data (dependency injection)
        """
        self.config = config
        self.model_trainer = model_trainer
        
        # Use provided base pipeline or create default one
        self.base_pipeline = base_pipeline or SyntheticOffPolicyEvaluationPipeline(config.base_config)
        
        # Use provided preprocessor or create one based on config
        self.data_preprocessor = data_preprocessor or create_preprocessor(
            config.retraining.retrain_model.strategy,
            config.retraining.retrain_model.strategy_params
        )
        
        # State for retrained model and metrics
        self._retrained_model: BaseEstimator = None
        self._retrain_performance: Dict[str, float] = None
        self._retrain_dataset_info: Dict[str, Any] = None
        self._preprocessing_info: Dict[str, Any] = None
    
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
        # Step 1: Generate logging policy data
        original_results, policy_data = self._generate_logging_policy_data(
            cutoff, exploration_rate
        )
        
        # Step 2: Split policy data into train and test
        train_policy_data, test_policy_data = train_test_split(
            policy_data,
            test_size=self.config.retraining.retrain_test_size,
            random_state=self.config.base_config.synthetic_data.random_state,
            stratify=policy_data['is_fraud'] if len(policy_data['is_fraud'].unique()) > 1 else None
        )
        
        # Step 3: Retrain model on training data
        preprocessed_train_data = self._retrain_model(train_policy_data)
        
        # Step 4: Evaluate retrained model on test data
        new_scores, binary_predictions = self._predict_on_test_data(test_policy_data)
        
        # Step 5: Evaluate retrained model performance using counterfactual estimation
        ope_metrics_results = self._evaluate_retrained_model(test_policy_data, binary_predictions)
        
        # Step 6: Compile final results
        return self._compile_results(
            original_results, 
            ope_metrics_results
        )

    def _generate_logging_policy_data(
        self, 
        cutoff: float = None, 
        exploration_rate: float = None
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Generate logging policy data using the base pipeline's generate_policy_data method.
        
        Args:
            cutoff: Score threshold for the original logging policy
            exploration_rate: Rate of exploration for blocked transactions
            
        Returns:
            Tuple of (original_results, policy_data)
        """
        # Generate policy data directly without running the full pipeline
        policy_data = self.base_pipeline.generate_policy_data(
            cutoff=cutoff,
            exploration_rate=exploration_rate
        )
        
        # Create minimal original_results structure with the information we need
        # This ensures compatibility with the rest of the pipeline
        original_results = {
            'model_performance': self.base_pipeline.get_model_performance(),
            'dataset_info': self.base_pipeline.get_dataset_info()
        }
        
        return original_results, policy_data

    def _retrain_model(self, train_policy_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess training data using strategy and train model.
        
        Args:
            train_policy_data: DataFrame containing training policy data
            
        Returns:
            Preprocessed training data
        """
        # Step 1: Use preprocessor to prepare training data
        X_train, y_train, sample_weights = self.data_preprocessor.prepare_training_data(train_policy_data)
        
        # Store preprocessing info for reporting
        self._preprocessing_info = self.data_preprocessor.get_strategy_info()
        
        # Step 2: Train new model using injected trainer with sample weights
        self._retrained_model = self.model_trainer.train_model(
            X_train, y_train, self.config.retraining.retrain_model.base_model, sample_weights
        )
        
        # Step 3: Calculate dataset info based on preprocessed data
        self._retrain_dataset_info = self._calculate_dataset_info_from_preprocessed(X_train, y_train)
        
        # Return preprocessed training data summary
        return self._create_preprocessed_data_summary(X_train, y_train)

    def _predict_on_test_data(self, test_policy_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions on test data using the retrained model.
        
        Args:
            test_policy_data: DataFrame containing test policy data
            
        Returns:
            Tuple of (new_scores, binary_predictions)
        """
        # Extract feature columns directly from test data (agnostic to training strategy)
        feature_columns = [col for col in test_policy_data.columns 
                          if col.startswith('feature_') or col.startswith('x')]
        if len(feature_columns) == 0:
            raise ValueError("No feature columns found in test data. Expected columns starting with 'feature_' or 'x'")
        
        X_test = test_policy_data[feature_columns]
        y_test = test_policy_data['is_fraud']
        
        # Calculate retrained model performance on test data
        self._retrain_performance = self.model_trainer.calculate_performance(
            self._retrained_model, X_test, y_test
        )
        
        # Generate new model scores on test set
        new_scores = self._retrained_model.predict_proba(X_test)[:, 1]  # Probability of fraud
        
        # Convert to binary predictions using classification threshold
        binary_predictions = (new_scores > self.config.retraining.retrain_model.classification_threshold).astype(int)
        
        return new_scores, binary_predictions

    def _evaluate_retrained_model(
        self, 
        test_policy_data: pd.DataFrame, 
        binary_predictions: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate retrained model performance using counterfactual estimation.
        
        Args:
            test_policy_data: Test policy data for counterfactual estimation
            binary_predictions: Binary predictions from retrained model (for all test transactions)
            
        Returns:
            Dictionary containing OPE metrics results
        """
        # Create counterfactual estimator with test data
        estimator = CounterfactualEstimator(
            self.config.base_config.counterfactual_estimator,
            test_policy_data
        )
        
        # Filter binary predictions to only allowed transactions (same filtering as estimator does internally)
        allowed_mask = test_policy_data['policy_action'] == 'allow'
        allowed_predictions = binary_predictions[allowed_mask]
        
        # Use the filtered predictions for counterfactual estimation
        return estimator.estimate_ope_metrics(allowed_predictions)

    def _compile_results(
        self,
        original_results: Dict[str, Any],
        ope_metrics_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compile final results from all pipeline steps.
        
        Args:
            original_results: Results from base pipeline
            ope_metrics_results: Counterfactual estimation results
            
        Returns:
            Simplified results dictionary with only key information
        """
        # Compile results with only the three key pieces of information
        results = {
            'original_results': original_results,
            'retrained_model_performance': self._retrain_performance,
            'ope_metrics': ope_metrics_results
        }
            
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
    
    def _calculate_dataset_info_from_preprocessed(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Calculate information about the preprocessed dataset."""
        return {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'fraud_rate': y.mean(),
            'n_fraud': y.sum(),
            'n_legitimate': (y == 0).sum(),
            'preprocessing_strategy': self._preprocessing_info.get('strategy', 'unknown') if self._preprocessing_info else 'unknown'
        }
    
    def _create_preprocessed_data_summary(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Create a summary of the preprocessed training data."""
        return {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'feature_names': list(X.columns),
            'fraud_rate': y.mean(),
            'n_fraud': y.sum(),
            'n_legitimate': (y == 0).sum(),
            'class_distribution': y.value_counts().to_dict()
        }
    
    def _calculate_summary_statistics(
        self, 
        policy_data: pd.DataFrame,
        preprocessed_train_summary: Dict[str, Any],
        test_policy_data: pd.DataFrame,
        new_scores: np.ndarray,
        binary_predictions: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics for the retraining pipeline."""
        total_transactions = len(policy_data)
        total_preprocessed_for_training = preprocessed_train_summary['n_samples']
        
        # Original model performance on policy data
        original_approval_rate = (policy_data['model_action'] == 'allow').mean()
        original_allowed_data = policy_data[policy_data['model_action'] == 'allow']
        original_fraud_rate = original_allowed_data['is_fraud'].mean() if len(original_allowed_data) > 0 else 0.0
        
        # New model predictions statistics
        new_block_rate = binary_predictions.mean()
        new_allow_rate = 1 - new_block_rate
        
        # Calculate fraud rate for new model's allowed transactions
        # Note: binary_predictions correspond to test_policy_data rows
        allowed_by_new_model_mask = binary_predictions == 0  # 0 means allow
        new_model_fraud_rate = test_policy_data['is_fraud'][allowed_by_new_model_mask].mean() if allowed_by_new_model_mask.sum() > 0 else 0.0
        
        return {
            'statistics': {
                'total_transactions': total_transactions,
                'total_preprocessed_for_training': total_preprocessed_for_training,
                'original_approval_rate': original_approval_rate,
                'original_fraud_rate': original_fraud_rate,
                'new_model_block_rate': new_block_rate,
                'new_model_allow_rate': new_allow_rate,
                'new_model_fraud_rate': new_model_fraud_rate,
                'classification_threshold': self.config.retraining.retrain_model.classification_threshold,
                'preprocessing_strategy': self._preprocessing_info.get('strategy', 'unknown') if self._preprocessing_info else 'unknown'
            }
        } 