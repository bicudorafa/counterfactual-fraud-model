"""Refactored Synthetic Retraining Pipeline for Counterfactual Fraud Model Simulation.

This pipeline extends the synthetic off-policy evaluation by adding model retraining
functionality. Uses composition and dependency injection for loose coupling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

from ..config import SyntheticRetrainingConfig, LoggingPolicyConfig, RetrainingConfig
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
        self._retrained_model: Optional[BaseEstimator] = None
        self._retrain_performance: Optional[Dict[str, float]] = None
        self._retrain_dataset_info: Optional[Dict[str, Any]] = None
        self._preprocessing_info: Optional[Dict[str, Any]] = None
        
        # State for logging policy data
        self._original_results: Optional[Dict[str, Any]] = None
        self._train_policy_data: Optional[pd.DataFrame] = None
        self._test_policy_data: Optional[pd.DataFrame] = None
    
    def run_pipeline(
        self,
        logging_policy_cutoff: float = None,
        logging_policy_exploration_rate: float = None,
        retraining_config: RetrainingConfig = None,
    ) -> Dict[str, Any]:
        """
        Execute the complete synthetic retraining pipeline.
        
        Args:
            logging_policy_cutoff: Score threshold for the original logging policy (overrides config if provided)
            logging_policy_exploration_rate: Rate of exploration for blocked transactions (overrides config if provided)
            retraining_config: Complete retraining configuration (overrides config if provided)
            
        Returns:
            Dictionary containing statistics, metrics, parameters, model performance, 
            dataset info, and optionally data for both original and retrained models
        """
        # Steps 1-2: Generate logging policy data and split into train/test
        self.generate_logging_policy_data(logging_policy_cutoff, logging_policy_exploration_rate)
        
        # Steps 3-6: Retrain model and evaluate performance
        return self.run_retrain_pipeline(retraining_config)

    def run_retrain_pipeline(
        self,
        retraining_config: RetrainingConfig = None,
    ) -> Dict[str, Any]:
        """
        Execute the model retraining and evaluation pipeline on previously generated logging policy data.
        
        This method executes Steps 3-6 from the complete pipeline:
        3. Retrain model on training data
        4. Evaluate retrained model on test data  
        5. Evaluate retrained model performance using counterfactual estimation
        6. Compile final results
        
        Args:
            retraining_config: Complete retraining configuration (overrides config if provided)
            
        Returns:
            Dictionary containing statistics, metrics, parameters, model performance, 
            dataset info, and optionally data for both original and retrained models
            
        Raises:
            ValueError: If logging policy data has not been generated yet
        """
        # Validate that logging policy data has been generated
        if self._train_policy_data is None or self._test_policy_data is None:
            raise ValueError("Logging policy data must be generated before running retrain pipeline. Call generate_logging_policy_data() first.")
        
        # Use provided config or fall back to instance config
        effective_config = retraining_config if retraining_config is not None else self.config.retraining
        
        # Step 3: Retrain model on training data
        self._retrain_model(self._train_policy_data, effective_config)
        
        # Step 4: Evaluate retrained model on test data
        new_scores, binary_predictions = self._predict_on_test_data(self._test_policy_data, effective_config)
        
        # Step 5: Evaluate retrained model performance using counterfactual estimation
        ope_metrics_results = self._evaluate_retrained_model(self._test_policy_data, binary_predictions, new_scores)
        
        return {
            'original_results': self._original_results,
            'retrained_model_performance': self._retrain_performance,
            'ope_metrics': ope_metrics_results
        }

    def generate_logging_policy_data(
        self, 
        logging_policy_cutoff: float = None, 
        logging_policy_exploration_rate: float = None
    ) -> None:
        """
        Generate logging policy data and split into training and test sets.
        
        This method combines Steps 1 and 2 from the pipeline:
        1. Generate logging policy data using the base pipeline
        2. Split the policy data into training and test sets
        
        The results are stored as instance attributes:
        - self._original_results: Original model performance and dataset info
        - self._train_policy_data: Training subset of policy data
        - self._test_policy_data: Test subset of policy data
        
        Args:
            logging_policy_cutoff: Score threshold for the original logging policy (overrides config if provided)
            logging_policy_exploration_rate: Rate of exploration for blocked transactions (overrides config if provided)
        """
        # Step 1: Generate policy data using base pipeline
        policy_data = self.base_pipeline.generate_policy_data(
            cutoff=logging_policy_cutoff,
            exploration_rate=logging_policy_exploration_rate
        )
        
        # Create minimal original_results structure with the information we need
        # This ensures compatibility with the rest of the pipeline
        self._original_results = {
            'model_performance': self.base_pipeline.get_model_performance(policy_data, logging_policy_cutoff),
            'dataset_info': self.base_pipeline.get_dataset_info()
        }
        
        # Step 2: Split policy data into train and test sets
        self._train_policy_data, self._test_policy_data = train_test_split(
            policy_data,
            test_size=self.config.retraining.retrain_test_size,
            random_state=self.config.base_config.synthetic_data.random_state,
            stratify=policy_data['is_fraud'] if len(policy_data['is_fraud'].unique()) > 1 else None
        )

    def _retrain_model(self, train_policy_data: pd.DataFrame, config: RetrainingConfig) -> None:
        """
        Preprocess training data using strategy and train model.
        
        Args:
            train_policy_data: DataFrame containing training policy data
            config: Retraining configuration to use
            
        Returns:
            Preprocessed training data
        """
        # Create preprocessor using the provided config
        data_preprocessor = create_preprocessor(
            config.retrain_model.strategy,
            config.retrain_model.strategy_params
        )
        
        # Step 1: Use preprocessor to prepare training data
        X_train, y_train, sample_weights = data_preprocessor.prepare_training_data(train_policy_data)
        
        # Store preprocessing info for reporting
        self._preprocessing_info = data_preprocessor.get_strategy_info()
        
        # Step 2: Train new model using injected trainer with sample weights
        self._retrained_model = self.model_trainer.train_model(
            X_train, y_train, config.retrain_model.base_model, sample_weights
        )
        
        # Step 3: Calculate dataset info based on preprocessed data
        self._retrain_dataset_info = self._create_preprocessed_data_summary(X_train, y_train)

    def _predict_on_test_data(self, test_policy_data: pd.DataFrame, config: RetrainingConfig) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions on test data using the retrained model.
        
        Args:
            test_policy_data: DataFrame containing test policy data
            config: Retraining configuration to use
            
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
            self._retrained_model, X_test, y_test, config.retrain_model.classification_threshold
        )
        
        # Generate new model scores on test set
        new_scores = self._retrained_model.predict_proba(X_test)[:, 1]  # Probability of fraud
        
        # Convert to binary predictions using classification threshold
        binary_predictions = (new_scores > config.retrain_model.classification_threshold).astype(int)
        
        return new_scores, binary_predictions

    def _evaluate_retrained_model(
        self, 
        test_policy_data: pd.DataFrame, 
        binary_predictions: np.ndarray,
        new_scores: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate retrained model performance using counterfactual estimation.
        
        Args:
            test_policy_data: Test policy data for counterfactual estimation
            binary_predictions: Binary predictions from retrained model (for all test transactions)
            new_scores: Probability scores from retrained model (for all test transactions)
            
        Returns:
            Dictionary containing OPE metrics results
        """
        # Create counterfactual estimator with test data
        estimator = CounterfactualEstimator(
            self.config.base_config.counterfactual_estimator,
            test_policy_data
        )
        
        # Filter both predictions and scores to only allowed transactions (same filtering as estimator does internally)
        allowed_mask = test_policy_data['policy_action'] == 'allow'
        allowed_predictions = binary_predictions[allowed_mask]
        allowed_scores = new_scores[allowed_mask]
        
        # Use both filtered predictions and scores for counterfactual estimation
        return estimator.estimate_ope_metrics_vectorized(allowed_predictions, allowed_scores)
    
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
    
    def get_original_results(self) -> Dict[str, Any]:
        """Get the original model performance and dataset info."""
        if self._original_results is None:
            raise ValueError("Logging policy data must be generated before accessing original results")
        return self._original_results.copy()
    
    def get_train_policy_data(self) -> pd.DataFrame:
        """Get the training subset of policy data."""
        if self._train_policy_data is None:
            raise ValueError("Logging policy data must be generated before accessing training data")
        return self._train_policy_data.copy()
    
    def get_test_policy_data(self) -> pd.DataFrame:
        """Get the test subset of policy data."""
        if self._test_policy_data is None:
            raise ValueError("Logging policy data must be generated before accessing test data")
        return self._test_policy_data.copy()
    
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