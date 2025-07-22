"""Synthetic Retraining Pipeline for Counterfactual Fraud Model Simulation."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from .synthetic_pipeline import SyntheticOffPolicyEvaluationPipeline
from .counterfactual_estimator import CounterfactualValuesEstimator
import lightgbm as lgb


class SyntheticRetrainingPipeline:
    """
    Pipeline that retrains a model on allowed transactions and evaluates its performance.
    
    This pipeline extends the synthetic off-policy evaluation by:
    1. Generating policy data using the original synthetic pipeline
    2. Filtering to only allowed transactions (model_action == 'allow')
    3. Training a new model on this filtered data
    4. Generating new model scores and converting to binary predictions
    5. Using counterfactual estimation to evaluate the new model's performance
    """
    
    def __init__(
        self,
        # Original pipeline parameters
        n_samples: int = 100_000,
        n_features: int = 30,
        n_informative: int = 15,
        n_redundant: int = 5,
        n_repeated: int = 0,
        n_clusters_per_class: int = 2,
        weights: Optional[List[float]] = None,
        flip_y: float = 0.01,
        class_sep: float = 1.0,
        model_type: str = "lightgbm",
        model_params: Optional[Dict[str, Any]] = None,
        test_size: float = 0.3,
        n_bootstrap: int = 5000,
        # New retraining parameters
        retrain_test_size: float = 0.3,
        classification_threshold: float = 0.05,
        retrain_model_type: str = "lightgbm",
        retrain_model_params: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize SyntheticRetrainingPipeline.
        
        Args:
            # Original pipeline args (same as SyntheticOffPolicyEvaluationPipeline)
            n_samples: Number of samples to generate
            n_features: Total number of features  
            n_informative: Number of informative features
            n_redundant: Number of redundant features
            n_repeated: Number of duplicated features
            n_clusters_per_class: Number of clusters per class
            weights: Class balance (e.g., [0.99, 0.01] for imbalanced fraud data)
            flip_y: Fraction of samples whose class is flipped (label noise)
            class_sep: Factor multiplying the hypercube size
            model_type: Type of original model ("lightgbm", "random_forest", "logistic")
            model_params: Additional parameters for the original model
            test_size: Fraction of data to use for testing original model
            n_bootstrap: Number of bootstrap repetitions for counterfactual estimation
            
            # New retraining args
            retrain_test_size: Fraction of allowed data to use for testing retrained model
            classification_threshold: Threshold for converting new scores to binary predictions
            retrain_model_type: Type of model to retrain ("lightgbm", "random_forest", "logistic")
            retrain_model_params: Additional parameters for the retrained model
            random_state: Random seed for reproducibility across all components
        """
        # Validate inputs
        self._validate_initialization_params(
            n_samples, n_features, n_informative, n_redundant, 
            test_size, n_bootstrap, retrain_test_size, classification_threshold
        )
        
        # Store all parameters
        self._retrain_test_size = retrain_test_size
        self._classification_threshold = classification_threshold
        self._retrain_model_type = retrain_model_type
        self._retrain_model_params = retrain_model_params or {}
        self._random_state = random_state
        
        # Initialize the original synthetic pipeline
        self._original_pipeline = SyntheticOffPolicyEvaluationPipeline(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=n_repeated,
            n_clusters_per_class=n_clusters_per_class,
            weights=weights,
            flip_y=flip_y,
            class_sep=class_sep,
            model_type=model_type,
            model_params=model_params,
            test_size=test_size,
            n_bootstrap=n_bootstrap,
            random_state=random_state
        )
        
        # Store retrained model and performance metrics
        self._retrained_model = None
        self._retrain_performance = None
        self._retrain_dataset_info = None
    
    @property
    def params(self) -> Dict[str, Any]:
        """Return the current parameters."""
        original_params = self._original_pipeline.params
        retrain_params = {
            'retrain_test_size': self._retrain_test_size,
            'classification_threshold': self._classification_threshold,
            'retrain_model_type': self._retrain_model_type,
            'retrain_model_params': self._retrain_model_params,
        }
        return {**original_params, **retrain_params}
    
    def _validate_initialization_params(
        self, 
        n_samples: int,
        n_features: int,
        n_informative: int,
        n_redundant: int,
        test_size: float,
        n_bootstrap: int,
        retrain_test_size: float,
        classification_threshold: float
    ) -> None:
        """Validate initialization parameters."""
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if n_informative <= 0:
            raise ValueError("n_informative must be positive")
        if n_informative > n_features:
            raise ValueError("n_informative cannot exceed n_features")
        if not 0 <= test_size <= 1:
            raise ValueError("test_size must be between 0 and 1")
        if not 0 <= retrain_test_size <= 1:
            raise ValueError("retrain_test_size must be between 0 and 1")
        if not 0 <= classification_threshold <= 1:
            raise ValueError("classification_threshold must be between 0 and 1")
        if n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be positive")
    
    def _validate_pipeline_params(self, cutoff: float, exploration_rate: float) -> None:
        """Validate pipeline execution parameters."""
        if not 0 <= cutoff <= 1:
            raise ValueError("cutoff must be between 0 and 1")
        if not 0 <= exploration_rate <= 1:
            raise ValueError("exploration_rate must be between 0 and 1")
    
    def _create_retrained_model(self) -> Any:
        """Create and return a new model instance for retraining."""
        if self._retrain_model_type == "lightgbm":
            default_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': self._random_state
            }
            params = {**default_params, **self._retrain_model_params}
            return lgb.LGBMClassifier(**params)
        
        elif self._retrain_model_type == "random_forest":
            default_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': self._random_state
            }
            params = {**default_params, **self._retrain_model_params}
            return RandomForestClassifier(**params)
        
        elif self._retrain_model_type == "logistic":
            default_params = {
                'random_state': self._random_state,
                'max_iter': 1000
            }
            params = {**default_params, **self._retrain_model_params}
            return LogisticRegression(**params)
        
        else:
            raise ValueError(f"Unsupported retrain_model_type: {self._retrain_model_type}")
    
    def _calculate_model_performance(
        self, 
        model: Any, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Calculate performance metrics for the retrained model."""
        # Get predictions and probabilities
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        
        # Calculate metrics
        performance = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return performance
    
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
                'classification_threshold': self._classification_threshold
            }
        }
    
    def get_retrained_model_performance(self) -> Dict[str, float]:
        """
        Get performance metrics of the retrained model.
        
        Returns:
            Dictionary with performance metrics
        """
        if self._retrain_performance is None:
            raise ValueError("Pipeline must be run before accessing retrained model performance")
        return self._retrain_performance.copy()
    
    def get_retrained_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the retrained dataset.
        
        Returns:
            Dictionary with dataset information
        """
        if self._retrain_dataset_info is None:
            raise ValueError("Pipeline must be run before accessing retrained dataset info")
        return self._retrain_dataset_info.copy()
    
    def run_pipeline(
        self,
        cutoff: float = 0.05,
        exploration_rate: float = 0.05,
        include_data: bool = True
    ) -> Dict[str, Any]:
        """
        Execute the complete synthetic retraining pipeline.
        
        Args:
            cutoff: Score threshold for the original logging policy (0-1)
            exploration_rate: Rate of exploration for blocked transactions (0-1)
            include_data: Whether to include the full dataset in results
            
        Returns:
            Dictionary containing statistics, metrics, parameters, model performance, 
            dataset info, and optionally data for both original and retrained models
        """
        # Validate parameters
        self._validate_pipeline_params(cutoff, exploration_rate)
        
        # Step 1: Run original pipeline to get policy_data
        original_results = self._original_pipeline.run_pipeline(
            cutoff=cutoff, 
            exploration_rate=exploration_rate, 
            include_data=True
        )
        policy_data = original_results['data']
        
        # Step 2: Filter to only allowed transactions (model_action == 'allow')
        allowed_data = policy_data[policy_data['model_action'] == 'allow'].copy()
        
        if len(allowed_data) == 0:
            raise ValueError("No transactions with model_action == 'allow' found. Cannot retrain model.")
        
        # Step 3: Prepare features (exclude model_scores) and target
        feature_columns = [col for col in allowed_data.columns 
                          if col.startswith('feature_') or col.startswith('x')]
        if len(feature_columns) == 0:
            raise ValueError("No feature columns found in data. Expected columns starting with 'feature_' or 'x'")
        
        X = allowed_data[feature_columns]
        y = allowed_data['is_fraud']
        
        # Step 4: Split into train/test for retraining
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self._retrain_test_size,
            random_state=self._random_state,
            stratify=y if len(y.unique()) > 1 else None
        )
        
        # Step 5: Train new model
        retrained_model = self._create_retrained_model()
        retrained_model.fit(X_train, y_train)
        
        # Step 6: Generate new model scores on test set
        new_scores = retrained_model.predict_proba(X_test)[:, 1]  # Probability of fraud
        
        # Step 7: Convert to binary predictions using classification threshold
        binary_predictions = (new_scores > self._classification_threshold).astype(int)
        
        # Step 8: Calculate retrained model performance
        self._retrain_performance = self._calculate_model_performance(retrained_model, X_test, y_test)
        self._retrain_dataset_info = self._calculate_dataset_info(allowed_data)
        
        # Step 9: Use CounterfactualValuesEstimator with binary predictions
        # We need to create a subset of allowed_data corresponding to X_test for counterfactual estimation
        test_indices = X_test.index
        test_allowed_data = allowed_data.loc[test_indices].copy()
        
        estimator = CounterfactualValuesEstimator(
            data=test_allowed_data,
            n_bootstrap=self._original_pipeline._n_bootstrap,
            random_state=self._random_state
        )
        
        # Use the binary predictions directly for counterfactual estimation
        ope_metrics_results = estimator.estimate_ope_metrics(binary_predictions)
        
        # Step 10: Calculate summary statistics
        summary_results = self._calculate_summary_statistics(
            policy_data, allowed_data, test_allowed_data, new_scores, binary_predictions
        )
        
        # Step 11: Compile parameters used in this run
        run_parameters = self.params.copy()
        run_parameters.update({
            'cutoff': cutoff,
            'exploration_rate': exploration_rate
        })
        
        # Step 12: Compile results
        results = {
            **summary_results,
            'ope_metrics': ope_metrics_results,
            'original_model_performance': original_results['model_performance'],
            'retrained_model_performance': self._retrain_performance,
            'original_dataset_info': original_results['dataset_info'],
            'retrained_dataset_info': self._retrain_dataset_info,
            'parameters': run_parameters
        }
        
        if include_data:
            results['original_data'] = policy_data
            results['retrained_data'] = test_allowed_data
            results['new_scores'] = new_scores
            results['binary_predictions'] = binary_predictions
            
        return results
    
    def get_params(self) -> Dict[str, Any]:
        """Return the current parameters."""
        return self.params 