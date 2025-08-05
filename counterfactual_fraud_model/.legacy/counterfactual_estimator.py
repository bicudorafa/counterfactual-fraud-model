"""Counterfactual Values Estimator for Fraud Model Evaluation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class CounterfactualValuesEstimator:
    """
    Applies counterfactual evaluation to estimate policy metrics with confidence intervals.
    
    Uses importance sampling and bootstrap estimation to calculate metrics
    for fraud detection policies under counterfactual assumptions.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        n_bootstrap: int = 5000,
        random_state: Optional[int] = None
    ):
        """
        Initialize CounterfactualValuesEstimator.
        
        Args:
            data: DataFrame with columns: is_fraud, model_scores, propensity_score, model_action, policy_action
            n_bootstrap: Number of bootstrap repetitions
            random_state: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        
        # Available metrics - calculate all by default
        self.available_metrics = {
            'precision': self._weighted_precision,
            'recall': self._weighted_recall,
            'f1': self._weighted_f1,
            'fraud_rate': self._weighted_fraud_rate
        }
        
        # Always calculate all available metrics
        self.metrics = list(self.available_metrics.keys())
        
        # Validate data has required columns
        required_columns = ['is_fraud', 'model_scores', 'propensity_score', 'model_action', 'policy_action']
        self._validate_data(data, required_columns)
        
        # Create observed_data as instance attribute
        self.observed_data = data[data['policy_action'] == 'allow'].copy()
        
        if len(self.observed_data) == 0:
            raise ValueError("No transactions with policy_action == 'allow' found. "
                           "Counterfactual evaluation is impossible without observed (allowed) transactions.")
        
        # Calculate importance sampling weights
        self.observed_data['weight'] = 1.0 / self.observed_data['propensity_score']
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _validate_data(self, data: pd.DataFrame, required_columns: List[str]) -> None:
        """Validate that data has all required columns."""
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def _weighted_precision(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           weights: np.ndarray) -> float:
        """Calculate weighted precision."""
        if len(y_pred) == 0 or np.sum(y_pred * weights) == 0:
            return 0.0
        
        true_positives = np.sum(((y_true == 1) & (y_pred == 1)) * weights)
        predicted_positives = np.sum(y_pred * weights)
        
        return true_positives / predicted_positives
    
    def _weighted_recall(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        weights: np.ndarray) -> float:
        """Calculate weighted recall."""
        if len(y_true) == 0 or np.sum(y_true * weights) == 0:
            return 0.0
        
        true_positives = np.sum(((y_true == 1) & (y_pred == 1)) * weights)
        actual_positives = np.sum(y_true * weights)
        
        return true_positives / actual_positives
    
    def _weighted_f1(self, y_true: np.ndarray, y_pred: np.ndarray, 
                    weights: np.ndarray) -> float:
        """Calculate weighted F1 score."""
        precision = self._weighted_precision(y_true, y_pred, weights)
        recall = self._weighted_recall(y_true, y_pred, weights)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _weighted_fraud_rate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           weights: np.ndarray) -> float:
        """Calculate weighted fraud rate (percentage of transactions that are fraudulent)."""
        if len(y_true) == 0 or np.sum(weights) == 0:
            return 0.0
        
        fraud_weight = np.sum((y_true == 1) * weights)
        total_weight = np.sum(weights)
        
        return fraud_weight / total_weight
    
    def estimate_ope_metrics(self, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Estimate counterfactual metrics with confidence intervals using importance sampling.
        
        This method evaluates how well a policy with the given predictions would perform
        using counterfactual estimation. It applies importance sampling to estimate
        metrics for fraud detection policies under counterfactual assumptions.
        
        Args:
            y_pred: Binary prediction array (0/1) of the same size as observed_data.is_fraud
            
        Returns:
            Dictionary with metrics and their statistics (mean, p025, p975, std, n_bootstrap)
        """
        if len(y_pred) != len(self.observed_data):
            raise ValueError(f"y_pred length ({len(y_pred)}) must match observed_data length ({len(self.observed_data)})")
        
        if not np.all(np.isin(y_pred, [0, 1])):
            raise ValueError("y_pred must contain only 0s and 1s")
        
        # Get relevant columns
        y_true = self.observed_data['is_fraud'].values
        weights = self.observed_data['weight'].values
        
        # Bootstrap estimation
        n_samples = len(self.observed_data)
        metric_results = {metric: [] for metric in self.metrics}
        
        for _ in range(self.n_bootstrap):
            # Poisson bootstrap: sample with poisson weights
            poisson_weights = np.random.poisson(1, n_samples)
            
            # Apply poisson weights to importance sampling weights
            bootstrap_weights = weights * poisson_weights
            
            # Skip if all weights are zero
            if np.sum(bootstrap_weights) == 0:
                continue
            
            # Calculate metrics for this bootstrap sample
            for metric_name in self.metrics:
                metric_func = self.available_metrics[metric_name]
                metric_value = metric_func(y_true, y_pred, bootstrap_weights)
                metric_results[metric_name].append(metric_value)
        
        # Calculate statistics for each metric
        results = {}
        for metric_name, values in metric_results.items():
            if len(values) > 0:
                results[metric_name] = {
                    'mean': np.mean(values),
                    'p025': np.percentile(values, 2.5),
                    'p975': np.percentile(values, 97.5),
                    'std': np.std(values),
                    'n_bootstrap': len(values)
                }
            else:
                results[metric_name] = {
                    'mean': 0.0,
                    'p025': 0.0,
                    'p975': 0.0,
                    'std': 0.0,
                    'n_bootstrap': 0
                }
        
        return results
    
    def estimate_policy_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Estimate metrics for the original model policy (model_action).
        
        Evaluates how well the original model policy would perform by converting
        model_action to binary predictions (block=1, allow=0).
        
        Returns:
            Dictionary with metrics and their statistics (mean, p025, p975, std, n_bootstrap)
        """
        if 'model_action' not in self.observed_data.columns:
            raise ValueError("model_action column not found in data. Required for policy metrics evaluation.")
        
        y_pred = (self.observed_data['model_action'] == 'block').astype(int)
        return self.estimate_ope_metrics(y_pred)
    
    def estimate_threshold_metrics(self, threshold: float) -> Dict[str, Dict[str, float]]:
        """
        Estimate metrics for a threshold-based policy.
        
        Evaluates how well a policy with the given threshold would perform
        by converting model scores to binary predictions based on the threshold.
        
        Args:
            threshold: Score threshold for converting model scores to binary predictions
            
        Returns:
            Dictionary with metrics and their statistics (mean, p025, p975, std, n_bootstrap)
        """
        y_pred = (self.observed_data['model_scores'] > threshold).astype(int)
        return self.estimate_ope_metrics(y_pred)
    
    def get_params(self) -> dict:
        """Return the current parameters."""
        return {
            'n_bootstrap': self.n_bootstrap,
            'random_state': self.random_state
        } 