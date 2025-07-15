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
        n_bootstrap: int = 1000,
        metrics: Optional[List[str]] = None,
        random_state: Optional[int] = None
    ):
        """
        Initialize CounterfactualValuesEstimator.
        
        Args:
            n_bootstrap: Number of bootstrap repetitions
            metrics: List of metric names to calculate ('precision', 'recall', 'f1')
            random_state: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        
        if metrics is None:
            metrics = ['precision', 'recall']
        self.metrics = metrics
        
        # Available metrics
        self.available_metrics = {
            'precision': self._weighted_precision,
            'recall': self._weighted_recall,
            'f1': self._weighted_f1
        }
        
        # Validate requested metrics
        for metric in self.metrics:
            if metric not in self.available_metrics:
                raise ValueError(f"Unknown metric: {metric}. Available metrics: {list(self.available_metrics.keys())}")
        
        if random_state is not None:
            np.random.seed(random_state)
    
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
    
    def estimate_metrics(
        self, 
        data: pd.DataFrame, 
        policy_threshold: float = 0.5
    ) -> Dict[str, Dict[str, float]]:
        """
        Estimate counterfactual metrics with confidence intervals.
        
        Args:
            data: DataFrame from LoggingPolicyGenerator
            policy_threshold: Threshold for converting scores to binary predictions
            
        Returns:
            Dictionary with metrics and their statistics (mean, p025, p975)
        """
        # Filter to only observed transactions (those that were allowed)
        observed_data = data[data['action'] == 'allow'].copy()
        
        if len(observed_data) == 0:
            raise ValueError("No observed transactions available for estimation")
        
        # Calculate importance sampling weights
        observed_data['weight'] = 1.0 / observed_data['propensity_score']
        
        # Get relevant columns
        y_true = observed_data['is_fraud'].values
        scores = observed_data['model_scores'].values
        weights = observed_data['weight'].values
        
        # Convert scores to binary predictions based on threshold
        y_pred = (scores > policy_threshold).astype(int)
        
        # Bootstrap estimation
        n_samples = len(observed_data)
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
    
    def estimate_policy_metrics(
        self,
        data: pd.DataFrame,
        policy_threshold: float = 0.5
    ) -> Dict[str, Dict[str, float]]:
        """
        Estimate metrics for a specific policy threshold.
        
        This method evaluates how well the original policy (based on cutoff)
        would perform using counterfactual estimation.
        
        Args:
            data: DataFrame from LoggingPolicyGenerator
            policy_threshold: Score threshold for the policy to evaluate
            
        Returns:
            Dictionary with policy metrics and confidence intervals
        """
        # Filter to only observed transactions
        observed_data = data[data['action'] == 'allow'].copy()
        
        if len(observed_data) == 0:
            raise ValueError("No observed transactions available for estimation")
        
        # Calculate weights
        observed_data['weight'] = 1.0 / observed_data['propensity_score']
        
        # Create binary predictions based on policy threshold
        y_true = observed_data['is_fraud'].values
        y_pred = (observed_data['model_scores'] > policy_threshold).astype(int)
        weights = observed_data['weight'].values
        
        # Bootstrap estimation
        n_samples = len(observed_data)
        metric_results = {metric: [] for metric in self.metrics}
        
        for _ in range(self.n_bootstrap):
            # Poisson bootstrap
            poisson_weights = np.random.poisson(1, n_samples)
            bootstrap_weights = weights * poisson_weights
            
            if np.sum(bootstrap_weights) == 0:
                continue
            
            # Calculate metrics
            for metric_name in self.metrics:
                metric_func = self.available_metrics[metric_name]
                metric_value = metric_func(y_true, y_pred, bootstrap_weights)
                metric_results[metric_name].append(metric_value)
        
        # Calculate statistics
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
    
    def get_params(self) -> dict:
        """Return the current parameters."""
        return {
            'n_bootstrap': self.n_bootstrap,
            'metrics': self.metrics,
            'random_state': self.random_state
        } 