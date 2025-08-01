"""Optimized Counterfactual Values Estimator for Fraud Model Evaluation.

This module provides optimized functionality to estimate counterfactual values
for fraud detection models using off-policy evaluation techniques.

Performance optimizations:
- Pre-convert pandas columns to numpy arrays for faster indexing
- Use vectorized numpy operations instead of pandas DataFrame operations
- Use Poisson bootstrap for faster estimation (similar to original implementation)
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.metrics import average_precision_score

from ..config import CounterfactualEstimatorConfig
from ..protocols import CounterfactualEstimatorProtocol


class CounterfactualEstimator(CounterfactualEstimatorProtocol):
    """
    Optimized estimator for counterfactual values using importance sampling.
    
    This class allows evaluation of how a new policy would perform by using
    historical data from a logging policy through off-policy evaluation methods.
    
    Optimizations include:
    - Pre-conversion of pandas columns to numpy arrays for faster bootstrap operations
    - Vectorized numpy operations instead of pandas DataFrame operations  
    - Poisson bootstrap for faster estimation
    - Elimination of repeated indexing overhead
    """
    
    def __init__(self, config: CounterfactualEstimatorConfig, data: pd.DataFrame):
        """
        Initialize CounterfactualEstimator.
        
        Args:
            config: Configuration object containing estimation parameters
            data: DataFrame with required columns for counterfactual evaluation
        """
        self.config = config
        
        # Validate data has required columns
        required_columns = ['is_fraud', 'model_scores', 'propensity_score', 'model_action', 'policy_action']
        self._validate_data(data, required_columns)
        
        # Store observed data (transactions that were allowed by the policy)
        self.observed_data = data[data['policy_action'] == 'allow'].copy()
        
        if len(self.observed_data) == 0:
            raise ValueError(
                "No transactions with policy_action == 'allow' found. "
                "Counterfactual evaluation is impossible without observed (allowed) transactions."
            )
        
        # Pre-convert to numpy arrays for faster bootstrap operations
        self._precompute_arrays()
        
        # Available metrics
        self.action_metrics = {
            'precision': self._weighted_precision,
            'recall': self._weighted_recall,
            # Stop using f1 for now, it isn't that informative for the simulations
            # 'f1': self._weighted_f1,
            'fraud_rate': self._weighted_fraud_rate,
        }
        self.proba_metrics = {
            'average_precision': self._weighted_average_precision
        }
        
        # Set random seed for reproducibility
        if config.random_state is not None:
            np.random.seed(config.random_state)
    
    def _precompute_arrays(self):
        """Convert DataFrame columns to numpy arrays for faster bootstrap operations."""
        self.is_fraud_array = self.observed_data['is_fraud'].values
        self.propensity_scores_array = self.observed_data['propensity_score'].values
        self.model_action_array = self.observed_data['model_action'].values
        self.model_scores_array = self.observed_data['model_scores'].values
        self.n_observed = len(self.observed_data)
        
        # Pre-compute importance sampling weights
        self.importance_weights = 1.0 / self.propensity_scores_array
    
    def estimate_policy_metrics(self, is_vectorized: bool = True) -> Dict[str, any]:
        """
        Estimate counterfactual policy metrics for the original model policy.
        
        Args:
            is_vectorized: If True, uses the vectorized implementation for better performance
                         with large datasets. If False, uses the loop-based implementation.
                         Default False for backward compatibility.
        
        Returns:
            Dictionary with estimated metrics and confidence intervals
        """
        # Use model_action as the counterfactual policy (converted to numpy)
        model_policy = (self.model_action_array == 'block').astype(int)
        model_policy_proba = self.model_scores_array
        if is_vectorized:
            return self.estimate_ope_metrics_vectorized(model_policy, model_policy_proba)
        else:
            return self.estimate_ope_metrics(model_policy, model_policy_proba)
    
    # TODO: better design this method to accept dataframes with the original structure from data
    def estimate_ope_metrics(self, new_actions: np.ndarray, new_actions_proba: np.ndarray) -> Dict[str, any]:
        """
        Off-policy evaluation metrics estimation using Poisson bootstrap (loop-based implementation).
        
        This is the standard, memory-efficient implementation that works well for most scenarios.
        For large datasets (n_observed > 1,000) with sufficient memory, consider using
        estimate_ope_metrics_vectorized() for better performance (1.2-1.7x speedup).
        
        Performance characteristics:
        - Memory usage: Low, constant footprint
        - Best for: n_observed < 1,000 or memory-constrained environments
        - Speed: Good for small/medium problems, slower for large problems
        
        Args:
            new_actions: Array of new policy actions (0=allow, 1=block)
            new_actions_proba: Array of new policy action probabilities/scores
            
        Returns:
            Dictionary with estimated metrics and confidence intervals
            
        See Also:
            estimate_ope_metrics_vectorized: Faster alternative for large datasets
        """
        return self._estimate_ope_metrics_loop_based(new_actions, new_actions_proba)
    
    def estimate_ope_metrics_vectorized(self, new_actions: np.ndarray, new_actions_proba: np.ndarray) -> Dict[str, any]:
        """
        Vectorized off-policy evaluation metrics estimation using matrix operations.
        
        This implementation generates all Poisson weights at once and uses matrix operations
        to compute all bootstrap samples simultaneously, potentially leveraging BLAS threading.
        
        Performance characteristics:
        - Best for: n_observed > 1,000 and sufficient memory available
        - Memory usage: ~8 bytes × n_observed × n_bootstrap (can be substantial)
        - Speedup: 1.2-1.7x faster than loop-based for medium/large problems
        
        Args:
            new_actions: Array of new policy actions (0=allow, 1=block)
            new_actions_proba: Array of new policy action probabilities/scores
            
        Returns:
            Dictionary with estimated metrics and confidence intervals
        """
        # Convert to numpy arrays if they're not already
        if not isinstance(new_actions, np.ndarray):
            new_actions = np.array(new_actions)
        if not isinstance(new_actions_proba, np.ndarray):
            new_actions_proba = np.array(new_actions_proba)
            
        if len(new_actions) != self.n_observed:
            raise ValueError(
                f"Length of new_actions ({len(new_actions)}) must match "
                f"number of observed transactions ({self.n_observed})"
            )
        
        if len(new_actions_proba) != self.n_observed:
            raise ValueError(
                f"Length of new_actions_proba ({len(new_actions_proba)}) must match "
                f"number of observed transactions ({self.n_observed})"
            )
        
        # Performance and memory warnings
        memory_elements = self.n_observed * self.config.n_bootstrap
        estimated_memory_mb = memory_elements * 8 / (1024 * 1024)  # 8 bytes per float64
        estimated_memory_gb = estimated_memory_mb / 1024  # Convert to GB
        
        if self.n_observed < 1000:
            warnings.warn(
                f"Using vectorized method with small dataset (n_observed={self.n_observed}). "
                f"The loop-based method estimate_ope_metrics() may be more efficient for small datasets. "
                f"Consider using the standard method for n_observed < 1,000.",
                UserWarning,
                stacklevel=2
            )
        
        if estimated_memory_gb > 8.0:  # Warn if estimated memory > 8GB (approaching 10GB limit)
            warnings.warn(
                f"Vectorized method will allocate ~{estimated_memory_gb:.2f}GB of memory "
                f"({self.n_observed:,} observations × {self.config.n_bootstrap:,} bootstrap samples). "
                f"This approaches the 10GB memory limit. Consider reducing n_bootstrap or using "
                f"the loop-based method estimate_ope_metrics() if memory is constrained.",
                UserWarning,
                stacklevel=2
            )
        elif estimated_memory_gb > 5.0:  # Info warning for moderate memory usage
            warnings.warn(
                f"Vectorized method will allocate ~{estimated_memory_gb:.2f}GB of memory "
                f"({self.n_observed:,} observations × {self.config.n_bootstrap:,} bootstrap samples). "
                f"Monitor memory usage if running on constrained systems.",
                UserWarning,
                stacklevel=2
            )
        
        if memory_elements > 1_250_000_000:  # ~10GB threshold (1.25B elements * 8 bytes = 10GB)
            warnings.warn(
                f"Very large matrix operation requested ({memory_elements:,} elements, "
                f"~{estimated_memory_gb:.2f}GB). This may cause memory issues or system instability. "
                f"Consider using the loop-based method estimate_ope_metrics() or reducing the problem size.",
                UserWarning,
                stacklevel=2
            )
        
        # Generate all Poisson weights at once: shape (n_observed, n_bootstrap)
        poisson_matrix = np.random.poisson(1, (self.n_observed, self.config.n_bootstrap))
        
        # Compute all bootstrap weights at once using broadcasting
        # Shape: (n_observed, n_bootstrap)
        bootstrap_weights_matrix = self.importance_weights[:, np.newaxis] * poisson_matrix
        
        # Results storage
        bootstrap_results = {}
        
        # Process action-based metrics (need new_actions)
        for metric_name, metric_func in self.action_metrics.items():
            # Compute metrics for all bootstrap samples at once
            bootstrap_values = self._compute_metric_vectorized(
                metric_func, new_actions, bootstrap_weights_matrix
            )
            
            # Calculate statistics (compatible format with original)
            bootstrap_results[metric_name] = {
                'mean': float(np.mean(bootstrap_values)),
                'p025': float(np.percentile(bootstrap_values, 2.5)),
                'p975': float(np.percentile(bootstrap_values, 97.5)),
                'n_bootstrap': len(bootstrap_values),
            }
        
        # Process probability-based metrics (need new_actions_proba)
        for metric_name, metric_func in self.proba_metrics.items():
            # Compute metrics for all bootstrap samples at once
            bootstrap_values = self._compute_metric_vectorized(
                metric_func, new_actions_proba, bootstrap_weights_matrix
            )
            
            # Calculate statistics (compatible format with original)
            bootstrap_results[metric_name] = {
                'mean': float(np.mean(bootstrap_values)),
                'p025': float(np.percentile(bootstrap_values, 2.5)),
                'p975': float(np.percentile(bootstrap_values, 97.5)),
                'n_bootstrap': len(bootstrap_values),
            }
        
        return bootstrap_results
    
    def _compute_metric_vectorized(self, metric_func, actions_or_proba: np.ndarray, 
                                 bootstrap_weights_matrix: np.ndarray) -> np.ndarray:
        """
        Compute a metric for all bootstrap samples using vectorized operations.
        
        Args:
            metric_func: The metric function to compute
            actions_or_proba: Either new_actions or new_actions_proba array
            bootstrap_weights_matrix: Matrix of bootstrap weights (n_observed, n_bootstrap)
            
        Returns:
            Array of metric values for each bootstrap sample
        """
        bootstrap_values = []
        
        # Check if we can vectorize further (currently still uses a loop over bootstrap samples)
        for i in range(self.config.n_bootstrap):
            bootstrap_weights = bootstrap_weights_matrix[:, i]
            
            # Skip if all weights are zero
            if np.sum(bootstrap_weights) == 0:
                continue
                
            # Calculate metric for this bootstrap sample
            bootstrap_value = metric_func(actions_or_proba, bootstrap_weights)
            bootstrap_values.append(bootstrap_value)
        
        return np.array(bootstrap_values)
    
    def _estimate_ope_metrics_loop_based(self, new_actions: np.ndarray, new_actions_proba: np.ndarray) -> Dict[str, any]:
        """
        Original loop-based off-policy evaluation metrics estimation using Poisson bootstrap.
        
        Performance characteristics:
        - Best for: n_observed < 1,000 or memory-constrained environments
        - Memory usage: Low, constant memory footprint
        - Slower for large problems where vectorized method could provide 1.2-1.7x speedup
        
        Args:
            new_actions: Array of new policy actions (0=allow, 1=block)
            new_actions_proba: Array of new policy action probabilities/scores
            
        Returns:
            Dictionary with estimated metrics and confidence intervals
        """
        # Convert to numpy arrays if they're not already
        if not isinstance(new_actions, np.ndarray):
            new_actions = np.array(new_actions)
        if not isinstance(new_actions_proba, np.ndarray):
            new_actions_proba = np.array(new_actions_proba)
            
        if len(new_actions) != self.n_observed:
            raise ValueError(
                f"Length of new_actions ({len(new_actions)}) must match "
                f"number of observed transactions ({self.n_observed})"
            )
        
        if len(new_actions_proba) != self.n_observed:
            raise ValueError(
                f"Length of new_actions_proba ({len(new_actions_proba)}) must match "
                f"number of observed transactions ({self.n_observed})"
            )
        
        # Performance warnings for suboptimal usage
        if self.n_observed > 1000 and self.config.n_bootstrap > 100:
            estimated_memory_mb = self.n_observed * self.config.n_bootstrap * 8 / (1024 * 1024)
            estimated_memory_gb = estimated_memory_mb / 1024
            
            if estimated_memory_gb < 8.0:  # Only suggest vectorized if memory is reasonable (< 8GB)
                expected_speedup = min(1.7, 1.0 + (self.n_observed / 10000))  # Rough speedup estimate
                
                if estimated_memory_gb < 1.0:
                    memory_str = f"~{estimated_memory_mb:.1f}MB"
                else:
                    memory_str = f"~{estimated_memory_gb:.2f}GB"
                
                warnings.warn(
                    f"Using loop-based method with large dataset (n_observed={self.n_observed:,}, "
                    f"n_bootstrap={self.config.n_bootstrap:,}). The vectorized method "
                    f"estimate_ope_metrics_vectorized() could provide ~{expected_speedup:.1f}x speedup "
                    f"with {memory_str} memory usage. Consider using the vectorized method "
                    f"for better performance if memory allows.",
                    UserWarning,
                    stacklevel=2
                )
            else:
                warnings.warn(
                    f"Large problem size detected (n_observed={self.n_observed:,}, "
                    f"n_bootstrap={self.config.n_bootstrap:,}). The vectorized method could provide "
                    f"better performance but would require ~{estimated_memory_gb:.2f}GB memory "
                    f"(approaching 10GB limit). Consider using estimate_ope_metrics_vectorized() only "
                    f"if sufficient memory is available, or reduce n_bootstrap for better performance.",
                    UserWarning,
                    stacklevel=2
                )
        
        # Poisson bootstrap estimation (faster than regular bootstrap)
        bootstrap_results = {}
        
        # Process action-based metrics (need new_actions)
        for metric_name, metric_func in self.action_metrics.items():
            bootstrap_values = []
            
            for _ in range(self.config.n_bootstrap):
                # Poisson bootstrap: sample with Poisson weights
                poisson_weights = np.random.poisson(1, self.n_observed)
                
                # Apply Poisson weights to importance sampling weights
                bootstrap_weights = self.importance_weights * poisson_weights
                
                # Skip if all weights are zero
                if np.sum(bootstrap_weights) == 0:
                    continue
                
                # Calculate metric for this bootstrap sample
                bootstrap_value = metric_func(new_actions, bootstrap_weights)
                bootstrap_values.append(bootstrap_value)
            
            # Calculate statistics (compatible format with original)
            bootstrap_values = np.array(bootstrap_values)
            bootstrap_results[metric_name] = {
                'mean': float(np.mean(bootstrap_values)),
                'p025': float(np.percentile(bootstrap_values, 2.5)),
                'p975': float(np.percentile(bootstrap_values, 97.5)),
                'n_bootstrap': len(bootstrap_values),
            }
        
        # Process probability-based metrics (need new_actions_proba)
        for metric_name, metric_func in self.proba_metrics.items():
            bootstrap_values = []
            
            for _ in range(self.config.n_bootstrap):
                # Poisson bootstrap: sample with Poisson weights
                poisson_weights = np.random.poisson(1, self.n_observed)
                
                # Apply Poisson weights to importance sampling weights
                bootstrap_weights = self.importance_weights * poisson_weights
                
                # Skip if all weights are zero
                if np.sum(bootstrap_weights) == 0:
                    continue
                
                # Calculate metric for this bootstrap sample
                bootstrap_value = metric_func(new_actions_proba, bootstrap_weights)
                bootstrap_values.append(bootstrap_value)
            
            # Calculate statistics (compatible format with original)
            bootstrap_values = np.array(bootstrap_values)
            bootstrap_results[metric_name] = {
                'mean': float(np.mean(bootstrap_values)),
                'p025': float(np.percentile(bootstrap_values, 2.5)),
                'p975': float(np.percentile(bootstrap_values, 97.5)),
                'n_bootstrap': len(bootstrap_values),
            }
        
        return bootstrap_results
    
    def get_config(self) -> CounterfactualEstimatorConfig:
        """Get the current configuration."""
        return self.config
    
    def _validate_data(self, data: pd.DataFrame, required_columns: List[str]) -> None:
        """Validate that data contains required columns."""
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Data is missing required columns: {missing_columns}")
    
    def _weighted_precision(self, new_actions: np.ndarray, weights: np.ndarray) -> float:
        """
        Calculate weighted precision using Poisson bootstrap weights.
        
        Precision = True Positives / (True Positives + False Positives)
        where True Positive = correctly blocked fraud, False Positive = incorrectly blocked legitimate
        
        Args:
            new_actions: Array of new policy actions (0=allow, 1=block)
            weights: Bootstrap weights (importance_weights * poisson_weights)
        """
        # Transactions that would be blocked by the new policy (action = 1)
        blocked_mask = (new_actions == 1)
        
        if not blocked_mask.any():
            return 0.0  # No blocked transactions, no precision calculation possible
        
        # Among blocked transactions, how many are actually fraud? (True Positives)
        blocked_is_fraud = self.is_fraud_array[blocked_mask]
        blocked_weights = weights[blocked_mask]
        
        if np.sum(blocked_weights) == 0:
            return 0.0
        
        # True positives: blocked transactions that are actually fraud
        true_positives = np.sum(blocked_is_fraud * blocked_weights)
        # Total blocked (predicted positives)
        total_blocked = np.sum(blocked_weights)
        
        return true_positives / total_blocked
    
    def _weighted_recall(self, new_actions: np.ndarray, weights: np.ndarray) -> float:
        """
        Calculate weighted recall using Poisson bootstrap weights.
        
        Recall = True Positives / (True Positives + False Negatives)
        where True Positive = correctly blocked fraud, False Negative = incorrectly allowed fraud
        
        Args:
            new_actions: Array of new policy actions (0=allow, 1=block)
            weights: Bootstrap weights (importance_weights * poisson_weights)
        """
        # All actual fraud cases
        fraud_mask = (self.is_fraud_array == 1)
        
        if not fraud_mask.any():
            return 1.0  # No fraud cases, perfect recall by definition
        
        # Among fraud cases, how many would be blocked by the new policy? (True Positives)
        fraud_actions = new_actions[fraud_mask]
        fraud_weights = weights[fraud_mask]
        
        if np.sum(fraud_weights) == 0:
            return 0.0
        
        # Fraud cases that would be blocked (action = 1) - True Positives
        blocked_fraud_mask = (fraud_actions == 1)
        true_positives = np.sum(fraud_weights[blocked_fraud_mask])
        # Total fraud cases (actual positives)
        total_fraud = np.sum(fraud_weights)
        
        return true_positives / total_fraud
    
    def _weighted_f1(self, new_actions: np.ndarray, weights: np.ndarray) -> float:
        """
        Calculate weighted F1-score using Poisson bootstrap weights.
        
        Args:
            new_actions: Array of new policy actions (0=allow, 1=block)
            weights: Bootstrap weights (importance_weights * poisson_weights)
        """
        precision = self._weighted_precision(new_actions, weights)
        recall = self._weighted_recall(new_actions, weights)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _weighted_fraud_rate(self, new_actions: np.ndarray, weights: np.ndarray) -> float:
        """
        Calculate weighted fraud rate using Poisson bootstrap weights.
        
        This calculates the fraud rate among transactions that would be ALLOWED by the new policy.
        
        Args:
            new_actions: Array of new policy actions (0=allow, 1=block)
            weights: Bootstrap weights (importance_weights * poisson_weights)
        """
        # Transactions that would be allowed by the new policy (action = 0)
        allowed_mask = (new_actions == 0)
        
        if not allowed_mask.any():
            return 0.0  # No allowed transactions
        
        # Among allowed transactions, what's the fraud rate?
        allowed_is_fraud = self.is_fraud_array[allowed_mask]
        allowed_weights = weights[allowed_mask]
        
        if np.sum(allowed_weights) == 0:
            return 0.0
        
        weighted_fraud = np.sum(allowed_is_fraud * allowed_weights)
        weighted_total = np.sum(allowed_weights)
        
        return weighted_fraud / weighted_total
    
    def _weighted_average_precision(self, new_actions_proba: np.ndarray, weights: np.ndarray) -> float:
        """
        Calculate weighted average precision using sklearn's average_precision_score.
        
        Average precision summarizes a precision-recall curve as the weighted mean of 
        precisions achieved at each threshold, with the increase in recall from the 
        previous threshold used as the weight.
        
        Args:
            new_actions_proba: Array of new policy action probabilities/scores for AP calculation
            weights: Bootstrap weights (importance_weights * poisson_weights)
            
        Returns:
            Average precision score
        """
        # Check if we have any positive weights
        if np.sum(weights) == 0:
            return 0.0
        
        # Check if we have any fraud cases
        if not self.is_fraud_array.any():
            return 0.0
        
        # Calculate average precision using model scores and true fraud labels
        # sklearn handles weight scaling internally, so no normalization needed
        ap_score = average_precision_score(
            y_true=self.is_fraud_array,
            y_score=new_actions_proba,
            sample_weight=weights
        )
        return ap_score 