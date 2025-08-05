"""Refactored Logging Policy Generator for Counterfactual Fraud Model Simulation.

This generator simulates fraud prevention system policy with exploration
for counterfactual evaluation. Configured via Pydantic models for type safety.
"""

import numpy as np
import pandas as pd

from ..config import LoggingPolicyConfig
from ..protocols import LoggingPolicyGeneratorProtocol


class LoggingPolicyGenerator(LoggingPolicyGeneratorProtocol):
    """
    Simulates fraud prevention system policy with exploration for counterfactual evaluation.
    
    Generates propensity scores and actions based on model scores, implementing
    the exploration-exploitation tradeoff for fraud detection.
    """
    
    def __init__(self, config: LoggingPolicyConfig):
        """
        Initialize LoggingPolicyGenerator with configuration.
        
        Args:
            config: Configuration object containing policy parameters
        """
        self.config = config
        
        # Set random seed if provided
        if config.random_state is not None:
            np.random.seed(config.random_state)
    
    def generate_policy(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate logging policy with propensity scores and actions.
        
        Processes transactions row-wise: transactions below cutoff are always allowed,
        transactions above cutoff are sampled based on exploration rate.
        Only calculates propensity scores for transactions that are actually allowed.
        
        Args:
            data: DataFrame from DataGenerator with model scores
            
        Returns:
            DataFrame with added columns: propensity_score, model_action, policy_action
        """
        # Create a copy to avoid modifying the original data
        result_data = data.copy()
        
        # Use model_scores as the main score for policy decisions
        scores = result_data['model_scores'].values
        n_transactions = len(scores)
        
        # Initialize arrays
        model_actions = np.empty(n_transactions, dtype='<U5')  # 'allow' or 'block'
        policy_actions = np.empty(n_transactions, dtype='<U5')  # 'allow' or 'block'
        propensity_scores = np.zeros(n_transactions)
        
        # Process each transaction
        for i in range(n_transactions):
            score = scores[i]
            
            # Model action is purely based on cutoff
            if score <= self.config.cutoff:
                model_actions[i] = 'allow'
            else:
                model_actions[i] = 'block'
            
            # Policy action includes exploration
            if score <= self.config.cutoff:
                # Transactions below cutoff are always allowed
                policy_actions[i] = 'allow'
                propensity_scores[i] = 1.0
            else:
                # Transactions above cutoff: sample based on exploration rate
                if np.random.random() < self.config.exploration_rate:
                    # Selected for exploration - allow and set propensity score
                    policy_actions[i] = 'allow'
                    propensity_scores[i] = self.config.exploration_rate
                else:
                    # Blocked - propensity score irrelevant (set to 0)
                    policy_actions[i] = 'block'
                    propensity_scores[i] = 0.0
        
        # Add columns to the result DataFrame
        result_data['propensity_score'] = propensity_scores
        result_data['model_action'] = model_actions
        result_data['policy_action'] = policy_actions
        
        return result_data
    
    def get_config(self) -> LoggingPolicyConfig:
        """Get the current configuration."""
        return self.config 