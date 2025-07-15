"""Logging Policy Generator for Counterfactual Fraud Model Simulation."""

import numpy as np
import pandas as pd
from typing import Optional


class LoggingPolicyGenerator:
    """
    Simulates fraud prevention system policy with exploration for counterfactual evaluation.
    
    Generates propensity scores and actions based on model scores, implementing
    the exploration-exploitation tradeoff for fraud detection.
    """
    
    def __init__(
        self,
        cutoff: float = 0.05,
        exploration_rate: float = 0.5,
        propensity_type: str = "uniform",
        random_state: Optional[int] = None
    ):
        """
        Initialize LoggingPolicyGenerator.
        
        Args:
            cutoff: Score threshold above which transactions would be blocked
            exploration_rate: Base exploration rate for blocked transactions  
            propensity_type: Type of propensity function ("uniform", remaining types will be implemented later)
            random_state: Random seed for reproducibility
        """
        self.cutoff = cutoff
        self.exploration_rate = exploration_rate
        self.propensity_type = propensity_type
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def generate_policy(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate logging policy with propensity scores and actions.
        
        Processes transactions row-wise: transactions below cutoff are always allowed,
        transactions above cutoff are sampled based on exploration rate.
        Only calculates propensity scores for transactions that are actually allowed.
        
        Args:
            data: DataFrame from DataGenerator with model scores (modified in place)
            
        Returns:
            DataFrame with added columns: propensity_score, action (same as input)
        """
        # Use model_scores as the main score for policy decisions
        scores = data['model_scores'].values
        n_transactions = len(scores)
        
        # Initialize arrays
        actions = np.empty(n_transactions, dtype='<U5')  # 'allow' or 'block'
        propensity_scores = np.zeros(n_transactions)
        
        # Process each transaction
        for i in range(n_transactions):
            score = scores[i]
            
            if score <= self.cutoff:
                # Transactions below cutoff are always allowed
                actions[i] = 'allow'
                propensity_scores[i] = 1.0
            else:
                # Transactions above cutoff: sample based on exploration rate
                if np.random.random() < self.exploration_rate:
                    # Selected for exploration - allow and set propensity score
                    actions[i] = 'allow'
                    propensity_scores[i] = self.exploration_rate
                else:
                    # Blocked - propensity score irrelevant (set to 0)
                    actions[i] = 'block'
                    propensity_scores[i] = 0.0
        
        # Add columns directly to the input DataFrame
        data['propensity_score'] = propensity_scores
        data['action'] = actions
        
        return data
    
    def get_params(self) -> dict:
        """Return the current parameters."""
        return {
            'cutoff': self.cutoff,
            'exploration_rate': self.exploration_rate,
            'propensity_type': self.propensity_type,
            'random_state': self.random_state
        } 