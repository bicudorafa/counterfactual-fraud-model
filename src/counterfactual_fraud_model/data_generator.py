"""Data Generator for Counterfactual Fraud Model Simulation."""

import numpy as np
import pandas as pd
from scipy.stats import beta
from typing import Optional


class DataGenerator:
    """
    Generates synthetic fraud model data for counterfactual evaluation.
    
    Creates model scores from beta distribution, adds noise, and generates
    fraud labels based on the resulting scores.
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta_param: float = 5.0,
        mean: float = 0.0,
        sd: float = 0.5,
        sample_size: int = 50_000,
        random_state: Optional[int] = None
    ):
        """
        Initialize DataGenerator with distribution parameters.
        
        Args:
            alpha: Alpha parameter for beta distribution
            beta_param: Beta parameter for beta distribution  
            mean: Mean for normal error distribution
            sd: Standard deviation for normal error distribution
            sample_size: Number of samples to generate
            random_state: Random seed for reproducibility
        """
        self.alpha = alpha
        self.beta_param = beta_param
        self.mean = mean
        self.sd = sd
        self.sample_size = sample_size
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def generate_data(self) -> pd.DataFrame:
        """
        Generate synthetic fraud model data.
        
        Returns:
            DataFrame with columns: model_scores, model_error, 
                                  adjusted_scores, fraud
        """
        # Generate model scores from beta distribution
        model_scores = beta.rvs(
            a=self.alpha, 
            b=self.beta_param, 
            size=self.sample_size,
            random_state=self.random_state
        )
        
        # Generate model error from normal distribution
        model_error = np.random.normal(
            self.mean, self.sd, self.sample_size
        )
        
        # Sum model error and model score, clip to [0, 1]
        fraud_probabilities = np.clip(model_scores + model_error, 0, 1)
        
        # Generate fraud labels based on adjusted scores using binomial sampling
        # Higher adjusted scores should have higher probability of fraud
        fraud = np.random.binomial(1, fraud_probabilities, self.sample_size)
        
        # Create DataFrame
        data = pd.DataFrame({
            'model_scores': model_scores,
            'is_fraud': fraud
        })
        
        return data
    
    def get_params(self) -> dict:
        """Return the current parameters."""
        return {
            'alpha': self.alpha,
            'beta_param': self.beta_param,
            'mean': self.mean,
            'sd': self.sd,
            'sample_size': self.sample_size,
            'random_state': self.random_state
        } 