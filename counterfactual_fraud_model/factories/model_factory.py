"""Model factory for creating machine learning models.

Centralized factory for creating and configuring ML models to eliminate
code duplication across components.
"""

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ..config import ModelConfig, ModelType
from ..protocols import ModelFactoryProtocol

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class ModelFactory(ModelFactoryProtocol):
    """Factory for creating machine learning models based on configuration."""
    
    def create_model(self, config: ModelConfig) -> BaseEstimator:
        """Create a model instance based on configuration.
        
        Args:
            config: Model configuration specifying type and parameters
            
        Returns:
            Configured scikit-learn compatible model instance
            
        Raises:
            ValueError: If model type is unsupported or LightGBM is not available
        """
        if config.model_type == ModelType.LIGHTGBM:
            return self._create_lightgbm_model(config)
        elif config.model_type == ModelType.RANDOM_FOREST:
            return self._create_random_forest_model(config)
        elif config.model_type == ModelType.LOGISTIC:
            return self._create_logistic_model(config)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
    
    def _create_lightgbm_model(self, config: ModelConfig) -> BaseEstimator:
        """Create a LightGBM classifier."""
        if not LIGHTGBM_AVAILABLE:
            raise ValueError("LightGBM is not available. Install with: pip install lightgbm")
        
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
            'random_state': config.random_state
        }
        
        # Merge with user-provided parameters
        params = {**default_params, **config.model_params}
        
        return lgb.LGBMClassifier(**params)
    
    def _create_random_forest_model(self, config: ModelConfig) -> BaseEstimator:
        """Create a Random Forest classifier."""
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': config.random_state
        }
        
        # Merge with user-provided parameters
        params = {**default_params, **config.model_params}
        
        return RandomForestClassifier(**params)
    
    def _create_logistic_model(self, config: ModelConfig) -> BaseEstimator:
        """Create a Logistic Regression classifier."""
        default_params = {
            'random_state': config.random_state,
            'max_iter': 1000
        }
        
        # Merge with user-provided parameters
        params = {**default_params, **config.model_params}
        
        return LogisticRegression(**params)


# Default factory instance
default_model_factory = ModelFactory() 