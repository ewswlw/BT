"""
Configuration management for CAD OAS prediction model.
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for the CAD OAS prediction model."""
    
    # Data paths
    data_path: str = "../data_pipelines/data_processed/no_er_daily.csv"
    output_dir: str = "ml spreads/outputs"
    
    # Target and features
    target_column: str = "cad_oas"
    feature_columns: List[str] = None
    
    # Lag constraints (all features must be lagged 5-90 days)
    min_lag: int = 5
    max_lag: int = 90
    lag_periods: List[int] = None
    
    # Walk-forward validation
    window_size: int = 756  # 3 years training
    test_size: int = 63     # 1 quarter testing
    step_size: int = 21     # Retrain monthly
    
    # Feature engineering
    baseline_lags: List[int] = None
    momentum_periods: List[int] = None
    volatility_periods: List[int] = None
    ma_periods: List[int] = None
    
    # Model parameters
    random_state: int = 42
    test_size_ratio: float = 0.2
    
    # Performance targets
    target_r2: float = 0.85
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.feature_columns is None:
            # All columns except Date and target
            self.feature_columns = [
                'us_hy_oas', 'us_ig_oas', 'tsx', 'vix', 'us_3m_10y',
                'us_growth_surprises', 'us_inflation_surprises', 'us_lei_yoy',
                'us_hard_data_surprises', 'us_equity_revisions', 'us_economic_regime',
                'spx_1bf_eps', 'spx_1bf_sales', 'tsx_1bf_eps', 'tsx_1bf_sales'
            ]
        
        if self.lag_periods is None:
            self.lag_periods = [5, 10, 20, 60, 90]
        
        if self.baseline_lags is None:
            self.baseline_lags = [5, 10, 20, 60, 90]
        
        if self.momentum_periods is None:
            self.momentum_periods = [1, 3, 5, 10, 20, 40, 60]
        
        if self.volatility_periods is None:
            self.volatility_periods = [5, 10, 20, 60]
        
        if self.ma_periods is None:
            self.ma_periods = [10, 20, 60]
        
        # Create output directories
        os.makedirs(f"{self.output_dir}/models", exist_ok=True)
        os.makedirs(f"{self.output_dir}/features", exist_ok=True)
        os.makedirs(f"{self.output_dir}/predictions", exist_ok=True)
        os.makedirs(f"{self.output_dir}/reports", exist_ok=True)

# Global configuration instance
config = ModelConfig()

# Model hyperparameters for different algorithms
MODEL_PARAMS = {
    'linear_regression': {
        'fit_intercept': True
    },
    'ridge': {
        'alpha': 1.0,
        'fit_intercept': True
    },
    'lasso': {
        'alpha': 0.01,
        'fit_intercept': True,
        'max_iter': 1000
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    },
    'lightgbm': {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': 7,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'force_col_wise': True,
        'seed': 42
    },
    'xgboost': {
        'objective': 'reg:squarederror',
        'learning_rate': 0.05,
        'max_depth': 7,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42
    }
}

# Feature engineering configurations
FEATURE_CONFIG = {
    'technical_indicators': {
        'rsi_periods': [14, 30],
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bollinger_window': 20,
        'bollinger_std': 2,
        'stochastic_period': 14
    },
    'statistical_features': {
        'zscore_windows': [20, 60, 252],
        'percentile_windows': [20, 60, 252],
        'skew_kurt_windows': [20, 60, 252]
    },
    'cross_asset_features': {
        'spread_pairs': [
            ('cad_oas', 'us_ig_oas'),
            ('cad_oas', 'us_hy_oas'),
            ('us_ig_oas', 'us_hy_oas')
        ],
        'vix_regime_percentiles': [25, 50, 75]
    }
}
