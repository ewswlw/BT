"""
Core modules for CDX backtesting framework.
"""

from .data_loader import DataLoader, CSVDataProvider, DataValidator
from .feature_engineering import (
    FeatureEngineer,
    TechnicalFeatureEngineer,
    AdvancedFeatureEngineer,
    FeatureSelector
)
from .backtest_engine import BacktestEngine, PortfolioManager, BacktestResult
from .validation import (
    WalkForwardValidator,
    BiasChecker,
    ProbabilisticSharpeRatio,
    DeflatedSharpeRatio,
    MonteCarloValidator
)
from .metrics import MetricsCalculator

__all__ = [
    'DataLoader',
    'CSVDataProvider',
    'DataValidator',
    'FeatureEngineer',
    'TechnicalFeatureEngineer',
    'AdvancedFeatureEngineer',
    'FeatureSelector',
    'BacktestEngine',
    'PortfolioManager',
    'BacktestResult',
    'WalkForwardValidator',
    'BiasChecker',
    'ProbabilisticSharpeRatio',
    'DeflatedSharpeRatio',
    'MonteCarloValidator',
    'MetricsCalculator',
]

