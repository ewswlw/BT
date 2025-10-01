"""
Core backtesting framework components.
"""

from .config import DataConfig, PortfolioConfig, ReportingConfig, BaseConfig, create_config_from_dict
from .data_loader import DataLoader, CSVDataProvider
from .feature_engineering import TechnicalFeatureEngineer, CrossAssetFeatureEngineer, MultiAssetFeatureEngineer
from .portfolio import PortfolioEngine, BacktestResult
from .metrics import MetricsCalculator
from .reporting import ReportGenerator

__all__ = [
    "DataConfig",
    "PortfolioConfig", 
    "ReportingConfig",
    "BaseConfig",
    "create_config_from_dict",
    "DataLoader",
    "CSVDataProvider",
    "TechnicalFeatureEngineer",
    "CrossAssetFeatureEngineer",
    "MultiAssetFeatureEngineer", 
    "PortfolioEngine",
    "BacktestResult",
    "MetricsCalculator",
    "ReportGenerator",
] 