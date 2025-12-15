"""
Core backtesting framework components.
"""

from .config import DataConfig, PortfolioConfig, ReportingConfig, BaseConfig, create_config_from_dict
from .data_loader import DataLoader, CSVDataProvider, MultiIndexCSVDataProvider
from .feature_engineering import TechnicalFeatureEngineer, CrossAssetFeatureEngineer, MultiAssetFeatureEngineer
from .portfolio import PortfolioEngine, BacktestResult
from .metrics import MetricsCalculator
from .reporting import ReportGenerator

# Import validation module (optional, for backward compatibility)
try:
    from .validation import ValidationFramework, ValidationConfig, ValidationReportGenerator
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    ValidationFramework = None
    ValidationConfig = None
    ValidationReportGenerator = None

__all__ = [
    "DataConfig",
    "PortfolioConfig", 
    "ReportingConfig",
    "BaseConfig",
    "create_config_from_dict",
    "DataLoader",
    "CSVDataProvider",
    "MultiIndexCSVDataProvider",
    "TechnicalFeatureEngineer",
    "CrossAssetFeatureEngineer",
    "MultiAssetFeatureEngineer", 
    "PortfolioEngine",
    "BacktestResult",
    "MetricsCalculator",
    "ReportGenerator",
]

if VALIDATION_AVAILABLE:
    __all__.extend([
        "ValidationFramework",
        "ValidationConfig",
        "ValidationReportGenerator",
    ]) 