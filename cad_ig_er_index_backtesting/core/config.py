"""
Configuration management for the backtesting framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import yaml
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    file_path: str
    date_column: str = "Date"
    resample_frequency: Optional[str] = None  # Allow null for raw data
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    assets: List[str] = field(default_factory=list)
    benchmark_asset: str = "cad_ig_er_index"
    trading_asset: str = "cad_ig_er_index"  # Added to match reference implementations
    required_columns: List[str] = field(default_factory=list)
    
    # Multi-index CSV support
    use_multi_index: bool = False  # Enable multi-index CSV loading
    multi_index_header_rows: int = 2  # Number of header rows (rows 1-2)
    date_header_row: int = 2  # Row index for Date column (0-indexed, so row 3 = index 2)
    
    # Asset class filtering
    asset_classes_include: Optional[List[str]] = None  # Asset classes to include
    asset_classes_exclude: Optional[List[str]] = None  # Asset classes to exclude
    
    # Column filtering (scoped by asset class)
    columns_include: Optional[Dict[str, List[str]]] = None  # {"asset_class": ["col1", "col2"]}
    columns_exclude: Optional[Dict[str, List[str]]] = None  # {"asset_class": ["col1"]}
    
    # Missing data handling
    missing_data_strategy: str = "forward_fill"  # forward_fill, backward_fill, interpolate, drop, fill_value
    missing_data_fill_value: float = 0.0  # Value for fill_value strategy
    
    # Outlier detection
    outlier_detection_enabled: bool = False
    outlier_method: str = "iqr"  # iqr, zscore
    outlier_threshold: float = 3.0  # For zscore method
    outlier_action: str = "flag"  # flag, remove, clip
    
    # Gap detection
    gap_detection_enabled: bool = False
    max_gap_days: int = 7
    gap_action: str = "flag"  # flag, fill, drop
    
    # Data validation
    data_validation_enabled: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    # Column transformations
    column_rename_map: Optional[Dict[str, str]] = None  # {"old_name": "new_name"}
    force_numeric: bool = True
    
    # Multi-index flattening
    flatten_columns: bool = True  # Flatten multi-index to single level
    # Note: Always use 2nd level (column names) as final column names


@dataclass
class PortfolioConfig:
    """Configuration for portfolio backtesting."""
    initial_capital: float = 100000
    frequency: str = "W"
    fees: float = 0.0
    slippage: float = 0.0
    position_size: Union[float, str] = 1.0
    leverage: float = 1.0
    cash_buffer: float = 0.0


@dataclass
class ReportingConfig:
    """Configuration for report generation."""
    output_dir: str = "outputs/reports"
    generate_html: bool = True
    generate_pdf: bool = False
    include_plots: bool = True
    metrics_precision: int = 4
    plot_style: str = "seaborn"
    plot_size: tuple = (12, 8)


@dataclass
class OptimizationConfig:
    """Configuration for parameter optimization."""
    method: str = "grid_search"  # grid_search, bayesian, genetic
    objective: str = "sharpe_ratio"
    n_trials: int = 100
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class BaseConfig:
    """Main configuration container."""
    data: DataConfig
    portfolio: PortfolioConfig
    reporting: ReportingConfig
    optimization: Optional[OptimizationConfig] = None
    random_seed: Optional[int] = 42
    verbose: bool = True
    log_level: str = "INFO"


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def create_config_from_dict(config_dict: Dict[str, Any]) -> BaseConfig:
    """Create BaseConfig from dictionary."""
    # Extract data config dict and ensure all optional fields have defaults
    data_dict = config_dict.get('data', {})
    
    # Create DataConfig with defaults for missing optional fields
    data_config = DataConfig(**data_dict)
    
    portfolio_config = PortfolioConfig(**config_dict.get('portfolio', {}))
    reporting_config = ReportingConfig(**config_dict.get('reporting', {}))
    
    optimization_config = None
    if 'optimization' in config_dict:
        optimization_config = OptimizationConfig(**config_dict['optimization'])
    
    return BaseConfig(
        data=data_config,
        portfolio=portfolio_config,
        reporting=reporting_config,
        optimization=optimization_config,
        random_seed=config_dict.get('random_seed', 42),
        verbose=config_dict.get('verbose', True),
        log_level=config_dict.get('log_level', 'INFO')
    )


def save_config(config: BaseConfig, output_path: str):
    """Save configuration to YAML file."""
    config_dict = {
        'data': config.data.__dict__,
        'portfolio': config.portfolio.__dict__,
        'reporting': config.reporting.__dict__,
        'random_seed': config.random_seed,
        'verbose': config.verbose,
        'log_level': config.log_level
    }
    
    if config.optimization:
        config_dict['optimization'] = config.optimization.__dict__
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2) 