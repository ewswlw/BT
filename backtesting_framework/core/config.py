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
    resample_frequency: str = "W-FRI"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    assets: List[str] = field(default_factory=list)
    benchmark_asset: str = "cad_ig_er_index"
    trading_asset: str = "cad_ig_er_index"  # Added to match reference implementations
    required_columns: List[str] = field(default_factory=list)


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
    data_config = DataConfig(**config_dict.get('data', {}))
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