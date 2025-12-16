---
name: backtesting-patterns
description: Core architectural patterns and best practices for the CAD IG ER Index backtesting framework including configuration, data loading, portfolio management, and signal generation
---

# Backtesting Framework Patterns

## Overview
This skill provides guidance on the core patterns and architecture of our backtesting framework. Use this when working with the framework's core components.

## When to Use This Skill
Claude should use this when:
- Working with core backtesting components (config, data_loader, portfolio, metrics, reporting)
- Understanding the overall framework architecture
- Implementing new features that integrate with existing components
- Debugging issues in the backtesting pipeline

## Core Architecture

### Backtesting Flow
```
Load Config → Load Data → Engineer Features → Create Strategy →
Generate Signals → Run Portfolio Engine → Calculate Metrics →
Generate Reports → Run Validation (optional)
```

### Directory Structure
```
cad_ig_er_index_backtesting/
├── core/                    # Core framework components
│   ├── config.py           # Configuration management
│   ├── data_loader.py      # Data providers
│   ├── portfolio.py        # Portfolio backtesting engine
│   ├── feature_engineering.py
│   ├── metrics.py          # Performance metrics
│   ├── reporting.py        # HTML/PDF report generation
│   └── validation/         # López de Prado validation framework
├── strategies/             # Strategy implementations
└── main.py                 # CLI entry point
```

## Configuration Pattern

### Configuration Structure
The framework uses YAML-based configuration with dataclass models:

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml

@dataclass
class DataConfig:
    file_path: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    handle_missing: str = "forward_fill"
    outlier_detection: bool = False

@dataclass
class PortfolioConfig:
    initial_capital: float = 100.0
    fees: float = 0.001
    slippage: float = 0.001
```

### Loading Configuration
```python
# In config.py
def load_config(config_path: str) -> BaseConfig:
    """Load and validate configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return BaseConfig(**config_dict)
```

**Best Practices:**
- Always validate configuration parameters
- Use dataclasses for type safety
- Provide sensible defaults
- Document all configuration options in config.yaml

## Data Loading Pattern

### Data Provider Interface
```python
from abc import ABC, abstractmethod
import pandas as pd

class DataProvider(ABC):
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess data."""
        pass
```

### Supported Formats
1. **CSVDataProvider**: Single-column price data
2. **MultiIndexCSVDataProvider**: Multi-index with asset/field structure

**Best Practices:**
- Always handle missing data according to config
- Validate date ranges
- Support both simple and complex data formats
- Implement proper error handling for file I/O

## Portfolio Backtesting Pattern

### Integration with VectorBT
```python
import vectorbt as vbt

def run_backtest(
    data: pd.DataFrame,
    entry_signals: pd.Series,
    exit_signals: pd.Series,
    config: PortfolioConfig
) -> vbt.Portfolio:
    """Run backtest using VectorBT."""
    return vbt.Portfolio.from_signals(
        data,
        entries=entry_signals,
        exits=exit_signals,
        init_cash=config.initial_capital,
        fees=config.fees,
        slippage=config.slippage
    )
```

### BacktestResult Dataclass
```python
@dataclass
class BacktestResult:
    portfolio: vbt.Portfolio
    metrics: Dict[str, float]
    signals: pd.DataFrame
    strategy_name: str
```

**Best Practices:**
- Use VectorBT for portfolio simulation
- Always calculate comprehensive metrics
- Store both portfolio object and key metrics
- Include signal data for debugging

## Signal Generation Pattern

### Signal Format
Signals must be boolean pandas Series:
- **entry_signals**: True when entering position (0 → 1)
- **exit_signals**: True when exiting position (1 → 0)

```python
def generate_signals(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Generate entry and exit signals."""
    entry_signals = (data['signal'] > threshold).astype(bool)
    exit_signals = (data['signal'] < -threshold).astype(bool)
    return entry_signals, exit_signals
```

**Best Practices:**
- Signals must be mutually exclusive (never both True)
- Align signals with data index
- Support both daily and weekly rebalancing
- Document signal logic clearly

## Feature Engineering Pattern

### Feature Engineer Classes
```python
class TechnicalFeatureEngineer:
    """Generate technical indicators (RSI, MACD, Bollinger, etc.)."""

    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical features to data."""
        # 40+ indicators configured in config.yaml
        pass
```

**Available Feature Engineers:**
1. **TechnicalFeatureEngineer**: Price-based indicators
2. **CrossAssetFeatureEngineer**: Multi-asset relationships
3. **MultiAssetFeatureEngineer**: Combined signals

**Best Practices:**
- Make feature engineering configurable via YAML
- Support feature selection
- Handle NaN values from indicators
- Document each feature's calculation

## Metrics and Reporting Pattern

### Metrics Calculation
```python
def calculate_metrics(portfolio: vbt.Portfolio) -> Dict[str, float]:
    """Calculate comprehensive performance metrics."""
    return {
        'total_return': portfolio.total_return(),
        'sharpe_ratio': portfolio.sharpe_ratio(),
        'max_drawdown': portfolio.max_drawdown(),
        'win_rate': portfolio.win_rate(),
        # ... additional metrics
    }
```

### Report Generation
```python
from jinja2 import Template

def generate_html_report(
    results: List[BacktestResult],
    config: ReportingConfig
) -> str:
    """Generate HTML report using Jinja2 templates."""
    template = Template(html_template)
    return template.render(results=results, config=config)
```

**Best Practices:**
- Use VectorBT and QuantStats for metrics
- Generate both HTML and PDF reports
- Include detailed statistics files
- Make report precision configurable

## Factory Pattern for Strategies

### StrategyFactory
```python
class StrategyFactory:
    @staticmethod
    def create_strategy(
        strategy_name: str,
        config: dict
    ) -> BaseStrategy:
        """Create strategy instance from configuration."""
        strategy_map = {
            'cross_asset_momentum': CrossAssetMomentumStrategy,
            'vol_adaptive_momentum': VolAdaptiveMomentumStrategy,
            # ... other strategies
        }
        return strategy_map[strategy_name](config)
```

**Best Practices:**
- Use factory pattern for strategy creation
- Support strategy selection via config
- Enable/disable strategies dynamically
- List available strategies via CLI

## CLI Pattern

### Main Entry Point
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Configuration file')
    parser.add_argument('--strategies', nargs='+', help='Specific strategies')
    parser.add_argument('--run-validation', action='store_true')
    parser.add_argument('--examples', action='store_true')
    args = parser.parse_args()
```

**Supported Commands:**
- `--config`: Custom configuration file
- `--strategies`: Run specific strategies
- `--run-validation`: Execute validation framework
- `--examples`: Show usage examples
- `--list-strategies`: List available strategies

## Common Pitfalls to Avoid

1. **Look-Ahead Bias**: Never use future data in signal generation
2. **Missing Data**: Always handle NaN values before backtesting
3. **Signal Alignment**: Ensure signals align with price data index
4. **Configuration Validation**: Validate all config parameters before use
5. **Memory Management**: For large datasets, use chunking or sampling
6. **Type Safety**: Always use type hints and dataclasses
7. **Timezone Handling**: Be consistent with timezone-aware/naive datetimes

## Testing Requirements

- Unit tests for all core components
- Integration tests for full pipeline
- Fixtures for sample data in `conftest.py`
- Use pytest framework with coverage reporting
- Test both simple and multi-index data formats

## Dependencies

**Core:**
- vectorbt: Portfolio backtesting
- pandas: Data manipulation
- numpy: Numerical computing
- PyYAML: Configuration parsing

**Reporting:**
- quantstats: Performance metrics
- plotly, matplotlib: Visualization
- Jinja2: Report templates

## Code Quality Standards

- **black**: Code formatting
- **isort**: Import sorting (profile: black)
- **flake8**: Linting (max-line-length: 88, ignore: E203, W503)
- **mypy**: Type checking
- **pytest**: Testing with coverage

Always run quality checks before committing:
```bash
black .
isort .
flake8 .
mypy .
pytest --cov
```
