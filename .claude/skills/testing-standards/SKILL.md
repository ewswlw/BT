---
name: testing-standards
description: Testing standards and best practices for the backtesting framework including pytest patterns, fixtures, mocking, integration tests, and test coverage requirements
---

# Testing Standards Guide

## Overview
Comprehensive testing standards for the backtesting framework. All code must be thoroughly tested before deployment.

## When to Use This Skill
Claude should use this when:
- Writing unit tests for new features
- Creating integration tests
- Setting up test fixtures
- Mocking external dependencies
- Debugging test failures
- Ensuring test coverage

## Testing Framework

### Test Structure
```
tests/
├── conftest.py                    # Shared fixtures
├── test_config.py                 # Configuration tests
├── test_data_loader.py            # Data loading tests
├── test_feature_engineering.py    # Feature engineering tests
├── test_portfolio.py              # Portfolio backtesting tests
├── test_strategies.py             # Strategy tests
├── test_metrics.py                # Metrics calculation tests
├── test_validation_framework.py   # Validation tests
├── test_integration.py            # End-to-end tests
└── run_tests.py                   # Test runner script
```

### Running Tests
```bash
# Run all tests with coverage
pytest --cov=cad_ig_er_index_backtesting --cov-report=html

# Run specific test file
pytest tests/test_strategies.py

# Run specific test
pytest tests/test_strategies.py::test_cross_asset_momentum

# Run with verbose output
pytest -v

# Run with print statements visible
pytest -s

# Run tests in parallel
pytest -n auto
```

## Fixtures (conftest.py)

### Sample Data Fixtures
```python
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Generate sample single-asset price data."""
    dates = pd.date_range('2020-01-01', periods=252, freq='B')

    # Generate realistic price series
    returns = np.random.randn(252) * 0.01
    prices = 100 * (1 + returns).cumprod()

    return pd.DataFrame({
        'Close': prices,
        'Open': prices * (1 + np.random.randn(252) * 0.001),
        'High': prices * (1 + np.abs(np.random.randn(252) * 0.005)),
        'Low': prices * (1 - np.abs(np.random.randn(252) * 0.005)),
        'Volume': np.random.randint(1000000, 10000000, 252)
    }, index=dates)


@pytest.fixture
def multi_asset_data() -> pd.DataFrame:
    """Generate multi-asset data with price and volatility."""
    dates = pd.date_range('2020-01-01', periods=252, freq='B')

    assets = ['ASSET1', 'ASSET2', 'VIX']
    data = {}

    for asset in assets:
        returns = np.random.randn(252) * (0.01 if asset != 'VIX' else 0.05)
        base_price = 20 if asset == 'VIX' else 100
        data[asset] = base_price * (1 + returns).cumprod()

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def multi_index_data() -> pd.DataFrame:
    """Generate DataFrame with MultiIndex columns (asset, field)."""
    dates = pd.date_range('2020-01-01', periods=100, freq='B')

    assets = ['ASSET1', 'ASSET2']
    fields = ['Close', 'Volume']

    columns = pd.MultiIndex.from_product([assets, fields])
    data = np.random.randn(100, len(columns))
    data[:, 1::2] = np.abs(data[:, 1::2]) * 1000000  # Volumes
    data[:, 0::2] = 100 * (1 + data[:, 0::2] * 0.01).cumprod(axis=0)  # Prices

    return pd.DataFrame(data, index=dates, columns=columns)


@pytest.fixture
def temp_csv_file(tmp_path, sample_data):
    """Create temporary CSV file with sample data."""
    file_path = tmp_path / "test_data.csv"
    sample_data.to_csv(file_path)
    return file_path


@pytest.fixture
def temp_multi_index_csv(tmp_path, multi_index_data):
    """Create temporary multi-index CSV file."""
    file_path = tmp_path / "test_multi_index.csv"
    multi_index_data.to_csv(file_path)
    return file_path


@pytest.fixture
def sample_signals(sample_data) -> tuple:
    """Generate sample entry and exit signals."""
    # Simple momentum signals
    returns = sample_data['Close'].pct_change(20)

    entry_signals = (returns > 0.05).astype(bool)
    exit_signals = (returns < -0.05).astype(bool)

    return entry_signals, exit_signals


@pytest.fixture
def sample_config() -> dict:
    """Sample configuration dictionary."""
    return {
        'data': {
            'file_path': 'dummy.csv',
            'start_date': '2020-01-01',
            'end_date': '2020-12-31'
        },
        'portfolio': {
            'initial_capital': 100.0,
            'fees': 0.001,
            'slippage': 0.001
        },
        'reporting': {
            'output_dir': 'outputs/test',
            'generate_html': True
        }
    }
```

### Configuration Fixtures
```python
@pytest.fixture
def config_file(tmp_path) -> Path:
    """Create temporary config.yaml file."""
    config_content = """
data:
  file_path: test_data.csv
  start_date: '2020-01-01'
  end_date: '2020-12-31'

portfolio:
  initial_capital: 100.0
  fees: 0.001
  slippage: 0.001
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)
    return config_path
```

## Unit Test Patterns

### Testing Configuration
```python
# tests/test_config.py

def test_load_config(config_file):
    """Test configuration loading from YAML."""
    from core.config import load_config

    config = load_config(config_file)

    assert config.data.file_path == 'test_data.csv'
    assert config.portfolio.initial_capital == 100.0
    assert config.portfolio.fees == 0.001


def test_config_validation():
    """Test configuration validation."""
    from core.config import DataConfig

    # Valid config
    valid = DataConfig(file_path='data.csv')
    assert valid.file_path == 'data.csv'

    # Invalid: negative fees should raise error
    with pytest.raises(ValueError):
        PortfolioConfig(fees=-0.001)
```

### Testing Data Loaders
```python
# tests/test_data_loader.py

def test_csv_data_provider(temp_csv_file):
    """Test CSV data loading."""
    from core.data_loader import CSVDataProvider

    provider = CSVDataProvider(str(temp_csv_file))
    data = provider.load_data()

    assert isinstance(data, pd.DataFrame)
    assert 'Close' in data.columns
    assert len(data) == 252
    assert isinstance(data.index, pd.DatetimeIndex)


def test_multi_index_csv_provider(temp_multi_index_csv):
    """Test multi-index CSV loading."""
    from core.data_loader import MultiIndexCSVDataProvider

    provider = MultiIndexCSVDataProvider(str(temp_multi_index_csv))
    data = provider.load_data()

    assert isinstance(data.columns, pd.MultiIndex)
    assert data.columns.nlevels == 2


def test_missing_data_handling(temp_csv_file):
    """Test missing data handling strategies."""
    from core.data_loader import CSVDataProvider

    # Add some NaN values
    data = pd.read_csv(temp_csv_file, index_col=0)
    data.loc[data.index[10:15], 'Close'] = np.nan
    data.to_csv(temp_csv_file)

    # Test forward fill
    provider = CSVDataProvider(str(temp_csv_file), handle_missing='forward_fill')
    loaded = provider.load_data()
    assert not loaded['Close'].isna().any()

    # Test drop
    provider = CSVDataProvider(str(temp_csv_file), handle_missing='drop')
    loaded = provider.load_data()
    assert len(loaded) < 252
```

### Testing Strategies
```python
# tests/test_strategies.py

def test_base_strategy_interface():
    """Test that BaseStrategy cannot be instantiated."""
    from strategies.base_strategy import BaseStrategy

    with pytest.raises(TypeError):
        BaseStrategy({})


def test_cross_asset_momentum_signals(multi_asset_data):
    """Test CrossAssetMomentumStrategy signal generation."""
    from strategies.cross_asset_momentum import CrossAssetMomentumStrategy

    config = {
        'lookback_period': 20,
        'rebalance_frequency': 'W'
    }

    strategy = CrossAssetMomentumStrategy(config)
    entry, exit = strategy.generate_signals(multi_asset_data)

    # Assertions
    assert isinstance(entry, pd.Series)
    assert isinstance(exit, pd.Series)
    assert entry.dtype == bool
    assert exit.dtype == bool
    assert len(entry) == len(multi_asset_data)

    # Signals should not overlap
    assert not (entry & exit).any()

    # Should have some signals
    assert entry.sum() > 0 or exit.sum() > 0


def test_lightgbm_strategy_training(sample_data):
    """Test LightGBM strategy model training."""
    from strategies.lightgbm_strategy import LightGBMStrategy

    # Add some features
    sample_data['RSI'] = 50 + np.random.randn(len(sample_data)) * 10
    sample_data['SMA_20'] = sample_data['Close'].rolling(20).mean()

    config = {
        'forward_periods': 5,
        'num_leaves': 15,
        'learning_rate': 0.1
    }

    strategy = LightGBMStrategy(config)

    # Train model
    strategy.train_model(sample_data.dropna())

    assert strategy.model is not None
    assert strategy.feature_cols is not None

    # Generate signals
    entry, exit = strategy.generate_signals(sample_data.dropna())

    assert isinstance(entry, pd.Series)
    assert isinstance(exit, pd.Series)


def test_strategy_factory():
    """Test StrategyFactory creation."""
    from strategies.strategy_factory import StrategyFactory

    config = {'lookback_period': 20}

    strategy = StrategyFactory.create_strategy('cross_asset_momentum', config)

    assert strategy is not None
    assert hasattr(strategy, 'generate_signals')

    # Invalid strategy name
    with pytest.raises(KeyError):
        StrategyFactory.create_strategy('nonexistent_strategy', config)
```

### Testing Portfolio Backtesting
```python
# tests/test_portfolio.py

def test_backtest_execution(sample_data, sample_signals):
    """Test portfolio backtesting."""
    from core.portfolio import run_backtest
    from core.config import PortfolioConfig

    config = PortfolioConfig(
        initial_capital=100.0,
        fees=0.001,
        slippage=0.001
    )

    entry, exit = sample_signals
    portfolio = run_backtest(
        sample_data['Close'],
        entry,
        exit,
        config
    )

    assert portfolio is not None
    assert hasattr(portfolio, 'total_return')
    assert hasattr(portfolio, 'sharpe_ratio')


def test_backtest_result_dataclass(sample_data, sample_signals):
    """Test BacktestResult dataclass."""
    from core.portfolio import BacktestResult, run_backtest
    from core.config import PortfolioConfig

    config = PortfolioConfig()
    entry, exit = sample_signals

    portfolio = run_backtest(sample_data['Close'], entry, exit, config)
    metrics = {'sharpe': 1.2, 'returns': 0.15}

    result = BacktestResult(
        portfolio=portfolio,
        metrics=metrics,
        signals=pd.DataFrame({'entry': entry, 'exit': exit}),
        strategy_name='test_strategy'
    )

    assert result.strategy_name == 'test_strategy'
    assert result.metrics['sharpe'] == 1.2
```

### Testing Metrics
```python
# tests/test_metrics.py

def test_sharpe_ratio_calculation():
    """Test Sharpe ratio calculation."""
    from core.metrics import calculate_sharpe_ratio

    # Known returns
    returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])

    sharpe = calculate_sharpe_ratio(returns, periods_per_year=252)

    assert isinstance(sharpe, float)
    assert not np.isnan(sharpe)


def test_max_drawdown():
    """Test maximum drawdown calculation."""
    from core.metrics import calculate_max_drawdown

    # Create equity curve with known drawdown
    equity = pd.Series([100, 110, 105, 95, 100, 115])

    max_dd = calculate_max_drawdown(equity)

    # Maximum drawdown should be from 110 to 95 = -13.6%
    expected_dd = (95 - 110) / 110
    assert np.isclose(max_dd, expected_dd, rtol=0.01)


def test_metrics_with_zero_volatility():
    """Test metrics handle zero volatility gracefully."""
    from core.metrics import calculate_sharpe_ratio

    # Zero volatility returns
    returns = pd.Series([0.0] * 100)

    sharpe = calculate_sharpe_ratio(returns)

    # Should return NaN or 0, not raise error
    assert np.isnan(sharpe) or sharpe == 0
```

### Testing Feature Engineering
```python
# tests/test_feature_engineering.py

def test_technical_feature_engineer(sample_data):
    """Test technical feature generation."""
    from core.feature_engineering import TechnicalFeatureEngineer

    config = {
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'bollinger_period': 20
    }

    engineer = TechnicalFeatureEngineer(config)
    features = engineer.add_features(sample_data)

    assert 'RSI' in features.columns
    assert 'MACD' in features.columns
    assert 'Bollinger_Upper' in features.columns
    assert len(features) == len(sample_data)


def test_cross_asset_feature_engineer(multi_asset_data):
    """Test cross-asset feature generation."""
    from core.feature_engineering import CrossAssetFeatureEngineer

    config = {
        'correlation_period': 20
    }

    engineer = CrossAssetFeatureEngineer(config)
    features = engineer.add_features(multi_asset_data)

    assert 'correlation_ASSET1_ASSET2' in features.columns or \
           features.filter(like='correlation').shape[1] > 0
```

## Integration Tests

### End-to-End Pipeline Test
```python
# tests/test_integration.py

def test_full_pipeline(temp_csv_file, tmp_path):
    """Test complete backtesting pipeline."""
    from core.config import load_config, BaseConfig, DataConfig, PortfolioConfig
    from core.data_loader import CSVDataProvider
    from strategies.strategy_factory import StrategyFactory
    from core.portfolio import run_backtest

    # Setup config
    config = BaseConfig(
        data=DataConfig(file_path=str(temp_csv_file)),
        portfolio=PortfolioConfig(initial_capital=100.0)
    )

    # Load data
    provider = CSVDataProvider(config.data.file_path)
    data = provider.load_data()

    # Create strategy
    strategy_config = {'lookback_period': 20, 'rebalance_frequency': 'W'}
    strategy = StrategyFactory.create_strategy('cross_asset_momentum', strategy_config)

    # Generate signals
    entry, exit = strategy.generate_signals(data)

    # Run backtest
    portfolio = run_backtest(data['Close'], entry, exit, config.portfolio)

    # Verify results
    assert portfolio is not None
    assert hasattr(portfolio, 'total_return')

    total_return = portfolio.total_return()
    assert isinstance(total_return, (int, float))
```

## Mocking and Patching

### Mocking External APIs
```python
from unittest.mock import Mock, patch

def test_data_fetch_with_mock():
    """Test data fetching with mocked API."""
    from core.data_loader import APIDataProvider

    mock_response = pd.DataFrame({
        'Close': [100, 101, 102],
        'Volume': [1000, 1100, 1200]
    })

    with patch('core.data_loader.fetch_api_data', return_value=mock_response):
        provider = APIDataProvider('https://api.example.com')
        data = provider.load_data()

        assert len(data) == 3
        assert data['Close'].iloc[0] == 100
```

## Test Coverage Requirements

### Coverage Goals
- **Overall**: Minimum 80% code coverage
- **Core modules**: Minimum 90% coverage
- **Strategies**: Minimum 85% coverage
- **Critical paths**: 100% coverage (config loading, portfolio backtesting)

### Checking Coverage
```bash
# Generate HTML coverage report
pytest --cov=cad_ig_er_index_backtesting --cov-report=html

# View report
open htmlcov/index.html

# Show missing lines
pytest --cov=cad_ig_er_index_backtesting --cov-report=term-missing
```

## Testing Best Practices

### 1. Test Naming
```python
# GOOD: Descriptive test names
def test_strategy_generates_valid_signals_for_momentum_crossover():
    pass

def test_portfolio_handles_zero_capital_gracefully():
    pass

# BAD: Vague test names
def test_strategy():
    pass

def test_portfolio_1():
    pass
```

### 2. Arrange-Act-Assert Pattern
```python
def test_sharpe_ratio():
    # Arrange
    returns = pd.Series([0.01, 0.02, -0.01, 0.015])
    expected_sharpe = 1.5  # Known value

    # Act
    sharpe = calculate_sharpe_ratio(returns)

    # Assert
    assert np.isclose(sharpe, expected_sharpe, rtol=0.1)
```

### 3. One Assertion Per Test (when possible)
```python
# GOOD: Focused test
def test_data_has_datetime_index(sample_data):
    assert isinstance(sample_data.index, pd.DatetimeIndex)

def test_data_has_required_columns(sample_data):
    assert 'Close' in sample_data.columns

# ACCEPTABLE: Related assertions
def test_signal_properties(sample_signals):
    entry, exit = sample_signals
    assert entry.dtype == bool
    assert exit.dtype == bool
    assert not (entry & exit).any()
```

### 4. Test Edge Cases
```python
def test_strategy_with_empty_data():
    """Test strategy handles empty data."""
    strategy = MyStrategy({})
    empty_data = pd.DataFrame()

    with pytest.raises(ValueError):
        strategy.generate_signals(empty_data)

def test_strategy_with_single_row():
    """Test strategy with minimal data."""
    strategy = MyStrategy({'lookback': 20})
    minimal_data = pd.DataFrame({'Close': [100]})

    # Should handle gracefully
    entry, exit = strategy.generate_signals(minimal_data)
    assert len(entry) == 1
```

### 5. Parametrize Tests
```python
@pytest.mark.parametrize("lookback,expected_signals", [
    (10, 5),
    (20, 8),
    (50, 12),
])
def test_strategy_with_different_lookbacks(sample_data, lookback, expected_signals):
    """Test strategy with various lookback periods."""
    config = {'lookback_period': lookback}
    strategy = MomentumStrategy(config)

    entry, exit = strategy.generate_signals(sample_data)

    assert entry.sum() >= expected_signals
```

## Continuous Integration

### Pre-commit Hook
```bash
# .git/hooks/pre-commit
#!/bin/bash

# Run tests before commit
pytest tests/ -v

if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi

# Check coverage
coverage run -m pytest
coverage report --fail-under=80

if [ $? -ne 0 ]; then
    echo "Coverage below 80%. Commit aborted."
    exit 1
fi
```

## Common Testing Pitfalls

1. **Not Using Fixtures**: Duplicating test data setup
2. **Testing Implementation**: Test behavior, not internal details
3. **Fragile Tests**: Tests that break with minor code changes
4. **No Edge Case Testing**: Only testing happy path
5. **Slow Tests**: Not using mocks for external dependencies
6. **Unclear Failures**: Poor assertion messages
7. **Test Interdependence**: Tests depending on execution order

## Testing Checklist

Before submitting code:

- [ ] All tests pass (`pytest`)
- [ ] Coverage > 80% (`pytest --cov`)
- [ ] New features have unit tests
- [ ] Edge cases tested
- [ ] Integration test added if applicable
- [ ] Fixtures used appropriately
- [ ] Mocks used for external dependencies
- [ ] Test names are descriptive
- [ ] No commented-out tests
- [ ] Tests are fast (< 1s per test ideally)
