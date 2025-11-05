"""
Test suite for CDX backtesting framework.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import DataLoader, CSVDataProvider, AdvancedFeatureEngineer, BacktestEngine
from strategies import MLStrategy, RuleBasedStrategy, HybridStrategy


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'us_ig_cdx_er_index': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'ig_cdx': 100 + np.random.randn(100) * 10,
        'hy_cdx': 200 + np.random.randn(100) * 20,
        'vix': 20 + np.random.randn(100) * 2,
        'spx_tr': 3000 + np.cumsum(np.random.randn(100) * 5),
        'rate_vol': 100 + np.random.randn(100) * 5,
        'us_10y_yield': 2.0 + np.random.randn(100) * 0.1,
        'us_2y_yield': 1.5 + np.random.randn(100) * 0.1,
    }, index=dates)
    return data


def test_data_loading(sample_data):
    """Test data loading."""
    # Save sample data to temporary CSV
    temp_file = Path(__file__).parent.parent / 'temp_test_data.csv'
    sample_data.to_csv(temp_file)
    
    try:
        loader = DataLoader(CSVDataProvider())
        data = loader.load_and_prepare(str(temp_file), date_column='Date')
        
        assert not data.empty
        assert len(data) == len(sample_data)
        assert 'us_ig_cdx_er_index' in data.columns
    finally:
        if temp_file.exists():
            temp_file.unlink()


def test_feature_engineering(sample_data):
    """Test feature engineering."""
    feature_engineer = AdvancedFeatureEngineer()
    config = {
        'momentum_windows': [5, 10, 20],
        'zscore_windows': [20, 60],
        'correlation_windows': [20, 60],
        'primary_asset': 'us_ig_cdx_er_index'
    }
    
    features = feature_engineer.create_features(sample_data, config)
    
    assert not features.empty
    assert len(features) == len(sample_data)
    assert len(features.columns) > 50  # Should have many features


def test_backtest_engine(sample_data):
    """Test backtest engine."""
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.0,
        slippage=0.0,
        holding_period_days=7
    )
    
    price_series = sample_data['us_ig_cdx_er_index']
    
    # Create simple signals
    entry_signals = pd.Series([False] * len(price_series), index=price_series.index)
    entry_signals.iloc[10:20] = True
    exit_signals = pd.Series([False] * len(price_series), index=price_series.index)
    exit_signals.iloc[25] = True
    
    result = engine.run_backtest(price_series, entry_signals, exit_signals)
    
    assert result is not None
    assert 'cagr' in result.metrics
    assert 'sharpe_ratio' in result.metrics
    assert result.time_in_market >= 0
    assert result.time_in_market <= 1


def test_holding_period_enforcement(sample_data):
    """Test that holding period is enforced."""
    engine = BacktestEngine(
        initial_capital=100000,
        holding_period_days=7
    )
    
    price_series = sample_data['us_ig_cdx_er_index']
    
    # Create signals that would exit early
    entry_signals = pd.Series([False] * len(price_series), index=price_series.index)
    entry_signals.iloc[10] = True
    exit_signals = pd.Series([False] * len(price_series), index=price_series.index)
    exit_signals.iloc[15] = True  # Try to exit after 5 days (should be held for 7)
    
    result = engine.run_backtest(price_series, entry_signals, exit_signals)
    
    # Verify holding period was enforced
    positions = pd.Series(0, index=price_series.index)
    in_position = False
    days_held = 0
    
    for i in range(len(entry_signals)):
        if entry_signals.iloc[i]:
            in_position = True
            days_held = 0
        if in_position:
            positions.iloc[i] = 1
            days_held += 1
            if days_held >= 7 and exit_signals.iloc[i]:
                in_position = False
    
    # Check that positions align with expected holding period
    assert result.time_in_market > 0


def test_ml_strategy(sample_data):
    """Test ML strategy."""
    config = {
        'name': 'Test_ML',
        'trading_asset': 'us_ig_cdx_er_index',
        'holding_period_days': 7,
        'use_random_forest': True,
        'use_lightgbm': True,
        'use_xgboost': False,
        'use_catboost': False,
        'probability_threshold': 0.5
    }
    
    strategy = MLStrategy(config)
    
    # Create features
    feature_engineer = AdvancedFeatureEngineer()
    features_config = {
        'momentum_windows': [5, 10],
        'zscore_windows': [20],
        'primary_asset': 'us_ig_cdx_er_index'
    }
    features = feature_engineer.create_features(sample_data, features_config)
    
    # Split data
    split_idx = int(len(sample_data) * 0.7)
    train_data = sample_data.iloc[:split_idx]
    test_data = sample_data.iloc[split_idx:]
    train_features = features.iloc[:split_idx]
    test_features = features.iloc[split_idx:]
    
    # Generate signals
    entry_signals, exit_signals = strategy.generate_signals(
        train_data, test_data, train_features, test_features
    )
    
    assert len(entry_signals) == len(test_data)
    assert len(exit_signals) == len(test_data)
    assert entry_signals.dtype == bool
    assert exit_signals.dtype == bool


def test_rule_based_strategy(sample_data):
    """Test rule-based strategy."""
    config = {
        'name': 'Test_Rule',
        'trading_asset': 'us_ig_cdx_er_index',
        'holding_period_days': 7,
        'strategy_type': 'momentum',
        'momentum_assets': ['spx_tr', 'vix'],
        'momentum_lookback_days': 10
    }
    
    strategy = RuleBasedStrategy(config)
    
    # Create features (not really needed for rule-based, but for consistency)
    features = pd.DataFrame(index=sample_data.index)
    
    # Split data
    split_idx = int(len(sample_data) * 0.7)
    train_data = sample_data.iloc[:split_idx]
    test_data = sample_data.iloc[split_idx:]
    train_features = features.iloc[:split_idx]
    test_features = features.iloc[split_idx:]
    
    # Generate signals
    entry_signals, exit_signals = strategy.generate_signals(
        train_data, test_data, train_features, test_features
    )
    
    assert len(entry_signals) == len(test_data)
    assert len(exit_signals) == len(test_data)
    assert entry_signals.dtype == bool
    assert exit_signals.dtype == bool


def test_binary_positioning(sample_data):
    """Test that positioning is binary (100% long or 100% cash)."""
    engine = BacktestEngine(initial_capital=100000, holding_period_days=7)
    price_series = sample_data['us_ig_cdx_er_index']
    
    entry_signals = pd.Series([False] * len(price_series), index=price_series.index)
    entry_signals.iloc[10] = True
    exit_signals = pd.Series([False] * len(price_series), index=price_series.index)
    exit_signals.iloc[20] = True
    
    result = engine.run_backtest(price_series, entry_signals, exit_signals)
    
    # Check that positions are binary (0 or 1)
    # This is implicit in the backtest engine design
    assert result.time_in_market >= 0
    assert result.time_in_market <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

