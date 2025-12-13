"""
Comprehensive tests for portfolio management and backtesting engine.
Tests PortfolioEngine and BacktestResult.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cad_ig_er_index_backtesting.core.portfolio import PortfolioEngine, BacktestResult
from cad_ig_er_index_backtesting.core.config import PortfolioConfig


class TestPortfolioEngine:
    """Test PortfolioEngine."""
    
    def create_test_data(self):
        """Create test price data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        return pd.DataFrame({'price': prices}, index=dates)
    
    def create_test_signals(self, n=100):
        """Create test entry/exit signals."""
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        entry = pd.Series([False] * n, index=dates)
        exit = pd.Series([False] * n, index=dates)
        
        # Create some signals
        entry.iloc[10] = True
        exit.iloc[20] = True
        entry.iloc[30] = True
        exit.iloc[40] = True
        
        return entry, exit
    
    def test_basic_backtest(self):
        """Test basic backtest execution."""
        data = self.create_test_data()
        entry, exit = self.create_test_signals()
        
        config = PortfolioConfig(
            initial_capital=100000,
            frequency='D',
            fees=0.0,
            slippage=0.0
        )
        
        engine = PortfolioEngine(config)
        result = engine.run_backtest(data, entry, exit, asset_column='price')
        
        assert isinstance(result, BacktestResult)
        assert len(result.returns) == len(data)
        assert len(result.equity_curve) == len(data)
        assert result.equity_curve.iloc[0] == config.initial_capital
    
    def test_backtest_with_fees(self):
        """Test backtest with fees."""
        data = self.create_test_data()
        entry, exit = self.create_test_signals()
        
        config = PortfolioConfig(
            initial_capital=100000,
            fees=0.001,  # 0.1% fee
            slippage=0.0
        )
        
        engine = PortfolioEngine(config)
        result = engine.run_backtest(data, entry, exit, asset_column='price')
        
        # With fees, equity should be lower
        assert result.equity_curve.iloc[-1] < config.initial_capital * (data['price'].iloc[-1] / data['price'].iloc[0])
    
    def test_backtest_with_slippage(self):
        """Test backtest with slippage."""
        data = self.create_test_data()
        entry, exit = self.create_test_signals()
        
        config = PortfolioConfig(
            initial_capital=100000,
            fees=0.0,
            slippage=0.0005  # 0.05% slippage
        )
        
        engine = PortfolioEngine(config)
        result = engine.run_backtest(data, entry, exit, asset_column='price')
        
        assert isinstance(result, BacktestResult)
    
    def test_backtest_with_leverage(self):
        """Test backtest with leverage."""
        data = self.create_test_data()
        entry, exit = self.create_test_signals()
        
        config = PortfolioConfig(
            initial_capital=100000,
            leverage=2.0  # 2x leverage
        )
        
        engine = PortfolioEngine(config)
        result = engine.run_backtest(data, entry, exit, asset_column='price')
        
        # With leverage, returns should be amplified
        assert isinstance(result, BacktestResult)
    
    def test_backtest_no_signals(self):
        """Test backtest with no trading signals."""
        data = self.create_test_data()
        entry = pd.Series([False] * len(data), index=data.index)
        exit = pd.Series([False] * len(data), index=data.index)
        
        config = PortfolioConfig(initial_capital=100000)
        engine = PortfolioEngine(config)
        result = engine.run_backtest(data, entry, exit, asset_column='price')
        
        # Should return cash position
        assert result.equity_curve.iloc[-1] == config.initial_capital
        assert result.returns.sum() == 0
    
    def test_backtest_all_long(self):
        """Test backtest with all long positions."""
        data = self.create_test_data()
        entry = pd.Series([True] * len(data), index=data.index)
        exit = pd.Series([False] * len(data), index=data.index)
        
        config = PortfolioConfig(initial_capital=100000)
        engine = PortfolioEngine(config)
        result = engine.run_backtest(data, entry, exit, asset_column='price')
        
        # Should track price performance
        assert result.equity_curve.iloc[-1] != config.initial_capital
    
    def test_backtest_mismatched_index(self):
        """Test backtest with mismatched data/signal indices."""
        data = self.create_test_data()
        dates = pd.date_range('2020-01-05', periods=50, freq='D')  # Different dates
        entry = pd.Series([True] * 50, index=dates)
        exit = pd.Series([False] * 50, index=dates)
        
        config = PortfolioConfig(initial_capital=100000)
        engine = PortfolioEngine(config)
        
        # Should handle index alignment
        result = engine.run_backtest(data, entry, exit, asset_column='price')
        assert isinstance(result, BacktestResult)
    
    def test_backtest_empty_data(self):
        """Test backtest with empty data."""
        data = pd.DataFrame({'price': []}, index=pd.DatetimeIndex([]))
        entry = pd.Series([], dtype=bool, index=pd.DatetimeIndex([]))
        exit = pd.Series([], dtype=bool, index=pd.DatetimeIndex([]))
        
        config = PortfolioConfig(initial_capital=100000)
        engine = PortfolioEngine(config)
        
        with pytest.raises((ValueError, IndexError)):
            engine.run_backtest(data, entry, exit, asset_column='price')
    
    def test_backtest_missing_asset_column(self):
        """Test backtest with missing asset column."""
        data = self.create_test_data()
        entry, exit = self.create_test_signals()
        
        config = PortfolioConfig(initial_capital=100000)
        engine = PortfolioEngine(config)
        
        # Should use first column if asset_column is None
        result = engine.run_backtest(data, entry, exit, asset_column=None)
        assert isinstance(result, BacktestResult)


class TestBacktestResult:
    """Test BacktestResult dataclass."""
    
    def test_backtest_result_creation(self):
        """Test creating BacktestResult."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        entry = pd.Series([False] * 100, index=dates)
        exit = pd.Series([False] * 100, index=dates)
        returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        equity = 100000 * (1 + returns).cumprod()
        
        result = BacktestResult(
            strategy_name="test_strategy",
            portfolio=None,
            entry_signals=entry,
            exit_signals=exit,
            returns=returns,
            equity_curve=equity,
            metrics={'sharpe': 1.0},
            config={'test': True},
            trades_count=5,
            time_in_market=0.5
        )
        
        assert result.strategy_name == "test_strategy"
        assert len(result.returns) == 100
        assert result.trades_count == 5
        assert result.time_in_market == 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

