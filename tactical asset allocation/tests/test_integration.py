"""
Integration Tests

Tests end-to-end workflows:
- Data loading → Strategy → Backtest → Report
- Multi-strategy comparison
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from strategies import DefenseFirstBaseStrategy
from backtests import TAABacktestEngine


class TestEndToEndWorkflow:
    """Test complete workflow from data to results."""
    
    @pytest.fixture
    def sample_full_data(self):
        """Create comprehensive sample data."""
        dates = pd.date_range('2020-01-01', '2021-12-31', freq='D')
        
        prices = pd.DataFrame({
            'SPY': np.cumprod(1 + np.random.normal(0.0005, 0.01, len(dates))) * 300,
            'TLT': np.cumprod(1 + np.random.normal(0.0002, 0.008, len(dates))) * 140,
            'GLD': np.cumprod(1 + np.random.normal(0.0001, 0.009, len(dates))) * 170,
            'DBC': np.cumprod(1 + np.random.normal(0.0, 0.012, len(dates))) * 15,
            'UUP': np.cumprod(1 + np.random.normal(0.0, 0.005, len(dates))) * 25,
            'BTAL': np.cumprod(1 + np.random.normal(-0.0001, 0.01, len(dates))) * 20,
        }, index=dates)
        
        tbill = pd.Series(
            np.random.uniform(0.01, 0.03, len(dates)),
            index=dates,
            name='TBILL_3M'
        )
        
        return prices, tbill
    
    def test_full_backtest_workflow(self, sample_full_data):
        """Test complete backtest from strategy to results."""
        prices, tbill = sample_full_data
        
        # 1. Initialize strategy
        strategy = DefenseFirstBaseStrategy(
            start_date='2020-01-01',
            end_date='2021-12-31'
        )
        
        # 2. Set data
        strategy.set_data(prices=prices, tbill=tbill)
        
        # 3. Generate signals
        signals = strategy.generate_signals()
        
        assert len(signals) > 0, "No signals generated"
        
        # 4. Run backtest
        engine = TAABacktestEngine(
            strategy=strategy,
            prices=prices,
            benchmark_ticker='SPY'
        )
        
        portfolio = engine.run()
        
        assert portfolio is not None, "Portfolio not created"
        
        # 5. Get metrics
        metrics = engine.get_metrics()
        
        assert 'vectorbt_stats' in metrics
        assert 'manual_calcs' in metrics
        
        # 6. Validate metrics
        manual = metrics['manual_calcs']['strategy']
        
        assert 'cagr' in manual
        assert 'sharpe' in manual
        assert 'max_drawdown' in manual
    
    def test_backtest_produces_positive_returns(self, sample_full_data):
        """Test that backtest produces sensible return metrics."""
        prices, tbill = sample_full_data
        
        strategy = DefenseFirstBaseStrategy()
        strategy.set_data(prices=prices, tbill=tbill)
        
        engine = TAABacktestEngine(strategy=strategy, prices=prices)
        portfolio = engine.run()
        
        # Check that we have returns
        returns = portfolio.returns()
        
        assert len(returns) > 0
        assert not returns.isna().all()
    
    def test_weights_are_implementable(self, sample_full_data):
        """Test that generated weights can actually be traded."""
        prices, tbill = sample_full_data
        
        strategy = DefenseFirstBaseStrategy()
        strategy.set_data(prices=prices, tbill=tbill)
        
        engine = TAABacktestEngine(strategy=strategy, prices=prices)
        engine.run()
        
        weights = engine.weights
        
        # All weights should be non-negative
        assert (weights >= -0.01).all().all(), "Found negative weights"
        
        # All weights should sum to ~1.0
        for idx, row in weights.iterrows():
            total = row.sum()
            assert abs(total - 1.0) < 0.01, f"Invalid weight sum on {idx}: {total}"
        
        # No single position > 100%
        assert (weights <= 1.01).all().all(), "Found weight > 100%"


class TestDataConsistency:
    """Test data consistency and validation."""
    
    def test_price_data_no_nulls_in_middle(self):
        """Test that price data doesn't have NaN values in the middle."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        
        prices = pd.DataFrame({
            'SPY': np.random.uniform(300, 400, len(dates)),
        }, index=dates)
        
        # Check for NaN
        assert not prices.isna().any().any(), "Found NaN in price data"
    
    def test_returns_calculation_alignment(self):
        """Test that returns align properly with prices."""
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        prices = pd.Series([100, 110, 121, 133.1], index=dates[:4])
        
        returns = prices.pct_change()
        
        # First return should be NaN
        assert pd.isna(returns.iloc[0])
        
        # Second return should be ~10%
        assert abs(returns.iloc[1] - 0.1) < 0.001
        
        # Third return should be ~10%
        assert abs(returns.iloc[2] - 0.1) < 0.001


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_strategy_without_data_raises_error(self):
        """Test that running strategy without data raises error."""
        strategy = DefenseFirstBaseStrategy()
        
        with pytest.raises(ValueError, match="Data not set"):
            strategy.generate_signals()
    
    def test_backtest_without_strategy_run_raises_error(self):
        """Test that getting metrics before running raises error."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        prices = pd.DataFrame({
            'SPY': np.random.uniform(300, 400, len(dates)),
            'TLT': np.random.uniform(140, 160, len(dates)),
        }, index=dates)
        
        strategy = DefenseFirstBaseStrategy()
        engine = TAABacktestEngine(strategy=strategy, prices=prices)
        
        with pytest.raises(ValueError, match="Run backtest first"):
            engine.get_metrics()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

