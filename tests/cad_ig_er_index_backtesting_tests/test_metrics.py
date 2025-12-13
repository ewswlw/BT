"""
Comprehensive tests for metrics calculation.
Tests MetricsCalculator with various scenarios.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cad_ig_er_index_backtesting.core.metrics import MetricsCalculator


class TestMetricsCalculator:
    """Test MetricsCalculator."""
    
    def create_test_returns(self, n=252, mean=0.001, std=0.02):
        """Create test returns series."""
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        returns = np.random.normal(mean, std, n)
        return pd.Series(returns, index=dates)
    
    def test_basic_metrics(self):
        """Test basic metrics calculation."""
        returns = self.create_test_returns()
        calculator = MetricsCalculator(frequency='D')
        metrics = calculator.calculate_comprehensive_metrics(returns)
        
        assert 'total_return' in metrics
        assert 'cagr' in metrics
        assert 'volatility' in metrics
        assert 'sharpe' in metrics
    
    def test_metrics_with_benchmark(self):
        """Test metrics calculation with benchmark."""
        returns = self.create_test_returns()
        benchmark = self.create_test_returns(mean=0.0005, std=0.015)
        
        calculator = MetricsCalculator(frequency='D')
        metrics = calculator.calculate_comprehensive_metrics(returns, benchmark)
        
        assert 'benchmark_total_return' in metrics
        assert 'benchmark_cagr' in metrics
        assert 'excess_return' in metrics
    
    def test_different_frequencies(self):
        """Test metrics with different frequencies."""
        returns = self.create_test_returns(n=52)
        
        for freq in ['D', 'W', 'M']:
            calculator = MetricsCalculator(frequency=freq)
            metrics = calculator.calculate_comprehensive_metrics(returns)
            assert 'cagr' in metrics
            assert 'volatility' in metrics
    
    def test_negative_returns(self):
        """Test metrics with negative returns."""
        returns = self.create_test_returns(mean=-0.001, std=0.02)
        calculator = MetricsCalculator(frequency='D')
        metrics = calculator.calculate_comprehensive_metrics(returns)
        
        assert metrics['total_return'] < 0
        assert metrics['cagr'] < 0
    
    def test_zero_returns(self):
        """Test metrics with zero returns."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        returns = pd.Series([0.0] * 100, index=dates)
        
        calculator = MetricsCalculator(frequency='D')
        metrics = calculator.calculate_comprehensive_metrics(returns)
        
        assert metrics['total_return'] == 0
        assert metrics['cagr'] == 0
        assert metrics['volatility'] == 0
    
    def test_empty_returns(self):
        """Test metrics with empty returns."""
        returns = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        calculator = MetricsCalculator(frequency='D')
        
        with pytest.raises((ValueError, IndexError, ZeroDivisionError)):
            calculator.calculate_comprehensive_metrics(returns)
    
    def test_drawdown_metrics(self):
        """Test drawdown calculation."""
        # Create returns with a drawdown
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        returns = pd.Series([0.01] * 50 + [-0.05] * 20 + [0.01] * 30, index=dates)
        
        calculator = MetricsCalculator(frequency='D')
        metrics = calculator.calculate_comprehensive_metrics(returns)
        
        assert 'max_drawdown' in metrics
        assert metrics['max_drawdown'] < 0
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = self.create_test_returns(mean=0.001, std=0.01)
        calculator = MetricsCalculator(frequency='D', risk_free_rate=0.0)
        metrics = calculator.calculate_comprehensive_metrics(returns)
        
        assert 'sharpe' in metrics
        assert isinstance(metrics['sharpe'], (int, float))
    
    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        returns = self.create_test_returns()
        calculator = MetricsCalculator(frequency='D')
        metrics = calculator.calculate_comprehensive_metrics(returns)
        
        assert 'sortino' in metrics
    
    def test_skew_kurtosis(self):
        """Test skewness and kurtosis calculation."""
        returns = self.create_test_returns()
        calculator = MetricsCalculator(frequency='D')
        metrics = calculator.calculate_comprehensive_metrics(returns)
        
        assert 'skew' in metrics
        assert 'kurtosis' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

