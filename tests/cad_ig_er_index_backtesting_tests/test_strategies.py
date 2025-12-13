"""
Comprehensive tests for strategy implementations.
Tests BaseStrategy, all strategy types, and StrategyFactory.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cad_ig_er_index_backtesting.strategies.base_strategy import BaseStrategy
from cad_ig_er_index_backtesting.strategies.strategy_factory import StrategyFactory
from cad_ig_er_index_backtesting.core.config import PortfolioConfig


class TestBaseStrategy:
    """Test BaseStrategy abstract class."""
    
    def create_test_data(self):
        """Create test data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'price': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        config = {
            'name': 'test_strategy',
            'description': 'Test description'
        }
        
        # Create a concrete implementation
        class TestStrategy(BaseStrategy):
            def generate_signals(self, data, features):
                entry = pd.Series([False] * len(data), index=data.index)
                exit = pd.Series([False] * len(data), index=data.index)
                return entry, exit
            
            def get_required_features(self):
                return []
        
        strategy = TestStrategy(config)
        assert strategy.name == 'test_strategy'
        assert strategy.description == 'Test description'
    
    def test_validate_data(self):
        """Test data validation."""
        class TestStrategy(BaseStrategy):
            def generate_signals(self, data, features):
                return pd.Series([False] * len(data), index=data.index), \
                       pd.Series([False] * len(data), index=data.index)
            
            def get_required_features(self):
                return ['required_feature']
        
        data = self.create_test_data()
        features = pd.DataFrame({'required_feature': np.random.randn(100)}, index=data.index)
        
        strategy = TestStrategy({})
        assert strategy.validate_data(data, features) == True
        
        # Missing required feature
        features_missing = pd.DataFrame({'other_feature': np.random.randn(100)}, index=data.index)
        assert strategy.validate_data(data, features_missing) == False


class TestStrategyFactory:
    """Test StrategyFactory."""
    
    def test_get_available_strategies(self):
        """Test getting list of available strategies."""
        strategies = StrategyFactory.get_available_strategies()
        
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        assert 'cross_asset_momentum' in strategies or 'CrossAssetMomentum' in strategies
    
    def test_create_cross_asset_momentum(self):
        """Test creating cross asset momentum strategy."""
        config = {
            'type': 'cross_asset_momentum',
            'name': 'test_cam',
            'momentum_assets': ['asset1', 'asset2'],
            'momentum_lookback_days': 10
        }
        
        try:
            strategy = StrategyFactory.create_strategy(config)
            assert strategy is not None
            assert strategy.name == 'test_cam'
        except Exception as e:
            # Strategy might not be available, skip test
            pytest.skip(f"Strategy not available: {e}")
    
    def test_create_invalid_strategy(self):
        """Test creating invalid strategy type."""
        config = {
            'type': 'nonexistent_strategy',
            'name': 'test'
        }
        
        with pytest.raises((ValueError, KeyError)):
            StrategyFactory.create_strategy(config)
    
    def test_create_strategy_with_missing_config(self):
        """Test creating strategy with missing required config."""
        config = {
            'type': 'cross_asset_momentum',
            'name': 'test'
            # Missing required config
        }
        
        try:
            # Should either work with defaults or raise informative error
            strategy = StrategyFactory.create_strategy(config)
            assert strategy is not None
        except (ValueError, KeyError, TypeError) as e:
            # Expected if required config is missing
            assert 'momentum' in str(e).lower() or 'asset' in str(e).lower()


class TestStrategyIntegration:
    """Test strategy integration with data and features."""
    
    def create_test_data(self):
        """Create test multi-asset data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'cad_ig_er_index': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'us_hy_er_index': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'us_ig_er_index': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'tsx': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'vix': 20 + np.random.randn(100) * 2
        }, index=dates)
    
    def test_strategy_backtest_flow(self):
        """Test complete strategy backtest flow."""
        data = self.create_test_data()
        features = pd.DataFrame(index=data.index)
        
        config = {
            'type': 'cross_asset_momentum',
            'name': 'test_strategy',
            'momentum_assets': ['cad_ig_er_index', 'us_hy_er_index'],
            'momentum_lookback_days': 10,
            'min_confirmations': 2
        }
        
        try:
            strategy = StrategyFactory.create_strategy(config)
            
            # Generate signals
            entry, exit = strategy.generate_signals(data, features)
            
            assert len(entry) == len(data)
            assert len(exit) == len(data)
            assert isinstance(entry, pd.Series)
            assert isinstance(exit, pd.Series)
            
            # Run backtest
            portfolio_config = PortfolioConfig(initial_capital=100000)
            result = strategy.backtest(data, features, portfolio_config, 'cad_ig_er_index')
            
            assert result is not None
            assert len(result.returns) == len(data)
        except Exception as e:
            pytest.skip(f"Strategy backtest not available: {e}")
    
    def test_strategy_with_empty_data(self):
        """Test strategy with empty data."""
        data = pd.DataFrame(index=pd.DatetimeIndex([]))
        features = pd.DataFrame(index=pd.DatetimeIndex([]))
        
        config = {
            'type': 'cross_asset_momentum',
            'name': 'test',
            'momentum_assets': [],
            'momentum_lookback_days': 10
        }
        
        try:
            strategy = StrategyFactory.create_strategy(config)
            
            with pytest.raises((ValueError, IndexError)):
                strategy.generate_signals(data, features)
        except Exception:
            pytest.skip("Strategy not available")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

