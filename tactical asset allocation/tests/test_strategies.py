"""
Unit Tests for TAA Strategies

Tests:
- Strategy initialization
- Signal generation logic
- Weight calculation
- Momentum calculations
- Filtering logic
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from strategies import (
    VIXTimingStrategy,
    DefenseFirstBaseStrategy,
    DefenseFirstLeveredStrategy,
    SectorRotationStrategy
)


class TestVIXTimingStrategy:
    """Test VIX Timing Strategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create VIX Timing strategy instance."""
        return VIXTimingStrategy()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample price and VIX data."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        
        # Create sample prices
        prices = pd.DataFrame({
            'SPY': np.random.uniform(300, 400, len(dates)),
            'SPXL': np.random.uniform(50, 100, len(dates)),
            'TLT': np.random.uniform(140, 160, len(dates)),
            'GLD': np.random.uniform(170, 180, len(dates)),
            'DBC': np.random.uniform(15, 20, len(dates)),
            'UUP': np.random.uniform(25, 27, len(dates)),
            'BTAL': np.random.uniform(20, 25, len(dates)),
        }, index=dates)
        
        # Create sample VIX
        vix = pd.Series(
            np.random.uniform(15, 30, len(dates)),
            index=dates,
            name='VIX'
        )
        
        return prices, vix
    
    def test_strategy_initialization(self, strategy):
        """Test strategy initializes correctly."""
        assert strategy.name == 'VIX_TIMING'
        assert len(strategy.lookbacks) == 3
        assert strategy.use_multi_lookback is True
    
    def test_strategy_info(self, strategy):
        """Test strategy info contains required fields."""
        info = strategy.get_strategy_info()
        
        assert 'name' in info
        assert 'paper' in info
        assert 'start_date' in info
        assert 'end_date' in info
        assert 'rules' in info
    
    def test_simple_allocation_risk_on(self, strategy, sample_data):
        """Test simple VIX filter when RV < VIX (risk-on)."""
        prices, vix = sample_data
        strategy.set_data(prices=prices, vix=vix)
        
        # Force low realized vol (< VIX)
        date = prices.index[-1]
        
        # Mock low volatility
        strategy.prices['SPY'].iloc[-20:] = 350  # Flat prices = low vol
        strategy.vix.iloc[-1] = 25  # High VIX
        
        allocation = strategy._simple_allocation(date, lookback=20)
        
        assert allocation == 1.0, "Should be 100% SPXL when RV < VIX"
    
    def test_simple_allocation_risk_off(self, strategy, sample_data):
        """Test simple VIX filter when RV >= VIX (risk-off)."""
        prices, vix = sample_data
        strategy.set_data(prices=prices, vix=vix)
        
        date = prices.index[-1]
        
        # Mock high volatility
        strategy.prices['SPY'].iloc[-20:] = np.random.uniform(300, 400, 20)
        strategy.vix.iloc[-1] = 10  # Low VIX
        
        allocation = strategy._simple_allocation(date, lookback=20)
        
        assert allocation == 0.0, "Should be 0% SPXL when RV >= VIX"
    
    def test_generate_signals_returns_dataframe(self, strategy, sample_data):
        """Test signal generation returns DataFrame."""
        prices, vix = sample_data
        strategy.set_data(prices=prices, vix=vix)
        
        signals = strategy.generate_signals()
        
        assert isinstance(signals, pd.DataFrame)
        assert len(signals) > 0
        assert signals.index.name is None or signals.index.name == ''
    
    def test_weights_sum_to_one(self, strategy, sample_data):
        """Test that generated weights sum to ~1.0."""
        prices, vix = sample_data
        strategy.set_data(prices=prices, vix=vix)
        
        signals = strategy.generate_signals()
        
        # Check each row sums to ~1.0
        for idx, row in signals.iterrows():
            total = row.sum()
            assert abs(total - 1.0) < 0.01, f"Weights don't sum to 1.0 on {idx}: {total}"


class TestDefenseFirstBaseStrategy:
    """Test Defense First Base Strategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create Defense First Base strategy instance."""
        return DefenseFirstBaseStrategy()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for Defense First."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        
        prices = pd.DataFrame({
            'SPY': np.random.uniform(300, 400, len(dates)),
            'TLT': np.random.uniform(140, 160, len(dates)),
            'GLD': np.random.uniform(170, 180, len(dates)),
            'DBC': np.random.uniform(15, 20, len(dates)),
            'UUP': np.random.uniform(25, 27, len(dates)),
            'BTAL': np.random.uniform(20, 25, len(dates)),
        }, index=dates)
        
        tbill = pd.Series(
            np.random.uniform(0.01, 0.03, len(dates)),
            index=dates,
            name='TBILL_3M'
        )
        
        return prices, tbill
    
    def test_strategy_initialization(self, strategy):
        """Test strategy initializes correctly."""
        assert strategy.name == 'DEFENSE_FIRST_BASE'
        assert strategy.fallback_asset == 'SPY'
        assert len(strategy.fixed_weights) == 4
        assert sum(strategy.fixed_weights) == 1.0
    
    def test_momentum_calculation(self, strategy):
        """Test momentum calculation for an asset."""
        # Create simple price series
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        prices = pd.Series(range(100, 100 + len(dates)), index=dates)
        
        momentum = strategy.calculate_momentum(prices)
        
        assert isinstance(momentum, float)
        assert momentum > 0  # Trending up
    
    def test_generate_signals_returns_dataframe(self, strategy, sample_data):
        """Test signal generation returns DataFrame."""
        prices, tbill = sample_data
        strategy.set_data(prices=prices, tbill=tbill)
        
        signals = strategy.generate_signals()
        
        assert isinstance(signals, pd.DataFrame)
        assert len(signals) > 0
    
    def test_weights_sum_to_one(self, strategy, sample_data):
        """Test that generated weights sum to ~1.0."""
        prices, tbill = sample_data
        strategy.set_data(prices=prices, tbill=tbill)
        
        signals = strategy.generate_signals()
        
        for idx, row in signals.iterrows():
            total = row.sum()
            assert abs(total - 1.0) < 0.01, f"Weights don't sum to 1.0 on {idx}"


class TestDefenseFirstLeveredStrategy:
    """Test Defense First Leveraged Strategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create Defense First Leveraged strategy instance."""
        return DefenseFirstLeveredStrategy()
    
    def test_strategy_initialization(self, strategy):
        """Test strategy initializes correctly."""
        assert strategy.name == 'DEFENSE_FIRST_LEVERED'
        assert strategy.fallback_asset == 'SPXL'  # Key difference
    
    def test_inherits_from_base(self, strategy):
        """Test that leveraged version inherits from base."""
        assert isinstance(strategy, DefenseFirstBaseStrategy)
    
    def test_strategy_info_shows_spxl(self, strategy):
        """Test strategy info shows SPXL as fallback."""
        info = strategy.get_strategy_info()
        
        assert info['fallback_asset'] == 'SPXL'
        assert 'SPXL' in info['name']


class TestSectorRotationStrategy:
    """Test Sector Rotation Strategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create Sector Rotation strategy instance."""
        return SectorRotationStrategy()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample sector data."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        
        sectors = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
        
        prices = pd.DataFrame({
            sector: np.random.uniform(50, 100, len(dates))
            for sector in sectors
        }, index=dates)
        
        # Add SPY for benchmarking
        prices['SPY'] = np.random.uniform(300, 400, len(dates))
        
        return prices
    
    def test_strategy_initialization(self, strategy):
        """Test strategy initializes correctly."""
        assert strategy.name == 'SECTOR_ROTATION'
        assert strategy.top_n_sectors == 5
        assert len(strategy.lookbacks) == 4
    
    def test_composite_rank_calculation(self, strategy, sample_data):
        """Test composite rank calculation."""
        strategy.set_data(prices=sample_data)
        
        date = sample_data.index[-1]
        sectors = ['XLB', 'XLE', 'XLF']
        
        ranks = strategy._calculate_composite_ranks(sectors, date)
        
        assert len(ranks) == len(sectors)
        for sector in sectors:
            assert sector in ranks
            assert isinstance(ranks[sector], (int, float))
    
    def test_momentum_signals(self, strategy, sample_data):
        """Test momentum signal calculation."""
        strategy.set_data(prices=sample_data)
        
        date = sample_data.index[-1]
        sectors = ['XLB', 'XLE']
        
        scores = strategy._calculate_momentum_signals(sectors, date)
        
        assert len(scores) == len(sectors)
        for sector in sectors:
            assert isinstance(scores[sector], (int, float))
    
    def test_volatility_signals(self, strategy, sample_data):
        """Test volatility signal calculation."""
        strategy.set_data(prices=sample_data)
        
        date = sample_data.index[-1]
        sectors = ['XLB', 'XLE']
        
        scores = strategy._calculate_volatility_signals(sectors, date)
        
        assert len(scores) == len(sectors)
        for sector in sectors:
            assert scores[sector] > 0  # Volatility always positive
    
    def test_generate_signals_returns_dataframe(self, strategy, sample_data):
        """Test signal generation returns DataFrame."""
        strategy.set_data(prices=sample_data)
        
        signals = strategy.generate_signals()
        
        assert isinstance(signals, pd.DataFrame)
        assert len(signals) > 0
    
    def test_weights_sum_to_one(self, strategy, sample_data):
        """Test that generated weights sum to ~1.0."""
        strategy.set_data(prices=sample_data)
        
        signals = strategy.generate_signals()
        
        for idx, row in signals.iterrows():
            total = row.sum()
            assert abs(total - 1.0) < 0.01, f"Weights don't sum to 1.0 on {idx}"
    
    def test_selects_top_n_sectors(self, strategy, sample_data):
        """Test that strategy selects exactly top_n_sectors."""
        strategy.set_data(prices=sample_data)
        
        signals = strategy.generate_signals()
        
        for idx, row in signals.iterrows():
            # Count non-zero non-cash allocations
            non_cash = row.drop('CASH', errors='ignore')
            selected = (non_cash > 0).sum()
            
            # Should be <= top_n_sectors (can be less if negative momentum)
            assert selected <= strategy.top_n_sectors


class TestBaseStrategyMethods:
    """Test base strategy utility methods."""
    
    @pytest.fixture
    def strategy(self):
        """Create a concrete strategy instance for testing base methods."""
        return DefenseFirstBaseStrategy()
    
    def test_calculate_returns(self, strategy):
        """Test daily returns calculation."""
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        prices = pd.DataFrame({
            'SPY': [100, 101, 102, 101, 103, 104, 105, 104, 106, 107],
        }, index=dates)
        
        returns = strategy.calculate_returns(prices)
        
        assert len(returns) == len(prices)
        assert returns.iloc[0].isna().all()  # First row should be NaN
        assert abs(returns['SPY'].iloc[1] - 0.01) < 0.001  # 1% return
    
    def test_calculate_realized_volatility(self, strategy):
        """Test realized volatility calculation."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        returns = pd.Series(
            np.random.normal(0.001, 0.02, len(dates)),
            index=dates
        )
        
        vol = strategy.calculate_realized_volatility(returns, lookback=20)
        
        assert isinstance(vol, float)
        assert vol > 0
    
    def test_normalize_weights(self, strategy):
        """Test weight normalization."""
        weights = {'SPY': 0.3, 'TLT': 0.5, 'GLD': 0.2}
        
        normalized = strategy.normalize_weights(weights)
        
        total = sum(normalized.values())
        assert abs(total - 1.0) < 0.001


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

