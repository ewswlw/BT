"""
Unit Tests for ETF Inception Dates and Dynamic Asset Manager

Tests:
- ETF inception date registry accuracy
- Dynamic asset availability logic
- Sector universe evolution
- Defensive asset availability
"""

import pytest
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from data.etf_inception_dates import (
    ETF_INCEPTION_DATES,
    STRATEGY_START_DATES,
    DynamicAssetManager
)


class TestETFInceptionDates:
    """Test ETF inception date registry."""
    
    def test_all_major_etfs_present(self):
        """Verify all required ETFs are in registry."""
        required_etfs = [
            'SPY', 'SPXL', 'TLT', 'GLD', 'DBC', 'UUP', 'BTAL',
            'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY',
            'XLRE', 'XLC'
        ]
        
        for etf in required_etfs:
            assert etf in ETF_INCEPTION_DATES, f"{etf} missing from registry"
    
    def test_inception_dates_are_datetime(self):
        """Verify all inception dates are datetime objects."""
        for etf, date in ETF_INCEPTION_DATES.items():
            assert isinstance(date, datetime), f"{etf} date is not datetime"
    
    def test_spxl_inception_date(self):
        """Verify SPXL inception date (critical for leveraged strategy)."""
        expected = datetime(2008, 11, 5)
        assert ETF_INCEPTION_DATES['SPXL'] == expected
    
    def test_btal_inception_date(self):
        """Verify BTAL inception date (critical for defensive strategies)."""
        expected = datetime(2011, 5, 11)
        assert ETF_INCEPTION_DATES['BTAL'] == expected
    
    def test_sector_etfs_original_9(self):
        """Verify original 9 sector ETFs all launched same day."""
        original_9 = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
        expected_date = datetime(1998, 12, 16)
        
        for sector in original_9:
            assert ETF_INCEPTION_DATES[sector] == expected_date, \
                f"{sector} has incorrect inception date"
    
    def test_xlre_inception_date(self):
        """Verify XLRE (Real Estate) inception date."""
        expected = datetime(2015, 10, 7)
        assert ETF_INCEPTION_DATES['XLRE'] == expected
    
    def test_xlc_inception_date(self):
        """Verify XLC (Communication Services) inception date."""
        expected = datetime(2018, 6, 18)
        assert ETF_INCEPTION_DATES['XLC'] == expected


class TestDynamicAssetManager:
    """Test Dynamic Asset Manager functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create asset manager instance."""
        return DynamicAssetManager()
    
    def test_get_available_assets_all_available(self, manager):
        """Test when all requested assets are available."""
        date = pd.Timestamp('2020-01-01')
        assets = ['SPY', 'TLT', 'GLD']
        
        available = manager.get_available_assets(date, assets)
        
        assert len(available) == 3
        assert set(available) == set(assets)
    
    def test_get_available_assets_btal_not_yet_launched(self, manager):
        """Test BTAL availability before inception (May 2011)."""
        date = pd.Timestamp('2010-01-01')
        assets = ['TLT', 'GLD', 'DBC', 'UUP', 'BTAL']
        
        available = manager.get_available_assets(date, assets)
        
        # BTAL should not be available
        assert 'BTAL' not in available
        assert set(available) == {'TLT', 'GLD', 'DBC', 'UUP'}
    
    def test_get_available_assets_btal_after_launch(self, manager):
        """Test BTAL availability after inception."""
        date = pd.Timestamp('2012-01-01')
        assets = ['TLT', 'GLD', 'DBC', 'UUP', 'BTAL']
        
        available = manager.get_available_assets(date, assets)
        
        # BTAL should be available
        assert 'BTAL' in available
        assert len(available) == 5
    
    def test_get_sector_universe_original_9(self, manager):
        """Test sector universe before XLRE launch (1999-2015)."""
        date = pd.Timestamp('2010-01-01')
        
        sectors = manager.get_sector_universe(date)
        
        expected = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
        assert len(sectors) == 9
        assert set(sectors) == set(expected)
    
    def test_get_sector_universe_with_xlre(self, manager):
        """Test sector universe after XLRE but before XLC (2015-2018)."""
        date = pd.Timestamp('2016-01-01')
        
        sectors = manager.get_sector_universe(date)
        
        assert len(sectors) == 10
        assert 'XLRE' in sectors
        assert 'XLC' not in sectors
    
    def test_get_sector_universe_all_11(self, manager):
        """Test sector universe after XLC launch (2018+)."""
        date = pd.Timestamp('2020-01-01')
        
        sectors = manager.get_sector_universe(date)
        
        assert len(sectors) == 11
        assert 'XLRE' in sectors
        assert 'XLC' in sectors
    
    def test_get_defensive_universe_before_btal(self, manager):
        """Test defensive universe before BTAL (2007-2011)."""
        date = pd.Timestamp('2010-01-01')
        
        defensives = manager.get_defensive_universe(date)
        
        expected = ['TLT', 'GLD', 'DBC', 'UUP']
        assert len(defensives) == 4
        assert set(defensives) == set(expected)
    
    def test_get_defensive_universe_after_btal(self, manager):
        """Test defensive universe after BTAL launch."""
        date = pd.Timestamp('2012-01-01')
        
        defensives = manager.get_defensive_universe(date)
        
        expected = ['TLT', 'GLD', 'DBC', 'UUP', 'BTAL']
        assert len(defensives) == 5
        assert set(defensives) == set(expected)
    
    def test_is_asset_available_true(self, manager):
        """Test is_asset_available returns True for available asset."""
        date = pd.Timestamp('2020-01-01')
        
        assert manager.is_asset_available(date, 'SPY') is True
    
    def test_is_asset_available_false(self, manager):
        """Test is_asset_available returns False for unavailable asset."""
        date = pd.Timestamp('2010-01-01')
        
        assert manager.is_asset_available(date, 'BTAL') is False
    
    def test_get_inception_date(self, manager):
        """Test get_inception_date returns correct date."""
        spy_inception = manager.get_inception_date('SPY')
        
        assert spy_inception == datetime(1993, 1, 29)
    
    def test_get_inception_date_unknown_asset(self, manager):
        """Test get_inception_date raises error for unknown asset."""
        with pytest.raises(ValueError, match="Unknown asset"):
            manager.get_inception_date('UNKNOWN')
    
    def test_get_available_assets_unknown_asset(self, manager):
        """Test get_available_assets raises error for unknown asset."""
        date = pd.Timestamp('2020-01-01')
        
        with pytest.raises(ValueError, match="Unknown asset"):
            manager.get_available_assets(date, ['SPY', 'UNKNOWN'])


class TestStrategyStartDates:
    """Test strategy start date constraints."""
    
    def test_all_strategies_defined(self):
        """Verify all 4 strategies have start dates."""
        required_strategies = [
            'vix_timing',
            'defense_first_base',
            'defense_first_levered',
            'sector_rotation'
        ]
        
        for strategy in required_strategies:
            assert strategy in STRATEGY_START_DATES, \
                f"{strategy} missing from STRATEGY_START_DATES"
    
    def test_vix_timing_start_date(self):
        """Verify VIX Timing starts at 2013-01-01 (per paper)."""
        expected = datetime(2013, 1, 1)
        assert STRATEGY_START_DATES['vix_timing'] == expected
    
    def test_defense_first_base_start_date(self):
        """Verify Defense First Base starts at 2008-02-01 (per paper)."""
        expected = datetime(2008, 2, 1)
        assert STRATEGY_START_DATES['defense_first_base'] == expected
    
    def test_defense_first_levered_constrained_by_spxl(self):
        """Verify Defense First Leveraged can't start before SPXL inception."""
        spxl_inception = ETF_INCEPTION_DATES['SPXL']
        strategy_start = STRATEGY_START_DATES['defense_first_levered']
        
        assert strategy_start >= spxl_inception, \
            "Strategy starts before SPXL was available"
    
    def test_sector_rotation_start_date(self):
        """Verify Sector Rotation starts at 1999-12-01 (per paper)."""
        expected = datetime(1999, 12, 1)
        assert STRATEGY_START_DATES['sector_rotation'] == expected


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

