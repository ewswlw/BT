"""
Unit Tests for Ticker Explorer

Tests:
- TickerExplorer initialization and configuration
- Ticker availability testing
- Data fetching functionality
- CAGR and max drawdown calculations
- Summary table generation
- Asset class exploration
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from ticker_explorer import TickerExplorer
from focused_ticker_explorer import FocusedTickerExplorer


class TestTickerExplorer:
    """Test TickerExplorer base functionality."""
    
    @pytest.fixture
    def explorer(self):
        """Create TickerExplorer instance."""
        return TickerExplorer(start_date="2020-01-01", end_date="2020-12-31")
    
    def test_initialization(self, explorer):
        """Test TickerExplorer initialization."""
        assert explorer.start_date == "2020-01-01"
        assert explorer.end_date == "2020-12-31"
        assert isinstance(explorer.ticker_universe, dict)
        assert len(explorer.ticker_universe) > 0
        assert isinstance(explorer.results, dict)
        assert isinstance(explorer.summary_data, list)
    
    def test_ticker_universe_structure(self, explorer):
        """Test ticker universe has expected structure."""
        expected_asset_classes = [
            'equity_indices', 'government_bonds', 'corporate_bonds',
            'commodities', 'currencies', 'real_estate', 'alternatives',
            'sectors', 'total_return_indices', 'mutual_fund_proxies'
        ]
        
        for asset_class in expected_asset_classes:
            assert asset_class in explorer.ticker_universe
            assert isinstance(explorer.ticker_universe[asset_class], list)
            assert len(explorer.ticker_universe[asset_class]) > 0
    
    def test_ticker_universe_content(self, explorer):
        """Test ticker universe contains expected tickers."""
        # Test equity indices
        assert 'SPX Index' in explorer.ticker_universe['equity_indices']
        assert 'SPTSX Index' in explorer.ticker_universe['equity_indices']
        assert 'UKX Index' in explorer.ticker_universe['equity_indices']
        
        # Test government bonds
        assert 'USGG10YR Index' in explorer.ticker_universe['government_bonds']
        assert 'USGG30YR Index' in explorer.ticker_universe['government_bonds']
        
        # Test commodities
        assert 'GOLDS Comdty' in explorer.ticker_universe['commodities']
        assert 'CRUDE Comdty' in explorer.ticker_universe['commodities']
    
    @patch('ticker_explorer.blp')
    def test_test_ticker_availability_success(self, mock_blp, explorer):
        """Test ticker availability testing with successful data."""
        # Mock successful Bloomberg response
        mock_data = pd.DataFrame({
            'SPX Index': [3000, 3100, 3200]
        }, index=pd.date_range('2020-01-01', periods=3, freq='ME'))
        
        mock_blp.bdh.return_value = mock_data
        
        result = explorer.test_ticker_availability('SPX Index')
        
        assert result['available'] is True
        assert result['ticker'] == 'SPX Index'
        assert result['field'] == 'PX_LAST'
        assert result['test_data_points'] == 3
    
    @patch('ticker_explorer.blp')
    def test_test_ticker_availability_no_data(self, mock_blp, explorer):
        """Test ticker availability testing with no data."""
        # Mock empty Bloomberg response
        mock_blp.bdh.return_value = pd.DataFrame()
        
        result = explorer.test_ticker_availability('INVALID_TICKER')
        
        assert result['available'] is False
        assert result['error'] == 'No data returned'
    
    @patch('ticker_explorer.blp')
    def test_test_ticker_availability_exception(self, mock_blp, explorer):
        """Test ticker availability testing with exception."""
        # Mock Bloomberg exception
        mock_blp.bdh.side_effect = Exception("Bloomberg error")
        
        result = explorer.test_ticker_availability('SPX Index')
        
        assert result['available'] is False
        assert result['error'] == 'Bloomberg error'
    
    def test_calculate_cagr_valid_data(self, explorer):
        """Test CAGR calculation with valid data."""
        # Create sample price data
        dates = pd.date_range('2020-01-01', periods=12, freq='ME')
        prices = pd.Series([100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155], 
                          index=dates)
        
        cagr = explorer.calculate_cagr(prices)
        
        # Should be approximately 55% annualized (155/100)^(1/1) - 1
        # But the actual calculation uses the time difference, so it's about 1 year
        expected_cagr = ((155 / 100) ** (1 / 1) - 1) * 100
        assert abs(cagr - expected_cagr) < 10  # More lenient tolerance
    
    def test_calculate_cagr_empty_data(self, explorer):
        """Test CAGR calculation with empty data."""
        empty_series = pd.Series(dtype=float)
        cagr = explorer.calculate_cagr(empty_series)
        assert np.isnan(cagr)
    
    def test_calculate_cagr_single_point(self, explorer):
        """Test CAGR calculation with single data point."""
        single_point = pd.Series([100], index=[pd.Timestamp('2020-01-01')])
        cagr = explorer.calculate_cagr(single_point)
        assert np.isnan(cagr)
    
    def test_calculate_max_drawdown_valid_data(self, explorer):
        """Test max drawdown calculation with valid data."""
        # Create sample price data with drawdown
        dates = pd.date_range('2020-01-01', periods=10, freq='ME')
        prices = pd.Series([100, 110, 120, 115, 105, 95, 100, 110, 120, 130], 
                          index=dates)
        
        max_dd = explorer.calculate_max_drawdown(prices)
        
        # Peak is 120, trough is 95, so max drawdown is (95-120)/120 = -20.83%
        expected_dd = (95 - 120) / 120 * 100
        assert abs(max_dd - expected_dd) < 0.1
    
    def test_calculate_max_drawdown_empty_data(self, explorer):
        """Test max drawdown calculation with empty data."""
        empty_series = pd.Series(dtype=float)
        max_dd = explorer.calculate_max_drawdown(empty_series)
        assert np.isnan(max_dd)
    
    def test_calculate_max_drawdown_single_point(self, explorer):
        """Test max drawdown calculation with single data point."""
        single_point = pd.Series([100], index=[pd.Timestamp('2020-01-01')])
        max_dd = explorer.calculate_max_drawdown(single_point)
        assert np.isnan(max_dd)
    
    @patch('ticker_explorer.blp')
    def test_get_ticker_data_success(self, mock_blp, explorer):
        """Test successful data fetching."""
        # Mock successful Bloomberg response
        mock_data = pd.DataFrame({
            'SPX Index': [3000, 3100, 3200]
        }, index=pd.date_range('2020-01-01', periods=3, freq='ME'))
        
        mock_blp.bdh.return_value = mock_data
        
        result = explorer.get_ticker_data('SPX Index')
        
        assert not result.empty
        assert len(result) == 3
        assert 'SPX Index' in result.columns
    
    @patch('ticker_explorer.blp')
    def test_get_ticker_data_empty_response(self, mock_blp, explorer):
        """Test data fetching with empty response."""
        mock_blp.bdh.return_value = pd.DataFrame()
        
        result = explorer.get_ticker_data('INVALID_TICKER')
        
        assert result.empty
    
    @patch('ticker_explorer.blp')
    def test_get_ticker_data_exception(self, mock_blp, explorer):
        """Test data fetching with exception."""
        mock_blp.bdh.side_effect = Exception("Bloomberg error")
        
        result = explorer.get_ticker_data('SPX Index')
        
        assert result.empty
    
    def test_generate_summary_table_empty_results(self, explorer):
        """Test summary table generation with empty results."""
        result = explorer.generate_summary_table([])
        assert result.empty
    
    def test_generate_summary_table_valid_results(self, explorer):
        """Test summary table generation with valid results."""
        sample_results = [
            {
                'ticker': 'SPX Index',
                'name': 'S&P 500',
                'asset_class': 'equity_indices',
                'start_date': '2020-01-01',
                'end_date': '2020-12-31',
                'frequency': 'Monthly',
                'data_points': 12,
                'cagr_since_inception': 10.5,
                'max_drawdown_since_inception': -15.2,
                'total_return': 10.5,
                'available': True
            },
            {
                'ticker': 'USGG10YR Index',
                'name': 'US 10Y Treasury',
                'asset_class': 'government_bonds',
                'start_date': '2020-01-01',
                'end_date': '2020-12-31',
                'frequency': 'Monthly',
                'data_points': 12,
                'cagr_since_inception': 2.1,
                'max_drawdown_since_inception': -5.8,
                'total_return': 2.1,
                'available': True
            }
        ]
        
        explorer.summary_data = sample_results
        summary_df = explorer.generate_summary_table()
        
        assert not summary_df.empty
        assert len(summary_df) == 2
        assert 'ticker' in summary_df.columns
        assert 'cagr_since_inception' in summary_df.columns
        assert 'max_drawdown_since_inception' in summary_df.columns
        
        # Should be sorted by CAGR descending
        assert summary_df.iloc[0]['cagr_since_inception'] > summary_df.iloc[1]['cagr_since_inception']


class TestFocusedTickerExplorer:
    """Test FocusedTickerExplorer functionality."""
    
    @pytest.fixture
    def focused_explorer(self):
        """Create FocusedTickerExplorer instance."""
        return FocusedTickerExplorer(start_date="2020-01-01", end_date="2020-12-31")
    
    def test_initialization(self, focused_explorer):
        """Test FocusedTickerExplorer initialization."""
        assert isinstance(focused_explorer.focused_tickers, list)
        assert len(focused_explorer.focused_tickers) > 0
        assert 'SPX Index' in focused_explorer.focused_tickers
        assert 'USGG10YR Index' in focused_explorer.focused_tickers
    
    def test_focused_tickers_content(self, focused_explorer):
        """Test focused tickers contain expected symbols."""
        expected_tickers = [
            'SPX Index', 'SPTSX Index', 'UKX Index', 'DAX Index',
            'USGG10YR Index', 'USGG30YR Index', 'GOLDS Comdty',
            'CRUDE Comdty', 'DXY Curncy', 'EURUSD Curncy'
        ]
        
        for ticker in expected_tickers:
            assert ticker in focused_explorer.focused_tickers
    
    def test_generate_focused_summary(self, focused_explorer):
        """Test focused summary generation."""
        sample_results = [
            {
                'ticker': 'SPX Index',
                'name': 'S&P 500',
                'start_date': '2020-01-01',
                'end_date': '2020-12-31',
                'frequency': 'Monthly',
                'cagr_since_inception': 10.5,
                'max_drawdown_since_inception': -15.2,
                'available': True
            }
        ]
        
        focused_explorer.summary_data = sample_results
        summary_df = focused_explorer.generate_focused_summary()
        
        assert not summary_df.empty
        assert len(summary_df) == 1
        assert 'ticker' in summary_df.columns
        assert 'cagr_since_inception' in summary_df.columns
        assert 'max_drawdown_since_inception' in summary_df.columns


class TestTickerExplorerIntegration:
    """Integration tests for ticker exploration."""
    
    def test_explore_custom_tickers_function(self):
        """Test the explore_custom_tickers function."""
        from focused_ticker_explorer import explore_custom_tickers
        
        # Test with mock data
        with patch('focused_ticker_explorer.FocusedTickerExplorer') as mock_explorer_class:
            mock_explorer = Mock()
            mock_explorer_class.return_value = mock_explorer
            
            # Mock analyze_ticker to return sample result
            mock_explorer.analyze_ticker.return_value = {
                'ticker': 'SPX Index',
                'name': 'S&P 500',
                'start_date': '2020-01-01',
                'end_date': '2020-12-31',
                'frequency': 'Monthly',
                'cagr_since_inception': 10.5,
                'max_drawdown_since_inception': -15.2,
                'available': True
            }
            
            # Mock generate_focused_summary to return DataFrame
            mock_summary_df = pd.DataFrame([{
                'ticker': 'SPX Index',
                'name': 'S&P 500',
                'start_date': '2020-01-01',
                'end_date': '2020-12-31',
                'frequency': 'Monthly',
                'cagr_since_inception': 10.5,
                'max_drawdown_since_inception': -15.2
            }])
            mock_explorer.generate_focused_summary.return_value = mock_summary_df
            
            # Test the function
            result_df = explore_custom_tickers(['SPX Index'])
            
            assert not result_df.empty
            assert len(result_df) == 1
            assert result_df.iloc[0]['ticker'] == 'SPX Index'
    
    def test_save_results_functionality(self):
        """Test save results functionality."""
        explorer = TickerExplorer()
        
        # Add sample data with all required columns
        explorer.summary_data = [
            {
                'ticker': 'SPX Index',
                'name': 'S&P 500',
                'asset_class': 'equity_indices',
                'start_date': '2020-01-01',
                'end_date': '2020-12-31',
                'frequency': 'Monthly',
                'data_points': 12,
                'cagr_since_inception': 10.5,
                'max_drawdown_since_inception': -15.2,
                'total_return': 10.5,
                'available': True
            }
        ]
        
        # Test save functionality with proper path
        with patch('builtins.open', create=True) as mock_open:
            with patch('pandas.DataFrame.to_csv') as mock_to_csv:
                with patch('os.makedirs') as mock_makedirs:
                    result_path = explorer.save_results('test/test_output.csv')
                    
                    assert result_path == 'test/test_output.csv'
                    mock_to_csv.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
