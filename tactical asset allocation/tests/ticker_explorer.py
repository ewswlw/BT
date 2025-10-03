"""
Comprehensive Ticker Explorer for xbbg

This module provides tools to explore and analyze historical cumulative return data
across broad asset classes using Bloomberg's xbbg API. It focuses on finding data
going back to the 1970s with monthly frequency.

Key Features:
- Comprehensive ticker universe covering major asset classes
- Historical data exploration with inception date detection
- CAGR and max drawdown calculations
- Summary table generation with all key metrics
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

try:
    from xbbg import blp
except ImportError:
    print("Warning: xbbg not available. Install with: pip install xbbg")
    blp = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TickerExplorer:
    """
    Comprehensive tool for exploring historical ticker data across asset classes.
    
    This class provides methods to:
    1. Test ticker availability and data quality
    2. Fetch historical data with proper error handling
    3. Calculate key performance metrics (CAGR, max drawdown)
    4. Generate comprehensive summary tables
    """
    
    def __init__(self, start_date: str = "1970-01-01", end_date: str = None):
        """
        Initialize the TickerExplorer.
        
        Args:
            start_date: Start date for data exploration (default: 1970-01-01)
            end_date: End date for data exploration (default: today)
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        # Comprehensive ticker universe organized by asset class
        self.ticker_universe = self._build_ticker_universe()
        
        # Results storage
        self.results = {}
        self.summary_data = []
        
    def _build_ticker_universe(self) -> Dict[str, List[str]]:
        """
        Build comprehensive ticker universe covering major asset classes.
        
        Returns:
            Dictionary mapping asset classes to lists of tickers
        """
        return {
            # Major Equity Indices
            'equity_indices': [
                'SPX Index',           # S&P 500
                'SPTSX Index',         # S&P/TSX Composite (Canada)
                'UKX Index',           # FTSE 100
                'DAX Index',           # DAX (Germany)
                'NKY Index',           # Nikkei 225
                'HSI Index',          # Hang Seng
                'ASX Index',          # ASX 200 (Australia)
                'MXEF Index',         # MSCI Emerging Markets
                'MXWO Index',         # MSCI World
                'MXUS Index',         # MSCI USA
                'MXEU Index',         # MSCI Europe
                'MXAP Index',         # MSCI Asia Pacific
                'RTY Index',          # Russell 2000
                'NDX Index',          # NASDAQ 100
                'VIX Index',          # VIX
            ],
            
            # Fixed Income - Government Bonds
            'government_bonds': [
                'USGG10YR Index',      # US 10Y Treasury
                'USGG30YR Index',      # US 30Y Treasury
                'USGG2YR Index',       # US 2Y Treasury
                'USGG5YR Index',       # US 5Y Treasury
                'GT10 Govt Index',     # Global Government Bonds
                'GTGBP10Y Index',      # UK 10Y Gilt
                'GTDEM10Y Index',       # German 10Y Bund
                'GTJPY10Y Index',      # Japanese 10Y JGB
                'GTCAD10Y Index',      # Canadian 10Y Bond
                'GTAUD10Y Index',      # Australian 10Y Bond
            ],
            
            # Fixed Income - Corporate Bonds
            'corporate_bonds': [
                'LUACTRUU Index',     # US Investment Grade Corporate
                'LF98TRUU Index',     # US High Yield Corporate
                'I05510CA Index',     # Canadian Investment Grade
                'LEGATRUU Index',     # European Investment Grade
                'LEHYTRUU Index',     # European High Yield
                'LEGATRJP Index',     # Japanese Investment Grade
            ],
            
            # Commodities
            'commodities': [
                'GOLDS Comdty',        # Gold
                'SILVER Comdty',       # Silver
                'COPPER Comdty',       # Copper
                'CRUDE Comdty',        # WTI Crude Oil
                'BRENT Comdty',        # Brent Crude Oil
                'NATGAS Comdty',       # Natural Gas
                'CORN Comdty',         # Corn
                'WHEAT Comdty',        # Wheat
                'SOYBEAN Comdty',      # Soybeans
                'SUGAR Comdty',        # Sugar
                'COFFEE Comdty',       # Coffee
                'COTTON Comdty',       # Cotton
            ],
            
            # Currency Indices
            'currencies': [
                'DXY Curncy',         # US Dollar Index
                'EURUSD Curncy',       # EUR/USD
                'GBPUSD Curncy',       # GBP/USD
                'USDJPY Curncy',      # USD/JPY
                'USDCAD Curncy',       # USD/CAD
                'AUDUSD Curncy',      # AUD/USD
                'USDCHF Curncy',      # USD/CHF
                'NZDUSD Curncy',      # NZD/USD
            ],
            
            # Real Estate
            'real_estate': [
                'FTSEEPRA Index',     # FTSE EPRA/NAREIT Global
                'FTSEEPUS Index',     # FTSE EPRA/NAREIT US
                'FTSEEPEU Index',     # FTSE EPRA/NAREIT Europe
                'FTSEEPAP Index',      # FTSE EPRA/NAREIT Asia Pacific
                'FTSEEPCA Index',     # FTSE EPRA/NAREIT Canada
            ],
            
            # Alternative Indices
            'alternatives': [
                'HFRXGL Index',       # HFRI Fund Weighted Composite
                'HFRXEH Index',       # HFRI Equity Hedge
                'HFRXED Index',       # HFRI Event Driven
                'HFRXAR Index',       # HFRI Arbitrage
                'HFRXMF Index',       # HFRI Macro
                'HFRXRS Index',       # HFRI Relative Value
            ],
            
            # Sector Indices (S&P 500 Sectors)
            'sectors': [
                'SP500-10 Index',     # Energy
                'SP500-15 Index',     # Materials
                'SP500-20 Index',     # Industrials
                'SP500-25 Index',     # Consumer Discretionary
                'SP500-30 Index',     # Consumer Staples
                'SP500-35 Index',     # Health Care
                'SP500-40 Index',     # Financials
                'SP500-45 Index',     # Information Technology
                'SP500-50 Index',     # Communication Services
                'SP500-55 Index',     # Utilities
                'SP500-60 Index',     # Real Estate
            ],
            
            # Total Return Indices (if available)
            'total_return_indices': [
                'SPXT Index',         # S&P 500 Total Return
                'SPTSXT Index',       # S&P/TSX Total Return
                'UKXT Index',         # FTSE 100 Total Return
                'DAXT Index',         # DAX Total Return
                'NKYT Index',         # Nikkei 225 Total Return
            ],
            
            # Mutual Fund Proxies (if available)
            'mutual_fund_proxies': [
                'VFINX US Equity',    # Vanguard 500 Index Fund
                'VTSMX US Equity',     # Vanguard Total Stock Market
                'VGTSX US Equity',     # Vanguard Total International Stock
                'VBMFX US Equity',     # Vanguard Total Bond Market
                'VGPMX US Equity',     # Vanguard Precious Metals
            ]
        }
    
    def test_ticker_availability(self, ticker: str, field: str = 'PX_LAST') -> Dict:
        """
        Test if a ticker is available and has data in the specified date range.
        
        Args:
            ticker: Bloomberg ticker symbol
            field: Bloomberg field to test (default: PX_LAST)
            
        Returns:
            Dictionary with availability information
        """
        if blp is None:
            return {'available': False, 'error': 'xbbg not available'}
        
        try:
            # Test with a small date range first
            test_start = '2020-01-01'
            test_end = '2020-12-31'
            
            data = blp.bdh(
                ticker, 
                field, 
                test_start, 
                test_end,
                Per='MONTHLY'
            )
            
            if data.empty:
                return {
                    'available': False,
                    'error': 'No data returned',
                    'ticker': ticker,
                    'field': field
                }
            
            # Check for actual data (not all NaN)
            if data.isnull().all().all():
                return {
                    'available': False,
                    'error': 'All data is NaN',
                    'ticker': ticker,
                    'field': field
                }
            
            return {
                'available': True,
                'ticker': ticker,
                'field': field,
                'test_data_points': len(data),
                'date_range': f"{data.index.min()} to {data.index.max()}"
            }
            
        except Exception as e:
            return {
                'available': False,
                'error': str(e),
                'ticker': ticker,
                'field': field
            }
    
    def get_ticker_data(self, ticker: str, field: str = 'PX_LAST', 
                       start_date: str = None, end_date: str = None,
                       frequency: str = 'MONTHLY') -> pd.DataFrame:
        """
        Fetch historical data for a ticker.
        
        Args:
            ticker: Bloomberg ticker symbol
            field: Bloomberg field to fetch
            start_date: Start date (default: self.start_date)
            end_date: End date (default: self.end_date)
            frequency: Data frequency ('DAILY', 'WEEKLY', 'MONTHLY')
            
        Returns:
            DataFrame with historical data
        """
        if blp is None:
            raise ImportError("xbbg not available")
        
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date
        
        try:
            data = blp.bdh(
                ticker,
                field,
                start_date,
                end_date,
                Per=frequency
            )
            
            if data.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()
            
            # Clean the data
            data = data.dropna()
            
            if data.empty:
                logger.warning(f"All data is NaN for {ticker}")
                return pd.DataFrame()
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_cagr(self, data: pd.Series, start_value: float = None, 
                       end_value: float = None) -> float:
        """
        Calculate Compound Annual Growth Rate (CAGR).
        
        Args:
            data: Price series
            start_value: Starting value (default: first non-null value)
            end_value: Ending value (default: last non-null value)
            
        Returns:
            CAGR as a percentage
        """
        if data.empty or data.isnull().all():
            return np.nan
        
        # Use first and last non-null values
        clean_data = data.dropna()
        if len(clean_data) < 2:
            return np.nan
        
        start_val = start_value or clean_data.iloc[0]
        end_val = end_value or clean_data.iloc[-1]
        
        if start_val <= 0 or end_val <= 0:
            return np.nan
        
        # Calculate years between start and end
        years = (clean_data.index[-1] - clean_data.index[0]).days / 365.25
        
        if years <= 0:
            return np.nan
        
        # Calculate CAGR
        cagr = ((end_val / start_val) ** (1 / years) - 1) * 100
        return cagr
    
    def calculate_max_drawdown(self, data: pd.Series) -> float:
        """
        Calculate maximum drawdown from peak.
        
        Args:
            data: Price series
            
        Returns:
            Maximum drawdown as a percentage
        """
        if data.empty or data.isnull().all():
            return np.nan
        
        clean_data = data.dropna()
        if len(clean_data) < 2:
            return np.nan
        
        # Calculate running maximum (peak)
        running_max = clean_data.expanding().max()
        
        # Calculate drawdown from peak
        drawdown = (clean_data - running_max) / running_max * 100
        
        # Return maximum drawdown (most negative)
        return drawdown.min()
    
    def analyze_ticker(self, ticker: str, field: str = 'PX_LAST') -> Dict:
        """
        Perform comprehensive analysis of a single ticker.
        
        Args:
            ticker: Bloomberg ticker symbol
            field: Bloomberg field to analyze
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing {ticker} ({field})")
        
        # Test availability first
        availability = self.test_ticker_availability(ticker, field)
        if not availability['available']:
            return {
                'ticker': ticker,
                'field': field,
                'available': False,
                'error': availability['error']
            }
        
        # Fetch historical data
        data = self.get_ticker_data(ticker, field)
        if data.empty:
            return {
                'ticker': ticker,
                'field': field,
                'available': False,
                'error': 'No data available'
            }
        
        # Extract the price series
        price_series = data.iloc[:, 0]  # First column
        
        # Calculate metrics
        start_date = price_series.index[0]
        end_date = price_series.index[-1]
        frequency = 'Monthly'  # We're using monthly data
        
        cagr = self.calculate_cagr(price_series)
        max_dd = self.calculate_max_drawdown(price_series)
        
        # Get ticker name (try to get from Bloomberg)
        try:
            name_data = blp.bdp(ticker, 'NAME')
            name = name_data.iloc[0, 0] if not name_data.empty else ticker
        except:
            name = ticker
        
        result = {
            'ticker': ticker,
            'name': name,
            'field': field,
            'available': True,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'frequency': frequency,
            'data_points': len(price_series),
            'cagr_since_inception': round(cagr, 2) if not np.isnan(cagr) else None,
            'max_drawdown_since_inception': round(max_dd, 2) if not np.isnan(max_dd) else None,
            'start_value': price_series.iloc[0],
            'end_value': price_series.iloc[-1],
            'total_return': round(((price_series.iloc[-1] / price_series.iloc[0]) - 1) * 100, 2)
        }
        
        return result
    
    def explore_asset_class(self, asset_class: str, field: str = 'PX_LAST') -> List[Dict]:
        """
        Explore all tickers in a specific asset class.
        
        Args:
            asset_class: Asset class name from ticker_universe
            field: Bloomberg field to analyze
            
        Returns:
            List of analysis results for each ticker
        """
        if asset_class not in self.ticker_universe:
            logger.error(f"Unknown asset class: {asset_class}")
            return []
        
        tickers = self.ticker_universe[asset_class]
        results = []
        
        logger.info(f"Exploring {len(tickers)} tickers in {asset_class}")
        
        for ticker in tickers:
            try:
                result = self.analyze_ticker(ticker, field)
                result['asset_class'] = asset_class
                results.append(result)
                
                # Store in results
                self.results[f"{ticker}_{field}"] = result
                
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {str(e)}")
                results.append({
                    'ticker': ticker,
                    'asset_class': asset_class,
                    'field': field,
                    'available': False,
                    'error': str(e)
                })
        
        return results
    
    def explore_all_asset_classes(self, field: str = 'PX_LAST') -> Dict[str, List[Dict]]:
        """
        Explore all asset classes.
        
        Args:
            field: Bloomberg field to analyze
            
        Returns:
            Dictionary mapping asset classes to their results
        """
        all_results = {}
        
        for asset_class in self.ticker_universe.keys():
            logger.info(f"Exploring asset class: {asset_class}")
            results = self.explore_asset_class(asset_class, field)
            all_results[asset_class] = results
            
            # Add to summary data
            self.summary_data.extend(results)
        
        return all_results
    
    def generate_summary_table(self, results: List[Dict] = None) -> pd.DataFrame:
        """
        Generate a comprehensive summary table.
        
        Args:
            results: List of analysis results (default: self.summary_data)
            
        Returns:
            DataFrame with summary information
        """
        if results is None:
            results = self.summary_data
        
        if not results:
            logger.warning("No results to summarize")
            return pd.DataFrame()
        
        # Filter for available tickers only
        available_results = [r for r in results if r.get('available', False)]
        
        if not available_results:
            logger.warning("No available tickers found")
            return pd.DataFrame()
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(available_results)
        
        # Select and order columns
        columns = [
            'ticker', 'name', 'asset_class', 'start_date', 'end_date', 
            'frequency', 'data_points', 'cagr_since_inception', 
            'max_drawdown_since_inception', 'total_return'
        ]
        
        summary_df = summary_df[columns]
        
        # Sort by CAGR descending
        summary_df = summary_df.sort_values('cagr_since_inception', ascending=False)
        
        return summary_df
    
    def save_results(self, output_path: str = None) -> str:
        """
        Save results to CSV file.
        
        Args:
            output_path: Path to save results (default: auto-generated)
            
        Returns:
            Path to saved file
        """
        if not self.summary_data:
            logger.warning("No data to save")
            return None
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"tactical asset allocation/tests/ticker_exploration_results_{timestamp}.csv"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate and save summary table
        summary_df = self.generate_summary_table()
        summary_df.to_csv(output_path, index=False)
        
        logger.info(f"Results saved to: {output_path}")
        return output_path
    
    def get_available_tickers_by_asset_class(self) -> Dict[str, List[str]]:
        """
        Get list of available tickers organized by asset class.
        
        Returns:
            Dictionary mapping asset classes to available tickers
        """
        available_by_class = {}
        
        for asset_class, tickers in self.ticker_universe.items():
            available_tickers = []
            
            for ticker in tickers:
                # Check if we have results for this ticker
                for key, result in self.results.items():
                    if result.get('ticker') == ticker and result.get('available', False):
                        available_tickers.append(ticker)
                        break
            
            if available_tickers:
                available_by_class[asset_class] = available_tickers
        
        return available_by_class


def main():
    """
    Main function to run the ticker exploration.
    """
    print("=== Comprehensive Ticker Explorer ===")
    print("Exploring historical data across broad asset classes...")
    
    # Initialize explorer
    explorer = TickerExplorer()
    
    print(f"Date range: {explorer.start_date} to {explorer.end_date}")
    print(f"Total tickers to explore: {sum(len(tickers) for tickers in explorer.ticker_universe.values())}")
    
    # Explore all asset classes
    try:
        all_results = explorer.explore_all_asset_classes()
        
        # Generate summary
        summary_df = explorer.generate_summary_table()
        
        if not summary_df.empty:
            print(f"\n=== SUMMARY TABLE ===")
            print(f"Found {len(summary_df)} available tickers")
            print("\nTop 10 by CAGR:")
            print(summary_df.head(10).to_string(index=False))
            
            print(f"\nBottom 10 by CAGR:")
            print(summary_df.tail(10).to_string(index=False))
            
            # Save results
            output_path = explorer.save_results()
            print(f"\nResults saved to: {output_path}")
            
            # Show asset class breakdown
            print(f"\n=== ASSET CLASS BREAKDOWN ===")
            available_by_class = explorer.get_available_tickers_by_asset_class()
            for asset_class, tickers in available_by_class.items():
                print(f"{asset_class}: {len(tickers)} available tickers")
        
        else:
            print("No available tickers found")
    
    except Exception as e:
        print(f"Error during exploration: {str(e)}")
        logger.error(f"Exploration failed: {str(e)}")


if __name__ == "__main__":
    main()
