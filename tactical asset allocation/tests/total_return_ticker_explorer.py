"""
Total Return Ticker Explorer

This module provides comprehensive exploration of Bloomberg total return indices
and representative mutual fund total return series for tactical asset allocation.

Focus: Total return series only (indices or mutual fund proxies)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TotalReturnAsset:
    """Data class for total return asset information."""
    ticker: str
    name: str
    asset_class: str
    sub_class: str
    region: str
    inception_date: Optional[str] = None
    description: str = ""

class TotalReturnTickerExplorer:
    """
    Explorer for Bloomberg total return indices and mutual fund total return series.
    
    Focuses exclusively on total return data for proper tactical asset allocation analysis.
    """
    
    def __init__(self, start_date: str = "1980-01-01", end_date: str = None):
        """
        Initialize the Total Return Ticker Explorer.
        
        Args:
            start_date: Start date for analysis (default: 1980-01-01)
            end_date: End date for analysis (default: today)
        """
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
        
        # Initialize results storage
        self.results = {}
        self.summary_data = []
        
        # Define comprehensive total return universe
        self.total_return_universe = self._build_total_return_universe()
        
        logger.info(f"Initialized Total Return Ticker Explorer")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        logger.info(f"Total return universe: {len(self.total_return_universe)} assets")
    
    def _build_total_return_universe(self) -> List[TotalReturnAsset]:
        """Build comprehensive universe of total return assets."""
        
        universe = []
        
        # EQUITY TOTAL RETURN INDICES
        equity_indices = [
            TotalReturnAsset("SPXT Index", "S&P 500 Total Return", "Equity", "Large Cap", "US", "1980-01-01", "S&P 500 Total Return Index"),
            TotalReturnAsset("SPTSXT Index", "S&P/TSX Total Return", "Equity", "Large Cap", "Canada", "1980-01-01", "S&P/TSX Composite Total Return Index"),
            TotalReturnAsset("UKXT Index", "FTSE 100 Total Return", "Equity", "Large Cap", "UK", "1980-01-01", "FTSE 100 Total Return Index"),
            TotalReturnAsset("DAXT Index", "DAX Total Return", "Equity", "Large Cap", "Germany", "1980-01-01", "DAX Total Return Index"),
            TotalReturnAsset("NKYT Index", "Nikkei 225 Total Return", "Equity", "Large Cap", "Japan", "1980-01-01", "Nikkei 225 Total Return Index"),
            TotalReturnAsset("MXUST Index", "MSCI USA Total Return", "Equity", "Large Cap", "US", "1980-01-01", "MSCI USA Total Return Index"),
            TotalReturnAsset("MXEUT Index", "MSCI Europe Total Return", "Equity", "Large Cap", "Europe", "1980-01-01", "MSCI Europe Total Return Index"),
            TotalReturnAsset("MXAPT Index", "MSCI Asia Pacific Total Return", "Equity", "Large Cap", "Asia Pacific", "1980-01-01", "MSCI Asia Pacific Total Return Index"),
            TotalReturnAsset("RTYT Index", "Russell 2000 Total Return", "Equity", "Small Cap", "US", "1980-01-01", "Russell 2000 Total Return Index"),
            TotalReturnAsset("NDXT Index", "NASDAQ 100 Total Return", "Equity", "Large Cap", "US", "1980-01-01", "NASDAQ 100 Total Return Index"),
            TotalReturnAsset("MXEF Index", "MSCI Emerging Markets", "Equity", "Large Cap", "Emerging Markets", "1980-01-01", "MSCI Emerging Markets Net Total Return Index"),
            TotalReturnAsset("MXWO Index", "MSCI World", "Equity", "Large Cap", "Global", "1980-01-01", "MSCI World Net Total Return Index"),
        ]
        
        # FIXED INCOME TOTAL RETURN INDICES
        fixed_income_indices = [
            TotalReturnAsset("LUACTRUU Index", "US Investment Grade Corporate TR", "Fixed Income", "Corporate", "US", "1980-01-01", "Bloomberg US Investment Grade Corporate Total Return Index"),
            TotalReturnAsset("LF98TRUU Index", "US High Yield Corporate TR", "Fixed Income", "High Yield", "US", "1980-01-01", "Bloomberg US High Yield Corporate Total Return Index"),
            TotalReturnAsset("I05510CA Index", "Canadian Investment Grade TR", "Fixed Income", "Corporate", "Canada", "1980-01-01", "Bloomberg Canadian Investment Grade Corporate Total Return Index"),
            TotalReturnAsset("GTGBP10Y Index", "UK 10Y Gilt Total Return", "Fixed Income", "Government", "UK", "1980-01-01", "UK 10Y Gilt Total Return Index"),
            TotalReturnAsset("GTDEM10Y Index", "German 10Y Bund Total Return", "Fixed Income", "Government", "Germany", "1980-01-01", "German 10Y Bund Total Return Index"),
            TotalReturnAsset("GTJPY10Y Index", "Japanese 10Y JGB Total Return", "Fixed Income", "Government", "Japan", "1980-01-01", "Japanese 10Y JGB Total Return Index"),
            TotalReturnAsset("USGG10YR Index", "US 10Y Treasury Total Return", "Fixed Income", "Government", "US", "1980-01-01", "US 10Y Treasury Total Return Index"),
            TotalReturnAsset("USGG30YR Index", "US 30Y Treasury Total Return", "Fixed Income", "Government", "US", "1980-01-01", "US 30Y Treasury Total Return Index"),
            TotalReturnAsset("USGG2YR Index", "US 2Y Treasury Total Return", "Fixed Income", "Government", "US", "1980-01-01", "US 2Y Treasury Total Return Index"),
            TotalReturnAsset("USGG5YR Index", "US 5Y Treasury Total Return", "Fixed Income", "Government", "US", "1980-01-01", "US 5Y Treasury Total Return Index"),
        ]
        
        # REAL ESTATE TOTAL RETURN INDICES
        real_estate_indices = [
            TotalReturnAsset("FTSEEPRA Index", "FTSE EPRA/NAREIT Global TR", "Real Estate", "REITs", "Global", "1980-01-01", "FTSE EPRA/NAREIT Global Real Estate Total Return Index"),
            TotalReturnAsset("FTSEEPUS Index", "FTSE EPRA/NAREIT US TR", "Real Estate", "REITs", "US", "1980-01-01", "FTSE EPRA/NAREIT US Real Estate Total Return Index"),
            TotalReturnAsset("FTSEEPEU Index", "FTSE EPRA/NAREIT Europe TR", "Real Estate", "REITs", "Europe", "1980-01-01", "FTSE EPRA/NAREIT Europe Real Estate Total Return Index"),
            TotalReturnAsset("FTSEEPAP Index", "FTSE EPRA/NAREIT Asia Pacific TR", "Real Estate", "REITs", "Asia Pacific", "1980-01-01", "FTSE EPRA/NAREIT Asia Pacific Real Estate Total Return Index"),
        ]
        
        # COMMODITY TOTAL RETURN INDICES
        commodity_indices = [
            TotalReturnAsset("GOLDS Comdty", "Gold Spot", "Commodity", "Precious Metals", "Global", "1980-01-01", "Gold Spot Price (can be converted to TR)"),
            TotalReturnAsset("SILVER Comdty", "Silver Spot", "Commodity", "Precious Metals", "Global", "1980-01-01", "Silver Spot Price (can be converted to TR)"),
            TotalReturnAsset("CRUDE Comdty", "WTI Crude Oil", "Commodity", "Energy", "Global", "1980-01-01", "WTI Crude Oil Spot Price (can be converted to TR)"),
            TotalReturnAsset("BRENT Comdty", "Brent Crude Oil", "Commodity", "Energy", "Global", "1980-01-01", "Brent Crude Oil Spot Price (can be converted to TR)"),
            TotalReturnAsset("BCOM Index", "Bloomberg Commodity Index TR", "Commodity", "Broad", "Global", "1980-01-01", "Bloomberg Commodity Index Total Return"),
            TotalReturnAsset("DJPY Index", "Dow Jones Commodity Index TR", "Commodity", "Broad", "Global", "1980-01-01", "Dow Jones Commodity Index Total Return"),
        ]
        
        # REPRESENTATIVE MUTUAL FUND TOTAL RETURN SERIES
        equity_mutual_funds = [
            TotalReturnAsset("VFINX US Equity", "Vanguard 500 Index Fund", "Equity", "Large Cap", "US", "1976-08-31", "Vanguard 500 Index Fund Total Return"),
            TotalReturnAsset("VTSMX US Equity", "Vanguard Total Stock Market Index Fund", "Equity", "Broad", "US", "1992-04-27", "Vanguard Total Stock Market Index Fund Total Return"),
            TotalReturnAsset("VTSAX US Equity", "Vanguard Total Stock Market Index Fund Admiral", "Equity", "Broad", "US", "2000-11-13", "Vanguard Total Stock Market Index Fund Admiral Total Return"),
            TotalReturnAsset("VEXMX US Equity", "Vanguard Extended Market Index Fund", "Equity", "Mid/Small Cap", "US", "1987-12-31", "Vanguard Extended Market Index Fund Total Return"),
            TotalReturnAsset("VIMAX US Equity", "Vanguard Mid-Cap Index Fund Admiral", "Equity", "Mid Cap", "US", "2011-01-26", "Vanguard Mid-Cap Index Fund Admiral Total Return"),
            TotalReturnAsset("VSMAX US Equity", "Vanguard Small-Cap Index Fund Admiral", "Equity", "Small Cap", "US", "2011-01-26", "Vanguard Small-Cap Index Fund Admiral Total Return"),
            TotalReturnAsset("VTMGX US Equity", "Vanguard Developed Markets Index Fund Admiral", "Equity", "Large Cap", "Developed Markets", "2010-05-26", "Vanguard Developed Markets Index Fund Admiral Total Return"),
            TotalReturnAsset("VEMAX US Equity", "Vanguard Emerging Markets Index Fund Admiral", "Equity", "Large Cap", "Emerging Markets", "2007-05-04", "Vanguard Emerging Markets Index Fund Admiral Total Return"),
        ]
        
        bond_mutual_funds = [
            TotalReturnAsset("VBMFX US Equity", "Vanguard Total Bond Market Index Fund", "Fixed Income", "Broad", "US", "1986-12-11", "Vanguard Total Bond Market Index Fund Total Return"),
            TotalReturnAsset("VBTLX US Equity", "Vanguard Total Bond Market Index Fund Admiral", "Fixed Income", "Broad", "US", "2001-11-13", "Vanguard Total Bond Market Index Fund Admiral Total Return"),
            TotalReturnAsset("VFITX US Equity", "Vanguard Intermediate-Term Treasury Fund", "Fixed Income", "Government", "US", "1991-05-15", "Vanguard Intermediate-Term Treasury Fund Total Return"),
            TotalReturnAsset("VFISX US Equity", "Vanguard Short-Term Treasury Fund", "Fixed Income", "Government", "US", "1991-05-15", "Vanguard Short-Term Treasury Fund Total Return"),
            TotalReturnAsset("VUSTX US Equity", "Vanguard Long-Term Treasury Fund", "Fixed Income", "Government", "US", "1986-12-11", "Vanguard Long-Term Treasury Fund Total Return"),
            TotalReturnAsset("VWEIX US Equity", "Vanguard Long-Term Investment-Grade Fund", "Fixed Income", "Corporate", "US", "1993-06-01", "Vanguard Long-Term Investment-Grade Fund Total Return"),
            TotalReturnAsset("VWESX US Equity", "Vanguard Long-Term Investment-Grade Fund Admiral", "Fixed Income", "Corporate", "US", "2001-11-13", "Vanguard Long-Term Investment-Grade Fund Admiral Total Return"),
        ]
        
        # Combine all assets
        universe.extend(equity_indices)
        universe.extend(fixed_income_indices)
        universe.extend(real_estate_indices)
        universe.extend(commodity_indices)
        universe.extend(equity_mutual_funds)
        universe.extend(bond_mutual_funds)
        
        return universe
    
    def get_available_assets(self, start_date: str = None) -> List[TotalReturnAsset]:
        """
        Get assets available from a specific start date.
        
        Args:
            start_date: Start date to check availability
            
        Returns:
            List of available assets
        """
        if start_date is None:
            start_date = self.start_date
        else:
            start_date = pd.Timestamp(start_date)
        
        available = []
        for asset in self.total_return_universe:
            if asset.inception_date is None or pd.Timestamp(asset.inception_date) <= start_date:
                available.append(asset)
        
        return available
    
    def analyze_total_return_asset(self, asset: TotalReturnAsset, field: str = 'PX_LAST') -> Dict:
        """
        Analyze a total return asset.
        
        Args:
            asset: TotalReturnAsset object
            field: Bloomberg field to analyze
            
        Returns:
            Analysis results dictionary
        """
        try:
            # Test availability first
            available = self.test_ticker_availability(asset.ticker, field)
            
            if not available:
                return {
                    'ticker': asset.ticker,
                    'name': asset.name,
                    'asset_class': asset.asset_class,
                    'sub_class': asset.sub_class,
                    'region': asset.region,
                    'available': False,
                    'error': 'No data available'
                }
            
            # Get data
            data = self.get_ticker_data(asset.ticker, field)
            
            if data is None or data.empty:
                return {
                    'ticker': asset.ticker,
                    'name': asset.name,
                    'asset_class': asset.asset_class,
                    'sub_class': asset.sub_class,
                    'region': asset.region,
                    'available': False,
                    'error': 'No data returned'
                }
            
            # Calculate metrics
            cagr = self.calculate_cagr(data[field])
            max_dd = self.calculate_max_drawdown(data[field])
            
            return {
                'ticker': asset.ticker,
                'name': asset.name,
                'asset_class': asset.asset_class,
                'sub_class': asset.sub_class,
                'region': asset.region,
                'start_date': data.index.min().strftime('%Y-%m-%d'),
                'end_date': data.index.max().strftime('%Y-%m-%d'),
                'frequency': 'Monthly',
                'cagr_since_inception': cagr,
                'max_drawdown_since_inception': max_dd,
                'data_points': len(data),
                'total_return': (data[field].iloc[-1] / data[field].iloc[0] - 1) * 100,
                'available': True
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {asset.ticker}: {str(e)}")
            return {
                'ticker': asset.ticker,
                'name': asset.name,
                'asset_class': asset.asset_class,
                'sub_class': asset.sub_class,
                'region': asset.region,
                'available': False,
                'error': str(e)
            }
    
    def test_ticker_availability(self, ticker: str, field: str = 'PX_LAST') -> bool:
        """
        Test if ticker data is available.
        
        Args:
            ticker: Bloomberg ticker
            field: Bloomberg field
            
        Returns:
            True if data is available
        """
        try:
            # Mock implementation for testing
            # In real implementation, this would use xbbg to test availability
            return True
        except Exception as e:
            logger.error(f"Error testing availability for {ticker}: {str(e)}")
            return False
    
    def get_ticker_data(self, ticker: str, field: str = 'PX_LAST') -> Optional[pd.DataFrame]:
        """
        Get ticker data from Bloomberg.
        
        Args:
            ticker: Bloomberg ticker
            field: Bloomberg field
            
        Returns:
            DataFrame with ticker data
        """
        try:
            # Mock implementation for testing
            # In real implementation, this would use xbbg to fetch data
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='ME')
            np.random.seed(42)  # For reproducible results
            prices = 100 * np.cumprod(1 + np.random.normal(0.01, 0.15, len(dates)))
            
            data = pd.DataFrame({
                field: prices
            }, index=dates)
            
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def calculate_cagr(self, price_series: pd.Series) -> float:
        """
        Calculate Compound Annual Growth Rate.
        
        Args:
            price_series: Price series
            
        Returns:
            CAGR as percentage
        """
        if len(price_series) < 2:
            return 0.0
        
        start_price = price_series.iloc[0]
        end_price = price_series.iloc[-1]
        
        # Calculate years
        start_date = price_series.index[0]
        end_date = price_series.index[-1]
        years = (end_date - start_date).days / 365.25
        
        if years <= 0:
            return 0.0
        
        # Calculate CAGR
        cagr = ((end_price / start_price) ** (1 / years) - 1) * 100
        return round(cagr, 2)
    
    def calculate_max_drawdown(self, price_series: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            price_series: Price series
            
        Returns:
            Maximum drawdown as percentage
        """
        if len(price_series) < 2:
            return 0.0
        
        # Calculate running maximum
        running_max = price_series.expanding().max()
        
        # Calculate drawdown
        drawdown = (price_series - running_max) / running_max * 100
        
        # Return minimum (most negative) drawdown
        max_dd = drawdown.min()
        return round(max_dd, 2)
    
    def generate_total_return_summary(self, results: List[Dict] = None) -> pd.DataFrame:
        """
        Generate summary table for total return analysis.
        
        Args:
            results: List of analysis results
            
        Returns:
            DataFrame with summary
        """
        if results is None:
            results = self.summary_data
        
        if not results:
            logger.warning("No results to summarize")
            return pd.DataFrame()
        
        # Filter for available assets only
        available_results = [r for r in results if r.get('available', False)]
        
        if not available_results:
            logger.warning("No available assets found")
            return pd.DataFrame()
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(available_results)
        
        # Select columns for summary table
        columns = [
            'ticker', 'name', 'asset_class', 'sub_class', 'region',
            'start_date', 'end_date', 'frequency', 'cagr_since_inception', 
            'max_drawdown_since_inception', 'data_points', 'total_return'
        ]
        
        # Ensure all columns exist
        for col in columns:
            if col not in summary_df.columns:
                summary_df[col] = None
        
        summary_df = summary_df[columns]
        
        # Sort by CAGR descending
        summary_df = summary_df.sort_values('cagr_since_inception', ascending=False)
        
        return summary_df
    
    def save_total_return_results(self, output_path: str = None) -> str:
        """
        Save total return analysis results to CSV file.
        
        Args:
            output_path: Path to save results
            
        Returns:
            Path to saved file
        """
        if not self.summary_data:
            logger.warning("No data to save")
            return None
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"tactical asset allocation/tests/total_return_results_{timestamp}.csv"
        
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate and save summary table
        summary_df = self.generate_total_return_summary()
        summary_df.to_csv(output_path, index=False)
        
        logger.info(f"Total return results saved to: {output_path}")
        return output_path


def explore_total_return_universe(start_date: str = "1980-01-01", end_date: str = None) -> pd.DataFrame:
    """
    Explore the complete total return universe.
    
    Args:
        start_date: Start date for analysis
        end_date: End date for analysis
        
    Returns:
        DataFrame with results
    """
    explorer = TotalReturnTickerExplorer(start_date, end_date)
    
    logger.info(f"Exploring total return universe from {start_date}")
    
    results = []
    for asset in explorer.total_return_universe:
        try:
            result = explorer.analyze_total_return_asset(asset)
            results.append(result)
        except Exception as e:
            logger.error(f"Error analyzing {asset.ticker}: {str(e)}")
            results.append({
                'ticker': asset.ticker,
                'name': asset.name,
                'asset_class': asset.asset_class,
                'sub_class': asset.sub_class,
                'region': asset.region,
                'available': False,
                'error': str(e)
            })
    
    # Generate summary
    explorer.summary_data = results
    summary_df = explorer.generate_total_return_summary()
    
    return summary_df


if __name__ == "__main__":
    # Example usage
    print("=== Total Return Ticker Explorer ===")
    
    explorer = TotalReturnTickerExplorer()
    
    print(f"Total return universe: {len(explorer.total_return_universe)} assets")
    print(f"Available from 1980: {len(explorer.get_available_assets('1980-01-01'))} assets")
    
    # Show asset breakdown by class
    asset_classes = {}
    for asset in explorer.total_return_universe:
        if asset.asset_class not in asset_classes:
            asset_classes[asset.asset_class] = 0
        asset_classes[asset.asset_class] += 1
    
    print("\nAsset class breakdown:")
    for asset_class, count in asset_classes.items():
        print(f"  {asset_class}: {count} assets")
