"""
Comprehensive Total Return Analysis Script

This script analyzes 55 total return series starting from 1980 to today,
providing detailed performance metrics in a nicely formatted table.

Output includes:
- Ticker, Name, Start/Inception Date, End Date
- Smallest Frequency, CAGR since inception
- YTD Return, 3yr CAGR, 5yr CAGR, 10yr CAGR
- Max Drawdown Since Inception
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import Bloomberg data fetching functionality
from xbbg import blp

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

class ComprehensiveTotalReturnAnalyzer:
    """
    Comprehensive analyzer for 55 total return series starting from 1980.
    """
    
    def __init__(self, start_date: str = "1980-01-01", end_date: str = None):
        """
        Initialize the Comprehensive Total Return Analyzer.
        
        Args:
            start_date: Start date for analysis (default: 1980-01-01)
            end_date: End date for analysis (default: today)
        """
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
        
        # Initialize results storage
        self.results = {}
        self.summary_data = []
        
        # Define comprehensive total return universe (55 assets)
        self.total_return_universe = self._build_comprehensive_universe()
        
        logger.info(f"Initialized Comprehensive Total Return Analyzer")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        logger.info(f"Total return universe: {len(self.total_return_universe)} assets")
    
    def _build_comprehensive_universe(self) -> List[TotalReturnAsset]:
        """Build comprehensive universe of 55 total return assets."""
        
        universe = []
        
        # EQUITY TOTAL RETURN INDICES (20 assets)
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
            TotalReturnAsset("VFINX US Equity", "Vanguard 500 Index Fund", "Equity", "Large Cap", "US", "1976-08-31", "Vanguard 500 Index Fund Total Return"),
            TotalReturnAsset("VTSMX US Equity", "Vanguard Total Stock Market Index Fund", "Equity", "Broad", "US", "1992-04-27", "Vanguard Total Stock Market Index Fund Total Return"),
            TotalReturnAsset("VTSAX US Equity", "Vanguard Total Stock Market Index Fund Admiral", "Equity", "Broad", "US", "2000-11-13", "Vanguard Total Stock Market Index Fund Admiral Total Return"),
            TotalReturnAsset("VEXMX US Equity", "Vanguard Extended Market Index Fund", "Equity", "Mid/Small Cap", "US", "1987-12-31", "Vanguard Extended Market Index Fund Total Return"),
            TotalReturnAsset("VIMAX US Equity", "Vanguard Mid-Cap Index Fund Admiral", "Equity", "Mid Cap", "US", "2011-01-26", "Vanguard Mid-Cap Index Fund Admiral Total Return"),
            TotalReturnAsset("VSMAX US Equity", "Vanguard Small-Cap Index Fund Admiral", "Equity", "Small Cap", "US", "2011-01-26", "Vanguard Small-Cap Index Fund Admiral Total Return"),
            TotalReturnAsset("VTMGX US Equity", "Vanguard Developed Markets Index Fund Admiral", "Equity", "Large Cap", "Developed Markets", "2010-05-26", "Vanguard Developed Markets Index Fund Admiral Total Return"),
            TotalReturnAsset("VEMAX US Equity", "Vanguard Emerging Markets Index Fund Admiral", "Equity", "Large Cap", "Emerging Markets", "2007-05-04", "Vanguard Emerging Markets Index Fund Admiral Total Return"),
        ]
        
        # FIXED INCOME TOTAL RETURN INDICES (17 assets)
        fixed_income_indices = [
            TotalReturnAsset("LUACTRUU Index", "US Investment Grade Corporate TR", "Fixed Income", "Corporate", "US", "1980-01-01", "Bloomberg US Investment Grade Corporate Total Return Index"),
            TotalReturnAsset("LF98TRUU Index", "US High Yield Corporate TR", "Fixed Income", "High Yield", "US", "1980-01-01", "Bloomberg US High Yield Corporate Total Return Index"),
            TotalReturnAsset("I05510CA Index", "Canadian Investment Grade TR", "Fixed Income", "Corporate", "Canada", "1980-01-01", "Bloomberg Canadian Investment Grade Corporate Total Return Index"),
            TotalReturnAsset("GTGBP10Y Index", "UK 10Y Gilt Total Return", "Fixed Income", "Government", "UK", "1980-01-01", "UK 10Y Gilt Total Return Index"),
            TotalReturnAsset("GTDEM10Y Index", "German 10Y Bund Total Return", "Fixed Income", "Government", "Germany", "1980-01-01", "German 10Y Bund Total Return Index"),
            TotalReturnAsset("GTJPY10Y Index", "Japanese 10Y JGB Total Return", "Fixed Income", "Government", "Japan", "1980-01-01", "Japanese 10Y JGB Total Return Index"),
            # Note: USGG*TR Index are aliases for price indices, not true total return indices
            # Using mutual fund proxies instead for proper total return data
            TotalReturnAsset("VBMFX US Equity", "Vanguard Total Bond Market Index Fund", "Fixed Income", "Broad", "US", "1986-12-11", "Vanguard Total Bond Market Index Fund Total Return"),
            TotalReturnAsset("VBTLX US Equity", "Vanguard Total Bond Market Index Fund Admiral", "Fixed Income", "Broad", "US", "2001-11-13", "Vanguard Total Bond Market Index Fund Admiral Total Return"),
            TotalReturnAsset("VFITX US Equity", "Vanguard Intermediate-Term Treasury Fund", "Fixed Income", "Government", "US", "1991-05-15", "Vanguard Intermediate-Term Treasury Fund Total Return"),
            TotalReturnAsset("VFISX US Equity", "Vanguard Short-Term Treasury Fund", "Fixed Income", "Government", "US", "1991-05-15", "Vanguard Short-Term Treasury Fund Total Return"),
            TotalReturnAsset("VUSTX US Equity", "Vanguard Long-Term Treasury Fund", "Fixed Income", "Government", "US", "1986-12-11", "Vanguard Long-Term Treasury Fund Total Return"),
            TotalReturnAsset("VWEIX US Equity", "Vanguard Long-Term Investment-Grade Fund", "Fixed Income", "Corporate", "US", "1993-06-01", "Vanguard Long-Term Investment-Grade Fund Total Return"),
            TotalReturnAsset("VWESX US Equity", "Vanguard Long-Term Investment-Grade Fund Admiral", "Fixed Income", "Corporate", "US", "2001-11-13", "Vanguard Long-Term Investment-Grade Fund Admiral Total Return"),
        ]
        
        # REAL ESTATE TOTAL RETURN INDICES (4 assets)
        real_estate_indices = [
            TotalReturnAsset("FTSEEPRA Index", "FTSE EPRA/NAREIT Global TR", "Real Estate", "REITs", "Global", "1980-01-01", "FTSE EPRA/NAREIT Global Real Estate Total Return Index"),
            TotalReturnAsset("FTSEEPUS Index", "FTSE EPRA/NAREIT US TR", "Real Estate", "REITs", "US", "1980-01-01", "FTSE EPRA/NAREIT US Real Estate Total Return Index"),
            TotalReturnAsset("FTSEEPEU Index", "FTSE EPRA/NAREIT Europe TR", "Real Estate", "REITs", "Europe", "1980-01-01", "FTSE EPRA/NAREIT Europe Real Estate Total Return Index"),
            TotalReturnAsset("FTSEEPAP Index", "FTSE EPRA/NAREIT Asia Pacific TR", "Real Estate", "REITs", "Asia Pacific", "1980-01-01", "FTSE EPRA/NAREIT Asia Pacific Real Estate Total Return Index"),
        ]
        
        # COMMODITY TOTAL RETURN INDICES (6 assets)
        commodity_indices = [
            TotalReturnAsset("GOLDS Comdty", "Gold Spot", "Commodity", "Precious Metals", "Global", "1980-01-01", "Gold Spot Price (convertible to TR)"),
            TotalReturnAsset("SILVER Comdty", "Silver Spot", "Commodity", "Precious Metals", "Global", "1980-01-01", "Silver Spot Price (convertible to TR)"),
            TotalReturnAsset("CRUDE Comdty", "WTI Crude Oil", "Commodity", "Energy", "Global", "1980-01-01", "WTI Crude Oil Spot Price (convertible to TR)"),
            TotalReturnAsset("BRENT Comdty", "Brent Crude Oil", "Commodity", "Energy", "Global", "1980-01-01", "Brent Crude Oil Spot Price (convertible to TR)"),
            TotalReturnAsset("BCOM Index", "Bloomberg Commodity Index TR", "Commodity", "Broad", "Global", "1980-01-01", "Bloomberg Commodity Index Total Return"),
            TotalReturnAsset("DJPY Index", "Dow Jones Commodity Index TR", "Commodity", "Broad", "Global", "1980-01-01", "Dow Jones Commodity Index Total Return"),
        ]
        
        # CASH/EQUIVALENT TOTAL RETURN INDICES (Note: USGG*TR Index are aliases for price indices)
        # Using international government bill indices and removing fake USGG*TR indices
        cash_equivalent_indices = [
            # Note: USGG*TR Index are aliases for price indices, not true total return indices
            # Keeping only international indices that may be legitimate
            TotalReturnAsset("GTGBP3MTR Index", "UK 3M Gilt Total Return", "Cash", "Government Bills", "UK", "1980-01-01", "UK 3M Gilt Total Return Index"),
            TotalReturnAsset("GTDEM3MTR Index", "German 3M Bund Total Return", "Cash", "Government Bills", "Germany", "1980-01-01", "German 3M Bund Total Return Index"),
            TotalReturnAsset("GTJPY3MTR Index", "Japanese 3M JGB Total Return", "Cash", "Government Bills", "Japan", "1980-01-01", "Japanese 3M JGB Total Return Index"),
        ]
        
        # Combine all assets (reduced from 55 to ~40 after removing fake USGG*TR indices)
        universe.extend(equity_indices)      # 20 assets
        universe.extend(fixed_income_indices)  # 13 assets (removed 4 fake USGG*TR)
        universe.extend(real_estate_indices)   # 4 assets
        universe.extend(commodity_indices)     # 6 assets
        universe.extend(cash_equivalent_indices)  # 3 assets (removed 5 fake USGG*TR)
        
        return universe
    
    def analyze_comprehensive_universe(self) -> List[Dict]:
        """
        Analyze the comprehensive universe of total return assets.
        
        Returns:
            List of analysis results
        """
        results = []
        
        logger.info(f"Analyzing {len(self.total_return_universe)} total return assets")
        
        for i, asset in enumerate(self.total_return_universe, 1):
            logger.info(f"Processing {i}/{len(self.total_return_universe)}: {asset.ticker}")
            
            try:
                result = self.analyze_total_return_asset(asset)
                results.append(result)
                
                # Store in results
                self.results[f"{asset.ticker}"] = result
                
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
        
        return results
    
    def analyze_total_return_asset(self, asset: TotalReturnAsset) -> Dict:
        """
        Analyze a total return asset with comprehensive metrics.
        
        Args:
            asset: TotalReturnAsset object
            
        Returns:
            Analysis results dictionary
        """
        try:
            # Get data
            data = self.get_ticker_data(asset.ticker)
            
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
            
            # Calculate comprehensive metrics
            start_date = data.index.min()
            end_date = data.index.max()
            frequency = self.determine_frequency(data)
            
            # CAGR since inception
            cagr_since_inception = self.calculate_cagr(data['price'], start_date, end_date)
            
            # YTD Return
            ytd_return = self.calculate_ytd_return(data['price'])
            
            # 3yr CAGR
            cagr_3yr = self.calculate_period_cagr(data['price'], years=3)
            
            # 5yr CAGR
            cagr_5yr = self.calculate_period_cagr(data['price'], years=5)
            
            # 10yr CAGR
            cagr_10yr = self.calculate_period_cagr(data['price'], years=10)
            
            # Max drawdown since inception
            max_drawdown = self.calculate_max_drawdown(data['price'])
            
            return {
                'ticker': asset.ticker,
                'name': asset.name,
                'asset_class': asset.asset_class,
                'sub_class': asset.sub_class,
                'region': asset.region,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'frequency': frequency,
                'cagr_since_inception': cagr_since_inception,
                'ytd_return': ytd_return,
                'cagr_3yr': cagr_3yr,
                'cagr_5yr': cagr_5yr,
                'cagr_10yr': cagr_10yr,
                'max_drawdown_since_inception': max_drawdown,
                'data_points': len(data),
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
    
    def get_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Get ticker data from Bloomberg API.
        
        Args:
            ticker: Bloomberg ticker
            
        Returns:
            DataFrame with ticker data
        """
        try:
            logger.info(f"Fetching Bloomberg data for {ticker}")
            
            # Try different fields for total return data - prioritize total return fields
            fields_to_try = ['TOT_RETURN_INDEX_NET_DVDS', 'TOT_RETURN_INDEX', 'PX_LAST']
            
            for field in fields_to_try:
                try:
                    # Fetch data from Bloomberg
                    data = blp.bdh(
                        ticker,
                        field,
                        self.start_date.strftime('%Y-%m-%d'),
                        self.end_date.strftime('%Y-%m-%d'),
                        Per='DAILY'
                    )
                    
                    if not data.empty and not data.isnull().all().all():
                        # Convert to DataFrame with proper column name
                        df = pd.DataFrame({
                            'price': data.iloc[:, 0]  # Use first column
                        })
                        df.index.name = 'Date'
                        
                        # Ensure index is properly formatted as DatetimeIndex
                        if not isinstance(df.index, pd.DatetimeIndex):
                            df.index = pd.to_datetime(df.index)
                        
                        # Validate the fetched data
                        self._validate_bloomberg_data(df, ticker, field)
                        
                        logger.info(f"Successfully fetched {ticker} data using field {field}")
                        return df
                        
                except Exception as e:
                    logger.debug(f"Failed to fetch {ticker} with field {field}: {str(e)}")
                    continue
            
            # If all fields fail, try with basic price data
            try:
                logger.warning(f"All total return fields failed for {ticker}, trying basic price data")
                data = blp.bdh(
                    ticker,
                    'PX_LAST',
                    self.start_date.strftime('%Y-%m-%d'),
                    self.end_date.strftime('%Y-%m-%d'),
                    Per='DAILY'
                )
                
                if not data.empty and not data.isnull().all().all():
                    df = pd.DataFrame({
                        'price': data.iloc[:, 0]
                    })
                    df.index.name = 'Date'
                    
                    # Ensure index is properly formatted as DatetimeIndex
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    
                    logger.warning(f"Using basic price data for {ticker} (not total return)")
                    return df
                    
            except Exception as e:
                logger.error(f"Failed to fetch any data for {ticker}: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def _validate_bloomberg_data(self, data: pd.DataFrame, ticker: str, field: str) -> None:
        """
        Validate Bloomberg data for quality and completeness.
        
        Args:
            data: Bloomberg price data
            ticker: Bloomberg ticker
            field: Bloomberg field used
        """
        try:
            # Basic data quality checks
            if data.empty:
                logger.error(f"{ticker}: No data returned from Bloomberg")
                return
            
            if data['price'].isnull().all():
                logger.error(f"{ticker}: All price data is null")
                return
            
            # Check data completeness
            total_days = len(data)
            valid_days = data['price'].notna().sum()
            completeness = (valid_days / total_days) * 100
            
            if completeness < 50:
                logger.warning(f"{ticker}: Low data completeness ({completeness:.1f}%)")
            
            # Check for extreme values
            price_data = data['price'].dropna()
            if len(price_data) > 0:
                min_price = price_data.min()
                max_price = price_data.max()
                
                # Check for negative prices (shouldn't happen for indices/funds)
                if min_price < 0:
                    logger.error(f"{ticker}: Negative prices detected (min: {min_price})")
                
                # Check for extremely high prices
                if max_price > 1000000:
                    logger.warning(f"{ticker}: Very high prices detected (max: {max_price})")
                
                # Check for zero prices
                if min_price == 0:
                    logger.warning(f"{ticker}: Zero prices detected")
            
            # Calculate basic statistics
            if len(price_data) > 1:
                daily_returns = price_data.pct_change().dropna()
                if len(daily_returns) > 0:
                    annual_return = daily_returns.mean() * 252 * 100
                    annual_volatility = daily_returns.std() * np.sqrt(252) * 100
                    
                    # Calculate YTD return
                    current_year = pd.Timestamp.now().year
                    year_start = pd.Timestamp(f"{current_year}-01-01")
                    ytd_data = data[data.index >= year_start]
                    if len(ytd_data) > 1:
                        ytd_return = ((ytd_data['price'].iloc[-1] / ytd_data['price'].iloc[0]) - 1) * 100
                    else:
                        ytd_return = 0
                    
                    logger.info(f"{ticker} ({field}): {valid_days}/{total_days} days ({completeness:.1f}%), "
                              f"Annual return: {annual_return:.2f}%, YTD: {ytd_return:.2f}%, Vol: {annual_volatility:.2f}%")
            
        except Exception as e:
            logger.error(f"Error validating Bloomberg data for {ticker}: {str(e)}")
    
    def determine_frequency(self, data: pd.DataFrame) -> str:
        """
        Determine the smallest frequency of the data.
        
        Args:
            data: DataFrame with datetime index
            
        Returns:
            Frequency string
        """
        if len(data) < 2:
            return "Unknown"
        
        # Calculate time differences
        time_diffs = data.index.to_series().diff().dropna()
        
        # Get the most common time difference
        most_common_diff = time_diffs.mode().iloc[0] if not time_diffs.mode().empty else time_diffs.iloc[0]
        
        if most_common_diff <= pd.Timedelta(days=1):
            return "Daily"
        elif most_common_diff <= pd.Timedelta(days=7):
            return "Weekly"
        elif most_common_diff <= pd.Timedelta(days=31):
            return "Monthly"
        elif most_common_diff <= pd.Timedelta(days=90):
            return "Quarterly"
        else:
            return "Annual"
    
    def calculate_cagr(self, price_series: pd.Series, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
        """
        Calculate Compound Annual Growth Rate.
        
        Args:
            price_series: Price series
            start_date: Start date
            end_date: End date
            
        Returns:
            CAGR as percentage
        """
        if len(price_series) < 2:
            return 0.0
        
        start_price = price_series.iloc[0]
        end_price = price_series.iloc[-1]
        
        # Calculate years
        years = (end_date - start_date).days / 365.25
        
        if years <= 0:
            return 0.0
        
        # Calculate CAGR
        cagr = ((end_price / start_price) ** (1 / years) - 1) * 100
        return round(cagr, 2)
    
    def calculate_ytd_return(self, price_series: pd.Series) -> float:
        """
        Calculate Year-to-Date return.
        
        Args:
            price_series: Price series
            
        Returns:
            YTD return as percentage
        """
        if len(price_series) < 2:
            return 0.0
        
        # Get current year start
        current_year = pd.Timestamp.now().year
        year_start = pd.Timestamp(f"{current_year}-01-01")
        
        # Find closest date to year start
        year_start_data = price_series[price_series.index >= year_start]
        if year_start_data.empty:
            return 0.0
        
        start_price = year_start_data.iloc[0]
        end_price = price_series.iloc[-1]
        
        ytd_return = ((end_price / start_price) - 1) * 100
        return round(ytd_return, 2)
    
    def calculate_period_cagr(self, price_series: pd.Series, years: int) -> float:
        """
        Calculate CAGR for a specific period.
        
        Args:
            price_series: Price series
            years: Number of years for the period
            
        Returns:
            CAGR as percentage
        """
        if len(price_series) < 2:
            return 0.0
        
        # Calculate cutoff date
        cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=years)
        
        # Find data from cutoff date onwards
        period_data = price_series[price_series.index >= cutoff_date]
        if len(period_data) < 2:
            return 0.0
        
        start_price = period_data.iloc[0]
        end_price = period_data.iloc[-1]
        
        # Calculate actual years
        actual_years = (period_data.index[-1] - period_data.index[0]).days / 365.25
        
        if actual_years <= 0:
            return 0.0
        
        cagr = ((end_price / start_price) ** (1 / actual_years) - 1) * 100
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
    
    def generate_comprehensive_table(self, results: List[Dict] = None) -> pd.DataFrame:
        """
        Generate comprehensive table with all requested metrics.
        
        Args:
            results: List of analysis results
            
        Returns:
            DataFrame with comprehensive table
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
        
        # Create comprehensive DataFrame
        df = pd.DataFrame(available_results)
        
        # Select and rename columns for the final table
        columns_mapping = {
            'ticker': 'Ticker',
            'name': 'Name',
            'start_date': 'Start/Inception Date',
            'end_date': 'End Date',
            'frequency': 'Smallest Frequency',
            'cagr_since_inception': 'CAGR Since Inception (%)',
            'ytd_return': 'YTD Return (%)',
            'cagr_3yr': '3yr CAGR (%)',
            'cagr_5yr': '5yr CAGR (%)',
            'cagr_10yr': '10yr CAGR (%)',
            'max_drawdown_since_inception': 'Max Drawdown Since Inception (%)'
        }
        
        # Ensure all columns exist
        for col in columns_mapping.keys():
            if col not in df.columns:
                df[col] = None
        
        # Select and rename columns
        final_df = df[list(columns_mapping.keys())].copy()
        final_df.columns = list(columns_mapping.values())
        
        # Sort by CAGR since inception descending
        final_df = final_df.sort_values('CAGR Since Inception (%)', ascending=False)
        
        return final_df
    
    def save_comprehensive_results(self, output_path: str = None) -> str:
        """
        Save comprehensive analysis results to CSV file.
        
        Args:
            output_path: Path to save results
            
        Returns:
            Path to saved file
        """
        if not self.summary_data:
            logger.warning("No data to save")
            return None
        
        if output_path is None:
            # Save in the tests directory
            output_path = os.path.join(os.path.dirname(__file__), "comprehensive_total_return_analysis.csv")
        
        # Ensure directory exists (only if there's a directory path)
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate and save comprehensive table
        comprehensive_df = self.generate_comprehensive_table()
        comprehensive_df.to_csv(output_path, index=False)
        
        logger.info(f"Comprehensive results saved to: {output_path}")
        return output_path


def main():
    """
    Main function to run comprehensive total return analysis.
    """
    print("=" * 80)
    print("COMPREHENSIVE TOTAL RETURN ANALYSIS")
    print("55 Total Return Series | 1980 Start Date to Today")
    print("=" * 80)
    print()
    
    # Initialize analyzer
    analyzer = ComprehensiveTotalReturnAnalyzer(start_date="1980-01-01")
    
    print(f"Analyzing {len(analyzer.total_return_universe)} total return assets")
    print(f"Date range: {analyzer.start_date.strftime('%Y-%m-%d')} to {analyzer.end_date.strftime('%Y-%m-%d')}")
    print()
    
    try:
        # Analyze comprehensive universe
        results = analyzer.analyze_comprehensive_universe()
        
        # Store results in analyzer
        analyzer.summary_data = results
        
        # Generate comprehensive table
        comprehensive_df = analyzer.generate_comprehensive_table()
        
        if not comprehensive_df.empty:
            print("=" * 80)
            print("COMPREHENSIVE TOTAL RETURN ANALYSIS RESULTS")
            print("=" * 80)
            print()
            
            # Display the table
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', 30)
            
            print(comprehensive_df.to_string(index=False))
            print()
            
            # Save results
            output_path = analyzer.save_comprehensive_results()
            print(f"Results saved to: {output_path}")
            print()
            
            # Show summary statistics
            print("=" * 80)
            print("SUMMARY STATISTICS")
            print("=" * 80)
            print()
            
            print(f"Total Assets Analyzed: {len(comprehensive_df)}")
            print(f"Average CAGR Since Inception: {comprehensive_df['CAGR Since Inception (%)'].mean():.2f}%")
            print(f"Median CAGR Since Inception: {comprehensive_df['CAGR Since Inception (%)'].median():.2f}%")
            print(f"Average YTD Return: {comprehensive_df['YTD Return (%)'].mean():.2f}%")
            print(f"Average 3yr CAGR: {comprehensive_df['3yr CAGR (%)'].mean():.2f}%")
            print(f"Average 5yr CAGR: {comprehensive_df['5yr CAGR (%)'].mean():.2f}%")
            print(f"Average 10yr CAGR: {comprehensive_df['10yr CAGR (%)'].mean():.2f}%")
            print(f"Average Max Drawdown: {comprehensive_df['Max Drawdown Since Inception (%)'].mean():.2f}%")
            print()
            
            # Asset class breakdown
            print("ASSET CLASS BREAKDOWN:")
            asset_class_counts = {}
            for asset in analyzer.total_return_universe:
                if asset.asset_class not in asset_class_counts:
                    asset_class_counts[asset.asset_class] = 0
                asset_class_counts[asset.asset_class] += 1
            
            for asset_class, count in asset_class_counts.items():
                print(f"  {asset_class}: {count} assets")
            
        else:
            print("No available assets found")
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        logger.error(f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    main()
