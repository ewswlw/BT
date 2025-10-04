"""
SPXL Synthetic Series Builder

Constructs a synthetic triple-leveraged S&P 500 series extending back to the 1970s
by replicating the SPXL ETF methodology, then splicing with actual SPXL data.

Exact Methodology (per prospectus):
-----------------------------------
1. Track 3x daily PRICE return of S&P 500 Index (not total return)
2. Daily rebalancing with leverage reset
3. Costs deducted:
   - Expense ratio: 0.91% annually
   - Financing cost: (Fed Funds + 40bps) annually
4. Splice at SPXL inception: November 5, 2008
5. Dividends: Ignored (tracks price index only)

Author: Created for YTM Capital
Date: 2025-09-29
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Tuple, Optional
from xbbg import blp
import quantstats as qs

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#############################################################
# CONFIGURATION
#############################################################

# SPXL ETF Parameters
SPXL_INCEPTION_DATE = '2008-11-05'
SPXL_TICKER = 'SPXL US Equity'
SPXL_EXPENSE_RATIO = 0.0091  # 0.91% annually
LEVERAGE_FACTOR = 3.0
FINANCING_SPREAD = 0.0060  # 60 basis points

# Bloomberg Tickers
SPX_TICKER = 'SPX Index'  # S&P 500 Price Return Index (NOT total return)
FED_FUNDS_TICKER = 'FEDL01 Index'  # Federal Funds Effective Rate

# Date Range
SYNTHETIC_START_DATE = '1970-01-02'
END_DATE = datetime.now().strftime('%Y-%m-%d')

# Trading days per year
TRADING_DAYS_PER_YEAR = 252

# Initial NAV
INITIAL_NAV = 100.0

# Output settings
OUTPUT_DIR = 'processed'
OUTPUT_FILE = 'spxl_synthetic_series.csv'

#############################################################
# CORE CLASS
#############################################################

class SPXLSyntheticBuilder:
    """
    Builder class for creating synthetic SPXL series using exact prospectus methodology.
    """
    
    def __init__(
        self,
        synthetic_start: str = SYNTHETIC_START_DATE,
        spxl_inception: str = SPXL_INCEPTION_DATE,
        end_date: str = END_DATE,
        leverage: float = LEVERAGE_FACTOR,
        expense_ratio: float = SPXL_EXPENSE_RATIO,
        financing_spread: float = FINANCING_SPREAD,
        initial_nav: float = INITIAL_NAV
    ):
        """
        Initialize the SPXL synthetic series builder.
        
        Args:
            synthetic_start: Start date for synthetic series
            spxl_inception: SPXL ETF inception date
            end_date: End date for data fetch
            leverage: Leverage factor (3.0 for SPXL)
            expense_ratio: Annual expense ratio (0.0091 for SPXL)
            financing_spread: Spread over Fed Funds (0.0040 = 40bps)
            initial_nav: Starting NAV value for synthetic series
        """
        self.synthetic_start = synthetic_start
        self.spxl_inception = spxl_inception
        self.end_date = end_date
        self.leverage = leverage
        self.expense_ratio = expense_ratio
        self.financing_spread = financing_spread
        self.initial_nav = initial_nav
        
        # Calculate daily costs
        self.daily_expense = expense_ratio / TRADING_DAYS_PER_YEAR
        
        logger.info("=" * 80)
        logger.info("SPXL SYNTHETIC SERIES BUILDER - INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Methodology: 3x S&P 500 PRICE RETURN (not total return)")
        logger.info(f"Synthetic period: {synthetic_start} to {spxl_inception}")
        logger.info(f"Actual ETF period: {spxl_inception} to {end_date}")
        logger.info(f"Leverage: {leverage}x")
        logger.info(f"Expense Ratio: {expense_ratio:.2%} annually ({self.daily_expense:.6%} daily)")
        logger.info(f"Financing: Fed Funds + {financing_spread:.2%} ({financing_spread*10000:.0f}bps)")
        logger.info("=" * 80)
    
    def fetch_spx_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch S&P 500 Price Index data from Bloomberg.
        
        Args:
            start_date: Optional custom start date
            end_date: Optional custom end date
            
        Returns:
            DataFrame with SPX prices (Date index, 'spx_price' column)
        """
        start = start_date or self.synthetic_start
        end = end_date or self.spxl_inception
        
        logger.info(f"Fetching S&P 500 Price Index ({SPX_TICKER}) from Bloomberg...")
        logger.info(f"Date range: {start} to {end}")
        
        try:
            df = blp.bdh(
                tickers=SPX_TICKER,
                flds='PX_LAST',
                start_date=start,
                end_date=end,
                Per='DAILY'
            )
            
            # Flatten columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['spx_price']
            else:
                df.columns = ['spx_price']
            
            # Remove any NaN values
            df = df.dropna()
            
            logger.info(f"✓ Fetched {len(df)} days of SPX data")
            logger.info(f"  First date: {df.index[0]}, Last date: {df.index[-1]}")
            logger.info(f"  Starting price: {df['spx_price'].iloc[0]:.2f}")
            logger.info(f"  Ending price: {df['spx_price'].iloc[-1]:.2f}")
            
            return df
            
        except Exception as e:
            logger.error(f"✗ Error fetching S&P 500 data: {str(e)}")
            raise
    
    def fetch_fed_funds_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch Federal Funds Effective Rate data from Bloomberg.
        
        Args:
            start_date: Optional custom start date
            end_date: Optional custom end date
            
        Returns:
            DataFrame with Fed Funds rates (Date index, 'fed_funds_rate' column, as decimal)
        """
        start = start_date or self.synthetic_start
        end = end_date or self.spxl_inception
        
        logger.info(f"Fetching Federal Funds Rate ({FED_FUNDS_TICKER}) from Bloomberg...")
        logger.info(f"Date range: {start} to {end}")
        
        try:
            df = blp.bdh(
                tickers=FED_FUNDS_TICKER,
                flds='PX_LAST',
                start_date=start,
                end_date=end,
                Per='DAILY'
            )
            
            # Flatten columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['fed_funds_rate']
            else:
                df.columns = ['fed_funds_rate']
            
            # Convert from percentage to decimal (e.g., 5.0 -> 0.05)
            df['fed_funds_rate'] = df['fed_funds_rate'] / 100
            
            # Forward fill for non-trading days
            df = df.ffill()
            
            logger.info(f"✓ Fetched {len(df)} days of Fed Funds data")
            logger.info(f"  First date: {df.index[0]}, Last date: {df.index[-1]}")
            logger.info(f"  Average rate: {df['fed_funds_rate'].mean():.4%}")
            logger.info(f"  Min rate: {df['fed_funds_rate'].min():.4%}, Max rate: {df['fed_funds_rate'].max():.4%}")
            
            return df
            
        except Exception as e:
            logger.error(f"✗ Error fetching Fed Funds data: {str(e)}")
            raise
    
    def fetch_spxl_data(self) -> pd.DataFrame:
        """
        Fetch actual SPXL ETF data from Bloomberg.
        
        Returns:
            DataFrame with SPXL prices (Date index, 'spxl_price' column)
        """
        logger.info(f"Fetching actual SPXL ETF data ({SPXL_TICKER}) from Bloomberg...")
        logger.info(f"Date range: {self.spxl_inception} to {self.end_date}")
        
        try:
            df = blp.bdh(
                tickers=SPXL_TICKER,
                flds='PX_LAST',
                start_date=self.spxl_inception,
                end_date=self.end_date,
                Per='DAILY'
            )
            
            # Flatten columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['spxl_price']
            else:
                df.columns = ['spxl_price']
            
            # Remove any NaN values
            df = df.dropna()
            
            logger.info(f"✓ Fetched {len(df)} days of SPXL data")
            logger.info(f"  First date: {df.index[0]}, Last date: {df.index[-1]}")
            logger.info(f"  Starting price: {df['spxl_price'].iloc[0]:.2f}")
            logger.info(f"  Current price: {df['spxl_price'].iloc[-1]:.2f}")
            
            return df
            
        except Exception as e:
            logger.error(f"✗ Error fetching SPXL data: {str(e)}")
            raise
    
    def calculate_synthetic_series(
        self, 
        spx_data: pd.DataFrame, 
        fed_funds_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate synthetic SPXL series using exact prospectus methodology.
        
        Methodology:
        1. Calculate daily S&P 500 price returns
        2. Apply 3x leverage
        3. Subtract expense ratio (daily)
        4. Subtract financing cost: (Fed Funds + 40bps) / 252
        5. Compound daily starting at initial NAV
        
        Args:
            spx_data: DataFrame with SPX prices
            fed_funds_data: DataFrame with Fed Funds rates
            
        Returns:
            DataFrame with synthetic SPXL series
        """
        logger.info("=" * 80)
        logger.info("CALCULATING SYNTHETIC SPXL SERIES")
        logger.info("=" * 80)
        
        # Merge datasets
        df = pd.merge(spx_data, fed_funds_data, left_index=True, right_index=True, how='left')
        
        # Forward fill Fed Funds for any missing dates
        df['fed_funds_rate'] = df['fed_funds_rate'].ffill()
        
        # Fill any remaining NaN at the beginning with first valid value
        df['fed_funds_rate'] = df['fed_funds_rate'].bfill()
        
        # Step 1: Calculate SPX daily returns
        df['spx_return'] = df['spx_price'].pct_change()
        
        # Step 2: Apply 3x leverage
        df['leveraged_return'] = df['spx_return'] * self.leverage
        
        # Step 3: Calculate expense cost (daily)
        df['expense_cost'] = self.daily_expense
        
        # Step 4: Calculate financing cost (daily)
        # Financing = (Fed Funds + spread) / 252
        # Per prospectus: financing rate is Fed Funds + 40bps (already accounts for leverage)
        df['financing_cost'] = (df['fed_funds_rate'] + self.financing_spread) / TRADING_DAYS_PER_YEAR
        
        # Step 5: Calculate net return
        df['net_return'] = df['leveraged_return'] - df['expense_cost'] - df['financing_cost']
        
        # Step 6: Calculate synthetic NAV by compounding
        # Handle first row (NaN return)
        net_returns = df['net_return'].fillna(0)
        cumulative_factor = (1 + net_returns).cumprod()
        df['synthetic_nav'] = self.initial_nav * cumulative_factor
        
        # Drop first row if it has NaN SPX return
        df = df.dropna(subset=['spx_return'])
        
        # Calculate summary statistics
        total_return_pct = (df['synthetic_nav'].iloc[-1] / df['synthetic_nav'].iloc[0] - 1) * 100
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (((df['synthetic_nav'].iloc[-1] / df['synthetic_nav'].iloc[0]) ** (1/years)) - 1) * 100
        avg_financing_cost = df['financing_cost'].mean() * TRADING_DAYS_PER_YEAR * 100
        
        logger.info(f"Synthetic series calculated: {len(df)} trading days")
        logger.info(f"Date range: {df.index[0]} to {df.index[-1]} ({years:.1f} years)")
        logger.info(f"Starting NAV: ${df['synthetic_nav'].iloc[0]:.2f}")
        logger.info(f"Ending NAV: ${df['synthetic_nav'].iloc[-1]:.2f}")
        logger.info(f"Total Return: {total_return_pct:.2f}%")
        logger.info(f"CAGR: {cagr:.2f}%")
        logger.info(f"Average Financing Cost: {avg_financing_cost:.2f}% annually")
        logger.info(f"Average Fed Funds Rate: {df['fed_funds_rate'].mean():.4%}")
        logger.info("=" * 80)
        
        return df
    
    def calculate_actual_series(
        self, 
        spxl_data: pd.DataFrame, 
        splice_value: float
    ) -> pd.DataFrame:
        """
        Calculate actual SPXL index series normalized to splice with synthetic.
        
        Args:
            spxl_data: DataFrame with actual SPXL prices
            splice_value: NAV value from synthetic series at inception date
            
        Returns:
            DataFrame with actual SPXL series normalized
        """
        logger.info("=" * 80)
        logger.info("CALCULATING ACTUAL SPXL SERIES")
        logger.info("=" * 80)
        
        df = spxl_data.copy()
        
        # Calculate returns
        df['spxl_return'] = df['spxl_price'].pct_change()
        
        # Normalize to start at the synthetic series ending value
        first_price = df['spxl_price'].iloc[0]
        df['spxl_nav'] = (df['spxl_price'] / first_price) * splice_value
        
        # Calculate summary statistics
        total_return_pct = (df['spxl_nav'].iloc[-1] / df['spxl_nav'].iloc[0] - 1) * 100
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (((df['spxl_nav'].iloc[-1] / df['spxl_nav'].iloc[0]) ** (1/years)) - 1) * 100
        
        logger.info(f"Actual SPXL series calculated: {len(df)} trading days")
        logger.info(f"Date range: {df.index[0]} to {df.index[-1]} ({years:.1f} years)")
        logger.info(f"Starting NAV (at inception): ${df['spxl_nav'].iloc[0]:.2f}")
        logger.info(f"Current NAV: ${df['spxl_nav'].iloc[-1]:.2f}")
        logger.info(f"Total Return: {total_return_pct:.2f}%")
        logger.info(f"CAGR: {cagr:.2f}%")
        logger.info("=" * 80)
        
        return df
    
    def build_complete_series(self) -> pd.DataFrame:
        """
        Build complete SPXL series combining synthetic and actual data.
        
        Returns:
            DataFrame with complete series (Date index, columns: nav, daily_return, source)
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("BUILDING COMPLETE SPXL SYNTHETIC SERIES")
        logger.info("=" * 80)
        logger.info("")
        
        # Step 1: Fetch all data for synthetic period
        spx_data = self.fetch_spx_data()
        fed_funds_data = self.fetch_fed_funds_data()
        
        # Step 2: Calculate synthetic series
        synthetic_df = self.calculate_synthetic_series(spx_data, fed_funds_data)
        
        # Step 3: Fetch and calculate actual SPXL series
        spxl_data = self.fetch_spxl_data()
        splice_value = synthetic_df['synthetic_nav'].iloc[-1]
        actual_df = self.calculate_actual_series(spxl_data, splice_value)
        
        # Step 4: Create standardized output dataframes
        logger.info("Splicing synthetic and actual series...")
        
        synthetic_result = pd.DataFrame({
            'nav': synthetic_df['synthetic_nav'],
            'daily_return': synthetic_df['net_return'],
            'source': 'synthetic',
            'spx_return': synthetic_df['spx_return'],
            'fed_funds_rate': synthetic_df['fed_funds_rate'],
            'financing_cost': synthetic_df['financing_cost']
        })
        
        actual_result = pd.DataFrame({
            'nav': actual_df['spxl_nav'],
            'daily_return': actual_df['spxl_return'],
            'source': 'actual',
            'spx_return': np.nan,
            'fed_funds_rate': np.nan,
            'financing_cost': np.nan
        })
        
        # Step 5: Concatenate
        complete_series = pd.concat([synthetic_result, actual_result])
        complete_series.index.name = 'Date'
        
        # Ensure index is DatetimeIndex
        if not isinstance(complete_series.index, pd.DatetimeIndex):
            complete_series.index = pd.to_datetime(complete_series.index)
        
        # Calculate overall statistics
        total_return_pct = (complete_series['nav'].iloc[-1] / complete_series['nav'].iloc[0] - 1) * 100
        years = (complete_series.index[-1] - complete_series.index[0]).days / 365.25
        cagr = (((complete_series['nav'].iloc[-1] / complete_series['nav'].iloc[0]) ** (1/years)) - 1) * 100
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("COMPLETE SERIES SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total trading days: {len(complete_series):,}")
        logger.info(f"  • Synthetic period: {len(synthetic_result):,} days")
        logger.info(f"  • Actual SPXL period: {len(actual_result):,} days")
        logger.info(f"Date range: {complete_series.index[0]} to {complete_series.index[-1]}")
        logger.info(f"Time span: {years:.1f} years")
        logger.info(f"Starting NAV: ${complete_series['nav'].iloc[0]:.2f}")
        logger.info(f"Ending NAV: ${complete_series['nav'].iloc[-1]:,.2f}")
        logger.info(f"Total Return: {total_return_pct:,.2f}%")
        logger.info(f"CAGR: {cagr:.2f}%")
        logger.info("=" * 80)
        logger.info("")
        
        return complete_series
    
    def save_series(self, series: pd.DataFrame, output_path: str = None) -> str:
        """
        Save the complete series to CSV.
        
        Args:
            series: DataFrame to save
            output_path: Optional custom output path
            
        Returns:
            Path where file was saved
        """
        if output_path is None:
            # Get the directory of the current file (tactical asset allocation/data/)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(current_dir, OUTPUT_DIR, OUTPUT_FILE)
        
        # Check if the processed directory exists, if not create it
        processed_dir = os.path.join(current_dir, OUTPUT_DIR)
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        
        # Save to CSV
        series.to_csv(output_path)
        logger.info(f"✓ Series saved to: {output_path}")
        logger.info(f"  File size: {os.path.getsize(output_path):,} bytes")
        
        return output_path


#############################################################
# CALIBRATION & VALIDATION
#############################################################

def validate_synthetic_vs_actual(builder: SPXLSyntheticBuilder) -> pd.DataFrame:
    """
    Validate synthetic methodology by comparing to actual SPXL in overlap period.
    
    This runs the synthetic model from SPXL inception to present and compares
    it to the actual SPXL performance to check for systematic bias.
    
    Args:
        builder: SPXLSyntheticBuilder instance
        
    Returns:
        DataFrame with comparison metrics
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("VALIDATION: SYNTHETIC vs ACTUAL (Overlap Period)")
    logger.info("=" * 80)
    logger.info("")
    
    # Fetch data for overlap period (SPXL inception to present)
    spx_data = builder.fetch_spx_data(
        start_date=builder.spxl_inception,
        end_date=builder.end_date
    )
    fed_funds_data = builder.fetch_fed_funds_data(
        start_date=builder.spxl_inception,
        end_date=builder.end_date
    )
    spxl_data = builder.fetch_spxl_data()
    
    # Calculate synthetic for overlap period
    synthetic_overlap = builder.calculate_synthetic_series(spx_data, fed_funds_data)
    
    # Normalize both to start at 100
    first_spxl_price = spxl_data['spxl_price'].iloc[0]
    spxl_data['spxl_normalized'] = (spxl_data['spxl_price'] / first_spxl_price) * 100
    
    first_synthetic_nav = synthetic_overlap['synthetic_nav'].iloc[0]
    synthetic_overlap['synthetic_normalized'] = (synthetic_overlap['synthetic_nav'] / first_synthetic_nav) * 100
    
    # Merge for comparison
    comparison = pd.merge(
        synthetic_overlap[['synthetic_normalized']],
        spxl_data[['spxl_normalized']],
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    comparison['difference'] = comparison['spxl_normalized'] - comparison['synthetic_normalized']
    comparison['difference_pct'] = (comparison['difference'] / comparison['synthetic_normalized']) * 100
    
    # Calculate statistics
    final_synthetic = comparison['synthetic_normalized'].iloc[-1]
    final_actual = comparison['spxl_normalized'].iloc[-1]
    tracking_error = final_actual - final_synthetic
    tracking_error_pct = (tracking_error / final_synthetic) * 100
    
    # Calculate CAGRs
    years = (comparison.index[-1] - comparison.index[0]).days / 365.25
    synthetic_cagr = (((final_synthetic / 100) ** (1/years)) - 1) * 100
    actual_cagr = (((final_actual / 100) ** (1/years)) - 1) * 100
    cagr_diff = actual_cagr - synthetic_cagr
    
    # Calculate total returns
    synthetic_total_return = (final_synthetic / 100 - 1) * 100
    actual_total_return = (final_actual / 100 - 1) * 100
    return_diff = actual_total_return - synthetic_total_return
    
    logger.info(f"Overlap period: {comparison.index[0]} to {comparison.index[-1]}")
    logger.info(f"Trading days: {len(comparison)} ({years:.2f} years)")
    logger.info(f"")
    logger.info(f"Final Values (both start at 100):")
    logger.info(f"  • Synthetic model: {final_synthetic:.2f}")
    logger.info(f"  • Actual SPXL: {final_actual:.2f}")
    logger.info(f"  • Difference: {tracking_error:.2f} ({tracking_error_pct:+.2f}%)")
    logger.info(f"")
    logger.info(f"Total Returns:")
    logger.info(f"  • Synthetic model: {synthetic_total_return:,.2f}%")
    logger.info(f"  • Actual SPXL: {actual_total_return:,.2f}%")
    logger.info(f"  • Difference: {return_diff:+.2f}%")
    logger.info(f"")
    logger.info(f"CAGR:")
    logger.info(f"  • Synthetic model: {synthetic_cagr:.2f}%")
    logger.info(f"  • Actual SPXL: {actual_cagr:.2f}%")
    logger.info(f"  • Difference: {cagr_diff:+.2f}%")
    logger.info(f"")
    logger.info(f"Tracking Error Statistics:")
    logger.info(f"  • Mean difference: {comparison['difference'].mean():.4f}")
    logger.info(f"  • Std dev: {comparison['difference'].std():.4f}")
    logger.info(f"  • Max difference: {comparison['difference'].max():.4f}")
    logger.info(f"  • Min difference: {comparison['difference'].min():.4f}")
    logger.info("=" * 80)
    logger.info("")
    
    return comparison


#############################################################
# QUANTSTATS ANALYSIS
#############################################################

def create_quantstats_comparison(complete_series: pd.DataFrame, output_dir: str = None):
    """
    Create quantstats tearsheet comparing SPXL synthetic vs SPX buy-and-hold.
    
    Args:
        complete_series: Complete SPXL series with NAV and returns
        output_dir: Optional custom output directory for tearsheet
        
    Returns:
        Tuple of (spxl_returns, spx_returns) as Series
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("QUANTSTATS ANALYSIS: SPXL vs SPX Buy-and-Hold")
    logger.info("=" * 80)
    logger.info("")
    
    # Get date range
    start_date = complete_series.index[0].strftime('%Y-%m-%d')
    end_date = complete_series.index[-1].strftime('%Y-%m-%d')
    
    # Fetch SPX Total Return Index (SPTR)
    logger.info(f"Fetching SPX Total Return Index (SPTR Index) from Bloomberg...")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    try:
        spx_tr_data = blp.bdh(
            tickers='SPTR Index',
            flds='PX_LAST',
            start_date=start_date,
            end_date=end_date,
            Per='DAILY'
        )
        
        # Flatten columns
        if isinstance(spx_tr_data.columns, pd.MultiIndex):
            spx_tr_data.columns = ['spx_tr_price']
        else:
            spx_tr_data.columns = ['spx_tr_price']
        
        logger.info(f"Fetched {len(spx_tr_data)} days of SPX Total Return data")
        logger.info(f"  Starting price: {spx_tr_data['spx_tr_price'].iloc[0]:.2f}")
        logger.info(f"  Ending price: {spx_tr_data['spx_tr_price'].iloc[-1]:.2f}")
        
        # Calculate returns
        spxl_returns = complete_series['daily_return'].copy()
        spx_returns = spx_tr_data['spx_tr_price'].pct_change()
        
        # Align the series
        combined = pd.DataFrame({
            'SPXL_3x': spxl_returns,
            'SPX_Total_Return': spx_returns
        }).dropna()
        
        # Ensure DatetimeIndex and remove timezone if present
        if not isinstance(combined.index, pd.DatetimeIndex):
            combined.index = pd.to_datetime(combined.index)
        if hasattr(combined.index, 'tz') and combined.index.tz is not None:
            combined.index = combined.index.tz_localize(None)
        
        logger.info(f"")
        logger.info(f"Aligned series: {len(combined)} trading days")
        logger.info(f"Date range: {combined.index[0]} to {combined.index[-1]}")
        
        # Set output directory
        if output_dir is None:
            # Get the directory of the current file (tactical asset allocation/data/)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(current_dir, OUTPUT_DIR)
        
        # Check if the processed directory exists, if not create it
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate tearsheet
        output_file = os.path.join(output_dir, 'spxl_vs_spx_tearsheet.html')
        logger.info(f"")
        logger.info(f"Generating quantstats tearsheet...")
        logger.info(f"Output file: {output_file}")
        
        # Extend pandas functionality for quantstats
        qs.extend_pandas()
        
        # Generate HTML tearsheet
        qs.reports.html(
            combined['SPXL_3x'],
            benchmark=combined['SPX_Total_Return'],
            output=output_file,
            title='SPXL 3x Leveraged vs SPX Total Return (1970-2025)'
        )
        
        logger.info(f"Tearsheet generated successfully!")
        logger.info(f"  File size: {os.path.getsize(output_file):,} bytes")
        
        # Also print metrics to console
        logger.info("")
        logger.info("=" * 80)
        logger.info("PERFORMANCE METRICS COMPARISON")
        logger.info("=" * 80)
        
        # Calculate metrics
        spxl_cagr = qs.stats.cagr(combined['SPXL_3x']) * 100
        spx_cagr = qs.stats.cagr(combined['SPX_Total_Return']) * 100
        
        spxl_sharpe = qs.stats.sharpe(combined['SPXL_3x'])
        spx_sharpe = qs.stats.sharpe(combined['SPX_Total_Return'])
        
        spxl_sortino = qs.stats.sortino(combined['SPXL_3x'])
        spx_sortino = qs.stats.sortino(combined['SPX_Total_Return'])
        
        spxl_maxdd = qs.stats.max_drawdown(combined['SPXL_3x']) * 100
        spx_maxdd = qs.stats.max_drawdown(combined['SPX_Total_Return']) * 100
        
        spxl_vol = qs.stats.volatility(combined['SPXL_3x']) * 100
        spx_vol = qs.stats.volatility(combined['SPX_Total_Return']) * 100
        
        logger.info(f"")
        logger.info(f"CAGR:")
        logger.info(f"  SPXL 3x:           {spxl_cagr:>8.2f}%")
        logger.info(f"  SPX Total Return:  {spx_cagr:>8.2f}%")
        logger.info(f"  Difference:        {spxl_cagr - spx_cagr:>8.2f}%")
        logger.info(f"")
        logger.info(f"Sharpe Ratio:")
        logger.info(f"  SPXL 3x:           {spxl_sharpe:>8.2f}")
        logger.info(f"  SPX Total Return:  {spx_sharpe:>8.2f}")
        logger.info(f"")
        logger.info(f"Sortino Ratio:")
        logger.info(f"  SPXL 3x:           {spxl_sortino:>8.2f}")
        logger.info(f"  SPX Total Return:  {spx_sortino:>8.2f}")
        logger.info(f"")
        logger.info(f"Max Drawdown:")
        logger.info(f"  SPXL 3x:           {spxl_maxdd:>8.2f}%")
        logger.info(f"  SPX Total Return:  {spx_maxdd:>8.2f}%")
        logger.info(f"")
        logger.info(f"Volatility (Annual):")
        logger.info(f"  SPXL 3x:           {spxl_vol:>8.2f}%")
        logger.info(f"  SPX Total Return:  {spx_vol:>8.2f}%")
        logger.info("=" * 80)
        logger.info("")
        
        return combined['SPXL_3x'], combined['SPX_Total_Return']
        
    except Exception as e:
        logger.error(f"Error in quantstats analysis: {str(e)}")
        raise


#############################################################
# MAIN EXECUTION
#############################################################

def main():
    """
    Main execution function.
    """
    # Create builder instance
    builder = SPXLSyntheticBuilder()
    
    # Run validation first
    logger.info("Running validation to compare synthetic vs actual SPXL (2008-present)...")
    validation_df = validate_synthetic_vs_actual(builder)
    
    # Build complete series
    complete_series = builder.build_complete_series()
    
    # Save to file
    output_path = builder.save_series(complete_series)
    
    # Generate quantstats comparison
    logger.info("Generating quantstats comparison and tearsheet...")
    spxl_returns, spx_returns = create_quantstats_comparison(complete_series)
    
    # Display samples
    print("\n" + "=" * 80)
    print("SAMPLE OUTPUT - First 10 rows (Synthetic Period):")
    print("=" * 80)
    print(complete_series.head(10).to_string())
    
    print("\n" + "=" * 80)
    print("SAMPLE OUTPUT - Around Inception Date (Splice Point):")
    print("=" * 80)
    inception_date = pd.Timestamp(SPXL_INCEPTION_DATE)
    mask = (complete_series.index >= inception_date - pd.Timedelta(days=5)) & \
           (complete_series.index <= inception_date + pd.Timedelta(days=5))
    print(complete_series[mask].to_string())
    
    print("\n" + "=" * 80)
    print("SAMPLE OUTPUT - Last 10 rows (Current Period):")
    print("=" * 80)
    print(complete_series.tail(10).to_string())
    
    print("\n" + "=" * 80)
    print(f"COMPLETE! Output saved to: {output_path}")
    print("=" * 80)
    
    return complete_series


if __name__ == "__main__":
    complete_series = main()
