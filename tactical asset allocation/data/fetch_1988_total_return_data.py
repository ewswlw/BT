"""
Bloomberg Total Return Index Data Fetcher with SPXL Synthetic Series Builder
Fetches TOT_RETURN_INDEX_GROSS_DVDS for equity/commodity indices and PX_LAST for bond indices back to 1988-01-01
Also builds SPXL synthetic series in real-time using exact prospectus methodology with validation.

Provides comprehensive data quality diagnostics for trading/backtesting applications.
Includes equity indices (S&P 500, MSCI World, etc.), commodity indices (BCOM, etc.), bond indices (US Corporate, Treasury, etc.), and SPXL (3x S&P).
Features sequential execution of Bloomberg data fetch and SPXL synthetic building for reliable Bloomberg API access.
"""

import pandas as pd
import numpy as np
from xbbg import blp
from datetime import datetime
import logging
from typing import Tuple, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

# Complete ticker to Security name mapping from the Bloomberg terminal image
ticker_to_security = {
    # Equity Indices
    'SPTSX Index': 'S&P/TSX Composite Index',
    'SPX Index': 'S&P 500 INDEX',
    'HSI Index': 'Hang Seng Index',
    'MSDLWI Index': 'MSCI World Local',
    'MXCA Index': 'MSCI Canada Index',
    'MXWO Index': 'MSCI World Index',
    'MXEA Index': 'MSCI EAFE Index',
    'MXJP Index': 'MSCI Japan Index',
    'CCMP Index': 'NASDAQ Composite Index',
    'FNRE Index': 'FTSE Nareit Equity REITs Index',
    'MSCZVAL Index': 'US VALUE',
    'MXWO000V Index': 'MSCI World Value Index',
    'MXWO000G Index': 'MSCI World Growth Index',
    'M1US000V Index': 'MSCI USA Value Net Total Retur',
    'MXJP000V Index': 'MSCI Japan Value Index',
    'MXWO000W Index': 'MSCI World Value Index',
    'RLV Index': 'Russell 1000 Value Index',
    'MXEF Index': 'MSCI Emerging Markets Index',
    
    # Leveraged ETF (from CSV data)
    'SPXL': '3x S&P',
    
    # Commodity Indices
    'BCOM Index': 'Bloomberg Commodity Index',
    'BCOMGC Index': 'Bloomberg Gold Subindex',
    'BCOMSI Index': 'Bloomberg Silver Subindex',
    
    # Bond/Fixed Income Indices
    'LUACTRUU Index': 'Bloomberg US Corporate Total Return Index',
    'LBUSTRUU Index': 'Bloomberg US Long Treasury Total Return Index',
    'IO0087US Index': 'Bloomberg U.S. Long Credit Total Return Index',
    'LD14TRUU Index': 'Bloomberg US Govt Total Return Index',
    'LD16TRUU Index': 'Bloomberg 1-3 Yr Gov Total Return Index',
    'LD28TRUU Index': 'Bloomberg Intermediate US High Yield Total Return Index'
}

TARGET_START_DATE = '1988-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')

# SPXL Synthetic Series Configuration
SPXL_INCEPTION_DATE = '2008-11-05'
SPXL_TICKER = 'SPXL US Equity'
SPXL_EXPENSE_RATIO = 0.0091  # 0.91% annually
LEVERAGE_FACTOR = 3.0
FINANCING_SPREAD = 0.0060  # 60 basis points
SPX_TICKER = 'SPX Index'  # S&P 500 Price Return Index (NOT total return)
FED_FUNDS_TICKER = 'FEDL01 Index'  # Federal Funds Effective Rate
SYNTHETIC_START_DATE = '1970-01-02'
TRADING_DAYS_PER_YEAR = 252
INITIAL_NAV = 100.0

# Field mapping for different asset classes
FIELD_MAPPING = {
    'equity': 'TOT_RETURN_INDEX_GROSS_DVDS',
    'commodity': 'TOT_RETURN_INDEX_GROSS_DVDS', 
    'bond': 'PX_LAST'
}

# Asset class mapping for each ticker
ASSET_CLASS_MAPPING = {
    # Equity Indices
    'SPTSX Index': 'equity', 'SPX Index': 'equity', 'HSI Index': 'equity',
    'MSDLWI Index': 'equity', 'MXCA Index': 'equity', 'MXWO Index': 'equity',
    'MXEA Index': 'equity', 'MXJP Index': 'equity', 'CCMP Index': 'equity',
    'FNRE Index': 'equity', 'MSCZVAL Index': 'equity', 'MXWO000V Index': 'equity',
    'MXWO000G Index': 'equity', 'M1US000V Index': 'equity', 'MXJP000V Index': 'equity',
    'MXWO000W Index': 'equity', 'RLV Index': 'equity', 'MXEF Index': 'equity',
    
    # Leveraged ETF (from CSV data)
    'SPXL': 'equity',
    
    # Commodity Indices
    'BCOM Index': 'commodity', 'BCOMGC Index': 'commodity', 'BCOMSI Index': 'commodity',
    
    # Bond/Fixed Income Indices
    'LUACTRUU Index': 'bond', 'LUAITRUU Index': 'bond', 'LD06TRUU Index': 'bond',
    'IO2778US Index': 'bond', 'LD07TRUU Index': 'bond', 'LULCTRUU Index': 'bond',
    'LUCRTRUU Index': 'bond', 'LD24TRUU Index': 'bond', 'LD18TRUU Index': 'bond',
    'LD10TRUU Index': 'bond', 'LBUSTRUU Index': 'bond', 'IO0087US Index': 'bond',
    'LD08TRUU Index': 'bond', 'LD09TRUU Index': 'bond', 'LD11TRUU Index': 'bond',
    'LD12TRUU Index': 'bond', 'LD13TRUU Index': 'bond', 'LD14TRUU Index': 'bond',
    'LD15TRUU Index': 'bond', 'LD16TRUU Index': 'bond', 'LD17TRUU Index': 'bond',
    'LD19TRUU Index': 'bond', 'LD20TRUU Index': 'bond', 'LD21TRUU Index': 'bond',
    'LD22TRUU Index': 'bond', 'LD23TRUU Index': 'bond', 'LD25TRUU Index': 'bond',
    'LD26TRUU Index': 'bond', 'LD27TRUU Index': 'bond', 'LD28TRUU Index': 'bond'
}

# ============================================================================
# SPXL SYNTHETIC BUILDER
# ============================================================================

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
        
        print("=" * 80)
        print("SPXL SYNTHETIC SERIES BUILDER - INITIALIZED")
        print("=" * 80)
        print(f"Methodology: 3x S&P 500 PRICE RETURN (not total return)")
        print(f"Synthetic period: {synthetic_start} to {spxl_inception}")
        print(f"Actual ETF period: {spxl_inception} to {end_date}")
        print(f"Leverage: {leverage}x")
        print(f"Expense Ratio: {expense_ratio:.2%} annually ({self.daily_expense:.6%} daily)")
        print(f"Financing: Fed Funds + {financing_spread:.2%} ({financing_spread*10000:.0f}bps)")
        print("=" * 80)
    
    def fetch_spx_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch S&P 500 Price Index data from Bloomberg."""
        start = start_date or self.synthetic_start
        end = end_date or self.spxl_inception
        
        print(f"Fetching S&P 500 Price Index ({SPX_TICKER}) from Bloomberg...")
        print(f"Date range: {start} to {end}")
        
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
            
            print(f"[OK] Fetched {len(df)} days of SPX data")
            print(f"  First date: {df.index[0]}, Last date: {df.index[-1]}")
            print(f"  Starting price: {df['spx_price'].iloc[0]:.2f}")
            print(f"  Ending price: {df['spx_price'].iloc[-1]:.2f}")
            
            return df
            
        except Exception as e:
            print(f"[ERROR] Error fetching S&P 500 data: {str(e)}")
            raise
    
    def fetch_fed_funds_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch Federal Funds Effective Rate data from Bloomberg."""
        start = start_date or self.synthetic_start
        end = end_date or self.spxl_inception
        
        print(f"Fetching Federal Funds Rate ({FED_FUNDS_TICKER}) from Bloomberg...")
        print(f"Date range: {start} to {end}")
        
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
            
            print(f"[OK] Fetched {len(df)} days of Fed Funds data")
            print(f"  First date: {df.index[0]}, Last date: {df.index[-1]}")
            print(f"  Average rate: {df['fed_funds_rate'].mean():.4%}")
            print(f"  Min rate: {df['fed_funds_rate'].min():.4%}, Max rate: {df['fed_funds_rate'].max():.4%}")
            
            return df
            
        except Exception as e:
            print(f"[ERROR] Error fetching Fed Funds data: {str(e)}")
            raise
    
    def fetch_spxl_data(self) -> pd.DataFrame:
        """Fetch actual SPXL ETF data from Bloomberg."""
        print(f"Fetching actual SPXL ETF data ({SPXL_TICKER}) from Bloomberg...")
        print(f"Date range: {self.spxl_inception} to {self.end_date}")
        
        try:
            print(f"Attempting to fetch SPXL data with ticker: {SPXL_TICKER}")
            df = blp.bdh(
                tickers=SPXL_TICKER,
                flds='PX_LAST',
                start_date=self.spxl_inception,
                end_date=self.end_date,
                Per='DAILY'
            )
            print(f"Raw SPXL data shape: {df.shape}")
            print(f"Raw SPXL data columns: {df.columns}")
            
            # Check if we got any data
            if len(df) == 0:
                raise ValueError(f"No SPXL data returned from Bloomberg for ticker {SPXL_TICKER}")
            
            # Flatten columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['spxl_price']
            else:
                df.columns = ['spxl_price']
            
            # Remove any NaN values
            df = df.dropna()
            
            # Check again after dropping NaN
            if len(df) == 0:
                raise ValueError(f"No valid SPXL data after removing NaN values")
            
            print(f"[OK] Fetched {len(df)} days of SPXL data")
            print(f"  First date: {df.index[0]}, Last date: {df.index[-1]}")
            print(f"  Starting price: {df['spxl_price'].iloc[0]:.2f}")
            print(f"  Current price: {df['spxl_price'].iloc[-1]:.2f}")
            
            return df
            
        except Exception as e:
            print(f"[ERROR] Error fetching SPXL data: {str(e)}")
            raise
    
    def calculate_synthetic_series(
        self, 
        spx_data: pd.DataFrame, 
        fed_funds_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate synthetic SPXL series using exact prospectus methodology."""
        print("=" * 80)
        print("CALCULATING SYNTHETIC SPXL SERIES")
        print("=" * 80)
        
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
        df['financing_cost'] = (df['fed_funds_rate'] + self.financing_spread) / TRADING_DAYS_PER_YEAR
        
        # Step 5: Calculate net return
        df['net_return'] = df['leveraged_return'] - df['expense_cost'] - df['financing_cost']
        
        # Step 6: Calculate synthetic NAV by compounding
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
        
        print(f"Synthetic series calculated: {len(df)} trading days")
        print(f"Date range: {df.index[0]} to {df.index[-1]} ({years:.1f} years)")
        print(f"Starting NAV: ${df['synthetic_nav'].iloc[0]:.2f}")
        print(f"Ending NAV: ${df['synthetic_nav'].iloc[-1]:.2f}")
        print(f"Total Return: {total_return_pct:.2f}%")
        print(f"CAGR: {cagr:.2f}%")
        print(f"Average Financing Cost: {avg_financing_cost:.2f}% annually")
        print(f"Average Fed Funds Rate: {df['fed_funds_rate'].mean():.4%}")
        print("=" * 80)
        
        return df
    
    def calculate_actual_series(
        self, 
        spxl_data: pd.DataFrame, 
        splice_value: float
    ) -> pd.DataFrame:
        """Calculate actual SPXL index series normalized to splice with synthetic."""
        print("=" * 80)
        print("CALCULATING ACTUAL SPXL SERIES")
        print("=" * 80)
        
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
        
        print(f"Actual SPXL series calculated: {len(df)} trading days")
        print(f"Date range: {df.index[0]} to {df.index[-1]} ({years:.1f} years)")
        print(f"Starting NAV (at inception): ${df['spxl_nav'].iloc[0]:.2f}")
        print(f"Current NAV: ${df['spxl_nav'].iloc[-1]:.2f}")
        print(f"Total Return: {total_return_pct:.2f}%")
        print(f"CAGR: {cagr:.2f}%")
        print("=" * 80)
        
        return df
    
    def build_complete_series(self) -> pd.DataFrame:
        """Build complete SPXL series combining synthetic and actual data."""
        print("")
        print("=" * 80)
        print("BUILDING COMPLETE SPXL SYNTHETIC SERIES")
        print("=" * 80)
        print("")
        
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
        print("Splicing synthetic and actual series...")
        
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
        
        print("")
        print("=" * 80)
        print("COMPLETE SERIES SUMMARY")
        print("=" * 80)
        print(f"Total trading days: {len(complete_series):,}")
        print(f"  • Synthetic period: {len(synthetic_result):,} days")
        print(f"  • Actual SPXL period: {len(actual_result):,} days")
        print(f"Date range: {complete_series.index[0]} to {complete_series.index[-1]}")
        print(f"Time span: {years:.1f} years")
        print(f"Starting NAV: ${complete_series['nav'].iloc[0]:.2f}")
        print(f"Ending NAV: ${complete_series['nav'].iloc[-1]:,.2f}")
        print(f"Total Return: {total_return_pct:,.2f}%")
        print(f"CAGR: {cagr:.2f}%")
        print("=" * 80)
        print("")
        
        return complete_series


def validate_synthetic_vs_actual(builder: SPXLSyntheticBuilder) -> pd.DataFrame:
    """
    Validate synthetic methodology by comparing to actual SPXL in overlap period.
    """
    print("")
    print("=" * 80)
    print("VALIDATION: SYNTHETIC vs ACTUAL (Overlap Period)")
    print("=" * 80)
    print("")
    
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
    
    print(f"Overlap period: {comparison.index[0]} to {comparison.index[-1]}")
    print(f"Trading days: {len(comparison)} ({years:.2f} years)")
    print(f"")
    print(f"Final Values (both start at 100):")
    print(f"  • Synthetic model: {final_synthetic:.2f}")
    print(f"  • Actual SPXL: {final_actual:.2f}")
    print(f"  • Difference: {tracking_error:.2f} ({tracking_error_pct:+.2f}%)")
    print(f"")
    print(f"Total Returns:")
    print(f"  • Synthetic model: {synthetic_total_return:,.2f}%")
    print(f"  • Actual SPXL: {actual_total_return:,.2f}%")
    print(f"  • Difference: {return_diff:+.2f}%")
    print(f"")
    print(f"CAGR:")
    print(f"  • Synthetic model: {synthetic_cagr:.2f}%")
    print(f"  • Actual SPXL: {actual_cagr:.2f}%")
    print(f"  • Difference: {cagr_diff:+.2f}%")
    print(f"")
    print(f"Tracking Error Statistics:")
    print(f"  • Mean difference: {comparison['difference'].mean():.4f}")
    print(f"  • Std dev: {comparison['difference'].std():.4f}")
    print(f"  • Max difference: {comparison['difference'].max():.4f}")
    print(f"  • Min difference: {comparison['difference'].min():.4f}")
    print("=" * 80)
    print("")
    
    return comparison


# ============================================================================
# SPXL DATA LOADER
# ============================================================================

def build_spxl_data():
    """
    Build SPXL synthetic series data in real-time (no CSV caching).
    
    Returns:
        pd.DataFrame: DataFrame with Date index and SPXL NAV values
    """
    
    print(f"\nBuilding SPXL synthetic series data in real-time...")
    
    # Create builder instance
    builder = SPXLSyntheticBuilder()
    
    # Run validation first
    print("Running validation to compare synthetic vs actual SPXL (2008-present)...")
    validation_df = validate_synthetic_vs_actual(builder)
    
    # Build complete series
    complete_series = builder.build_complete_series()
    
    # Filter to start from 1988-01-01 to align with other assets
    start_date = pd.Timestamp(TARGET_START_DATE)
    df_spxl = complete_series.loc[start_date:].copy()
    
    # Keep only the NAV column and rename it to match our naming convention
    df_spxl = df_spxl[['nav']].copy()
    df_spxl.columns = ['3x S&P']
    
    # Forward fill any missing data
    df_spxl = df_spxl.ffill()
    
    print(f"[OK] Built SPXL data: {len(df_spxl)} rows from {df_spxl.index.min().strftime('%Y-%m-%d')} to {df_spxl.index.max().strftime('%Y-%m-%d')}")
    
    return df_spxl


# ============================================================================
# DATA FETCH
# ============================================================================

def fetch_bloomberg_data():
    """Fetch Bloomberg data for equity/commodity and bond indices."""
    tickers = list(ticker_to_security.keys())
    
    # Group tickers by field (exclude SPXL as it comes from synthetic builder)
    equity_tickers = [t for t in tickers if ASSET_CLASS_MAPPING[t] in ['equity', 'commodity'] and t != 'SPXL']
    bond_tickers = [t for t in tickers if ASSET_CLASS_MAPPING[t] == 'bond']
    
    print(f"Fetching Bloomberg data...")
    print(f"Equity/Commodity tickers ({len(equity_tickers)}): {FIELD_MAPPING['equity']}")
    print(f"Bond tickers ({len(bond_tickers)}): {FIELD_MAPPING['bond']}")
    
    # Fetch equity/commodity data
    df_equity = None
    if equity_tickers:
        df_equity = blp.bdh(
            tickers=equity_tickers,
            flds=FIELD_MAPPING['equity'],
            start_date=TARGET_START_DATE,
            end_date=END_DATE
        )
    
    # Fetch bond data
    df_bonds = None
    if bond_tickers:
        df_bonds = blp.bdh(
            tickers=bond_tickers,
            flds=FIELD_MAPPING['bond'],
            start_date=TARGET_START_DATE,
            end_date=END_DATE
        )
    
    return df_equity, df_bonds


def fetch_total_return_data():
    """
    Fetch total return index data from Bloomberg with comprehensive quality checks.
    Now includes real-time SPXL synthetic series building with validation.
    
    Returns:
        pd.DataFrame: DataFrame with Date index and security names as columns
    """
    
    print("=" * 100)
    print(f"BLOOMBERG DATA FETCH - TOTAL RETURN INDICES (EQUITY + COMMODITY + BONDS + SPXL)")
    print("=" * 100)
    print(f"Fields: Equity/Commodity={FIELD_MAPPING['equity']}, Bonds={FIELD_MAPPING['bond']}")
    print(f"Period: {TARGET_START_DATE} to {END_DATE}")
    print(f"Total Tickers: {len(ticker_to_security)} (Equity: 19, Commodity: 3, Bonds: 29, SPXL: 1)")
    print("=" * 100)

    # Run Bloomberg data fetch and SPXL building sequentially (Bloomberg API doesn't handle parallel access well)
    print(f"\nRunning Bloomberg data fetch and SPXL synthetic building...")
    
    # Fetch Bloomberg data first
    print("Step 1: Fetching Bloomberg data...")
    df_equity, df_bonds = fetch_bloomberg_data()
    
    # Build SPXL data
    print("Step 2: Building SPXL synthetic series...")
    df_spxl = build_spxl_data()
    
    # Combine the dataframes
    dataframes = []
    if df_equity is not None:
        dataframes.append(df_equity)
    if df_bonds is not None:
        dataframes.append(df_bonds)
    if df_spxl is not None:
        dataframes.append(df_spxl)
    
    if dataframes:
        df_raw = pd.concat(dataframes, axis=1)
    else:
        raise ValueError("No data fetched from Bloomberg or SPXL builder")

    print(f"[OK] Received {len(df_raw)} rows from Bloomberg + SPXL synthetic builder")

    # ============================================================================
    # DATA PREPROCESSING
    # ============================================================================

    # Flatten MultiIndex columns if necessary
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.droplevel(1)

    # Ensure all columns are strings (not tuples)
    df_raw.columns = [str(col) if isinstance(col, tuple) else col for col in df_raw.columns]

    # Rename columns to Security names
    df = df_raw.copy()
    
    # Create a mapping for Bloomberg columns (extract ticker from column names)
    column_mapping = {}
    for col in df.columns:
        if col == '3x S&P':  # SPXL column is already clean
            column_mapping[col] = col
        else:
            # Extract ticker from Bloomberg column names like "('SPX Index', 'TOT_RETURN_INDEX_GROSS_DVDS')"
            # or "SPX Index"
            if col.startswith("('") and col.endswith("')"):
                # MultiIndex format: extract the ticker part
                ticker_part = col.split("', '")[0][2:]  # Remove "('" and get ticker
                if ticker_part in ticker_to_security:
                    column_mapping[col] = ticker_to_security[ticker_part]
                else:
                    column_mapping[col] = col
            else:
                # Simple format: direct ticker name
                if col in ticker_to_security:
                    column_mapping[col] = ticker_to_security[col]
                else:
                    column_mapping[col] = col
    
    df.columns = [column_mapping[col] for col in df.columns]

    # Reset index to make Date a column, then set it back
    df = df.reset_index()
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # Remove duplicate dates (keep first occurrence)
    df = df[~df.index.duplicated(keep='first')]
    
    # Forward fill all missing data (weekends, holidays, gaps)
    df = df.ffill()
    print(f"[OK] Applied forward-fill to handle missing data")

    return df


def analyze_data_quality(df):
    """
    Perform comprehensive data quality analysis.
    
    Args:
        df: DataFrame with Date index and security columns
        
    Returns:
        list: List of dictionaries containing security statistics
    """
    
    # ============================================================================
    # DATA QUALITY ASSESSMENT
    # ============================================================================

    print("\n" + "=" * 100)
    print("DATA QUALITY REPORT")
    print("=" * 100)

    # 1. Date Range Analysis
    print(f"\n{'DATE RANGE ANALYSIS':<50}")
    print("-" * 100)
    print(f"{'Earliest Date in Dataset:':<40} {df.index.min().strftime('%Y-%m-%d')}")
    print(f"{'Latest Date in Dataset:':<40} {df.index.max().strftime('%Y-%m-%d')}")
    print(f"{'Total Trading Days:':<40} {len(df)}")
    years_span = df.index.max().year - df.index.min().year
    print(f"{'Expected Days (approx 252/year):':<40} ~{years_span * 252}")

    # 2. Per-Security Analysis
    print(f"\n{'PER-SECURITY DATA QUALITY':<50}")
    print("-" * 100)
    print(f"{'Security':<45} {'Start Date':<12} {'End Date':<12} {'Coverage':<8} {'Missing':<8} {'Status'}")
    print("-" * 100)

    security_stats = []

    for col in df.columns:
        first_valid_idx = df[col].first_valid_index()
        last_valid_idx = df[col].last_valid_index()
        
        if first_valid_idx is not None:
            first_date = first_valid_idx
            last_date = last_valid_idx
            
            # Calculate coverage
            subset = df.loc[first_date:last_date, col]
            non_null_count = subset.notna().sum()
            total_count = len(subset)
            coverage_pct = (non_null_count / total_count * 100) if total_count > 0 else 0
            missing_count = total_count - non_null_count
            
            # Status determination
            target_date = pd.Timestamp(TARGET_START_DATE)
            if first_date <= target_date:
                status = "[OK] GOOD"
            elif first_date <= pd.Timestamp(TARGET_START_DATE) + pd.Timedelta(days=90):
                status = "[!] NEAR"
            else:
                status = "[X] LATE"
            
            security_stats.append({
                'Security': col,
                'Start': first_date,
                'End': last_date,
                'Coverage': coverage_pct,
                'Missing': missing_count,
                'Status': status
            })
            
            print(f"{col:<45} {first_date.strftime('%Y-%m-%d'):<12} {last_date.strftime('%Y-%m-%d'):<12} "
                  f"{coverage_pct:>6.2f}% {missing_count:>7} {status}")
        else:
            print(f"{col:<45} {'NO DATA':<12} {'NO DATA':<12} {'0.00%':>7} {'ALL':>7} [X] NO DATA")
            security_stats.append({
                'Security': col,
                'Start': None,
                'End': None,
                'Coverage': 0,
                'Missing': len(df),
                'Status': '[X] NO DATA'
            })

    return security_stats


def print_summary_statistics(security_stats):
    """Print summary statistics about data quality."""
    
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)

    target_date = pd.Timestamp(TARGET_START_DATE)
    good_start = len([s for s in security_stats if s['Start'] and s['Start'] <= target_date])
    high_coverage_99 = len([s for s in security_stats if s['Coverage'] > 99])
    high_coverage_95 = len([s for s in security_stats if s['Coverage'] > 95])
    low_coverage = len([s for s in security_stats if s['Coverage'] < 90])

    print(f"\nSecurities with data starting on or before {TARGET_START_DATE}: {good_start}")
    print(f"Securities with >99% coverage: {high_coverage_99}")
    print(f"Securities with >95% coverage: {high_coverage_95}")
    print(f"Securities with <90% coverage: {low_coverage}")


def check_data_integrity(df):
    """Check for data anomalies and integrity issues."""
    
    print("\n" + "=" * 100)
    print("DATA INTEGRITY CHECKS")
    print("=" * 100)

    # Check for negative values (shouldn't happen in total return indices)
    negative_counts = (df < 0).sum()
    if negative_counts.sum() > 0:
        print("\n[!] WARNING: Negative values detected (should not occur in return indices):")
        print(negative_counts[negative_counts > 0])
    else:
        print("\n[OK] No negative values detected (GOOD)")

    # Check for zeros (suspicious in return indices)
    zero_counts = (df == 0).sum()
    if zero_counts.sum() > 0:
        print("\n[!] WARNING: Zero values detected:")
        print(zero_counts[zero_counts > 0])
    else:
        print("\n[OK] No zero values detected (GOOD)")

    # Check for extreme jumps (>50% daily change - likely data error)
    print("\n" + "-" * 100)
    print("Extreme Daily Returns Check (>50% single-day move):")
    pct_change = df.pct_change()
    extreme_moves = (pct_change.abs() > 0.5).sum()
    if extreme_moves.sum() > 0:
        print("\n[!] WARNING: Extreme single-day moves detected:")
        print(extreme_moves[extreme_moves > 0])
        print("\nDetails of extreme moves:")
        for col in extreme_moves[extreme_moves > 0].index:
            extreme_dates = pct_change[col][pct_change[col].abs() > 0.5]
            for date, value in extreme_dates.items():
                print(f"  {col}: {date.strftime('%Y-%m-%d')} - {value*100:.2f}% change")
    else:
        print("[OK] No extreme daily moves detected (GOOD)")


def print_dataframe_info(df):
    """Print DataFrame structure and info."""
    
    print("\n" + "=" * 100)
    print("DATAFRAME INFO")
    print("=" * 100)
    df.info()

    print("\n" + "=" * 100)
    print("FIRST 10 ROWS")
    print("=" * 100)
    print(df.head(10).to_string())

    print("\n" + "=" * 100)
    print("LAST 10 ROWS")
    print("=" * 100)
    print(df.tail(10).to_string())

    print("\n" + "=" * 100)
    print("DESCRIPTIVE STATISTICS (Index Levels)")
    print("=" * 100)
    print(df.describe().to_string())


def analyze_cumulative_returns(df):
    """Analyze and display cumulative returns for each security."""
    
    print("\n" + "=" * 100)
    print("CUMULATIVE RETURN ANALYSIS")
    print("=" * 100)

    print("\nTotal Return Since Inception (Per Security):")
    print("-" * 100)
    print(f"{'Security':<45} {'Start Date':<12} {'Start Value':<15} {'End Value':<15} {'Total Return':<15}")
    print("-" * 100)

    returns_data = []

    for col in df.columns:
        first_valid = df[col].first_valid_index()
        last_valid = df[col].last_valid_index()
        
        if first_valid and last_valid:
            start_val = df.loc[first_valid, col]
            end_val = df.loc[last_valid, col]
            total_return = ((end_val / start_val) - 1) * 100
            
            returns_data.append({
                'Security': col,
                'Start_Date': first_valid,
                'Total_Return_Pct': total_return
            })
            
            print(f"{col:<45} {first_valid.strftime('%Y-%m-%d'):<12} {start_val:>14.2f} {end_val:>14.2f} {total_return:>13.1f}%")

    # Sort by total return
    returns_df = pd.DataFrame(returns_data).sort_values('Total_Return_Pct', ascending=False)
    
    print("\n" + "-" * 100)
    print("RANKING BY TOTAL RETURN:")
    print("-" * 100)
    for idx, row in returns_df.iterrows():
        print(f"{row['Security']:<45} {row['Total_Return_Pct']:>13.1f}%")


def print_recommendations(security_stats):
    """Print final recommendations for data usage."""
    
    print("\n" + "=" * 100)
    print("DATA ENGINEER RECOMMENDATIONS")
    print("=" * 100)

    target_date = pd.Timestamp(TARGET_START_DATE)
    good_securities = [s for s in security_stats if s['Start'] and s['Start'] <= target_date]
    good_coverage = [s for s in security_stats if s['Coverage'] > 95]

    print(f"\n[OK] {len(good_securities)} securities have data starting on or before {TARGET_START_DATE}")
    print(f"[OK] {len(good_coverage)} securities have >95% coverage within their date range")

    if len(good_securities) < len(security_stats):
        late_securities = [s['Security'] for s in security_stats 
                          if not s['Start'] or s['Start'] > target_date]
        print(f"\n[!] {len(late_securities)} securities start after {TARGET_START_DATE}:")
        for sec in late_securities:
            matching_stat = next(s for s in security_stats if s['Security'] == sec)
            if matching_stat['Start']:
                print(f"   - {sec:<45} (starts {matching_stat['Start'].strftime('%Y-%m-%d')})")
            else:
                print(f"   - {sec:<45} (no data)")

    print("\nRECOMMENDATION:")
    if len(good_securities) >= 10:
        print("[OK] Sufficient securities available for multi-asset backtesting from 1988")
        print("[OK] Data quality appears good for trading strategy development")
    else:
        print("[!] Limited securities available from 1988 - consider adjusting start date")


def calculate_performance_metrics(df):
    """
    Calculate comprehensive performance metrics for all securities.
    
    Args:
        df: DataFrame with Date index and security columns (forward-filled)
        
    Returns:
        pd.DataFrame: Performance metrics table
    """
    
    print("\n" + "=" * 100)
    print("CALCULATING COMPREHENSIVE PERFORMANCE METRICS")
    print("=" * 100)
    
    metrics_list = []
    latest_date = df.index.max()
    
    for col in df.columns:
        print(f"Processing: {col}")
        
        # Get valid data range
        series = df[col].dropna()
        if len(series) == 0:
            continue
            
        start_date = series.index.min()
        end_date = series.index.max()
        
        # 1. Security name
        security_name = col
        
        # 2. Start Date
        start_dt = start_date.strftime('%Y-%m-%d')
        
        # 3. Frequency (auto-detect)
        days_span = (end_date - start_date).days
        years_span = days_span / 365.25
        obs_count = len(series)
        obs_per_year = obs_count / years_span if years_span > 0 else 0
        
        if obs_per_year >= 240:
            frequency = "Daily"
        elif obs_per_year >= 48:
            frequency = "Weekly"
        elif obs_per_year >= 10:
            frequency = "Monthly"
        else:
            frequency = "Other"
        
        # Find latest point in 2024 for YTD calculation
        dates_2024 = series.loc[series.index.year == 2024]
        if len(dates_2024) > 0:
            end_2024 = dates_2024.index.max()
            start_2025 = series.loc[series.index >= '2025-01-01'].index.min() if len(series.loc[series.index >= '2025-01-01']) > 0 else None
            
            if start_2025 and end_2024:
                ytd_start_val = series.loc[end_2024]
                ytd_end_val = series.loc[end_date]
                ytd_return = ((ytd_end_val / ytd_start_val) - 1) * 100
            else:
                ytd_return = np.nan
        else:
            ytd_return = np.nan
        
        # Rolling period returns
        def calculate_cagr(start_val, end_val, years):
            """Calculate CAGR given start/end values and time period."""
            if start_val <= 0 or end_val <= 0 or years <= 0:
                return np.nan
            return (((end_val / start_val) ** (1 / years)) - 1) * 100
        
        # 1-year return
        one_yr_ago = end_date - pd.Timedelta(days=365)
        if start_date <= one_yr_ago:
            one_yr_val = series.loc[series.index >= one_yr_ago].iloc[0]
            one_yr_return = ((series.loc[end_date] / one_yr_val) - 1) * 100
        else:
            one_yr_return = np.nan
        
        # 3-year CAGR
        three_yr_ago = end_date - pd.Timedelta(days=365*3)
        if start_date <= three_yr_ago:
            three_yr_val = series.loc[series.index >= three_yr_ago].iloc[0]
            three_yr_cagr = calculate_cagr(three_yr_val, series.loc[end_date], 3)
        else:
            three_yr_cagr = np.nan
        
        # 5-year CAGR
        five_yr_ago = end_date - pd.Timedelta(days=365*5)
        if start_date <= five_yr_ago:
            five_yr_val = series.loc[series.index >= five_yr_ago].iloc[0]
            five_yr_cagr = calculate_cagr(five_yr_val, series.loc[end_date], 5)
        else:
            five_yr_cagr = np.nan
        
        # 10-year CAGR
        ten_yr_ago = end_date - pd.Timedelta(days=365*10)
        if start_date <= ten_yr_ago:
            ten_yr_val = series.loc[series.index >= ten_yr_ago].iloc[0]
            ten_yr_cagr = calculate_cagr(ten_yr_val, series.loc[end_date], 10)
        else:
            ten_yr_cagr = np.nan
        
        # 15-year CAGR
        fifteen_yr_ago = end_date - pd.Timedelta(days=365*15)
        if start_date <= fifteen_yr_ago:
            fifteen_yr_val = series.loc[series.index >= fifteen_yr_ago].iloc[0]
            fifteen_yr_cagr = calculate_cagr(fifteen_yr_val, series.loc[end_date], 15)
        else:
            fifteen_yr_cagr = np.nan
        
        # Since Inception CAGR
        inception_years = (end_date - start_date).days / 365.25
        inception_cagr = calculate_cagr(series.iloc[0], series.iloc[-1], inception_years)
        
        # Max Drawdown calculation
        cummax = series.cummax()
        drawdown = (series - cummax) / cummax
        max_dd = drawdown.min() * 100  # Convert to percentage
        max_dd_date = drawdown.idxmin()
        
        # Find the peak before the trough
        trough_date = max_dd_date
        peak_before_trough = series.loc[:trough_date].idxmax()
        
        # Calculate recovery period (days from trough to new all-time high)
        series_after_trough = series.loc[trough_date:]
        peak_value_at_trough = cummax.loc[trough_date]
        
        # Find when it recovered to a new high
        recovery_dates = series_after_trough[series_after_trough >= peak_value_at_trough].index
        
        if len(recovery_dates) > 0:
            recovery_date = recovery_dates[0]
            # Count trading days between trough and recovery
            days_to_recovery = len(series.loc[trough_date:recovery_date]) - 1  # -1 to exclude start date
        else:
            days_to_recovery = "Not Recovered"
        
        # Compile metrics
        metrics_list.append({
            'Security': security_name,
            'Start Date': start_dt,
            'Frequency': frequency,
            'YTD (%)': ytd_return,
            '1yr (%)': one_yr_return,
            '3yr CAGR (%)': three_yr_cagr,
            '5yr CAGR (%)': five_yr_cagr,
            '10yr CAGR (%)': ten_yr_cagr,
            '15yr CAGR (%)': fifteen_yr_cagr,
            'Inception CAGR (%)': inception_cagr,
            'Max Drawdown (%)': max_dd,
            'Worst DD Date': max_dd_date.strftime('%Y-%m-%d'),
            'Days to Recovery': days_to_recovery
        })
    
    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    return metrics_df


def print_performance_metrics_table(metrics_df):
    """Print the performance metrics table in a nicely formatted way, sorted by inception CAGR."""
    
    print("\n" + "=" * 100)
    print("COMPREHENSIVE PERFORMANCE METRICS TABLE (Sorted by Inception CAGR)")
    print("=" * 100)
    
    # Sort by inception CAGR (highest to lowest), handling NaN values
    display_df = metrics_df.copy()
    display_df = display_df.sort_values('Inception CAGR (%)', ascending=False, na_position='last')
    
    # Format numeric columns
    numeric_cols = ['YTD (%)', '1yr (%)', '3yr CAGR (%)', '5yr CAGR (%)', 
                    '10yr CAGR (%)', '15yr CAGR (%)', 'Inception CAGR (%)', 'Max Drawdown (%)']
    
    for col in numeric_cols:
        display_df[col] = display_df[col].apply(
            lambda x: f"{x:>6.2f}" if pd.notna(x) else "   N/A"
        )
    
    print("\n" + display_df.to_string(index=False))
    
    print("\n" + "=" * 100)
    print("LEGEND:")
    print("  N/A = Insufficient history for the time period")
    print("  Not Recovered = Asset has not yet reached new all-time high since worst drawdown")
    print("=" * 100)
    
    return display_df


def main():
    """Main execution function."""
    
    # Fetch data
    df = fetch_total_return_data()
    
    # Analyze quality
    security_stats = analyze_data_quality(df)
    
    # Print statistics
    print_summary_statistics(security_stats)
    
    # Check integrity
    check_data_integrity(df)
    
    # Print info
    print_dataframe_info(df)
    
    # Analyze returns
    analyze_cumulative_returns(df)
    
    # Print recommendations
    print_recommendations(security_stats)
    
    # Calculate and display comprehensive performance metrics
    metrics_df = calculate_performance_metrics(df)
    print_performance_metrics_table(metrics_df)
    
    # Save DataFrame to CSV
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "processed", "1988 assets 3x.csv")
    
    # Reset index to make Date a column (first column)
    df_to_save = df.reset_index()
    df_to_save.rename(columns={df_to_save.columns[0]: 'Date'}, inplace=True)
    
    # Save to CSV
    df_to_save.to_csv(csv_path, index=False)
    print(f"\n[OK] Data saved to: {csv_path}")
    print(f"     Shape: {df_to_save.shape}")
    print(f"     Columns: {list(df_to_save.columns)}")
    
    print("\n" + "=" * 100)
    print("DATA FETCH COMPLETE")
    print("=" * 100)
    print(f"\nDataFrame stored in variable 'df_final'")
    print(f"Performance metrics stored in variable 'metrics_df'")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df, metrics_df


if __name__ == "__main__":
    df_final, metrics_df = main()

