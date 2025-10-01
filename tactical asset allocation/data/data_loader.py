"""
Bloomberg Data Loader using xbbg

Fetches ETF price data, VIX, and T-bill rates from Bloomberg.
Implements caching to avoid redundant API calls.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import pickle
from datetime import datetime
import xbbg
from xbbg import blp

from .etf_inception_dates import DynamicAssetManager, ETF_INCEPTION_DATES


class DataLoader:
    """
    Bloomberg data loader with caching support.
    
    Fetches adjusted close prices for ETFs, VIX index, and T-bill rates.
    Caches data locally to minimize Bloomberg API calls.
    """
    
    # Bloomberg ticker mappings
    BLOOMBERG_TICKERS = {
        # Equity Indices
        'SPY': 'SPY US Equity',
        'SPXL': 'SPXL US Equity',
        
        # Defensive Assets
        'TLT': 'TLT US Equity',
        'GLD': 'GLD US Equity',
        'DBC': 'DBC US Equity',
        'UUP': 'UUP US Equity',
        'BTAL': 'BTAL US Equity',
        
        # Sector ETFs
        'XLB': 'XLB US Equity',
        'XLC': 'XLC US Equity',
        'XLE': 'XLE US Equity',
        'XLF': 'XLF US Equity',
        'XLI': 'XLI US Equity',
        'XLK': 'XLK US Equity',
        'XLP': 'XLP US Equity',
        'XLRE': 'XLRE US Equity',
        'XLU': 'XLU US Equity',
        'XLV': 'XLV US Equity',
        'XLY': 'XLY US Equity',
        
        # Indices
        'VIX': 'VIX Index',
        
        # Rates
        'TBILL_3M': 'USGG3M Index',  # 3-Month US Government Bond Yield
    }
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize data loader.
        
        Args:
            cache_dir: Directory for caching data (default: ./data/processed/)
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent / 'processed'
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.asset_manager = DynamicAssetManager()
        
    def fetch_prices(
        self, 
        tickers: List[str], 
        start_date: str, 
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch adjusted close prices for list of tickers.
        
        Args:
            tickers: List of ticker symbols (e.g., ['SPY', 'TLT'])
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with dates as index and tickers as columns
        """
        cache_key = f"prices_{'_'.join(sorted(tickers))}_{start_date}_{end_date}.pkl"
        cache_file = self.cache_dir / cache_key
        
        # Check cache
        if use_cache and cache_file.exists():
            print(f"Loading cached price data: {cache_file.name}")
            return pd.read_pickle(cache_file)
        
        print(f"Fetching price data from Bloomberg for {len(tickers)} tickers...")
        
        # Convert tickers to Bloomberg format
        bbg_tickers = [self.BLOOMBERG_TICKERS[t] for t in tickers]
        
        # Fetch from Bloomberg
        try:
            # Use xbbg to fetch historical data
            data = blp.bdh(
                tickers=bbg_tickers,
                flds='PX_LAST',  # Adjusted close price
                start_date=start_date,
                end_date=end_date
            )
            
            # Ensure DatetimeIndex
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Clean up column names (remove Bloomberg suffix)
            if isinstance(data.columns, pd.MultiIndex):
                # If multi-level columns, take first level
                data.columns = data.columns.get_level_values(0)
            
            # Map back to simple ticker names
            ticker_map = {v: k for k, v in self.BLOOMBERG_TICKERS.items()}
            data.columns = [ticker_map.get(col, col) for col in data.columns]
            
            # Forward fill missing data (weekends, holidays)
            data = data.ffill()
            
            # Cache the data
            data.to_pickle(cache_file)
            print(f"Cached price data: {cache_file.name}")
            
            return data
            
        except Exception as e:
            print(f"Error fetching data from Bloomberg: {e}")
            raise
    
    def fetch_vix(
        self,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.Series:
        """
        Fetch VIX index data.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data
            
        Returns:
            Series with VIX values
        """
        cache_file = self.cache_dir / f"vix_{start_date}_{end_date}.pkl"
        
        if use_cache and cache_file.exists():
            print(f"Loading cached VIX data: {cache_file.name}")
            return pd.read_pickle(cache_file)
        
        print("Fetching VIX data from Bloomberg...")
        
        try:
            data = blp.bdh(
                tickers='VIX Index',
                flds='PX_LAST',
                start_date=start_date,
                end_date=end_date
            )
            
            # Ensure DatetimeIndex
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Convert to Series
            if isinstance(data, pd.DataFrame):
                vix_series = data.iloc[:, 0]
            else:
                vix_series = data
            
            vix_series.name = 'VIX'
            
            # Ensure DatetimeIndex on series too
            if not isinstance(vix_series.index, pd.DatetimeIndex):
                vix_series.index = pd.to_datetime(vix_series.index)
            
            # Forward fill
            vix_series = vix_series.ffill()
            
            # Cache
            vix_series.to_pickle(cache_file)
            print(f"Cached VIX data: {cache_file.name}")
            
            return vix_series
            
        except Exception as e:
            print(f"Error fetching VIX data: {e}")
            raise
    
    def fetch_tbill_rate(
        self,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.Series:
        """
        Fetch 3-month T-bill rate (USGG3M Index) from Bloomberg.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data
            
        Returns:
            Series with T-bill rates (in decimal form, e.g., 0.05 = 5%)
        """
        cache_file = self.cache_dir / f"tbill_{start_date}_{end_date}.pkl"
        
        if use_cache and cache_file.exists():
            print(f"Loading cached T-bill data: {cache_file.name}")
            return pd.read_pickle(cache_file)
        
        print("Fetching T-bill rate from Bloomberg...")
        
        try:
            data = blp.bdh(
                tickers='USGG3M Index',
                flds='PX_LAST',
                start_date=start_date,
                end_date=end_date
            )
            
            # Ensure DatetimeIndex
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Convert to Series
            if isinstance(data, pd.DataFrame):
                tbill_series = data.iloc[:, 0]
            else:
                tbill_series = data
            
            tbill_series.name = 'TBILL_3M'
            
            # Ensure DatetimeIndex on series too
            if not isinstance(tbill_series.index, pd.DatetimeIndex):
                tbill_series.index = pd.to_datetime(tbill_series.index)
            
            # Convert from percentage to decimal (e.g., 5.0 -> 0.05)
            tbill_series = tbill_series / 100.0
            
            # Forward fill
            tbill_series = tbill_series.ffill()
            
            # Cache
            tbill_series.to_pickle(cache_file)
            print(f"Cached T-bill data: {cache_file.name}")
            
            return tbill_series
            
        except Exception as e:
            print(f"Error fetching T-bill data: {e}")
            raise
    
    def load_all_strategy_data(
        self,
        strategy_name: str,
        start_date: Optional[str] = None,
        end_date: str = '2025-08-31',  # Default to latest paper end date
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all data needed for a specific strategy.
        
        Args:
            strategy_name: One of ['vix_timing', 'defense_first_base', 
                                   'defense_first_levered', 'sector_rotation']
            start_date: Override start date (uses strategy default if None)
            end_date: End date
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary with keys: 'prices', 'vix', 'tbill'
        """
        from .etf_inception_dates import STRATEGY_START_DATES
        
        if start_date is None:
            start_date = STRATEGY_START_DATES[strategy_name].strftime('%Y-%m-%d')
        
        print(f"\n{'='*80}")
        print(f"Loading data for: {strategy_name.upper()}")
        print(f"Period: {start_date} to {end_date}")
        print(f"{'='*80}\n")
        
        # Determine required assets
        if strategy_name == 'vix_timing':
            tickers = ['SPY', 'SPXL', 'TLT', 'GLD', 'DBC', 'UUP', 'BTAL']
        elif strategy_name == 'defense_first_base':
            tickers = ['SPY', 'TLT', 'GLD', 'DBC', 'UUP', 'BTAL']
        elif strategy_name == 'defense_first_levered':
            tickers = ['SPY', 'SPXL', 'TLT', 'GLD', 'DBC', 'UUP', 'BTAL']
        elif strategy_name == 'sector_rotation':
            tickers = ['SPY', 'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 
                      'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Fetch price data
        prices = self.fetch_prices(tickers, start_date, end_date, use_cache)
        
        # Fetch VIX (needed for vix_timing and potentially others)
        vix = self.fetch_vix(start_date, end_date, use_cache)
        
        # Fetch T-bill rate (needed for all strategies)
        tbill = self.fetch_tbill_rate(start_date, end_date, use_cache)
        
        # Align all data to same dates
        common_dates = prices.index.intersection(vix.index).intersection(tbill.index)
        
        return {
            'prices': prices.loc[common_dates],
            'vix': vix.loc[common_dates],
            'tbill': tbill.loc[common_dates],
        }
    
    def clear_cache(self):
        """Clear all cached data files."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print("Cache cleared")

