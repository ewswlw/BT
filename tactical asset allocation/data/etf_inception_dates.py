"""
ETF Inception Date Registry

Manages dynamic asset availability based on ETF launch dates.
Used to ensure backtests only use assets that existed at each point in time.
"""

from datetime import datetime
from typing import List
import pandas as pd


# ETF Inception Dates (verified from Bloomberg/provider websites)
ETF_INCEPTION_DATES = {
    # Equity Indices
    'SPY': datetime(1993, 1, 29),      # SPDR S&P 500 ETF Trust
    'SPXL': datetime(2008, 11, 5),     # Direxion Daily S&P 500 Bull 3X Shares
    
    # Defensive Assets
    'TLT': datetime(2002, 7, 26),      # iShares 20+ Year Treasury Bond ETF
    'GLD': datetime(2004, 11, 18),     # SPDR Gold Shares
    'DBC': datetime(2006, 2, 3),       # Invesco DB Commodity Index Tracking Fund
    'UUP': datetime(2007, 2, 26),      # Invesco DB US Dollar Index Bullish Fund
    'BTAL': datetime(2011, 5, 11),     # AGFiQ US Market Neutral Anti-Beta Fund
    
    # Sector ETFs - Original 9 (all launched same day)
    'XLB': datetime(1998, 12, 16),     # Materials Select Sector SPDR Fund
    'XLE': datetime(1998, 12, 16),     # Energy Select Sector SPDR Fund
    'XLF': datetime(1998, 12, 16),     # Financial Select Sector SPDR Fund
    'XLI': datetime(1998, 12, 16),     # Industrial Select Sector SPDR Fund
    'XLK': datetime(1998, 12, 16),     # Technology Select Sector SPDR Fund
    'XLP': datetime(1998, 12, 16),     # Consumer Staples Select Sector SPDR Fund
    'XLU': datetime(1998, 12, 16),     # Utilities Select Sector SPDR Fund
    'XLV': datetime(1998, 12, 16),     # Health Care Select Sector SPDR Fund
    'XLY': datetime(1998, 12, 16),     # Consumer Discretionary Select Sector SPDR Fund
    
    # Later Sector ETFs
    'XLRE': datetime(2015, 10, 7),     # Real Estate Select Sector SPDR Fund
    'XLC': datetime(2018, 6, 18),      # Communication Services Select Sector SPDR Fund
}


# Strategy Start Dates (matching academic papers)
STRATEGY_START_DATES = {
    'vix_timing': datetime(2013, 1, 1),              # Match Quantpedia study (2013-2025)
    'defense_first_base': datetime(2008, 2, 1),      # Match paper replication period
    'defense_first_levered': datetime(2008, 11, 5),  # SPXL inception date
    'sector_rotation': datetime(1999, 12, 1),        # Match paper (1999-2025)
}


class DynamicAssetManager:
    """
    Manages dynamic asset availability based on inception dates.
    
    This ensures backtests only use ETFs that existed at each point in time,
    matching academic paper methodology.
    
    Examples:
        >>> manager = DynamicAssetManager()
        >>> manager.get_available_assets('2010-01-01', ['TLT', 'GLD', 'BTAL'])
        ['TLT', 'GLD']  # BTAL not yet launched
        
        >>> manager.get_available_assets('2012-01-01', ['TLT', 'GLD', 'BTAL'])
        ['TLT', 'GLD', 'BTAL']  # All available
    """
    
    def __init__(self):
        self.inception_dates = ETF_INCEPTION_DATES
    
    def get_available_assets(self, date: pd.Timestamp, asset_list: List[str]) -> List[str]:
        """
        Return only assets from asset_list that exist on given date.
        
        Args:
            date: Date to check availability
            asset_list: List of asset tickers to filter
            
        Returns:
            List of available asset tickers
        """
        date = pd.Timestamp(date)
        available = []
        
        for asset in asset_list:
            if asset not in self.inception_dates:
                raise ValueError(f"Unknown asset: {asset}. Add to ETF_INCEPTION_DATES.")
            
            inception = pd.Timestamp(self.inception_dates[asset])
            if date >= inception:
                available.append(asset)
        
        return available
    
    def get_sector_universe(self, date: pd.Timestamp) -> List[str]:
        """
        Return available sector ETFs for given date.
        
        Sector universe evolves:
        - 1998-12-16 to 2015-10-06: 9 original sectors
        - 2015-10-07 to 2018-06-17: 10 sectors (+ XLRE)
        - 2018-06-18 onwards: 11 sectors (+ XLC)
        
        Args:
            date: Date to check sector availability
            
        Returns:
            List of available sector tickers
        """
        date = pd.Timestamp(date)
        
        # Original 9 sectors (launched Dec 1998)
        original_9 = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
        
        if date < pd.Timestamp(self.inception_dates['XLRE']):
            return original_9
        elif date < pd.Timestamp(self.inception_dates['XLC']):
            # 10 sectors (add Real Estate)
            return ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
        else:
            # All 11 sectors
            return ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
    
    def get_defensive_universe(self, date: pd.Timestamp) -> List[str]:
        """
        Return available defensive assets for given date.
        
        Defensive universe evolves:
        - 2007-02-26 to 2011-05-10: 4 assets (TLT, GLD, DBC, UUP)
        - 2011-05-11 onwards: 5 assets (+ BTAL)
        
        Args:
            date: Date to check defensive asset availability
            
        Returns:
            List of available defensive asset tickers
        """
        date = pd.Timestamp(date)
        
        all_defensives = ['TLT', 'GLD', 'DBC', 'UUP', 'BTAL']
        return self.get_available_assets(date, all_defensives)
    
    def is_asset_available(self, date: pd.Timestamp, asset: str) -> bool:
        """Check if a single asset is available on given date."""
        date = pd.Timestamp(date)
        if asset not in self.inception_dates:
            return False
        return date >= pd.Timestamp(self.inception_dates[asset])
    
    def get_inception_date(self, asset: str) -> datetime:
        """Get inception date for a specific asset."""
        if asset not in self.inception_dates:
            raise ValueError(f"Unknown asset: {asset}")
        return self.inception_dates[asset]

