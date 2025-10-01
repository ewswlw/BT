"""
Base Strategy Class

Abstract base class for all TAA strategies.
Provides common functionality for signal generation, rebalancing, and metrics.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.etf_inception_dates import DynamicAssetManager


class BaseTAAStrategy(ABC):
    """
    Abstract base class for Tactical Asset Allocation strategies.
    
    All strategies must implement:
    - generate_signals(): Monthly signal generation logic
    - get_strategy_info(): Strategy metadata and rules
    """
    
    def __init__(
        self,
        name: str,
        start_date: str,
        end_date: str = '2025-09-30'
    ):
        """
        Initialize base strategy.
        
        Args:
            name: Strategy name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        self.name = name
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        
        # Asset manager for dynamic availability
        self.asset_manager = DynamicAssetManager()
        
        # Data containers
        self.prices: Optional[pd.DataFrame] = None
        self.vix: Optional[pd.Series] = None
        self.tbill: Optional[pd.Series] = None
        
        # Signals and weights
        self.signals: Optional[pd.DataFrame] = None
        self.weights: Optional[pd.DataFrame] = None
        
    def set_data(
        self,
        prices: pd.DataFrame,
        vix: Optional[pd.Series] = None,
        tbill: Optional[pd.Series] = None
    ):
        """
        Set market data for strategy.
        
        Args:
            prices: DataFrame of adjusted close prices
            vix: Series of VIX values
            tbill: Series of T-bill rates (decimal form)
        """
        self.prices = prices
        self.vix = vix
        self.tbill = tbill
        
        # Validate data
        self._validate_data()
    
    def _validate_data(self):
        """Validate that required data is available."""
        if self.prices is None:
            raise ValueError("Price data not set. Call set_data() first.")
        
        if len(self.prices) == 0:
            raise ValueError("Price data is empty.")
        
        # Check for NaN values
        if self.prices.isnull().any().any():
            print(f"Warning: NaN values found in price data")
            print(self.prices.isnull().sum())
    
    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals for the strategy.
        
        Returns:
            DataFrame with dates as index and weights for each asset as columns
        """
        pass
    
    @abstractmethod
    def get_strategy_info(self) -> Dict:
        """
        Return strategy metadata and rules.
        
        Returns:
            Dictionary with strategy information
        """
        pass
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily returns from prices.
        
        Args:
            prices: DataFrame of prices
            
        Returns:
            DataFrame of daily returns
        """
        return prices.pct_change()
    
    def calculate_momentum(
        self,
        prices: pd.Series,
        lookback_periods: List[int] = [21, 63, 126, 252]
    ) -> float:
        """
        Calculate multi-period momentum score.
        
        Used by Defense First and Sector Rotation strategies.
        
        Args:
            prices: Price series for an asset
            lookback_periods: List of lookback periods in days
            
        Returns:
            Average annualized momentum across periods
        """
        momentum_values = []
        
        for period in lookback_periods:
            if len(prices) < period + 1:
                continue
                
            # Calculate return over period
            ret = (prices.iloc[-1] / prices.iloc[-(period+1)] - 1)
            
            # Annualize
            ann_ret = ret * (252 / period)
            momentum_values.append(ann_ret)
        
        if len(momentum_values) == 0:
            return 0.0
        
        return np.mean(momentum_values)
    
    def calculate_realized_volatility(
        self,
        returns: pd.Series,
        lookback: int = 20
    ) -> float:
        """
        Calculate realized volatility (annualized).
        
        Args:
            returns: Daily return series
            lookback: Lookback period in days
            
        Returns:
            Annualized volatility
        """
        if len(returns) < lookback:
            return np.nan
        
        vol = returns.iloc[-lookback:].std() * np.sqrt(252)
        return vol
    
    def get_monthly_rebalance_dates(self, prices: pd.DataFrame) -> pd.DatetimeIndex:
        """
        Get month-end dates for rebalancing.
        
        Args:
            prices: DataFrame with DatetimeIndex
            
        Returns:
            DatetimeIndex of month-end dates
        """
        # Resample to month-end and get last available date each month
        monthly = prices.resample('ME').last()  # 'ME' for month-end (pandas 2.2+)
        return monthly.index
    
    def get_signal_date(self, rebalance_date: pd.Timestamp) -> pd.Timestamp:
        """
        Get signal generation date (T-1 to avoid lookahead).
        
        Args:
            rebalance_date: Month-end rebalance date
            
        Returns:
            Signal date (previous trading day)
        """
        # Find nearest trading day on or before rebalance date
        valid_dates = self.prices.index[self.prices.index <= rebalance_date]
        
        if len(valid_dates) == 0:
            # No valid dates before rebalance date
            return rebalance_date
        
        # Get the last trading day on or before rebalance date
        last_trading_day = valid_dates[-1]
        
        # Find its index
        idx = self.prices.index.get_loc(last_trading_day)
        
        # Return previous trading day (T-1)
        if idx > 0:
            return self.prices.index[idx - 1]
        else:
            return last_trading_day
    
    def normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.0.
        
        Args:
            weights: Dictionary of asset -> weight
            
        Returns:
            Normalized weights
        """
        total = sum(weights.values())
        if total == 0:
            return weights
        
        return {k: v / total for k, v in weights.items()}
    
    def format_weight_df(
        self,
        weight_dict: Dict[pd.Timestamp, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Convert weight dictionary to DataFrame.
        
        Args:
            weight_dict: Nested dict of {date: {asset: weight}}
            
        Returns:
            DataFrame with dates as index and assets as columns
        """
        df = pd.DataFrame.from_dict(weight_dict, orient='index')
        df = df.fillna(0.0)
        # CRITICAL: Sort index chronologically to ensure proper rebalancing order
        df = df.sort_index()
        return df
    
    def __repr__(self):
        return f"{self.name} ({self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')})"

