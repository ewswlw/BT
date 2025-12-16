"""
TSX/S&P 500 Momentum Strategy implementation.
Weekly rebalancing strategy using 3 momentum signals.
"""

from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class TSXSPXMomentumStrategy(BaseStrategy):
    """
    TSX/S&P 500 Momentum Strategy.
    
    Strategy Logic:
    - Invest when at least 2 out of 3 momentum signals are positive:
      1. TSX 4-week momentum > 0
      2. S&P 500 4-week momentum > 0
      3. TSX 8-week momentum > 0
    - Otherwise hold cash
    - Weekly rebalancing on Mondays
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Parameters
        self.tsx_column = config.get('tsx_column', 'tsx')
        self.spx_column = config.get('spx_column', 's&p_500')
        self.trading_asset = config.get('trading_asset', 'cad_ig_er_index')
        
        # Momentum periods (in trading days)
        # 4 weeks = 20 trading days, 8 weeks = 40 trading days
        self.tsx_4week_days = config.get('tsx_4week_days', 20)
        self.spx_4week_days = config.get('spx_4week_days', 20)
        self.tsx_8week_days = config.get('tsx_8week_days', 40)
        
        # Minimum confirmations (2 out of 3)
        self.min_confirmations = config.get('min_confirmations', 2)

    def get_required_features(self) -> List[str]:
        """Return the list of asset columns required for momentum calculation."""
        return [self.tsx_column, self.spx_column]

    def generate_signals(self, data: pd.DataFrame, features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate entry and exit signals based on TSX/S&P momentum,
        rebalancing weekly on Mondays.
        
        Args:
            data: Daily price data DataFrame
            features: Not used in this strategy
            
        Returns:
            Tuple of (entry_signals, exit_signals)
        """
        # --- Monday-Only Trading Logic ---
        # 1. Identify Mondays for rebalancing
        is_monday = data.index.dayofweek == 0
        
        # 2. Calculate the 3 momentum signals
        confirmations = self._calculate_confirmations(data)
        
        # 3. Generate the signal for Mondays (at least 2 out of 3 positive)
        monday_signal = (confirmations >= self.min_confirmations)
        
        # Create a Series that only has signal values on Mondays
        signals_on_mondays = pd.Series(np.nan, index=data.index)
        signals_on_mondays[is_monday] = monday_signal[is_monday]
        
        # 4. Forward-fill the signal to hold position until the next Monday
        final_signals = signals_on_mondays.ffill().fillna(False)
        
        # --- Convert positions to entry/exit signals ---
        positions = final_signals.astype(int)
        positions_shifted = positions.shift(1).fillna(0)
        
        entry_signals = (positions == 1) & (positions_shifted == 0)
        exit_signals = (positions == 0) & (positions_shifted == 1)
        
        return entry_signals, exit_signals

    def _calculate_confirmations(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate the number of positive momentum signals (out of 3).
        
        Returns:
            Series with count of positive signals (0-3)
        """
        confirmations = pd.Series(0, index=data.index)
        
        # Signal 1: TSX 4-week momentum > 0
        if self.tsx_column in data.columns:
            tsx_4week_mom = data[self.tsx_column].pct_change(self.tsx_4week_days)
            confirmations += (tsx_4week_mom > 0).astype(int)
        
        # Signal 2: S&P 500 4-week momentum > 0
        if self.spx_column in data.columns:
            spx_4week_mom = data[self.spx_column].pct_change(self.spx_4week_days)
            confirmations += (spx_4week_mom > 0).astype(int)
        
        # Signal 3: TSX 8-week momentum > 0
        if self.tsx_column in data.columns:
            tsx_8week_mom = data[self.tsx_column].pct_change(self.tsx_8week_days)
            confirmations += (tsx_8week_mom > 0).astype(int)
        
        return confirmations

    def get_signal_statistics(self, data: pd.DataFrame, features: pd.DataFrame) -> Dict:
        """Get detailed signal statistics for reporting."""
        confirmations = self._calculate_confirmations(data)
        is_monday = data.index.dayofweek == 0
        
        # Statistics should be based on the rebalancing day signals
        monday_confirmations = confirmations[is_monday]
        entry_signals, _ = self.generate_signals(data, features)
        
        return {
            'total_signals_generated': entry_signals.sum(),
            'signal_frequency': entry_signals.mean(),
            'average_confirmations_on_mondays': monday_confirmations.mean(),
            'tsx_4week_column': self.tsx_column,
            'spx_4week_column': self.spx_column,
        }
    
    def get_strategy_description(self) -> str:
        """Get human-readable strategy description."""
        return f"""TSX/S&P 500 Momentum Strategy (Weekly Rebalance)
        
Strategy Logic:
- Long-only, unlevered strategy
- Invest when at least {self.min_confirmations} out of 3 momentum signals are positive:
  1. {self.tsx_column} {self.tsx_4week_days}-day (4-week) momentum > 0
  2. {self.spx_column} {self.spx_4week_days}-day (4-week) momentum > 0
  3. {self.tsx_column} {self.tsx_8week_days}-day (8-week) momentum > 0
- Otherwise hold cash
- Trading Asset: {self.trading_asset}
- Rebalance on Monday, hold for next week
- Uses walk-forward evaluation (no look-ahead bias)

Parameters:
- TSX 4-week Lookback: {self.tsx_4week_days} days
- S&P 500 4-week Lookback: {self.spx_4week_days} days
- TSX 8-week Lookback: {self.tsx_8week_days} days
- Minimum Confirmations: {self.min_confirmations} out of 3 signals
"""

