"""
Cross-asset momentum strategy implementation.
Matches the logic from Cross_Asset_2Week_Momentum.py exactly.
"""

from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class CrossAssetMomentumStrategy(BaseStrategy):
    """
    Cross-asset momentum strategy.
    
    Strategy Logic:
    - On a weekly basis (Monday), calculate momentum for multiple indices.
    - Enter a long position if a minimum number of assets show positive momentum.
    - Hold the position until the next weekly rebalancing day.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Parameters updated for daily data with weekly rebalancing
        self.momentum_assets = config.get('momentum_assets', ['cad_ig_er_index', 'us_hy_er_index', 'us_ig_er_index', 'tsx'])
        self.momentum_lookback_days = config.get('momentum_lookback_days', 10) # e.g., 2 weeks * 5 days
        self.min_confirmations = config.get('min_confirmations', 3)
        self.trading_asset = config.get('trading_asset', 'cad_ig_er_index')

    def get_required_features(self) -> List[str]:
        """Return the list of asset columns required for momentum calculation."""
        return self.momentum_assets

    def generate_signals(self, data: pd.DataFrame, features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate entry and exit signals based on cross-asset momentum,
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
        
        # 2. Calculate momentum signals ONLY on Mondays
        confirmations = self._calculate_confirmations(data)
        
        # 3. Generate the signal for Mondays
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
        """Calculate the number of assets with positive momentum."""
        confirmations = pd.Series(0, index=data.index)
        
        for asset in self.momentum_assets:
            if asset in data.columns:
                # Use daily lookback period
                momentum = data[asset].pct_change(self.momentum_lookback_days)
                confirmations += (momentum > 0).astype(int)
        
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
        }
    
    def get_strategy_description(self) -> str:
        """Get human-readable strategy description."""
        return f"""Cross-Asset {self.momentum_lookback_days}-Day Momentum Strategy
        
Strategy Logic:
- Long-only, unlevered strategy
- Signal: invest when ≥{self.min_confirmations} of {len(self.momentum_assets)} indices show positive {self.momentum_lookback_days}-day momentum
- Indices: {', '.join(self.momentum_assets)}
- Trading Asset: {self.trading_asset}
- Rebalance on Monday, hold for next week
- Uses walk-forward evaluation (no look-ahead bias)

Parameters:
- Momentum Lookback: {self.momentum_lookback_days} days
- Minimum Confirmations: {self.min_confirmations} of {len(self.momentum_assets)} assets
- Momentum Assets: {self.momentum_assets}
""" 