"""
Multi-asset momentum strategy implementation.
Matches the logic from Multi-asset momentum.py exactly.
"""

from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class MultiAssetMomentumStrategy(BaseStrategy):
    """
    Multi-asset momentum strategy using daily data with weekly rebalancing.
    
    Strategy Logic:
    - On Mondays, calculate a combined momentum signal from multiple assets.
    - Enter a long position if the combined momentum exceeds a threshold.
    - Exit the position after a fixed number of holding days.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Parameters updated for daily data
        self.momentum_assets_map = config.get('momentum_assets_map', {
            'tsx': 'tsx',
            'us_hy': 'us_hy_er_index',
            'cad_ig': 'cad_ig_er_index'
        })
        self.momentum_lookback_days = config.get('momentum_lookback_days', 20)
        self.signal_threshold = config.get('signal_threshold', -0.005)
        self.exit_hold_days = config.get('exit_hold_days', 5)
        self.trading_asset = config.get('trading_asset', 'cad_ig_er_index')
        
    def get_required_features(self) -> List[str]:
        """Return required features - the actual price columns for momentum calculation."""
        return list(self.momentum_assets_map.values())
    
    def generate_signals(self, data: pd.DataFrame, features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate entry and exit signals, rebalancing weekly on Mondays.
        
        Args:
            data: Daily price data DataFrame  
            features: Not used, we calculate directly from data
            
        Returns:
            Tuple of (entry_signals, exit_signals)
        """
        # --- Monday-Only Trading Logic ---
        # 1. Identify Mondays for rebalancing
        is_monday = data.index.dayofweek == 0
        
        # 2. Calculate combined momentum signal for all days
        combined_momentum = self._calculate_combined_momentum(data)
        
        # 3. Generate entry signals ONLY on Mondays
        monday_entry_signal = (combined_momentum[is_monday] > self.signal_threshold)
        
        # Create a Series that only has entry signals on Mondays
        entry_signals_on_mondays = pd.Series(False, index=data.index)
        entry_signals_on_mondays[is_monday] = monday_entry_signal
        
        # --- Position & Exit Logic ---
        n = len(data)
        positions = np.zeros(n, dtype=int)
        hold_counter = 0

        for i in range(n):
            if entry_signals_on_mondays.iat[i]:
                positions[i] = 1
                hold_counter = self.exit_hold_days - 1
            elif hold_counter > 0:
                positions[i] = 1
                hold_counter -= 1
        
        # --- Convert positions to entry/exit signals ---
        pos_series = pd.Series(positions, index=data.index)
        pos_shifted = pos_series.shift(1).fillna(0)
        
        final_entry_signals = (pos_series == 1) & (pos_shifted == 0)
        final_exit_signals = (pos_series == 0) & (pos_shifted == 1)
        
        return final_entry_signals, final_exit_signals
    
    def _calculate_combined_momentum(self, data: pd.DataFrame) -> pd.Series:
        """Calculate combined momentum across multiple assets."""
        momentums = {}
        
        for asset_key, col_name in self.momentum_assets_map.items():
            if col_name in data.columns:
                # Use daily lookback period
                momentum = data[col_name].pct_change(self.momentum_lookback_days)
                momentums[asset_key] = momentum
        
        if momentums:
            combined_momentum = sum(momentums.values()) / len(momentums)
            return combined_momentum.fillna(0)
        else:
            return pd.Series(0.0, index=data.index)

    def get_signal_statistics(self, data: pd.DataFrame, features: pd.DataFrame) -> Dict:
        """Get detailed signal statistics for reporting."""
        entry_signals, _ = self.generate_signals(data, features)
        combined_momentum = self._calculate_combined_momentum(data)
        is_monday = data.index.dayofweek == 0
        
        return {
            'total_signals': entry_signals.sum(),
            'signal_frequency': entry_signals.mean(),
            'average_momentum_on_mondays': combined_momentum[is_monday].mean(),
            'momentum_volatility_on_mondays': combined_momentum[is_monday].std(),
        }
    
    def get_strategy_description(self) -> str:
        """Get human-readable strategy description."""
        return f"""Multi-Asset {self.momentum_lookback_days}-Day Momentum Strategy
        
Strategy Logic:
- Calculate {self.momentum_lookback_days}-day momentum for each asset
- Assets: {list(self.momentum_assets_map.keys())} -> {list(self.momentum_assets_map.values())}
- Combine individual momentums into average momentum signal
- Enter long position when combined momentum > {self.signal_threshold:.3f}
- Exit after {self.exit_hold_days} days or when signal weakens
- Trading Asset: {self.trading_asset}

Parameters:
- Momentum Lookback: {self.momentum_lookback_days} days
- Signal Threshold: {self.signal_threshold:.3f}
- Exit Hold: {self.exit_hold_days} days
- Assets Map: {self.momentum_assets_map}
""" 