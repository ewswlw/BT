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
    
    Strategy Logic (matching Cross_Asset_2Week_Momentum.py):
    - Calculate 2-week momentum for 4 indices: CAD IG ER, US HY ER, US IG ER, TSX
    - Enter when ≥3 of 4 indices show positive 2-week momentum
    - Exit when signal condition is no longer met
    - Rebalance on Friday close, hold for next week
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Parameters matching Cross_Asset_2Week_Momentum.py exactly
        self.momentum_assets = config.get('momentum_assets', ['cad_ig_er_index', 'us_hy_er_index', 'us_ig_er_index', 'tsx'])
        self.momentum_lookback_weeks = config.get('momentum_lookback_weeks', 2)
        self.min_confirmations = config.get('min_confirmations', 3)
        self.trading_asset = config.get('trading_asset', 'cad_ig_er_index')
        
    def get_required_features(self) -> List[str]:
        """Return required features - the actual price columns for momentum calculation."""
        return self.momentum_assets.copy()
    
    def generate_signals(self, data: pd.DataFrame, features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate entry and exit signals based on cross-asset 2-week momentum confirmation.
        Exactly matches the generate_signals function from Cross_Asset_2Week_Momentum.py
        
        Signal Logic:
        - Calculate 2-week momentum for 4 indices
        - Enter when ≥3 of 4 indices show positive 2-week momentum
        - Exit when signal condition is no longer met
        
        Args:
            data: Weekly price data DataFrame
            features: Features DataFrame (not used, we calculate directly from data)
            
        Returns:
            tuple[pd.Series, pd.Series]: entry_signals, exit_signals (Boolean series)
        """
        # Calculate 2-week momentum for each asset (matching reference exactly)
        momentum_signals = pd.DataFrame(index=data.index)
        
        for asset in self.momentum_assets:
            if asset in data.columns:
                # 2-week momentum: positive if current price > price 2 weeks ago
                momentum_signals[asset] = (data[asset].pct_change(self.momentum_lookback_weeks) > 0).astype(int)
            else:
                print(f"Warning: {asset} not found in data, skipping...")
                momentum_signals[asset] = 0
        
        # Signal when ≥3 of 4 indices show positive momentum
        confirmation_count = momentum_signals.sum(axis=1)
        signal = (confirmation_count >= self.min_confirmations).astype(int)
        
        # Generate entry/exit signals
        # Entry: signal goes from 0 to 1
        # Exit: signal goes from 1 to 0 (or use inverse of entry for continuous position)
        entry_signals = (signal == 1)  # Long when signal is active
        exit_signals = (signal == 0)   # Exit when signal is inactive
        
        # Additional statistics for reporting (matching reference)
        print(f"\nSignal Statistics:")
        print(f"Total signals generated: {signal.sum()}")
        print(f"Signal frequency: {signal.mean():.2%}")
        print(f"Average confirmations: {confirmation_count.mean():.2f}")
        
        return entry_signals, exit_signals
    
    def get_signal_statistics(self, data: pd.DataFrame, features: pd.DataFrame) -> Dict:
        """Get detailed signal statistics for reporting."""
        entry_signals, exit_signals = self.generate_signals(data, features)
        
        # Calculate momentum signals for additional stats
        momentum_signals = pd.DataFrame(index=data.index)
        for asset in self.momentum_assets:
            if asset in data.columns:
                momentum_signals[asset] = (data[asset].pct_change(self.momentum_lookback_weeks) > 0).astype(int)
            else:
                momentum_signals[asset] = 0
        
        confirmation_count = momentum_signals.sum(axis=1)
        
        return {
            'total_signals': entry_signals.sum(),
            'signal_frequency': entry_signals.mean(),
            'average_confirmations': confirmation_count.mean(),
            'time_in_market': entry_signals.mean(),
            'longest_signal_period': self._get_longest_consecutive(entry_signals),
            'longest_no_signal_period': self._get_longest_consecutive(~entry_signals),
            'average_signal_duration': self._get_average_signal_duration(entry_signals)
        }
    
    def _get_longest_consecutive(self, signals: pd.Series) -> int:
        """Get longest consecutive True values."""
        consecutive_counts = []
        current_count = 0
        
        for signal in signals:
            if signal:
                current_count += 1
            else:
                if current_count > 0:
                    consecutive_counts.append(current_count)
                current_count = 0
        
        if current_count > 0:
            consecutive_counts.append(current_count)
        
        return max(consecutive_counts) if consecutive_counts else 0
    
    def _get_average_signal_duration(self, signals: pd.Series) -> float:
        """Get average duration of signal periods."""
        durations = []
        current_duration = 0
        
        for signal in signals:
            if signal:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0
    
    def get_strategy_description(self) -> str:
        """Get human-readable strategy description."""
        return f"""Cross-Asset {self.momentum_lookback_weeks}-Week Momentum Strategy
        
Strategy Logic (matching Cross_Asset_2Week_Momentum.py):
- Long-only, unlevered strategy
- Signal: invest when ≥{self.min_confirmations} of {len(self.momentum_assets)} indices show positive {self.momentum_lookback_weeks}-week momentum
- Indices: {', '.join(self.momentum_assets)}
- Trading Asset: {self.trading_asset}
- Rebalance on Friday close, hold for next week
- Uses walk-forward evaluation (no look-ahead bias)

Parameters:
- Momentum Lookback: {self.momentum_lookback_weeks} weeks
- Minimum Confirmations: {self.min_confirmations} of {len(self.momentum_assets)} assets
- Momentum Assets: {self.momentum_assets}
""" 