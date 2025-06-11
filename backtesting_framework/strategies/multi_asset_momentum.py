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
    Multi-asset momentum strategy.
    
    Strategy Logic (matching Multi-asset momentum.py):
    - Calculate momentum for TSX, US HY ER, and CAD IG ER indices
    - Combine momentums into a single signal
    - Enter when combined momentum exceeds threshold (-0.005)
    - Exit based on shifted signal or when momentum weakens
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Parameters matching Multi-asset momentum.py exactly
        self.momentum_assets_map = config.get('momentum_assets_map', {
            'tsx': 'tsx',
            'us_hy': 'us_hy_er_index',
            'cad_ig': 'cad_ig_er_index'
        })
        self.momentum_lookback_periods = config.get('momentum_lookback_periods', 4)
        self.signal_threshold = config.get('signal_threshold', -0.005)
        self.exit_signal_shift_periods = config.get('exit_signal_shift_periods', 1)
        self.trading_asset = config.get('trading_asset', 'cad_ig_er_index')
        
    def get_required_features(self) -> List[str]:
        """Return required features - the actual price columns for momentum calculation."""
        return list(self.momentum_assets_map.values())
    
    def generate_signals(self, data: pd.DataFrame, features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate entry and exit signals based on multi-asset momentum.
        Exactly matches the calculate_multi_asset_momentum and generate_buy_sell_signals 
        functions from Multi-asset momentum.py
        
        Args:
            data: Weekly price data DataFrame  
            features: Features DataFrame (not used, we calculate directly from data)
            
        Returns:
            Tuple of (entry_signals, exit_signals)
        """
        # Calculate combined momentum from price data (matching reference exactly)
        combined_momentum = self.calculate_multi_asset_momentum(data)
        
        # Generate buy/sell signals (matching reference exactly)  
        entry_signals, exit_signals = self.generate_buy_sell_signals(combined_momentum)
        
        return entry_signals, exit_signals
    
    def calculate_multi_asset_momentum(self, weekly_df: pd.DataFrame) -> pd.Series:
        """
        Calculate combined momentum across multiple assets.
        Exactly matches calculate_multi_asset_momentum from Multi-asset momentum.py
        
        Args:
            weekly_df: DataFrame with weekly price data
            
        Returns:
            pd.Series: Combined momentum signal across all assets
        """
        momentums = {}
        
        for asset_key, col_name in self.momentum_assets_map.items():
            if col_name in weekly_df.columns:
                # Calculate momentum: (current_price / price_N_periods_ago) - 1
                momentum = weekly_df[col_name] / weekly_df[col_name].shift(self.momentum_lookback_periods) - 1
                momentums[asset_key] = momentum
                print(f"Calculated {self.momentum_lookback_periods}-period momentum for {asset_key} ({col_name})")
            else:
                print(f"Warning: {col_name} not found in data for {asset_key}")
        
        if momentums:
            # Average momentum across all available assets
            combined_momentum = sum(momentums.values()) / len(momentums)
            print(f"Combined momentum calculated from {len(momentums)} assets")
            return combined_momentum
        else:
            print("Warning: No assets found for momentum calculation")
            return pd.Series(0.0, index=weekly_df.index)
    
    def generate_buy_sell_signals(self, combined_momentum_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Generate buy and sell signals based on combined momentum.
        Exactly matches generate_buy_sell_signals from Multi-asset momentum.py
        
        Args:
            combined_momentum_series: Combined momentum signal
            
        Returns:
            tuple[pd.Series, pd.Series]: entry_signals, exit_signals
        """
        # Generate signal when momentum exceeds threshold
        momentum_signal = (combined_momentum_series > self.signal_threshold)
        
        # Entry signals: when momentum signal is True
        entry_signals = momentum_signal.astype(bool)
        
        # Exit signals: shifted entry signals (exit after holding period)
        if self.exit_signal_shift_periods > 0:
            exit_signals = entry_signals.shift(self.exit_signal_shift_periods).fillna(False)
        else:
            # Alternative: exit when signal turns False
            exit_signals = (~momentum_signal).astype(bool)
        
        print(f"Generated signals: {entry_signals.sum()} entry periods, threshold: {self.signal_threshold}")
        print(f"Signal frequency: {entry_signals.mean():.2%}")
        
        return entry_signals, exit_signals
    
    def calculate_combined_momentum(self, data: pd.DataFrame) -> pd.Series:
        """Calculate combined momentum from price data."""
        momentums = {}
        
        for asset_key, col_name in self.momentum_assets_map.items():
            if col_name in data.columns:
                momentum = data[col_name] / data[col_name].shift(self.momentum_lookback_periods) - 1
                momentums[asset_key] = momentum
        
        if momentums:
            combined_momentum = sum(momentums.values()) / len(momentums)
            return combined_momentum
        else:
            return pd.Series(0.0, index=data.index)
    
    def get_signal_statistics(self, data: pd.DataFrame, features: pd.DataFrame) -> Dict:
        """Get detailed signal statistics for reporting."""
        entry_signals, exit_signals = self.generate_signals(data, features)
        combined_momentum = self.calculate_combined_momentum(data)
        
        return {
            'total_signals': entry_signals.sum(),
            'signal_frequency': entry_signals.mean(),
            'average_momentum': combined_momentum.mean(),
            'momentum_volatility': combined_momentum.std(),
            'positive_momentum_periods': (combined_momentum > 0).sum(),
            'signal_threshold': self.signal_threshold,
            'average_signal_duration': self._get_average_signal_duration(entry_signals)
        }
    
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
        return f"""Multi-Asset {self.momentum_lookback_periods}-Period Momentum Strategy
        
Strategy Logic:
- Calculate {self.momentum_lookback_periods}-period momentum for each asset
- Assets: {list(self.momentum_assets_map.keys())} -> {list(self.momentum_assets_map.values())}
- Combine individual momentums into average momentum signal
- Enter long position when combined momentum > {self.signal_threshold:.3f}
- Exit after {self.exit_signal_shift_periods} periods or when signal weakens
- Trading Asset: {self.trading_asset}

Parameters:
- Momentum Lookback: {self.momentum_lookback_periods} periods
- Signal Threshold: {self.signal_threshold:.3f}
- Exit Shift: {self.exit_signal_shift_periods} periods
- Assets Map: {self.momentum_assets_map}
""" 