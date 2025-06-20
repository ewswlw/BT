"""
Volatility-Adaptive Momentum Strategy
"""

from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class VolAdaptiveMomentumStrategy(BaseStrategy):
    """
    A momentum strategy that adapts its entry threshold and holding period
    based on the prevailing volatility regime.

    Strategy Logic:
    - A VIX z-score is used as the primary volatility filter.
    - The entry threshold for the VIX z-score becomes stricter in high-volatility regimes.
    - A high-volatility regime is defined as the 20-day realized volatility being
      above its long-term median.
    - In calm regimes, a positive momentum filter is also applied.
    - The holding period is adaptive, increasing with stronger momentum, up to a defined maximum.
    """

    def __init__(self, config: Dict):
        """
        Initializes the Volatility-Adaptive Momentum Strategy.

        Args:
            config (Dict): A configuration dictionary with the following keys:
                - price_column (str): The name of the column for the main price series.
                - vix_column (str): The name of the column for the VIX index.
                - mom_lookback (int): Lookback period for momentum calculation (days).
                - vol_lookback (int): Lookback period for realized volatility (days).
                - vix_z_lookback (int): Lookback period for VIX z-score (days).
                - thr_low_vol (float): VIX z-score entry threshold in calm regimes.
                - thr_high_vol (float): VIX z-score entry threshold in high-volatility regimes.
                - max_hold (int): Maximum holding period for a position (days).
                - scale (float): Scaling factor for adaptive holding period calculation.
        """
        super().__init__(config)

        # Asset and indicator configuration
        self.price_column = config.get('price_column', 'cad_ig_er_index')
        self.vix_column = config.get('vix_column', 'vix')

        # Strategy parameters from the provided logic
        self.mom_lookback = config.get('mom_lookback', 20)
        self.vol_lookback = config.get('vol_lookback', 20)
        self.vix_z_lookback = config.get('vix_z_lookback', 252)
        self.thr_low_vol = config.get('thr_low_vol', 0.60)
        self.thr_high_vol = config.get('thr_high_vol', 0.00)
        self.max_hold = config.get('max_hold', 10)
        self.scale = config.get('scale', 0.005)

    def get_required_features(self) -> List[str]:
        """Returns the list of columns required from the input data."""
        return [self.price_column, self.vix_column]

    def generate_signals(self, data: pd.DataFrame, features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generates entry and exit signals for the Vol-Adaptive Momentum strategy.

        Args:
            data (pd.DataFrame): DataFrame containing the price and VIX data.
            features (pd.DataFrame): Not used in this strategy.

        Returns:
            Tuple[pd.Series, pd.Series]: A tuple of boolean Series for entry and exit signals.
        """
        price = data[self.price_column]
        vix = data[self.vix_column]

        # 1. Calculate Indicators
        daily_ret = price.pct_change()
        vix_z = (vix - vix.rolling(self.vix_z_lookback).mean()) / vix.rolling(self.vix_z_lookback).std()
        mom = price.pct_change(self.mom_lookback)
        realvol = daily_ret.rolling(self.vol_lookback).std()
        median_vol = realvol.median()

        # Fill NaNs created by rolling windows to avoid errors
        vix_z = vix_z.fillna(method='ffill').fillna(0)
        mom = mom.fillna(method='ffill').fillna(0)
        realvol = realvol.fillna(method='ffill').fillna(0)

        # 2. Generate Positions using the core logic
        n = len(price)
        pos = np.zeros(n, dtype=int)
        hold = 0

        for i in range(1, n):
            if hold > 0:
                pos[i] = 1
                hold -= 1
                continue

            high_vol_regime = realvol.iat[i] > median_vol
            thr = self.thr_high_vol if high_vol_regime else self.thr_low_vol

            if vix_z.iat[i] <= thr:
                if high_vol_regime or (mom.iat[i] > 0):
                    k = 1 + int(max(0, mom.iat[i]) / self.scale)
                    k = min(k, self.max_hold)
                    hold = k - 1 # Subtract 1 because we enter on the current day
                    pos[i] = 1
        
        # 3. Convert Positions to Entry/Exit Signals
        pos_series = pd.Series(pos, index=price.index)
        pos_shifted = pos_series.shift(1).fillna(0)
        
        entry_signals = (pos_series == 1) & (pos_shifted == 0)
        exit_signals = (pos_series == 0) & (pos_shifted == 1)

        return entry_signals, exit_signals 