"""
Rule-based strategies with momentum, mean reversion, and regime-based rules.
"""

from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base_strategy import BaseStrategy


class RuleBasedStrategy(BaseStrategy):
    """
    Rule-based strategy with multiple rule sets.
    
    Includes:
    - Momentum strategies (multi-asset, cross-asset)
    - Mean reversion strategies (z-score based)
    - Regime-based strategies (volatility regime, credit regime)
    - Combination strategies
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Trading parameters
        self.trading_asset = config.get('trading_asset', 'us_ig_cdx_er_index')
        
        # Strategy type
        self.strategy_type = config.get('strategy_type', 'momentum')  # momentum, mean_reversion, regime, combination
        
        # Momentum parameters
        self.momentum_assets = config.get('momentum_assets', ['spx_tr', 'nasdaq100', 'vix', 'gold_spot'])
        self.momentum_lookback_days = config.get('momentum_lookback_days', 20)
        self.momentum_threshold = config.get('momentum_threshold', 0.0)
        self.min_confirmations = config.get('min_confirmations', 2)
        
        # Mean reversion parameters
        self.mean_reversion_asset = config.get('mean_reversion_asset', 'us_ig_cdx_er_index')
        self.zscore_window = config.get('zscore_window', 60)
        self.zscore_lower = config.get('zscore_lower', -1.5)
        self.zscore_upper = config.get('zscore_upper', 1.5)
        
        # Regime parameters
        self.regime_asset = config.get('regime_asset', 'vix')
        self.regime_window = config.get('regime_window', 252)
        self.regime_percentile_low = config.get('regime_percentile_low', 0.2)
        self.regime_percentile_high = config.get('regime_percentile_high', 0.8)
        
        # Combination parameters
        self.use_momentum_filter = config.get('use_momentum_filter', True)
        self.use_regime_filter = config.get('use_regime_filter', True)
        self.use_mean_reversion_filter = config.get('use_mean_reversion_filter', False)
    
    def get_required_features(self) -> List[str]:
        """Return required features for this strategy."""
        required = [self.trading_asset]
        
        if self.strategy_type == 'momentum' or self.use_momentum_filter:
            required.extend([asset for asset in self.momentum_assets if asset])
        
        if self.strategy_type == 'regime' or self.use_regime_filter:
            if self.regime_asset:
                required.append(self.regime_asset)
        
        if self.strategy_type == 'mean_reversion' or self.use_mean_reversion_filter:
            if self.mean_reversion_asset:
                required.append(self.mean_reversion_asset)
        
        return list(set(required))
    
    def momentum_signal(self, data: pd.DataFrame) -> pd.Series:
        """Generate momentum-based signals."""
        signals = pd.Series(0, index=data.index, dtype=int)
        
        # Multi-asset momentum
        momentum_scores = {}
        
        for asset in self.momentum_assets:
            if asset not in data.columns:
                continue
            
            momentum = data[asset].pct_change(self.momentum_lookback_days)
            momentum_scores[asset] = momentum
        
        if not momentum_scores:
            return signals
        
        # Calculate average momentum
        momentum_df = pd.DataFrame(momentum_scores, index=data.index)
        avg_momentum = momentum_df.mean(axis=1)
        
        # Count confirmations
        confirmations = (momentum_df > self.momentum_threshold).sum(axis=1)
        
        # Generate signals
        signals = (
            (avg_momentum > self.momentum_threshold) & 
            (confirmations >= self.min_confirmations)
        ).astype(int)
        
        return signals
    
    def mean_reversion_signal(self, data: pd.DataFrame) -> pd.Series:
        """Generate mean reversion signals based on z-scores."""
        signals = pd.Series(0, index=data.index, dtype=int)
        
        if self.mean_reversion_asset not in data.columns:
            return signals
        
        asset_data = data[self.mean_reversion_asset]
        
        # Calculate z-score
        rolling_mean = asset_data.rolling(self.zscore_window).mean()
        rolling_std = asset_data.rolling(self.zscore_window).std()
        zscore = (asset_data - rolling_mean) / (rolling_std + 1e-8)
        
        # Generate signals: buy when z-score is low (oversold), sell when high (overbought)
        # For long-only strategy, we buy when oversold
        signals = (zscore < self.zscore_lower).astype(int)
        
        return signals
    
    def regime_signal(self, data: pd.DataFrame) -> pd.Series:
        """Generate regime-based signals."""
        signals = pd.Series(0, index=data.index, dtype=int)
        
        if self.regime_asset not in data.columns:
            return signals
        
        asset_data = data[self.regime_asset]
        
        # Calculate regime thresholds
        percentile_low = asset_data.rolling(self.regime_window).quantile(self.regime_percentile_low)
        percentile_high = asset_data.rolling(self.regime_window).quantile(self.regime_percentile_high)
        
        # For VIX-like assets: buy when regime is low (low volatility)
        # For other assets: buy when regime is favorable
        if 'vix' in self.regime_asset.lower():
            signals = (asset_data < percentile_low).astype(int)
        else:
            # For other assets, buy when above percentile (high regime)
            signals = (asset_data > percentile_high).astype(int)
        
        return signals
    
    def combination_signal(self, data: pd.DataFrame) -> pd.Series:
        """Generate combination signals using multiple filters."""
        signals = pd.Series(1, index=data.index, dtype=int)  # Start with all signals on
        
        # Apply momentum filter
        if self.use_momentum_filter:
            momentum_sig = self.momentum_signal(data)
            signals = signals & momentum_sig
        
        # Apply regime filter
        if self.use_regime_filter:
            regime_sig = self.regime_signal(data)
            signals = signals & regime_sig
        
        # Apply mean reversion filter
        if self.use_mean_reversion_filter:
            meanrev_sig = self.mean_reversion_signal(data)
            signals = signals & meanrev_sig
        
        return signals
    
    def generate_signals(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                        train_features: pd.DataFrame, test_features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate trading signals based on rule-based strategy.
        
        Args:
            train_data: Training price data
            test_data: Test price data
            train_features: Training features (not used for rule-based)
            test_features: Test features (not used for rule-based)
            
        Returns:
            Tuple of (entry_signals, exit_signals) for test_data
        """
        print(f"\n=== Rule-Based Strategy: {self.name} ===")
        print(f"Strategy type: {self.strategy_type}")
        
        # Combine train and test data for feature calculation
        # But only generate signals for test data
        # Remove any duplicate indices first
        train_data_clean = train_data[~train_data.index.duplicated(keep='first')]
        test_data_clean = test_data[~test_data.index.duplicated(keep='first')]
        combined_data = pd.concat([train_data_clean, test_data_clean])
        
        # Generate raw signals based on strategy type
        if self.strategy_type == 'momentum':
            raw_signals = self.momentum_signal(combined_data)
        elif self.strategy_type == 'mean_reversion':
            raw_signals = self.mean_reversion_signal(combined_data)
        elif self.strategy_type == 'regime':
            raw_signals = self.regime_signal(combined_data)
        elif self.strategy_type == 'combination':
            raw_signals = self.combination_signal(combined_data)
        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")
        
        # Extract signals for test data only
        raw_signals_test = raw_signals.loc[test_data_clean.index].fillna(0).astype(int)
        
        # Apply holding period
        print(f"Applying {self.holding_period_days}-day holding period...")
        adjusted_signals = self.apply_holding_period(raw_signals_test, self.holding_period_days)
        
        # Convert to entry/exit signals
        positions = adjusted_signals.astype(int)
        positions_shifted = positions.shift(1).fillna(0).astype(int)
        
        entry_signals = (positions == 1) & (positions_shifted == 0)
        exit_signals = (positions == 0) & (positions_shifted == 1)
        
        print(f"  Entry signals: {entry_signals.sum()}")
        print(f"  Exit signals: {exit_signals.sum()}")
        print(f"  Time in market: {positions.mean():.2%}")
        
        return entry_signals, exit_signals

