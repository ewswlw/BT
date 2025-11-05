"""
Regime-based ML strategy that uses different models/parameters for different market regimes.
"""

from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from .base_strategy import BaseStrategy
from .ml_strategy import MLStrategy
from core.feature_selection import FeatureSelector


class RegimeMLStrategy(BaseStrategy):
    """
    Regime-based ML strategy that adapts to different market regimes.
    
    Uses different models/parameters for:
    - High volatility vs low volatility
    - Trending vs mean-reverting
    - High momentum vs low momentum
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.trading_asset = config.get('trading_asset', 'us_ig_cdx_er_index')
        self.holding_period_days = config.get('holding_period_days', 7)
        
        # Regime configurations
        self.vol_regime_config = config.get('vol_regime_config', {})
        self.trend_regime_config = config.get('trend_regime_config', {})
        self.momentum_regime_config = config.get('momentum_regime_config', {})
        
        # Default regime configs if not provided
        if not self.vol_regime_config:
            self.vol_regime_config = {
                'high_vol': {
                    'probability_threshold': 0.50,
                    'forward_return_periods': 7,
                    'top_k_features': 120,
                    'ensemble_weights': [0.5, 0.5, 0.0, 0.0]
                },
                'low_vol': {
                    'probability_threshold': 0.40,
                    'forward_return_periods': 7,
                    'top_k_features': 150,
                    'ensemble_weights': [0.5, 0.5, 0.0, 0.0]
                }
            }
        
        # Regime strategies
        self.regime_strategies = {}
        self.feature_selector = None
    
    def get_required_features(self) -> List[str]:
        """Returns empty list as features are generated internally."""
        return []
    
    def identify_regime(self, data: pd.DataFrame, date: pd.Timestamp) -> str:
        """
        Identify current market regime.
        
        Returns: 'high_vol', 'low_vol', 'trending', 'mean_reverting', 'high_mom', 'low_mom'
        """
        # Remove duplicate indices first
        data_clean = data[~data.index.duplicated(keep='first')]
        
        # Get data up to current date (exclusive)
        mask = data_clean.index < date
        current_data = data_clean[mask]
        
        if len(current_data) == 0:
            # If no data before this date, use all data up to and including this date
            mask = data_clean.index <= date
            current_data = data_clean[mask]
        
        if len(current_data) < 60:
            return 'low_vol'  # Default for early data
        
        # Calculate volatility
        returns = current_data[self.trading_asset].pct_change()
        vol_20 = returns.rolling(20).std().iloc[-1]
        vol_60 = returns.rolling(60).std().iloc[-1]
        vol_252 = returns.rolling(252).std().iloc[-1] if len(current_data) >= 252 else vol_60
        
        # Volatility regime
        vol_percentile_80 = returns.rolling(252).std().quantile(0.8) if len(current_data) >= 252 else vol_60 * 1.5
        vol_percentile_20 = returns.rolling(252).std().quantile(0.2) if len(current_data) >= 252 else vol_60 * 0.5
        
        if vol_20 > vol_percentile_80:
            vol_regime = 'high_vol'
        elif vol_20 < vol_percentile_20:
            vol_regime = 'low_vol'
        else:
            vol_regime = 'low_vol'  # Default to low vol
        
        # Trend regime
        price_ma_20 = current_data[self.trading_asset].rolling(20).mean().iloc[-1]
        price_ma_60 = current_data[self.trading_asset].rolling(60).mean().iloc[-1]
        price_ma_252 = current_data[self.trading_asset].rolling(252).mean().iloc[-1] if len(current_data) >= 252 else price_ma_60
        
        if price_ma_20 > price_ma_60 > price_ma_252:
            trend_regime = 'trending'
        elif price_ma_20 < price_ma_60 < price_ma_252:
            trend_regime = 'trending'
        else:
            trend_regime = 'mean_reverting'
        
        # Momentum regime
        mom_20 = returns.rolling(20).mean().iloc[-1]
        mom_60 = returns.rolling(60).mean().iloc[-1]
        
        if mom_20 > mom_60 and mom_20 > 0:
            mom_regime = 'high_mom'
        elif mom_20 < mom_60 and mom_20 < 0:
            mom_regime = 'low_mom'
        else:
            mom_regime = 'low_mom'  # Default
        
        # Combine regimes (prioritize volatility)
        return vol_regime
    
    def create_target_for_regime(self, data: pd.DataFrame, regime: str, 
                                forward_periods: int) -> pd.Series:
        """Create target variable optimized for specific regime."""
        if self.trading_asset not in data.columns:
            raise ValueError(f"Trading asset {self.trading_asset} not found in data")
        
        forward_return = data[self.trading_asset].pct_change(forward_periods).shift(-forward_periods)
        
        # Different target definitions for different regimes
        if regime == 'high_vol':
            # In high vol, be more selective - use 75th percentile
            percentile_threshold = forward_return.quantile(0.75)
            absolute_threshold = 0.002  # 0.2% per period (more aggressive)
        else:  # low_vol
            # In low vol, use 70th percentile
            percentile_threshold = forward_return.quantile(0.70)
            absolute_threshold = 0.0015  # 0.15% per period
        
        threshold = max(percentile_threshold, absolute_threshold)
        target = (forward_return > threshold).astype(int)
        
        return target
    
    def generate_signals(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                        train_features: pd.DataFrame, test_features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate signals using regime-adaptive approach.
        """
        print(f"\n=== Regime-Based ML Strategy: {self.name} ===")
        
        # Remove duplicate indices
        train_data = train_data[~train_data.index.duplicated(keep='first')]
        test_data = test_data[~test_data.index.duplicated(keep='first')]
        train_features = train_features[~train_features.index.duplicated(keep='first')]
        test_features = test_features[~test_features.index.duplicated(keep='first')]
        
        # Combine train and test for regime identification
        combined_data = pd.concat([train_data, test_data])
        combined_features = pd.concat([train_features, test_features])
        
        # Identify regimes for each test date
        test_regimes = {}
        for date in test_data.index:
            regime = self.identify_regime(combined_data, date)
            test_regimes[date] = regime
        
        # Get unique regimes
        unique_regimes = list(set(test_regimes.values()))
        print(f"Identified regimes: {unique_regimes}")
        
        # Train separate models for each regime
        all_predictions = pd.Series(0.0, index=test_data.index, dtype=float)
        
        for regime in unique_regimes:
            print(f"\nTraining model for {regime} regime...")
            
            # Get regime config
            regime_config = self.vol_regime_config.get(regime, self.vol_regime_config.get('low_vol', {}))
            
            # Create target for this regime
            y_train = self.create_target_for_regime(
                train_data, regime, regime_config.get('forward_return_periods', 7)
            )
            
            # Align features and target
            train_features_aligned, y_train_aligned = train_features.align(y_train, join='inner', axis=0)
            valid_mask = ~(train_features_aligned.isnull().any(axis=1) | y_train_aligned.isnull())
            train_features_clean = train_features_aligned[valid_mask]
            y_train_clean = y_train_aligned[valid_mask]
            
            if len(train_features_clean) < 50:
                print(f"  Skipping {regime} - insufficient data")
                continue
            
            # Feature selection
            if regime_config.get('use_feature_selection', True):
                top_k = regime_config.get('top_k_features', 150)
                feature_selector = FeatureSelector(method='combined', top_k=top_k)
                train_features_clean = feature_selector.select_features(
                    train_features_clean, y_train_clean
                )
                selected_features = feature_selector.get_selected_features()
                print(f"  Selected {len(selected_features)} features")
            
            # Create strategy config for this regime
            strategy_config = {
                'name': f'ML_Regime_{regime}',
                'trading_asset': self.trading_asset,
                'holding_period_days': self.holding_period_days,
                'probability_threshold': regime_config.get('probability_threshold', 0.45),
                'forward_return_periods': regime_config.get('forward_return_periods', 7),
                'use_random_forest': True,
                'use_lightgbm': True,
                'use_xgboost': False,
                'use_catboost': False,
                'ensemble_weights': regime_config.get('ensemble_weights', [0.5, 0.5, 0.0, 0.0]),
                'use_feature_selection': False,  # Already done above
                'feature_selection_method': 'combined',
                'top_k_features': top_k
            }
            
            # Create custom strategy for this regime
            class RegimeMLStrategy(MLStrategy):
                def create_target(self, data: pd.DataFrame) -> pd.Series:
                    return self._create_target_for_regime(data, regime, regime_config.get('forward_return_periods', 7))
                
                def _create_target_for_regime(self, data: pd.DataFrame, regime: str, forward_periods: int) -> pd.Series:
                    forward_return = data[self.trading_asset].pct_change(forward_periods).shift(-forward_periods)
                    if regime == 'high_vol':
                        percentile_threshold = forward_return.quantile(0.75)
                        absolute_threshold = 0.002
                    else:
                        percentile_threshold = forward_return.quantile(0.70)
                        absolute_threshold = 0.0015
                    threshold = max(percentile_threshold, absolute_threshold)
                    return (forward_return > threshold).astype(int)
            
            try:
                regime_strategy = RegimeMLStrategy(strategy_config)
                regime_strategy.train_models(train_features_clean, y_train_clean)
                
                # Get test dates for this regime
                regime_test_dates = [date for date, r in test_regimes.items() if r == regime]
                if not regime_test_dates:
                    continue
                
                regime_test_data = test_data.loc[regime_test_dates]
                regime_test_features = test_features.loc[regime_test_dates]
                
                # Get predictions for this regime
                regime_test_features_aligned = regime_test_features.reindex(regime_test_data.index).ffill().fillna(0)
                regime_test_features_clean = regime_test_features_aligned.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # Ensure same features as training
                missing_features = set(train_features_clean.columns) - set(regime_test_features_clean.columns)
                for feat in missing_features:
                    regime_test_features_clean[feat] = 0
                regime_test_features_clean = regime_test_features_clean[train_features_clean.columns]
                
                # Get probabilities
                regime_proba = regime_strategy.predict_proba(regime_test_features_clean)
                regime_prob_threshold = regime_config.get('probability_threshold', 0.45)
                
                # Store predictions
                all_predictions.loc[regime_test_dates] = regime_proba[:, 1]
                
            except Exception as e:
                print(f"  Error training {regime} regime: {str(e)}")
                continue
        
        # Convert probabilities to signals
        # Use weighted probability threshold based on regime
        probability_threshold = 0.42  # Default
        
        entry_signals = (all_predictions > probability_threshold).astype(bool)
        exit_signals = (all_predictions < (probability_threshold - 0.1)).astype(bool)
        
        # Apply holding period
        print(f"\nApplying {self.holding_period_days}-day holding period...")
        positions = pd.Series(0, index=test_data.index, dtype=int)
        positions[entry_signals] = 1
        
        adjusted_positions = self.apply_holding_period(positions, self.holding_period_days)
        
        # Convert to entry/exit signals
        positions_shifted = adjusted_positions.shift(1).fillna(0).astype(int)
        final_entry_signals = (adjusted_positions == 1) & (positions_shifted == 0)
        final_exit_signals = (adjusted_positions == 0) & (positions_shifted == 1)
        
        print(f"  Entry signals: {final_entry_signals.sum()}")
        print(f"  Exit signals: {final_exit_signals.sum()}")
        print(f"  Time in market: {adjusted_positions.mean():.2%}")
        
        return final_entry_signals, final_exit_signals

