"""
Multi-timeframe ML strategy that combines predictions from different timeframes.
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


class MultiTimeframeStrategy(BaseStrategy):
    """
    Multi-timeframe ML strategy that combines predictions from different forward return periods.
    
    Uses multiple timeframes (5-day, 7-day, 10-day) and combines them for better signal quality.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.trading_asset = config.get('trading_asset', 'us_ig_cdx_er_index')
        self.holding_period_days = config.get('holding_period_days', 7)
        
        # Timeframe configurations
        self.timeframes = config.get('timeframes', [5, 7, 10])
        self.timeframe_weights = config.get('timeframe_weights', [0.3, 0.4, 0.3])
        
        # Base ML config
        self.base_ml_config = config.get('ml_config', {
            'use_random_forest': True,
            'use_lightgbm': True,
            'use_xgboost': False,
            'use_catboost': False,
        })
        
        # Probability threshold
        self.probability_threshold = config.get('probability_threshold', 0.42)
        
        # Feature selection
        self.use_feature_selection = config.get('use_feature_selection', True)
        self.feature_selection_method = config.get('feature_selection_method', 'combined')
        self.top_k_features = config.get('top_k_features', 150)
        
        # Store trained models for each timeframe
        self.timeframe_models = {}
    
    def get_required_features(self) -> List[str]:
        """Returns empty list as features are generated internally."""
        return []
    
    def create_target_for_timeframe(self, data: pd.DataFrame, forward_periods: int) -> pd.Series:
        """Create target variable for specific timeframe."""
        if self.trading_asset not in data.columns:
            raise ValueError(f"Trading asset {self.trading_asset} not found in data")
        
        forward_return = data[self.trading_asset].pct_change(forward_periods).shift(-forward_periods)
        
        # Use 70th percentile as threshold (selective)
        percentile_threshold = forward_return.quantile(0.70)
        
        # Absolute return threshold
        absolute_threshold = 0.0015  # 0.15% per period
        
        threshold = max(percentile_threshold, absolute_threshold)
        target = (forward_return > threshold).astype(int)
        
        return target
    
    def generate_signals(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                        train_features: pd.DataFrame, test_features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate signals using multi-timeframe approach.
        """
        print(f"\n=== Multi-Timeframe ML Strategy: {self.name} ===")
        
        # Remove duplicate indices
        train_data = train_data[~train_data.index.duplicated(keep='first')]
        test_data = test_data[~test_data.index.duplicated(keep='first')]
        train_features = train_features[~train_features.index.duplicated(keep='first')]
        test_features = test_features[~test_features.index.duplicated(keep='first')]
        
        # Align features and data
        common_index = train_data.index.intersection(train_features.index)
        train_data = train_data.loc[common_index]
        train_features = train_features.loc[common_index]
        
        # Store predictions for each timeframe
        timeframe_predictions = {}
        
        # Train and predict for each timeframe
        for i, timeframe in enumerate(self.timeframes):
            print(f"\nProcessing {timeframe}-day timeframe...")
            
            # Create target for this timeframe
            y_train = self.create_target_for_timeframe(train_data, timeframe)
            
            # Align features and target
            train_features_aligned, y_train_aligned = train_features.align(y_train, join='inner', axis=0)
            valid_mask = ~(train_features_aligned.isnull().any(axis=1) | y_train_aligned.isnull())
            train_features_clean = train_features_aligned[valid_mask]
            y_train_clean = y_train_aligned[valid_mask]
            
            if len(train_features_clean) < 100:
                print(f"  Skipping {timeframe}-day - insufficient data")
                continue
            
            # Feature selection
            train_features_selected = train_features_clean
            if self.use_feature_selection:
                feature_selector = FeatureSelector(
                    method=self.feature_selection_method, 
                    top_k=self.top_k_features
                )
                train_features_selected = feature_selector.select_features(
                    train_features_clean, y_train_clean
                )
                print(f"  Selected {len(feature_selector.get_selected_features())} features")
            
            # Create strategy config for this timeframe
            strategy_config = {
                'name': f'ML_Timeframe_{timeframe}',
                'trading_asset': self.trading_asset,
                'holding_period_days': self.holding_period_days,
                'probability_threshold': self.probability_threshold,
                'forward_return_periods': timeframe,
                'use_random_forest': self.base_ml_config.get('use_random_forest', True),
                'use_lightgbm': self.base_ml_config.get('use_lightgbm', True),
                'use_xgboost': self.base_ml_config.get('use_xgboost', False),
                'use_catboost': self.base_ml_config.get('use_catboost', False),
                'ensemble_weights': self.base_ml_config.get('ensemble_weights', [0.5, 0.5, 0.0, 0.0]),
                'use_feature_selection': False,  # Already done
                'feature_selection_method': self.feature_selection_method,
                'top_k_features': self.top_k_features
            }
            
            # Create custom strategy for this timeframe
            class TimeframeMLStrategy(MLStrategy):
                def create_target(self, data: pd.DataFrame) -> pd.Series:
                    forward_return = data[self.trading_asset].pct_change(timeframe).shift(-timeframe)
                    percentile_threshold = forward_return.quantile(0.70)
                    absolute_threshold = 0.0015
                    threshold = max(percentile_threshold, absolute_threshold)
                    return (forward_return > threshold).astype(int)
            
            try:
                timeframe_strategy = TimeframeMLStrategy(strategy_config)
                timeframe_strategy.train_models(train_features_selected, y_train_clean)
                
                # Get predictions for test data
                test_features_aligned = test_features.reindex(test_data.index).ffill().fillna(0)
                test_features_clean = test_features_aligned.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # Ensure same features as training
                missing_features = set(train_features_selected.columns) - set(test_features_clean.columns)
                for feat in missing_features:
                    test_features_clean[feat] = 0
                test_features_clean = test_features_clean[train_features_selected.columns]
                
                # Get probabilities
                timeframe_proba = timeframe_strategy.predict_proba(test_features_clean)
                timeframe_predictions[timeframe] = timeframe_proba[:, 1]
                
                print(f"  {timeframe}-day predictions generated")
                
            except Exception as e:
                print(f"  Error with {timeframe}-day timeframe: {str(e)}")
                continue
        
        if not timeframe_predictions:
            # Fallback to single timeframe if multi-timeframe fails
            print("Multi-timeframe failed, using single timeframe...")
            return self._fallback_single_timeframe(train_data, test_data, train_features, test_features)
        
        # Combine predictions from all timeframes
        print("\nCombining multi-timeframe predictions...")
        combined_predictions = pd.Series(0.0, index=test_data.index, dtype=float)
        
        for i, timeframe in enumerate(self.timeframes):
            if timeframe in timeframe_predictions:
                weight = self.timeframe_weights[i] if i < len(self.timeframe_weights) else 1.0 / len(timeframe_predictions)
                combined_predictions += pd.Series(timeframe_predictions[timeframe], index=test_data.index) * weight
        
        # Normalize by number of timeframes that contributed
        num_timeframes = len(timeframe_predictions)
        if num_timeframes > 0:
            combined_predictions = combined_predictions / sum(self.timeframe_weights[:num_timeframes])
        
        # Convert probabilities to signals
        entry_signals = (combined_predictions > self.probability_threshold).astype(bool)
        exit_signals = (combined_predictions < (self.probability_threshold - 0.1)).astype(bool)
        
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
    
    def _fallback_single_timeframe(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                                   train_features: pd.DataFrame, test_features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Fallback to single timeframe if multi-timeframe fails."""
        strategy_config = {
            'name': 'ML_Ensemble',
            'trading_asset': self.trading_asset,
            'holding_period_days': self.holding_period_days,
            'probability_threshold': self.probability_threshold,
            'forward_return_periods': 7,
            **self.base_ml_config,
            'use_feature_selection': self.use_feature_selection,
            'feature_selection_method': self.feature_selection_method,
            'top_k_features': self.top_k_features
        }
        
        strategy = MLStrategy(strategy_config)
        return strategy.generate_signals(train_data, test_data, train_features, test_features)

