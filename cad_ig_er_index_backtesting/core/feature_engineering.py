"""
Feature engineering components for technical analysis.
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class FeatureEngineer(ABC):
    """Abstract base class for feature engineering."""
    
    @abstractmethod
    def create_features(self, price_data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Create features from price data."""
        pass


class TechnicalFeatureEngineer(FeatureEngineer):
    """Technical analysis feature engineer."""
    
    def create_features(self, price_data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Create technical analysis features."""
        features = pd.DataFrame(index=price_data.index)
        
        # Get primary asset (first column or specified)
        primary_asset = config.get('primary_asset')
        if primary_asset and primary_asset in price_data.columns:
            price_series = price_data[primary_asset]
        else:
            price_series = price_data.iloc[:, 0]
        
        # Momentum features
        if 'momentum_periods' in config:
            for period in config['momentum_periods']:
                features[f"mom_{period}"] = price_series.pct_change(period)
        
        # Volatility features
        if 'volatility_windows' in config:
            returns = price_series.pct_change()
            for window in config['volatility_windows']:
                features[f"vol_{window}"] = returns.rolling(window).std()
        
        # SMA deviation features
        if 'sma_windows' in config:
            for window in config['sma_windows']:
                sma = price_series.rolling(window).mean()
                features[f"sma_{window}_dev"] = price_series / sma - 1
        
        # MACD features (matching genetic algo weekly.py exactly)
        if config.get('include_macd', False):
            fast_period = config.get('macd_fast', 12)
            slow_period = config.get('macd_slow', 26)
            signal_period = config.get('macd_signal', 9)
            
            ema_fast = price_series.ewm(span=fast_period, adjust=False).mean()
            ema_slow = price_series.ewm(span=slow_period, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            features["macd_diff"] = macd_line - signal_line
        
        # Stochastic oscillator (matching genetic algo weekly.py exactly)
        if config.get('include_stochastic', False):
            period = config.get('stochastic_k_period', 14)
            low_min = price_series.rolling(period).min()
            high_max = price_series.rolling(period).max()
            features["stoch_k"] = 100 * (price_series - low_min) / (high_max - low_min + 1e-8)
        
        # RSI
        if config.get('include_rsi', False):
            period = config.get('rsi_period', 14)
            features["rsi"] = self._calculate_rsi(price_series, period)
        
        # Bollinger Bands
        if config.get('include_bollinger', False):
            window = config.get('bollinger_window', 20)
            std_dev = config.get('bollinger_std', 2)
            sma = price_series.rolling(window).mean()
            std = price_series.rolling(window).std()
            features[f"bb_upper"] = sma + (std * std_dev)
            features[f"bb_lower"] = sma - (std * std_dev)
            features[f"bb_position"] = (price_series - features[f"bb_lower"]) / (features[f"bb_upper"] - features[f"bb_lower"])
        
        # OAS and VIX features (matching genetic algo weekly.py exactly)
        oas_vix_cols = ["cad_oas", "us_ig_oas", "us_hy_oas", "vix"]
        if config.get('include_oas_features', False):
            momentum_period = config.get('oas_momentum_period', 4)
            for col in oas_vix_cols:
                if col in price_data.columns:
                    # N-week momentum for risk indicators (matching reference)
                    features[f"{col}_mom{momentum_period}"] = price_data[col].pct_change(momentum_period)
                else:
                    # Fill with default value if column not found (matching reference)
                                         features[f"{col}_mom{momentum_period}"] = config.get('fill_na_value', 0.0)
        
        return features.fillna(config.get('fill_na_value', 0))
    
    def _calculate_rsi(self, price_series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        delta = price_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class CrossAssetFeatureEngineer(FeatureEngineer):
    """Cross-asset feature engineer."""
    
    def create_features(self, price_data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Create cross-asset features."""
        features = pd.DataFrame(index=price_data.index)
        
        # Asset correlation features
        if config.get('include_correlations', False):
            corr_window = config.get('correlation_window', 20)
            assets = config.get('correlation_assets', price_data.columns)
            
            for i, asset1 in enumerate(assets):
                if asset1 not in price_data.columns:
                    continue
                    
                ret1 = price_data[asset1].pct_change()
                for asset2 in assets[i+1:]:
                    if asset2 not in price_data.columns:
                        continue
                        
                    ret2 = price_data[asset2].pct_change()
                    features[f"{asset1}_{asset2}_corr"] = ret1.rolling(corr_window).corr(ret2)
        
        # Relative strength features
        if config.get('include_relative_strength', False):
            benchmark = config.get('relative_strength_benchmark', price_data.columns[0])
            if benchmark in price_data.columns:
                for asset in price_data.columns:
                    if asset != benchmark:
                        features[f"{asset}_vs_{benchmark}_ratio"] = price_data[asset] / price_data[benchmark]
                        features[f"{asset}_vs_{benchmark}_momentum"] = (price_data[asset] / price_data[benchmark]).pct_change(20)
        
        # Momentum confirmation features (for cross-asset momentum strategies)
        if config.get('momentum_assets', []):
            momentum_assets = config['momentum_assets']
            lookback = config.get('momentum_lookback_weeks', 2)
            
            momentum_signals = pd.DataFrame(index=price_data.index)
            for asset in momentum_assets:
                if asset in price_data.columns:
                    momentum_signals[f"{asset}_momentum"] = (price_data[asset].pct_change(lookback) > 0).astype(int)
                    features[f"{asset}_momentum_signal"] = momentum_signals[f"{asset}_momentum"]
            
            # Count of positive momentum signals
            if len(momentum_signals.columns) > 0:
                features["momentum_confirmation_count"] = momentum_signals.sum(axis=1)
                features["momentum_confirmation_pct"] = momentum_signals.mean(axis=1)
        
        return features.fillna(0)


class MultiAssetFeatureEngineer(FeatureEngineer):
    """Multi-asset momentum feature engineer."""
    
    def create_features(self, price_data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Create multi-asset momentum features."""
        features = pd.DataFrame(index=price_data.index)
        
        # Multi-asset momentum calculation
        if config.get('momentum_assets_map', {}):
            asset_map = config['momentum_assets_map']
            lookback = config.get('momentum_lookback_periods', 4)
            
            momentums = {}
            for asset_key, col_name in asset_map.items():
                if col_name in price_data.columns:
                    momentum = price_data[col_name] / price_data[col_name].shift(lookback) - 1
                    momentums[asset_key] = momentum
                    features[f"{asset_key}_momentum"] = momentum
            
            # Combined momentum
            if momentums:
                combined_momentum = sum(momentums.values()) / len(momentums)
                features["combined_momentum"] = combined_momentum
                
                # Momentum signals
                threshold = config.get('signal_threshold', -0.005)
                features["momentum_signal"] = (combined_momentum > threshold).astype(int)
        
        return features.fillna(0)


class FeatureSelector:
    """Feature selection utilities."""
    
    @staticmethod
    def select_by_correlation(features: pd.DataFrame, target: pd.Series, 
                            threshold: float = 0.05) -> List[str]:
        """Select features based on correlation with target."""
        correlations = features.corrwith(target).abs()
        selected = correlations[correlations > threshold].index.tolist()
        return selected
    
    @staticmethod
    def select_by_variance(features: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """Select features based on variance threshold."""
        variances = features.var()
        selected = variances[variances > threshold].index.tolist()
        return selected
    
    @staticmethod
    def remove_highly_correlated(features: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features."""
        corr_matrix = features.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        return features.drop(features[to_drop], axis=1)


def create_feature_engineer(engineer_type: str, **kwargs) -> FeatureEngineer:
    """Factory function to create feature engineers."""
    engineers = {
        'technical': TechnicalFeatureEngineer,
        'cross_asset': CrossAssetFeatureEngineer,
        'multi_asset': MultiAssetFeatureEngineer
    }
    
    if engineer_type not in engineers:
        raise ValueError(f"Unknown feature engineer type: {engineer_type}")
    
    return engineers[engineer_type](**kwargs) 