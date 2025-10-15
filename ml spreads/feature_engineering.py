"""
Feature engineering for CAD OAS prediction model.
Creates comprehensive features with proper lag constraints.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import warnings
# Config will be imported in __init__ to avoid circular imports

class FeatureEngineer:
    """Handles feature engineering with proper lag enforcement."""
    
    def __init__(self, config_obj=None):
        """Initialize FeatureEngineer with configuration."""
        # Import config here to avoid circular imports
        from config import config, FEATURE_CONFIG
        self.config = config_obj or config
        self.feature_config = FEATURE_CONFIG
        
    def create_baseline_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create baseline feature set with proper lagging."""
        print("Creating baseline features...")
        
        features_df = pd.DataFrame(index=data.index)
        
        # 1. Raw lagged values for all feature columns
        lagged_features = self._create_lagged_features(
            data, self.config.feature_columns, self.config.baseline_lags
        )
        features_df = pd.concat([features_df, lagged_features], axis=1)
        
        # 2. Returns over multiple horizons (with proper lagging)
        for col in self.config.feature_columns:
            if col not in data.columns:
                continue
            
            for period in self.config.momentum_periods:
                # Calculate returns, then lag them to ensure no look-ahead bias
                returns = data[col].pct_change(period)
                # Lag the returns by min_lag to ensure prediction is based on past data
                lagged_returns = returns.shift(self.config.min_lag)
                features_df[f'{col}_returns_{period}d_lag{self.config.min_lag}'] = lagged_returns
        
        # 3. Moving averages (with proper lagging)
        for col in self.config.feature_columns:
            if col not in data.columns:
                continue
                
            for period in self.config.ma_periods:
                ma = data[col].rolling(period).mean()
                # Lag the moving average by min_lag
                lagged_ma = ma.shift(self.config.min_lag)
                features_df[f'{col}_ma_{period}d_lag{self.config.min_lag}'] = lagged_ma
                
                # Price relative to moving average
                price_ma_ratio = data[col] / ma
                lagged_ratio = price_ma_ratio.shift(self.config.min_lag)
                features_df[f'{col}_ma_ratio_{period}d_lag{self.config.min_lag}'] = lagged_ratio
        
        # 4. Rolling volatility (with proper lagging)
        for col in self.config.feature_columns:
            if col not in data.columns:
                continue
                
            # Calculate daily returns first
            daily_returns = data[col].pct_change()
            
            for period in self.config.volatility_periods:
                vol = daily_returns.rolling(period).std()
                # Lag the volatility by min_lag
                lagged_vol = vol.shift(self.config.min_lag)
                features_df[f'{col}_vol_{period}d_lag{self.config.min_lag}'] = lagged_vol
        
        # Remove any remaining NaN values
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        print(f"Created {len(features_df.columns)} baseline features")
        return features_df
    
    def create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators with proper lagging."""
        print("Creating technical indicators...")
        
        features_df = pd.DataFrame(index=data.index)
        tech_config = self.feature_config['technical_indicators']
        
        # Focus on spread-related columns for technical indicators
        spread_columns = ['cad_oas', 'us_ig_oas', 'us_hy_oas']
        
        for col in spread_columns:
            if col not in data.columns:
                continue
            
            # RSI (Relative Strength Index)
            for period in tech_config['rsi_periods']:
                rsi = self._calculate_rsi(data[col], period)
                # Lag RSI by min_lag
                lagged_rsi = rsi.shift(self.config.min_lag)
                features_df[f'{col}_rsi_{period}d_lag{self.config.min_lag}'] = lagged_rsi
            
            # MACD
            macd_line, signal_line, macd_diff = self._calculate_macd(
                data[col], 
                tech_config['macd_fast'],
                tech_config['macd_slow'], 
                tech_config['macd_signal']
            )
            # Lag MACD features
            features_df[f'{col}_macd_diff_lag{self.config.min_lag}'] = macd_diff.shift(self.config.min_lag)
            features_df[f'{col}_macd_signal_lag{self.config.min_lag}'] = (macd_line > signal_line).astype(int).shift(self.config.min_lag)
            
            # Bollinger Bands
            bb_upper, bb_lower, bb_position = self._calculate_bollinger_bands(
                data[col],
                tech_config['bollinger_window'],
                tech_config['bollinger_std']
            )
            features_df[f'{col}_bb_position_lag{self.config.min_lag}'] = bb_position.shift(self.config.min_lag)
            features_df[f'{col}_bb_width_lag{self.config.min_lag}'] = ((bb_upper - bb_lower) / data[col]).shift(self.config.min_lag)
            
            # Stochastic Oscillator
            stoch_k = self._calculate_stochastic(data[col], tech_config['stochastic_period'])
            features_df[f'{col}_stoch_k_lag{self.config.min_lag}'] = stoch_k.shift(self.config.min_lag)
        
        # VIX technical indicators
        if 'vix' in data.columns:
            vix = data['vix']
            # VIX RSI
            vix_rsi = self._calculate_rsi(vix, 14)
            features_df[f'vix_rsi_14d_lag{self.config.min_lag}'] = vix_rsi.shift(self.config.min_lag)
            
            # VIX Bollinger Bands
            vix_bb_upper, vix_bb_lower, vix_bb_position = self._calculate_bollinger_bands(vix, 20, 2)
            features_df[f'vix_bb_position_lag{self.config.min_lag}'] = vix_bb_position.shift(self.config.min_lag)
        
        print(f"Created {len(features_df.columns)} technical indicator features")
        return features_df
    
    def create_cross_asset_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create cross-asset and statistical features."""
        print("Creating cross-asset and statistical features...")
        
        features_df = pd.DataFrame(index=data.index)
        cross_asset_config = self.feature_config['cross_asset_features']
        
        # 1. Spread differentials
        for spread1, spread2 in cross_asset_config['spread_pairs']:
            if spread1 in data.columns and spread2 in data.columns:
                # Raw spread difference
                spread_diff = data[spread1] - data[spread2]
                features_df[f'{spread1}_minus_{spread2}_lag{self.config.min_lag}'] = spread_diff.shift(self.config.min_lag)
                
                # Spread ratio
                spread_ratio = data[spread1] / data[spread2]
                features_df[f'{spread1}_over_{spread2}_lag{self.config.min_lag}'] = spread_ratio.shift(self.config.min_lag)
                
                # Spread difference momentum
                spread_diff_mom = spread_diff.pct_change(5)
                features_df[f'{spread1}_minus_{spread2}_mom5d_lag{self.config.min_lag}'] = spread_diff_mom.shift(self.config.min_lag)
        
        # 2. VIX regime indicators
        if 'vix' in data.columns:
            vix = data['vix']
            vix_percentiles = cross_asset_config['vix_regime_percentiles']
            
            # VIX regime (based on rolling percentiles)
            vix_rolling_pct = vix.rolling(252).quantile(0.5)  # 1-year median
            features_df[f'vix_above_median_lag{self.config.min_lag}'] = (vix > vix_rolling_pct).astype(int).shift(self.config.min_lag)
            
            # VIX percentile rank
            vix_pct_rank = vix.rolling(252).rank(pct=True)
            features_df[f'vix_percentile_rank_lag{self.config.min_lag}'] = vix_pct_rank.shift(self.config.min_lag)
            
            # VIX momentum
            for period in [5, 20]:
                vix_mom = vix.pct_change(period)
                features_df[f'vix_momentum_{period}d_lag{self.config.min_lag}'] = vix_mom.shift(self.config.min_lag)
        
        # 3. Yield curve features
        if 'us_3m_10y' in data.columns:
            yield_curve = data['us_3m_10y']
            # Yield curve level and changes
            features_df[f'yield_curve_level_lag{self.config.min_lag}'] = yield_curve.shift(self.config.min_lag)
            features_df[f'yield_curve_change_5d_lag{self.config.min_lag}'] = yield_curve.pct_change(5).shift(self.config.min_lag)
            features_df[f'yield_curve_change_20d_lag{self.config.min_lag}'] = yield_curve.pct_change(20).shift(self.config.min_lag)
        
        # 4. TSX vs Spread relationships
        if 'tsx' in data.columns and 'cad_oas' in data.columns:
            tsx = data['tsx']
            cad_oas = data['cad_oas']
            
            # TSX momentum vs CAD OAS changes
            tsx_mom = tsx.pct_change(20)
            cad_oas_change = cad_oas.pct_change(5)
            features_df[f'tsx_vs_cad_oas_momentum_lag{self.config.min_lag}'] = (tsx_mom / (cad_oas_change + 1e-8)).shift(self.config.min_lag)
            
            # TSX relative to moving average vs spread level
            tsx_ma_ratio = tsx / tsx.rolling(60).mean()
            features_df[f'tsx_ma60_ratio_vs_cad_oas_lag{self.config.min_lag}'] = (tsx_ma_ratio / (cad_oas / cad_oas.rolling(60).mean())).shift(self.config.min_lag)
        
        print(f"Created {len(features_df.columns)} cross-asset features")
        return features_df
    
    def create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features with proper lagging."""
        print("Creating statistical features...")
        
        features_df = pd.DataFrame(index=data.index)
        stat_config = self.feature_config['statistical_features']
        
        # Focus on key columns for statistical features
        key_columns = ['cad_oas', 'us_ig_oas', 'us_hy_oas', 'vix', 'tsx']
        
        for col in key_columns:
            if col not in data.columns:
                continue
            
            # Z-scores over rolling windows
            for window in stat_config['zscore_windows']:
                rolling_mean = data[col].rolling(window).mean()
                rolling_std = data[col].rolling(window).std()
                zscore = (data[col] - rolling_mean) / (rolling_std + 1e-8)
                features_df[f'{col}_zscore_{window}d_lag{self.config.min_lag}'] = zscore.shift(self.config.min_lag)
            
            # Percentile ranks
            for window in stat_config['percentile_windows']:
                pct_rank = data[col].rolling(window).rank(pct=True)
                features_df[f'{col}_percentile_rank_{window}d_lag{self.config.min_lag}'] = pct_rank.shift(self.config.min_lag)
            
            # Rolling skewness and kurtosis
            for window in stat_config['skew_kurt_windows']:
                returns = data[col].pct_change()
                rolling_skew = returns.rolling(window).skew()
                rolling_kurt = returns.rolling(window).kurt()
                features_df[f'{col}_skew_{window}d_lag{self.config.min_lag}'] = rolling_skew.shift(self.config.min_lag)
                features_df[f'{col}_kurt_{window}d_lag{self.config.min_lag}'] = rolling_kurt.shift(self.config.min_lag)
        
        print(f"Created {len(features_df.columns)} statistical features")
        return features_df
    
    def create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create regime and interaction features."""
        print("Creating regime and interaction features...")
        
        features_df = pd.DataFrame(index=data.index)
        
        # 1. Economic regime features
        if 'us_economic_regime' in data.columns:
            econ_regime = data['us_economic_regime']
            # Economic regime level and changes
            features_df[f'economic_regime_lag{self.config.min_lag}'] = econ_regime.shift(self.config.min_lag)
            features_df[f'economic_regime_change_lag{self.config.min_lag}'] = econ_regime.diff().shift(self.config.min_lag)
        
        # 2. Economic surprise interactions
        surprise_cols = ['us_growth_surprises', 'us_inflation_surprises', 'us_hard_data_surprises']
        for col in surprise_cols:
            if col in data.columns:
                surprise = data[col]
                # Interaction with CAD OAS
                if 'cad_oas' in data.columns:
                    interaction = surprise * data['cad_oas']
                    features_df[f'{col}_x_cad_oas_lag{self.config.min_lag}'] = interaction.shift(self.config.min_lag)
                
                # Surprise momentum
                surprise_mom = surprise.pct_change(5)
                features_df[f'{col}_momentum_5d_lag{self.config.min_lag}'] = surprise_mom.shift(self.config.min_lag)
        
        # 3. Equity revision features
        if 'us_equity_revisions' in data.columns:
            equity_rev = data['us_equity_revisions']
            # Equity revisions vs spread changes
            if 'cad_oas' in data.columns:
                spread_change = data['cad_oas'].pct_change(5)
                rev_spread_ratio = equity_rev / (spread_change + 1e-8)
                features_df[f'equity_rev_vs_spread_change_lag{self.config.min_lag}'] = rev_spread_ratio.shift(self.config.min_lag)
        
        # 4. VIX interaction features
        if 'vix' in data.columns and 'cad_oas' in data.columns:
            vix = data['vix']
            cad_oas = data['cad_oas']
            
            # VIX × CAD OAS interaction
            vix_cad_interaction = vix * cad_oas
            features_df[f'vix_x_cad_oas_lag{self.config.min_lag}'] = vix_cad_interaction.shift(self.config.min_lag)
            
            # VIX regime × spread change
            vix_regime = (vix > vix.rolling(252).median()).astype(int)
            spread_change = cad_oas.pct_change(5)
            vix_regime_spread = vix_regime * spread_change
            features_df[f'vix_regime_x_spread_change_lag{self.config.min_lag}'] = vix_regime_spread.shift(self.config.min_lag)
        
        # 5. Multi-asset momentum regimes
        momentum_assets = ['tsx', 'vix', 'cad_oas']
        available_assets = [col for col in momentum_assets if col in data.columns]
        
        if len(available_assets) >= 2:
            # Calculate momentum for each asset
            momentum_signals = pd.DataFrame(index=data.index)
            for asset in available_assets:
                momentum_signals[f'{asset}_momentum'] = data[asset].pct_change(20)
            
            # Average momentum across assets
            avg_momentum = momentum_signals.mean(axis=1)
            features_df[f'avg_asset_momentum_lag{self.config.min_lag}'] = avg_momentum.shift(self.config.min_lag)
            
            # Momentum dispersion (volatility of momentum across assets)
            momentum_dispersion = momentum_signals.std(axis=1)
            features_df[f'momentum_dispersion_lag{self.config.min_lag}'] = momentum_dispersion.shift(self.config.min_lag)
        
        print(f"Created {len(features_df.columns)} regime and interaction features")
        return features_df
    
    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different variables."""
        print("Creating interaction features...")
        
        features_df = pd.DataFrame(index=data.index)
        
        # 1. VIX × Spread interactions
        if 'vix' in data.columns and 'cad_oas' in data.columns:
            vix_lagged = data['vix'].shift(self.config.min_lag)
            cad_oas_lagged = data['cad_oas'].shift(self.config.min_lag)
            features_df[f'vix_x_cad_oas_lag{self.config.min_lag}'] = vix_lagged * cad_oas_lagged
            
            # VIX regime × CAD OAS
            vix_median = data['vix'].rolling(252).median()
            vix_regime = (data['vix'] > vix_median).astype(int)
            features_df[f'vix_regime_x_cad_oas_lag{self.config.min_lag}'] = vix_regime.shift(self.config.min_lag) * cad_oas_lagged
        
        # 2. Economic surprises × Spread interactions
        surprise_cols = ['us_growth_surprises', 'us_inflation_surprises', 'us_hard_data_surprises']
        for surprise_col in surprise_cols:
            if surprise_col in data.columns and 'cad_oas' in data.columns:
                surprise_lagged = data[surprise_col].shift(self.config.min_lag)
                cad_oas_lagged = data['cad_oas'].shift(self.config.min_lag)
                features_df[f'{surprise_col}_x_cad_oas_lag{self.config.min_lag}'] = surprise_lagged * cad_oas_lagged
        
        # 3. Equity revisions × Credit spread interactions
        if 'us_equity_revisions' in data.columns and 'cad_oas' in data.columns:
            equity_rev_lagged = data['us_equity_revisions'].shift(self.config.min_lag)
            cad_oas_lagged = data['cad_oas'].shift(self.config.min_lag)
            features_df[f'us_equity_revisions_x_cad_oas_lag{self.config.min_lag}'] = equity_rev_lagged * cad_oas_lagged
        
        # 4. Cross-asset momentum interactions
        if 'tsx' in data.columns and 'cad_oas' in data.columns:
            tsx_momentum = data['tsx'].pct_change(20)
            tsx_momentum_lagged = tsx_momentum.shift(self.config.min_lag)
            cad_oas_lagged = data['cad_oas'].shift(self.config.min_lag)
            features_df[f'tsx_momentum_x_cad_oas_lag{self.config.min_lag}'] = tsx_momentum_lagged * cad_oas_lagged
        
        # 5. Yield curve × Spread interactions
        if 'us_3m_10y' in data.columns and 'cad_oas' in data.columns:
            yield_curve_lagged = data['us_3m_10y'].shift(self.config.min_lag)
            cad_oas_lagged = data['cad_oas'].shift(self.config.min_lag)
            features_df[f'us_3m_10y_x_cad_oas_lag{self.config.min_lag}'] = yield_curve_lagged * cad_oas_lagged
        
        # 6. Multi-factor interactions
        if all(col in data.columns for col in ['vix', 'us_growth_surprises', 'cad_oas']):
            vix_lagged = data['vix'].shift(self.config.min_lag)
            growth_lagged = data['us_growth_surprises'].shift(self.config.min_lag)
            cad_oas_lagged = data['cad_oas'].shift(self.config.min_lag)
            features_df[f'vix_x_growth_x_cad_oas_lag{self.config.min_lag}'] = vix_lagged * growth_lagged * cad_oas_lagged
        
        # 7. Regime-dependent interactions
        if 'us_economic_regime' in data.columns and 'cad_oas' in data.columns:
            regime_lagged = data['us_economic_regime'].shift(self.config.min_lag)
            cad_oas_lagged = data['cad_oas'].shift(self.config.min_lag)
            features_df[f'economic_regime_x_cad_oas_lag{self.config.min_lag}'] = regime_lagged * cad_oas_lagged
        
        # 8. Volatility regime interactions
        if 'vix' in data.columns and 'cad_oas' in data.columns:
            vix_volatility = data['vix'].rolling(20).std()
            vix_vol_regime = (vix_volatility > vix_volatility.rolling(252).median()).astype(int)
            vix_vol_regime_lagged = vix_vol_regime.shift(self.config.min_lag)
            cad_oas_lagged = data['cad_oas'].shift(self.config.min_lag)
            features_df[f'vix_vol_regime_x_cad_oas_lag{self.config.min_lag}'] = vix_vol_regime_lagged * cad_oas_lagged
        
        # 9. Cross-spread interactions
        spread_cols = ['us_hy_oas', 'us_ig_oas']
        for spread_col in spread_cols:
            if spread_col in data.columns and 'cad_oas' in data.columns:
                spread_lagged = data[spread_col].shift(self.config.min_lag)
                cad_oas_lagged = data['cad_oas'].shift(self.config.min_lag)
                features_df[f'{spread_col}_x_cad_oas_lag{self.config.min_lag}'] = spread_lagged * cad_oas_lagged
        
        # 10. Momentum dispersion interactions
        if all(col in data.columns for col in ['tsx', 'vix', 'cad_oas']):
            tsx_momentum = data['tsx'].pct_change(20)
            vix_momentum = data['vix'].pct_change(20)
            momentum_dispersion = pd.concat([tsx_momentum, vix_momentum], axis=1).std(axis=1)
            momentum_dispersion_lagged = momentum_dispersion.shift(self.config.min_lag)
            cad_oas_lagged = data['cad_oas'].shift(self.config.min_lag)
            features_df[f'momentum_dispersion_x_cad_oas_lag{self.config.min_lag}'] = momentum_dispersion_lagged * cad_oas_lagged
        
        print(f"Created {len(features_df.columns)} interaction features")
        return features_df
    
    def _create_lagged_features(self, data: pd.DataFrame, feature_cols: List[str], 
                               lags: List[int]) -> pd.DataFrame:
        """Create lagged features ensuring minimum lag constraint."""
        lagged_df = pd.DataFrame(index=data.index)
        
        for col in feature_cols:
            if col not in data.columns:
                continue
                
            for lag in lags:
                if lag >= self.config.min_lag and lag <= self.config.max_lag:
                    lagged_df[f'{col}_lag{lag}'] = data[col].shift(lag)
        
        return lagged_df
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicators."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_diff = macd_line - signal_line
        return macd_line, signal_line, macd_diff
    
    def _calculate_bollinger_bands(self, series: pd.Series, window: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        position = (series - lower_band) / (upper_band - lower_band + 1e-10)
        return upper_band, lower_band, position
    
    def _calculate_stochastic(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Stochastic Oscillator."""
        low_min = series.rolling(period).min()
        high_max = series.rolling(period).max()
        stoch_k = 100 * (series - low_min) / (high_max - low_min + 1e-10)
        return stoch_k
