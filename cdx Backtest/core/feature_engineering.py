"""
Creative feature engineering for CDX backtesting.
Creates 150+ features from CDX spreads, yields, macro, equity, and interactions.
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from scipy import stats


class FeatureEngineer(ABC):
    """Abstract base class for feature engineering."""
    
    @abstractmethod
    def create_features(self, data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Create features from price data."""
        pass


class TechnicalFeatureEngineer(FeatureEngineer):
    """Technical analysis feature engineer."""
    
    def create_features(self, data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Create technical analysis features."""
        features = pd.DataFrame(index=data.index)
        
        # Get primary asset
        primary_asset = config.get('primary_asset', 'us_ig_cdx_er_index')
        if primary_asset not in data.columns:
            primary_asset = data.columns[0]
        
        price_series = data[primary_asset]
        
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
        
        return features.fillna(0)


class AdvancedFeatureEngineer(FeatureEngineer):
    """Advanced feature engineer with 150+ creative features."""
    
    def create_features(self, data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Create comprehensive feature set (150+ features).
        
        Categories:
        - CDX Spread Features
        - Rate Volatility Features
        - Yield Curve Features
        - Treasury Yield Features
        - Macro Surprise Features
        - Economic Indicators
        - Equity & Volatility Features
        - Commodities & FX
        - Financial Conditions
        - Cross-Asset Momentum
        - Time-Based Features
        - Interaction Features
        - Statistical Features
        """
        features = pd.DataFrame(index=data.index)
        
        # Momentum windows
        momentum_windows = config.get('momentum_windows', [5, 10, 20, 30, 60, 90, 120])
        zscore_windows = config.get('zscore_windows', [5, 10, 20, 40, 60, 120, 252])
        correlation_windows = config.get('correlation_windows', [20, 60, 120])
        
        print("Creating CDX Spread Features...")
        features = self._create_cdx_spread_features(data, features, momentum_windows, zscore_windows)
        
        print("Creating Rate Volatility Features...")
        features = self._create_rate_vol_features(data, features, momentum_windows, zscore_windows)
        
        print("Creating Yield Curve Features...")
        features = self._create_yield_curve_features(data, features, momentum_windows)
        
        print("Creating Treasury Yield Features...")
        features = self._create_treasury_yield_features(data, features, momentum_windows, zscore_windows)
        
        print("Creating Macro Surprise Features...")
        features = self._create_macro_surprise_features(data, features, momentum_windows, zscore_windows)
        
        print("Creating Economic Indicator Features...")
        features = self._create_economic_indicator_features(data, features, momentum_windows)
        
        print("Creating Equity & Volatility Features...")
        features = self._create_equity_volatility_features(data, features, momentum_windows, zscore_windows)
        
        print("Creating Commodities & FX Features...")
        features = self._create_commodities_fx_features(data, features, momentum_windows)
        
        print("Creating Financial Conditions Features...")
        features = self._create_financial_conditions_features(data, features, momentum_windows, zscore_windows)
        
        print("Creating Cross-Asset Momentum Features...")
        features = self._create_cross_asset_momentum_features(data, features, momentum_windows)
        
        print("Creating Time-Based Features...")
        features = self._create_time_based_features(data, features)
        
        print("Creating Interaction Features...")
        features = self._create_interaction_features(data, features)
        
        print("Creating Statistical Features...")
        features = self._create_statistical_features(data, features, correlation_windows)
        
        # Fill NaN values
        features = features.ffill().fillna(0)
        
        # Replace infinity values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        print(f"✓ Feature engineering complete: {len(features.columns)} features created")
        
        return features
    
    def _create_cdx_spread_features(self, data: pd.DataFrame, features: pd.DataFrame,
                                   momentum_windows: List[int], zscore_windows: List[int]) -> pd.DataFrame:
        """Create CDX spread features."""
        for col in ['ig_cdx', 'hy_cdx']:
            if col not in data.columns:
                continue
            
            # Momentum features
            for window in momentum_windows:
                features[f'{col}_mom_{window}'] = data[col].pct_change(window)
            
            # Z-score features
            for window in zscore_windows:
                rolling_mean = data[col].rolling(window).mean()
                rolling_std = data[col].rolling(window).std()
                features[f'{col}_zscore_{window}'] = (data[col] - rolling_mean) / (rolling_std + 1e-8)
            
            # Percentile features
            for window in [60, 120, 252]:
                rolling_percentile = data[col].rolling(window).apply(
                    lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x.dropna()) > 0 else 0.5
                )
                features[f'{col}_percentile_{window}'] = rolling_percentile
            
            # Change features
            for window in [1, 3, 5, 10, 20]:
                features[f'{col}_change_{window}'] = data[col].diff(window)
        
        # CDX spread ratios
        if 'ig_cdx' in data.columns and 'hy_cdx' in data.columns:
            features['ig_hy_ratio'] = data['ig_cdx'] / (data['hy_cdx'] + 1e-8)
            features['ig_hy_ratio_mom_20'] = features['ig_hy_ratio'].pct_change(20)
            
            # 6-month average ratios
            for col in ['ig_cdx', 'hy_cdx']:
                avg_6m = data[col].rolling(120).mean()
                features[f'{col}_vs_6m_avg'] = data[col] / (avg_6m + 1e-8) - 1
        
        return features
    
    def _create_rate_vol_features(self, data: pd.DataFrame, features: pd.DataFrame,
                                 momentum_windows: List[int], zscore_windows: List[int]) -> pd.DataFrame:
        """Create rate volatility features."""
        if 'rate_vol' not in data.columns:
            return features
        
        # Momentum and z-scores
        for window in momentum_windows:
            features[f'rate_vol_mom_{window}'] = data['rate_vol'].pct_change(window)
        
        for window in zscore_windows:
            rolling_mean = data['rate_vol'].rolling(window).mean()
            rolling_std = data['rate_vol'].rolling(window).std()
            features[f'rate_vol_zscore_{window}'] = (data['rate_vol'] - rolling_mean) / (rolling_std + 1e-8)
        
        # Regime detection (high/medium/low)
        percentiles_20 = data['rate_vol'].rolling(252).quantile(0.2)
        percentiles_80 = data['rate_vol'].rolling(252).quantile(0.8)
        features['rate_vol_regime_high'] = (data['rate_vol'] > percentiles_80).astype(int)
        features['rate_vol_regime_low'] = (data['rate_vol'] < percentiles_20).astype(int)
        
        # Volatility clustering (GARCH-like)
        returns = data['rate_vol'].pct_change()
        features['rate_vol_cluster_20'] = returns.rolling(20).std()
        features['rate_vol_cluster_60'] = returns.rolling(60).std()
        
        # Correlation with CDX ER
        if 'us_ig_cdx_er_index' in data.columns:
            cdx_returns = data['us_ig_cdx_er_index'].pct_change()
            vol_returns = returns
            features['rate_vol_cdx_corr_60'] = vol_returns.rolling(60).corr(cdx_returns)
        
        return features
    
    def _create_yield_curve_features(self, data: pd.DataFrame, features: pd.DataFrame,
                                   momentum_windows: List[int]) -> pd.DataFrame:
        """Create yield curve features."""
        # Yield curve spreads
        for col in ['us_2s10s_spread', 'us_3m10y_spread', 'us_5y30y_spread']:
            if col not in data.columns:
                continue
            
            # Momentum
            for window in momentum_windows[:5]:  # Use shorter windows
                features[f'{col}_mom_{window}'] = data[col].pct_change(window)
            
            # Change features
            for window in [1, 3, 5, 10]:
                features[f'{col}_change_{window}'] = data[col].diff(window)
            
            # Inversion indicators
            if 'spread' in col:
                features[f'{col}_inverted'] = (data[col] < 0).astype(int)
        
        # Breakeven inflation
        if 'us_10y_breakeven' in data.columns:
            for window in momentum_windows[:5]:
                features[f'us_10y_breakeven_mom_{window}'] = data['us_10y_breakeven'].pct_change(window)
        
        # Yield curve twist (difference between spreads)
        if 'us_2s10s_spread' in data.columns and 'us_3m10y_spread' in data.columns:
            features['yield_curve_twist'] = data['us_2s10s_spread'] - data['us_3m10y_spread']
            features['yield_curve_twist_mom_20'] = features['yield_curve_twist'].pct_change(20)
        
        return features
    
    def _create_treasury_yield_features(self, data: pd.DataFrame, features: pd.DataFrame,
                                      momentum_windows: List[int], zscore_windows: List[int]) -> pd.DataFrame:
        """Create treasury yield features."""
        yield_cols = ['us_10y_yield', 'us_2y_yield', 'us_30y_yield', 'us_3m_tbill']
        
        for col in yield_cols:
            if col not in data.columns:
                continue
            
            # Momentum
            for window in momentum_windows[:7]:  # Use shorter windows
                features[f'{col}_mom_{window}'] = data[col].pct_change(window)
            
            # Z-scores
            for window in zscore_windows[:5]:
                rolling_mean = data[col].rolling(window).mean()
                rolling_std = data[col].rolling(window).std()
                features[f'{col}_zscore_{window}'] = (data[col] - rolling_mean) / (rolling_std + 1e-8)
            
            # Change features
            for window in [1, 3, 5, 10, 20]:
                features[f'{col}_change_{window}'] = data[col].diff(window)
        
        # Fed funds rate differentials
        if 'fed_funds_rate' in data.columns:
            for col in yield_cols:
                if col in data.columns:
                    features[f'{col}_vs_ffr'] = data[col] - data['fed_funds_rate']
                    features[f'{col}_vs_ffr_mom_20'] = features[f'{col}_vs_ffr'].pct_change(20)
        
        return features
    
    def _create_macro_surprise_features(self, data: pd.DataFrame, features: pd.DataFrame,
                                      momentum_windows: List[int], zscore_windows: List[int]) -> pd.DataFrame:
        """Create macro surprise features."""
        surprise_cols = ['us_growth_surprises', 'us_inflation_surprises', 'us_hard_data_surprises']
        
        for col in surprise_cols:
            if col not in data.columns:
                continue
            
            # Momentum
            for window in momentum_windows[:5]:
                features[f'{col}_mom_{window}'] = data[col].pct_change(window)
            
            # Z-scores
            for window in zscore_windows[:5]:
                rolling_mean = data[col].rolling(window).mean()
                rolling_std = data[col].rolling(window).std()
                features[f'{col}_zscore_{window}'] = (data[col] - rolling_mean) / (rolling_std + 1e-8)
            
            # Regime detection
            features[f'{col}_positive'] = (data[col] > 0).astype(int)
            features[f'{col}_negative'] = (data[col] < 0).astype(int)
        
        # Equity revisions
        if 'us_equity_revisions' in data.columns:
            for window in momentum_windows[:5]:
                features[f'us_equity_revisions_mom_{window}'] = data['us_equity_revisions'].pct_change(window)
        
        return features
    
    def _create_economic_indicator_features(self, data: pd.DataFrame, features: pd.DataFrame,
                                          momentum_windows: List[int]) -> pd.DataFrame:
        """Create economic indicator features."""
        # LEI YOY
        if 'us_lei_yoy' in data.columns:
            for window in momentum_windows[:5]:
                features[f'us_lei_yoy_mom_{window}'] = data['us_lei_yoy'].pct_change(window)
            features['us_lei_yoy_change_20'] = data['us_lei_yoy'].diff(20)
        
        # Economic regime
        if 'us_economic_regime' in data.columns:
            features['us_economic_regime_change'] = data['us_economic_regime'].diff()
            features['us_economic_regime_positive'] = (data['us_economic_regime'] > 0).astype(int)
        
        # LEI vs CDX ER correlation
        if 'us_lei_yoy' in data.columns and 'us_ig_cdx_er_index' in data.columns:
            lei_returns = data['us_lei_yoy'].pct_change()
            cdx_returns = data['us_ig_cdx_er_index'].pct_change()
            features['lei_cdx_corr_120'] = lei_returns.rolling(120).corr(cdx_returns)
        
        return features
    
    def _create_equity_volatility_features(self, data: pd.DataFrame, features: pd.DataFrame,
                                         momentum_windows: List[int], zscore_windows: List[int]) -> pd.DataFrame:
        """Create equity and volatility features."""
        # SPX Total Return
        if 'spx_tr' in data.columns:
            for window in momentum_windows[:7]:
                features[f'spx_tr_mom_{window}'] = data['spx_tr'].pct_change(window)
        
        # Nasdaq 100
        if 'nasdaq100' in data.columns:
            for window in momentum_windows[:7]:
                features[f'nasdaq100_mom_{window}'] = data['nasdaq100'].pct_change(window)
        
        # VIX features
        if 'vix' in data.columns:
            for window in momentum_windows[:7]:
                features[f'vix_mom_{window}'] = data['vix'].pct_change(window)
            
            for window in zscore_windows[:5]:
                rolling_mean = data['vix'].rolling(window).mean()
                rolling_std = data['vix'].rolling(window).std()
                features[f'vix_zscore_{window}'] = (data['vix'] - rolling_mean) / (rolling_std + 1e-8)
            
            # VIX regime
            percentiles_20 = data['vix'].rolling(252).quantile(0.2)
            percentiles_80 = data['vix'].rolling(252).quantile(0.8)
            features['vix_regime_high'] = (data['vix'] > percentiles_80).astype(int)
            features['vix_regime_low'] = (data['vix'] < percentiles_20).astype(int)
        
        # VVIX features
        if 'vvix' in data.columns:
            for window in momentum_windows[:5]:
                features[f'vvix_mom_{window}'] = data['vvix'].pct_change(window)
            
            # VVIX vs VIX ratio
            if 'vix' in data.columns:
                features['vvix_vix_ratio'] = data['vvix'] / (data['vix'] + 1e-8)
                features['vvix_vix_ratio_mom_20'] = features['vvix_vix_ratio'].pct_change(20)
        
        # Equity vs CDX ER correlation
        if 'us_ig_cdx_er_index' in data.columns:
            cdx_returns = data['us_ig_cdx_er_index'].pct_change()
            
            if 'spx_tr' in data.columns:
                spx_returns = data['spx_tr'].pct_change()
                for window in [20, 60, 120]:
                    features[f'spx_cdx_corr_{window}'] = spx_returns.rolling(window).corr(cdx_returns)
        
        return features
    
    def _create_commodities_fx_features(self, data: pd.DataFrame, features: pd.DataFrame,
                                      momentum_windows: List[int]) -> pd.DataFrame:
        """Create commodities and FX features."""
        commodity_cols = ['gold_spot', 'dollar_index', 'wti_crude']
        
        for col in commodity_cols:
            if col not in data.columns:
                continue
            
            for window in momentum_windows[:7]:
                features[f'{col}_mom_{window}'] = data[col].pct_change(window)
        
        # Dollar index vs CDX ER correlation
        if 'dollar_index' in data.columns and 'us_ig_cdx_er_index' in data.columns:
            dollar_returns = data['dollar_index'].pct_change()
            cdx_returns = data['us_ig_cdx_er_index'].pct_change()
            for window in [20, 60, 120]:
                features[f'dollar_cdx_corr_{window}'] = dollar_returns.rolling(window).corr(cdx_returns)
        
        # Gold vs CDX ER correlation
        if 'gold_spot' in data.columns and 'us_ig_cdx_er_index' in data.columns:
            gold_returns = data['gold_spot'].pct_change()
            cdx_returns = data['us_ig_cdx_er_index'].pct_change()
            features['gold_cdx_corr_60'] = gold_returns.rolling(60).corr(cdx_returns)
        
        return features
    
    def _create_financial_conditions_features(self, data: pd.DataFrame, features: pd.DataFrame,
                                            momentum_windows: List[int], zscore_windows: List[int]) -> pd.DataFrame:
        """Create financial conditions features."""
        if 'us_bloomberg_fci' not in data.columns:
            return features
        
        # Momentum
        for window in momentum_windows[:7]:
            features[f'us_bloomberg_fci_mom_{window}'] = data['us_bloomberg_fci'].pct_change(window)
        
        # Z-scores
        for window in zscore_windows[:5]:
            rolling_mean = data['us_bloomberg_fci'].rolling(window).mean()
            rolling_std = data['us_bloomberg_fci'].rolling(window).std()
            features[f'us_bloomberg_fci_zscore_{window}'] = (data['us_bloomberg_fci'] - rolling_mean) / (rolling_std + 1e-8)
        
        # Regime detection
        percentiles_20 = data['us_bloomberg_fci'].rolling(252).quantile(0.2)
        percentiles_80 = data['us_bloomberg_fci'].rolling(252).quantile(0.8)
        features['us_bloomberg_fci_regime_high'] = (data['us_bloomberg_fci'] > percentiles_80).astype(int)
        features['us_bloomberg_fci_regime_low'] = (data['us_bloomberg_fci'] < percentiles_20).astype(int)
        
        return features
    
    def _create_cross_asset_momentum_features(self, data: pd.DataFrame, features: pd.DataFrame,
                                            momentum_windows: List[int]) -> pd.DataFrame:
        """Create cross-asset momentum features."""
        asset_cols = []
        for col in ['spx_tr', 'nasdaq100', 'vix', 'gold_spot', 'dollar_index']:
            if col in data.columns:
                asset_cols.append(col)
        
        if len(asset_cols) < 2:
            return features
        
        # Multi-asset momentum score
        momentum_scores = []
        window = 20  # Use 20-day momentum
        for col in asset_cols:
            momentum = data[col].pct_change(window)
            momentum_scores.append(momentum)
        
        if momentum_scores:
            momentum_df = pd.DataFrame(momentum_scores, index=asset_cols).T
            momentum_df.index = data.index
            features['multi_asset_momentum_score'] = momentum_df.mean(axis=1)
            
            # Momentum confirmation count
            features['momentum_confirmation_count'] = (momentum_df > 0).sum(axis=1)
            features['momentum_confirmation_pct'] = features['momentum_confirmation_count'] / len(asset_cols)
            
            # Cross-asset momentum divergence
            features['momentum_divergence'] = momentum_df.std(axis=1)
        
        return features
    
    def _create_time_based_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        # Day of week
        features['day_of_week'] = data.index.dayofweek
        features['is_monday'] = (features['day_of_week'] == 0).astype(int)
        features['is_friday'] = (features['day_of_week'] == 4).astype(int)
        
        # Month of year
        features['month'] = data.index.month
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        # Quarter
        features['quarter'] = data.index.quarter
        
        return features
    
    def _create_interaction_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        # CDX spread * VIX interaction
        if 'ig_cdx' in data.columns and 'vix' in data.columns:
            features['ig_cdx_vix_interaction'] = data['ig_cdx'] * data['vix']
            features['ig_cdx_vix_ratio'] = data['ig_cdx'] / (data['vix'] + 1e-8)
        
        # Yield curve * CDX spread interaction
        if 'us_2s10s_spread' in data.columns and 'ig_cdx' in data.columns:
            features['yield_curve_cdx_interaction'] = data['us_2s10s_spread'] * data['ig_cdx']
        
        # Credit spread * equity momentum interaction
        if 'ig_cdx' in data.columns and 'spx_tr' in data.columns:
            cdx_mom = data['ig_cdx'].pct_change(20)
            spx_mom = data['spx_tr'].pct_change(20)
            features['credit_equity_momentum_interaction'] = cdx_mom * spx_mom
        
        # Macro surprise * CDX momentum interaction
        if 'us_growth_surprises' in data.columns and 'ig_cdx' in data.columns:
            cdx_mom = data['ig_cdx'].pct_change(10)
            features['macro_cdx_interaction'] = data['us_growth_surprises'] * cdx_mom
        
        return features
    
    def _create_statistical_features(self, data: pd.DataFrame, features: pd.DataFrame,
                                   correlation_windows: List[int]) -> pd.DataFrame:
        """Create statistical features."""
        if 'us_ig_cdx_er_index' not in data.columns:
            return features
        
        cdx_returns = data['us_ig_cdx_er_index'].pct_change()
        
        # Rolling correlations with various assets
        for asset_col in ['spx_tr', 'dollar_index', 'gold_spot', 'vix', 'rate_vol']:
            if asset_col not in data.columns:
                continue
            
            asset_returns = data[asset_col].pct_change()
            for window in correlation_windows:
                features[f'cdx_{asset_col}_corr_{window}'] = cdx_returns.rolling(window).corr(asset_returns)
        
        # Rolling betas
        if 'spx_tr' in data.columns:
            spx_returns = data['spx_tr'].pct_change()
            for window in [60, 120]:
                covariance = cdx_returns.rolling(window).cov(spx_returns)
                spx_var = spx_returns.rolling(window).var()
                features[f'cdx_spx_beta_{window}'] = covariance / (spx_var + 1e-8)
        
        # Rolling skewness and kurtosis
        for window in [60, 120]:
            features[f'cdx_skewness_{window}'] = cdx_returns.rolling(window).skew()
            features[f'cdx_kurtosis_{window}'] = cdx_returns.rolling(window).kurt()
        
        # Enhanced regime indicators
        features = self._create_regime_indicators(data, features, cdx_returns)
        
        return features
    
    def _create_regime_indicators(self, data: pd.DataFrame, features: pd.DataFrame, 
                                 returns: pd.Series) -> pd.DataFrame:
        """Create enhanced regime indicators (volatility, trend, momentum regimes)."""
        primary_asset = data.columns[0] if len(data.columns) > 0 else None
        if primary_asset is None:
            return features
        
        price_series = data[primary_asset]
        
        # Volatility regime
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std()
        vol_252 = returns.rolling(252).std()
        
        # High/medium/low volatility regimes
        vol_percentile_20 = vol_252.quantile(0.2)
        vol_percentile_80 = vol_252.quantile(0.8)
        features['vol_regime_high'] = (vol_20 > vol_percentile_80).astype(int)
        features['vol_regime_low'] = (vol_20 < vol_percentile_20).astype(int)
        features['vol_regime_rising'] = (vol_20 > vol_60).astype(int)
        features['vol_regime_falling'] = (vol_20 < vol_60).astype(int)
        
        # Trend regime
        price_ma_20 = price_series.rolling(20).mean()
        price_ma_60 = price_series.rolling(60).mean()
        price_ma_252 = price_series.rolling(252).mean()
        
        features['trend_regime_uptrend'] = ((price_ma_20 > price_ma_60) & (price_ma_60 > price_ma_252)).astype(int)
        features['trend_regime_downtrend'] = ((price_ma_20 < price_ma_60) & (price_ma_60 < price_ma_252)).astype(int)
        features['trend_regime_above_ma'] = (price_series > price_ma_252).astype(int)
        
        # Momentum regime
        mom_20 = returns.rolling(20).mean()
        mom_60 = returns.rolling(60).mean()
        mom_percentile_20 = mom_60.quantile(0.2)
        mom_percentile_80 = mom_60.quantile(0.8)
        
        features['momentum_regime_high'] = (mom_20 > mom_percentile_80).astype(int)
        features['momentum_regime_low'] = (mom_20 < mom_percentile_20).astype(int)
        features['momentum_regime_positive'] = (mom_20 > 0).astype(int)
        
        # VIX regime (if available)
        if 'vix' in data.columns:
            vix_252 = data['vix'].rolling(252)
            vix_percentile_20 = vix_252.quantile(0.2)
            vix_percentile_80 = vix_252.quantile(0.8)
            features['vix_regime_high'] = (data['vix'] > vix_percentile_80).astype(int)
            features['vix_regime_low'] = (data['vix'] < vix_percentile_20).astype(int)
        
        # Credit spread regime (if available)
        if 'ig_cdx' in data.columns:
            cdx_252 = data['ig_cdx'].rolling(252)
            cdx_percentile_20 = cdx_252.quantile(0.2)
            cdx_percentile_80 = cdx_252.quantile(0.8)
            features['cdx_regime_wide'] = (data['ig_cdx'] > cdx_percentile_80).astype(int)
            features['cdx_regime_tight'] = (data['ig_cdx'] < cdx_percentile_20).astype(int)
        
        # Combined regime score (0-1 scale)
        regime_score = (
            features['vol_regime_low'].fillna(0) * 0.2 +
            features['trend_regime_uptrend'].fillna(0) * 0.3 +
            features['momentum_regime_positive'].fillna(0) * 0.3
        )
        if 'vix_regime_low' in features:
            regime_score += features['vix_regime_low'].fillna(0) * 0.2
        features['combined_regime_score'] = regime_score
        
        return features


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
        return features.drop(columns=to_drop, errors='ignore')

