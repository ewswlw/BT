"""
Comprehensive tests for feature engineering components.
Tests TechnicalFeatureEngineer, CrossAssetFeatureEngineer, MultiAssetFeatureEngineer.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cad_ig_er_index_backtesting.core.feature_engineering import (
    TechnicalFeatureEngineer, CrossAssetFeatureEngineer, MultiAssetFeatureEngineer
)


class TestTechnicalFeatureEngineer:
    """Test TechnicalFeatureEngineer."""
    
    def create_test_data(self):
        """Create test price data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        return pd.DataFrame({'price': prices}, index=dates)
    
    def test_momentum_features(self):
        """Test momentum feature creation."""
        data = self.create_test_data()
        config = {
            'primary_asset': 'price',
            'momentum_periods': [5, 10, 20]
        }
        
        engineer = TechnicalFeatureEngineer()
        features = engineer.create_features(data, config)
        
        assert 'mom_5' in features.columns
        assert 'mom_10' in features.columns
        assert 'mom_20' in features.columns
        assert len(features) == len(data)
    
    def test_volatility_features(self):
        """Test volatility feature creation."""
        data = self.create_test_data()
        config = {
            'primary_asset': 'price',
            'volatility_windows': [10, 20]
        }
        
        engineer = TechnicalFeatureEngineer()
        features = engineer.create_features(data, config)
        
        assert 'vol_10' in features.columns
        assert 'vol_20' in features.columns
    
    def test_sma_features(self):
        """Test SMA deviation features."""
        data = self.create_test_data()
        config = {
            'primary_asset': 'price',
            'sma_windows': [10, 20]
        }
        
        engineer = TechnicalFeatureEngineer()
        features = engineer.create_features(data, config)
        
        assert 'sma_10_dev' in features.columns
        assert 'sma_20_dev' in features.columns
    
    def test_macd_features(self):
        """Test MACD feature creation."""
        data = self.create_test_data()
        config = {
            'primary_asset': 'price',
            'include_macd': True,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        }
        
        engineer = TechnicalFeatureEngineer()
        features = engineer.create_features(data, config)
        
        assert 'macd_diff' in features.columns
    
    def test_rsi_features(self):
        """Test RSI feature creation."""
        data = self.create_test_data()
        config = {
            'primary_asset': 'price',
            'include_rsi': True,
            'rsi_period': 14
        }
        
        engineer = TechnicalFeatureEngineer()
        features = engineer.create_features(data, config)
        
        assert 'rsi' in features.columns
        assert features['rsi'].min() >= 0
        assert features['rsi'].max() <= 100
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands features."""
        data = self.create_test_data()
        config = {
            'primary_asset': 'price',
            'include_bollinger': True,
            'bollinger_window': 20,
            'bollinger_std': 2
        }
        
        engineer = TechnicalFeatureEngineer()
        features = engineer.create_features(data, config)
        
        assert 'bb_upper' in features.columns
        assert 'bb_lower' in features.columns
        assert 'bb_position' in features.columns
    
    def test_oas_features(self):
        """Test OAS features."""
        data = pd.DataFrame({
            'price': np.random.randn(100) * 10 + 100,
            'cad_oas': np.random.randn(100) * 5 + 50,
            'us_ig_oas': np.random.randn(100) * 5 + 50,
            'vix': np.random.randn(100) * 5 + 20
        }, index=pd.date_range('2020-01-01', periods=100, freq='D'))
        
        config = {
            'primary_asset': 'price',
            'include_oas_features': True,
            'oas_momentum_period': 4
        }
        
        engineer = TechnicalFeatureEngineer()
        features = engineer.create_features(data, config)
        
        assert 'cad_oas_mom4' in features.columns
        assert 'us_ig_oas_mom4' in features.columns
        assert 'vix_mom4' in features.columns
    
    def test_missing_primary_asset(self):
        """Test with missing primary asset falls back to first column."""
        data = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100)
        })
        config = {
            'primary_asset': 'nonexistent',
            'momentum_periods': [5]
        }
        
        engineer = TechnicalFeatureEngineer()
        features = engineer.create_features(data, config)
        
        assert 'mom_5' in features.columns
    
    def test_fill_na_value(self):
        """Test fill_na_value configuration."""
        data = self.create_test_data()
        data.iloc[10:15, 0] = np.nan  # Introduce NaN
        
        config = {
            'primary_asset': 'price',
            'momentum_periods': [5],
            'fill_na_value': 999.0
        }
        
        engineer = TechnicalFeatureEngineer()
        features = engineer.create_features(data, config)
        
        assert features['mom_5'].isna().sum() == 0


class TestCrossAssetFeatureEngineer:
    """Test CrossAssetFeatureEngineer."""
    
    def create_test_data(self):
        """Create test multi-asset data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'cad_ig_er_index': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'us_hy_er_index': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'us_ig_er_index': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'tsx': 100 + np.cumsum(np.random.randn(100) * 0.5)
        }, index=dates)
    
    def test_cross_asset_momentum(self):
        """Test cross-asset momentum features."""
        data = self.create_test_data()
        config = {
            'momentum_assets': ['cad_ig_er_index', 'us_hy_er_index', 'us_ig_er_index', 'tsx'],
            'momentum_lookback_days': 10
        }
        
        engineer = CrossAssetFeatureEngineer()
        features = engineer.create_features(data, config)
        
        # Should create momentum features for each asset
        assert len(features.columns) > 0
    
    def test_missing_assets(self):
        """Test with missing assets."""
        data = pd.DataFrame({
            'cad_ig_er_index': np.random.randn(100)
        })
        config = {
            'momentum_assets': ['cad_ig_er_index', 'missing_asset'],
            'momentum_lookback_days': 10
        }
        
        engineer = CrossAssetFeatureEngineer()
        # Should handle missing assets gracefully
        features = engineer.create_features(data, config)
        assert len(features) == len(data)


class TestMultiAssetFeatureEngineer:
    """Test MultiAssetFeatureEngineer."""
    
    def create_test_data(self):
        """Create test multi-asset data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'tsx': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'us_hy': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'cad_ig': 100 + np.cumsum(np.random.randn(100) * 0.5)
        }, index=dates)
    
    def test_multi_asset_features(self):
        """Test multi-asset feature creation."""
        data = self.create_test_data()
        config = {
            'momentum_assets_map': {
                'tsx': 'tsx',
                'us_hy': 'us_hy',
                'cad_ig': 'cad_ig'
            },
            'momentum_lookback_days': 20
        }
        
        engineer = MultiAssetFeatureEngineer()
        features = engineer.create_features(data, config)
        
        assert len(features.columns) > 0
        assert len(features) == len(data)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

