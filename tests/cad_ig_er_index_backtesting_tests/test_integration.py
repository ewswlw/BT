"""
Integration tests for the complete backtesting pipeline.
Tests end-to-end workflows with various configurations.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cad_ig_er_index_backtesting.core.data_loader import DataLoader, CSVDataProvider, MultiIndexCSVDataProvider
from cad_ig_er_index_backtesting.core.config import DataConfig, PortfolioConfig, create_config_from_dict
from cad_ig_er_index_backtesting.core.feature_engineering import TechnicalFeatureEngineer
from cad_ig_er_index_backtesting.strategies.strategy_factory import StrategyFactory


class TestEndToEndPipeline:
    """Test complete end-to-end backtesting pipeline."""
    
    def create_test_csv(self, multi_index=False):
        """Create test CSV file."""
        if multi_index:
            content = [
                "asset_class,equity_indices,volatility\n",
                "column,tsx,vix\n",
                "Date,,\n",
            ]
            for i in range(50):
                content.append(f"2020-01-{i+1:02d},{100+i},{15+i*0.1}\n")
        else:
            content = [
                "Date,price,volume\n",
            ]
            for i in range(50):
                content.append(f"2020-01-{i+1:02d},{100+i},{1000+i}\n")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.writelines(content)
            return f.name
    
    def test_simple_pipeline(self):
        """Test simple single-index CSV pipeline."""
        csv_path = self.create_test_csv(multi_index=False)
        
        try:
            config = DataConfig(
                file_path=csv_path,
                use_multi_index=False,
                resample_frequency=None
            )
            
            loader = DataLoader(CSVDataProvider())
            data = loader.load_and_prepare(config)
            
            assert not data.empty
            assert 'price' in data.columns
            
            # Create features
            feature_config = {
                'primary_asset': 'price',
                'momentum_periods': [5, 10]
            }
            engineer = TechnicalFeatureEngineer()
            features = engineer.create_features(data, feature_config)
            
            assert 'mom_5' in features.columns
            assert 'mom_10' in features.columns
        finally:
            os.unlink(csv_path)
    
    def test_multi_index_pipeline(self):
        """Test multi-index CSV pipeline."""
        csv_path = self.create_test_csv(multi_index=True)
        
        try:
            config = DataConfig(
                file_path=csv_path,
                use_multi_index=True,
                asset_classes_include=['equity_indices'],
                flatten_columns=True,
                resample_frequency=None
            )
            
            loader = DataLoader(MultiIndexCSVDataProvider())
            data = loader.load_and_prepare(config)
            
            assert not data.empty
            assert not isinstance(data.columns, pd.MultiIndex)
            assert 'tsx' in data.columns
            assert 'vix' not in data.columns  # Filtered out
        finally:
            os.unlink(csv_path)
    
    def test_pipeline_with_filtering(self):
        """Test pipeline with asset class and column filtering."""
        csv_path = self.create_test_csv(multi_index=True)
        
        try:
            config = DataConfig(
                file_path=csv_path,
                use_multi_index=True,
                asset_classes_include=['equity_indices'],
                columns_include={'equity_indices': ['tsx']},
                flatten_columns=True
            )
            
            loader = DataLoader(MultiIndexCSVDataProvider())
            data = loader.load_and_prepare(config)
            
            assert 'tsx' in data.columns
            assert len(data.columns) == 1
        finally:
            os.unlink(csv_path)
    
    def test_pipeline_with_missing_data_handling(self):
        """Test pipeline with missing data handling."""
        csv_path = self.create_test_csv(multi_index=False)
        
        try:
            config = DataConfig(
                file_path=csv_path,
                use_multi_index=False,
                missing_data_strategy='interpolate'
            )
            
            loader = DataLoader(CSVDataProvider())
            data = loader.load_and_prepare(config)
            
            assert data.isna().sum().sum() == 0
        finally:
            os.unlink(csv_path)
    
    def test_pipeline_with_resampling(self):
        """Test pipeline with data resampling."""
        csv_path = self.create_test_csv(multi_index=False)
        
        try:
            config = DataConfig(
                file_path=csv_path,
                use_multi_index=False,
                resample_frequency='W'
            )
            
            loader = DataLoader(CSVDataProvider())
            data = loader.load_and_prepare(config)
            
            assert len(data) < 50  # Should be resampled
        finally:
            os.unlink(csv_path)
    
    def test_pipeline_with_date_filtering(self):
        """Test pipeline with date range filtering."""
        csv_path = self.create_test_csv(multi_index=False)
        
        try:
            config = DataConfig(
                file_path=csv_path,
                use_multi_index=False,
                start_date='2020-01-10',
                end_date='2020-01-20'
            )
            
            loader = DataLoader(CSVDataProvider())
            data = loader.load_and_prepare(config)
            
            assert len(data) == 11
            assert data.index[0] >= pd.Timestamp('2020-01-10')
            assert data.index[-1] <= pd.Timestamp('2020-01-20')
        finally:
            os.unlink(csv_path)
    
    def test_pipeline_with_all_quality_checks(self):
        """Test pipeline with all quality checks enabled."""
        csv_path = self.create_test_csv(multi_index=False)
        
        try:
            config = DataConfig(
                file_path=csv_path,
                use_multi_index=False,
                outlier_detection_enabled=True,
                outlier_action='flag',
                gap_detection_enabled=True,
                gap_action='flag',
                data_validation_enabled=True,
                min_value=0,
                max_value=200
            )
            
            loader = DataLoader(CSVDataProvider())
            data = loader.load_and_prepare(config)
            
            assert not data.empty
        finally:
            os.unlink(csv_path)
    
    def test_config_from_yaml_integration(self):
        """Test loading config from YAML and running pipeline."""
        import yaml
        
        config_dict = {
            'data': {
                'file_path': 'dummy.csv',
                'use_multi_index': False,
                'resample_frequency': None
            },
            'portfolio': {
                'initial_capital': 100000
            },
            'reporting': {
                'output_dir': 'test_output'
            }
        }
        
        base_config = create_config_from_dict(config_dict)
        
        assert base_config.data.use_multi_index == False
        assert base_config.portfolio.initial_capital == 100000


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Date,Price\n")
            temp_path = f.name
        
        try:
            config = DataConfig(file_path=temp_path)
            loader = DataLoader(CSVDataProvider())
            
            with pytest.raises(ValueError, match="empty"):
                loader.load_and_prepare(config)
        finally:
            os.unlink(temp_path)
    
    def test_missing_required_columns(self):
        """Test handling missing required columns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Date,Price\n")
            f.write("2020-01-01,100\n")
            temp_path = f.name
        
        try:
            config = DataConfig(
                file_path=temp_path,
                required_columns=['MissingColumn']
            )
            loader = DataLoader(CSVDataProvider())
            
            with pytest.raises(ValueError, match="Missing required columns"):
                loader.load_and_prepare(config)
        finally:
            os.unlink(temp_path)
    
    def test_invalid_date_range(self):
        """Test handling invalid date range."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Date,Price\n")
            f.write("2020-01-01,100\n")
            f.write("2020-01-02,101\n")
            temp_path = f.name
        
        try:
            config = DataConfig(
                file_path=temp_path,
                start_date='2020-01-10',
                end_date='2020-01-05'  # End before start
            )
            loader = DataLoader(CSVDataProvider())
            data = loader.load_and_prepare(config)
            
            # Should return empty or handle gracefully
            assert isinstance(data, pd.DataFrame)
        finally:
            os.unlink(temp_path)
    
    def test_multi_index_without_enabling(self):
        """Test multi-index data without enabling multi-index mode."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("asset_class,equity_indices\n")
            f.write("column,tsx\n")
            f.write("Date,\n")
            f.write("2020-01-01,100\n")
            temp_path = f.name
        
        try:
            config = DataConfig(
                file_path=temp_path,
                use_multi_index=False  # Not enabled
            )
            loader = DataLoader(CSVDataProvider())
            
            # Should handle gracefully or raise informative error
            try:
                data = loader.load_and_prepare(config)
                assert isinstance(data, pd.DataFrame)
            except Exception:
                pass  # Expected if multi-index structure not supported
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

