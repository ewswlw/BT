"""
Comprehensive tests for data loading and preprocessing components.
Tests all edge cases for CSVDataProvider, MultiIndexCSVDataProvider, and DataLoader.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cad_ig_er_index_backtesting.core.data_loader import (
    CSVDataProvider, MultiIndexCSVDataProvider, DataLoader, DataValidator
)
from cad_ig_er_index_backtesting.core.config import DataConfig


class TestCSVDataProvider:
    """Test CSVDataProvider with various edge cases."""
    
    def test_load_simple_csv(self):
        """Test loading a simple CSV file."""
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Date,Price,Volume\n")
            f.write("2020-01-01,100.0,1000\n")
            f.write("2020-01-02,101.0,1100\n")
            f.write("2020-01-03,102.0,1200\n")
            temp_path = f.name
        
        try:
            config = DataConfig(file_path=temp_path, date_column="Date")
            provider = CSVDataProvider()
            df = provider.load_data(config)
            
            assert not df.empty
            assert len(df) == 3
            assert isinstance(df.index, pd.DatetimeIndex)
            assert "Price" in df.columns
            assert "Volume" in df.columns
        finally:
            os.unlink(temp_path)
    
    def test_load_csv_with_date_filtering(self):
        """Test CSV loading with date range filtering."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Date,Price\n")
            for i in range(10):
                f.write(f"2020-01-{i+1:02d},100.{i}\n")
            temp_path = f.name
        
        try:
            config = DataConfig(
                file_path=temp_path,
                date_column="Date",
                start_date="2020-01-03",
                end_date="2020-01-07"
            )
            provider = CSVDataProvider()
            df = provider.load_data(config)
            
            assert len(df) == 5
            assert df.index[0] == pd.Timestamp("2020-01-03")
            assert df.index[-1] == pd.Timestamp("2020-01-07")
        finally:
            os.unlink(temp_path)
    
    def test_load_csv_invalid_file(self):
        """Test loading non-existent file raises error."""
        config = DataConfig(file_path="nonexistent_file.csv")
        provider = CSVDataProvider()
        
        with pytest.raises(ValueError):
            provider.load_data(config)
    
    def test_load_csv_empty_file(self):
        """Test loading empty CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Date,Price\n")
            temp_path = f.name
        
        try:
            config = DataConfig(file_path=temp_path)
            provider = CSVDataProvider()
            df = provider.load_data(config)
            
            assert df.empty
        finally:
            os.unlink(temp_path)
    
    def test_load_csv_malformed_dates(self):
        """Test loading CSV with malformed dates."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Date,Price\n")
            f.write("invalid-date,100.0\n")
            f.write("2020-01-02,101.0\n")
            temp_path = f.name
        
        try:
            config = DataConfig(file_path=temp_path)
            provider = CSVDataProvider()
            df = provider.load_data(config)
            
            # Should handle invalid dates gracefully
            assert len(df) >= 1
        finally:
            os.unlink(temp_path)


class TestMultiIndexCSVDataProvider:
    """Test MultiIndexCSVDataProvider with various edge cases."""
    
    def test_load_multi_index_csv(self):
        """Test loading multi-index CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("asset_class,equity_indices,equity_indices,volatility\n")
            f.write("column,tsx,s&p_500,vix\n")
            f.write("Date,,,\n")
            f.write("2020-01-01,100.0,200.0,15.0\n")
            f.write("2020-01-02,101.0,201.0,16.0\n")
            temp_path = f.name
        
        try:
            config = DataConfig(
                file_path=temp_path,
                use_multi_index=True,
                multi_index_header_rows=2,
                date_header_row=2
            )
            provider = MultiIndexCSVDataProvider()
            df = provider.load_data(config)
            
            assert not df.empty
            assert isinstance(df.columns, pd.MultiIndex)
            assert len(df.columns.levels) == 2
            assert len(df) == 2
        finally:
            os.unlink(temp_path)
    
    def test_load_multi_index_with_date_filtering(self):
        """Test multi-index CSV loading with date filtering."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("asset_class,equity_indices\n")
            f.write("column,tsx\n")
            f.write("Date,\n")
            for i in range(10):
                f.write(f"2020-01-{i+1:02d},{100+i}\n")
            temp_path = f.name
        
        try:
            config = DataConfig(
                file_path=temp_path,
                use_multi_index=True,
                start_date="2020-01-03",
                end_date="2020-01-07"
            )
            provider = MultiIndexCSVDataProvider()
            df = provider.load_data(config)
            
            assert len(df) == 5
        finally:
            os.unlink(temp_path)
    
    def test_load_multi_index_invalid_structure(self):
        """Test loading malformed multi-index CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("invalid,structure\n")
            temp_path = f.name
        
        try:
            config = DataConfig(file_path=temp_path, use_multi_index=True)
            provider = MultiIndexCSVDataProvider()
            
            with pytest.raises(ValueError):
                provider.load_data(config)
        finally:
            os.unlink(temp_path)


class TestDataLoader:
    """Test DataLoader with all filtering, quality checks, and transformations."""
    
    def create_test_dataframe(self, multi_index=False):
        """Helper to create test DataFrame."""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        
        if multi_index:
            columns = pd.MultiIndex.from_tuples([
                ('equity_indices', 'tsx'),
                ('equity_indices', 's&p_500'),
                ('volatility', 'vix'),
                ('economic_indicators', 'us_economic_regime')
            ])
        else:
            columns = ['tsx', 's&p_500', 'vix', 'us_economic_regime']
        
        data = np.random.randn(20, 4) * 10 + 100
        df = pd.DataFrame(data, index=dates, columns=columns)
        return df
    
    def test_load_and_prepare_simple(self):
        """Test basic load_and_prepare functionality."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Date,Price\n")
            for i in range(10):
                f.write(f"2020-01-{i+1:02d},{100+i}\n")
            temp_path = f.name
        
        try:
            config = DataConfig(file_path=temp_path)
            loader = DataLoader(CSVDataProvider())
            df = loader.load_and_prepare(config)
            
            assert not df.empty
            assert len(df) == 10
        finally:
            os.unlink(temp_path)
    
    def test_filter_by_asset_classes_include(self):
        """Test filtering by asset class include list."""
        df = self.create_test_dataframe(multi_index=True)
        config = DataConfig(
            file_path="dummy",
            use_multi_index=True,
            asset_classes_include=['equity_indices']
        )
        
        loader = DataLoader(CSVDataProvider())
        filtered = loader._filter_by_asset_classes(df, config)
        
        assert len(filtered.columns) == 2
        assert all('equity_indices' in str(col[0]) for col in filtered.columns)
    
    def test_filter_by_asset_classes_exclude(self):
        """Test filtering by asset class exclude list."""
        df = self.create_test_dataframe(multi_index=True)
        config = DataConfig(
            file_path="dummy",
            use_multi_index=True,
            asset_classes_exclude=['volatility']
        )
        
        loader = DataLoader(CSVDataProvider())
        filtered = loader._filter_by_asset_classes(df, config)
        
        assert 'volatility' not in [col[0] for col in filtered.columns]
    
    def test_filter_by_columns_include(self):
        """Test filtering columns within asset classes."""
        df = self.create_test_dataframe(multi_index=True)
        config = DataConfig(
            file_path="dummy",
            use_multi_index=True,
            columns_include={'equity_indices': ['tsx']}
        )
        
        loader = DataLoader(CSVDataProvider())
        filtered = loader._filter_by_columns(df, config)
        
        equity_cols = [col for col in filtered.columns if col[0] == 'equity_indices']
        assert len(equity_cols) == 1
        assert equity_cols[0][1] == 'tsx'
    
    def test_filter_by_columns_exclude(self):
        """Test excluding columns within asset classes."""
        df = self.create_test_dataframe(multi_index=True)
        config = DataConfig(
            file_path="dummy",
            use_multi_index=True,
            columns_exclude={'equity_indices': ['s&p_500']}
        )
        
        loader = DataLoader(CSVDataProvider())
        filtered = loader._filter_by_columns(df, config)
        
        equity_cols = [col for col in filtered.columns if col[0] == 'equity_indices']
        assert 's&p_500' not in [col[1] for col in equity_cols]
    
    def test_handle_missing_data_forward_fill(self):
        """Test forward fill missing data strategy."""
        df = pd.DataFrame({
            'A': [1, np.nan, 3, np.nan, 5],
            'B': [10, 20, np.nan, 40, 50]
        })
        config = DataConfig(file_path="dummy", missing_data_strategy="forward_fill")
        
        loader = DataLoader(CSVDataProvider())
        result = loader._handle_missing_data(df, config)
        
        assert not result['A'].isna().any()
        assert result['A'].iloc[1] == 1  # Forward filled
        assert result['A'].iloc[3] == 3  # Forward filled
    
    def test_handle_missing_data_backward_fill(self):
        """Test backward fill missing data strategy."""
        df = pd.DataFrame({
            'A': [1, np.nan, 3, np.nan, 5]
        })
        config = DataConfig(file_path="dummy", missing_data_strategy="backward_fill")
        
        loader = DataLoader(CSVDataProvider())
        result = loader._handle_missing_data(df, config)
        
        assert not result['A'].isna().any()
        assert result['A'].iloc[1] == 3  # Backward filled
        assert result['A'].iloc[3] == 5  # Backward filled
    
    def test_handle_missing_data_interpolate(self):
        """Test interpolate missing data strategy."""
        df = pd.DataFrame({
            'A': [1, np.nan, 3, np.nan, 5]
        })
        config = DataConfig(file_path="dummy", missing_data_strategy="interpolate")
        
        loader = DataLoader(CSVDataProvider())
        result = loader._handle_missing_data(df, config)
        
        assert not result['A'].isna().any()
    
    def test_handle_missing_data_drop(self):
        """Test drop missing data strategy."""
        df = pd.DataFrame({
            'A': [1, np.nan, 3, np.nan, 5]
        })
        config = DataConfig(file_path="dummy", missing_data_strategy="drop")
        
        loader = DataLoader(CSVDataProvider())
        result = loader._handle_missing_data(df, config)
        
        assert result['A'].isna().sum() == 0
    
    def test_handle_missing_data_fill_value(self):
        """Test fill value missing data strategy."""
        df = pd.DataFrame({
            'A': [1, np.nan, 3, np.nan, 5]
        })
        config = DataConfig(
            file_path="dummy",
            missing_data_strategy="fill_value",
            missing_data_fill_value=999.0
        )
        
        loader = DataLoader(CSVDataProvider())
        result = loader._handle_missing_data(df, config)
        
        assert (result['A'] == 999.0).sum() == 2
    
    def test_detect_and_handle_outliers_flag(self):
        """Test outlier detection with flag action."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 100, 5, 6, 7]  # 100 is an outlier
        })
        config = DataConfig(
            file_path="dummy",
            outlier_detection_enabled=True,
            outlier_method="iqr",
            outlier_action="flag"
        )
        
        loader = DataLoader(CSVDataProvider())
        result = loader._detect_and_handle_outliers(df, config)
        
        # Should not modify data, just flag
        assert result['A'].iloc[3] == 100
    
    def test_detect_and_handle_outliers_remove(self):
        """Test outlier detection with remove action."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 100, 5, 6, 7]
        }, index=pd.date_range('2020-01-01', periods=7))
        config = DataConfig(
            file_path="dummy",
            outlier_detection_enabled=True,
            outlier_method="iqr",
            outlier_action="remove"
        )
        
        loader = DataLoader(CSVDataProvider())
        result = loader._detect_and_handle_outliers(df, config)
        
        # Outlier row should be removed
        assert len(result) < len(df)
    
    def test_detect_and_handle_outliers_clip(self):
        """Test outlier detection with clip action."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 100, 5, 6, 7]
        })
        config = DataConfig(
            file_path="dummy",
            outlier_detection_enabled=True,
            outlier_method="iqr",
            outlier_action="clip"
        )
        
        loader = DataLoader(CSVDataProvider())
        result = loader._detect_and_handle_outliers(df, config)
        
        # Outlier should be clipped
        assert result['A'].iloc[3] < 100
    
    def test_detect_and_handle_gaps_flag(self):
        """Test gap detection with flag action."""
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        dates = dates.drop(dates[2])  # Create a gap
        df = pd.DataFrame({'A': range(len(dates))}, index=dates)
        
        config = DataConfig(
            file_path="dummy",
            gap_detection_enabled=True,
            max_gap_days=1,
            gap_action="flag"
        )
        
        loader = DataLoader(CSVDataProvider())
        result = loader._detect_and_handle_gaps(df, config)
        
        # Should not modify data
        assert len(result) == len(df)
    
    def test_validate_data_min_max(self):
        """Test data validation with min/max values."""
        df = pd.DataFrame({
            'A': [-10, 5, 15, 25, 35]
        })
        config = DataConfig(
            file_path="dummy",
            data_validation_enabled=True,
            min_value=0,
            max_value=30
        )
        
        loader = DataLoader(CSVDataProvider())
        result = loader._validate_data(df, config)
        
        assert result['A'].min() >= 0
        assert result['A'].max() <= 30
    
    def test_apply_type_conversion(self):
        """Test type conversion to numeric."""
        df = pd.DataFrame({
            'A': ['1', '2', '3'],
            'B': [1.0, 2.0, 3.0]
        })
        config = DataConfig(file_path="dummy", force_numeric=True)
        
        loader = DataLoader(CSVDataProvider())
        result = loader._apply_type_conversion(df, config)
        
        assert pd.api.types.is_numeric_dtype(result['A'])
    
    def test_flatten_multi_index(self):
        """Test flattening multi-index columns."""
        df = self.create_test_dataframe(multi_index=True)
        config = DataConfig(file_path="dummy", flatten_columns=True)
        
        loader = DataLoader(CSVDataProvider())
        result = loader._flatten_multi_index(df, config)
        
        assert not isinstance(result.columns, pd.MultiIndex)
        assert 'tsx' in result.columns
        assert 'vix' in result.columns
    
    def test_resample_data(self):
        """Test data resampling."""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        df = pd.DataFrame({'A': range(20)}, index=dates)
        config = DataConfig(file_path="dummy", resample_frequency="W")
        
        loader = DataLoader(CSVDataProvider())
        result = loader._resample_data(df, "W")
        
        assert len(result) < len(df)
        assert isinstance(result.index, pd.DatetimeIndex)
    
    def test_load_and_prepare_multi_index_pipeline(self):
        """Test complete multi-index pipeline."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("asset_class,equity_indices,volatility\n")
            f.write("column,tsx,vix\n")
            f.write("Date,,\n")
            for i in range(10):
                f.write(f"2020-01-{i+1:02d},{100+i},{15+i*0.1}\n")
            temp_path = f.name
        
        try:
            config = DataConfig(
                file_path=temp_path,
                use_multi_index=True,
                asset_classes_include=['equity_indices'],
                flatten_columns=True
            )
            loader = DataLoader(MultiIndexCSVDataProvider())
            df = loader.load_and_prepare(config)
            
            assert not df.empty
            assert not isinstance(df.columns, pd.MultiIndex)
            assert 'tsx' in df.columns
            assert 'vix' not in df.columns  # Excluded by filter
        finally:
            os.unlink(temp_path)
    
    def test_load_and_prepare_empty_data(self):
        """Test handling empty data."""
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
    
    def test_load_and_prepare_missing_required_columns(self):
        """Test validation of required columns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Date,Price\n")
            f.write("2020-01-01,100.0\n")
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


class TestDataValidator:
    """Test DataValidator utility methods."""
    
    def test_validate_price_data(self):
        """Test price data validation."""
        df = pd.DataFrame({
            'valid': [100, 101, 102],
            'all_nan': [np.nan, np.nan, np.nan],
            'negative': [-1, -2, -3],
            'infinite': [1, np.inf, 3]
        })
        
        results = DataValidator.validate_price_data(df, ['valid', 'all_nan', 'negative', 'infinite'])
        
        assert results['valid'] == True
        assert results['all_nan'] == False
        assert results['negative'] == True  # Has positive values check might fail
        assert results['infinite'] == False
    
    def test_check_data_gaps(self):
        """Test gap detection."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        dates = dates.drop(dates[5])  # Create a gap
        df = pd.DataFrame({'A': range(len(dates))}, index=dates)
        
        gaps = DataValidator.check_data_gaps(df, max_gap_days=1)
        
        assert len(gaps['A']) > 0
    
    def test_detect_outliers_iqr(self):
        """Test outlier detection using IQR method."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 100]  # 100 is an outlier
        })
        
        outliers = DataValidator.detect_outliers(df, method="iqr")
        
        assert 'A' in outliers
        assert len(outliers['A']) > 0
    
    def test_detect_outliers_zscore(self):
        """Test outlier detection using z-score method."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 100]
        })
        
        outliers = DataValidator.detect_outliers(df, method="zscore", threshold=2.0)
        
        assert 'A' in outliers
        assert len(outliers['A']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

