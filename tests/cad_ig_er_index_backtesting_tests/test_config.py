"""
Comprehensive tests for configuration management.
Tests DataConfig, PortfolioConfig, ReportingConfig, and config loading.
"""

import pytest
import yaml
import tempfile
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cad_ig_er_index_backtesting.core.config import (
    DataConfig, PortfolioConfig, ReportingConfig, OptimizationConfig,
    BaseConfig, create_config_from_dict, load_config, save_config
)


class TestDataConfig:
    """Test DataConfig dataclass."""
    
    def test_default_values(self):
        """Test DataConfig with default values."""
        config = DataConfig(file_path="test.csv")
        
        assert config.date_column == "Date"
        assert config.use_multi_index == False
        assert config.missing_data_strategy == "forward_fill"
        assert config.outlier_detection_enabled == False
        assert config.flatten_columns == True
    
    def test_all_fields(self):
        """Test DataConfig with all fields specified."""
        config = DataConfig(
            file_path="test.csv",
            date_column="Timestamp",
            use_multi_index=True,
            asset_classes_include=["equity_indices"],
            asset_classes_exclude=["forecasts"],
            columns_include={"equity_indices": ["tsx"]},
            missing_data_strategy="interpolate",
            outlier_detection_enabled=True,
            gap_detection_enabled=True,
            data_validation_enabled=False,
            column_rename_map={"old": "new"},
            force_numeric=False,
            flatten_columns=False
        )
        
        assert config.file_path == "test.csv"
        assert config.use_multi_index == True
        assert config.asset_classes_include == ["equity_indices"]
        assert config.missing_data_strategy == "interpolate"
    
    def test_optional_fields_none(self):
        """Test DataConfig with None values for optional fields."""
        config = DataConfig(
            file_path="test.csv",
            asset_classes_include=None,
            asset_classes_exclude=None,
            columns_include=None,
            columns_exclude=None,
            column_rename_map=None,
            min_value=None,
            max_value=None
        )
        
        assert config.asset_classes_include is None
        assert config.column_rename_map is None


class TestPortfolioConfig:
    """Test PortfolioConfig dataclass."""
    
    def test_default_values(self):
        """Test PortfolioConfig with default values."""
        config = PortfolioConfig()
        
        assert config.initial_capital == 100000
        assert config.frequency == "W"
        assert config.fees == 0.0
        assert config.leverage == 1.0
    
    def test_custom_values(self):
        """Test PortfolioConfig with custom values."""
        config = PortfolioConfig(
            initial_capital=50000,
            frequency="D",
            fees=0.001,
            slippage=0.0005,
            leverage=2.0
        )
        
        assert config.initial_capital == 50000
        assert config.frequency == "D"
        assert config.fees == 0.001
        assert config.leverage == 2.0


class TestConfigLoading:
    """Test configuration loading from YAML."""
    
    def test_load_config_from_yaml(self):
        """Test loading configuration from YAML file."""
        config_dict = {
            'data': {
                'file_path': 'test.csv',
                'use_multi_index': True,
                'asset_classes_include': ['equity_indices']
            },
            'portfolio': {
                'initial_capital': 50000
            },
            'reporting': {
                'output_dir': 'custom_output'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name
        
        try:
            loaded = load_config(temp_path)
            assert loaded['data']['file_path'] == 'test.csv'
            assert loaded['data']['use_multi_index'] == True
        finally:
            os.unlink(temp_path)
    
    def test_create_config_from_dict(self):
        """Test creating BaseConfig from dictionary."""
        config_dict = {
            'data': {
                'file_path': 'test.csv',
                'use_multi_index': True
            },
            'portfolio': {
                'initial_capital': 50000
            },
            'reporting': {
                'output_dir': 'test_output'
            },
            'random_seed': 42,
            'verbose': False
        }
        
        base_config = create_config_from_dict(config_dict)
        
        assert isinstance(base_config.data, DataConfig)
        assert isinstance(base_config.portfolio, PortfolioConfig)
        assert base_config.data.use_multi_index == True
        assert base_config.portfolio.initial_capital == 50000
        assert base_config.random_seed == 42
        assert base_config.verbose == False
    
    def test_create_config_with_defaults(self):
        """Test creating config with missing fields uses defaults."""
        config_dict = {
            'data': {
                'file_path': 'test.csv'
            }
        }
        
        base_config = create_config_from_dict(config_dict)
        
        # Should use defaults
        assert base_config.data.date_column == "Date"
        assert base_config.portfolio.initial_capital == 100000
        assert base_config.random_seed == 42
    
    def test_create_config_with_optimization(self):
        """Test creating config with optimization section."""
        config_dict = {
            'data': {'file_path': 'test.csv'},
            'portfolio': {},
            'reporting': {},
            'optimization': {
                'method': 'bayesian',
                'n_trials': 200
            }
        }
        
        base_config = create_config_from_dict(config_dict)
        
        assert base_config.optimization is not None
        assert base_config.optimization.method == 'bayesian'
        assert base_config.optimization.n_trials == 200
    
    def test_save_config(self):
        """Test saving configuration to YAML."""
        data_config = DataConfig(file_path="test.csv", use_multi_index=True)
        portfolio_config = PortfolioConfig(initial_capital=50000)
        reporting_config = ReportingConfig(output_dir="test_output")
        
        base_config = BaseConfig(
            data=data_config,
            portfolio=portfolio_config,
            reporting=reporting_config,
            random_seed=42
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            save_config(base_config, temp_path)
            
            # Verify it can be loaded back
            loaded = load_config(temp_path)
            assert loaded['data']['file_path'] == 'test.csv'
            assert loaded['data']['use_multi_index'] == True
            assert loaded['portfolio']['initial_capital'] == 50000
        finally:
            os.unlink(temp_path)
    
    def test_config_with_multi_index_settings(self):
        """Test config with all multi-index settings."""
        config_dict = {
            'data': {
                'file_path': 'test.csv',
                'use_multi_index': True,
                'multi_index_header_rows': 2,
                'date_header_row': 2,
                'asset_classes_include': ['equity_indices', 'volatility'],
                'asset_classes_exclude': ['forecasts'],
                'columns_include': {
                    'equity_indices': ['tsx', 's&p_500']
                },
                'columns_exclude': {
                    'economic_indicators': ['us_hard_data_surprises']
                },
                'missing_data_strategy': 'interpolate',
                'outlier_detection_enabled': True,
                'gap_detection_enabled': True,
                'data_validation_enabled': True,
                'min_value': 0.0,
                'max_value': 1000.0,
                'column_rename_map': {'old_name': 'new_name'},
                'force_numeric': True,
                'flatten_columns': True
            },
            'portfolio': {},
            'reporting': {}
        }
        
        base_config = create_config_from_dict(config_dict)
        
        assert base_config.data.use_multi_index == True
        assert len(base_config.data.asset_classes_include) == 2
        assert base_config.data.missing_data_strategy == 'interpolate'
        assert base_config.data.outlier_detection_enabled == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

