"""
Pytest configuration and fixtures for cad_ig_er_index_backtesting tests.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def sample_data():
    """Fixture providing sample price data."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'price': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)


@pytest.fixture
def multi_asset_data():
    """Fixture providing multi-asset data."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'cad_ig_er_index': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'us_hy_er_index': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'us_ig_er_index': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'tsx': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'vix': 20 + np.random.randn(100) * 2
    }, index=dates)


@pytest.fixture
def multi_index_data():
    """Fixture providing multi-index DataFrame."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    columns = pd.MultiIndex.from_tuples([
        ('equity_indices', 'tsx'),
        ('equity_indices', 's&p_500'),
        ('volatility', 'vix'),
        ('economic_indicators', 'us_economic_regime')
    ])
    data = np.random.randn(100, 4) * 10 + 100
    return pd.DataFrame(data, index=dates, columns=columns)


@pytest.fixture
def temp_csv_file():
    """Fixture providing temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("Date,Price,Volume\n")
        for i in range(20):
            f.write(f"2020-01-{i+1:02d},{100+i},{1000+i}\n")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_multi_index_csv():
    """Fixture providing temporary multi-index CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("asset_class,equity_indices,volatility\n")
        f.write("column,tsx,vix\n")
        f.write("Date,,\n")
        for i in range(20):
            f.write(f"2020-01-{i+1:02d},{100+i},{15+i*0.1}\n")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_signals():
    """Fixture providing sample entry/exit signals."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    entry = pd.Series([False] * 100, index=dates)
    exit = pd.Series([False] * 100, index=dates)
    
    # Add some signals
    entry.iloc[10] = True
    exit.iloc[20] = True
    entry.iloc[30] = True
    exit.iloc[40] = True
    
    return entry, exit

