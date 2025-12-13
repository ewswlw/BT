"""
Market Data Pipeline - Bloomberg Data Fetching and Processing

This module provides a comprehensive pipeline for fetching, processing, and organizing
financial market data from Bloomberg using the xbbg library.

KEY FEATURE: ASSET CLASS GROUPING WITH MULTIINDEX COLUMNS
==========================================================

The pipeline automatically organizes all time series columns by asset class using
pandas MultiIndex columns. This provides:

1. VISIBLE GROUPING IN CSV FILES:
   - CSV files have a two-row header showing asset class grouping
   - First row: Asset class names (e.g., 'cad_credit_spreads', 'us_credit_spreads', 'cad_credit_exposure', 'us_credit_exposure')
   - Second row: Column names (e.g., 'cad_oas', 'us_ig_oas', 'us_hy_oas')
   - Makes it easy to see which columns belong together when viewing CSV in Excel

2. ORGANIZED COLUMN STRUCTURE:
   - Columns are automatically grouped and ordered by asset class

3. EASY DATA ACCESS IN NOTEBOOKS:
   - Access entire asset classes: df['cad_credit_spreads']
   - Access specific columns: df[('cad_credit_spreads', 'cad_oas')]
   - See all asset classes: df.columns.get_level_values(0).unique()

CSV OUTPUT FORMAT:
------------------
When saved to CSV, the file structure looks like:

    asset_class,cad_credit_spreads,us_credit_spreads,us_credit_spreads,cad_credit_exposure,us_credit_exposure,...
    column,cad_oas,us_ig_oas,us_hy_oas,cad_ig_er_index,us_ig_er_index,...
    Date,120.5,95.2,350.8,100.0,100.0,...
    2024-01-01,120.5,95.2,350.8,100.0,100.0,...

USAGE EXAMPLES:
---------------
# Initialize pipeline
pipeline = DataPipeline()

# Process and save data (automatically organizes by asset class)
df = pipeline.process_data()
pipeline.save_dataset(df, 'output.csv')

# Load data (automatically detects MultiIndex structure)
df = pipeline.load_dataset('output.csv')

# Access data by asset class
cad_credit_spreads = df['cad_credit_spreads']  # DataFrame with CAD credit spread columns
us_credit_spreads = df['us_credit_spreads']    # DataFrame with US credit spread columns
cad_credit_exposure = df['cad_credit_exposure'] # DataFrame with CAD credit exposure columns
us_credit_exposure = df['us_credit_exposure']   # DataFrame with US credit exposure columns

# Access specific column
cad_oas = df[('cad_credit_spreads', 'cad_oas')]  # Series

# List all asset classes
asset_classes = df.columns.get_level_values(0).unique()
# Returns: ['cad_credit_spreads', 'us_credit_spreads', 'cad_credit_exposure', 'us_credit_exposure', 'equity_indices', ...]

# Get all columns in a specific asset class
cad_credit_cols = df['cad_credit_spreads'].columns.tolist()
# Returns: ['cad_oas']

# Plot all credit spreads together
df['cad_credit_spreads'].plot(title='CAD Credit Spreads Over Time')
df['us_credit_spreads'].plot(title='US Credit Spreads Over Time')

CUSTOMIZING ASSET CLASS GROUPS:
-------------------------------
To add new columns or modify groupings, edit the ASSET_CLASS_GROUPS dictionary
(defined around line 200). The order of asset classes in the dictionary determines
the column order in the output.

Example: Adding a new column to an existing asset class
    ASSET_CLASS_GROUPS['cad_credit_spreads'].append('new_spread_column')

Example: Creating a new asset class
    ASSET_CLASS_GROUPS['new_asset_class'] = ['column1', 'column2', 'column3']

Note: After modifying ASSET_CLASS_GROUPS, the COLUMN_TO_ASSET_CLASS mapping
is automatically rebuilt from the groups.

BACKWARD COMPATIBILITY:
-----------------------
The load_dataset() method automatically detects whether a CSV file has:
- MultiIndex structure (two header rows) - loads with header=[0, 1]
- Single-level structure (one header row) - loads normally

This ensures compatibility with older CSV files that don't have asset class grouping.

AUTHOR: Market Data Pipeline
LAST UPDATED: 2025-01-XX
"""

import os
import sys
import pandas as pd
import yaml
from datetime import datetime
from typing import Dict, Tuple, List, Union, Optional
from pathlib import Path
from xbbg import blp
import logging
import numpy as np
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

#############################################################
# CONFIGURATION SECTION
#############################################################

# Default date range parameters
DEFAULT_START_DATE = '2002-11-01'
DEFAULT_END_DATE = datetime.now().strftime('%Y-%m-%d')

# Data frequency and alignment defaults
DEFAULT_PERIODICITY = 'D'  # 'D'=daily, 'W'=weekly, 'M'=monthly
DEFAULT_ALIGN_START = True  # Align all series to start at the same date
DEFAULT_FILL = 'ffill'  # Forward fill missing values
DEFAULT_START_DATE_ALIGN = 'no'  # Align start dates

# Output settings
DEFAULT_OUTPUT_PATH = 'market data pipeline/processed market data/market_data_time_series.csv'

# Bloomberg API settings
BLOOMBERG_TIMEOUT = 10000  # milliseconds
BLOOMBERG_MAX_RETRIES = 3
BLOOMBERG_RETRY_DELAY = 2  # seconds
BLOOMBERG_BATCH_SIZE = 30  # number of tickers per batch
BLOOMBERG_FIELDS_BATCH_SIZE = 8  # number of fields per batch

# Error handling settings
ERROR_ON_MISSING_DATA = False  # Raise error if data is missing
ALLOW_PARTIAL_DATA = True  # Return partial data if some fields are missing

# Cache settings
USE_CACHE = True  # Use cache to avoid redundant API calls
CACHE_DIR = 'data_pipelines/cache'
CACHE_EXPIRY = 86400  # seconds (24 hours)

# Data processing settings
HANDLE_BAD_DATES = True  # Handle known bad dates in data
APPLY_CALENDAR_ADJUSTMENTS = True  # Apply calendar adjustments to align data
CONVERT_TIMEZONES = True  # Convert timezones to a consistent timezone
DEFAULT_TIMEZONE = 'America/New_York'  # Default timezone for data

# Resampling settings
RESAMPLE_METHOD = 'last'  # Method for resampling data to different frequencies
INTERPOLATION_METHOD = 'linear'  # Method for interpolating missing values

# Leveraged credit exposure index settings
LEVERAGE_MULTIPLIER = 3.0  # Leverage multiplier for credit exposure indices (default: 3x)
ANNUAL_BORROW_COST_BPS = 40.0  # Annualized borrow cost in basis points (default: 40bps = 0.004)

# Combined total return index settings (leveraged credit + risk-free rate)
LEVERAGED_ALLOCATION = 1.0  # Allocation to leveraged credit strategy (default: 100%)
RISK_FREE_ALLOCATION = 1.0  # Allocation to risk-free rate (default: 100%)

# Net return index fee settings (after gross return calculation)
ANNUAL_MANAGEMENT_FEE_PCT = 0.90  # Annualized management fee percentage (default: 0.90%)
ANNUAL_OPERATIONAL_FEE_PCT = 0.20  # Annualized operational fee percentage (default: 0.20%)
PERFORMANCE_FEE_PCT = 15.0  # Performance fee percentage (default: 15%, charged annually on positive returns, no HWM)

#############################################################
# SECURITIES CONFIGURATION
#############################################################

# Default securities and their mappings - match the exact securities from config
# Format: {(ticker, field): column_name}
DEFAULT_PRICE_MAPPING = {
    ('I05510CA Index', 'INDEX_OAS_TSY_BP'): 'cad_oas',
    ('I34227CA Index', 'INDEX_OAS_TSY_BP'): 'cad_short_oas',
    ('I34229CA Index', 'INDEX_OAS_TSY_BP'): 'cad_long_oas',
    ('I05523CA Index', 'INDEX_OAS_TSY_BP'): 'cad_credit_spreads_fins',
    ('I05520CA Index', 'INDEX_OAS_TSY_BP'): 'cad_credit_spreads_non_fins_ex_uts',
    ('I05517CA Index', 'INDEX_OAS_TSY_BP'): 'cad_credit_spreads_uts',
    ('I05515CA Index', 'INDEX_OAS_TSY_BP'): 'cad_credit_spreads_a_credits',
    ('I05516CA Index', 'INDEX_OAS_TSY_BP'): 'cad_credit_spreads_bbb_credits',
    ('I34069CA Index', 'INDEX_OAS_TSY_BP'): 'cad_credit_spreads_provs',
    ('I34336CA Index', 'INDEX_OAS_TSY_BP'): 'cad_credit_spreads_provs_longs',
    ('LF98TRUU Index', 'INDEX_OAS_TSY_BP'): 'us_hy_oas',
    ('LUACTRUU Index', 'INDEX_OAS_TSY_BP'): 'us_ig_oas',
    ('IBOXUMAE MKIT Curncy', 'ROLL_ADJUSTED_MID_PRICE'): 'cdx_ig',
    ('IBOXHYSE MKIT Curncy', 'ROLL_ADJUSTED_MID_PRICE'): 'cdx_hy',
    ('SPTSX Index', 'PX_LAST'): 'tsx',
    ('SPXT Index', 'PX_LAST'): 's&p_500',
    ('ERIXCDIG Index', 'PX_LAST'): 'cdx_ig_er',
    ('UISYMH5S Index', 'PX_LAST'): 'cdx_hy_er',
    ('VIX Index', 'PX_LAST'): 'vix',
    ('USYC3M30 Index', 'PX_LAST'): 'us_3m_10y',
    ('BCMPUSGR Index', 'PX_LAST'): 'us_growth_surprises',
    ('BCMPUSIF Index', 'PX_LAST'): 'us_inflation_surprises',
    ('LEI YOY  Index', 'PX_LAST'): 'us_lei_yoy',
    ('.HARDATA G Index', 'PX_LAST'): 'us_hard_data_surprises',
    ('CGERGLOB Index', 'PX_LAST'): 'us_equity_revisions',
    ('.ECONREGI G Index', 'PX_LAST'): 'us_economic_regime',
    ('GCAN3M Index', 'PX_LAST'): 'cad_3m_tbill',
}

# Excess Return YTD Mappings
# Format: {(ticker, field): column_name}
DEFAULT_ER_YTD_MAPPING = {
    ('I05510CA Index', 'INDEX_EXCESS_RETURN_YTD'): 'cad_ig_er',
    ('I34227CA Index', 'INDEX_EXCESS_RETURN_YTD'): 'cad_ig_short_er',
    ('I34229CA Index', 'INDEX_EXCESS_RETURN_YTD'): 'cad_ig_long_er',
    ('I34069CA Index', 'INDEX_EXCESS_RETURN_YTD'): 'cad_credit_spreads_provs_er',
    ('I34336CA Index', 'INDEX_EXCESS_RETURN_YTD'): 'cad_credit_spreads_provs_longs_er',
    ('LF98TRUU Index', 'INDEX_EXCESS_RETURN_YTD'): 'us_hy_er',
    ('LUACTRUU Index', 'INDEX_EXCESS_RETURN_YTD'): 'us_ig_er',
}

# Bloomberg Override Mappings
# Format: {(ticker, field, override): column_name}
DEFAULT_BLOOMBERG_OVERRIDES_MAPPING = {
    ('SPX Index', 'BEST_EPS', '1BF'): 'spx_1bf_eps',
    ('SPX Index', 'BEST_SALES', '1BF'): 'spx_1bf_sales', 
    ('SPTSX Index', 'BEST_EPS', '1BF'): 'tsx_1bf_eps',
    ('SPTSX Index', 'BEST_SALES', '1BF'): 'tsx_1bf_sales',
}

#############################################################
# ASSET CLASS GROUPING
#############################################################

# Define asset class groupings for column organization
# Order determines the column order in final output
ASSET_CLASS_GROUPS = {
    'risk_free_rates': [
        'cad_3m_tbill',
        'cad_3m_tbill_tr_index',
    ],
    'cad_credit_spreads': [
        'cad_oas',
        'cad_short_oas',
        'cad_long_oas',
        'cad_credit_spreads_fins',
        'cad_credit_spreads_non_fins_ex_uts',
        'cad_credit_spreads_uts',
        'cad_credit_spreads_a_credits',
        'cad_credit_spreads_bbb_credits',
        'cad_credit_spreads_provs',
        'cad_credit_spreads_provs_longs',
    ],
    'us_credit_spreads': [
        'us_ig_oas', 
        'us_hy_oas',
        'cdx_ig',
        'cdx_hy',
    ],
    'cad_credit_exposure': [
        'cad_ig_er_index',
        'cad_ig_short_er_index',
        'cad_ig_long_er_index',
        'cad_credit_spreads_provs_er_index',
        'cad_credit_spreads_provs_longs_er_index',
    ],
    'levered_credit_er_index': [
        'cad_ig_er_index_3x',
        'cad_ig_short_er_index_3x',
        'cad_ig_long_er_index_3x',
        'cad_ig_gross_return_index_3x',
        'cad_ig_short_gross_return_index_3x',
        'cad_ig_long_gross_return_index_3x',
        'cad_ig_net_return_index_3x',
        'cad_ig_short_net_return_index_3x',
        'cad_ig_long_net_return_index_3x',
    ],
    'us_credit_exposure': [
        'us_ig_er_index',
        'us_hy_er_index',
        'cdx_ig_er',
        'cdx_hy_er',
    ],
    'equity_indices': [
        'tsx',
        's&p_500',
    ],
    'volatility': [
        'vix',
    ],
    'yield_curves': [
        'us_3m_10y',
    ],
    'economic_indicators': [
        'us_growth_surprises',
        'us_inflation_surprises',
        'us_lei_yoy',
        'us_hard_data_surprises',
        'us_equity_revisions',
        'us_economic_regime',
    ],
    'forecasts': [
        'spx_1bf_eps',
        'spx_1bf_sales',
        'tsx_1bf_eps',
        'tsx_1bf_sales',
    ],
}

# Create reverse mapping: column_name -> asset_class
COLUMN_TO_ASSET_CLASS = {}
for asset_class, columns in ASSET_CLASS_GROUPS.items():
    for col in columns:
        COLUMN_TO_ASSET_CLASS[col] = asset_class

# Known bad dates in data
# Format following the config file structure: 
# {'date_key': {'column': 'column_name', 'action': 'action_type'}}
DEFAULT_BAD_DATES = {
    '2005-11-15': {
        'column': 'cad_oas',
        'action': 'use_previous'
    },
    '2005-11-15_non_fins': {
        'column': 'cad_credit_spreads_non_fins_ex_uts',
        'action': 'use_previous'
    },
    '2005-11-15_bbb': {
        'column': 'cad_credit_spreads_bbb_credits',
        'action': 'use_previous'
    },
    '2002-10-01_lei': {
        'column': 'us_lei_yoy',
        'action': 'forward_fill'
    },
    '2002-10-01_regime': {
        'column': 'us_economic_regime',
        'action': 'forward_fill'
    },
    '2002-10-01_revisions': {
        'column': 'us_equity_revisions',
        'action': 'forward_fill'
    }
}

#############################################################
# DATA FETCH SELECTION
#############################################################

# Toggle which data categories to fetch (set to False to skip)
# Default is to fetch all data categories
ENABLE_OHLC_DATA = True
ENABLE_ER_YTD_DATA = True
ENABLE_BLOOMBERG_OVERRIDES_DATA = True

# To select specific securities, simply modify the mapping dictionaries above
# by adding or removing entries as needed.

#############################################################
# END CONFIGURATION SECTION
#############################################################

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import utility functions
import logging

def convert_er_ytd_to_index(df):
    """
    Convert excess return YTD data to an index.
    Args:
        df: DataFrame containing excess return YTD columns
    Returns:
        DataFrame with excess return columns converted to indices
    """
    import pandas as pd
    logger = logging.getLogger(__name__)
    logger.info(f"[convert_er_ytd_to_index] Called with DataFrame shape: {df.shape}")
    result = pd.DataFrame(index=df.index)
    try:
        for column in df.columns:
            logger.info(f"[convert_er_ytd_to_index] Processing column: {column}")
            daily_returns = df[column].diff()
            index_values = (1 + daily_returns/100).cumprod() * 100
            result[f"{column}_index"] = index_values
            logger.info(f"[convert_er_ytd_to_index] Completed index for column: {column}")
    except Exception as e:
        logger.error(f"[convert_er_ytd_to_index] Error processing DataFrame: {e}")
        return pd.DataFrame(index=df.index)
    logger.info(f"[convert_er_ytd_to_index] Result DataFrame shape: {result.shape}")
    return result

def merge_dfs(*dfs_args, fill=None, start_date_align="no"):
    """
    Merge multiple DataFrames or Series with options for handling missing values and date alignment.
    Args:
        *dfs_args: Variable number of DataFrames/Series to merge
        fill (str, optional): Fill method for NaN values. Options: None, 'ffill', 'bfill', 'interpolate'
        start_date_align (str, optional): Whether to align start dates. Options: 'yes', 'no'
    Returns:
        pd.DataFrame: Merged DataFrame with date index
    """
    import pandas as pd
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"[merge_dfs] Called with {len(dfs_args)} DataFrames/Series")
    processed_dfs = {}
    first_valid_dates = {}
    # Convert arguments to dict with names
    for idx, df in enumerate(dfs_args):
        name = getattr(df, 'name', f'df_{idx}')
        processed_dfs[name] = df.copy()
        # Track first valid index for alignment
        if hasattr(df, 'first_valid_index'):
            first_valid_idx = df.first_valid_index()
            if first_valid_idx is not None:
                first_valid_dates[name] = first_valid_idx
    # Align start dates if requested
    if str(start_date_align).lower() == "yes" and first_valid_dates:
        latest_start = max(first_valid_dates.values())
        logger.info(f"[merge_dfs] Aligning all data to start from: {latest_start}")
        for name in processed_dfs:
            df = processed_dfs[name]
            mask = df.index >= latest_start
            processed_dfs[name] = df.loc[mask].copy()
    # Merge all DataFrames
    merged_df = None
    for name, df in processed_dfs.items():
        if merged_df is None:
            merged_df = df.copy()
        else:
            merged_df = pd.merge(merged_df, df, left_index=True, right_index=True, how='outer')
    if merged_df is None:
        return pd.DataFrame()
    # Apply fill method if specified
    if fill is not None:
        if fill == 'ffill':
            for col in merged_df.columns:
                first_valid_idx = merged_df[col].first_valid_index()
                if first_valid_idx is not None:
                    mask = merged_df.index >= first_valid_idx
                    merged_df.loc[mask, col] = merged_df.loc[mask, col].ffill()
        elif fill == 'bfill':
            for col in merged_df.columns:
                first_valid_idx = merged_df[col].first_valid_index()
                if first_valid_idx is not None:
                    mask = merged_df.index >= first_valid_idx
                    merged_df.loc[mask, col] = merged_df.loc[mask, col].bfill()
        elif fill == 'interpolate':
            for col in merged_df.columns:
                first_valid_idx = merged_df[col].first_valid_index()
                if first_valid_idx is not None:
                    mask = merged_df.index >= first_valid_idx
                    merged_df.loc[mask, col] = merged_df.loc[mask, col].interpolate(method='time')
    logger.info(f"[merge_dfs] Result DataFrame shape: {merged_df.shape}")
    return merged_df


# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPipeline:
    def __init__(
        self,
        config_path: str = None,
        start_date: str = None,
        end_date: str = None,
        periodicity: str = None,
        align_start: bool = None,
        fill: str = None,
        start_date_align: str = None,
        ohlc_mapping: Dict[Tuple[str, str], str] = None,
        er_ytd_mapping: Dict[Tuple[str, str], str] = None,
        bloomberg_overrides_mapping: Dict[Tuple[str, str, str], str] = None,
        bad_dates: Dict[str, Dict[str, str]] = None,
        enable_ohlc_data: bool = ENABLE_OHLC_DATA,
        enable_er_ytd_data: bool = ENABLE_ER_YTD_DATA,
        enable_bloomberg_overrides_data: bool = ENABLE_BLOOMBERG_OVERRIDES_DATA,
    ):
        """Initialize the DataPipeline class with configuration parameters.
        
        Args:
            config_path (str, optional): Path to YAML config file
            start_date (str, optional): Start date for data fetch. Defaults to '2002-11-01'
            end_date (str, optional): End date for data fetch. Defaults to current date
            periodicity (str, optional): Data frequency ('D'=daily, 'W'=weekly, 'M'=monthly). Defaults to 'D'
            align_start (bool, optional): Align all series to start at the same date. Defaults to True
            fill (str, optional): Fill method for missing values. Defaults to 'ffill'
            start_date_align (str, optional): Whether to align start dates. Defaults to 'yes'
            ohlc_mapping (Dict[Tuple[str, str], str], optional): Mapping for main price data
            er_ytd_mapping (Dict[Tuple[str, str], str], optional): Mapping for excess return data
            bloomberg_overrides_mapping (Dict[Tuple[str, str, str], str], optional): Mapping for forecast data
            bad_dates (Dict[str, Dict[str, str]], optional): Dates with known data issues
            enable_ohlc_data (bool, optional): Whether to fetch OHLC data. Defaults to ENABLE_OHLC_DATA
            enable_er_ytd_data (bool, optional): Whether to fetch ER YTD data. Defaults to ENABLE_ER_YTD_DATA
            enable_bloomberg_overrides_data (bool, optional): Whether to fetch Bloomberg overrides data. Defaults to ENABLE_BLOOMBERG_OVERRIDES_DATA
        """
        # Load configuration from YAML if provided
        if config_path:
            self.load_config(config_path)
        else:
            # Set basic parameters with minimal defaults
            self.start_date = start_date or DEFAULT_START_DATE
            self.end_date = end_date or DEFAULT_END_DATE
            self.periodicity = periodicity or DEFAULT_PERIODICITY
            self.align_start = align_start if align_start is not None else DEFAULT_ALIGN_START
            self.fill = fill or DEFAULT_FILL
            self.start_date_align = start_date_align or DEFAULT_START_DATE_ALIGN
            
            # Initialize mappings with defaults or provided values
            self.ohlc_mapping = ohlc_mapping or DEFAULT_PRICE_MAPPING.copy()
            self.er_ytd_mapping = er_ytd_mapping or DEFAULT_ER_YTD_MAPPING.copy()
            self.bloomberg_overrides_mapping = bloomberg_overrides_mapping or DEFAULT_BLOOMBERG_OVERRIDES_MAPPING.copy()
            self.bad_dates = bad_dates or DEFAULT_BAD_DATES.copy()
            
            # Initialize data fetch settings
            self.enable_ohlc_data = enable_ohlc_data
            self.enable_er_ytd_data = enable_er_ytd_data
            self.enable_bloomberg_overrides_data = enable_bloomberg_overrides_data

    def load_config(self, config_path: str):
        """Load configuration from a YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        
            pipeline_config = config.get('data_pipeline', {})
            
            self.start_date = pipeline_config.get('start_date', DEFAULT_START_DATE)
            self.end_date = pipeline_config.get('end_date') or DEFAULT_END_DATE
            self.periodicity = pipeline_config.get('periodicity', DEFAULT_PERIODICITY)
            self.align_start = pipeline_config.get('align_start', DEFAULT_ALIGN_START)
            self.fill = pipeline_config.get('fill', DEFAULT_FILL)
            self.start_date_align = pipeline_config.get('start_date_align', DEFAULT_START_DATE_ALIGN)
            
            # Load mappings
            mappings_config = pipeline_config.get('mappings', {})
            
            # Convert mapping strings to tuples
            self.ohlc_mapping = {}
            self.er_ytd_mapping = {}
            self.bloomberg_overrides_mapping = {}
            
            for mapping_str, column_name in mappings_config.get('ohlc_mapping', {}).items():
                ticker, field = mapping_str.split('|')
                self.ohlc_mapping[(ticker.strip(), field.strip())] = column_name
            
            for mapping_str, column_name in mappings_config.get('er_ytd_mapping', {}).items():
                ticker, field = mapping_str.split('|')
                self.er_ytd_mapping[(ticker.strip(), field.strip())] = column_name
                
            # Parse the new bloomberg_overrides_mapping
            for mapping_str, column_name in mappings_config.get('bloomberg_overrides_mapping', {}).items():
                parts = mapping_str.split('|')
                ticker = parts[0].strip()
                field = parts[1].strip()
                override = parts[2].strip()
                self.bloomberg_overrides_mapping[(ticker, field, override)] = column_name
            
            self.bad_dates = pipeline_config.get('bad_dates', {})
            
            # Load output settings
            output_config = pipeline_config.get('output', {})
            self.default_output_path = output_config.get('default_path', DEFAULT_OUTPUT_PATH)

            # Load data fetch settings
            data_fetch_config = pipeline_config.get('data_fetch', {})
            self.enable_ohlc_data = data_fetch_config.get('enable_ohlc_data', ENABLE_OHLC_DATA)
            self.enable_er_ytd_data = data_fetch_config.get('enable_er_ytd_data', ENABLE_ER_YTD_DATA)
            self.enable_bloomberg_overrides_data = data_fetch_config.get('enable_bloomberg_overrides_data', ENABLE_BLOOMBERG_OVERRIDES_DATA)

        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {str(e)}")
            raise
            
    def merge_dfs(self, *dfs_args, fill=None, start_date_align="no"):
        """
        Merge multiple DataFrames or Series with options for handling missing values and date alignment.
        
        Args:
            *dfs_args: Variable number of DataFrames/Series to merge
            fill (str, optional): Fill method for NaN values. Options: None, 'ffill', 'bfill', 'interpolate'
            start_date_align (str, optional): Whether to align start dates. Options: 'yes', 'no'
            
        Returns:
            pd.DataFrame: Merged DataFrame with date index
        """
        # Convert arguments to dictionary
        dfs = {f'df_{i}': df for i, df in enumerate(dfs_args)}
        
        # Convert Series to DataFrame and ensure datetime index
        processed_dfs = {}
        for name, df in dfs.items():
            # Skip None or empty DataFrames
            if df is None or (hasattr(df, 'empty') and df.empty):
                continue
                
            # Convert Series to DataFrame if necessary
            if isinstance(df, pd.Series):
                df = df.to_frame()
            
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    logger.warning(f"Failed to convert index to datetime for {name}: {str(e)}")
                    continue
            
            processed_dfs[name] = df
        
        # Return empty DataFrame if no valid inputs
        if not processed_dfs:
            return pd.DataFrame()
        
        # Find first valid date for each DataFrame
        first_valid_dates = {}
        for name, df in processed_dfs.items():
            for col in df.columns:
                first_valid_idx = df[col].first_valid_index()
                if first_valid_idx is not None:
                    if name not in first_valid_dates:
                        first_valid_dates[name] = first_valid_idx
                    else:
                        first_valid_dates[name] = max(first_valid_dates[name], first_valid_idx)
        
        # Align start dates if requested
        start_date_align = str(start_date_align).lower()
        if start_date_align == "yes" and first_valid_dates:
            # Find the latest start date among all DataFrames
            latest_start = max(first_valid_dates.values())
            logger.info(f"Aligning all data to start from: {latest_start}")
            
            # Trim all DataFrames to start from this date
            for name in processed_dfs:
                df = processed_dfs[name]
                mask = df.index >= latest_start
                processed_dfs[name] = df.loc[mask].copy()
        
        # Merge all DataFrames
        merged_df = None
        for name, df in processed_dfs.items():
            if merged_df is None:
                merged_df = df.copy()
            else:
                # Merge with outer join to keep all dates
                merged_df = pd.merge(merged_df, df, 
                                   left_index=True, right_index=True, 
                                   how='outer')
        
        if merged_df is None:
            return pd.DataFrame()
        
        # Apply fill method if specified
        if fill is not None:
            if fill == 'ffill':
                # Forward fill but only after first valid value for each column
                for col in merged_df.columns:
                    first_valid_idx = merged_df[col].first_valid_index()
                    if first_valid_idx is not None:
                        mask = merged_df.index >= first_valid_idx
                        merged_df.loc[mask, col] = merged_df.loc[mask, col].ffill()
            
            elif fill == 'bfill':
                # Backward fill but only after first valid value for each column
                for col in merged_df.columns:
                    first_valid_idx = merged_df[col].first_valid_index()
                    if first_valid_idx is not None:
                        mask = merged_df.index >= first_valid_idx
                        merged_df.loc[mask, col] = merged_df.loc[mask, col].bfill()
            
            elif fill == 'interpolate':
                # Interpolate but only after first valid value for each column
                for col in merged_df.columns:
                    first_valid_idx = merged_df[col].first_valid_index()
                    if first_valid_idx is not None:
                        mask = merged_df.index >= first_valid_idx
                        merged_df.loc[mask, col] = merged_df.loc[mask, col].interpolate(method='time')
        
        return merged_df

    def update_parameters(self, **kwargs):
        """Update any of the class parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"DataPipeline has no attribute '{key}'")

        if 'end_date' in kwargs and kwargs['end_date'] is None:
            self.end_date = datetime.now().strftime('%Y-%m-%d')

    def fetch_bloomberg_data(self, mapping: Optional[Dict[Tuple[str, str], str]] = None) -> pd.DataFrame:
        """Fetch data from Bloomberg using xbbg."""
        try:
            mapping_to_use = mapping if mapping is not None else self.ohlc_mapping
            securities = list(set(security for security, _ in mapping_to_use.keys()))
            fields = list(set(field for _, field in mapping_to_use.keys()))

            logger.info(f"Fetching Bloomberg data for {len(securities)} securities")
            
            # Map periodicity to Bloomberg format
            per_map = {
                'M': 'MONTHLY',
                'MONTHLY': 'MONTHLY',
                'W': 'WEEKLY',
                'WEEKLY': 'WEEKLY',
                'D': 'DAILY',
                'DAILY': 'DAILY'
            }
            bloomberg_per = per_map.get(self.periodicity.upper(), 'DAILY')
            
            df = blp.bdh(
                tickers=securities,
                flds=fields,
                start_date=self.start_date,
                end_date=self.end_date,
                Per=bloomberg_per
            )

            renamed_df = pd.DataFrame(index=df.index)
            for (security, field), new_name in mapping_to_use.items():
                if (security, field) in df.columns:
                    renamed_df[new_name] = df[(security, field)]
                else:
                    logger.warning(f"Column ({security}, {field}) not found in Bloomberg data")

            return renamed_df
        
        except Exception as e:
            logger.error(f"Error fetching Bloomberg data: {str(e)}")
            raise
            
    def fetch_bloomberg_overrides_data(self) -> pd.DataFrame:
        """Fetch data from Bloomberg with overrides for BEST_FPERIOD_OVERRIDE."""
        try:
            result_df = pd.DataFrame()
            
            # Group by ticker and field to minimize API calls
            ticker_field_groups = {}
            for (ticker, field, override), column in self.bloomberg_overrides_mapping.items():
                if (ticker, field) not in ticker_field_groups:
                    ticker_field_groups[(ticker, field)] = []
                ticker_field_groups[(ticker, field)].append((override, column))
            
            # Track progress for logging
            total_items = len(self.bloomberg_overrides_mapping)
            processed_items = 0
            
            # Map periodicity to Bloomberg format
            per_map = {
                'M': 'MONTHLY',
                'MONTHLY': 'MONTHLY',
                'W': 'WEEKLY',
                'WEEKLY': 'WEEKLY',
                'D': 'DAILY',
                'DAILY': 'DAILY'
            }
            bloomberg_per = per_map.get(self.periodicity.upper(), 'DAILY')
            
            # Fetch data for each ticker and field combination
            for (ticker, field), override_columns in ticker_field_groups.items():
                for override, column in override_columns:
                    processed_items += 1
                    logger.info(f"Fetching Bloomberg data with override ({processed_items}/{total_items}): {ticker}, {field}, {override}")
                    
                    try:
                        data = blp.bdh(
                            ticker, 
                            field, 
                            self.start_date, 
                            self.end_date,
                            Per=bloomberg_per,
                            BEST_FPERIOD_OVERRIDE=override
                        )
                        
                        # Convert to Series with proper column name
                        series = pd.Series(data.values.flatten(), index=data.index, name=column)
                        
                        # Add to result DataFrame
                        if result_df.empty:
                            result_df = pd.DataFrame(index=series.index)
                        result_df[column] = series
                        
                    except Exception as e:
                        logger.error(f"Error fetching data for {ticker}, {field}, {override}: {str(e)}")
            
            return result_df
        
        except Exception as e:
            logger.error(f"Error fetching Bloomberg override data: {str(e)}")
            raise

    def convert_er_ytd_to_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert excess return YTD data to indices.
        
        The Bloomberg INDEX_EXCESS_RETURN_YTD field returns year-to-date percentage returns.
        This method converts these YTD returns into an index starting at 100 for each calendar year,
        where the index value for a given day is based on the previous year-end index value
        multiplied by (1 + YTD return/100).
        """
        try:
            result = pd.DataFrame(index=df.index)
            er_columns = list(self.er_ytd_mapping.values())
            
            # Extract year from index using datetime properties
            years = [date.year for date in df.index]
            df_years = pd.Series(years, index=df.index)
            
            for column in df.columns:
                if column in er_columns:
                    # Initialize the index at 100 for the starting date
                    index_values = pd.Series(np.nan, index=df.index)
                    prev_year_end = 100  # Start with 100
                    
                    # Process each year separately
                    for year in sorted(set(years)):
                        # Get data for this year
                        year_indices = [i for i, y in enumerate(years) if y == year]
                        if not year_indices:
                            continue
                            
                        # Get dates for this year
                        year_dates = df.index[year_indices]
                        if len(year_dates) == 0:
                            continue
                            
                        # Calculate index for each day in the year
                        for date in year_dates:
                            if pd.notna(df.loc[date, column]):
                                # Index = previous year-end value * (1 + YTD return/100)
                                index_values.loc[date] = prev_year_end * (1 + df.loc[date, column]/100)
                        
                        # Update prev_year_end for next year if we have data for the last day of the year
                        last_date_of_year = year_dates[-1]
                        if pd.notna(index_values.loc[last_date_of_year]):
                            prev_year_end = index_values.loc[last_date_of_year]
                    
                    # Fill any remaining NaN values using the specified fill method
                    index_values = index_values.ffill() if self.fill == 'ffill' else index_values.bfill()
                    result[f"{column}_index"] = index_values
            
            return result
        
        except Exception as e:
            logger.error(f"Error converting excess returns to index: {str(e)}")
            raise

    def print_high_low_statistics(self, data: pd.DataFrame = None):
        """
        Print high and low values with dates for each time series in the dataset.
        
        Args:
            data (pd.DataFrame, optional): Data to analyze. If None, uses the last processed data.
                Must have MultiIndex columns with (asset_class, column) structure.
        """
        try:
            df_to_analyze = data if data is not None else self.data
            
            if df_to_analyze is None:
                logger.warning("No data available for high/low statistics")
                return
            
            # Check if DataFrame has MultiIndex columns
            if not isinstance(df_to_analyze.columns, pd.MultiIndex):
                logger.warning("DataFrame does not have MultiIndex columns, skipping statistics")
                return
            
            print("\n" + "=" * 100)
            print("TIME SERIES HIGH/LOW STATISTICS")
            print("=" * 100)
            
            # Iterate through each column
            for (asset_class, column) in df_to_analyze.columns:
                series = df_to_analyze[(asset_class, column)]
                
                # Remove NaN values for accurate statistics
                valid_series = series.dropna()
                
                if len(valid_series) == 0:
                    print(f"\n{asset_class} | {column}: No valid data")
                    continue
                
                # Find high and low values
                high_value = valid_series.max()
                low_value = valid_series.min()
                
                # Find dates when high and low occurred
                high_date = valid_series.idxmax()
                low_date = valid_series.idxmin()
                
                # Format dates
                high_date_str = high_date.strftime('%Y-%m-%d') if isinstance(high_date, pd.Timestamp) else str(high_date)
                low_date_str = low_date.strftime('%Y-%m-%d') if isinstance(low_date, pd.Timestamp) else str(low_date)
                
                print(f"\n{asset_class} | {column}:")
                print(f"  High: {high_value:.4f} on {high_date_str}")
                print(f"  Low:  {low_value:.4f} on {low_date_str}")
                print(f"  Range: {high_value - low_value:.4f} ({((high_value - low_value) / low_value * 100):.2f}%)")
            
            print("\n" + "=" * 100)
            
        except Exception as e:
            logger.error(f"Error printing high/low statistics: {str(e)}")

    def visualize_data(self, data: pd.DataFrame = None, output_dir: str = None, asset_class: str = None):
        """
        Visualize time series data by asset class and save as interactive HTML files.
        
        Creates one HTML file per asset class with separate subplots for each series.
        Uses dark mode styling for professional appearance.
        
        Args:
            data (pd.DataFrame, optional): Data to visualize. If None, uses the last processed data.
                Must have MultiIndex columns with (asset_class, column) structure.
            output_dir (str, optional): Directory to save HTML files. If None, saves to 'visuals' folder.
            asset_class (str, optional): Specific asset class to visualize. If None, visualizes all asset classes.
            
        Returns:
            List[str]: List of paths to saved HTML files
            
        Raises:
            ValueError: If no data is available or Plotly is not installed
            Exception: If there's an error during visualization
        """
        if not PLOTLY_AVAILABLE:
            raise ValueError("Plotly is not installed. Please install it with: poetry add plotly")
        
        try:
            # Use provided data or last processed data
            df_to_plot = data if data is not None else self.data
            
            if df_to_plot is None:
                raise ValueError("No data available for visualization. Please process data first.")
            
            # Check if DataFrame has MultiIndex columns
            if not isinstance(df_to_plot.columns, pd.MultiIndex):
                raise ValueError("DataFrame must have MultiIndex columns with (asset_class, column) structure.")
            
            # Set output directory
            if output_dir is None:
                output_dir = os.path.join(os.path.dirname(__file__), "visuals")
            os.makedirs(output_dir, exist_ok=True)
            
            # Get asset classes to visualize
            all_asset_classes = df_to_plot.columns.get_level_values(0).unique()
            if asset_class:
                if asset_class not in all_asset_classes:
                    raise ValueError(f"Asset class '{asset_class}' not found in data. Available: {list(all_asset_classes)}")
                asset_classes_to_plot = [asset_class]
            else:
                asset_classes_to_plot = all_asset_classes
            
            saved_paths = []
            
            # Create visualization for each asset class
            for asset_class_name in asset_classes_to_plot:
                # Get columns for this asset class
                asset_class_df = df_to_plot[asset_class_name]
                
                if asset_class_df.empty:
                    logger.warning(f"No data found for asset class '{asset_class_name}', skipping.")
                    continue
                
                # Get number of series (columns) in this asset class
                num_series = len(asset_class_df.columns)
                
                # Create subplots: one per series, arranged vertically
                fig = make_subplots(
                    rows=num_series,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    subplot_titles=[col for col in asset_class_df.columns],
                    row_heights=[1] * num_series
                )
                
                # Define color palette (professional colors that work well in dark mode)
                colors = [
                    '#1f77b4',  # Blue
                    '#ff7f0e',  # Orange
                    '#2ca02c',  # Green
                    '#d62728',  # Red
                    '#9467bd',  # Purple
                    '#8c564b',  # Brown
                    '#e377c2',  # Pink
                    '#7f7f7f',  # Gray
                    '#bcbd22',  # Yellow-green
                    '#17becf',  # Cyan
                ]
                
                # Add trace for each series
                for idx, column in enumerate(asset_class_df.columns):
                    color = colors[idx % len(colors)]
                    fig.add_trace(
                        go.Scatter(
                            x=asset_class_df.index,
                            y=asset_class_df[column],
                            mode='lines',
                            name=column,
                            line=dict(color=color, width=2),
                            hovertemplate=f'<b>{column}</b><br>' +
                                        'Date: %{x|%Y-%m-%d}<br>' +
                                        'Value: %{y:.2f}<extra></extra>',
                            showlegend=False
                        ),
                        row=idx + 1,
                        col=1
                    )
                
                # Update layout with dark mode styling
                fig.update_layout(
                    title={
                        'text': f'{asset_class_name.replace("_", " ").title()} - Time Series',
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 24, 'color': '#d4d4d4'}
                    },
                    height=300 * num_series,  # 300px per subplot
                    template='plotly_dark',
                    paper_bgcolor='#1e1e1e',
                    plot_bgcolor='#252526',
                    font=dict(color='#d4d4d4', size=12),
                    hovermode='x unified',
                    showlegend=False,
                    dragmode='zoom'  # Enable zoom by default
                )
                
                # Update x-axis labels (only show on bottom subplot)
                for i in range(1, num_series + 1):
                    fig.update_xaxes(
                        title_text="Date" if i == num_series else "",
                        showgrid=True,
                        gridcolor='#3c3c3c',
                        gridwidth=1,
                        zeroline=False,
                        row=i,
                        col=1
                    )
                    fig.update_yaxes(
                        showgrid=True,
                        gridcolor='#3c3c3c',
                        gridwidth=1,
                        zeroline=True,
                        zerolinecolor='#555555',
                        zerolinewidth=1,
                        row=i,
                        col=1
                    )
                
                # Save to HTML file
                filename = f"{asset_class_name}.html"
                output_path = os.path.join(output_dir, filename)
                fig.write_html(
                    output_path, 
                    config={
                        'displayModeBar': True,
                        'modeBarButtonsToAdd': ['zoom', 'pan', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
                        'displaylogo': False
                    }
                )
                saved_paths.append(output_path)
                logger.info(f"Visualization saved: {output_path} ({num_series} series)")
            
            logger.info(f"Created {len(saved_paths)} visualization file(s)")
            return saved_paths
            
        except Exception as e:
            logger.error(f"Error in data visualization: {str(e)}")
            raise

    def process_data(self) -> pd.DataFrame:
        """Main method to fetch and process all data."""
        try:
            # Get the complete dataset including Bloomberg overrides
            final_df = self.get_full_dataset()
            logger.info("Successfully fetched and processed all data")

            # Store the processed data
            self.data = final_df

            # Redundant bad date handling removed, this is now fully handled in clean_data()

            return final_df
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean known data issues and handle missing values.
        
        This method:
        1. Handles known bad data points that need to be replaced
        2. Drops any rows with missing values
        3. Logs data completeness information
        
        Args:
            df (pd.DataFrame): DataFrame to clean
            
        Returns:
            pd.DataFrame: Cleaned DataFrame with no missing values
        """
        try:
            cleaned_df = df.copy()
            
            # Log initial data state
            initial_rows = len(cleaned_df)
            initial_missing = cleaned_df.isnull().sum()
            if initial_missing.any():
                logger.info("Initial missing values:")
                for col, count in initial_missing[initial_missing > 0].items():
                    logger.info(f"  {col}: {count} missing values")
        
            # Handle known bad dates from config format
            # Format in bad_dates is: {'date_key': {'column': 'col_name', 'action': 'action_type'}}
            if self.bad_dates:
                for date_key, info in self.bad_dates.items():
                    if 'column' in info and 'action' in info:
                        column = info['column']
                        action = info['action']
                        
                        # Handle different date_key formats
                        if '_' in date_key:
                            # Something like '2002-10-01_lei'
                            date = date_key.split('_')[0]
                        else:
                            # Just a date string
                            date = date_key
                            
                        # Convert to datetime
                        try:
                            date = pd.to_datetime(date)
                        except:
                            logger.warning(f"Could not parse date from {date_key}")
                            continue
                            
                        # Check if column exists, but don't warn if it doesn't
                        if column not in cleaned_df.columns:
                            logger.debug(f"Column {column} for bad date rule not found in current dataset, skipping.")
                            continue
                            
                        # Check if date exists
                        if date not in cleaned_df.index:
                            logger.warning(f"Date {date} not found in data")
                            continue
                            
                        logger.info(f"Processing bad date {date} for column {column}, action: {action}")
                        
                        # Apply action
                        if action == 'use_previous':
                            # Get previous value
                            prev_values = cleaned_df.loc[cleaned_df.index < date, column]
                            if not prev_values.empty:
                                prev_value = prev_values.iloc[-1]
                                cleaned_df.loc[date, column] = prev_value
                                logger.info(f"  Used previous value {prev_value} for {column} on {date}")
                            else:
                                logger.warning(f"  No previous values for {column} before {date}")
                                
                        elif action == 'forward_fill':
                            # This was previously implemented incorrectly as a backfill.
                            # The goal of ffill on a single point is to use the *previous* value.
                            # Let's correct this to use a proper forward fill.
                            cleaned_df[column] = cleaned_df[column].ffill()
                            logger.info(f"  Forward filled column {column} up to {date}")
                                
                        elif action == 'interpolate':
                            # Get values before and after
                            before_idx = cleaned_df.index.get_indexer([date], method='pad')[0]
                            after_idx = cleaned_df.index.get_indexer([date], method='backfill')[0]
                            
                            if before_idx >= 0 and after_idx < len(cleaned_df) and before_idx != after_idx:
                                before_date = cleaned_df.index[before_idx]
                                after_date = cleaned_df.index[after_idx]
                                before_val = cleaned_df.loc[before_date, column]
                                after_val = cleaned_df.loc[after_date, column]
                                
                                # Calculate interpolated value
                                days_diff = (after_date - before_date).days
                                target_diff = (date - before_date).days
                                ratio = target_diff / days_diff if days_diff > 0 else 0
                                interp_val = before_val + (after_val - before_val) * ratio
                                
                                cleaned_df.loc[date, column] = interp_val
                                logger.info(f"  Interpolated to {interp_val} for {column} on {date}")
                            else:
                                logger.warning(f"  Could not interpolate for {column} on {date}")
                        
                        else:
                            logger.warning(f"  Unknown action {action} for {column} on {date}")
                
            # Apply more generic cleaning for the standard format from DEFAULT_BAD_DATES
            if HANDLE_BAD_DATES and hasattr(self, 'bad_dates_standardized'):
                for ticker, date_range in self.bad_dates_standardized.items():
                    # Extract ticker base name for column matching
                    ticker_base = ticker.split(' ')[0]
                    
                    # Find columns that might be related to this ticker
                    related_columns = [col for col in cleaned_df.columns if ticker_base.lower() in col.lower()]
                    
                    if related_columns:
                        # Extract date range
                        start_date = pd.to_datetime(date_range.get('start_date'))
                        end_date = pd.to_datetime(date_range.get('end_date'))
                        
                        if start_date and end_date:
                            # Create mask for the bad date range
                            bad_date_mask = (cleaned_df.index >= start_date) & (cleaned_df.index <= end_date)
                            
                            if bad_date_mask.any():
                                logger.info(f"Handling bad dates for {ticker} from {start_date} to {end_date}")
                                
                                for col in related_columns:
                                    # Similar logic as before for interpolation...
                                    # (code omitted for brevity)
                                    pass
            
            # Don't drop rows with missing values here - let the fill method handle it
            # This allows columns with different start dates to coexist
            # The fill parameter (default 'ffill') will handle missing values after cleaning
            logger.info(f"Data cleaning complete. Rows: {len(cleaned_df):,}, Columns: {len(cleaned_df.columns)}")
            
            return cleaned_df
        
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise

    def get_full_dataset(self) -> pd.DataFrame:
        """Get complete dataset with price data, excess return indices, and Bloomberg override data."""
        try:
            dfs_to_merge = []
            
            if self.enable_ohlc_data:
                logger.info("Fetching main price data...")
                df_ohlc = self.fetch_bloomberg_data(mapping=self.ohlc_mapping)
                dfs_to_merge.append(df_ohlc)
            
            if self.enable_er_ytd_data:
                logger.info("Fetching excess return YTD data...")
                er_ytd_df = self.fetch_bloomberg_data(mapping=self.er_ytd_mapping)
                logger.info("Converting excess returns to indices...")
                er_index_df = self.convert_er_ytd_to_index(er_ytd_df)
                dfs_to_merge.append(er_index_df)
            
            if self.enable_bloomberg_overrides_data:
                logger.info("Fetching Bloomberg data with overrides...")
                bloomberg_overrides_df = self.fetch_bloomberg_overrides_data()
                dfs_to_merge.append(bloomberg_overrides_df)
            
            logger.info("Merging datasets...")
            # Use our integrated merge_dfs method instead of the imported one
            final_df = self.merge_dfs(*dfs_to_merge,
                                fill=self.fill, start_date_align=self.start_date_align)
            
            logger.info("Cleaning data...")
            final_df = self.clean_data(final_df)

            # Align start date after cleaning to ensure we are aligning based on clean, non-null data
            if self.start_date_align == 'yes':
                non_null_df = final_df.dropna(how='any')
                if not non_null_df.empty:
                    first_complete_date = non_null_df.index[0]
                    final_df = final_df[final_df.index >= first_complete_date]
                    logger.info(f"Aligned data to start from first complete date: {first_complete_date}")

            if self.fill:
                final_df = final_df.ffill() if self.fill == 'ffill' else final_df.bfill()
            
            # Calculate total return index for CAD 3M T-bill
            final_df = self.calculate_cad_3m_tbill_tr_index(final_df)
            
            # Calculate leveraged credit exposure excess return indices
            final_df = self.calculate_levered_credit_er_indices(final_df)
            
            # Calculate combined gross return indices (leveraged credit + risk-free rate)
            final_df = self.calculate_combined_gross_return_indices(final_df)
            
            # Calculate net return indices (gross return adjusted for fees)
            final_df = self.calculate_net_return_indices(final_df)
            
            # Organize columns by asset class with MultiIndex structure
            final_df = self.organize_columns_by_asset_class(final_df)
            
            return final_df
        
        except Exception as e:
            logger.error(f"Error getting full dataset: {str(e)}")
            raise

    def calculate_cad_3m_tbill_tr_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate total return index for CAD 3M T-bill starting at 100.
        
        The index compounds daily using the annualized rate with Actual/365 day count convention.
        Rate is converted from percentage to decimal (divide by 100).
        Missing rates are forward filled.
        
        Formula: index[t] = index[t-1] * (1 + rate[t]/100 * days_between/365)
        
        Args:
            df: DataFrame with 'cad_3m_tbill' column
            
        Returns:
            DataFrame with added 'cad_3m_tbill_tr_index' column
        """
        logger.info("Calculating CAD 3M T-bill total return index...")
        
        # Check if cad_3m_tbill column exists
        if 'cad_3m_tbill' not in df.columns:
            logger.warning("cad_3m_tbill column not found. Skipping total return index calculation.")
            return df
        
        # Get the rate series
        rate_series = df['cad_3m_tbill'].copy()
        
        # Check if we have any valid data
        if rate_series.isna().all():
            logger.warning("cad_3m_tbill has no valid data. Skipping total return index calculation.")
            return df
        
        # Forward fill missing rates
        rate_series = rate_series.ffill()
        
        # Get first valid date from rate series (not necessarily start_date)
        first_valid_date = rate_series.first_valid_index()
        if first_valid_date is None:
            logger.warning("cad_3m_tbill has no valid data. Skipping total return index calculation.")
            return df
        
        # Use the later of start_date or first_valid_date
        start_date = pd.Timestamp(self.start_date)
        if first_valid_date > start_date:
            start_date = first_valid_date
            logger.info(f"Starting CAD 3M T-bill TR index from first valid data date: {start_date}")
        
        end_date = df.index.max()
        
        # Create full date range (daily)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Reindex rate series to full date range and forward fill
        rate_series_full = rate_series.reindex(date_range).ffill()
        
        # Check if we have valid rates after reindexing
        if rate_series_full.isna().all():
            logger.warning("cad_3m_tbill has no valid data after reindexing. Skipping total return index calculation.")
            return df
        
        # Get first valid rate value to fill any leading NaNs
        first_valid_idx = rate_series_full.first_valid_index()
        if first_valid_idx is not None:
            first_valid_rate = rate_series_full.loc[first_valid_idx]
            # Fill any leading NaNs with first valid rate (backfill then forward fill)
            rate_series_full = rate_series_full.bfill().fillna(first_valid_rate)
        
        logger.info(f"Rate series stats: min={rate_series_full.min():.4f}, max={rate_series_full.max():.4f}, mean={rate_series_full.mean():.4f}, first_value={rate_series_full.iloc[0]:.4f}")
        
        # Initialize index series starting at 100
        tr_index = pd.Series(index=date_range, dtype=float)
        tr_index.iloc[0] = 100.0
        
        # Calculate daily returns and compound
        for i in range(1, len(tr_index)):
            prev_date = tr_index.index[i-1]
            curr_date = tr_index.index[i]
            
            # Calculate days between (Actual/365)
            days_between = (curr_date - prev_date).days
            
            # Get rate for current period (convert from percentage to decimal)
            rate_value = rate_series_full.iloc[i]
            
            # Skip if rate is still NaN (shouldn't happen after fillna, but safety check)
            if pd.isna(rate_value):
                tr_index.iloc[i] = tr_index.iloc[i-1]  # Keep previous value
                continue
            
            rate_decimal = rate_value / 100.0
            
            # Calculate daily return factor
            # For daily compounding: (1 + annual_rate * days/365)
            return_factor = 1 + (rate_decimal * days_between / 365.0)
            
            # Compound the index
            tr_index.iloc[i] = tr_index.iloc[i-1] * return_factor
        
        logger.info(f"TR index stats: start={tr_index.iloc[0]:.4f}, end={tr_index.iloc[-1]:.4f}, min={tr_index.min():.4f}, max={tr_index.max():.4f}")
        
        # Reindex to match original dataframe index
        tr_index_aligned = tr_index.reindex(df.index).ffill()
        
        # Fill any remaining NaN values with 100 (before first valid date)
        tr_index_aligned = tr_index_aligned.fillna(100.0)
        
        # Add to dataframe
        df['cad_3m_tbill_tr_index'] = tr_index_aligned
        
        logger.info(f"CAD 3M T-bill total return index calculated. Final value: {tr_index_aligned.iloc[-1]:.4f}")
        
        return df

    def calculate_levered_credit_er_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate leveraged total return indices for CAD credit exposure excess return indices.
        
        Creates 3x leveraged indices starting at 100 with annualized borrow cost deduction.
        Leverage is applied to period returns, and borrow cost is adjusted based on actual
        days between observations.
        
        Formula: leveraged_return[t] = (underlying_return[t] × leverage_multiplier) - 
                                        (annual_borrow_cost × days_between / 365)
        
        The leveraged return is then compounded: index[t] = index[t-1] × (1 + leveraged_return[t]/100)
        
        Args:
            df: DataFrame with underlying excess return index columns:
                - 'cad_ig_er_index'
                - 'cad_ig_short_er_index'
                - 'cad_ig_long_er_index'
            
        Returns:
            DataFrame with added leveraged index columns:
                - 'cad_ig_er_index_3x'
                - 'cad_ig_short_er_index_3x'
                - 'cad_ig_long_er_index_3x'
        """
        logger.info("Calculating leveraged credit exposure excess return indices...")
        
        # Define mapping: underlying column -> leveraged column
        underlying_to_levered = {
            'cad_ig_er_index': 'cad_ig_er_index_3x',
            'cad_ig_short_er_index': 'cad_ig_short_er_index_3x',
            'cad_ig_long_er_index': 'cad_ig_long_er_index_3x',
        }
        
        # Get configuration values
        leverage_multiplier = getattr(self, 'leverage_multiplier', LEVERAGE_MULTIPLIER)
        annual_borrow_cost_bps = getattr(self, 'annual_borrow_cost_bps', ANNUAL_BORROW_COST_BPS)
        annual_borrow_cost_decimal = annual_borrow_cost_bps / 10000.0  # Convert bps to decimal
        
        logger.info(f"Using leverage multiplier: {leverage_multiplier}x, annual borrow cost: {annual_borrow_cost_bps}bps")
        
        # Process each underlying index
        for underlying_col, levered_col in underlying_to_levered.items():
            if underlying_col not in df.columns:
                logger.warning(f"{underlying_col} column not found. Skipping {levered_col} calculation.")
                continue
            
            # Get underlying index series
            underlying_series = df[underlying_col].copy()
            
            # Check if we have any valid data
            if underlying_series.isna().all():
                logger.warning(f"{underlying_col} has no valid data. Skipping {levered_col} calculation.")
                continue
            
            # Forward fill missing values before calculating returns
            underlying_series = underlying_series.ffill()
            
            # Get first valid date
            first_valid_date = underlying_series.first_valid_index()
            if first_valid_date is None:
                logger.warning(f"{underlying_col} has no valid data. Skipping {levered_col} calculation.")
                continue
            
            # Get date range from first valid date to end
            end_date = df.index.max()
            date_range = df.index[df.index >= first_valid_date]
            
            if len(date_range) == 0:
                logger.warning(f"No valid dates for {underlying_col}. Skipping {levered_col} calculation.")
                continue
            
            # Initialize leveraged index series starting at 100
            levered_index = pd.Series(index=date_range, dtype=float)
            levered_index.iloc[0] = 100.0
            
            # Calculate leveraged returns and compound
            for i in range(1, len(levered_index)):
                prev_date = levered_index.index[i-1]
                curr_date = levered_index.index[i]
                
                # Get underlying index values
                prev_underlying = underlying_series.loc[prev_date]
                curr_underlying = underlying_series.loc[curr_date]
                
                # Skip if either value is NaN (shouldn't happen after ffill, but safety check)
                if pd.isna(prev_underlying) or pd.isna(curr_underlying):
                    levered_index.iloc[i] = levered_index.iloc[i-1]  # Keep previous value
                    continue
                
                # Calculate underlying period return (as percentage)
                underlying_return_pct = ((curr_underlying / prev_underlying) - 1) * 100
                
                # Calculate days between observations
                days_between = (curr_date - prev_date).days
                
                # Calculate leveraged return: (underlying_return × leverage) - (borrow_cost × days/365)
                leveraged_return_pct = (underlying_return_pct * leverage_multiplier) - \
                                       (annual_borrow_cost_decimal * 100 * days_between / 365.0)
                
                # Compound the index: index[t] = index[t-1] × (1 + leveraged_return/100)
                levered_index.iloc[i] = levered_index.iloc[i-1] * (1 + leveraged_return_pct / 100.0)
            
            # Reindex to match original dataframe index and forward fill
            levered_index_aligned = levered_index.reindex(df.index).ffill()
            
            # Fill any remaining NaN values with 100 (before first valid date)
            levered_index_aligned = levered_index_aligned.fillna(100.0)
            
            # Add to dataframe
            df[levered_col] = levered_index_aligned
            
            logger.info(f"{levered_col} calculated. Start: {levered_index.iloc[0]:.4f}, "
                       f"End: {levered_index.iloc[-1]:.4f}, "
                       f"Min: {levered_index.min():.4f}, Max: {levered_index.max():.4f}")
        
        logger.info("Leveraged credit exposure excess return indices calculation complete.")
        
        return df

    def calculate_combined_gross_return_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate combined gross return indices combining leveraged credit exposure with risk-free rate.
        
        Creates indices using additive returns: gross return = leveraged return + risk-free return.
        The combined index compounds daily by adding the period returns from both strategies.
        
        Formula: 
            leveraged_return[t] = (leveraged_index[t] / leveraged_index[t-1]) - 1
            tbill_return[t] = (tbill_index[t] / tbill_index[t-1]) - 1
            combined_return[t] = leveraged_return[t] + tbill_return[t]
            combined_index[t] = combined_index[t-1] * (1 + combined_return[t])
        
        This represents a strategy that earns both returns simultaneously (additive, not weighted average).
        The combined index starts at 100 and compounds using the sum of returns from both strategies.
        
        Args:
            df: DataFrame with leveraged index columns and cad_3m_tbill_tr_index:
                - 'cad_ig_er_index_3x'
                - 'cad_ig_short_er_index_3x'
                - 'cad_ig_long_er_index_3x'
                - 'cad_3m_tbill_tr_index'
            
        Returns:
            DataFrame with added combined gross return index columns:
                - 'cad_ig_gross_return_index_3x'
                - 'cad_ig_short_gross_return_index_3x'
                - 'cad_ig_long_gross_return_index_3x'
        """
        logger.info("Calculating combined gross return indices (leveraged credit + risk-free rate)...")
        
        # Check if cad_3m_tbill_tr_index exists
        if 'cad_3m_tbill_tr_index' not in df.columns:
            logger.warning("cad_3m_tbill_tr_index column not found. Skipping combined gross return index calculation.")
            return df
        
        # Get configuration values (both default to 1.0 for 100% each)
        leveraged_allocation = getattr(self, 'leveraged_allocation', LEVERAGED_ALLOCATION)
        risk_free_allocation = getattr(self, 'risk_free_allocation', RISK_FREE_ALLOCATION)
        
        logger.info(f"Using allocations: {leveraged_allocation:.0%} leveraged credit, {risk_free_allocation:.0%} risk-free rate")
        
        # Define mapping: leveraged column -> combined column
        leveraged_to_combined = {
            'cad_ig_er_index_3x': 'cad_ig_gross_return_index_3x',
            'cad_ig_short_er_index_3x': 'cad_ig_short_gross_return_index_3x',
            'cad_ig_long_er_index_3x': 'cad_ig_long_gross_return_index_3x',
        }
        
        # Get tbill index series and forward fill missing values
        tbill_series = df['cad_3m_tbill_tr_index'].copy().ffill()
        
        # Check if we have any valid tbill data
        if tbill_series.isna().all():
            logger.warning("cad_3m_tbill_tr_index has no valid data. Skipping combined gross return index calculation.")
            return df
        
        # Process each leveraged index
        for leveraged_col, combined_col in leveraged_to_combined.items():
            if leveraged_col not in df.columns:
                logger.warning(f"{leveraged_col} column not found. Skipping {combined_col} calculation.")
                continue
            
            # Get leveraged index series
            leveraged_series = df[leveraged_col].copy()
            
            # Check if we have any valid data
            if leveraged_series.isna().all():
                logger.warning(f"{leveraged_col} has no valid data. Skipping {combined_col} calculation.")
                continue
            
            # Forward fill missing values
            leveraged_series = leveraged_series.ffill()
            tbill_series_filled = tbill_series.ffill()
            
            # Find common date range (where both series have data)
            leveraged_valid = leveraged_series.notna()
            tbill_valid = tbill_series_filled.notna()
            common_valid = leveraged_valid & tbill_valid
            
            if not common_valid.any():
                logger.warning(f"No common valid dates between {leveraged_col} and cad_3m_tbill_tr_index. Skipping {combined_col} calculation.")
                continue
            
            # Get first valid date
            first_valid_date = common_valid.idxmax() if common_valid.any() else None
            if first_valid_date is None:
                logger.warning(f"No valid dates for {combined_col}. Skipping calculation.")
                continue
            
            # Get date range from first valid date to end
            date_range = df.index[df.index >= first_valid_date]
            
            if len(date_range) == 0:
                logger.warning(f"No valid dates for {combined_col}. Skipping calculation.")
                continue
            
            # Initialize combined index series starting at 100
            combined_index = pd.Series(index=date_range, dtype=float)
            combined_index.iloc[0] = 100.0
            
            # Calculate combined returns using additive method
            # Gross return = leveraged return + risk-free return (additive, not weighted average)
            for i in range(1, len(combined_index)):
                prev_date = combined_index.index[i-1]
                curr_date = combined_index.index[i]
                
                # Get index values
                prev_leveraged = leveraged_series.loc[prev_date]
                curr_leveraged = leveraged_series.loc[curr_date]
                prev_tbill = tbill_series_filled.loc[prev_date]
                curr_tbill = tbill_series_filled.loc[curr_date]
                
                # Skip if any value is NaN (shouldn't happen after ffill, but safety check)
                if pd.isna(prev_leveraged) or pd.isna(curr_leveraged) or pd.isna(prev_tbill) or pd.isna(curr_tbill):
                    combined_index.iloc[i] = combined_index.iloc[i-1]  # Keep previous value
                    continue
                
                # Calculate period returns (as decimals, not percentages)
                leveraged_return = (curr_leveraged / prev_leveraged) - 1.0
                tbill_return = (curr_tbill / prev_tbill) - 1.0
                
                # Combined return is additive: leveraged return + risk-free return
                combined_return = leveraged_return + tbill_return
                
                # Compound the combined index
                # Formula: combined_index[t] = combined_index[t-1] * (1 + combined_return)
                combined_index.iloc[i] = combined_index.iloc[i-1] * (1.0 + combined_return)
            
            # Reindex to match original dataframe index and forward fill
            combined_index_aligned = combined_index.reindex(df.index).ffill()
            
            # Fill any remaining NaN values with 100 (before first valid date)
            combined_index_aligned = combined_index_aligned.fillna(100.0)
            
            # Add to dataframe
            df[combined_col] = combined_index_aligned
            
            logger.info(f"{combined_col} calculated. Start: {combined_index.iloc[0]:.4f}, "
                       f"End: {combined_index.iloc[-1]:.4f}, "
                       f"Min: {combined_index.min():.4f}, Max: {combined_index.max():.4f}")
        
        logger.info("Combined gross return indices calculation complete.")
        
        return df

    def calculate_net_return_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate net return indices adjusting gross return indices for fees.
        
        Deducts:
        - Management fee: 0.90% annualized (deducted daily)
        - Operational fee: 0.20% annualized (deducted daily)
        - Performance fee: 15% of positive annual returns (deducted at year-end, no HWM)
        
        Formula:
            Daily net return = daily_gross_return - (management_fee + operational_fee)/365
            At year-end: if annual_return > 0, deduct 15% performance fee
        
        Args:
            df: DataFrame with gross return index columns:
                - 'cad_ig_gross_return_index_3x'
                - 'cad_ig_short_gross_return_index_3x'
                - 'cad_ig_long_gross_return_index_3x'
            
        Returns:
            DataFrame with added net return index columns:
                - 'cad_ig_net_return_index_3x'
                - 'cad_ig_short_net_return_index_3x'
                - 'cad_ig_long_net_return_index_3x'
        """
        logger.info("Calculating net return indices (gross return adjusted for fees)...")
        
        # Get fee configuration values
        annual_mgmt_fee_pct = getattr(self, 'annual_management_fee_pct', ANNUAL_MANAGEMENT_FEE_PCT)
        annual_operational_fee_pct = getattr(self, 'annual_operational_fee_pct', ANNUAL_OPERATIONAL_FEE_PCT)
        performance_fee_pct = getattr(self, 'performance_fee_pct', PERFORMANCE_FEE_PCT)
        
        # Calculate daily fee rates (all fees applied daily at equivalent annualized rates)
        daily_mgmt_fee = annual_mgmt_fee_pct / 100.0 / 365.0  # Convert to daily decimal
        daily_operational_fee = annual_operational_fee_pct / 100.0 / 365.0  # Convert to daily decimal
        performance_fee_multiplier = 1.0 - (performance_fee_pct / 100.0)  # Performance fee multiplier (1 - 15% = 0.85)
        
        logger.info(f"Using fees: Management {annual_mgmt_fee_pct}% annualized (applied daily), "
                   f"Operational {annual_operational_fee_pct}% annualized (applied daily), "
                   f"Performance {performance_fee_pct}% applied daily to positive returns")
        
        # Define mapping: gross column -> net column
        gross_to_net = {
            'cad_ig_gross_return_index_3x': 'cad_ig_net_return_index_3x',
            'cad_ig_short_gross_return_index_3x': 'cad_ig_short_net_return_index_3x',
            'cad_ig_long_gross_return_index_3x': 'cad_ig_long_net_return_index_3x',
        }
        
        # Process each gross return index
        for gross_col, net_col in gross_to_net.items():
            if gross_col not in df.columns:
                logger.warning(f"{gross_col} column not found. Skipping {net_col} calculation.")
                continue
            
            # Get gross return index series
            gross_series = df[gross_col].copy()
            
            # Check if we have any valid data
            if gross_series.isna().all():
                logger.warning(f"{gross_col} has no valid data. Skipping {net_col} calculation.")
                continue
            
            # Forward fill missing values
            gross_series = gross_series.ffill()
            
            # Get first valid date
            first_valid_date = gross_series.first_valid_index()
            if first_valid_date is None:
                logger.warning(f"{gross_col} has no valid data. Skipping {net_col} calculation.")
                continue
            
            # Get date range from first valid date to end
            date_range = df.index[df.index >= first_valid_date]
            
            if len(date_range) == 0:
                logger.warning(f"No valid dates for {gross_col}. Skipping {net_col} calculation.")
                continue
            
            # Initialize net return index series starting at 100
            net_index = pd.Series(index=date_range, dtype=float)
            net_index.iloc[0] = 100.0
            
            # Calculate net index with daily fee application
            # Formula: net_return = (gross_return - mgmt_fee - operational_fee) * (1 - performance_fee) for positive returns
            #          net_return = (gross_return - mgmt_fee - operational_fee) for negative returns
            for i in range(1, len(net_index)):
                prev_date = net_index.index[i-1]
                curr_date = net_index.index[i]
                
                # Get gross return index values
                prev_gross = gross_series.loc[prev_date]
                curr_gross = gross_series.loc[curr_date]
                
                # Skip if any value is NaN
                if pd.isna(prev_gross) or pd.isna(curr_gross):
                    net_index.iloc[i] = net_index.iloc[i-1]  # Keep previous value
                    continue
                
                # Calculate daily gross return (as decimal)
                daily_gross_return = (curr_gross / prev_gross) - 1.0
                
                # Calculate daily net return after fees
                # Subtract daily management and operational fees
                daily_return_after_mgmt_ops = daily_gross_return - daily_mgmt_fee - daily_operational_fee
                
                # Apply performance fee (15%) to positive returns only
                if daily_return_after_mgmt_ops > 0:
                    daily_net_return = daily_return_after_mgmt_ops * performance_fee_multiplier
                else:
                    daily_net_return = daily_return_after_mgmt_ops
                
                # Compound the net index: net_index[t] = net_index[t-1] * (1 + daily_net_return)
                net_index.iloc[i] = net_index.iloc[i-1] * (1.0 + daily_net_return)
            
            # Reindex to match original dataframe index and forward fill
            net_index_aligned = net_index.reindex(df.index).ffill()
            
            # Fill any remaining NaN values with 100 (before first valid date)
            net_index_aligned = net_index_aligned.fillna(100.0)
            
            # Add to dataframe
            df[net_col] = net_index_aligned
            
            logger.info(f"{net_col} calculated. Start: {net_index.iloc[0]:.4f}, "
                       f"End: {net_index.iloc[-1]:.4f}, "
                       f"Min: {net_index.min():.4f}, Max: {net_index.max():.4f}")
        
        logger.info("Net return indices calculation complete.")
        
        return df

    def organize_columns_by_asset_class(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Organize DataFrame columns with MultiIndex structure (asset_class, column_name).
        
        This creates a hierarchical column structure where:
        - Level 0: Asset class (e.g., 'cad_credit_spreads', 'us_credit_spreads', 'cad_credit_exposure', 'us_credit_exposure')
        - Level 1: Column name (e.g., 'cad_oas', 'us_ig_oas', 'us_hy_oas')
        
        When saved to CSV, this creates a two-row header showing the grouping.
        
        Args:
            df: DataFrame with columns to organize
            
        Returns:
            DataFrame with MultiIndex columns organized by asset class
        """
        # Create new column structure with (asset_class, column_name) tuples
        new_columns = []
        for col in df.columns:
            asset_class = COLUMN_TO_ASSET_CLASS.get(col, 'other')
            new_columns.append((asset_class, col))
        
        # Create MultiIndex
        df.columns = pd.MultiIndex.from_tuples(new_columns, names=['asset_class', 'column'])
        
        # Reorder by asset class groups (maintain order within groups)
        ordered_tuples = []
        for asset_class in ASSET_CLASS_GROUPS.keys():
            for col in ASSET_CLASS_GROUPS[asset_class]:
                if (asset_class, col) in df.columns:
                    ordered_tuples.append((asset_class, col))
        
        # Add any ungrouped columns at the end
        for col_tuple in df.columns:
            if col_tuple not in ordered_tuples:
                ordered_tuples.append(col_tuple)
        
        # Reorder DataFrame
        reordered_df = df[ordered_tuples].copy()
        
        logger.info(f"Organized columns with MultiIndex by asset class. Groups: {list(ASSET_CLASS_GROUPS.keys())}")
        logger.info(f"Column structure: {len(reordered_df.columns)} columns across {reordered_df.columns.nlevels} levels")
        
        return reordered_df

    def save_dataset(self, df: pd.DataFrame = None, output_path=None, date_format=None):
        """
        Save the dataset to a file.
        
        Args:
            df (pd.DataFrame, optional): DataFrame to save. If not provided, it will be fetched.
            output_path (str): Path to save the dataset to.
            date_format (str): Format to use for the date index.
            
        Returns:
            str: The path to the saved dataset.
        """
        # Get project root directory if not already set
        if not hasattr(self, 'project_root'):
            self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
        # Use config default path if none provided
        if output_path is None:
            output_path = getattr(self, 'default_output_path', DEFAULT_OUTPUT_PATH)
            
        # Handle relative paths
        if not os.path.isabs(output_path):
            output_path = os.path.join(self.project_root, output_path)
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            if df is None:
                df = self.get_full_dataset()
            
            # Ensure index name is 'Date'
            df.index.name = 'Date'
            
            # Handle MultiIndex columns in CSV
            if isinstance(df.columns, pd.MultiIndex):
                # Save with MultiIndex header (creates two-row header in CSV)
                df.to_csv(output_path)
                logger.info(f"Saved dataset with MultiIndex columns (asset_class, column) to {output_path}")
            else:
                # Standard single-level columns
                if date_format:
                    df.to_csv(output_path, date_format=date_format)
                else:
                    df.to_csv(output_path)
            
            logger.info(f"Dataset saved to {output_path} with shape {df.shape}")
            if isinstance(df.columns, pd.MultiIndex):
                logger.info(f"Column structure: {df.columns.nlevels} level(s), {len(df.columns)} total columns")
                logger.info(f"Asset classes: {df.columns.get_level_values(0).unique().tolist()}")
            
            # Generate data quality log file
            try:
                self.generate_data_log(df=df, csv_path=output_path)
            except Exception as e:
                logger.warning(f"Could not generate data log file: {str(e)}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving dataset: {str(e)}")
            raise

    def generate_data_log(self, df: pd.DataFrame = None, csv_path: str = None, log_file: str = None):
        """
        Generate a comprehensive log file with data quality information.
        
        Creates a log file similar to parquet_stats.log with df.info(), shape, date range,
        missing data statistics, and sample rows (head/tail).
        
        Args:
            df (pd.DataFrame, optional): DataFrame to analyze. If None, loads from csv_path.
            csv_path (str, optional): Path to CSV file to load. Required if df is None.
            log_file (str, optional): Path to log file. Defaults to bond_data/logs/market_data_stats.log
            
        Returns:
            str: Path to the generated log file
        """
        from io import StringIO
        
        try:
            # Load data if not provided
            if df is None:
                if csv_path is None:
                    csv_path = getattr(self, 'default_output_path', DEFAULT_OUTPUT_PATH)
                df = self.load_dataset(csv_path, parse_dates=True)
            
            # Set up log file path
            if log_file is None:
                # Get project root directory
                if not hasattr(self, 'project_root'):
                    self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                log_file = os.path.join(self.project_root, 'bond_data', 'logs', 'market_data_stats.log')
            
            # Ensure log directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # Set up logger for this specific log file
            log_logger = logging.getLogger('market_data_stats')
            log_logger.setLevel(logging.INFO)
            
            # Remove existing handlers to avoid duplicates
            log_logger.handlers = []
            
            # Create file handler
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8', errors='replace')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            log_logger.addHandler(file_handler)
            
            # Get current timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Write header
            log_logger.info("=" * 100)
            log_logger.info(f"MARKET DATA TIME SERIES DIAGNOSTICS @ {timestamp}")
            log_logger.info("-" * 100)
            
            # Dataset information
            csv_path_display = csv_path if csv_path else "In-memory DataFrame"
            log_logger.info(f"Dataset: market_data_time_series.csv")
            log_logger.info(f"Path: {os.path.abspath(csv_path_display) if csv_path else 'N/A (in-memory)'}")
            log_logger.info(f"Shape: {len(df):,} rows x {len(df.columns)} columns")
            
            # Date range
            log_logger.info(f"Date Range: {df.index.min()} to {df.index.max()}")
            log_logger.info(f"Total Days: {(df.index.max() - df.index.min()).days + 1:,}")
            log_logger.info(f"Business Days: {len(df):,}")
            
            # Column structure information
            if isinstance(df.columns, pd.MultiIndex):
                log_logger.info(f"Column Structure: MultiIndex with {df.columns.nlevels} levels")
                log_logger.info(f"Asset Classes: {len(df.columns.get_level_values(0).unique())}")
                log_logger.info(f"Total Columns: {len(df.columns)}")
                
                # Asset class breakdown
                log_logger.info("\nAsset Class Breakdown:")
                asset_class_counts = df.columns.get_level_values(0).value_counts().sort_index()
                for asset_class, count in asset_class_counts.items():
                    log_logger.info(f"  {asset_class}: {count} column(s)")
            else:
                log_logger.info(f"Column Structure: Single-level ({len(df.columns)} columns)")
            
            # DataFrame info()
            log_logger.info("\nDataFrame info():")
            # Capture df.info() output
            info_buffer = StringIO()
            df.info(buf=info_buffer)
            info_output = info_buffer.getvalue()
            # Log each line
            for line in info_output.split('\n'):
                if line.strip():
                    log_logger.info(f"    {line}")
            
            # Missing data statistics
            log_logger.info("\nMissing Data Statistics:")
            missing_data = df.isnull().sum()
            missing_pct = (missing_data / len(df) * 100).round(2)
            
            if isinstance(df.columns, pd.MultiIndex):
                # Group by asset class for MultiIndex
                missing_by_asset = {}
                for (asset_class, column) in df.columns:
                    if asset_class not in missing_by_asset:
                        missing_by_asset[asset_class] = {'total': 0, 'columns': []}
                    missing_count = missing_data[(asset_class, column)]
                    missing_by_asset[asset_class]['total'] += missing_count
                    if missing_count > 0:
                        missing_by_asset[asset_class]['columns'].append((column, missing_count, missing_pct[(asset_class, column)]))
                
                for asset_class, info in missing_by_asset.items():
                    if info['total'] > 0:
                        log_logger.info(f"  {asset_class}: {info['total']:,} missing values across {len(info['columns'])} column(s)")
                        for column, count, pct in info['columns'][:5]:  # Show first 5 columns
                            log_logger.info(f"    {column}: {count:,} ({pct}%)")
                        if len(info['columns']) > 5:
                            log_logger.info(f"    ... and {len(info['columns']) - 5} more column(s)")
                    else:
                        log_logger.info(f"  {asset_class}: No missing data")
            else:
                # Single-level columns
                columns_with_missing = missing_data[missing_data > 0].sort_values(ascending=False)
                if len(columns_with_missing) > 0:
                    log_logger.info(f"Columns with missing data: {len(columns_with_missing)}")
                    for column, count in columns_with_missing.head(10).items():
                        pct = missing_pct[column]
                        log_logger.info(f"  {column}: {count:,} ({pct}%)")
                    if len(columns_with_missing) > 10:
                        log_logger.info(f"  ... and {len(columns_with_missing) - 10} more column(s)")
                else:
                    log_logger.info("No missing data")
            
            # Basic statistics for numeric columns
            log_logger.info("\nBasic Statistics (numeric columns only):")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stats_df = df[numeric_cols].describe()
                # Log summary statistics
                log_logger.info(f"  Numeric columns: {len(numeric_cols)}")
                log_logger.info(f"  Mean values range: {stats_df.loc['mean'].min():.4f} to {stats_df.loc['mean'].max():.4f}")
                log_logger.info(f"  Std dev range: {stats_df.loc['std'].min():.4f} to {stats_df.loc['std'].max():.4f}")
            else:
                log_logger.info("  No numeric columns found")
            
            # Head sample
            log_logger.info("\nHead (3 rows):")
            head_str = df.head(3).to_string()
            for line in head_str.split('\n'):
                log_logger.info(f"    {line}")
            
            # Tail sample
            log_logger.info("\nTail (3 rows):")
            tail_str = df.tail(3).to_string()
            for line in tail_str.split('\n'):
                log_logger.info(f"    {line}")
            
            # Footer
            log_logger.info("\n" + "=" * 100)
            
            logger.info(f"Market data statistics log generated: {log_file}")
            return log_file
            
        except Exception as e:
            logger.error(f"Error generating data log: {str(e)}")
            raise

    def load_dataset(self, file_path, parse_dates=True):
        """
        Load a dataset from a file.
        
        Args:
            file_path (str): Path to the dataset. Can be absolute or relative to project root.
            parse_dates (bool): Whether to parse the dates.
            
        Returns:
            pd.DataFrame: The loaded dataset with proper datetime index.
        """
        # Handle relative paths
        if not os.path.isabs(file_path):
            file_path = os.path.join(project_root, file_path)
            
        # Ensure file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found at: {file_path}")
            
        try:
            # Check if CSV has MultiIndex header (two header rows)
            # Read first few lines to detect header structure
            with open(file_path, 'r') as f:
                first_line = f.readline()
                second_line = f.readline()
                f.seek(0)
                
                # If second line starts with a date or number, it's single-level
                # If second line looks like column names, it's MultiIndex
                try:
                    # Try to parse first column as date
                    first_col = second_line.split(',')[0].strip()
                    pd.to_datetime(first_col)
                    has_multiindex = False
                except:
                    # If first column can't be parsed as date, check if it looks like a column name
                    first_col = second_line.split(',')[0].strip()
                    # If it's empty or looks like a date string, it's probably single-level
                    if not first_col or pd.to_datetime(first_col, errors='coerce') is not pd.NaT:
                        has_multiindex = False
                    else:
                        # Otherwise, assume MultiIndex (second row contains column names)
                        has_multiindex = True
            
            # Read CSV with appropriate header structure
            if has_multiindex:
                df = pd.read_csv(file_path, index_col=0, header=[0, 1])
                # Ensure MultiIndex names are set
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns.names = ['asset_class', 'column']
            else:
                df = pd.read_csv(file_path, index_col=0)
            
            # Handle datetime index
            if parse_dates:
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                
            logger.info(f"Loaded dataset from {file_path} with shape {df.shape}")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            if isinstance(df.columns, pd.MultiIndex):
                logger.info(f"MultiIndex columns: {df.columns.nlevels} levels, {len(df.columns)} total columns")
                logger.info(f"Asset classes: {df.columns.get_level_values(0).unique().tolist()}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset from {file_path}: {str(e)}")
            raise

if __name__ == "__main__":
    # Initialize the pipeline using the default constants defined at the top of the file
    pipeline = DataPipeline()
    
    # Run the pipeline and get the data
    df = pipeline.process_data()

    # Save the dataset to CSV using the default path defined at the top of the file
    saved_path = pipeline.save_dataset(df, DEFAULT_OUTPUT_PATH)
    print(f"\nDataset saved to: {saved_path}")

    # Print basic information about the resulting dataset
    print("\nDataset Information:")
    print(f"Date Range: {df.index.min()} to {df.index.max()}")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumns:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    print("\nLast few rows:")
    print(df.tail())
    print("\nDataset Info:")
    print(df.info())
    
    # Generate data quality log file
    try:
        log_path = pipeline.generate_data_log(df=df, csv_path=saved_path)
        print(f"\nData quality log generated: {log_path}")
    except Exception as e:
        logger.warning(f"Could not generate data log file: {str(e)}")
    
    # Generate visualizations for all asset classes
    if PLOTLY_AVAILABLE:
        try:
            print("\nGenerating visualizations...")
            viz_paths = pipeline.visualize_data(df)
            print(f"\nVisualizations created: {len(viz_paths)} file(s)")
            for path in viz_paths:
                print(f"  - {path}")
        except Exception as e:
            logger.warning(f"Could not generate visualizations: {str(e)}")
    else:
        print("\nNote: Plotly not available. Install with 'poetry add plotly' to generate visualizations.")
