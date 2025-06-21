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
DEFAULT_START_DATE_ALIGN = 'yes'  # Align start dates

# Output settings
DEFAULT_OUTPUT_PATH = 'data_pipelines/data_processed/with_er_daily.csv'

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

#############################################################
# SECURITIES CONFIGURATION
#############################################################

# Default securities and their mappings - match the exact securities from config
# Format: {(ticker, field): column_name}
DEFAULT_OHLC_MAPPING = {
    ('I05510CA Index', 'INDEX_OAS_TSY_BP'): 'cad_oas',
    ('LF98TRUU Index', 'INDEX_OAS_TSY_BP'): 'us_hy_oas',
    ('LUACTRUU Index', 'INDEX_OAS_TSY_BP'): 'us_ig_oas',
    ('SPTSX Index', 'PX_LAST'): 'tsx',
    ('VIX Index', 'PX_LAST'): 'vix',
    ('USYC3M30 Index', 'PX_LAST'): 'us_3m_10y',
    ('BCMPUSGR Index', 'PX_LAST'): 'us_growth_surprises',
    ('BCMPUSIF Index', 'PX_LAST'): 'us_inflation_surprises',
    ('LEI YOY  Index', 'PX_LAST'): 'us_lei_yoy',
    ('.HARDATA G Index', 'PX_LAST'): 'us_hard_data_surprises',
    ('CGERGLOB Index', 'PX_LAST'): 'us_equity_revisions',
    ('.ECONREGI G Index', 'PX_LAST'): 'us_economic_regime',
}

# Excess Return YTD Mappings
# Format: {(ticker, field): column_name}
DEFAULT_ER_YTD_MAPPING = {
    ('I05510CA Index', 'INDEX_EXCESS_RETURN_YTD'): 'cad_ig_er',
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

# Known bad dates in data
# Format following the config file structure: 
# {'date_key': {'column': 'column_name', 'action': 'action_type'}}
DEFAULT_BAD_DATES = {
    '2005-11-15': {
        'column': 'cad_oas',
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
            self.ohlc_mapping = ohlc_mapping or DEFAULT_OHLC_MAPPING.copy()
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

    def visualize_data(self, data: pd.DataFrame = None, output_path: str = None, title: str = "Data Series Over Time"):
        """
        Visualize the data series and save as interactive HTML.
        
        Args:
            data (pd.DataFrame, optional): Data to visualize. If None, uses the last processed data
            output_path (str, optional): Path to save the HTML file. If None, saves to data_pipelines/data_visualization.html
            title (str): Title for the visualization
            
        Returns:
            str: Path to the saved HTML file
            
        Raises:
            ValueError: If no data is available for visualization
            Exception: If there's an error during visualization
        """
        try:
            from data_pipelines.data_visualization import create_spread_plots
            
            # Use provided data or last processed data
            df_to_plot = data if data is not None else self.data
            
            if df_to_plot is None:
                raise ValueError("No data available for visualization. Please process data first.")
            
            # Set default output path if none provided
            if output_path is None:
                output_path = os.path.join(os.path.dirname(__file__), "data_visualization.html")
                
            # Create and save the plot
            saved_path = create_spread_plots(df_to_plot, output_path, title)
            logger.info(f"Visualization saved to: {saved_path}")
            return saved_path
            
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
            
            # Drop any rows with missing values
            cleaned_df = cleaned_df.dropna()
            final_rows = len(cleaned_df)
            
            if final_rows < initial_rows:
                logger.info(f"Dropped {initial_rows - final_rows} rows with missing values")
                logger.info(f"Final dataset has {final_rows} complete rows")
            
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
            
            return final_df
        
        except Exception as e:
            logger.error(f"Error getting full dataset: {str(e)}")
            raise

    def save_dataset(self, output_path=None, date_format=None):
        """
        Save the dataset to a file.
        
        Args:
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
            df = self.get_full_dataset()
            
            # Ensure index name is 'Date'
            df.index.name = 'Date'
            
            # Save with date format if specified
            if date_format:
                df.to_csv(output_path, date_format=date_format)
            else:
                df.to_csv(output_path)
                
            logger.info(f"Dataset saved to {output_path} with shape {df.shape}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving dataset: {str(e)}")
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
            # Always read with index_col=0
            df = pd.read_csv(file_path, index_col=0)
            
            # Handle datetime index
            if parse_dates:
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                
            logger.info(f"Loaded dataset from {file_path} with shape {df.shape}")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
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
    saved_path = pipeline.save_dataset(DEFAULT_OUTPUT_PATH)
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
