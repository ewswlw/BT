"""
Market Data Selection and Filtering Script

This script loads market data from a CSV file, applies filtering and preprocessing
based on configuration, and saves a filtered dataset to a new CSV file.

The configuration dictionary at the top of this file allows you to:
- Filter by asset classes (include/exclude)
- Filter columns within asset classes (include/exclude)
- Filter by date range
- Handle missing data
- Detect and handle outliers
- Detect and handle gaps
- Validate data ranges
- Resample data frequency
- Rename columns
- Flatten multi-index columns to single level

Usage:
    python selected_data.py
"""

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from io import StringIO

# ============================================================================
# CONFIGURATION DICTIONARY
# ============================================================================
# Modify the values below to customize data selection and filtering

CONFIG = {
    # Input/Output paths
    "input_file": "processed market data/market_data_time_series.csv",
    "output_file": "processed market data/cad_ig_er_index_data_for_backtest.csv",
    
    # Multi-index CSV support
    "use_multi_index": True,  # Enable multi-index CSV loading
    "multi_index_header_rows": 2,
    "date_header_row": 2,
    
    # Asset class filtering
    "asset_classes_include": [
        "cad_credit_spreads",
        "cad_credit_exposure",
        "equity_indices",
        "volatility",
        "yield_curves",
        "economic_indicators",
        "forecasts"
    ],
    "asset_classes_exclude": None,  # null = exclude none
    
    # Column filtering (optional, scoped by asset class)
    "columns_include": {
        "cad_credit_spreads": ["cad_oas"],
        "cad_credit_exposure": ["cad_ig_er_index"],
        "equity_indices": ["tsx", "s&p_500"],
        "volatility": ["vix"],
        "yield_curves": ["us_3m_10y"],
        "economic_indicators": ["us_growth_surprises", "us_inflation_surprises", "us_lei_yoy", "us_hard_data_surprises", "us_equity_revisions", "us_economic_regime"],
        "forecasts": ["spx_1bf_eps", "spx_1bf_sales", "tsx_1bf_eps", "tsx_1bf_sales"]
    },
    "columns_exclude": None,  # null = exclude none
    
    # Date range filtering
    "start_date": None,  # null = no start limit, or "2002-11-01"
    "end_date": None,  # null = no end limit
    
    # Data resampling
    "resample_frequency": None,  # null = use raw daily data, or specify frequency like "W-FRI" (weekly Friday), "M" (monthly), "Q" (quarterly), etc.
    "resample_method": "last",  # Options: last, first, mean, sum - method to use when resampling
    
    # Start date alignment (align all columns to start on same date with no NAs)
    "align_start_date": True,  # True = trim to first date where all columns have data, False = keep original start date
    
    # Missing data handling
    "missing_data_strategy": "forward_fill",  # Options: forward_fill, backward_fill, interpolate, drop, fill_value
    "missing_data_fill_value": 0.0,
    
    # Outlier detection
    "outlier_detection_enabled": False,
    "outlier_method": "iqr",  # Options: iqr, zscore
    "outlier_threshold": 3.0,
    "outlier_action": "flag",  # Options: flag, remove, clip
    
    # Gap detection
    "gap_detection_enabled": False,
    "max_gap_days": 7,
    "gap_action": "flag",  # Options: flag, fill, drop
    
    # Data validation
    "data_validation_enabled": True,
    "min_value": None,
    "max_value": None,
    
    # Column transformations
    "column_rename_map": None,  # null = no renaming, or {"old_name": "new_name"}
    "force_numeric": True,
    
    # Multi-index flattening
    "flatten_columns": True,  # Always use 2nd level (column names)
}


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def is_multi_index(df: pd.DataFrame) -> bool:
    """Check if DataFrame has multi-index columns."""
    return isinstance(df.columns, pd.MultiIndex)


def load_multi_index_csv(file_path: str) -> pd.DataFrame:
    """
    Load CSV file with multi-index headers.
    
    CSV structure:
    - Row 1 (index 0): asset_class header row (starts with "asset_class")
    - Row 2 (index 1): column header row (starts with "column")
    - Row 3 (index 2): Date header row (starts with "Date")
    - Row 4+ (index 3+): actual data
    """
    # Read CSV with multi-index headers
    # header=[0, 1] uses rows 0 and 1 as multi-index column headers
    # skiprows=[2] skips row 2 (the Date header row)
    df = pd.read_csv(file_path, header=[0, 1], skiprows=[2])
    
    # The first column should be the Date column
    first_col = df.columns[0]
    df = df.set_index(first_col).sort_index()
    
    # Convert index to datetime
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    
    return df


def filter_by_asset_classes(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Filter by asset class include/exclude lists."""
    if not is_multi_index(df):
        return df
    
    if not config.get("asset_classes_include") and not config.get("asset_classes_exclude"):
        return df
    
    # Get level 0 (asset_class) values
    asset_classes = df.columns.get_level_values(0).unique()
    
    # Apply include filter
    if config.get("asset_classes_include"):
        asset_classes = [ac for ac in asset_classes if ac in config["asset_classes_include"]]
    
    # Apply exclude filter
    if config.get("asset_classes_exclude"):
        asset_classes = [ac for ac in asset_classes if ac not in config["asset_classes_exclude"]]
    
    # Filter columns
    if len(asset_classes) > 0:
        mask = df.columns.get_level_values(0).isin(asset_classes)
        df = df.loc[:, mask]
    
    return df


def filter_by_columns(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Filter columns within asset classes."""
    if not is_multi_index(df):
        return df
    
    if not config.get("columns_include") and not config.get("columns_exclude"):
        return df
    
    # Get all columns as tuples (asset_class, column)
    columns_to_keep = []
    
    for asset_class in df.columns.get_level_values(0).unique():
        # Get columns for this asset class
        asset_cols = df.columns[df.columns.get_level_values(0) == asset_class]
        col_names = asset_cols.get_level_values(1).unique()
        
        # Apply include filter
        if config.get("columns_include") and asset_class in config["columns_include"]:
            col_names = [col for col in col_names if col in config["columns_include"][asset_class]]
        
        # Apply exclude filter
        if config.get("columns_exclude") and asset_class in config["columns_exclude"]:
            col_names = [col for col in col_names if col not in config["columns_exclude"][asset_class]]
        
        # Add columns to keep list
        for col_name in col_names:
            columns_to_keep.append((asset_class, col_name))
    
    # Filter DataFrame
    if columns_to_keep:
        df = df.loc[:, columns_to_keep]
    
    return df


def handle_missing_data(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Apply missing data strategy."""
    strategy = config.get("missing_data_strategy", "forward_fill")
    
    if strategy == "forward_fill":
        df = df.ffill()
    elif strategy == "backward_fill":
        df = df.bfill()
    elif strategy == "interpolate":
        df = df.interpolate(method='linear')
    elif strategy == "drop":
        df = df.dropna()
    elif strategy == "fill_value":
        fill_value = config.get("missing_data_fill_value", 0.0)
        df = df.fillna(fill_value)
    else:
        raise ValueError(f"Unknown missing data strategy: {strategy}")
    
    return df


def detect_outliers(df: pd.DataFrame, method: str = "iqr", threshold: float = 3.0) -> Dict[str, pd.Series]:
    """Detect outliers in data."""
    outliers = {}
    
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col].dropna()
        
        if len(series) == 0:
            continue
        
        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            
        elif method == "zscore":
            mean = series.mean()
            std = series.std()
            if std == 0:
                continue
            z_scores = np.abs((series - mean) / std)
            outlier_mask = z_scores > threshold
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        outliers[col] = series[outlier_mask]
    
    return outliers


def detect_and_handle_outliers(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Detect and handle outliers."""
    if not config.get("outlier_detection_enabled", False):
        return df
    
    outliers = detect_outliers(
        df,
        config.get("outlier_method", "iqr"),
        config.get("outlier_threshold", 3.0)
    )
    
    action = config.get("outlier_action", "flag")
    
    if action == "flag":
        # Just flag, don't modify
        if outliers:
            total_outliers = sum(len(outlier_series) for outlier_series in outliers.values())
            print(f"Warning: Outliers detected - {total_outliers} total outliers across {len(outliers)} column(s)")
            for col, outlier_series in list(outliers.items())[:5]:  # Show first 5
                if len(outlier_series) > 0:
                    print(f"  {col}: {len(outlier_series)} outliers")
    elif action == "remove":
        # Remove outlier rows
        outlier_indices = set()
        for col, outlier_series in outliers.items():
            if len(outlier_series) > 0:
                outlier_indices.update(outlier_series.index)
        if outlier_indices:
            df = df.drop(index=list(outlier_indices))
            print(f"Removed {len(outlier_indices)} rows with outliers")
    elif action == "clip":
        # Clip outliers to bounds
        for col, outlier_series in outliers.items():
            if len(outlier_series) > 0:
                series = df[col]
                if config.get("outlier_method") == "iqr":
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        df[col] = series.clip(lower=lower_bound, upper=upper_bound)
                elif config.get("outlier_method") == "zscore":
                    mean = series.mean()
                    std = series.std()
                    if std > 0:
                        threshold = config.get("outlier_threshold", 3.0)
                        lower_bound = mean - threshold * std
                        upper_bound = mean + threshold * std
                        df[col] = series.clip(lower=lower_bound, upper=upper_bound)
        print(f"Clipped outliers in {len(outliers)} column(s)")
    
    return df


def check_data_gaps(df: pd.DataFrame, max_gap_days: int = 7) -> Dict[str, List]:
    """Check for large gaps in data."""
    gaps = {}
    
    for col in df.columns:
        series = df[col].dropna()
        if len(series) < 2:
            gaps[col] = []
            continue
        
        # Find gaps larger than threshold
        time_diffs = series.index.to_series().diff()
        large_gaps = time_diffs[time_diffs > pd.Timedelta(days=max_gap_days)]
        gaps[col] = large_gaps.index.tolist()
    
    return gaps


def detect_and_handle_gaps(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Detect and handle gaps."""
    if not config.get("gap_detection_enabled", False):
        return df
    
    gaps = check_data_gaps(df, config.get("max_gap_days", 7))
    
    action = config.get("gap_action", "flag")
    
    if action == "flag":
        # Just flag, don't modify
        total_gaps = sum(len(gap_list) for gap_list in gaps.values())
        if total_gaps > 0:
            print(f"Warning: {total_gaps} gaps detected")
    elif action == "fill":
        # Forward fill gaps
        df = df.ffill()
        print("Forward filled gaps")
    elif action == "drop":
        # Drop rows with gaps (complex, so we'll just forward fill)
        df = df.ffill()
        print("Forward filled gaps (drop action)")
    
    return df


def validate_data(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Validate data (min/max checks)."""
    if not config.get("data_validation_enabled", True):
        return df
    
    min_value = config.get("min_value")
    max_value = config.get("max_value")
    
    if min_value is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].clip(lower=min_value)
    
    if max_value is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].clip(upper=max_value)
    
    return df


def resample_data(df: pd.DataFrame, frequency: str, method: str = "last") -> pd.DataFrame:
    """
    Resample data to specified frequency.
    
    Args:
        df: DataFrame to resample
        frequency: Pandas frequency string (e.g., "W-FRI", "M", "Q")
        method: Resampling method - "last", "first", "mean", or "sum"
    
    Returns:
        Resampled DataFrame
    """
    try:
        if method == "last":
            resampled = df.resample(frequency).last()
        elif method == "first":
            resampled = df.resample(frequency).first()
        elif method == "mean":
            resampled = df.resample(frequency).mean()
        elif method == "sum":
            resampled = df.resample(frequency).sum()
        else:
            raise ValueError(f"Unknown resample method: {method}. Use 'last', 'first', 'mean', or 'sum'")
        
        # Forward fill any missing values after resampling
        resampled = resampled.ffill()
        return resampled
    except Exception as e:
        raise ValueError(f"Error resampling data to {frequency}: {str(e)}")


def align_start_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align all columns to start on the same date where all columns have non-null data.
    
    This ensures there are no NAs at the start of the dataset due to different start dates
    for different columns.
    
    Args:
        df: DataFrame to align
        
    Returns:
        DataFrame trimmed to start from first date where all columns have data
    """
    if df.empty:
        return df
    
    # Find the first date where all columns have non-null data
    # Check each row to see if all columns are non-null
    non_null_mask = df.notna().all(axis=1)
    
    if non_null_mask.any():
        # Find the first True value (first date with all non-null columns)
        first_valid_date = df.index[non_null_mask][0]
        df_aligned = df.loc[first_valid_date:].copy()
        
        print(f"Aligned start date: {first_valid_date} (original start: {df.index[0]})")
        print(f"Rows removed: {len(df) - len(df_aligned)}")
        
        return df_aligned
    else:
        print("Warning: No date found where all columns have data. Returning original DataFrame.")
        return df


def apply_type_conversion(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Force numeric conversion if enabled."""
    if not config.get("force_numeric", True):
        return df
    
    # Convert all columns to numeric where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except (ValueError, TypeError):
            # Keep non-numeric columns as-is
            pass
    
    return df


def flatten_multi_index(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Flatten multi-index columns using 2nd level (column names)."""
    if not config.get("flatten_columns", True):
        return df
    
    if is_multi_index(df):
        # Always use 2nd level (column names) as final column names
        df.columns = df.columns.get_level_values(1)
        
        # Apply column renaming after flattening if needed
        if config.get("column_rename_map"):
            df = df.rename(columns=config["column_rename_map"])
    
    return df


def generate_data_log(df: pd.DataFrame, csv_path: Path, log_file: Path = None) -> str:
    """
    Generate a comprehensive log file with data quality information.
    
    Creates a log file similar to market_data_stats.log with df.info(), shape, date range,
    missing data statistics, and sample rows (head/tail).
    
    Args:
        df: DataFrame to analyze
        csv_path: Path to the CSV file
        log_file: Path to log file. If None, uses default location in logs/logs/
        
    Returns:
        str: Path to the generated log file
    """
    try:
        # Set up log file path
        if log_file is None:
            # Get project root directory (parent of market data pipeline)
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            log_file = project_root / "logs" / "logs" / "cad_ig_er_index_backtest_data_stats.log"
        
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up logger for this specific log file
        log_logger = logging.getLogger('cad_ig_er_index_backtest_data_stats')
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
        log_logger.info(f"CAD IG ER INDEX BACKTEST DATA DIAGNOSTICS @ {timestamp}")
        log_logger.info("-" * 100)
        
        # Dataset information
        log_logger.info(f"Dataset: {csv_path.name}")
        log_logger.info(f"Path: {os.path.abspath(csv_path)}")
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
            log_logger.info(f"Columns: {', '.join(df.columns.tolist())}")
        
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
                for column, count in columns_with_missing.items():
                    pct = missing_pct[column]
                    log_logger.info(f"  {column}: {count:,} ({pct}%)")
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
        
        return str(log_file)
        
    except Exception as e:
        print(f"Warning: Error generating data log: {str(e)}")
        return None


def process_data(config: Dict) -> pd.DataFrame:
    """
    Main data processing pipeline.
    
    Applies all filtering and preprocessing steps in order:
    1. Load data (with multi-index support if enabled)
    2. Filter by asset classes
    3. Filter by columns within asset classes
    4. Filter by date range
    5. Handle missing data
    6. Detect and handle outliers
    7. Detect and handle gaps
    8. Validate data (min/max checks)
    9. Resample data if required
    10. Align start date (trim to first date where all columns have data)
    11. Apply type conversion
    12. Flatten multi-index columns
    """
    # Get script directory for relative paths
    script_dir = Path(__file__).parent
    input_path = script_dir / config["input_file"]
    output_path = script_dir / config["output_file"]
    
    # Validate input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Loading data from: {input_path}")
    
    # Load data
    if config.get("use_multi_index", True):
        df = load_multi_index_csv(str(input_path))
    else:
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.sort_index()
    
    print(f"Loaded data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    if is_multi_index(df):
        asset_classes = df.columns.get_level_values(0).unique()
        print(f"Asset classes: {list(asset_classes)}")
        print(f"Total columns: {len(df.columns)}")
    
    # Apply processing pipeline
    if config.get("use_multi_index", True) and is_multi_index(df):
        # 1. Filter by asset classes
        df = filter_by_asset_classes(df, config)
        print(f"After asset class filtering: {df.shape}")
        
        # 2. Filter by columns within asset classes
        df = filter_by_columns(df, config)
        print(f"After column filtering: {df.shape}")
    
    # 3. Filter by date range
    if config.get("start_date"):
        start = pd.to_datetime(config["start_date"])
        df = df[df.index >= start]
        print(f"After start date filter ({config['start_date']}): {df.shape}")
    
    if config.get("end_date"):
        end = pd.to_datetime(config["end_date"])
        df = df[df.index <= end]
        print(f"After end date filter ({config['end_date']}): {df.shape}")
    
    # 4. Handle missing data
    df = handle_missing_data(df, config)
    
    # 5. Detect and handle outliers
    df = detect_and_handle_outliers(df, config)
    
    # 6. Detect and handle gaps
    df = detect_and_handle_gaps(df, config)
    
    # 7. Validate data (min/max checks)
    df = validate_data(df, config)
    
    # 8. Resample data if required (before alignment to ensure consistent dates)
    if config.get("resample_frequency"):
        resample_method = config.get("resample_method", "last")
        df = resample_data(df, config["resample_frequency"], resample_method)
        print(f"After resampling to {config['resample_frequency']} (method: {resample_method}): {df.shape}")
    
    # 9. Align start date (trim to first date where all columns have data)
    if config.get("align_start_date", True):
        df = align_start_date(df)
        print(f"After start date alignment: {df.shape}")
    
    # 10. Apply type conversion
    df = apply_type_conversion(df, config)
    
    # 11. Flatten multi-index columns
    df = flatten_multi_index(df, config)
    
    print(f"\nFinal data shape: {df.shape}")
    print(f"Final date range: {df.index.min()} to {df.index.max()}")
    print(f"Final columns: {list(df.columns)}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.index.name = 'Date'
    df.to_csv(output_path)
    
    print(f"\nFiltered data saved to: {output_path}")
    
    # Generate data log file
    try:
        log_path = generate_data_log(df, output_path)
        if log_path:
            print(f"Data log generated: {log_path}")
    except Exception as e:
        print(f"Warning: Could not generate data log: {str(e)}")
    
    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Market Data Selection and Filtering Script")
    print("=" * 80)
    print()
    
    try:
        df = process_data(CONFIG)
        
        print()
        print("=" * 80)
        print("Processing completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print()
        print("=" * 80)
        print(f"ERROR: {str(e)}")
        print("=" * 80)
        raise

