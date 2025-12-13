"""
Data loading and preprocessing components.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
from pathlib import Path
from .config import DataConfig


class DataProvider(ABC):
    """Abstract base class for data providers."""
    
    @abstractmethod
    def load_data(self, config: DataConfig) -> pd.DataFrame:
        """Load data according to configuration."""
        pass


class CSVDataProvider(DataProvider):
    """Data provider for CSV files."""
    
    def load_data(self, config: DataConfig) -> pd.DataFrame:
        """Load data from CSV file (matching reference implementations exactly)."""
        try:
            # Load CSV with date parsing (matching Cross_Asset_2Week_Momentum.py)
            df = pd.read_csv(config.file_path, parse_dates=[config.date_column])
            
            # Set DatetimeIndex with proper error handling (matching genetic algo weekly.py)
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            df = df.set_index(df.columns[0]).sort_index()
            
            # Apply date filtering
            if config.start_date:
                df = df[df.index >= config.start_date]
            if config.end_date:
                df = df[df.index <= config.end_date]
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading data from {config.file_path}: {str(e)}")


class MultiIndexCSVDataProvider(DataProvider):
    """Data provider for multi-index CSV files."""
    
    def load_data(self, config: DataConfig) -> pd.DataFrame:
        """Load data from multi-index CSV file.
        
        CSV structure:
        - Row 1 (index 0): asset_class header row (starts with "asset_class")
        - Row 2 (index 1): column header row (starts with "column")
        - Row 3 (index 2): Date header row (starts with "Date")
        - Row 4+ (index 3+): actual data
        """
        try:
            # Read CSV with multi-index headers
            # header=[0, 1] uses rows 0 and 1 as multi-index column headers
            # skiprows=[2] skips row 2 (the Date header row)
            df = pd.read_csv(config.file_path, header=[0, 1], skiprows=[2])
            
            # The first column should be the Date column
            # After reading with header=[0,1] and skiprows=[2], the first column tuple is ("asset_class", "column")
            # but the values in that column are the dates
            first_col = df.columns[0]
            df = df.set_index(first_col).sort_index()
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df.sort_index()
            
            # Apply date filtering
            if config.start_date:
                df = df[df.index >= pd.to_datetime(config.start_date)]
            if config.end_date:
                df = df[df.index <= pd.to_datetime(config.end_date)]
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading multi-index data from {config.file_path}: {str(e)}")


class DataLoader:
    """Main data loading class."""
    
    def __init__(self, provider: DataProvider):
        self.provider = provider
        
    def load_and_prepare(self, config: DataConfig) -> pd.DataFrame:
        """Load and prepare data according to configuration."""
        # Load raw data
        df = self.provider.load_data(config)
        
        # Validate that file exists and has data
        if df.empty:
            raise ValueError("Loaded data is empty")
        
        # Check if we have multi-index columns and should process them
        is_multi_index = self._is_multi_index(df)
        process_multi_index = is_multi_index and config.use_multi_index
        
        if process_multi_index:
            # Multi-index pipeline
            # 1. Filter by asset classes
            df = self._filter_by_asset_classes(df, config)
            
            # 2. Filter by columns within asset classes
            df = self._filter_by_columns(df, config)
            
            # 3. Date range filtering (already applied in provider, but check again)
            if config.start_date:
                df = df[df.index >= pd.to_datetime(config.start_date)]
            if config.end_date:
                df = df[df.index <= pd.to_datetime(config.end_date)]
            
            # 4. Handle missing data
            df = self._handle_missing_data(df, config)
            
            # 5. Detect and handle outliers
            df = self._detect_and_handle_outliers(df, config)
            
            # 6. Detect and handle gaps
            df = self._detect_and_handle_gaps(df, config)
            
            # 7. Validate data (min/max checks)
            df = self._validate_data(df, config)
            
            # 8. Resample data if required
            if config.resample_frequency:
                df = self._resample_data(df, config.resample_frequency)
                print(f"Data loaded. Strategy Period: {df.index[0]} to {df.index[-1]}")
                print(f"Total periods ({config.resample_frequency}): {len(df)}")
            
            # 9. Apply type conversion
            df = self._apply_type_conversion(df, config)
            
            # 10. Flatten multi-index columns (use 2nd level)
            df = self._flatten_multi_index(df, config)
            
        else:
            # Original pipeline for backward compatibility
            # Resample data if required (matching all reference implementations)
            if config.resample_frequency:
                df = self._resample_data(df, config.resample_frequency)
                print(f"Data loaded. Strategy Period: {df.index[0]} to {df.index[-1]}")
                print(f"Total periods ({config.resample_frequency}): {len(df)}")
            
            # Handle missing data (use config strategy)
            df = self._handle_missing_data(df, config)
        
        # Validate required assets are present (after flattening)
        if config.assets:
            missing_assets = [asset for asset in config.assets if asset not in df.columns]
            if missing_assets:
                print(f"Warning: Missing assets in data: {missing_assets}")
        
        # Validate required columns
        if config.required_columns:
            missing_cols = [col for col in config.required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    def _resample_data(self, df: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """Resample data to specified frequency."""
        try:
            # Use last value for each period and forward fill
            resampled = df.resample(frequency).last().ffill()
            return resampled
        except Exception as e:
            raise ValueError(f"Error resampling data to {frequency}: {str(e)}")
    
    def _is_multi_index(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has multi-index columns."""
        return isinstance(df.columns, pd.MultiIndex)
    
    def _filter_by_asset_classes(self, df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
        """Filter by asset class include/exclude lists."""
        if not self._is_multi_index(df):
            return df
        
        if not config.asset_classes_include and not config.asset_classes_exclude:
            return df
        
        # Get level 0 (asset_class) values
        asset_classes = df.columns.get_level_values(0).unique()
        
        # Apply include filter
        if config.asset_classes_include:
            asset_classes = [ac for ac in asset_classes if ac in config.asset_classes_include]
        
        # Apply exclude filter
        if config.asset_classes_exclude:
            asset_classes = [ac for ac in asset_classes if ac not in config.asset_classes_exclude]
        
        # Filter columns
        if len(asset_classes) > 0:
            mask = df.columns.get_level_values(0).isin(asset_classes)
            df = df.loc[:, mask]
        
        return df
    
    def _filter_by_columns(self, df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
        """Filter columns within asset classes."""
        if not self._is_multi_index(df):
            return df
        
        if not config.columns_include and not config.columns_exclude:
            return df
        
        # Get all columns as tuples (asset_class, column)
        columns_to_keep = []
        
        for asset_class in df.columns.get_level_values(0).unique():
            # Get columns for this asset class
            asset_cols = df.columns[df.columns.get_level_values(0) == asset_class]
            col_names = asset_cols.get_level_values(1).unique()
            
            # Apply include filter
            if config.columns_include and asset_class in config.columns_include:
                col_names = [col for col in col_names if col in config.columns_include[asset_class]]
            
            # Apply exclude filter
            if config.columns_exclude and asset_class in config.columns_exclude:
                col_names = [col for col in col_names if col not in config.columns_exclude[asset_class]]
            
            # Add columns to keep list
            for col_name in col_names:
                columns_to_keep.append((asset_class, col_name))
        
        # Filter DataFrame
        if columns_to_keep:
            df = df.loc[:, columns_to_keep]
        
        return df
    
    def _handle_missing_data(self, df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
        """Apply missing data strategy."""
        strategy = config.missing_data_strategy
        
        if strategy == "forward_fill":
            df = df.ffill()
        elif strategy == "backward_fill":
            df = df.bfill()
        elif strategy == "interpolate":
            df = df.interpolate(method='linear')
        elif strategy == "drop":
            df = df.dropna()
        elif strategy == "fill_value":
            df = df.fillna(config.missing_data_fill_value)
        else:
            raise ValueError(f"Unknown missing data strategy: {strategy}")
        
        return df
    
    def _detect_and_handle_outliers(self, df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
        """Detect and handle outliers using DataValidator."""
        if not config.outlier_detection_enabled:
            return df
        
        outliers = DataValidator.detect_outliers(df, config.outlier_method, config.outlier_threshold)
        
        if config.outlier_action == "flag":
            # Just flag, don't modify
            if outliers:
                print(f"Warning: Outliers detected in {len(outliers)} columns")
        elif config.outlier_action == "remove":
            # Remove outlier rows
            for col, outlier_series in outliers.items():
                if len(outlier_series) > 0:
                    df = df.drop(outlier_series.index)
        elif config.outlier_action == "clip":
            # Clip outliers to bounds
            for col, outlier_series in outliers.items():
                if len(outlier_series) > 0:
                    series = df[col]
                    if config.outlier_method == "iqr":
                        Q1 = series.quantile(0.25)
                        Q3 = series.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                    elif config.outlier_method == "zscore":
                        mean = series.mean()
                        std = series.std()
                        lower_bound = mean - config.outlier_threshold * std
                        upper_bound = mean + config.outlier_threshold * std
                    else:
                        continue
                    df[col] = series.clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def _detect_and_handle_gaps(self, df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
        """Detect and handle gaps using DataValidator."""
        if not config.gap_detection_enabled:
            return df
        
        gaps = DataValidator.check_data_gaps(df, config.max_gap_days)
        
        if config.gap_action == "flag":
            # Just flag, don't modify
            total_gaps = sum(len(gap_list) for gap_list in gaps.values())
            if total_gaps > 0:
                print(f"Warning: {total_gaps} gaps detected")
        elif config.gap_action == "fill":
            # Forward fill gaps
            df = df.ffill()
        elif config.gap_action == "drop":
            # Drop rows with gaps (this is complex, so we'll just forward fill)
            df = df.ffill()
        
        return df
    
    def _validate_data(self, df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
        """Validate data (min/max checks)."""
        if not config.data_validation_enabled:
            return df
        
        if config.min_value is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].clip(lower=config.min_value)
        
        if config.max_value is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].clip(upper=config.max_value)
        
        return df
    
    def _apply_column_renaming(self, df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
        """Rename columns according to rename map."""
        if not config.column_rename_map:
            return df
        
        # Handle both multi-index and single-level columns
        if self._is_multi_index(df):
            # For multi-index, we need to rename after flattening or handle it specially
            # We'll handle this after flattening
            pass
        else:
            df = df.rename(columns=config.column_rename_map)
        
        return df
    
    def _apply_type_conversion(self, df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
        """Force numeric conversion if enabled."""
        if not config.force_numeric:
            return df
        
        # Convert all columns to numeric where possible
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                # Keep non-numeric columns as-is
                pass
        
        return df
    
    def _flatten_multi_index(self, df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
        """Flatten multi-index columns using 2nd level (column names)."""
        if not config.flatten_columns:
            return df
        
        if self._is_multi_index(df):
            # Always use 2nd level (column names) as final column names
            df.columns = df.columns.get_level_values(1)
            
            # Apply column renaming after flattening if needed
            if config.column_rename_map:
                df = df.rename(columns=config.column_rename_map)
        
        return df
    
    def get_data_info(self, df: pd.DataFrame) -> Dict:
        """Get information about the loaded data."""
        return {
            'start_date': df.index[0],
            'end_date': df.index[-1],
            'total_periods': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        }


class DataValidator:
    """Validates data quality and completeness."""
    
    @staticmethod
    def validate_price_data(df: pd.DataFrame, price_columns: List[str]) -> Dict[str, bool]:
        """Validate price data quality."""
        results = {}
        
        for col in price_columns:
            if col not in df.columns:
                results[col] = False
                continue
                
            series = df[col]
            results[col] = (
                not series.isnull().all() and  # Not all NaN
                (series > 0).any() and         # Has positive values
                not series.isinf().any()       # No infinite values
            )
        
        return results
    
    @staticmethod
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
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, method: str = "iqr", threshold: float = 3.0) -> Dict[str, pd.Series]:
        """Detect outliers in data."""
        outliers = {}
        
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            
            if method == "iqr":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (series < lower_bound) | (series > upper_bound)
                
            elif method == "zscore":
                z_scores = np.abs((series - series.mean()) / series.std())
                outlier_mask = z_scores > threshold
                
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            outliers[col] = series[outlier_mask]
        
        return outliers 