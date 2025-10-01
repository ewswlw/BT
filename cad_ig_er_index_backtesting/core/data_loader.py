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
        
        # Resample data if required (matching all reference implementations)
        if config.resample_frequency:
            df = self._resample_data(df, config.resample_frequency)
            print(f"Data loaded. Strategy Period: {df.index[0]} to {df.index[-1]}")
            print(f"Total periods ({config.resample_frequency}): {len(df)}")
        
        # Validate required assets are present
        if config.assets:
            missing_assets = [asset for asset in config.assets if asset not in df.columns]
            if missing_assets:
                print(f"Warning: Missing assets in data: {missing_assets}")
        
        # Validate required columns
        if config.required_columns:
            missing_cols = [col for col in config.required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Forward fill missing values
        df = df.ffill()
        
        return df
    
    def _resample_data(self, df: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """Resample data to specified frequency."""
        try:
            # Use last value for each period and forward fill
            resampled = df.resample(frequency).last().ffill()
            return resampled
        except Exception as e:
            raise ValueError(f"Error resampling data to {frequency}: {str(e)}")
    
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