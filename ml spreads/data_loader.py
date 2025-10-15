"""
Data loading and preprocessing for CAD OAS prediction model.
Handles lag enforcement and walk-forward validation splits.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Generator
import warnings
# Config will be imported in __init__ to avoid circular imports

class DataLoader:
    """Handles data loading with proper lag enforcement and validation splits."""
    
    def __init__(self, config_obj=None):
        """Initialize DataLoader with configuration."""
        # Import config here to avoid circular imports
        from config import config
        self.config = config_obj or config
        self.data = None
        self.features = None
        self.target = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the raw data."""
        print("Loading data from:", self.config.data_path)
        
        # Load data
        self.data = pd.read_csv(self.config.data_path)
        
        # Convert Date column to datetime and set as index
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.set_index('Date').sort_index()
        
        print(f"Loaded {len(self.data)} observations from {self.data.index.min()} to {self.data.index.max()}")
        print(f"Columns: {list(self.data.columns)}")
        
        # Check for missing values
        missing_data = self.data.isnull().sum()
        if missing_data.sum() > 0:
            print("\nMissing values detected:")
            print(missing_data[missing_data > 0])
            # Forward fill missing values (conservative approach)
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')
            print("Missing values filled using forward/backward fill")
        
        return self.data
    
    def create_lagged_features(self, data: pd.DataFrame, feature_cols: List[str], 
                              lags: List[int]) -> pd.DataFrame:
        """
        Create lagged features ensuring minimum lag constraint.
        
        Args:
            data: DataFrame with time series data
            feature_cols: List of column names to lag
            lags: List of lag periods (must be >= min_lag)
        
        Returns:
            DataFrame with lagged features
        """
        # Validate lag constraints
        valid_lags = [lag for lag in lags if lag >= self.config.min_lag and lag <= self.config.max_lag]
        if len(valid_lags) != len(lags):
            removed_lags = [lag for lag in lags if lag < self.config.min_lag or lag > self.config.max_lag]
            print(f"Warning: Removed lags {removed_lags} outside allowed range [{self.config.min_lag}, {self.config.max_lag}]")
        
        lagged_df = pd.DataFrame(index=data.index)
        
        for col in feature_cols:
            if col not in data.columns:
                print(f"Warning: Column {col} not found in data")
                continue
                
            for lag in valid_lags:
                lagged_df[f'{col}_lag{lag}'] = data[col].shift(lag)
        
        print(f"Created {len(lagged_df.columns)} lagged features from {len(feature_cols)} original columns")
        return lagged_df
    
    def create_baseline_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create baseline feature set with proper lagging."""
        print("Creating baseline features...")
        
        features_df = pd.DataFrame(index=data.index)
        
        # 1. Raw lagged values for all feature columns
        lagged_features = self.create_lagged_features(
            data, self.config.feature_columns, self.config.baseline_lags
        )
        features_df = pd.concat([features_df, lagged_features], axis=1)
        
        # 2. Returns over multiple horizons (with proper lagging)
        for col in self.config.feature_columns:
            if col not in data.columns:
                continue
            
            for period in self.config.momentum_periods:
                # Calculate returns, then lag them to ensure no look-ahead bias
                returns = data[col].pct_change(period)
                # Lag the returns by min_lag to ensure prediction is based on past data
                lagged_returns = returns.shift(self.config.min_lag)
                features_df[f'{col}_returns_{period}d_lag{self.config.min_lag}'] = lagged_returns
        
        # 3. Moving averages (with proper lagging)
        for col in self.config.feature_columns:
            if col not in data.columns:
                continue
                
            for period in self.config.ma_periods:
                ma = data[col].rolling(period).mean()
                # Lag the moving average by min_lag
                lagged_ma = ma.shift(self.config.min_lag)
                features_df[f'{col}_ma_{period}d_lag{self.config.min_lag}'] = lagged_ma
                
                # Price relative to moving average
                price_ma_ratio = data[col] / ma
                lagged_ratio = price_ma_ratio.shift(self.config.min_lag)
                features_df[f'{col}_ma_ratio_{period}d_lag{self.config.min_lag}'] = lagged_ratio
        
        # 4. Rolling volatility (with proper lagging)
        for col in self.config.feature_columns:
            if col not in data.columns:
                continue
                
            # Calculate daily returns first
            daily_returns = data[col].pct_change()
            
            for period in self.config.volatility_periods:
                vol = daily_returns.rolling(period).std()
                # Lag the volatility by min_lag
                lagged_vol = vol.shift(self.config.min_lag)
                features_df[f'{col}_vol_{period}d_lag{self.config.min_lag}'] = lagged_vol
        
        # Remove any remaining NaN and infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        print(f"Created {len(features_df.columns)} baseline features")
        return features_df
    
    def prepare_target(self, data: pd.DataFrame) -> pd.Series:
        """Prepare target variable (cad_oas at time t)."""
        if self.config.target_column not in data.columns:
            raise ValueError(f"Target column {self.config.target_column} not found in data")
        
        target = data[self.config.target_column].copy()
        print(f"Target variable: {self.config.target_column}")
        print(f"Target statistics: mean={target.mean():.2f}, std={target.std():.2f}")
        
        return target
    
    def create_walk_forward_splits(self, features: pd.DataFrame, target: pd.Series) -> Generator:
        """
        Create walk-forward validation splits with rolling windows.
        
        Yields:
            Tuple of (train_features, train_target, test_features, test_target, split_info)
        """
        total_samples = len(features)
        window_size = self.config.window_size
        test_size = self.config.test_size
        step_size = self.config.step_size
        
        print(f"\nCreating walk-forward splits:")
        print(f"  Total samples: {total_samples}")
        print(f"  Window size: {window_size} (training)")
        print(f"  Test size: {test_size}")
        print(f"  Step size: {step_size}")
        
        # Calculate number of splits
        max_start = total_samples - window_size - test_size
        num_splits = max(1, (max_start // step_size) + 1)
        print(f"  Number of splits: {num_splits}")
        
        for i in range(num_splits):
            start_idx = i * step_size
            train_end_idx = start_idx + window_size
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + test_size
            
            # Ensure we don't exceed data bounds
            if test_end_idx > total_samples:
                break
            
            # Extract splits
            train_features = features.iloc[start_idx:train_end_idx].copy()
            train_target = target.iloc[start_idx:train_end_idx].copy()
            test_features = features.iloc[test_start_idx:test_end_idx].copy()
            test_target = target.iloc[test_start_idx:test_end_idx].copy()
            
            # Get date ranges for this split
            train_dates = (train_features.index.min(), train_features.index.max())
            test_dates = (test_features.index.min(), test_features.index.max())
            
            split_info = {
                'split_number': i + 1,
                'train_dates': train_dates,
                'test_dates': test_dates,
                'train_size': len(train_features),
                'test_size': len(test_features)
            }
            
            yield train_features, train_target, test_features, test_target, split_info
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names."""
        if self.features is None:
            raise ValueError("Features not created yet. Call create_baseline_features() first.")
        return list(self.features.columns)
    
    def save_data_summary(self):
        """Save data summary to outputs directory."""
        if self.data is None:
            raise ValueError("No data loaded")
        
        summary = {
            'total_observations': len(self.data),
            'date_range': (self.data.index.min().strftime('%Y-%m-%d'), 
                          self.data.index.max().strftime('%Y-%m-%d')),
            'columns': list(self.data.columns),
            'target_column': self.config.target_column,
            'feature_columns': self.config.feature_columns,
            'missing_values': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.to_dict()
        }
        
        # Save as JSON
        import json
        summary_path = f"{self.config.output_dir}/reports/data_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Data summary saved to: {summary_path}")
        return summary
