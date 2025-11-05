"""
Base strategy class for all trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import core modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.backtest_engine import BacktestEngine, BacktestResult, PortfolioManager


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.description = config.get('description', '')
        self.holding_period_days = config.get('holding_period_days', 7)
        self.portfolio_manager = PortfolioManager()
        
    @abstractmethod
    def generate_signals(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                        train_features: pd.DataFrame, test_features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate entry and exit signals based on data and features.
        
        Args:
            train_data: Training price data DataFrame
            test_data: Test price data DataFrame
            train_features: Training features DataFrame
            test_features: Test features DataFrame
            
        Returns:
            Tuple of (entry_signals, exit_signals) as boolean Series for test_data
        """
        pass
    
    @abstractmethod
    def get_required_features(self) -> List[str]:
        """
        Return list of required feature names for this strategy.
        
        Returns:
            List of required feature column names
        """
        pass
    
    def validate_data(self, data: pd.DataFrame, features: pd.DataFrame) -> bool:
        """
        Validate that data contains required features and is properly formatted.
        
        Args:
            data: Price data DataFrame
            features: Features DataFrame
            
        Returns:
            True if data is valid
            
        Raises:
            ValueError: If required features are missing or data is invalid
        """
        # Check if data is empty
        if data.empty or features.empty:
            raise ValueError("Data or features DataFrame is empty")
        
        # Check if indices align
        if not data.index.equals(features.index):
            print("Warning: Data and features indices don't match exactly. Aligning...")
        
        # Check required features - look in both data and features
        required_features = self.get_required_features()
        if required_features:  # Only check if strategy specifies required features
            # Check in features first, then in data (for strategies that need price columns directly)
            missing_features = []
            for feature in required_features:
                if feature not in features.columns and feature not in data.columns:
                    missing_features.append(feature)
            
            if missing_features:
                raise ValueError(f"Missing required features/columns: {missing_features}")
        
        # Check for sufficient data
        if len(data) < 50:  # Minimum 50 periods
            raise ValueError(f"Insufficient data: {len(data)} periods. Need at least 50.")
        
        return True
    
    def apply_holding_period(self, raw_signals: pd.Series, holding_days: Optional[int] = None) -> pd.Series:
        """
        Apply minimum holding period constraint to signals.
        
        Args:
            raw_signals: Raw binary signals (1=enter, 0=no position)
            holding_days: Minimum days to hold position (uses self.holding_period_days if None)
            
        Returns:
            Adjusted signals with holding period enforced
        """
        holding_days = holding_days or self.holding_period_days
        return self.portfolio_manager.apply_holding_period(raw_signals, holding_days)
    
    def validate_signals(self, entry_signals: pd.Series, exit_signals: pd.Series) -> bool:
        """
        Validate that signals ensure binary positioning.
        
        Args:
            entry_signals: Entry signals
            exit_signals: Exit signals
            
        Returns:
            True if signals are valid
        """
        # Check that signals are boolean
        if entry_signals.dtype != bool:
            entry_signals = entry_signals.astype(bool)
        if exit_signals.dtype != bool:
            exit_signals = exit_signals.astype(bool)
        
        # Check that entry and exit signals don't overlap (will be handled by holding period)
        # This is just a basic validation
        
        return True
    
    def backtest(self, 
                 data: pd.DataFrame, 
                 features: pd.DataFrame,
                 price_column: str,
                 backtest_engine: BacktestEngine) -> BacktestResult:
        """
        Run complete backtest for this strategy.
        
        Args:
            data: Price data DataFrame
            features: Features DataFrame  
            price_column: Column name for price data
            backtest_engine: Backtesting engine
            
        Returns:
            BacktestResult object containing all results
        """
        # Remove duplicate indices
        data = data[~data.index.duplicated(keep='first')]
        features = features[~features.index.duplicated(keep='first')]
        
        # Align data and features
        common_index = data.index.intersection(features.index)
        data = data.loc[common_index]
        features = features.loc[common_index]
        
        # Validate data
        self.validate_data(data, features)
        
        # For backtesting, we need to split data
        # Simple approach: use all data as both train and test for full backtest
        entry_signals, exit_signals = self.generate_signals(
            data, data, features, features
        )
        
        # Validate signals
        self.validate_signals(entry_signals, exit_signals)
        
        # Run backtest
        price_series = data[price_column]
        result = backtest_engine.run_backtest(
            price_series, entry_signals, exit_signals, apply_holding_period=True
        )
        
        # Update result with strategy info
        result.strategy_name = self.name
        
        return result
    
    def get_strategy_info(self) -> Dict:
        """Get information about this strategy."""
        return {
            'name': self.name,
            'description': self.description,
            'class': self.__class__.__name__,
            'config': self.config,
            'required_features': self.get_required_features(),
            'holding_period_days': self.holding_period_days
        }
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        return self.__str__()

