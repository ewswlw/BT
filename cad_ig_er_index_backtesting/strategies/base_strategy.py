"""
Base strategy class for all trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np
from core.portfolio import PortfolioEngine, BacktestResult
from core.config import PortfolioConfig

# Import validation (optional)
try:
    from core.validation import ValidationFramework, ValidationConfig, ValidationResults
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    ValidationFramework = None
    ValidationConfig = None
    ValidationResults = None


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.description = config.get('description', '')
        self.validation_config = None
        self.validation_results = None
        
        # Try to get validation config from config dict
        if VALIDATION_AVAILABLE and 'validation' in config:
            try:
                from core.validation.validation_config import ValidationConfig
                self.validation_config = ValidationConfig(**config['validation'])
            except Exception:
                pass
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate entry and exit signals based on data and features.
        
        Args:
            data: Price data DataFrame
            features: Features DataFrame
            
        Returns:
            Tuple of (entry_signals, exit_signals) as boolean Series
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
    
    def backtest(self, 
                 data: pd.DataFrame, 
                 features: pd.DataFrame,
                 portfolio_config: PortfolioConfig,
                 asset_column: Optional[str] = None) -> BacktestResult:
        """
        Run complete backtest for this strategy.
        
        Args:
            data: Price data DataFrame
            features: Features DataFrame  
            portfolio_config: Portfolio configuration
            asset_column: Column to trade (if None, uses first column)
            
        Returns:
            BacktestResult object containing all results
        """
        # Validate data
        self.validate_data(data, features)
        
        # Generate signals
        entry_signals, exit_signals = self.generate_signals(data, features)
        
        # Run backtest using portfolio engine
        portfolio_engine = PortfolioEngine(portfolio_config)
        result = portfolio_engine.run_backtest(
            data, entry_signals, exit_signals, asset_column
        )
        
        # Update result with strategy info
        result.strategy_name = self.name
        
        return result
    
    def optimize_parameters(self, 
                           data: pd.DataFrame,
                           features: pd.DataFrame,
                           parameter_space: Dict,
                           portfolio_config: PortfolioConfig,
                           objective: str = 'sharpe_ratio') -> Dict:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            data: Price data
            features: Features data
            parameter_space: Dictionary of parameters to optimize
            portfolio_config: Portfolio configuration
            objective: Optimization objective ('sharpe_ratio', 'total_return', etc.)
            
        Returns:
            Dictionary with best parameters and results
        """
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(parameter_space.keys())
        param_values = list(parameter_space.values())
        
        best_score = -np.inf
        best_params = None
        best_result = None
        results = []
        
        for combination in product(*param_values):
            # Create parameter dictionary
            params = dict(zip(param_names, combination))
            
            # Update strategy config
            temp_config = self.config.copy()
            temp_config.update(params)
            
            try:
                # Create temporary strategy instance
                temp_strategy = self.__class__(temp_config)
                
                # Run backtest
                result = temp_strategy.backtest(data, features, portfolio_config)
                
                # Get objective score
                score = result.metrics.get(objective, -np.inf)
                
                results.append({
                    'parameters': params,
                    'score': score,
                    'metrics': result.metrics
                })
                
                # Update best if improved
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_result = result
                    
            except Exception as e:
                print(f"Error with parameters {params}: {e}")
                continue
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'best_result': best_result,
            'all_results': results
        }
    
    def get_strategy_info(self) -> Dict:
        """Get information about this strategy."""
        return {
            'name': self.name,
            'description': self.description,
            'class': self.__class__.__name__,
            'config': self.config,
            'required_features': self.get_required_features()
        }
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def _prepare_samples_info_sets(
        self,
        data: pd.DataFrame,
        prediction_horizon: int = 7
    ) -> Optional[List[Tuple]]:
        """
        Prepare samples_info_sets for purged CV.
        
        Creates (start_time, end_time) tuples for each sample based on
        prediction horizon. This assumes labels are forward-looking.
        
        Args:
            data: Price data DataFrame
            prediction_horizon: Number of periods ahead to predict
            
        Returns:
            List of (start_time, end_time) tuples or None
        """
        if data.empty or not isinstance(data.index, pd.DatetimeIndex):
            return None
        
        samples_info_sets = []
        for i in range(len(data)):
            start_time = data.index[i]
            # End time is start_time + prediction_horizon periods
            if i + prediction_horizon < len(data):
                end_time = data.index[i + prediction_horizon]
            else:
                end_time = data.index[-1]
            
            samples_info_sets.append((start_time, end_time))
        
        return samples_info_sets
    
    def run_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        returns: pd.Series,
        samples_info_sets: Optional[List[Tuple]] = None
    ) -> Optional['ValidationResults']:
        """
        Run validation framework for this strategy.
        
        Args:
            X: Features DataFrame
            y: Target Series
            returns: Returns Series
            samples_info_sets: Optional list of (start_time, end_time) tuples
            
        Returns:
            ValidationResults object or None if validation not available
        """
        if not VALIDATION_AVAILABLE or not self.validation_config:
            return None
        
        try:
            # Prepare samples_info_sets if not provided
            if samples_info_sets is None:
                samples_info_sets = self._prepare_samples_info_sets(X)
            
            # Create validation framework
            framework = ValidationFramework(self.validation_config)
            
            # Run validation
            results = framework.validate(X, y, returns, samples_info_sets)
            
            self.validation_results = results
            return results
        except Exception as e:
            print(f"Validation failed: {e}")
            return None 