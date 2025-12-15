"""
Walk-Forward Analysis for backtesting.

Simulates real-world trading by training on past data and testing
on future data, then moving forward in time.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass


@dataclass
class WalkForwardResult:
    """Result from a single walk-forward period."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    sharpe: Optional[float] = None
    return_: Optional[float] = None
    volatility: Optional[float] = None
    max_drawdown: Optional[float] = None
    metrics: Optional[Dict] = None


class WalkForwardAnalyzer:
    """
    Perform walk-forward analysis.
    
    Supports both expanding and rolling window approaches.
    """
    
    def __init__(
        self,
        train_period: int = 252,
        test_period: int = 63,
        step: int = 21,
        window_type: str = "expanding"
    ):
        """
        Initialize WalkForwardAnalyzer.
        
        Args:
            train_period: Training window size in periods
            test_period: Testing window size in periods
            step: Step size for rolling window in periods
            window_type: "expanding" or "rolling"
        """
        if train_period < 1:
            raise ValueError("train_period must be at least 1")
        if test_period < 1:
            raise ValueError("test_period must be at least 1")
        if step < 1:
            raise ValueError("step must be at least 1")
        if window_type not in ["expanding", "rolling"]:
            raise ValueError("window_type must be 'expanding' or 'rolling'")
        
        self.train_period = train_period
        self.test_period = test_period
        self.step = step
        self.window_type = window_type
    
    def analyze(
        self,
        data: pd.DataFrame,
        returns: pd.Series,
        model_fn: Optional[Callable] = None
    ) -> List[WalkForwardResult]:
        """
        Perform walk-forward analysis.
        
        Args:
            data: Feature data DataFrame
            returns: Returns Series
            model_fn: Optional function to train/evaluate model
                     Signature: (train_data, train_returns, test_data) -> metrics_dict
                     
        Returns:
            List of WalkForwardResult objects
        """
        results = []
        n_periods = len(data)
        
        if n_periods < self.train_period + self.test_period:
            # Not enough data
            return results
        
        i = 0
        while i + self.train_period + self.test_period <= n_periods:
            # Define windows
            if self.window_type == "expanding":
                train_start_idx = 0
            else:  # rolling
                train_start_idx = max(0, i - self.train_period)
            
            train_end_idx = i + self.train_period
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + self.test_period, n_periods)
            
            # Get data
            train_data = data.iloc[train_start_idx:train_end_idx]
            test_data = data.iloc[test_start_idx:test_end_idx]
            train_returns = returns.iloc[train_start_idx:train_end_idx]
            test_returns = returns.iloc[test_start_idx:test_end_idx]
            
            # Calculate basic metrics
            result = WalkForwardResult(
                train_start=data.index[train_start_idx],
                train_end=data.index[train_end_idx - 1],
                test_start=data.index[test_start_idx],
                test_end=data.index[test_end_idx - 1]
            )
            
            # Calculate performance metrics
            if len(test_returns) > 0:
                test_returns_clean = test_returns.dropna()
                if len(test_returns_clean) > 1:
                    result.return_ = float(test_returns_clean.mean() * 252)  # Annualized
                    result.volatility = float(test_returns_clean.std() * np.sqrt(252))
                    
                    if result.volatility > 0:
                        result.sharpe = result.return_ / result.volatility
                    
                    # Calculate max drawdown
                    equity_curve = (1 + test_returns_clean).cumprod()
                    rolling_max = equity_curve.expanding().max()
                    drawdown = (equity_curve - rolling_max) / rolling_max
                    result.max_drawdown = float(drawdown.min())
            
            # If model function provided, get additional metrics
            if model_fn is not None:
                try:
                    metrics = model_fn(train_data, train_returns, test_data)
                    result.metrics = metrics
                except Exception as e:
                    # Model evaluation failed, continue with basic metrics
                    pass
            
            results.append(result)
            
            # Move forward
            i += self.step
        
        return results
    
    def summarize_results(self, results: List[WalkForwardResult]) -> Dict:
        """
        Summarize walk-forward results.
        
        Args:
            results: List of WalkForwardResult objects
            
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {}
        
        sharpes = [r.sharpe for r in results if r.sharpe is not None]
        returns = [r.return_ for r in results if r.return_ is not None]
        drawdowns = [r.max_drawdown for r in results if r.max_drawdown is not None]
        
        summary = {
            'n_periods': len(results),
            'sharpe_mean': float(np.mean(sharpes)) if sharpes else None,
            'sharpe_std': float(np.std(sharpes)) if sharpes else None,
            'sharpe_min': float(np.min(sharpes)) if sharpes else None,
            'sharpe_max': float(np.max(sharpes)) if sharpes else None,
            'return_mean': float(np.mean(returns)) if returns else None,
            'return_std': float(np.std(returns)) if returns else None,
            'max_dd_mean': float(np.mean(drawdowns)) if drawdowns else None,
            'max_dd_worst': float(np.min(drawdowns)) if drawdowns else None,
        }
        
        return summary

