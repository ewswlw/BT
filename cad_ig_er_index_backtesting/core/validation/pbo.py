"""
Probability of Backtest Overfitting (PBO) calculation.

Measures the probability that backtest performance is due to overfitting
rather than genuine predictive power.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PBOResults:
    """Results from PBO calculation."""
    pbo: float
    is_sharpe: float
    oos_sharpe: float
    best_is_config: int
    best_is_sharpe: float
    best_is_oos_sharpe: float
    is_sharpes: List[float]
    oos_sharpes: List[float]


class ProbabilityBacktestOverfitting:
    """
    Calculate Probability of Backtest Overfitting (PBO).
    
    Based on López de Prado's methodology for detecting overfitting
    in backtest results.
    """
    
    def __init__(self):
        """Initialize PBO calculator."""
        pass
    
    def calculate_pbo(
        self,
        returns_matrix: pd.DataFrame,
        split_ratio: float = 0.5
    ) -> PBOResults:
        """
        Calculate Probability of Backtest Overfitting.
        
        Args:
            returns_matrix: DataFrame with returns for multiple configurations
                          Rows = time periods, Columns = configurations
            split_ratio: Ratio for IS/OOS split (default: 0.5)
            
        Returns:
            PBOResults object with PBO and related metrics
        """
        if returns_matrix.empty:
            raise ValueError("returns_matrix cannot be empty")
        
        n_configs = returns_matrix.shape[1]
        if n_configs < 2:
            raise ValueError("Need at least 2 configurations for PBO calculation")
        
        # Split into IS and OOS
        split_point = int(len(returns_matrix) * split_ratio)
        is_returns = returns_matrix.iloc[:split_point]
        oos_returns = returns_matrix.iloc[split_point:]
        
        # Calculate Sharpe ratios for each configuration
        is_sharpes = []
        oos_sharpes = []
        
        for col in returns_matrix.columns:
            is_ret = is_returns[col].dropna()
            oos_ret = oos_returns[col].dropna()
            
            if len(is_ret) > 1 and is_ret.std() > 0:
                is_sr = is_ret.mean() / is_ret.std() * np.sqrt(252)
            else:
                is_sr = 0.0
            
            if len(oos_ret) > 1 and oos_ret.std() > 0:
                oos_sr = oos_ret.mean() / oos_ret.std() * np.sqrt(252)
            else:
                oos_sr = 0.0
            
            is_sharpes.append(is_sr)
            oos_sharpes.append(oos_sr)
        
        is_sharpes = np.array(is_sharpes)
        oos_sharpes = np.array(oos_sharpes)
        
        # Find best IS configuration
        best_is_config = int(np.argmax(is_sharpes))
        best_is_sharpe = float(is_sharpes[best_is_config])
        best_is_oos_sharpe = float(oos_sharpes[best_is_config])
        
        # Calculate median OOS Sharpe
        median_oos = float(np.median(oos_sharpes))
        
        # Count configurations where OOS performance < median OOS
        n_worse = int(np.sum(oos_sharpes < median_oos))
        
        # PBO
        pbo = n_worse / n_configs
        
        return PBOResults(
            pbo=pbo,
            is_sharpe=float(np.mean(is_sharpes)),
            oos_sharpe=float(np.mean(oos_sharpes)),
            best_is_config=best_is_config,
            best_is_sharpe=best_is_sharpe,
            best_is_oos_sharpe=best_is_oos_sharpe,
            is_sharpes=is_sharpes.tolist(),
            oos_sharpes=oos_sharpes.tolist()
        )


def calculate_pbo(
    returns_matrix: pd.DataFrame,
    split_ratio: float = 0.5
) -> PBOResults:
    """
    Calculate Probability of Backtest Overfitting.
    
    Convenience function wrapper.
    
    Args:
        returns_matrix: DataFrame with returns for multiple configurations
        split_ratio: Ratio for IS/OOS split
        
    Returns:
        PBOResults object
    """
    calculator = ProbabilityBacktestOverfitting()
    return calculator.calculate_pbo(returns_matrix, split_ratio)

