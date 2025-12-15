"""
Deflated performance metrics for multiple testing adjustment.

Implements deflated Sharpe ratio and enhanced probabilistic Sharpe ratio
to account for multiple testing bias.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Optional


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_observations: int,
    skew: float = 0.0,
    kurt: float = 3.0
) -> float:
    """
    Calculate Deflated Sharpe Ratio.
    
    Adjusts Sharpe ratio to account for multiple testing bias.
    Returns the probability that the Sharpe ratio is genuine.
    
    Based on López de Prado's formula from "Advances in Financial ML"
    
    Args:
        sharpe: Estimated Sharpe ratio
        n_trials: Number of strategies tested
        n_observations: Number of observations
        skew: Skewness of returns (default: 0.0)
        kurt: Kurtosis of returns (default: 3.0 for normal)
        
    Returns:
        Deflated Sharpe ratio (probability that SR is genuine)
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_observations < 2:
        raise ValueError("n_observations must be at least 2")
    
    # Euler-Mascheroni constant
    euler_mascheroni = 0.5772156649
    
    # Expected maximum Sharpe ratio under null hypothesis
    # Using approximation from López de Prado
    if n_trials == 1:
        expected_max_sr = 0.0
    else:
        expected_max_sr = (
            (1 - euler_mascheroni) * norm.ppf(1 - 1.0 / n_trials) +
            euler_mascheroni * norm.ppf(1 - 1.0 / (n_trials * np.e))
        )
    
    # Variance of Sharpe ratio
    var_sr = (
        1 + (1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2)
    ) / (n_observations - 1)
    
    if var_sr <= 0:
        return 0.5  # Neutral probability if variance is invalid
    
    # Deflated Sharpe ratio
    dsr = norm.cdf((sharpe - expected_max_sr) / np.sqrt(var_sr))
    
    return max(0.0, min(1.0, dsr))  # Clamp to [0, 1]


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    benchmark_sr: float = 0.0,
    annual_factor: int = 252
) -> float:
    """
    Calculate Probabilistic Sharpe Ratio.
    
    Computes the probability that the estimated Sharpe ratio exceeds
    a benchmark, accounting for skewness and kurtosis.
    
    Formula: PSR(SR*) = Φ(((SR - SR*) √(N-1)) / √(1 - γ₃SR + (γ₄-1)/4 SR²))
    
    Where:
        SR = estimated Sharpe ratio
        SR* = benchmark Sharpe ratio
        N = number of observations
        γ₃ = skewness
        γ₄ = kurtosis
        Φ = standard normal CDF
    
    Args:
        returns: Series of returns
        benchmark_sr: Benchmark Sharpe ratio (default: 0.0)
        annual_factor: Annualization factor (default: 252 for daily)
        
    Returns:
        Probabilistic Sharpe ratio (probability that SR > benchmark)
    """
    if len(returns) < 3:
        return 0.5  # Neutral if insufficient data
    
    # Calculate statistics
    sr = (returns.mean() * annual_factor) / (returns.std() * np.sqrt(annual_factor))
    n = len(returns)
    
    # Calculate skewness and kurtosis
    skewness = returns.skew()
    kurt = returns.kurtosis() + 3  # scipy returns excess kurtosis
    
    # PSR formula
    numerator = (sr - benchmark_sr) * np.sqrt(n - 1)
    denominator = np.sqrt(1 - skewness * sr + (kurt - 1) / 4 * sr**2)
    
    if denominator == 0 or not np.isfinite(denominator):
        return 0.5
    
    z_score = numerator / denominator
    psr = norm.cdf(z_score)
    
    return max(0.0, min(1.0, psr))  # Clamp to [0, 1]


def min_backtest_length(
    target_sr: float,
    expected_sr: float,
    n_observations: int
) -> int:
    """
    Calculate minimum backtest length.
    
    Determines the minimum number of observations needed to have confidence
    in backtest results.
    
    Formula: MinBTL = ((SR* / SR)^2 - 1) × N
    
    Where:
        SR* = target Sharpe ratio
        SR = expected Sharpe ratio
        N = number of observations in original backtest
    
    Args:
        target_sr: Target Sharpe ratio
        expected_sr: Expected Sharpe ratio
        n_observations: Number of observations in original backtest
        
    Returns:
        Minimum backtest length in periods
    """
    if expected_sr == 0:
        return np.inf
    
    if target_sr <= 0 or expected_sr <= 0:
        return n_observations  # Return current length if invalid
    
    min_length = ((target_sr / expected_sr)**2 - 1) * n_observations
    
    return int(np.ceil(max(n_observations, min_length)))

