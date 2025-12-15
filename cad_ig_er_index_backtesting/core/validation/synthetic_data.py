"""
Synthetic data generation for robustness testing.

Generates synthetic price paths that preserve statistical properties
of real data for testing strategy robustness.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Callable
from scipy.stats import norm


class SyntheticDataGenerator:
    """
    Generate synthetic financial data for robustness testing.
    
    Creates synthetic price paths that preserve key statistical properties
    of the original data.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize SyntheticDataGenerator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        self.random_seed = random_seed
    
    def generate_returns(
        self,
        real_returns: pd.Series,
        n_simulations: int = 1000,
        method: str = "normal"
    ) -> pd.DataFrame:
        """
        Generate synthetic returns.
        
        Args:
            real_returns: Historical returns Series
            n_simulations: Number of synthetic paths to generate
            method: Generation method ("normal", "bootstrap", "garch")
            
        Returns:
            DataFrame with synthetic returns (columns = simulations)
        """
        n_periods = len(real_returns)
        
        if method == "normal":
            return self._generate_normal_returns(real_returns, n_simulations, n_periods)
        elif method == "bootstrap":
            return self._generate_bootstrap_returns(real_returns, n_simulations, n_periods)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _generate_normal_returns(
        self,
        real_returns: pd.Series,
        n_simulations: int,
        n_periods: int
    ) -> pd.DataFrame:
        """Generate returns using normal distribution."""
        # Estimate parameters
        mu = real_returns.mean()
        sigma = real_returns.std()
        
        # Generate synthetic returns
        synthetic_returns = np.random.normal(
            mu, sigma,
            size=(n_periods, n_simulations)
        )
        
        return pd.DataFrame(
            synthetic_returns,
            index=real_returns.index,
            columns=[f'sim_{i}' for i in range(n_simulations)]
        )
    
    def _generate_bootstrap_returns(
        self,
        real_returns: pd.Series,
        n_simulations: int,
        n_periods: int
    ) -> pd.DataFrame:
        """Generate returns using bootstrap resampling."""
        synthetic_returns = np.zeros((n_periods, n_simulations))
        
        for sim in range(n_simulations):
            # Sample with replacement
            bootstrap_sample = np.random.choice(
                real_returns.values,
                size=n_periods,
                replace=True
            )
            synthetic_returns[:, sim] = bootstrap_sample
        
        return pd.DataFrame(
            synthetic_returns,
            index=real_returns.index,
            columns=[f'sim_{i}' for i in range(n_simulations)]
        )
    
    def generate_prices(
        self,
        real_returns: pd.Series,
        initial_price: float = 100.0,
        n_simulations: int = 1000,
        method: str = "normal"
    ) -> pd.DataFrame:
        """
        Generate synthetic price paths.
        
        Args:
            real_returns: Historical returns Series
            initial_price: Starting price
            n_simulations: Number of synthetic paths
            method: Generation method
            
        Returns:
            DataFrame with synthetic prices
        """
        # Generate returns
        synthetic_returns = self.generate_returns(real_returns, n_simulations, method)
        
        # Convert to prices
        synthetic_prices = initial_price * (1 + synthetic_returns).cumprod(axis=0)
        
        return synthetic_prices
    
    def test_robustness(
        self,
        real_returns: pd.Series,
        strategy_fn: Callable,
        n_simulations: int = 1000,
        initial_price: float = 100.0
    ) -> Dict:
        """
        Test strategy robustness on synthetic data.
        
        Args:
            real_returns: Historical returns
            strategy_fn: Function that takes prices and returns performance metrics
                        Signature: (prices: pd.Series) -> Dict[str, float]
            n_simulations: Number of simulations
            initial_price: Starting price
            
        Returns:
            Dictionary with robustness test results
        """
        # Generate synthetic prices
        synthetic_prices = self.generate_prices(
            real_returns,
            initial_price,
            n_simulations
        )
        
        # Test strategy on each simulation
        results = []
        for col in synthetic_prices.columns:
            try:
                prices = synthetic_prices[col]
                metrics = strategy_fn(prices)
                results.append(metrics)
            except Exception as e:
                # Skip failed simulations
                continue
        
        if not results:
            return {
                'n_simulations': 0,
                'mean_performance': None,
                'std_performance': None,
                'success_rate': 0.0
            }
        
        # Aggregate results
        # Assuming metrics dict has a 'sharpe' or 'return' key
        performance_key = None
        for key in ['sharpe', 'return', 'total_return', 'cagr']:
            if key in results[0]:
                performance_key = key
                break
        
        if performance_key:
            performances = [r[performance_key] for r in results if performance_key in r]
        else:
            performances = [0.0] * len(results)
        
        return {
            'n_simulations': len(results),
            'mean_performance': float(np.mean(performances)) if performances else None,
            'std_performance': float(np.std(performances)) if performances else None,
            'min_performance': float(np.min(performances)) if performances else None,
            'max_performance': float(np.max(performances)) if performances else None,
            'success_rate': len([p for p in performances if p > 0]) / len(performances) if performances else 0.0,
            'performance_key': performance_key
        }

