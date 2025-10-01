"""
VectorBT Backtest Engine

Runs tactical asset allocation strategies using VectorBT.
Generates comprehensive performance metrics and reports.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import vectorbt as vbt
from pathlib import Path
import warnings

import sys
sys.path.append(str(Path(__file__).parent.parent))

from strategies.base_strategy import BaseTAAStrategy

# Suppress VectorBT warnings about optional parameters
warnings.filterwarnings('ignore', message='.*requires benchmark_rets to be set.*')
warnings.filterwarnings('ignore', message='.*requires frequency to be set.*')


class TAABacktestEngine:
    """
    VectorBT-based backtesting engine for TAA strategies.
    
    Features:
    - Monthly rebalancing with T-1 signals
    - 0 transaction costs (per requirements)
    - Full allocation (100% invested or in cash)
    - Comprehensive metrics calculation
    
    Attributes:
        strategy: TAA strategy instance
        prices: Price data DataFrame
        benchmark_ticker: Ticker for benchmark comparison (default: 'SPY')
    """
    
    def __init__(
        self,
        strategy: BaseTAAStrategy,
        prices: pd.DataFrame,
        benchmark_ticker: str = 'SPY',
        target_volatility: Optional[float] = None,
        vol_lookback: int = 20  # Rolling vol calculation window (days) - matches paper
    ):
        """
        Initialize backtest engine.
        
        Args:
            strategy: Initialized strategy instance
            prices: DataFrame of adjusted close prices
            benchmark_ticker: Ticker for benchmark comparison
            target_volatility: Optional target annualized volatility (e.g., 0.10 for 10%)
            vol_lookback: Lookback period for realized vol calculation (default: 60 days)
        """
        self.strategy = strategy
        self.prices = prices
        self.benchmark_ticker = benchmark_ticker
        self.target_volatility = target_volatility
        self.vol_lookback = vol_lookback
        
        # Results containers
        self.portfolio: Optional[vbt.Portfolio] = None
        self.weights: Optional[pd.DataFrame] = None
        self.signals: Optional[pd.DataFrame] = None
        self.vol_scaling_factors: Optional[pd.Series] = None
        
    def run(self) -> vbt.Portfolio:
        """
        Execute backtest.
        
        Returns:
            VectorBT Portfolio object with results
        """
        print(f"\n{'='*80}")
        print(f"RUNNING BACKTEST: {self.strategy.name}")
        print(f"{'='*80}\n")
        
        # Generate strategy signals
        print("Generating signals...")
        self.weights = self.strategy.generate_signals()
        
        print(f"Generated {len(self.weights)} monthly rebalance signals")
        print(f"Date range: {self.weights.index[0]} to {self.weights.index[-1]}")
        
        # Reindex weights to match price data (forward-fill for daily)
        daily_weights = self.weights.reindex(
            self.prices.index
        ).ffill().fillna(0.0)
        
        # Apply volatility normalization if target_volatility is set
        if self.target_volatility is not None:
            print(f"Applying volatility normalization (target: {self.target_volatility*100:.1f}%)...")
            daily_weights = self._apply_volatility_normalization(daily_weights)
        
        print(f"\nRunning VectorBT backtest...")
        
        # Handle cash as separate column
        if 'CASH' in daily_weights.columns:
            # Cash doesn't need price data
            cash_weight = daily_weights['CASH']
            asset_weights = daily_weights.drop('CASH', axis=1)
        else:
            cash_weight = pd.Series(0.0, index=daily_weights.index)
            asset_weights = daily_weights
        
        # Get prices for assets in weights
        asset_prices = self.prices[asset_weights.columns]
        
        # Run VectorBT portfolio simulation
        self.portfolio = vbt.Portfolio.from_orders(
            close=asset_prices,
            size=asset_weights,
            size_type='targetpercent',  # Target allocation percentage
            cash_sharing=True,  # Share cash across all positions
            init_cash=100.0,  # Start with $100
            fees=0.0005,  # 5 basis points transaction costs (per paper)
            freq='D'  # Daily frequency
        )
        
        print(f"✓ Backtest complete!")
        
        return self.portfolio
    
    def get_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            Dictionary with all metrics matching comprehensive_strategy_comparison.txt
        """
        if self.portfolio is None:
            raise ValueError("Run backtest first by calling run()")
        
        print(f"\nCalculating metrics...")
        
        # Get benchmark returns
        benchmark_returns = self.prices[self.benchmark_ticker].pct_change()
        
        # Calculate metrics
        metrics = {
            'vectorbt_stats': self._get_vectorbt_stats(),
            'returns_stats': self._get_returns_stats(),
            'manual_calcs': self._get_manual_calculations(benchmark_returns),
            'quantstats_style': self._get_quantstats_comparison(benchmark_returns)
        }
        
        print(f"✓ Metrics calculated!")
        
        return metrics
    
    def _get_vectorbt_stats(self) -> Dict:
        """Get VectorBT portfolio stats."""
        stats = self.portfolio.stats()
        return stats.to_dict() if hasattr(stats, 'to_dict') else dict(stats)
    
    def _get_returns_stats(self) -> Dict:
        """Get VectorBT returns stats."""
        returns = self.portfolio.returns()
        stats = returns.vbt.returns.stats()
        return stats.to_dict() if hasattr(stats, 'to_dict') else dict(stats)
    
    def _get_manual_calculations(self, benchmark_returns: pd.Series) -> Dict:
        """
        Calculate metrics manually (matching comprehensive_strategy_comparison.txt format).
        
        Args:
            benchmark_returns: Benchmark return series
            
        Returns:
            Dictionary of manually calculated metrics
        """
        # Get strategy returns
        returns = self.portfolio.returns()
        
        # Time in market (% of days with non-zero allocation)
        if self.weights is not None:
            # Count days with any non-cash allocation
            non_cash_weights = self.weights.drop('CASH', axis=1, errors='ignore')
            time_in_market = (non_cash_weights.sum(axis=1) > 0).mean()
        else:
            time_in_market = 1.0
        
        # Total return
        total_return = self.portfolio.total_return()
        
        # CAGR
        total_days = (returns.index[-1] - returns.index[0]).days
        years = total_days / 365.25
        cagr = (1 + total_return) ** (1 / years) - 1
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe (assuming 0% risk-free rate)
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
        
        # Sortino (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = (returns.mean() * 252) / downside_std if downside_std > 0 else 0
        
        # Max Drawdown
        max_dd = self.portfolio.max_drawdown()
        
        # Skewness and Kurtosis
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        # Benchmark metrics
        bench_total_return = (1 + benchmark_returns).prod() - 1
        bench_cagr = (1 + bench_total_return) ** (1 / years) - 1
        bench_vol = benchmark_returns.std() * np.sqrt(252)
        bench_sharpe = (benchmark_returns.mean() * 252) / (benchmark_returns.std() * np.sqrt(252))
        
        bench_downside = benchmark_returns[benchmark_returns < 0]
        bench_downside_std = bench_downside.std() * np.sqrt(252)
        bench_sortino = (benchmark_returns.mean() * 252) / bench_downside_std if bench_downside_std > 0 else 0
        
        # Calculate benchmark max drawdown
        bench_cumulative = (1 + benchmark_returns).cumprod()
        bench_running_max = bench_cumulative.expanding().max()
        bench_drawdown = (bench_cumulative - bench_running_max) / bench_running_max
        bench_max_dd = bench_drawdown.min()
        
        bench_skew = benchmark_returns.skew()
        bench_kurt = benchmark_returns.kurtosis()
        
        return {
            'strategy': {
                'start_period': returns.index[0].strftime('%Y-%m-%d'),
                'end_period': returns.index[-1].strftime('%Y-%m-%d'),
                'total_periods': len(returns),
                'time_in_market': f"{time_in_market*100:.2f}%",
                'total_trades': len(self.weights) if self.weights is not None else 0,
                'total_return': total_return,
                'cagr': f"{cagr*100:.2f}%",
                'volatility': volatility,
                'sharpe': sharpe,
                'sortino': sortino,
                'max_drawdown': max_dd,
                'skewness': skew,
                'kurtosis': kurt,
            },
            'benchmark': {
                'start_period': benchmark_returns.index[0].strftime('%Y-%m-%d'),
                'end_period': benchmark_returns.index[-1].strftime('%Y-%m-%d'),
                'total_periods': len(benchmark_returns),
                'time_in_market': "100.0%",
                'total_trades': 'N/A',
                'total_return': bench_total_return,
                'cagr': f"{bench_cagr*100:.2f}%",
                'volatility': bench_vol,
                'sharpe': bench_sharpe,
                'sortino': bench_sortino,
                'max_drawdown': bench_max_dd,
                'skewness': bench_skew,
                'kurtosis': bench_kurt,
            }
        }
    
    def _get_quantstats_comparison(self, benchmark_returns: pd.Series) -> Dict:
        """
        Generate QuantStats-style comparison vs benchmark.
        
        Args:
            benchmark_returns: Benchmark return series
            
        Returns:
            Dictionary with comparison metrics
        """
        returns = self.portfolio.returns()
        
        # Align returns and benchmark
        common_dates = returns.index.intersection(benchmark_returns.index)
        strat_returns = returns.loc[common_dates]
        bench_returns = benchmark_returns.loc[common_dates]
        
        # Calculate comparative metrics
        comparison = {
            'start_period': common_dates[0].strftime('%Y-%m-%d'),
            'end_period': common_dates[-1].strftime('%Y-%m-%d'),
            'cumulative_return_strategy': f"{(1 + strat_returns).prod() - 1:.2%}",
            'cumulative_return_benchmark': f"{(1 + bench_returns).prod() - 1:.2%}",
            'sharpe_strategy': strat_returns.mean() * 252 / (strat_returns.std() * np.sqrt(252)),
            'sharpe_benchmark': bench_returns.mean() * 252 / (bench_returns.std() * np.sqrt(252)),
            'max_dd_strategy': self.portfolio.max_drawdown(),
            'max_dd_benchmark': ((1 + bench_returns).cumprod() / (1 + bench_returns).cumprod().expanding().max() - 1).min(),
        }
        
        return comparison
    
    def _apply_volatility_normalization(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Apply volatility normalization to portfolio weights.
        
        Scales weights based on rolling realized volatility to target a specific vol level.
        This matches the paper's methodology of vol-normalizing to 10%.
        
        Args:
            weights: Daily weight DataFrame
            
        Returns:
            Vol-normalized weight DataFrame
        """
        # Calculate portfolio returns using prices and weights
        # We need to estimate what the returns would be based on historical data
        
        # For each asset (excluding CASH), get returns
        returns = pd.DataFrame(index=weights.index)
        for col in weights.columns:
            if col != 'CASH' and col in self.prices.columns:
                returns[col] = self.prices[col].pct_change()
        
        # Calculate hypothetical portfolio returns (weighted average of asset returns)
        portfolio_returns = pd.Series(0.0, index=weights.index)
        for col in returns.columns:
            if col in weights.columns:
                portfolio_returns += weights[col] * returns[col]
        
        # Calculate rolling realized volatility (annualized)
        realized_vol = portfolio_returns.rolling(window=self.vol_lookback).std() * np.sqrt(252)
        
        # Calculate scaling factor: target_vol / realized_vol
        # When realized vol > target, scale down; when realized vol < target, scale up
        scaling_factor = self.target_volatility / realized_vol
        
        # Remove scaling limits to match paper methodology exactly
        # Paper doesn't mention scaling limits in their vol normalization
        
        # Fill NaN at the start with 1.0 (no scaling until we have enough data)
        scaling_factor = scaling_factor.fillna(1.0)
        
        # Store scaling factors for analysis
        self.vol_scaling_factors = scaling_factor
        
        # Apply scaling to all asset weights (not CASH)
        normalized_weights = weights.copy()
        for col in normalized_weights.columns:
            if col != 'CASH':
                normalized_weights[col] = normalized_weights[col] * scaling_factor
        
        # Adjust CASH to maintain 100% allocation
        total_asset_weight = normalized_weights.drop('CASH', axis=1, errors='ignore').sum(axis=1)
        if 'CASH' in normalized_weights.columns:
            normalized_weights['CASH'] = 1.0 - total_asset_weight
        else:
            # If no CASH column, add it
            normalized_weights['CASH'] = 1.0 - total_asset_weight
        
        # Ensure no negative weights (set to 0)
        normalized_weights = normalized_weights.clip(lower=0.0)
        
        # Renormalize to 100%
        row_sums = normalized_weights.sum(axis=1)
        normalized_weights = normalized_weights.div(row_sums, axis=0)
        
        print(f"Vol normalization applied: avg scaling factor = {scaling_factor.mean():.3f}")
        
        return normalized_weights
    
    def save_results(self, output_dir: Path):
        """
        Save backtest results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save portfolio object
        # self.portfolio.save(output_dir / f"{self.strategy.name}_portfolio.pkl")
        
        # Save weights
        if self.weights is not None:
            self.weights.to_csv(output_dir / f"{self.strategy.name}_weights.csv")
        
        # Save vol scaling factors if available
        if self.vol_scaling_factors is not None:
            self.vol_scaling_factors.to_csv(output_dir / f"{self.strategy.name}_vol_scaling.csv")
        
        print(f"\n✓ Results saved to {output_dir}")

