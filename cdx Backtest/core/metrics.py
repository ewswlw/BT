"""
Comprehensive performance metrics calculation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False
    print("Warning: quantstats not available, using manual calculations")


class MetricsCalculator:
    """Comprehensive performance metrics calculator."""
    
    def __init__(self, risk_free_rate: float = 0.0, frequency: str = 'D'):
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency
        
        # Set annualization factor based on frequency
        self.annual_factor = {
            'D': 252,   # Daily
            'W': 52,    # Weekly  
            'M': 12,    # Monthly
            'Q': 4,     # Quarterly
            'Y': 1      # Yearly
        }.get(frequency, 252)
    
    def calculate_comprehensive_metrics(self, 
                                      returns: pd.Series, 
                                      benchmark_returns: Optional[pd.Series] = None,
                                      name: str = "Strategy") -> Dict:
        """Calculate comprehensive performance metrics."""
        
        if QUANTSTATS_AVAILABLE:
            return self._calculate_quantstats_metrics(returns, benchmark_returns, name)
        else:
            return self._calculate_manual_metrics(returns, benchmark_returns, name)
    
    def _calculate_quantstats_metrics(self, 
                                    returns: pd.Series,
                                    benchmark_returns: Optional[pd.Series] = None,
                                    name: str = "Strategy") -> Dict:
        """Calculate metrics using quantstats library."""
        metrics = {}
        
        try:
            # Basic metrics
            metrics['start_period'] = returns.index[0].strftime('%Y-%m-%d')
            metrics['end_period'] = returns.index[-1].strftime('%Y-%m-%d')
            metrics['risk_free_rate'] = f"{self.risk_free_rate:.1%}"
            
            # Return metrics
            metrics['total_return'] = qs.stats.comp(returns)
            metrics['cagr'] = qs.stats.cagr(returns)
            
            # Risk metrics
            metrics['volatility'] = qs.stats.volatility(returns, annualize=True)
            metrics['sharpe'] = qs.stats.sharpe(returns, rf=self.risk_free_rate)
            metrics['sortino'] = qs.stats.sortino(returns, rf=self.risk_free_rate)
            metrics['calmar'] = qs.stats.calmar(returns)
            
            # Drawdown metrics
            metrics['max_drawdown'] = qs.stats.max_drawdown(returns)
            
            # Other metrics
            metrics['skew'] = qs.stats.skew(returns)
            metrics['kurtosis'] = qs.stats.kurtosis(returns)
            
            # Win/Loss metrics
            win_rate, gain_pain_ratio = self._calculate_win_loss_metrics(returns)
            metrics['win_rate'] = win_rate
            metrics['gain_pain_ratio'] = gain_pain_ratio
            
            # Expected returns
            metrics['expected_daily'] = returns.mean()
            metrics['expected_yearly'] = returns.mean() * self.annual_factor
            
            # R-squared (if benchmark provided)
            if benchmark_returns is not None:
                aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
                if len(aligned_returns) > 1:
                    metrics['r_squared'] = aligned_returns.corr(aligned_benchmark) ** 2
                    metrics['information_ratio'] = self._calculate_information_ratio(
                        aligned_returns, aligned_benchmark
                    )
                else:
                    metrics['r_squared'] = 0.0
                    metrics['information_ratio'] = 0.0
            else:
                metrics['r_squared'] = 0.0
                metrics['information_ratio'] = 0.0
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            # Return basic metrics if quantstats fails
            return self._calculate_manual_metrics(returns, benchmark_returns, name)
        
        return metrics
    
    def _calculate_manual_metrics(self, 
                                returns: pd.Series,
                                benchmark_returns: Optional[pd.Series] = None,
                                name: str = "Strategy") -> Dict:
        """Calculate metrics manually when quantstats is not available."""
        metrics = {}
        
        # Basic info
        metrics['start_period'] = returns.index[0].strftime('%Y-%m-%d')
        metrics['end_period'] = returns.index[-1].strftime('%Y-%m-%d')
        metrics['risk_free_rate'] = f"{self.risk_free_rate:.1%}"
        
        # Return metrics
        total_return = (1 + returns).prod() - 1
        metrics['total_return'] = total_return
        
        years = len(returns) / self.annual_factor
        metrics['cagr'] = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Risk metrics
        metrics['volatility'] = returns.std() * np.sqrt(self.annual_factor)
        
        excess_returns = returns - self.risk_free_rate / self.annual_factor
        metrics['sharpe'] = (excess_returns.mean() * self.annual_factor) / metrics['volatility'] if metrics['volatility'] > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(self.annual_factor) if len(downside_returns) > 0 else 0
        metrics['sortino'] = (excess_returns.mean() * self.annual_factor) / downside_std if downside_std > 0 else 0
        
        # Drawdown
        equity_curve = (1 + returns).cumprod()
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        metrics['max_drawdown'] = drawdown.min()
        
        # Basic stats
        metrics['skew'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        
        # Calmar ratio
        metrics['calmar'] = metrics['cagr'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        
        # Win/Loss metrics
        win_rate, gain_pain_ratio = self._calculate_win_loss_metrics(returns)
        metrics['win_rate'] = win_rate
        metrics['gain_pain_ratio'] = gain_pain_ratio
        
        # Expected returns
        metrics['expected_daily'] = returns.mean()
        metrics['expected_yearly'] = returns.mean() * self.annual_factor
        
        # R-squared (if benchmark provided)
        if benchmark_returns is not None:
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
            if len(aligned_returns) > 1:
                metrics['r_squared'] = aligned_returns.corr(aligned_benchmark) ** 2
                metrics['information_ratio'] = self._calculate_information_ratio(
                    aligned_returns, aligned_benchmark
                )
            else:
                metrics['r_squared'] = 0.0
                metrics['information_ratio'] = 0.0
        else:
            metrics['r_squared'] = 0.0
            metrics['information_ratio'] = 0.0
        
        return metrics
    
    def _calculate_information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Information Ratio."""
        excess_returns = returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(self.annual_factor)
        return (excess_returns.mean() * self.annual_factor) / tracking_error if tracking_error > 0 else 0
    
    def _calculate_win_loss_metrics(self, returns: pd.Series) -> Tuple[float, float]:
        """Calculate win rate and gain/pain ratio."""
        win_rate = (returns > 0).mean()
        
        positive_sum = returns[returns > 0].sum()
        negative_sum = abs(returns[returns <= 0].sum())
        gain_pain_ratio = positive_sum / negative_sum if negative_sum > 0 else np.inf
        
        return win_rate, gain_pain_ratio
    
    def calculate_benchmark_comparison(self, 
                                    strategy_returns: pd.Series,
                                    benchmark_returns: pd.Series) -> Dict:
        """Calculate strategy vs benchmark comparison metrics."""
        # Align the series
        strategy_aligned, benchmark_aligned = strategy_returns.align(benchmark_returns, join='inner')
        
        strategy_metrics = self.calculate_comprehensive_metrics(strategy_aligned, benchmark_aligned, "Strategy")
        benchmark_metrics = self.calculate_comprehensive_metrics(benchmark_aligned, None, "Benchmark")
        
        # Calculate outperformance
        outperformance_abs = strategy_metrics['cagr'] - benchmark_metrics['cagr']
        outperformance_pct = (strategy_metrics['cagr'] / benchmark_metrics['cagr'] - 1) * 100 if benchmark_metrics['cagr'] != 0 else 0
        
        return {
            'strategy': strategy_metrics,
            'benchmark': benchmark_metrics,
            'outperformance_abs': outperformance_abs,
            'outperformance_pct': outperformance_pct,
            'excess_sharpe': strategy_metrics['sharpe'] - benchmark_metrics['sharpe']
        }

