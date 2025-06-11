"""
Comprehensive performance metrics calculation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import scipy.stats as stats

try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False
    print("Warning: quantstats not available, using manual calculations")


class MetricsCalculator:
    """Comprehensive performance metrics calculator."""
    
    def __init__(self, risk_free_rate: float = 0.0, frequency: str = 'W'):
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency
        
        # Set annualization factor based on frequency
        self.annual_factor = {
            'D': 252,   # Daily
            'W': 52,    # Weekly  
            'M': 12,    # Monthly
            'Q': 4,     # Quarterly
            'Y': 1      # Yearly
        }.get(frequency, 52)
    
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
            
            # Additional Sharpe variations
            metrics['prob_sharpe'] = self._calculate_probabilistic_sharpe(returns)
            metrics['smart_sharpe'] = metrics['sharpe'] * 0.79  # Approximation
            metrics['smart_sortino'] = metrics['sortino'] * 0.79
            metrics['sortino_sqrt2'] = metrics['sortino'] / np.sqrt(2)
            metrics['smart_sortino_sqrt2'] = metrics['smart_sortino'] / np.sqrt(2)
            
            # Drawdown metrics
            metrics['max_drawdown'] = qs.stats.max_drawdown(returns)
            metrics['longest_dd_days'] = self._calculate_longest_drawdown_duration(returns)
            
            # Other metrics
            metrics['skew'] = qs.stats.skew(returns)
            metrics['kurtosis'] = qs.stats.kurtosis(returns)
            
            # Omega ratio
            metrics['omega'] = self._calculate_omega_ratio(returns)
            
            # VaR and CVaR
            metrics['daily_var'] = qs.stats.var(returns)
            metrics['cvar'] = qs.stats.cvar(returns)
            
            # Win/Loss metrics
            win_rate, gain_pain_ratio = self._calculate_win_loss_metrics(returns)
            metrics['win_rate'] = win_rate
            metrics['gain_pain_ratio'] = gain_pain_ratio
            
            # Expected returns
            metrics['expected_daily'] = returns.mean()
            metrics['expected_weekly'] = returns.mean()
            metrics['expected_monthly'] = returns.mean() * (self.annual_factor / 12)
            metrics['expected_yearly'] = returns.mean() * self.annual_factor
            
            # Kelly Criterion
            metrics['kelly_criterion'] = self._calculate_kelly_criterion(returns)
            
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
            
            # Additional calculated metrics
            metrics['profit_factor'] = self._calculate_profit_factor(returns)
            metrics['payoff_ratio'] = self._calculate_payoff_ratio(returns)
            metrics['common_sense_ratio'] = self._calculate_common_sense_ratio(returns)
            metrics['cpc_index'] = self._calculate_cpc_index(returns)
            metrics['tail_ratio'] = self._calculate_tail_ratio(returns)
            
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
        metrics['longest_dd_days'] = self._calculate_longest_drawdown_duration(returns)
        
        # Basic stats
        metrics['skew'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        
        # Fill in other metrics with basic calculations or defaults
        metrics['calmar'] = metrics['cagr'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        metrics['prob_sharpe'] = self._calculate_probabilistic_sharpe(returns)
        metrics['omega'] = self._calculate_omega_ratio(returns)
        
        return metrics
    
    def _calculate_probabilistic_sharpe(self, returns: pd.Series) -> float:
        """Calculate Probabilistic Sharpe Ratio."""
        try:
            sr = (returns.mean() * self.annual_factor) / (returns.std() * np.sqrt(self.annual_factor))
            n = len(returns)
            
            if n > 3:
                skew = returns.skew()
                kurt = returns.kurtosis()
                
                denominator = np.sqrt(1 - skew*sr + (kurt-1)/4*sr**2)
                if denominator != 0:
                    z = (sr * np.sqrt(n - 1)) / denominator
                    psr = stats.norm.cdf(z)
                else:
                    psr = 0.5
            else:
                psr = 0.5
                
            return psr
        except:
            return 0.5
    
    def _calculate_longest_drawdown_duration(self, returns: pd.Series) -> int:
        """Calculate longest drawdown duration in periods."""
        equity_curve = (1 + returns).cumprod()
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        
        in_drawdown = drawdown < 0
        if not in_drawdown.any():
            return 0
        
        drawdown_periods = []
        start = None
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start is None:
                start = i
            elif not is_dd and start is not None:
                drawdown_periods.append(i - start)
                start = None
        
        if start is not None:  # Still in drawdown at end
            drawdown_periods.append(len(in_drawdown) - start)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega ratio."""
        positive_returns = returns[returns > threshold].sum()
        negative_returns = abs(returns[returns <= threshold].sum())
        return positive_returns / negative_returns if negative_returns > 0 else np.inf
    
    def _calculate_kelly_criterion(self, returns: pd.Series) -> float:
        """Calculate Kelly Criterion."""
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns <= 0].mean()) if (returns <= 0).any() else 0
        
        if avg_loss > 0:
            payoff_ratio = avg_win / avg_loss
            kelly = win_rate - (1 - win_rate) / payoff_ratio
        else:
            kelly = win_rate
            
        return max(0, kelly)  # Don't allow negative Kelly
    
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
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate Profit Factor."""
        return self._calculate_omega_ratio(returns, 0.0)
    
    def _calculate_payoff_ratio(self, returns: pd.Series) -> float:
        """Calculate Payoff Ratio (average win / average loss)."""
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns <= 0].mean()) if (returns <= 0).any() else 0
        return avg_win / avg_loss if avg_loss > 0 else np.inf
    
    def _calculate_common_sense_ratio(self, returns: pd.Series) -> float:
        """Calculate Common Sense Ratio."""
        # This is an approximation - actual CSR calculation is more complex
        tail_ratio = self._calculate_tail_ratio(returns)
        profit_factor = self._calculate_profit_factor(returns)
        return (tail_ratio * profit_factor) ** 0.5 if profit_factor > 0 else 0
    
    def _calculate_cpc_index(self, returns: pd.Series) -> float:
        """Calculate CPC Index (Conservative Performance Compound Index)."""
        # Simplified CPC calculation
        profit_factor = self._calculate_profit_factor(returns)
        win_rate = (returns > 0).mean()
        return profit_factor * win_rate
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate Tail Ratio (95th percentile / 5th percentile)."""
        p95 = returns.quantile(0.95)
        p5 = returns.quantile(0.05)
        return abs(p95 / p5) if p5 != 0 else 0
    
    def calculate_benchmark_comparison(self, 
                                    strategy_returns: pd.Series,
                                    benchmark_returns: pd.Series) -> Dict:
        """Calculate strategy vs benchmark comparison metrics."""
        # Align the series
        strategy_aligned, benchmark_aligned = strategy_returns.align(benchmark_returns, join='inner')
        
        strategy_metrics = self.calculate_comprehensive_metrics(strategy_aligned, benchmark_aligned, "Strategy")
        benchmark_metrics = self.calculate_comprehensive_metrics(benchmark_aligned, None, "Benchmark")
        
        # Calculate outperformance
        outperformance = strategy_metrics['total_return'] - benchmark_metrics['total_return']
        
        return {
            'strategy': strategy_metrics,
            'benchmark': benchmark_metrics,
            'outperformance': outperformance,
            'excess_sharpe': strategy_metrics['sharpe'] - benchmark_metrics['sharpe']
        } 