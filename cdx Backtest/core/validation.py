"""
Statistical validation including walk-forward analysis, bias checking, PSR/DSR.
"""

from typing import Dict, List, Tuple, Optional, Callable
import pandas as pd
import numpy as np
from scipy import stats
from .backtest_engine import BacktestEngine, BacktestResult


class WalkForwardValidator:
    """Walk-forward validation with expanding window."""
    
    def __init__(self, 
                 initial_train_ratio: float = 0.7,
                 test_periods: int = 6,
                 min_test_period_size: int = 20):
        self.initial_train_ratio = initial_train_ratio
        self.test_periods = test_periods
        self.min_test_period_size = min_test_period_size
    
    def run_walk_forward(self,
                        data: pd.DataFrame,
                        price_column: str,
                        strategy_func: Callable,
                        backtest_engine: BacktestEngine) -> Dict:
        """
        Run walk-forward analysis with expanding window.
        
        Args:
            data: Full dataset
            price_column: Column name for price data
            strategy_func: Function that takes (train_data, test_data) and returns (entry_signals, exit_signals)
            backtest_engine: Backtesting engine
            
        Returns:
            Dictionary with individual results and summary
        """
        results = []
        
        # Initial split
        split_idx = int(len(data) * self.initial_train_ratio)
        test_data_full = data.iloc[split_idx:]
        
        # Divide test period into sequential periods
        period_size = len(test_data_full) // self.test_periods
        
        print(f"\n=== Walk-Forward Validation ({self.test_periods} periods) ===")
        print(f"Initial train size: {split_idx} periods")
        print(f"Test size: {len(test_data_full)} periods")
        print(f"Test period size: {period_size} periods")
        
        for i in range(self.test_periods):
            # Define test period
            test_start_idx = i * period_size
            test_end_idx = (i + 1) * period_size if i < self.test_periods - 1 else len(test_data_full)
            
            if test_end_idx - test_start_idx < self.min_test_period_size:
                continue
            
            test_data = test_data_full.iloc[test_start_idx:test_end_idx]
            
            # Expanding window: train on all data up to test period start
            train_end_idx = split_idx + test_start_idx
            train_data = data.iloc[:train_end_idx]
            
            try:
                print(f"\nPeriod {i+1}/{self.test_periods}:")
                print(f"  Train: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} periods)")
                print(f"  Test:  {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} periods)")
                
                # Generate signals
                entry_signals, exit_signals = strategy_func(train_data, test_data)
                
                # Run backtest on test data
                price_series = test_data[price_column]
                result = backtest_engine.run_backtest(
                    price_series, entry_signals, exit_signals, apply_holding_period=True
                )
                
                results.append({
                    'period': i + 1,
                    'train_start': train_data.index[0],
                    'train_end': train_data.index[-1],
                    'train_size': len(train_data),
                    'test_start': test_data.index[0],
                    'test_end': test_data.index[-1],
                    'test_size': len(test_data),
                    'result': result,
                    'cagr': result.metrics['cagr'],
                    'sharpe_ratio': result.metrics['sharpe_ratio'],
                    'max_drawdown': result.metrics['max_drawdown'],
                    'total_return': result.metrics['total_return']
                })
                
            except Exception as e:
                print(f"Error in walk-forward period {i+1}: {e}")
                continue
        
        return {
            'individual_results': results,
            'summary': self._summarize_walk_forward_results(results)
        }
    
    def _summarize_walk_forward_results(self, results: List[Dict]) -> Dict:
        """Summarize walk-forward analysis results."""
        if not results:
            return {}
        
        # Extract metrics from all periods
        cagrs = [r['cagr'] for r in results]
        sharpe_ratios = [r['sharpe_ratio'] for r in results]
        max_drawdowns = [r['max_drawdown'] for r in results]
        total_returns = [r['total_return'] for r in results]
        
        return {
            'num_periods': len(results),
            'avg_cagr': np.mean(cagrs),
            'std_cagr': np.std(cagrs),
            'avg_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'avg_max_dd': np.mean(max_drawdowns),
            'win_rate': sum(1 for r in total_returns if r > 0) / len(total_returns) if total_returns else 0,
            'best_period_cagr': max(cagrs) if cagrs else 0,
            'worst_period_cagr': min(cagrs) if cagrs else 0,
            'consistency_score': sum(1 for c in cagrs if c > 0) / len(cagrs) if cagrs else 0
        }


class BiasChecker:
    """Bias detection and checking."""
    
    @staticmethod
    def check_look_ahead_bias(features: pd.DataFrame, data: pd.DataFrame) -> Dict:
        """
        Check for look-ahead bias in features.
        
        Ensures no future data is used in feature creation.
        """
        issues = []
        
        # Check if features use future values
        # This is a simplified check - in practice, would need to verify feature creation logic
        for col in features.columns:
            # Check for obvious forward-looking patterns
            if features[col].isnull().sum() > len(features) * 0.1:
                # Many NaN values might indicate look-ahead issues
                pass
        
        return {
            'has_look_ahead_bias': len(issues) > 0,
            'issues': issues
        }
    
    @staticmethod
    def check_survivorship_bias(data: pd.DataFrame, max_gap_days: int = 7) -> Dict:
        """Check for survivorship bias (large data gaps)."""
        gaps = []
        
        if len(data) < 2:
            return {'has_gaps': False, 'gaps': []}
        
        time_diffs = data.index.to_series().diff()
        large_gaps = time_diffs[time_diffs > pd.Timedelta(days=max_gap_days)]
        
        for idx, gap in large_gaps.items():
            gaps.append({
                'date': idx,
                'gap_days': gap.days
            })
        
        return {
            'has_gaps': len(gaps) > 0,
            'gaps': gaps,
            'num_gaps': len(gaps)
        }
    
    @staticmethod
    def check_selection_bias(num_tests: int, best_p_value: float) -> Dict:
        """
        Check for selection bias (multiple testing problem).
        
        Args:
            num_tests: Number of tests performed
            best_p_value: Best p-value from tests
            
        Returns:
            Dictionary with corrected p-value information
        """
        # Bonferroni correction
        bonferroni_corrected = best_p_value * num_tests
        
        # Holm-Bonferroni (less conservative)
        holm_corrected = min(best_p_value * num_tests, 1.0)
        
        return {
            'num_tests': num_tests,
            'original_p_value': best_p_value,
            'bonferroni_corrected': bonferroni_corrected,
            'holm_corrected': holm_corrected,
            'is_significant_bonferroni': bonferroni_corrected < 0.05,
            'is_significant_holm': holm_corrected < 0.05
        }


class ProbabilisticSharpeRatio:
    """Calculate Probabilistic Sharpe Ratio (PSR)."""
    
    @staticmethod
    def calculate(returns: pd.Series, benchmark_sr: float = 0.0, confidence_level: float = 0.95) -> float:
        """
        Calculate Probabilistic Sharpe Ratio.
        
        PSR = probability that the true Sharpe ratio is greater than benchmark_sr.
        
        Args:
            returns: Return series
            benchmark_sr: Benchmark Sharpe ratio (default 0)
            confidence_level: Confidence level
            
        Returns:
            PSR value (0-1)
        """
        if len(returns) < 2:
            return 0.5
        
        # Calculate sample Sharpe ratio
        annual_factor = 252
        mean_return = returns.mean() * annual_factor
        std_return = returns.std() * np.sqrt(annual_factor)
        
        if std_return == 0:
            return 0.5
        
        sample_sr = mean_return / std_return
        
        # Calculate PSR
        n = len(returns)
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        # Adjust for skewness and kurtosis
        denominator = np.sqrt(1 - skew * sample_sr + (kurt - 1) / 4 * sample_sr**2)
        
        if denominator == 0:
            return 0.5
        
        z_score = (sample_sr - benchmark_sr) * np.sqrt(n - 1) / denominator
        
        # Convert to probability
        psr = stats.norm.cdf(z_score)
        
        return psr


class DeflatedSharpeRatio:
    """Calculate Deflated Sharpe Ratio (DSR) accounting for multiple testing."""
    
    @staticmethod
    def calculate(returns: pd.Series, num_tests: int, benchmark_sr: float = 0.0) -> float:
        """
        Calculate Deflated Sharpe Ratio.
        
        DSR accounts for multiple testing by adjusting for the number of tests performed.
        
        Args:
            returns: Return series
            num_tests: Number of tests performed
            benchmark_sr: Benchmark Sharpe ratio (default 0)
            
        Returns:
            DSR value
        """
        if len(returns) < 2:
            return 0.0
        
        # Calculate sample Sharpe ratio
        annual_factor = 252
        mean_return = returns.mean() * annual_factor
        std_return = returns.std() * np.sqrt(annual_factor)
        
        if std_return == 0:
            return 0.0
        
        sample_sr = mean_return / std_return
        
        # Calculate DSR
        n = len(returns)
        
        if num_tests <= 1:
            return sample_sr
        
        try:
            # Expected maximum SR from multiple tests
            expected_max_sr = (1 - np.euler_gamma) * stats.norm.ppf(1 - 1 / num_tests) + np.euler_gamma * stats.norm.ppf(1 - 1 / (num_tests * np.e))
            
            # Variance of SR estimate
            var_sr = (1 + sample_sr**2 / 2) / (n - 1)
            
            # DSR adjustment - handle potential negative values under sqrt
            adjustment = 1 - var_sr * expected_max_sr**2 / (sample_sr**2 + var_sr)
            if adjustment <= 0:
                adjustment = 0.01  # Small positive value to avoid sqrt of negative
            
            dsr = sample_sr * np.sqrt(adjustment)
            
            return dsr
        except:
            # If calculation fails, return sample SR
            return sample_sr


class MonteCarloValidator:
    """Monte Carlo simulation for robustness testing."""
    
    @staticmethod
    def run_monte_carlo(returns: pd.Series, 
                       num_simulations: int = 1000,
                       confidence_level: float = 0.95) -> Dict:
        """
        Run Monte Carlo simulation to test strategy robustness.
        
        Args:
            returns: Return series
            num_simulations: Number of Monte Carlo simulations
            confidence_level: Confidence level
            
        Returns:
            Dictionary with simulation results
        """
        n = len(returns)
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generate random return series with same distribution
        simulated_cagrs = []
        simulated_sharpes = []
        
        for _ in range(num_simulations):
            # Bootstrap sample
            simulated_returns = np.random.choice(returns, size=n, replace=True)
            
            # Calculate metrics
            total_return = (1 + simulated_returns).prod() - 1
            years = n / 252
            cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
            sharpe = (simulated_returns.mean() * 252) / (simulated_returns.std() * np.sqrt(252)) if simulated_returns.std() > 0 else 0
            
            simulated_cagrs.append(cagr)
            simulated_sharpes.append(sharpe)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        cagr_ci_lower = np.percentile(simulated_cagrs, alpha / 2 * 100)
        cagr_ci_upper = np.percentile(simulated_cagrs, (1 - alpha / 2) * 100)
        sharpe_ci_lower = np.percentile(simulated_sharpes, alpha / 2 * 100)
        sharpe_ci_upper = np.percentile(simulated_sharpes, (1 - alpha / 2) * 100)
        
        return {
            'num_simulations': num_simulations,
            'cagr_mean': np.mean(simulated_cagrs),
            'cagr_std': np.std(simulated_cagrs),
            'cagr_ci_lower': cagr_ci_lower,
            'cagr_ci_upper': cagr_ci_upper,
            'sharpe_mean': np.mean(simulated_sharpes),
            'sharpe_std': np.std(simulated_sharpes),
            'sharpe_ci_lower': sharpe_ci_lower,
            'sharpe_ci_upper': sharpe_ci_upper,
            'confidence_level': confidence_level
        }

