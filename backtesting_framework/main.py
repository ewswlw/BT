#!/usr/bin/env python3
"""
Main execution script for the modular backtesting framework.
This script can be run from any directory.
"""

import argparse
import sys
import yaml
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from datetime import datetime

# Try to import additional libraries for enhanced statistics
try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False
    print("Warning: quantstats not available, using manual calculations")

try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    print("Warning: vectorbt not available, using manual calculations")

# Setup paths for running from any directory
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

# Change working directory to the framework directory
os.chdir(script_dir)

from core import (
    DataLoader, CSVDataProvider, 
    TechnicalFeatureEngineer, CrossAssetFeatureEngineer, MultiAssetFeatureEngineer,
    PortfolioEngine, MetricsCalculator, ReportGenerator,
    create_config_from_dict
)
from strategies import StrategyFactory


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)


def calculate_comprehensive_stats(result, benchmark_data, strategy_name):
    """Calculate comprehensive statistics for a strategy."""
    stats = {
        'strategy_name': strategy_name,
        'start_period': result.returns.index[0].strftime('%Y-%m-%d'),
        'end_period': result.returns.index[-1].strftime('%Y-%m-%d'),
        'total_periods': len(result.returns),
        'time_in_market': result.time_in_market,
        'trades_count': result.trades_count
    }
    
    # Calculate basic metrics
    strategy_returns = result.returns.dropna()
    benchmark_returns = benchmark_data.pct_change().dropna()
    
    # Align periods
    common_index = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_returns = strategy_returns.reindex(common_index)
    benchmark_returns = benchmark_returns.reindex(common_index)
    
    # Basic return metrics
    stats['total_return'] = (result.equity_curve.iloc[-1] / result.equity_curve.iloc[0]) - 1
    stats['benchmark_total_return'] = (benchmark_data.iloc[-1] / benchmark_data.iloc[0]) - 1
    
    # Annualized metrics (weekly data)
    annual_factor = 52
    stats['cagr'] = (1 + stats['total_return']) ** (annual_factor / len(strategy_returns)) - 1
    stats['benchmark_cagr'] = (1 + stats['benchmark_total_return']) ** (annual_factor / len(benchmark_returns)) - 1
    
    # Volatility
    stats['volatility'] = strategy_returns.std() * np.sqrt(annual_factor)
    stats['benchmark_volatility'] = benchmark_returns.std() * np.sqrt(annual_factor)
    
    # Sharpe ratio
    stats['sharpe'] = (strategy_returns.mean() * annual_factor) / stats['volatility'] if stats['volatility'] > 0 else 0
    stats['benchmark_sharpe'] = (benchmark_returns.mean() * annual_factor) / stats['benchmark_volatility'] if stats['benchmark_volatility'] > 0 else 0
    
    # Sortino ratio
    downside_strategy = strategy_returns[strategy_returns < 0]
    downside_benchmark = benchmark_returns[benchmark_returns < 0]
    
    downside_std_strategy = downside_strategy.std() * np.sqrt(annual_factor) if len(downside_strategy) > 0 else 0
    downside_std_benchmark = downside_benchmark.std() * np.sqrt(annual_factor) if len(downside_benchmark) > 0 else 0
    
    stats['sortino'] = (strategy_returns.mean() * annual_factor) / downside_std_strategy if downside_std_strategy > 0 else 0
    stats['benchmark_sortino'] = (benchmark_returns.mean() * annual_factor) / downside_std_benchmark if downside_std_benchmark > 0 else 0
    
    # Drawdown calculations
    rolling_max_strategy = result.equity_curve.expanding().max()
    drawdown_strategy = (result.equity_curve - rolling_max_strategy) / rolling_max_strategy
    
    rolling_max_benchmark = benchmark_data.expanding().max()
    drawdown_benchmark = (benchmark_data - rolling_max_benchmark) / rolling_max_benchmark
    
    stats['max_drawdown'] = drawdown_strategy.min()
    stats['benchmark_max_drawdown'] = drawdown_benchmark.min()
    
    # Additional metrics
    stats['skew'] = strategy_returns.skew()
    stats['benchmark_skew'] = benchmark_returns.skew()
    
    stats['kurtosis'] = strategy_returns.kurtosis()
    stats['benchmark_kurtosis'] = benchmark_returns.kurtosis()
    
    # Expected returns
    stats['expected_daily'] = strategy_returns.mean() / 5  # Weekly to daily
    stats['expected_weekly'] = strategy_returns.mean()
    stats['expected_monthly'] = strategy_returns.mean() * 4.33  # Weekly to monthly
    stats['expected_yearly'] = strategy_returns.mean() * annual_factor
    
    stats['benchmark_expected_daily'] = benchmark_returns.mean() / 5
    stats['benchmark_expected_weekly'] = benchmark_returns.mean()
    stats['benchmark_expected_monthly'] = benchmark_returns.mean() * 4.33
    stats['benchmark_expected_yearly'] = benchmark_returns.mean() * annual_factor
    
    # Risk metrics
    stats['daily_var'] = np.percentile(strategy_returns / 5, 5)  # 5% VaR daily approximation
    stats['benchmark_daily_var'] = np.percentile(benchmark_returns / 5, 5)
    
    stats['cvar'] = strategy_returns[strategy_returns <= stats['daily_var'] * 5].mean() / 5  # Expected shortfall
    stats['benchmark_cvar'] = benchmark_returns[benchmark_returns <= stats['benchmark_daily_var'] * 5].mean() / 5
    
    # Calculate VectorBT stats if available
    vectorbt_stats = {}
    vectorbt_returns_stats = {}
    
    if VECTORBT_AVAILABLE and hasattr(result, 'portfolio') and result.portfolio is not None:
        try:
            vectorbt_stats = result.portfolio.stats().to_dict()
            vectorbt_returns_stats = result.portfolio.returns().vbt.returns.stats().to_dict()
        except Exception as e:
            print(f"Warning: Could not calculate VectorBT stats: {e}")
    
    # Calculate QuantStats metrics if available
    quantstats_metrics = {}
    if QUANTSTATS_AVAILABLE:
        try:
            # Set up returns for quantstats
            qs_strategy_returns = strategy_returns.dropna()
            qs_benchmark_returns = benchmark_returns.dropna()
            
            # Helper function to safely extract values
            def safe_qs_metric(func, returns_series):
                try:
                    result = func(returns_series)
                    # Handle different return types from QuantStats
                    if hasattr(result, 'values'):
                        return float(result.values[0]) if len(result.values) > 0 else 0.0
                    elif isinstance(result, (int, float, np.number)):
                        return float(result)
                    else:
                        return 0.0
                except Exception as e:
                    print(f"Warning: QuantStats metric failed: {func.__name__}: {e}")
                    return 0.0
            
            # Calculate various QuantStats metrics with safe extraction
            quantstats_metrics = {
                'prob_sharpe_ratio': safe_qs_metric(qs.stats.probabilistic_sharpe_ratio, qs_strategy_returns),
                'smart_sharpe': safe_qs_metric(qs.stats.smart_sharpe, qs_strategy_returns),
                'smart_sortino': safe_qs_metric(qs.stats.smart_sortino, qs_strategy_returns),
                'omega': safe_qs_metric(qs.stats.omega, qs_strategy_returns),
                'calmar': safe_qs_metric(qs.stats.calmar, qs_strategy_returns),
                'gain_pain_ratio': safe_qs_metric(qs.stats.gain_to_pain_ratio, qs_strategy_returns),
                'payoff_ratio': safe_qs_metric(qs.stats.payoff_ratio, qs_strategy_returns),
                'profit_factor': safe_qs_metric(qs.stats.profit_factor, qs_strategy_returns),
                'tail_ratio': safe_qs_metric(qs.stats.tail_ratio, qs_strategy_returns),
                'common_sense_ratio': safe_qs_metric(qs.stats.common_sense_ratio, qs_strategy_returns),
                'longest_dd_days': safe_qs_metric(qs.stats.max_drawdown_duration, qs_strategy_returns),
                
                # Benchmark metrics
                'benchmark_prob_sharpe_ratio': safe_qs_metric(qs.stats.probabilistic_sharpe_ratio, qs_benchmark_returns),
                'benchmark_smart_sharpe': safe_qs_metric(qs.stats.smart_sharpe, qs_benchmark_returns),
                'benchmark_smart_sortino': safe_qs_metric(qs.stats.smart_sortino, qs_benchmark_returns),
                'benchmark_omega': safe_qs_metric(qs.stats.omega, qs_benchmark_returns),
                'benchmark_calmar': safe_qs_metric(qs.stats.calmar, qs_benchmark_returns),
                'benchmark_gain_pain_ratio': safe_qs_metric(qs.stats.gain_to_pain_ratio, qs_benchmark_returns),
                'benchmark_payoff_ratio': safe_qs_metric(qs.stats.payoff_ratio, qs_benchmark_returns),
                'benchmark_profit_factor': safe_qs_metric(qs.stats.profit_factor, qs_benchmark_returns),
                'benchmark_tail_ratio': safe_qs_metric(qs.stats.tail_ratio, qs_benchmark_returns),
                'benchmark_common_sense_ratio': safe_qs_metric(qs.stats.common_sense_ratio, qs_benchmark_returns),
                'benchmark_longest_dd_days': safe_qs_metric(qs.stats.max_drawdown_duration, qs_benchmark_returns),
            }
        except Exception as e:
            print(f"Warning: Could not calculate QuantStats metrics: {e}")
    
    return {
        'basic_stats': stats,
        'vectorbt_stats': vectorbt_stats,
        'vectorbt_returns_stats': vectorbt_returns_stats,
        'quantstats_metrics': quantstats_metrics
    }


def generate_strategy_description_from_config(strategy_name, strategy_config):
    """Generate trading rules description from strategy configuration."""
    
    if strategy_name.lower() == 'cross_asset_momentum':
        momentum_assets = strategy_config.get('momentum_assets', ['cad_ig_er_index', 'us_hy_er_index', 'us_ig_er_index', 'tsx'])
        momentum_lookback_weeks = strategy_config.get('momentum_lookback_weeks', 2)
        min_confirmations = strategy_config.get('min_confirmations', 3)
        
        return f"""Cross-Asset {momentum_lookback_weeks}-Week Momentum Strategy

TRADING RULES:
1. Signal Generation:
   - Calculate {momentum_lookback_weeks}-week momentum for {len(momentum_assets)} indices
   - Momentum = (Current Price / Price {momentum_lookback_weeks} weeks ago) - 1
   - Assets monitored: {', '.join(momentum_assets)}

2. Entry Condition:
   - Enter LONG when >={min_confirmations} of {len(momentum_assets)} indices show positive {momentum_lookback_weeks}-week momentum
   - Must meet confirmation threshold: {min_confirmations}/{len(momentum_assets)} assets

3. Exit Condition:
   - Exit when signal condition is no longer met
   - Exit when <{min_confirmations} of {len(momentum_assets)} assets show positive momentum

4. Position Management:
   - Long-only, no short positions
   - Full capital deployment when signal is active
   - Cash position when signal is inactive
   - Weekly rebalancing on Friday close

5. Risk Management:
   - No leverage, no margin
   - No stop losses or take profits
   - Position sizing: 100% of capital when invested

PARAMETERS:
- Momentum Lookback: {momentum_lookback_weeks} weeks
- Minimum Confirmations: {min_confirmations} of {len(momentum_assets)}
- Assets: {momentum_assets}
- Frequency: Weekly (Friday close)
- Universe: Multi-asset cross-confirmation"""

    elif strategy_name.lower() == 'multi_asset_momentum':
        momentum_assets_map = strategy_config.get('momentum_assets_map', {
            'tsx': 'tsx', 'us_hy': 'us_hy_er_index', 'cad_ig': 'cad_ig_er_index'
        })
        momentum_lookback_periods = strategy_config.get('momentum_lookback_periods', 4)
        signal_threshold = strategy_config.get('signal_threshold', -0.005)
        exit_signal_shift_periods = strategy_config.get('exit_signal_shift_periods', 1)
        
        return f"""Multi-Asset {momentum_lookback_periods}-Period Momentum Strategy

TRADING RULES:
1. Momentum Calculation:
   - Calculate {momentum_lookback_periods}-period momentum for each asset
   - Individual Momentum = (Current Price / Price {momentum_lookback_periods} periods ago) - 1
   - Assets: {list(momentum_assets_map.keys())} → {list(momentum_assets_map.values())}

2. Signal Combination:
   - Average individual momentums across all assets
   - Combined Momentum = Sum(Individual Momentums) / Number of Assets
   - Creates single composite momentum signal

3. Entry Condition:
   - Enter LONG when Combined Momentum > {signal_threshold:.3f}
   - Signal threshold allows slightly negative momentum for entry

4. Exit Condition:
   - Exit after {exit_signal_shift_periods} period(s) holding time
   - OR exit when Combined Momentum ≤ {signal_threshold:.3f}

5. Position Management:
   - Long-only strategy
   - Binary position: 100% invested or 100% cash
   - Weekly rebalancing

PARAMETERS:
- Momentum Lookback: {momentum_lookback_periods} periods
- Signal Threshold: {signal_threshold:.3f}
- Exit Shift: {exit_signal_shift_periods} periods
- Assets Map: {momentum_assets_map}
- Combination Method: Equal-weighted average"""

    elif strategy_name.lower() == 'genetic_algorithm':
        population_size = strategy_config.get('population_size', 120)
        max_generations = strategy_config.get('max_generations', 120)
        mutation_rate = strategy_config.get('mutation_rate', 0.40)
        crossover_rate = strategy_config.get('crossover_rate', 0.40)
        max_clauses_per_rule = strategy_config.get('max_clauses_per_rule', 4)
        fitness_drawdown_penalty_factor = strategy_config.get('fitness_drawdown_penalty_factor', 0.1)
        
        return f"""Genetic Algorithm Evolved Trading Strategy

TRADING RULES:
1. Feature Engineering (Technical Indicators):
   - Momentum: 1, 2, 3, 4, 6, 8, 12, 13, 26, 52 week returns
   - Volatility: 4, 8, 13, 26 week rolling volatility
   - SMA Deviation: Price deviation from 4, 8, 13, 26 week moving averages
   - MACD: 12/26 EMA crossover with 9-period signal line
   - Stochastic K: 14-period stochastic oscillator
   - Factor Momentum: 4-week momentum of CAD OAS, US IG OAS, US HY OAS, VIX

2. Rule Evolution Process:
   - Population Size: {population_size} random rule combinations
   - Generations: {max_generations} evolutionary iterations
   - Mutation Rate: {mutation_rate:.0%} chance of rule modification
   - Crossover Rate: {crossover_rate:.0%} chance of rule combination
   - Max Clauses: {max_clauses_per_rule} conditions per rule

3. Fitness Function:
   - Primary: Total Return from rule application
   - Penalty: Drawdown penalty factor = {fitness_drawdown_penalty_factor}
   - Fitness = Return - ({fitness_drawdown_penalty_factor} × Max Drawdown)
   - Selects for return while minimizing risk

4. Rule Structure:
   - Boolean combinations of technical conditions
   - Example: (momentum_4 > 0.02) & (vol_8 < 0.15) | (macd_diff > 0)
   - Complex multi-condition logic evolved automatically

5. Trading Execution:
   - Rules generate binary signals (invested/cash)
   - Long-only positions when rule conditions are met
   - Weekly rebalancing based on rule evaluation

PARAMETERS:
- Evolution: {population_size} population × {max_generations} generations
- Genetic Operators: {mutation_rate:.0%} mutation, {crossover_rate:.0%} crossover
- Rule Complexity: Max {max_clauses_per_rule} clauses per rule
- Fitness Penalty: {fitness_drawdown_penalty_factor} × drawdown
- Feature Universe: 20+ technical indicators"""

    else:
        return f"""Strategy: {strategy_name.upper()}

TRADING RULES:
Configuration-based strategy implementation.
Parameters: {strategy_config}

Note: Detailed trading rules not available for this strategy type.
Please check the strategy implementation for specific logic."""


def generate_comprehensive_stats_file(results, benchmark_data, config_dict):
    """Generate comprehensive statistics file for all strategies."""
    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    enhanced_config = config_dict.get('enhanced_reporting', {})
    file_name = enhanced_config.get('stats_file_name', 'comprehensive_strategy_comparison.txt')
    file_path = output_dir / file_name
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("COMPREHENSIVE STRATEGY COMPARISON REPORT\n")
        f.write("="*100 + "\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Total Strategies Analyzed: {len(results)}\n")
        f.write(f"Benchmark Asset: {config_dict.get('data', {}).get('benchmark_asset', 'N/A')}\n")
        f.write("="*100 + "\n\n")
        
        # Calculate comprehensive stats for all strategies
        all_stats = {}
        for strategy_name, result in results.items():
            all_stats[strategy_name] = calculate_comprehensive_stats(result, benchmark_data, strategy_name)
        
        # 1. VECTORBT PORTFOLIO STATS SECTION
        if enhanced_config.get('include_vectorbt_stats', True):
            f.write("="*100 + "\n")
            f.write("1. VECTORBT PORTFOLIO STATS (pf.stats())\n")
            f.write("="*100 + "\n")
            
            for strategy_name, stats in all_stats.items():
                f.write(f"\n{'-'*60}\n")
                f.write(f"{strategy_name.upper()} - VectorBT Portfolio Stats\n")
                f.write(f"{'-'*60}\n")
                
                if stats['vectorbt_stats']:
                    for key, value in stats['vectorbt_stats'].items():
                        if isinstance(value, (int, float)):
                            if 'return' in key.lower() or 'ratio' in key.lower():
                                f.write(f"{key:<30}: {value:>12.4f}\n")
                            else:
                                f.write(f"{key:<30}: {value:>12.2f}\n")
                        else:
                            f.write(f"{key:<30}: {str(value):>12}\n")
                else:
                    f.write("VectorBT stats not available for this strategy\n")
            
            f.write("\n")
        
        # 2. VECTORBT RETURNS STATS SECTION  
        if enhanced_config.get('include_vectorbt_stats', True):
            f.write("="*100 + "\n")
            f.write("2. VECTORBT RETURNS STATS (pf.vbt.returns.stats())\n")
            f.write("="*100 + "\n")
            
            for strategy_name, stats in all_stats.items():
                f.write(f"\n{'-'*60}\n")
                f.write(f"{strategy_name.upper()} - VectorBT Returns Stats\n")
                f.write(f"{'-'*60}\n")
                
                if stats['vectorbt_returns_stats']:
                    for key, value in stats['vectorbt_returns_stats'].items():
                        if isinstance(value, (int, float)):
                            if 'return' in key.lower() or 'ratio' in key.lower():
                                f.write(f"{key:<30}: {value:>12.4f}\n")
                            else:
                                f.write(f"{key:<30}: {value:>12.2f}\n")
                        else:
                            f.write(f"{key:<30}: {str(value):>12}\n")
                else:
                    f.write("VectorBT returns stats not available for this strategy\n")
            
            f.write("\n")
        
        # 3. MANUAL CALCULATIONS SECTION
        if enhanced_config.get('include_manual_calculations', True):
            f.write("="*100 + "\n")
            f.write("3. MANUAL PORTFOLIO CALCULATIONS\n")
            f.write("="*100 + "\n")
            
            for strategy_name, stats in all_stats.items():
                basic = stats['basic_stats']
                f.write(f"\n{'-'*60}\n")
                f.write(f"{strategy_name.upper()} - Manual Calculations\n")
                f.write(f"{'-'*60}\n")
                f.write(f"{'Metric':<30} {'Strategy':<15} {'Benchmark':<15}\n")
                f.write(f"{'-'*60}\n")
                f.write(f"{'Start Period':<30} {basic['start_period']:<15} {basic['start_period']:<15}\n")
                f.write(f"{'End Period':<30} {basic['end_period']:<15} {basic['end_period']:<15}\n")
                f.write(f"{'Total Periods':<30} {basic['total_periods']:<15} {basic['total_periods']:<15}\n")
                f.write(f"{'Time in Market':<30} {basic['time_in_market']:<15.2%} {'100.0%':<15}\n")
                f.write(f"{'Total Trades':<30} {basic['trades_count']:<15} {'N/A':<15}\n")
                f.write(f"\n")
                f.write(f"{'Total Return':<30} {basic['total_return']:<15.4f} {basic['benchmark_total_return']:<15.4f}\n")
                f.write(f"{'CAGR':<30} {basic['cagr']:<15.4f} {basic['benchmark_cagr']:<15.4f}\n")
                f.write(f"{'Volatility (ann.)':<30} {basic['volatility']:<15.4f} {basic['benchmark_volatility']:<15.4f}\n")
                f.write(f"{'Sharpe Ratio':<30} {basic['sharpe']:<15.4f} {basic['benchmark_sharpe']:<15.4f}\n")
                f.write(f"{'Sortino Ratio':<30} {basic['sortino']:<15.4f} {basic['benchmark_sortino']:<15.4f}\n")
                f.write(f"{'Max Drawdown':<30} {basic['max_drawdown']:<15.4f} {basic['benchmark_max_drawdown']:<15.4f}\n")
                f.write(f"{'Skewness':<30} {basic['skew']:<15.4f} {basic['benchmark_skew']:<15.4f}\n")
                f.write(f"{'Kurtosis':<30} {basic['kurtosis']:<15.4f} {basic['benchmark_kurtosis']:<15.4f}\n")
                f.write(f"\n")
            
        # 4. QUANTSTATS STYLE COMPARISON
        f.write("="*100 + "\n")
        f.write("4. QUANTSTATS STYLE COMPREHENSIVE COMPARISON\n")
        f.write("="*100 + "\n")
        
        # Create comparison table for all strategies
        benchmark_stats = all_stats[list(all_stats.keys())[0]]['basic_stats']  # Use first strategy's benchmark data
        
        for strategy_name, stats in all_stats.items():
            basic = stats['basic_stats']
            qs_metrics = stats['quantstats_metrics']
            
            f.write(f"\n{'-'*80}\n")
            f.write(f"{strategy_name.upper()} vs BENCHMARK\n")
            f.write(f"{'-'*80}\n")
            f.write(f"{'Metric':<35} {'Benchmark':<15} {'Strategy':<15}\n")
            f.write(f"{'-'*65}\n")
            
            # Time period info
            f.write(f"{'Start Period':<35} {basic['start_period']:<15} {basic['start_period']:<15}\n")
            f.write(f"{'End Period':<35} {basic['end_period']:<15} {basic['end_period']:<15}\n")
            f.write(f"{'Risk-Free Rate':<35} {'0.0%':<15} {'0.0%':<15}\n")
            time_in_market_str = f"{basic['time_in_market']:.1%}"
            f.write(f"{'Time in Market':<35} {'100.0%':<15} {time_in_market_str:<15}\n")
            f.write(f"\n")
            
            # Returns
            f.write(f"{'Cumulative Return':<35} {basic['benchmark_total_return']:>14.2%} {basic['total_return']:>14.2%}\n")
            f.write(f"{'CAGR%':<35} {basic['benchmark_cagr']:>14.2%} {basic['cagr']:>14.2%}\n")
            f.write(f"\n")
            
            # Risk-adjusted metrics
            f.write(f"{'Sharpe':<35} {basic['benchmark_sharpe']:>14.2f} {basic['sharpe']:>14.2f}\n")
            
            if qs_metrics:
                f.write(f"{'Prob. Sharpe Ratio':<35} {qs_metrics.get('benchmark_prob_sharpe_ratio', 0)*100:>13.2f}% {qs_metrics.get('prob_sharpe_ratio', 0)*100:>13.2f}%\n")
                f.write(f"{'Smart Sharpe':<35} {qs_metrics.get('benchmark_smart_sharpe', 0):>14.2f} {qs_metrics.get('smart_sharpe', 0):>14.2f}\n")
            
            f.write(f"{'Sortino':<35} {basic['benchmark_sortino']:>14.2f} {basic['sortino']:>14.2f}\n")
            
            if qs_metrics:
                f.write(f"{'Smart Sortino':<35} {qs_metrics.get('benchmark_smart_sortino', 0):>14.2f} {qs_metrics.get('smart_sortino', 0):>14.2f}\n")
                f.write(f"{'Omega':<35} {qs_metrics.get('benchmark_omega', 0):>14.2f} {qs_metrics.get('omega', 0):>14.2f}\n")
            
            f.write(f"\n")
            
            # Risk metrics
            f.write(f"{'Max Drawdown':<35} {basic['benchmark_max_drawdown']:>14.2%} {basic['max_drawdown']:>14.2%}\n")
            
            if qs_metrics:
                f.write(f"{'Longest DD Days':<35} {qs_metrics.get('benchmark_longest_dd_days', 0):>14.0f} {qs_metrics.get('longest_dd_days', 0):>14.0f}\n")
            
            f.write(f"{'Volatility (ann.)':<35} {basic['benchmark_volatility']:>14.2%} {basic['volatility']:>14.2%}\n")
            
            if qs_metrics:
                f.write(f"{'Calmar':<35} {qs_metrics.get('benchmark_calmar', 0):>14.2f} {qs_metrics.get('calmar', 0):>14.2f}\n")
            
            f.write(f"{'Skew':<35} {basic['benchmark_skew']:>14.2f} {basic['skew']:>14.2f}\n")
            f.write(f"{'Kurtosis':<35} {basic['benchmark_kurtosis']:>14.2f} {basic['kurtosis']:>14.2f}\n")
            f.write(f"\n")
            
            # Expected returns
            f.write(f"{'Expected Daily %':<35} {basic['benchmark_expected_daily']:>14.2%} {basic['expected_daily']:>14.2%}\n")
            f.write(f"{'Expected Weekly %':<35} {basic['benchmark_expected_weekly']:>14.2%} {basic['expected_weekly']:>14.2%}\n")
            f.write(f"{'Expected Monthly %':<35} {basic['benchmark_expected_monthly']:>14.2%} {basic['expected_monthly']:>14.2%}\n")
            f.write(f"{'Expected Yearly %':<35} {basic['benchmark_expected_yearly']:>14.2%} {basic['expected_yearly']:>14.2%}\n")
            f.write(f"{'Daily Value-at-Risk':<35} {basic['benchmark_daily_var']:>14.2%} {basic['daily_var']:>14.2%}\n")
            f.write(f"{'Expected Shortfall (cVaR)':<35} {basic['benchmark_cvar']:>14.2%} {basic['cvar']:>14.2%}\n")
            f.write(f"\n")
            
            # Additional metrics from QuantStats
            if qs_metrics:
                f.write(f"{'Gain/Pain Ratio':<35} {qs_metrics.get('benchmark_gain_pain_ratio', 0):>14.2f} {qs_metrics.get('gain_pain_ratio', 0):>14.2f}\n")
                f.write(f"{'Payoff Ratio':<35} {qs_metrics.get('benchmark_payoff_ratio', 0):>14.2f} {qs_metrics.get('payoff_ratio', 0):>14.2f}\n")
                f.write(f"{'Profit Factor':<35} {qs_metrics.get('benchmark_profit_factor', 0):>14.2f} {qs_metrics.get('profit_factor', 0):>14.2f}\n")
                f.write(f"{'Common Sense Ratio':<35} {qs_metrics.get('benchmark_common_sense_ratio', 0):>14.2f} {qs_metrics.get('common_sense_ratio', 0):>14.2f}\n")
                f.write(f"{'Tail Ratio':<35} {qs_metrics.get('benchmark_tail_ratio', 0):>14.1f} {qs_metrics.get('tail_ratio', 0):>14.1f}\n")
            
            f.write(f"\n")
        
        # Summary comparison table of all strategies
        f.write("="*100 + "\n")
        f.write("5. STRATEGY SUMMARY COMPARISON\n")
        f.write("="*100 + "\n")
        
        f.write(f"{'Strategy':<25} {'Total Return':<15} {'CAGR':<10} {'Sharpe':<10} {'Sortino':<10} {'Max DD':<12} {'Volatility':<12}\n")
        f.write(f"{'-'*94}\n")
        
        # Benchmark row
        bench_stats = benchmark_stats
        f.write(f"{'BENCHMARK':<25} {bench_stats['benchmark_total_return']:>14.2%} {bench_stats['benchmark_cagr']:>9.2%} {bench_stats['benchmark_sharpe']:>9.2f} {bench_stats['benchmark_sortino']:>9.2f} {bench_stats['benchmark_max_drawdown']:>11.2%} {bench_stats['benchmark_volatility']:>11.2%}\n")
        
        # Strategy rows
        for strategy_name, stats in all_stats.items():
            basic = stats['basic_stats']
            f.write(f"{strategy_name.upper():<25} {basic['total_return']:>14.2%} {basic['cagr']:>9.2%} {basic['sharpe']:>9.2f} {basic['sortino']:>9.2f} {basic['max_drawdown']:>11.2%} {basic['volatility']:>11.2%}\n")
        
        f.write(f"\n")
        
        # 6. TRADING RULES SECTION
        f.write("="*100 + "\n")
        f.write("6. DYNAMIC TRADING RULES\n")
        f.write("="*100 + "\n")
        f.write("Detailed trading rules and parameters for each strategy (extracted dynamically from code):\n\n")
        
        for strategy_name, stats in all_stats.items():
            # Try to get trading rules from the result object
            try:
                # Get the strategy object from results to extract trading rules
                result = results[strategy_name]
                if hasattr(result, 'config') and 'strategy' in result.config:
                    # Try to recreate strategy to get description
                    strategy_config = result.config.get('strategy', {})
                    strategy_config['type'] = strategy_name
                    strategy_config['name'] = strategy_name
                    
                    try:
                        from strategies import StrategyFactory
                        strategy_obj = StrategyFactory.create_strategy(strategy_config)
                        
                        if hasattr(strategy_obj, 'get_strategy_description'):
                            description = strategy_obj.get_strategy_description()
                        else:
                            description = generate_strategy_description_from_config(strategy_name, strategy_config)
                    except:
                        description = generate_strategy_description_from_config(strategy_name, config_dict.get(strategy_name.lower(), {}))
                else:
                    description = generate_strategy_description_from_config(strategy_name, config_dict.get(strategy_name.lower(), {}))
                    
            except Exception as e:
                description = generate_strategy_description_from_config(strategy_name, config_dict.get(strategy_name.lower(), {}))
            
            f.write(f"{'-'*80}\n")
            f.write(f"{strategy_name.upper()} TRADING RULES\n")
            f.write(f"{'-'*80}\n")
            f.write(description)
            f.write(f"\n\n")
        
        f.write("="*100 + "\n")
        f.write("IMPORTANT NOTES\n")
        f.write("="*100 + "\n")
        f.write("* Because risk-free rate is assumed at 0% to be conservative, Sharpe ratio is\n")
        f.write("  really just a measure of absolute return per unit of risk.\n")
        f.write("* All calculations assume weekly rebalancing on Friday close.\n")
        f.write("* No transaction costs, slippage, or taxes are included in calculations.\n")
        f.write("* Results represent theoretical performance and may not be achievable in practice.\n")
        f.write("* Past performance does not guarantee future results.\n")
        f.write("\n")
        f.write("="*100 + "\n")
        f.write("END OF COMPREHENSIVE STRATEGY COMPARISON REPORT\n")
        f.write("="*100 + "\n")
    
    print(f"\n✓ Comprehensive statistics file saved: {file_path}")
    return str(file_path)


def run_single_strategy(strategy_name: str, config_dict: dict):
    """Run a single strategy."""
    print(f"\n{'='*80}")
    print(f"RUNNING STRATEGY: {strategy_name.upper()}")
    print(f"{'='*80}")
    
    # Set random seed (matching reference implementations)
    import random
    import numpy as np
    random_seed = config_dict.get('random_seed', 7)
    random.seed(random_seed)
    np.random.seed(random_seed)
    print(f"Random seed set to: {random_seed}")
    
    try:
        # Create configuration objects
        base_config = create_config_from_dict(config_dict)
        
        # Load and prepare data
        print("Loading data...")
        data_loader = DataLoader(CSVDataProvider())
        data = data_loader.load_and_prepare(base_config.data)
        
        print(f"Data loaded. Period: {data.index[0]} to {data.index[-1]}")
        print(f"Total periods: {len(data)}")
        
        # Create features based on strategy type
        print("Engineering features...")
        if strategy_name.lower() == 'cross_asset_momentum':
            feature_engineer = CrossAssetFeatureEngineer()
            feature_config = {
                **config_dict.get('features', {}),
                **config_dict.get('cross_asset_momentum', {})
            }
        elif strategy_name.lower() == 'multi_asset_momentum':
            feature_engineer = MultiAssetFeatureEngineer()
            feature_config = {
                **config_dict.get('features', {}),
                **config_dict.get('multi_asset_momentum', {})
            }
        else:
            feature_engineer = TechnicalFeatureEngineer()
            feature_config = config_dict.get('features', {})
        
        features = feature_engineer.create_features(data, feature_config)
        print(f"Features created: {len(features.columns)} features")
        
        # Create strategy
        strategy_config = {
            'type': strategy_name,
            'name': strategy_name,
            **config_dict.get(strategy_name.lower(), {}),
            **config_dict.get('features', {}),
            'random_seed': config_dict.get('random_seed', 7)
        }
        
        strategy = StrategyFactory.create_strategy(strategy_config)
        print(f"Strategy created: {strategy.name}")
        
        # Run backtest
        print("Running backtest...")
        result = strategy.backtest(data, features, base_config.portfolio, base_config.data.benchmark_asset)
        
        # Generate report
        print("Generating report...")
        report_generator = ReportGenerator(base_config.reporting.output_dir)
        benchmark_data = data[base_config.data.benchmark_asset]
        
        report_text, html_path = report_generator.generate_strategy_report(
            result, benchmark_data, base_config.reporting.generate_html
        )
        
        # Print report
        print(report_text)
        
        # Save artifacts
        artifacts = report_generator.save_artifacts(result)
        print(f"\nArtifacts saved:")
        for artifact_type, path in artifacts.items():
            print(f"  {artifact_type}: {path}")
        
        if html_path:
            print(f"\nHTML Report: {html_path}")
            
    except Exception as e:
        print(f"Error running strategy {strategy_name}: {e}")
        import traceback
        traceback.print_exc()


def run_strategy_comparison(strategy_names: list, config_dict: dict):
    """Run and compare multiple strategies."""
    print(f"\n{'='*100}")
    print(f"RUNNING STRATEGY COMPARISON")
    print(f"Strategies: {', '.join(strategy_names)}")
    print(f"{'='*100}")
    
    try:
        # Create configuration objects
        base_config = create_config_from_dict(config_dict)
        
        # Load and prepare data
        print("Loading data...")
        data_loader = DataLoader(CSVDataProvider())
        data = data_loader.load_and_prepare(base_config.data)
        
        print(f"Data loaded. Period: {data.index[0]} to {data.index[-1]}")
        print(f"Total periods: {len(data)}")
        
        results = {}
        
        for strategy_name in strategy_names:
            print(f"\n{'-'*60}")
            print(f"Running {strategy_name}...")
            print(f"{'-'*60}")
            
            try:
                # Create features based on strategy type
                if strategy_name.lower() == 'cross_asset_momentum':
                    feature_engineer = CrossAssetFeatureEngineer()
                    feature_config = {
                        **config_dict.get('features', {}),
                        **config_dict.get('cross_asset_momentum', {})
                    }
                elif strategy_name.lower() == 'multi_asset_momentum':
                    feature_engineer = MultiAssetFeatureEngineer()
                    feature_config = {
                        **config_dict.get('features', {}),
                        **config_dict.get('multi_asset_momentum', {})
                    }
                else:
                    feature_engineer = TechnicalFeatureEngineer()
                    feature_config = config_dict.get('features', {})
                
                features = feature_engineer.create_features(data, feature_config)
                
                # Create strategy
                strategy_config = {
                    'type': strategy_name,
                    'name': strategy_name,
                    **config_dict.get(strategy_name.lower(), {}),
                    **config_dict.get('features', {}),
                    'random_seed': config_dict.get('random_seed', 7)
                }
                
                strategy = StrategyFactory.create_strategy(strategy_config)
                
                # Run backtest
                result = strategy.backtest(data, features, base_config.portfolio, base_config.data.benchmark_asset)
                results[strategy_name] = result
                
                print(f"✓ {strategy_name} completed")
                print(f"  Total Return: {result.metrics['total_return']:.2%}")
                print(f"  Sharpe Ratio: {result.metrics['sharpe_ratio']:.3f}")
                print(f"  Max Drawdown: {result.metrics['max_drawdown']:.2%}")
                
            except Exception as e:
                print(f"✗ Error running {strategy_name}: {e}")
                continue
        
        if results:
            # Generate comparison report
            print(f"\n{'='*80}")
            print("STRATEGY COMPARISON SUMMARY")
            print(f"{'='*80}")
            
            report_generator = ReportGenerator(base_config.reporting.output_dir)
            benchmark_data = data[base_config.data.benchmark_asset]
            
            comparison_report = report_generator.generate_strategy_comparison_report(
                results, benchmark_data
            )
            print(comparison_report)
            
            # Generate comprehensive statistics file if enabled
            enhanced_config = config_dict.get('enhanced_reporting', {})
            if enhanced_config.get('generate_detailed_stats_file', True):
                try:
                    stats_file_path = generate_comprehensive_stats_file(results, benchmark_data, config_dict)
                    print(f"\n✓ Comprehensive statistics saved to: {stats_file_path}")
                except Exception as e:
                    print(f"✗ Error generating comprehensive statistics file: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Save individual reports
            for strategy_name, result in results.items():
                try:
                    report_text, html_path = report_generator.generate_strategy_report(
                        result, benchmark_data, base_config.reporting.generate_html
                    )
                    artifacts = report_generator.save_artifacts(result)
                    print(f"\n✓ {strategy_name} reports saved")
                except Exception as e:
                    print(f"✗ Error saving {strategy_name} reports: {e}")
                    
    except Exception as e:
        print(f"Error in strategy comparison: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Modular Backtesting Framework")
    parser.add_argument("--config", default="configs/base_config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--strategy", help="Single strategy to run")
    parser.add_argument("--compare", nargs="+", 
                       help="Compare multiple strategies")
    parser.add_argument("--list-strategies", action="store_true",
                       help="List available strategies")
    
    args = parser.parse_args()
    
    # List available strategies
    if args.list_strategies:
        print("Available Strategies:")
        for name, strategy_class in StrategyFactory.get_available_strategies().items():
            print(f"  - {name}: {strategy_class.__name__}")
        return
    
    # Load configuration file (we're now always in the framework directory)
    config_file = Path(args.config)
    if not config_file.exists():
        print(f"Config file not found: {config_file.absolute()}")
        print("Please ensure the config file exists or provide a valid path.")
        print(f"Current working directory: {Path.cwd()}")
        sys.exit(1)
    
    print(f"Using config file: {config_file}")
    
    config_dict = load_config(str(config_file))
    
    # Validate data path (we're now always in the framework directory)
    data_path = Path(config_dict['data']['file_path'])
    if not data_path.exists():
        # Try the original data location as fallback
        fallback_path = Path("../../cad ig er index weekly backtests/data_pipelines/data_processed/with_er_daily.csv")
        if fallback_path.exists():
            config_dict['data']['file_path'] = str(fallback_path)
            data_path = fallback_path
        else:
            print(f"Data file not found: {data_path.absolute()}")
            print("Please update the file_path in the config file or ensure data exists.")
            sys.exit(1)
    
    print(f"Using data file: {data_path}")
    
    # Run based on arguments
    if args.strategy:
        run_single_strategy(args.strategy, config_dict)
    elif args.compare:
        run_strategy_comparison(args.compare, config_dict)
    else:
        # Get strategies from config or use all available
        strategies_config = config_dict.get('strategies', {})
        if strategies_config.get('run_all_by_default', True):
            # Use enabled strategies from config
            enabled_strategies = strategies_config.get('enabled', [])
            if enabled_strategies:
                print(f"Running enabled strategies from config: {', '.join(enabled_strategies)}")
                run_strategy_comparison(enabled_strategies, config_dict)
            else:
                # Fallback to all available strategies
                available_strategies = list(StrategyFactory.get_available_strategies().keys())
                print(f"No enabled strategies in config. Running all available strategies:")
                print(f"Strategies: {', '.join(available_strategies)}")
                run_strategy_comparison(available_strategies, config_dict)
        else:
            print("Strategy execution disabled in config. Use --strategy or --compare to run specific strategies.")


if __name__ == "__main__":
    main() 