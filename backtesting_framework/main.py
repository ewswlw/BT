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
        qs_strategy_returns = strategy_returns.dropna()
        qs_benchmark_returns = benchmark_returns.dropna()

        # Define a list of metrics to calculate
        qs_metric_names = [
            'smart_sharpe', 'smart_sortino', 'calmar', 'payoff_ratio', 
            'profit_factor', 'tail_ratio', 'common_sense_ratio'
        ]

        for metric_name in qs_metric_names:
            try:
                # Get the function from qs.stats
                metric_func = getattr(qs.stats, metric_name)
                
                # Calculate for strategy
                strat_metric = metric_func(qs_strategy_returns)
                quantstats_metrics[metric_name] = float(strat_metric)

                # Calculate for benchmark
                bench_metric = metric_func(qs_benchmark_returns)
                quantstats_metrics[f'benchmark_{metric_name}'] = float(bench_metric)

            except Exception as e:
                print(f"Warning: Could not calculate QuantStats metric '{metric_name}': {e}")
                quantstats_metrics[metric_name] = 0.0
                quantstats_metrics[f'benchmark_{metric_name}'] = 0.0
    
    return {
        'basic_stats': stats,
        'vectorbt_stats': vectorbt_stats,
        'vectorbt_returns_stats': vectorbt_returns_stats,
        'quantstats_metrics': quantstats_metrics
    }


def generate_strategy_description_from_config(strategy_name, strategy_config):
    """Generate trading rules description from strategy configuration."""
    
    if strategy_name.lower() == 'cross_asset_momentum':
        momentum_assets = strategy_config.get('momentum_assets', [])
        momentum_lookback_days = strategy_config.get('momentum_lookback_days', 10)
        min_confirmations = strategy_config.get('min_confirmations', 3)
        trading_asset = strategy_config.get('trading_asset', 'cad_ig_er_index')

        return f"""Cross-Asset {momentum_lookback_days}-Day Momentum Strategy (Weekly Rebalance)

TRADING RULES:
1. Data & Frequency:
   - This strategy operates on DAILY data but only makes trading decisions once per week.
   - Rebalancing Day: Monday.

2. Signal Universe:
   - A basket of {len(momentum_assets)} diverse assets is monitored to gauge market breadth.
   - Assets: {', '.join(momentum_assets)}

3. Momentum Calculation (Monday Only):
   - For each asset, a {momentum_lookback_days}-day momentum is calculated on Monday.
   - Momentum = (Monday's Price / Price {momentum_lookback_days} trading days ago) - 1.

4. Entry Condition (Monday Only):
   - A position is entered in '{trading_asset}' if enough assets show positive momentum.
   - Entry Threshold: LONG when >= {min_confirmations} of the {len(momentum_assets)} assets have positive momentum.

5. Position Management:
   - If the entry condition is met on Monday, the position is held for the entire week.
   - The position is exited on the next Monday if the entry condition is no longer met.

PARAMETERS:
- Trading Asset: {trading_asset}
- Momentum Lookback: {momentum_lookback_days} trading days
- Confirmation Threshold: {min_confirmations} out of {len(momentum_assets)} assets
- Rebalancing Frequency: Weekly (Monday)
"""

    elif strategy_name.lower() == 'vol_adaptive_momentum':
        price_col = strategy_config.get('price_column', 'cad_ig_er_index')
        vix_col = strategy_config.get('vix_column', 'vix')
        mom_lookback = strategy_config.get('mom_lookback', 20)
        vol_lookback = strategy_config.get('vol_lookback', 20)
        vix_z_lookback = strategy_config.get('vix_z_lookback', 252)
        thr_low_vol = strategy_config.get('thr_low_vol', 0.60)
        thr_high_vol = strategy_config.get('thr_high_vol', 0.00)
        max_hold = strategy_config.get('max_hold', 10)
        scale = strategy_config.get('scale', 0.005)

        return f"""Volatility-Adaptive {mom_lookback}-Day Momentum Strategy

TRADING RULES:
1. Core Indicators:
   - Price Series: {price_col}
   - Volatility Index: {vix_col}
   - Momentum: {mom_lookback}-day percentage change of the price series.
   - Realized Volatility: {vol_lookback}-day rolling standard deviation of daily returns.
   - Volatility Filter: {vix_z_lookback}-day z-score of the VIX index.

2. Volatility Regime Detection:
   - A "High Volatility" regime is active if the current {vol_lookback}-day realized volatility is
     greater than its long-run median value.
   - Otherwise, a "Calm" regime is active.

3. Entry Conditions:
   - The primary entry signal is the VIX z-score falling below a certain threshold.
   - In Calm Regimes: Enter LONG if VIX z-score <= {thr_low_vol:.2f} AND {mom_lookback}-day momentum is positive.
   - In High-Vol Regimes: Enter LONG if VIX z-score <= {thr_high_vol:.2f} (stricter threshold). The
     momentum filter is bypassed in high-volatility regimes.

4. Position Management (Adaptive Holding Period):
   - Upon entry, the holding period ('k') is dynamically calculated.
   - k = 1 + (Positive Momentum / {scale})
   - The holding period is capped at a maximum of {max_hold} days.
   - The position is held for 'k' days, after which the entry conditions are re-evaluated.
   - This allows the strategy to hold positions longer during strong momentum trends.

PARAMETERS:
- Momentum Lookback: {mom_lookback} days
- Realized Volatility Lookback: {vol_lookback} days
- VIX Z-Score Lookback: {vix_z_lookback} days
- Calm Vol Threshold (VIX z-score): {thr_low_vol:.2f}
- High Vol Threshold (VIX z-score): {thr_high_vol:.2f}
- Max Holding Period: {max_hold} days
- Momentum Scale for Holding: {scale}
"""

    elif strategy_name.lower() == 'multi_asset_momentum':
        assets_map = strategy_config.get('momentum_assets_map', {})
        lookback = strategy_config.get('momentum_lookback_days', 20)
        threshold = strategy_config.get('signal_threshold', -0.005)
        exit_hold_days = strategy_config.get('exit_hold_days', 5)
        trading_asset = strategy_config.get('trading_asset', 'cad_ig_er_index')
        
        asset_values = list(assets_map.values())

        return f"""Multi-Asset {lookback}-Day Combined Momentum (Weekly Rebalance)

TRADING RULES:
1. Data & Frequency:
   - This strategy operates on DAILY data but only makes trading decisions once per week.
   - Rebalancing Day: Monday.

2. Combined Momentum (Monday Only):
   - On Monday, a {lookback}-day momentum is calculated for {len(asset_values)} assets.
   - These are averaged into a single composite momentum signal for the market trend.

3. Entry Condition (Monday Only):
   - A LONG position is entered in '{trading_asset}' if the composite signal is positive.
   - Entry Signal: Combined Momentum > {threshold:.4f}

4. Exit Condition (Time-Based):
   - Once entered, the position is held for a fixed number of days.
   - Holding Period: {exit_hold_days} trading days.
   - The position is exited after the holding period, regardless of the momentum signal.

PARAMETERS:
- Trading Asset: {trading_asset}
- Momentum Lookback: {lookback} trading days
- Entry Threshold: > {threshold:.4f}
- Exit Holding Period: {exit_hold_days} trading days
- Rebalancing Frequency: Weekly (Monday)
"""
        
    elif strategy_name.lower() == 'genetic_algorithm':
        pop_size = strategy_config.get('population_size', 120)
        generations = strategy_config.get('max_generations', 120)
        penalty = strategy_config.get('fitness_drawdown_penalty_factor', 0.1)

        return f"""Genetic Algorithm Evolved Strategy (Weekly Rebalance)

TRADING RULES:
1. Data & Frequency:
   - This strategy uses a genetic algorithm to find rules using DAILY data features.
   - The final, evolved rule is only applied once per week on Monday.
   - Rebalancing Day: Monday.

2. Feature Engineering (The Gene Pool):
   - A wide range of daily technical indicators (momentum, volatility, etc.) are created.
   - These daily features serve as the building blocks for the evolutionary process.

3. Rule Evolution Process:
   - An initial population of {pop_size} random rules is evolved over {generations} generations.
   - Rules are selected based on historical fitness (profitability vs. risk).

4. Fitness Function (Survival of the Fittest):
   - Fitness = Total Return - ({penalty} * Max Drawdown).
   - This guides the evolution towards high-return, low-risk rules.

5. Final Rule Execution (Monday Only):
   - The single best-performing rule is applied every Monday to the daily features.
   - If the rule is TRUE on Monday, a LONG position is entered and held for the week.
   - If FALSE, the position is exited.

PARAMETERS FOR EVOLUTION:
- Population Size: {pop_size} rules per generation
- Max Generations: {generations} iterations
- Drawdown Penalty Factor: {penalty}
- Rebalancing Frequency: Weekly (Monday)
"""

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
    """Main function to run the backtesting framework."""
    parser = argparse.ArgumentParser(description="Modular Backtesting Framework")
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/config.yaml', 
        help='Path to the configuration file.'
    )
    parser.add_argument(
        '--strategies',
        nargs='+',
        type=str,
        help='Run specific strategies by name, overriding the config file. e.g., --strategies vol_adaptive_momentum genetic_algorithm'
    )
    parser.add_argument(
        '--list-strategies',
        action='store_true',
        help='List all available strategies and exit.'
    )
    
    args = parser.parse_args()

    # If --list-strategies is used, print them and exit
    if args.list_strategies:
        available_strategies = StrategyFactory.get_available_strategies()
        print("Available strategies:")
        for name in available_strategies:
            print(f"- {name}")
        sys.exit(0)

    # Load configuration
    config_dict = load_config(args.config)
    if not config_dict:
        print(f"Config file not found: {os.path.abspath(args.config)}")
        print("Please ensure the config file exists or provide a valid path.")
        return

    # Determine which strategies to run
    if args.strategies:
        # User has specified strategies via command line
        enabled_strategies = args.strategies
        print(f"Running user-specified strategies: {', '.join(enabled_strategies)}")
    else:
        # Use strategies from the config file
        enabled_strategies = config_dict.get('strategies', {}).get('enabled', [])
        print(f"Running enabled strategies from config: {', '.join(enabled_strategies)}")

    if not enabled_strategies:
        print("No strategies enabled in config or provided via arguments. Exiting.")
        return

    if len(enabled_strategies) > 1:
        run_strategy_comparison(enabled_strategies, config_dict)
    elif len(enabled_strategies) == 1:
        run_single_strategy(enabled_strategies[0], config_dict)


if __name__ == '__main__':
    main() 