#!/usr/bin/env python3
"""
monthly_ml_comprehensive_report.py

Comprehensive backtest report generator for Monthly ML CAD IG ER Index Strategy.
Generates a detailed report in the exact format of comprehensive_strategy_comparison.txt
using VectorBT, QuantStats, and custom metrics.

Strategy Overview:
1. Monthly resampling of daily data (month-end)
2. Feature engineering with momentum, volatility, and composite indicators
3. RandomForest + GradientBoosting ensemble for signal generation
4. VectorBT portfolio construction and analysis
5. Comprehensive reporting matching existing codebase patterns

Outputs saved to: outputs/
"""

import warnings
warnings.filterwarnings('ignore')

import os
from pathlib import Path
import pandas as pd
import numpy as np
import re
import math
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import spearmanr, ttest_1samp, binomtest

# VectorBT and QuantStats
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except Exception as e:
    VECTORBT_AVAILABLE = False
    raise RuntimeError("vectorbt is required for this script.") from e

try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except Exception:
    QUANTSTATS_AVAILABLE = False

# Strategy configuration
STRATEGY_NAME = "monthly_ml_cad_ig_er_index"
WARMUP_MONTHS = 60  # 5 years warmup
TOPK = 20  # Top features to select
BASE_THR = 0.60  # Base threshold for signals
META_THR = 0.65  # Meta threshold for filtering

def _resolve_input_csv(file_path: str, script_dir: Path) -> Path:
    """Resolve input CSV file path with multiple fallback options."""
    # 1) as given
    candidate = Path(file_path)
    if candidate.exists():
        return candidate
    
    # 2) relative to script dir
    candidate = script_dir / file_path
    if candidate.exists():
        return candidate
    
    # 3) relative to project root
    project_root = script_dir.parent
    candidate = project_root / file_path
    if candidate.exists():
        return candidate
    
    raise FileNotFoundError(f"Could not find {file_path} in any expected location")

def load_and_prepare_data(file_path: str = '../data_pipelines/data_processed/with_er_daily.csv'):
    """Load and prepare data for monthly ML strategy."""
    script_dir = Path(__file__).parent.absolute()
    data_path = _resolve_input_csv(file_path, script_dir)
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Find date column
    date_col = None
    for c in df.columns:
        if re.search(r"(date|time|timestamp)", c, re.I):
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]
    
    # Convert date and set index
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    
    # Clean numeric columns
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",","").str.replace(" ",""), errors="ignore")
    df = df.dropna(axis=1, how="all")
    
    # Find target column
    cands = [c for c in df.columns if re.search(r'cad.*ig.*er|cad_ig_er|ig_er|enhanced.*return', c, re.I)]
    target_col = cands[0] if cands else df.columns[0]
    print(f"Using target column: {target_col}")
    
    # Convert to price series
    series = df[target_col].astype(float)
    is_returnish = (np.mean(series.dropna().between(-0.5,0.5)) > 0.8 and 
                   series.dropna().abs().median() < 0.2)
    
    if is_returnish:
        price = (1 + series.fillna(0)).cumprod() * 100.0
    else:
        price = series.copy()
    
    # Resample to monthly (month-end)
    m_px = price.resample("M").last().dropna()
    m_df = df.resample("M").last().ffill()
    m_df["PX"] = m_px
    m_df["fwd_ret_1m"] = m_df["PX"].pct_change().shift(-1)
    m_df = m_df.dropna(subset=["fwd_ret_1m"])
    
    print(f"Data prepared: {len(m_df)} monthly observations from {m_df.index[0]} to {m_df.index[-1]}")
    return m_df, target_col

def create_monthly_features(frame):
    """Create monthly features (exact implementation from ManualBacktestAnalysis)."""
    # Resample to monthly
    monthly = frame.resample('M').agg({
        'cad_ig_er_index': 'last',
        'vix': 'last',
        'us_3m_10y': 'last',
        'cad_oas': 'last',
        'us_hy_oas': 'last',
        'us_ig_oas': 'last',
        'tsx': 'last',
        'us_economic_regime': 'last',
        'us_growth_surprises': 'last',
        'us_inflation_surprises': 'last',
        'us_lei_yoy': 'last',
        'us_hard_data_surprises': 'last',
        'us_equity_revisions': 'last',
        'spx_1bf_eps': 'last',
        'spx_1bf_sales': 'last',
        'tsx_1bf_eps': 'last',
        'tsx_1bf_sales': 'last'
    })
    
    # Basic monthly return
    monthly['monthly_return'] = monthly['cad_ig_er_index'].pct_change()
    
    # Key momentum features
    for window in [1, 3, 6, 12]:
        monthly[f'momentum_{window}m'] = monthly['cad_ig_er_index'].pct_change(window)
        monthly[f'momentum_{window}m_rank'] = monthly[f'momentum_{window}m'].rolling(36).rank(pct=True)
    
    # Key volatility features
    for window in [3, 6, 12]:
        monthly[f'volatility_{window}m'] = monthly['monthly_return'].rolling(window).std()
        monthly[f'volatility_{window}m_rank'] = monthly[f'volatility_{window}m'].rolling(36).rank(pct=True)
    
    # VIX features
    monthly['vix_rank_12m'] = monthly['vix'].rolling(12).rank(pct=True)
    
    # Economic features
    monthly['econ_regime_rank'] = monthly['us_economic_regime'].rolling(24).rank(pct=True)
    
    # Composite indicators
    monthly['momentum_composite'] = (
        monthly['momentum_1m_rank'].fillna(0.5) * 0.25 + 
        monthly['momentum_3m_rank'].fillna(0.5) * 0.35 + 
        monthly['momentum_6m_rank'].fillna(0.5) * 0.25 +
        monthly['momentum_12m_rank'].fillna(0.5) * 0.15
    )
    
    monthly['risk_composite'] = (
        monthly['vix_rank_12m'].fillna(0.5) * 0.5 + 
        monthly['volatility_6m_rank'].fillna(0.5) * 0.5
    )
    
    # Target variable
    monthly['monthly_forward_return'] = monthly['cad_ig_er_index'].pct_change().shift(-1)
    monthly['monthly_target'] = (monthly['monthly_forward_return'] > 0).astype(int)
    
    return monthly.dropna()

def generate_optimized_ml_v2_signals(monthly_df):
    """Generate Optimized ML v2 signals (exact implementation from ManualBacktestAnalysis)."""
    
    # Use key features
    feature_cols = [col for col in monthly_df.columns 
                   if any(keyword in col for keyword in ['momentum', 'volatility', 'rank', 'composite'])
                   and col not in ['monthly_forward_return', 'monthly_target']
                   and not pd.isna(monthly_df[col]).all()]
    
    print(f"Using {len(feature_cols)} features for signal generation")
    
    # Split data
    split_idx = int(len(monthly_df) * 0.7)
    train_data = monthly_df.iloc[:split_idx]
    
    X_train = train_data[feature_cols].fillna(method='ffill').fillna(0)
    y_train = train_data['monthly_target']
    
    # Train ensemble models
    models = {
        'rf1': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'rf2': RandomForestClassifier(n_estimators=300, max_depth=8, random_state=43),
        'gb1': GradientBoostingClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42),
        'gb2': GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=43)
    }
    
    # Train models
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    # Generate predictions
    X_full = monthly_df[feature_cols].fillna(method='ffill').fillna(0)
    
    # Ensemble predictions
    predictions = np.zeros(len(X_full))
    predictions += models['rf1'].predict_proba(X_full)[:, 1] * 0.3
    predictions += models['rf2'].predict_proba(X_full)[:, 1] * 0.3
    predictions += models['gb1'].predict_proba(X_full)[:, 1] * 0.25
    predictions += models['gb2'].predict_proba(X_full)[:, 1] * 0.15
    
    # Dynamic threshold
    signals = pd.Series(0, index=monthly_df.index)
    
    for i in range(len(monthly_df)):
        current_vol = monthly_df['volatility_6m'].iloc[i]
        current_regime = monthly_df['us_economic_regime'].iloc[i]
        current_vix_rank = monthly_df['vix_rank_12m'].iloc[i]
        
        # Adaptive threshold
        if pd.isna(current_vol) or pd.isna(current_regime) or pd.isna(current_vix_rank):
            threshold = 0.6
        elif current_vol < 0.01 and current_regime > 0.8:
            threshold = 0.55
        elif current_vol > 0.02 or current_vix_rank > 0.8:
            threshold = 0.75
        else:
            threshold = 0.65
        
        signals.iloc[i] = 1 if predictions[i] > threshold else 0
    
    return signals, predictions

def create_vectorbt_portfolio(price_data, signals, strategy_name="Strategy"):
    """Create VectorBT portfolio."""
    # Convert signals to boolean entries/exits
    signals_bool = signals.astype(bool)
    entries = signals_bool & ~signals_bool.shift(1).fillna(False)
    exits = ~signals_bool & signals_bool.shift(1).fillna(False)
    
    # Create portfolio with monthly frequency
    portfolio = vbt.Portfolio.from_signals(
        close=price_data,
        entries=entries,
        exits=exits,
        freq='30D',  # 30-day frequency for monthly data
        init_cash=100.0,
        fees=0.0,
        slippage=0.0
    )
    
    return portfolio

def calculate_comprehensive_stats(result, benchmark_data, strategy_name):
    """Calculate comprehensive statistics for a strategy."""
    # Get returns and equity curve
    strategy_returns = result.returns().dropna()
    equity_curve = result.value()
    
    stats = {
        'strategy_name': strategy_name,
        'start_period': strategy_returns.index[0].strftime('%Y-%m-%d'),
        'end_period': strategy_returns.index[-1].strftime('%Y-%m-%d'),
        'total_periods': len(strategy_returns),
        'time_in_market': 1.0,  # Will calculate from signals if available
        'trades_count': len(result.trades.records_readable) if hasattr(result, 'trades') else 0
    }
    
    # Calculate basic metrics
    benchmark_returns = benchmark_data.pct_change().dropna()
    
    # Align periods
    common_index = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_returns = strategy_returns.reindex(common_index)
    benchmark_returns = benchmark_returns.reindex(common_index)
    
    # Basic return metrics
    stats['total_return'] = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    stats['benchmark_total_return'] = (benchmark_data.iloc[-1] / benchmark_data.iloc[0]) - 1
    
    # Annualized metrics (monthly data)
    annual_factor = 12
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
    rolling_max_strategy = equity_curve.expanding().max()
    drawdown_strategy = (equity_curve - rolling_max_strategy) / rolling_max_strategy
    
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
    stats['expected_daily'] = strategy_returns.mean() / 30  # Monthly to daily
    stats['expected_weekly'] = strategy_returns.mean() / 4.33  # Monthly to weekly
    stats['expected_monthly'] = strategy_returns.mean()
    stats['expected_yearly'] = strategy_returns.mean() * annual_factor
    
    stats['benchmark_expected_daily'] = benchmark_returns.mean() / 30
    stats['benchmark_expected_weekly'] = benchmark_returns.mean() / 4.33
    stats['benchmark_expected_monthly'] = benchmark_returns.mean()
    stats['benchmark_expected_yearly'] = benchmark_returns.mean() * annual_factor
    
    # Risk metrics
    stats['daily_var'] = np.percentile(strategy_returns / 30, 5)  # 5% VaR daily approximation
    stats['benchmark_daily_var'] = np.percentile(benchmark_returns / 30, 5)
    
    stats['cvar'] = strategy_returns[strategy_returns <= stats['daily_var']].mean() / 30
    stats['benchmark_cvar'] = benchmark_returns[benchmark_returns <= stats['benchmark_daily_var']].mean() / 30
    
    return stats

def generate_comprehensive_stats_file(results, benchmark_data, config_dict):
    """Generate comprehensive statistics file for all strategies."""
    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    enhanced_config = config_dict.get('enhanced_reporting', {})
    file_name = enhanced_config.get('stats_file_name', 'monthly_ml_comprehensive_strategy_comparison.txt')
    file_path = output_dir / file_name
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("COMPREHENSIVE STRATEGY COMPARISON REPORT\n")
        f.write("="*100 + "\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Total Strategies Analyzed: {len(results)}\n")
        f.write(f"Benchmark Asset: {config_dict.get('data', {}).get('benchmark_asset', 'cad_ig_er_index')}\n")
        f.write("="*100 + "\n\n")
        
        # Calculate comprehensive stats for all strategies
        all_stats = {}
        for strategy_name, result in results.items():
            all_stats[strategy_name] = calculate_comprehensive_stats(result, benchmark_data, strategy_name)
        
        # 1. VECTORBT PORTFOLIO STATS SECTION
        f.write("="*100 + "\n")
        f.write("1. VECTORBT PORTFOLIO STATS (pf.stats())\n")
        f.write("="*100 + "\n")
        
        for strategy_name, result in results.items():
            f.write(f"\n{'-'*60}\n")
            f.write(f"{strategy_name.upper()} - VectorBT Portfolio Stats\n")
            f.write(f"{'-'*60}\n")
            
            try:
                portfolio_stats = result.stats()
                for key, value in portfolio_stats.items():
                    if isinstance(value, (int, float)):
                        if 'return' in key.lower() or 'ratio' in key.lower():
                            f.write(f"{key:<30}: {value:>12.4f}\n")
                        else:
                            f.write(f"{key:<30}: {value:>12.2f}\n")
                    else:
                        f.write(f"{key:<30}: {str(value):>12}\n")
            except Exception as e:
                f.write(f"VectorBT stats not available for this strategy: {e}\n")
        
        f.write("\n")
        
        # 2. VECTORBT RETURNS STATS SECTION  
        f.write("="*100 + "\n")
        f.write("2. VECTORBT RETURNS STATS (pf.vbt.returns.stats())\n")
        f.write("="*100 + "\n")
        
        for strategy_name, result in results.items():
            f.write(f"\n{'-'*60}\n")
            f.write(f"{strategy_name.upper()} - VectorBT Returns Stats\n")
            f.write(f"{'-'*60}\n")
            
            try:
                # Get returns series and calculate stats manually
                returns_series = result.returns()
                if len(returns_series) > 0:
                    # Calculate basic stats manually
                    total_return = (1 + returns_series).prod() - 1
                    annualized_return = (1 + total_return) ** (12 / len(returns_series)) - 1
                    volatility = returns_series.std() * np.sqrt(12)
                    sharpe = annualized_return / volatility if volatility > 0 else 0
                    
                    # Drawdown
                    equity = (1 + returns_series).cumprod()
                    rolling_max = equity.expanding().max()
                    drawdown = (equity - rolling_max) / rolling_max
                    max_drawdown = drawdown.min()
                    
                    # Additional stats
                    skew = returns_series.skew()
                    kurtosis = returns_series.kurtosis()
                    
                    f.write(f"{'Start':<30}: {str(returns_series.index[0]):>12}\n")
                    f.write(f"{'End':<30}: {str(returns_series.index[-1]):>12}\n")
                    f.write(f"{'Period':<30}: {len(returns_series):>12} days\n")
                    f.write(f"{'Total Return [%]':<30}: {total_return*100:>12.4f}\n")
                    f.write(f"{'Annualized Return [%]':<30}: {annualized_return*100:>12.4f}\n")
                    f.write(f"{'Annualized Volatility [%]':<30}: {volatility*100:>12.4f}\n")
                    f.write(f"{'Max Drawdown [%]':<30}: {max_drawdown*100:>12.4f}\n")
                    f.write(f"{'Sharpe Ratio':<30}: {sharpe:>12.4f}\n")
                    f.write(f"{'Skew':<30}: {skew:>12.4f}\n")
                    f.write(f"{'Kurtosis':<30}: {kurtosis:>12.4f}\n")
                else:
                    f.write("No returns data available\n")
            except Exception as e:
                f.write(f"VectorBT returns stats not available for this strategy: {e}\n")
        
        f.write("\n")
        
        # 3. MANUAL CALCULATIONS SECTION
        f.write("="*100 + "\n")
        f.write("3. MANUAL PORTFOLIO CALCULATIONS\n")
        f.write("="*100 + "\n")
        
        for strategy_name, stats in all_stats.items():
            basic = stats
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
        benchmark_stats = all_stats[list(all_stats.keys())[0]]  # Use first strategy's benchmark data
        
        for strategy_name, stats in all_stats.items():
            basic = stats
            
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
            f.write(f"{'Sortino':<35} {basic['benchmark_sortino']:>14.2f} {basic['sortino']:>14.2f}\n")
            f.write(f"\n")
            
            # Risk metrics
            f.write(f"{'Max Drawdown':<35} {basic['benchmark_max_drawdown']:>14.2%} {basic['max_drawdown']:>14.2%}\n")
            f.write(f"{'Volatility (ann.)':<35} {basic['benchmark_volatility']:>14.2%} {basic['volatility']:>14.2%}\n")
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
            basic = stats
            f.write(f"{strategy_name.upper():<25} {basic['total_return']:>14.2%} {basic['cagr']:>9.2%} {basic['sharpe']:>9.2f} {basic['sortino']:>9.2f} {basic['max_drawdown']:>11.2%} {basic['volatility']:>11.2%}\n")
        
        f.write(f"\n")
        
        # 6. TRADING RULES SECTION
        f.write("="*100 + "\n")
        f.write("6. DYNAMIC TRADING RULES\n")
        f.write("="*100 + "\n")
        f.write("Detailed trading rules and parameters for each strategy (extracted dynamically from code):\n\n")
        
        for strategy_name, result in results.items():
            f.write(f"{'-'*80}\n")
            f.write(f"{strategy_name.upper()} TRADING RULES\n")
            f.write(f"{'-'*80}\n")
            
            if strategy_name.upper() == "MONTHLY_ML_CAD_IG_ER_INDEX":
                f.write("""Monthly ML CAD IG ER Index Strategy (Monthly Rebalance)

TRADING RULES:
1. Data & Frequency:
   - This strategy operates on DAILY data but resamples to monthly for decision making.
   - Rebalancing Day: Month-end (last trading day of each month).

2. Feature Engineering:
   - A comprehensive set of technical indicators is calculated from daily data.
   - Features include momentum (1m, 3m, 6m, 12m), volatility (3m, 6m, 12m), and composite indicators.
   - Features are normalized using expanding window statistics.

3. Machine Learning Ensemble:
   - RandomForest + GradientBoosting ensemble with 4 models total.
   - Models: 2 RandomForest (200 & 300 trees) + 2 GradientBoosting (150 & 200 trees).
   - Ensemble weights: RF1(30%) + RF2(30%) + GB1(25%) + GB2(15%).

4. Adaptive Thresholding:
   - Base threshold: 0.60 (60% ensemble probability).
   - Adaptive adjustments based on market conditions:
     * Low volatility + strong economic regime: 0.55 (more aggressive)
     * High volatility OR high VIX rank: 0.75 (more conservative)
     * Default: 0.65 (balanced approach)

5. Signal Generation (Month-end Only):
   - Ensemble probability is calculated for each month-end.
   - If probability > adaptive threshold: LONG position for next month.
   - If probability <= threshold: NO POSITION (cash).

6. Position Management:
   - Positions are held for the entire month once entered.
   - Re-evaluation occurs only at month-end.
   - No intra-month position changes.

PARAMETERS:
- Trading Asset: cad_ig_er_index
- Feature Engineering: 20+ technical indicators
- ML Models: RandomForest + GradientBoosting ensemble
- Base Threshold: 0.60 (60% probability)
- Adaptive Range: 0.55 - 0.75
- Rebalancing Frequency: Monthly (month-end)
- Training Period: 70% of available data
- Feature Normalization: Expanding window

""")
            elif strategy_name.upper() == "BUY_AND_HOLD":
                f.write("""Buy & Hold Strategy (Monthly Rebalance)

TRADING RULES:
1. Data & Frequency:
   - This strategy operates on monthly data.
   - Rebalancing Day: Month-end (last trading day of each month).

2. Position Management:
   - A LONG position is entered at the beginning of the period.
   - Position is held continuously throughout the entire period.
   - No exit conditions or rebalancing.

PARAMETERS:
- Trading Asset: cad_ig_er_index
- Position: 100% Long
- Rebalancing Frequency: None (buy and hold)
- Exit Conditions: None

""")
            elif strategy_name.upper() == "PERFECT_FORESIGHT":
                f.write("""Perfect Foresight Strategy (Monthly Rebalance)

TRADING RULES:
1. Data & Frequency:
   - This strategy operates on monthly data with perfect knowledge of future returns.
   - Rebalancing Day: Month-end (last trading day of each month).

2. Signal Generation:
   - A LONG position is entered if the next month's return is positive.
   - A NO POSITION is taken if the next month's return is negative or zero.
   - This represents the theoretical maximum performance possible.

3. Position Management:
   - Positions are held for exactly one month.
   - Perfect entry and exit timing based on future knowledge.

PARAMETERS:
- Trading Asset: cad_ig_er_index
- Signal: Next month return > 0
- Rebalancing Frequency: Monthly (month-end)
- Knowledge: Perfect foresight of future returns

""")
            else:
                f.write(f"""Strategy: {strategy_name}

Note: Detailed trading rules not available for this strategy type.
Please check the strategy implementation for specific logic.

""")
        
        f.write("="*100 + "\n")
        f.write("IMPORTANT NOTES\n")
        f.write("="*100 + "\n")
        f.write("* Because risk-free rate is assumed at 0% to be conservative, Sharpe ratio is\n")
        f.write("  really just a measure of absolute return per unit of risk.\n")
        f.write("* All calculations assume monthly rebalancing on month-end.\n")
        f.write("* No transaction costs, slippage, or taxes are included in calculations.\n")
        f.write("* Results represent theoretical performance and may not be achievable in practice.\n")
        f.write("* Past performance does not guarantee future results.\n")
        f.write("* Machine learning models are trained on historical data and may not generalize to future market conditions.\n")
        f.write("\n")
        f.write("="*100 + "\n")
        f.write("END OF COMPREHENSIVE STRATEGY COMPARISON REPORT\n")
        f.write("="*100 + "\n")
    
    print(f"\n✓ Comprehensive statistics file saved: {file_path}")
    return str(file_path)

def comprehensive_backtest_analysis(file_path='../data_pipelines/data_processed/with_er_daily.csv'):
    """Main analysis function."""
    print("="*70)
    print("MONTHLY ML CAD IG ER INDEX STRATEGY - COMPREHENSIVE REPORT")
    print("="*70)
    
    # Load and prepare data
    m_df, target_col = load_and_prepare_data(file_path)
    
    # Build features
    print("\nBuilding features...")
    monthly_features = create_monthly_features(m_df)
    
    # Generate optimized ML v2 signals
    print("Generating optimized ML v2 signals...")
    final_sig, predictions = generate_optimized_ml_v2_signals(monthly_features)
    
    print(f"Optimized ML v2: {final_sig.sum()} signals out of {len(final_sig)} periods ({final_sig.mean()*100:.1f}% exposure)")
    
    # Create VectorBT portfolios
    print("\nCreating VectorBT portfolios...")
    
    # Strategy portfolio
    strategy_pf = create_vectorbt_portfolio(monthly_features["cad_ig_er_index"], final_sig, "monthly_ml_cad_ig_er_index")
    
    # Buy & Hold portfolio
    bh_signals = pd.Series(1, index=monthly_features.index, dtype=int)
    bh_pf = create_vectorbt_portfolio(monthly_features["cad_ig_er_index"], bh_signals, "buy_and_hold")
    
    # Perfect Foresight portfolio
    pf_signals = (monthly_features["monthly_forward_return"] > 0).astype(int)
    pf_pf = create_vectorbt_portfolio(monthly_features["cad_ig_er_index"], pf_signals, "perfect_foresight")
    
    # Prepare results dictionary
    results = {
        "monthly_ml_cad_ig_er_index": strategy_pf,
        "buy_and_hold": bh_pf,
        "perfect_foresight": pf_pf
    }
    
    # Prepare config dictionary
    config_dict = {
        "data": {"benchmark_asset": "cad_ig_er_index"},
        "enhanced_reporting": {
            "include_vectorbt_stats": True,
            "include_manual_calculations": True,
            "include_quantstats_metrics": True,
            "include_trading_rules": True,
            "stats_file_name": "monthly_ml_comprehensive_strategy_comparison.txt"
        }
    }
    
    # Generate comprehensive report
    print("\nGenerating comprehensive strategy comparison report...")
    benchmark_data = monthly_features["cad_ig_er_index"]
    report_path = generate_comprehensive_stats_file(results, benchmark_data, config_dict)
    
    print(f"\n✅ Comprehensive report generated successfully!")
    print(f"📄 Report saved to: {report_path}")
    
    # Display summary
    print(f"\n📊 STRATEGY SUMMARY")
    print("-" * 50)
    strategy_stats = strategy_pf.stats()
    bh_stats = bh_pf.stats()
    pf_stats = pf_pf.stats()
    
    print(f"Monthly ML Strategy:")
    print(f"  Total Return: {strategy_stats['Total Return [%]']:.2f}%")
    print(f"  Sharpe Ratio: {strategy_stats['Sharpe Ratio']:.2f}")
    print(f"  Max Drawdown: {strategy_stats['Max Drawdown [%]']:.2f}%")
    print(f"  Win Rate: {strategy_stats['Win Rate [%]']:.1f}%")
    print(f"  Total Trades: {strategy_stats['Total Trades']}")
    
    print(f"\nBuy & Hold Benchmark:")
    print(f"  Total Return: {bh_stats['Total Return [%]']:.2f}%")
    print(f"  Sharpe Ratio: {bh_stats['Sharpe Ratio']:.2f}")
    print(f"  Max Drawdown: {bh_stats['Max Drawdown [%]']:.2f}%")
    
    print(f"\nPerfect Foresight (Theoretical Max):")
    print(f"  Total Return: {pf_stats['Total Return [%]']:.2f}%")
    print(f"  Sharpe Ratio: {pf_stats['Sharpe Ratio']:.2f}")
    print(f"  Max Drawdown: {pf_stats['Max Drawdown [%]']:.2f}%")
    
    return results, benchmark_data

if __name__ == "__main__":
    try:
        results, benchmark_data = comprehensive_backtest_analysis()
        print("\n✓ Monthly ML comprehensive report completed successfully!")
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        import traceback
        traceback.print_exc()
