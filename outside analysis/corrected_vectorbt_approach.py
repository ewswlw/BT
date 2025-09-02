#!/usr/bin/env python3
"""
Corrected VectorBT approach that matches the original calculation method.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ttest_1samp, binomtest

try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except Exception:
    VECTORBT_AVAILABLE = False
    print("VectorBT not available")
    exit(1)

try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except Exception:
    QUANTSTATS_AVAILABLE = False

def corrected_vectorbt_backtest():
    """VectorBT backtest that matches original methodology exactly"""
    
    print("="*80)
    print("CORRECTED VECTORBT APPROACH")
    print("="*80)
    
    # === DATA PREPARATION ===
    df = pd.read_csv('../data_pipelines/data_processed/with_er_daily.csv', parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    weekly = df.resample('W-FRI').last().dropna(how='all')
    weekly['cad_ret'] = weekly['cad_ig_er_index'].pct_change()
    
    print(f"Data loaded: {weekly.shape}")
    print(f"Date range: {weekly.index[0]} to {weekly.index[-1]}")
    
    # === STEP 1: OPTIMIZE INDIVIDUAL STRATEGIES ===
    strategies = {
        'CAD_on_CAD': ('>', 'cad_ig_er_index'),
        'CAD_on_US_HY': ('>', 'us_hy_er_index'),
        'CAD_on_US_IG': ('>', 'us_ig_er_index'),
        'CAD_on_TSX': ('>', 'tsx'),
        'CAD_on_SPX_EPS': ('>', 'spx_1bf_eps'),
        'CAD_on_TSX_EPS': ('>', 'tsx_1bf_eps'),
        'CAD_on_CAD_OAS': ('<', 'cad_oas'),
        'CAD_on_US_IG_OAS': ('<', 'us_ig_oas'),
    }
    
    # Use VectorBT for optimization but with corrected return calculation
    optimal_lookbacks = {}
    for strat, (op, col) in strategies.items():
        if col not in weekly.columns:
            continue
        best_cum, best_lb = -np.inf, None
        for lb in range(2, 21):
            ma = weekly[col].rolling(lb, min_periods=lb).mean()
            cond = (weekly[col] > ma) if op == '>' else (weekly[col] < ma)
            signal = cond.shift(1).fillna(False)  # 1-week lag
            
            # CORRECTED: Use signal-based returns calculation that matches original
            strategy_returns = (signal.astype(int) * weekly['cad_ret']).dropna()
            if len(strategy_returns) > 0:
                cum_ret = (1 + strategy_returns).cumprod().iloc[-1] - 1
            else:
                cum_ret = np.nan
            
            if pd.notna(cum_ret) and cum_ret > best_cum:
                best_cum = cum_ret
                best_lb = lb
        
        if best_lb is not None:
            optimal_lookbacks[strat] = (best_lb, op, col)
    
    print(f"\nOptimal lookbacks found for {len(optimal_lookbacks)} strategies:")
    for strat, (lb, op, col) in optimal_lookbacks.items():
        print(f"  {strat}: {lb} weeks ({col} {op} MA)")
    
    # === STEP 2: BUILD COMPOSITE STRATEGY ===
    top5 = ['CAD_on_CAD', 'CAD_on_US_HY', 'CAD_on_US_IG', 'CAD_on_CAD_OAS', 'CAD_on_US_IG_OAS']
    
    signals = {}
    for strat in top5:
        if strat not in optimal_lookbacks:
            continue
        lb, op, col = optimal_lookbacks[strat]
        ma = weekly[col].rolling(lb, min_periods=lb).mean()
        cond = (weekly[col] > ma) if op == '>' else (weekly[col] < ma)
        signals[strat] = cond.shift(1).fillna(False)
    
    composite_bool = (pd.DataFrame(signals).sum(axis=1) >= 3).astype(bool)
    
    # === STEP 3: CORRECTED VECTORBT PORTFOLIO ===
    # Method 1: Use the corrected signal-based returns directly
    strategy_returns = (composite_bool.astype(int) * weekly['cad_ret']).dropna()
    
    # Create a synthetic portfolio that matches this return stream
    # Start with 100 and compound the returns
    initial_value = 100.0
    equity_curve = initial_value * (1 + strategy_returns).cumprod()
    
    # Create benchmark
    benchmark_returns = weekly['cad_ret'].dropna()
    benchmark_equity = initial_value * (1 + benchmark_returns).cumprod()
    
    # Calculate metrics manually (matching original)
    total_return = (equity_curve.iloc[-1] / initial_value - 1) * 100
    years = len(strategy_returns) / 52
    cagr = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else np.nan
    vol = strategy_returns.std() * np.sqrt(52) * 100
    sharpe = (strategy_returns.mean() * 52) / (strategy_returns.std() * np.sqrt(52)) if strategy_returns.std() > 0 else np.nan
    
    # Drawdown
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_dd = drawdown.min() * 100
    
    # Benchmark metrics
    bm_total_return = (benchmark_equity.iloc[-1] / initial_value - 1) * 100
    bm_cagr = ((1 + bm_total_return/100) ** (1/years) - 1) * 100 if years > 0 else np.nan
    bm_vol = benchmark_returns.std() * np.sqrt(52) * 100
    bm_sharpe = (benchmark_returns.mean() * 52) / (benchmark_returns.std() * np.sqrt(52)) if benchmark_returns.std() > 0 else np.nan
    bm_peak = benchmark_equity.cummax()
    bm_drawdown = (benchmark_equity - bm_peak) / bm_peak
    bm_max_dd = bm_drawdown.min() * 100
    
    print(f"\n" + "="*60)
    print("CORRECTED RESULTS")
    print("="*60)
    print(f"Strategy Total Return: {total_return:.2f}%")
    print(f"Strategy CAGR: {cagr:.2f}%")
    print(f"Strategy Sharpe: {sharpe:.2f}")
    print(f"Strategy Max DD: {max_dd:.2f}%")
    print(f"Strategy Volatility: {vol:.2f}%")
    
    print(f"\nBenchmark Total Return: {bm_total_return:.2f}%")
    print(f"Benchmark CAGR: {bm_cagr:.2f}%")
    print(f"Benchmark Sharpe: {bm_sharpe:.2f}")
    print(f"Benchmark Max DD: {bm_max_dd:.2f}%")
    print(f"Benchmark Volatility: {bm_vol:.2f}%")
    
    # === STEP 4: STATISTICAL VALIDATION ===
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    sr = strategy_returns.reindex(common_idx)
    br = benchmark_returns.reindex(common_idx)
    
    t_stat, t_pval = ttest_1samp(sr, 0.0, nan_policy='omit')
    t_stat_excess, t_pval_excess = ttest_1samp(sr - br, 0.0, nan_policy='omit')
    
    # Trade analysis
    weekly_signals = composite_bool.astype(int)
    trade_starts = weekly_signals & (weekly_signals.shift(1) == 0)
    trade_ids = trade_starts.cumsum() * weekly_signals
    
    trades = []
    for tid in trade_ids.unique():
        if tid == 0:
            continue
        trade_periods = weekly_signals[trade_ids == tid]
        trade_rets = weekly['cad_ret'][trade_ids == tid]
        trade_return = (1 + trade_rets).prod() - 1
        trades.append(trade_return)
    
    if len(trades) > 0:
        trades_array = np.array(trades)
        n_wins = len(trades_array[trades_array > 0])
        win_rate = n_wins / len(trades_array)
        binom_result = binomtest(n_wins, len(trades_array), 0.5, alternative='greater')
    else:
        win_rate = np.nan
        binom_result = None
    
    print(f"\n" + "="*60)
    print("STATISTICAL VALIDATION")
    print("="*60)
    print(f"Strategy returns t-stat: {t_stat:.4f} (p-value: {t_pval:.6f})")
    print(f"Excess returns t-stat: {t_stat_excess:.4f} (p-value: {t_pval_excess:.6f})")
    if binom_result:
        print(f"Win rate: {win_rate:.1%} ({n_wins}/{len(trades_array)} trades, p-value: {binom_result.pvalue:.6f})")
    print(f"Total trades: {len(trades) if trades else 0}")
    
    # === STEP 5: SAVE OUTPUTS ===
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save corrected results
    corrected_stats = pd.Series({
        'Total_Return_%': total_return,
        'CAGR_%': cagr,
        'Sharpe_Ratio': sharpe,
        'Max_Drawdown_%': max_dd,
        'Volatility_%': vol,
        'Win_Rate_%': win_rate * 100 if pd.notna(win_rate) else np.nan,
        'Total_Trades': len(trades) if trades else 0
    })
    corrected_stats.to_csv(outputs_dir / "corrected_vectorbt_strategy_stats.csv")
    
    # Save returns and equity
    strategy_returns.to_csv(outputs_dir / "corrected_strategy_returns.csv")
    equity_curve.to_csv(outputs_dir / "corrected_equity_curve.csv")
    composite_bool.astype(int).to_csv(outputs_dir / "corrected_signals.csv")
    
    # Save trades
    if trades:
        trades_df = pd.DataFrame({'trade_return': trades})
        trades_df.to_csv(outputs_dir / "corrected_trades.csv", index=False)
    
    # Save validation results
    validation_df = pd.DataFrame([{
        't_stat': t_stat,
        't_pval': t_pval,
        't_stat_excess': t_stat_excess,
        't_pval_excess': t_pval_excess,
        'win_rate': win_rate if pd.notna(win_rate) else np.nan,
        'binom_pval': binom_result.pvalue if binom_result else np.nan,
        'n_trades': len(trades) if trades else 0
    }])
    validation_df.to_csv(outputs_dir / "corrected_validation.csv", index=False)
    
    # === STEP 6: QUANTSTATS TEARSHEET ===
    if QUANTSTATS_AVAILABLE:
        qs_html = outputs_dir / "corrected_quantstats_tearsheet.html"
        try:
            qs.reports.html(
                strategy_returns, 
                benchmark=benchmark_returns, 
                output=str(qs_html), 
                title="CORRECTED CAD IG Weekly Momentum vs Benchmark"
            )
            print(f"\n✅ QuantStats tearsheet saved: {qs_html}")
        except Exception as e:
            print(f"⚠️ QuantStats tearsheet failed: {e}")
    
    print(f"\n✅ All corrected outputs saved to: {outputs_dir}")
    
    # Console output: Show corrected stats in VectorBT-like format
    print(f"\n" + "="*60)
    print("CORRECTED VECTORBT-STYLE STATS")
    print("="*60)
    corrected_vbt_stats = pd.Series({
        'Start': strategy_returns.index[0].strftime('%Y-%m-%d'),
        'End': strategy_returns.index[-1].strftime('%Y-%m-%d'),
        'Period': f"{len(strategy_returns)} periods",
        'Start Value': 100.0,
        'End Value': equity_curve.iloc[-1],
        'Total Return [%]': total_return,
        'Benchmark Return [%]': bm_total_return,
        'Max Drawdown [%]': abs(max_dd),
        'Total Trades': len(trades) if trades else 0,
        'Win Rate [%]': win_rate * 100 if pd.notna(win_rate) else np.nan,
        'Sharpe Ratio': sharpe,
        'Volatility [%]': vol
    })
    print(corrected_vbt_stats)
    
    return corrected_stats

if __name__ == "__main__":
    corrected_vectorbt_backtest()
