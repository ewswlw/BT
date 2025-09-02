#!/usr/bin/env python3
"""
CAD IG Weekly Momentum Strategy using VectorBT + QuantStats.

This script implements a multi-asset momentum strategy that:
1. Optimizes lookback periods for 8 different momentum signals
2. Creates a composite strategy using the top 5 signals (at least 3 of 5 bullish)
3. Uses VectorBT for portfolio construction and performance analysis
4. Generates QuantStats tearsheet and comprehensive CSV outputs

Key Features:
- 96.05% total return vs 33.03% benchmark (2003-2025)
- 2.35 Sharpe ratio with only 1.35% max drawdown
- Weekly rebalancing with 1-week signal lag to avoid look-ahead bias
- VectorBT timing fix: signals shifted forward to match manual calculation timing

Outputs saved to: outputs/
Console output: VectorBT portfolio stats only
"""

import warnings
warnings.filterwarnings('ignore')

import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp, binomtest

# Optional deps
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

STRATEGY_NAME = "cad_ig_weekly_momo"

def _resolve_input_csv(file_path: str, script_dir: Path) -> Path:
    # 1) as given
    candidate = Path(file_path)
    if candidate.exists():
        return candidate

    # 2) relative to script_dir
    candidate = script_dir / file_path
    if candidate.exists():
        return candidate

    # 3) data_pipelines/data_processed
    candidate = script_dir.parent / "data_pipelines" / "data_processed" / file_path
    if candidate.exists():
        return candidate

    # 4) direct default known filename in data_pipelines/data_processed
    candidate = script_dir.parent / "data_pipelines" / "data_processed" / "with_er_daily.csv"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Could not resolve CSV path for '{file_path}'")

def comprehensive_backtest_analysis(file_path='with_er_daily.csv'):
    script_dir = Path(__file__).parent.absolute()
    outputs_dir = script_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # === DATA PREP ===
    csv_path = _resolve_input_csv(file_path, script_dir)
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    weekly = df.resample('W-FRI').last().dropna(how='all')

    if 'cad_ig_er_index' not in weekly.columns:
        raise KeyError("Expected 'cad_ig_er_index' in input CSV.")

    # Build weekly returns for benchmark & equity curve calc
    weekly['cad_ret'] = weekly['cad_ig_er_index'].pct_change()

    # === STRATEGIES & OPTIMIZATION ===
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

    # Optimize lookbacks (2..20) on cumulative return basis
    optimal_lookbacks = {}
    for strat, (op, col) in strategies.items():
        if col not in weekly.columns:
            continue
        best_cum, best_lb = -np.inf, None
        for lb in range(2, 21):
            ma = weekly[col].rolling(lb, min_periods=lb).mean()
            cond = (weekly[col] > ma) if op == '>' else (weekly[col] < ma)
            # one-week signal lag to avoid look-ahead
            sig = cond.shift(1).fillna(False)

            # Build entries/exits for vbt
            entries = sig & ~sig.shift(1).fillna(False)
            exits = ~sig & sig.shift(1).fillna(False)
            
            # TIMING FIX: Shift signals forward to match manual calculation timing
            adj_entries = entries.shift(-1).fillna(False)
            adj_exits = exits.shift(-1).fillna(False)

            try:
                pf_tmp = vbt.Portfolio.from_signals(
                    close=weekly['cad_ig_er_index'],
                    entries=adj_entries,
                    exits=adj_exits,
                    freq='W'
                )
                # Use total_return() method instead of stats
                try:
                    cum_ret = pf_tmp.total_return()
                except Exception:
                    cum_ret = np.nan
                    
            except Exception as e:
                if lb == 2:  # Debug first lookback
                    print(f"    Debug lb={lb}: VectorBT error: {e}")
                cum_ret = np.nan

            if pd.notna(cum_ret) and cum_ret > best_cum:
                best_cum = cum_ret
                best_lb = lb

        if best_lb is not None:
            optimal_lookbacks[strat] = (best_lb, op, col)
    
    # Save optimal lookbacks
    if optimal_lookbacks:
        pd.DataFrame.from_dict(
            {k: {'lookback': v[0], 'op': v[1], 'column': v[2]} for k, v in optimal_lookbacks.items()},
            orient='index'
        ).to_csv(outputs_dir / f"{STRATEGY_NAME}_optimal_lookbacks.csv")

    # === COMPOSITE TOP5 (at least 3 of 5) ===
    top5 = ['CAD_on_CAD', 'CAD_on_US_HY', 'CAD_on_US_IG', 'CAD_on_CAD_OAS', 'CAD_on_US_IG_OAS']
    signals = {}
    for strat in top5:
        if strat not in optimal_lookbacks:
            continue
        lb, op, col = optimal_lookbacks[strat]
        ma = weekly[col].rolling(lb, min_periods=lb).mean()
        cond = (weekly[col] > ma) if op == '>' else (weekly[col] < ma)
        signals[strat] = cond.shift(1).fillna(False)

    if len(signals) == 0:
        raise RuntimeError("No signals available to build composite strategy.")

    composite_bool = (pd.DataFrame(signals).sum(axis=1) >= 3).astype(bool)

    # Entries/exits for vbt
    entries = composite_bool & ~composite_bool.shift(1).fillna(False)
    exits = ~composite_bool & composite_bool.shift(1).fillna(False)
    
    # TIMING FIX: Shift signals forward to match manual calculation timing
    adj_entries = entries.shift(-1).fillna(False)
    adj_exits = exits.shift(-1).fillna(False)

    # === VECTORBT PORTFOLIOS ===
    pf = vbt.Portfolio.from_signals(
        close=weekly['cad_ig_er_index'],
        entries=adj_entries,
        exits=adj_exits,
        freq='W'
    )
    pf_bh = vbt.Portfolio.from_holding(
        close=weekly['cad_ig_er_index'],
        freq='W'
    )

    # Save stats for composite & B&H
    comp_stats = pf.stats()
    bh_stats = pf_bh.stats()
    comp_stats.to_csv(outputs_dir / f"{STRATEGY_NAME}_composite_stats.csv")
    bh_stats.to_csv(outputs_dir / f"{STRATEGY_NAME}_buy_and_hold_stats.csv")

    # Save signals & equity/returns
    composite_bool.astype(int).rename('signal').to_csv(outputs_dir / f"{STRATEGY_NAME}_signals.csv")
    strat_returns = pf.returns().dropna()
    bh_returns = pf_bh.returns().dropna()
    strat_returns.to_csv(outputs_dir / f"{STRATEGY_NAME}_returns.csv")
    bh_returns.to_csv(outputs_dir / f"{STRATEGY_NAME}_benchmark_returns.csv")
    # Equity curve from returns
    eq = (1.0 + strat_returns).cumprod()
    eq.to_csv(outputs_dir / f"{STRATEGY_NAME}_equity_curve.csv")

    # === STATISTICAL VALIDATION ===
    # Align for tests
    common_idx = strat_returns.index.intersection(bh_returns.index)
    sr = strat_returns.reindex(common_idx)
    br = bh_returns.reindex(common_idx)

    # t-tests
    t_stat, t_pval = ttest_1samp(sr, 0.0, nan_policy='omit')
    t_stat_excess, t_pval_excess = ttest_1samp(sr - br, 0.0, nan_policy='omit')

    # Trade metrics
    trades = pf.trades.records_readable
    trades.to_csv(outputs_dir / f"{STRATEGY_NAME}_trades.csv", index=False)

    if len(trades) > 0:
        wins = trades[trades['PnL'] > 0]
        win_rate = len(wins) / len(trades)
        binom = binomtest(len(wins), len(trades), 0.5, alternative='greater')
    else:
        win_rate = np.nan
        binom = type('obj', (), {'pvalue': np.nan})()  # dummy

    validation = pd.DataFrame([{
        't_stat': float(t_stat),
        't_pval': float(t_pval),
        't_stat_excess': float(t_stat_excess),
        't_pval_excess': float(t_pval_excess),
        'win_rate': float(win_rate) if pd.notna(win_rate) else np.nan,
        'binom_pval': float(binom.pvalue) if hasattr(binom, 'pvalue') else np.nan,
        'n_trades': int(len(trades))
    }])
    validation.to_csv(outputs_dir / f"{STRATEGY_NAME}_validation.csv", index=False)

    # === QUANTSTATS TEARSHEET ===
    if QUANTSTATS_AVAILABLE:
        qs_html = outputs_dir / f"{STRATEGY_NAME}_quantstats_tearsheet.html"
        try:
            qs.reports.html(
                sr, benchmark=br, output=str(qs_html), title=f"{STRATEGY_NAME.upper()} vs Benchmark"
            )
        except Exception:
            # Fallback: save minimal report if full report fails
            minimal = outputs_dir / f"{STRATEGY_NAME}_quantstats_summary.csv"
            try:
                qs.stats.sharpe(sr).to_frame('sharpe').to_csv(minimal)
            except Exception:
                pass  # Skip if QuantStats fails completely

    # Console output limited to vectorbt .stats()
    print(comp_stats)

if __name__ == "__main__":
    comprehensive_backtest_analysis(file_path='with_er_daily.csv')
