# Tactical Asset Allocation Backtesting with VectorBT
# Strategy 1: Top-2 3/6/12m blend, InvVol, Cap60%, Breadth≥2, Duration RO, 20% GSCI overlay
# Strategy 2: Top-3 1/3/12m blend, InvVol, Cap50%, Breadth≥3, Strict Duration vs Cash

import pandas as pd
import numpy as np
import vectorbt as vbt
import os
from pathlib import Path
import quantstats as qs

def main():
    # ---------------------
    # Load data
    # ---------------------
    script_dir = Path(__file__).parent.absolute()
    csv_path = script_dir / "processed data" / "asset total returns series from the 1970s.csv"
    
    if not csv_path.exists():
        print(f"Error: Data file not found at {csv_path}")
        return
    
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.set_index(date_col).sort_index()
    X = df.select_dtypes(include=[float, int]).ffill()
    rets = X.pct_change().fillna(0.0)
    cols = list(X.columns)
    
    print(f"Data loaded: {X.shape[0]} rows, {X.shape[1]} columns")
    print(f"Date range: {X.index[0].date()} to {X.index[-1].date()}")
    
    # ---------------------
    # Detect key columns
    # ---------------------
    def find_col(substrs, default=None):
        for c in cols:
            lc = c.lower()
            if any(s in lc for s in substrs):
                return c
        return default
    
    benchmark = "Russell 2000 Net TR"
    r2000 = find_col(["russell 2000"])
    reits = find_col(["reit"])
    gsci = "S&P GSCI Total Return" if "S&P GSCI Total Return" in cols else find_col(["gsci","commod"])
    credit = "Bloomberg US Credit TR" if "Bloomberg US Credit TR" in cols else find_col(["credit"])
    t_bill = find_col(["t-bill","t bill","3m","bill"])
    treasury_long = find_col(["treasury tr"])
    aggregate = "Bloomberg US Aggregate TR" if "Bloomberg US Aggregate TR" in cols else find_col(["aggregate tr","agg tr"])
    
    safe_cash = [c for c in [t_bill] if c]
    safe_duration = [c for c in [treasury_long, aggregate] if c]
    risk_cols = [c for c in cols if c not in safe_cash and c not in safe_duration]
    if credit and credit not in risk_cols:
        risk_cols.append(credit)
    
    print(f"\nAsset classification:")
    print(f"Benchmark: {benchmark}")
    print(f"Risk assets: {risk_cols}")
    print(f"Safe duration: {safe_duration}")
    print(f"Safe cash: {safe_cash}")
    
    # ---------------------
    # Weekly schedule
    # ---------------------
    decision_days = X.resample("W-FRI").last().index
    di = X.index
    start_days = pd.DatetimeIndex([di[di.searchsorted(d, side="right")] for d in decision_days if di.searchsorted(d, side="right") < len(di)])
    end_days = list(start_days[1:]) + [di[-1]]
    
    print(f"Weekly rebalancing: {len(decision_days)} decision points")
    
    # ---------------------
    # Precompute momentum/vol
    # ---------------------
    def roll_momentum(df_levels, cols, lb):
        return df_levels[cols].pct_change().add(1).rolling(lb).apply(np.prod, raw=True)-1.0
    
    print("Computing momentum and volatility metrics...")
    mom21 = roll_momentum(X, risk_cols, 21)
    mom63 = roll_momentum(X, risk_cols, 63)
    mom126 = roll_momentum(X, risk_cols, 126)
    mom252 = roll_momentum(X, risk_cols, 252)
    vol60 = rets[risk_cols].rolling(60).std(ddof=0)
    
    roll_max_60 = X[risk_cols].rolling(60, min_periods=1).max()
    dd60 = X[risk_cols]/roll_max_60 - 1.0
    
    def treasury_mom(lb):
        if treasury_long is None: return None
        return X[treasury_long].pct_change().add(1).rolling(lb).apply(np.prod, raw=True)-1.0
    t6 = treasury_mom(126)
    t12 = treasury_mom(252)
    
    agg6 = X[aggregate].pct_change().add(1).rolling(126).apply(np.prod, raw=True)-1.0 if aggregate else None
    agg12 = X[aggregate].pct_change().add(1).rolling(252).apply(np.prod, raw=True)-1.0 if aggregate else None
    
    # ---------------------
    # Strategy 1
    # ---------------------
    def run_strategy_1():
        """
        Top-2 by 3/6/12m blend (equal), inverse-vol, cap=60%, breadth≥2 (6m>0 count),
        risk-off to duration; +20% GSCI overlay when trending.
        """
        print("Running Strategy 1...")
        blended = (mom63 + mom126 + mom252) / 3.0
        long_row_df = mom252
        breadth_df = mom126
    
        out = pd.Series(0.0, index=di)
        for d, s, e in zip(decision_days, start_days, end_days):
            pos = di.searchsorted(d, side="right") - 1
            if pos < 0: continue
            de = di[pos]
            if de not in blended.index: continue
    
            row = blended.loc[de].dropna()
            long_row = long_row_df.loc[de].dropna()
            br_row = breadth_df.loc[de].dropna()
    
            breadth_count = int((br_row > 0).sum())
            go_risk = breadth_count >= 2
    
            weights = {}
            if go_risk:
                ranked = row.sort_values(ascending=False).index.tolist()
                eligible = [c for c in ranked if long_row.get(c, -1e9) > 0]
                winners = eligible[:2]
                if len(winners) == 0:
                    go_risk = False
                else:
                    vs = vol60.loc[de, winners].replace(0, np.nan).dropna()
                    if len(vs) == 0:
                        for c in winners: weights[c] = 1.0/len(winners)
                    else:
                        inv = 1.0/vs; raw = inv/inv.sum()
                        for c in winners: weights[c] = float(raw.get(c, 0.0))
                    # 20% GSCI overlay
                    if gsci and gsci not in winners and long_row.get(gsci, 0.0) > 0:
                        ow = 0.20
                        for k in list(weights.keys()): weights[k] *= (1.0 - ow)
                        weights[gsci] = weights.get(gsci, 0.0) + ow
    
            if not go_risk:
                if len(safe_duration) > 0:
                    w = 1.0/len(safe_duration); weights = {c: w for c in safe_duration}
                elif len(safe_cash) > 0:
                    w = 1.0/len(safe_cash); weights = {c: w for c in safe_cash}
    
            # cap=60% and renormalize
            if weights:
                for k in list(weights.keys()): weights[k] = min(weights[k], 0.60)
                ssum = sum(weights.values())
                if ssum > 1e-12:
                    for k in list(weights.keys()): weights[k] = weights[k]/ssum
    
            win = rets.loc[s:e]
            out.loc[win.index] = sum(win[k]*w for k, w in weights.items()) if weights else 0.0
        return out
    
    # ---------------------
    # Strategy 2
    # ---------------------
    def run_strategy_2():
        """
        Top-3 by 1/3/12m blend (0.2/0.4/0.4), breadth≥3 & median>0, inverse-vol, cap=50%,
        REIT dd60>-12% and long>0, IG only if Treasury 6m & 12m >0, adaptive GSCI 10–25%,
        risk-off: duration only if both t6 & t12>0 else T-Bills; prefer stronger of Treasury vs Aggregate.
        """
        print("Running Strategy 2...")
        mom_blend = 0.2*mom21 + 0.4*mom63 + 0.4*mom252
        long_row_df = mom252
    
        out = pd.Series(0.0, index=di)
        for d, s, e in zip(decision_days, start_days, end_days):
            pos = di.searchsorted(d, side="right") - 1
            if pos < 0: continue
            de = di[pos]
            if de not in mom_blend.index: continue
    
            row = mom_blend.loc[de].dropna()
            long_row = long_row_df.loc[de].dropna()
    
            breadth = int((row > 0).sum())
            median_ok = row.median() > 0
            go_risk = (breadth >= 3) and median_ok
    
            ranked = row.sort_values(ascending=False).index.tolist()
            eligible = []
            for c in ranked:
                if long_row.get(c, -1e9) <= 0:
                    continue
                # REIT guard
                if reits and c == reits and dd60.loc[de, c] <= -0.12:
                    continue
                # IG guard
                if credit and c == credit and (t6 is None or t12 is None or t6.loc[de] <= 0 or t12.loc[de] <= 0):
                    continue
                eligible.append(c)
    
            winners = eligible[:3] if go_risk else []
            weights = {}
    
            if winners:
                vs = vol60.loc[de, winners].replace(0, np.nan).dropna()
                if len(vs) == 0:
                    for c in winners: weights[c] = 1.0/len(winners)
                else:
                    inv = 1.0/vs; raw = inv/inv.sum()
                    for c in winners: weights[c] = float(raw.get(c, 0.0))
                # Adaptive GSCI overlay 10–25%
                if gsci and gsci not in winners and long_row.get(gsci, 0.0) > 0:
                    strength = max(0.0, min(1.0, long_row.get(gsci, 0.0)/0.2))
                    ow = 0.10 + (0.25 - 0.10) * strength
                    for k in list(weights.keys()): weights[k] *= (1.0 - ow)
                    weights[gsci] = weights.get(gsci, 0.0) + ow
            else:
                # strict duration-vs-cash
                use_duration = (t6 is not None and t12 is not None and t6.loc[de] > 0 and t12.loc[de] > 0)
                if use_duration and len(safe_duration) > 0:
                    if treasury_long and aggregate and agg6 is not None and agg12 is not None:
                        tscore = (t6.loc[de] if de in t6.index else 0) + (t12.loc[de] if de in t12.index else 0)
                        ascore = (agg6.loc[de] if de in agg6.index else 0) + (agg12.loc[de] if de in agg12.index else 0)
                        pick = treasury_long if tscore > ascore and treasury_long in safe_duration else (aggregate if aggregate in safe_duration else safe_duration[0])
                        weights = {pick: 1.0}
                    else:
                        w = 1.0/len(safe_duration); weights = {c: w for c in safe_duration}
                else:
                    if len(safe_cash) > 0:
                        w = 1.0/len(safe_cash); weights = {c: w for c in safe_cash}
    
            # cap=50% and renormalize
            if weights:
                for k in list(weights.keys()): weights[k] = min(weights[k], 0.50)
                ssum = sum(weights.values())
                if ssum > 1e-12:
                    for k in list(weights.keys()): weights[k] = weights[k]/ssum
    
            win = rets.loc[s:e]
            out.loc[win.index] = sum(win[k]*w for k, w in weights.items()) if weights else 0.0
        return out
    
    # ---------------------
    # Run strategies and create vectorbt portfolios
    # ---------------------
    sr1 = run_strategy_1()
    sr2 = run_strategy_2()
    benchmark_returns = rets[benchmark].dropna()
    
    print("Creating VectorBT portfolios...")
    
    # Align all series
    common_dates = sr1.index.intersection(sr2.index).intersection(benchmark_returns.index)
    sr1_aligned = sr1.loc[common_dates]
    sr2_aligned = sr2.loc[common_dates]
    bench_aligned = benchmark_returns.loc[common_dates]
    
    # Create price series for close prices (cumulative returns starting at 100)
    sr1_prices = (1 + sr1_aligned).cumprod() * 100
    sr2_prices = (1 + sr2_aligned).cumprod() * 100
    bench_prices = (1 + bench_aligned).cumprod() * 100
    
    # Create simple entry/exit signals (always invested when returns != 0)
    entries1 = sr1_aligned != 0
    exits1 = sr1_aligned == 0
    
    entries2 = sr2_aligned != 0
    exits2 = sr2_aligned == 0
    
    # Skip VectorBT - it's overcomplicating things. Use direct returns.
    print("\n" + "="*80)
    print("STRATEGY PERFORMANCE (USING ACTUAL RETURNS)")
    print("="*80)
    
    def print_strategy_stats(returns, name):
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        annualized_vol = returns.std() * np.sqrt(252)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns / rolling_max - 1)
        max_dd = drawdown.min()
        
        calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0
        
        print(f"\n{name}")
        print("-" * 50)
        print(f"Total Return: {total_return:.2%}")
        print(f"CAGR: {annualized_return:.2%}")
        print(f"Annualized Volatility: {annualized_vol:.2%}")
        print(f"Sharpe Ratio: {sharpe:.3f}")
        print(f"Max Drawdown: {max_dd:.2%}")
        print(f"Calmar Ratio: {calmar:.3f}")
        print(f"Data points: {len(returns)}")
        print(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
        
        return {
            'total_return': total_return,
            'cagr': annualized_return,
            'volatility': annualized_vol,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'calmar': calmar
        }
    
    stats1 = print_strategy_stats(sr1_aligned, "STRATEGY 1 - Top-2 3/6/12m")
    stats2 = print_strategy_stats(sr2_aligned, "STRATEGY 2 - Top-3 1/3/12m")
    stats_bench = print_strategy_stats(bench_aligned, f"BENCHMARK - {benchmark}")
    
    # Generate QuantStats tearsheets with correct data
    print("\n" + "="*80)
    print("GENERATING QUANTSTATS TEARSHEETS (CORRECTED)")
    print("="*80)
    
    # Create outputs directory
    outputs_dir = script_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    # Generate tearsheets for each strategy
    strategy1_path = outputs_dir / "tactical_strategy1_quantstats_tearsheet.html"
    strategy2_path = outputs_dir / "tactical_strategy2_quantstats_tearsheet.html"
    benchmark_path = outputs_dir / "tactical_benchmark_quantstats_tearsheet.html"
    
    try:
        # Print QuantStats metrics for debugging
        print(f"\n📊 QUANTSTATS METRICS - STRATEGY 1:")
        print("=" * 60)
        qs1_stats = qs.stats.to_drawdown_series(sr1_aligned)
        
        print(f"Start Period: {sr1_aligned.index[0].date()}")
        print(f"End Period: {sr1_aligned.index[-1].date()}")
        print(f"Risk-Free Rate: 0.0%")
        print(f"Time in Market: {qs.stats.exposure(sr1_aligned):.1%}")
        print(f"Cumulative Return: {qs.stats.comp(sr1_aligned):.2%}")
        print(f"CAGR: {qs.stats.cagr(sr1_aligned):.2%}")
        print(f"Sharpe: {qs.stats.sharpe(sr1_aligned):.2f}")
        print(f"Prob. Sharpe Ratio: {qs.stats.probabilistic_sharpe_ratio(sr1_aligned):.1%}")
        print(f"Smart Sharpe: {qs.stats.smart_sharpe(sr1_aligned):.2f}")
        print(f"Sortino: {qs.stats.sortino(sr1_aligned):.2f}")
        print(f"Smart Sortino: {qs.stats.smart_sortino(sr1_aligned):.2f}")
        print(f"Sortino/√2: {qs.stats.sortino(sr1_aligned)/np.sqrt(2):.2f}")
        print(f"Smart Sortino/√2: {qs.stats.smart_sortino(sr1_aligned)/np.sqrt(2):.2f}")
        print(f"Omega: {qs.stats.omega(sr1_aligned):.2f}")
        print(f"Max Drawdown: {qs.stats.max_drawdown(sr1_aligned):.2%}")
        print(f"Longest DD Days: {qs.stats.longest_dd_days(sr1_aligned)}")
        print(f"Volatility (ann.): {qs.stats.volatility(sr1_aligned):.2%}")
        print(f"R²: {qs.stats.r_squared(sr1_aligned, bench_aligned):.2f}")
        print(f"Information Ratio: {qs.stats.information_ratio(sr1_aligned, bench_aligned):.2f}")
        print(f"Calmar: {qs.stats.calmar(sr1_aligned):.2f}")
        print(f"Skew: {qs.stats.skew(sr1_aligned):.2f}")
        print(f"Kurtosis: {qs.stats.kurtosis(sr1_aligned):.2f}")
        print(f"Expected Daily: {qs.stats.expected_return(sr1_aligned, aggregate_returns='daily'):.2%}")
        print(f"Expected Monthly: {qs.stats.expected_return(sr1_aligned, aggregate_returns='monthly'):.2%}")
        print(f"Expected Yearly: {qs.stats.expected_return(sr1_aligned, aggregate_returns='yearly'):.2%}")
        print(f"Kelly Criterion: {qs.stats.kelly_criterion(sr1_aligned):.2%}")
        print(f"Risk of Ruin: {qs.stats.risk_of_ruin(sr1_aligned):.1%}")
        print(f"Daily Value-at-Risk: {qs.stats.value_at_risk(sr1_aligned):.2%}")
        print(f"Expected Shortfall (cVaR): {qs.stats.conditional_value_at_risk(sr1_aligned):.2%}")
        
        print(f"Generating Strategy 1 tearsheet: {strategy1_path}")
        qs.reports.html(
            sr1_aligned, 
            benchmark=bench_aligned,
            output=str(strategy1_path),
            title="Tactical Asset Allocation Strategy 1 - Top-2 3/6/12m (CORRECTED)"
        )
        
        # Print QuantStats metrics for Strategy 2
        print(f"\n📊 QUANTSTATS METRICS - STRATEGY 2:")
        print("=" * 60)
        
        print(f"Start Period: {sr2_aligned.index[0].date()}")
        print(f"End Period: {sr2_aligned.index[-1].date()}")
        print(f"Risk-Free Rate: 0.0%")
        print(f"Time in Market: {qs.stats.exposure(sr2_aligned):.1%}")
        print(f"Cumulative Return: {qs.stats.comp(sr2_aligned):.2%}")
        print(f"CAGR: {qs.stats.cagr(sr2_aligned):.2%}")
        print(f"Sharpe: {qs.stats.sharpe(sr2_aligned):.2f}")
        print(f"Prob. Sharpe Ratio: {qs.stats.probabilistic_sharpe_ratio(sr2_aligned):.1%}")
        print(f"Smart Sharpe: {qs.stats.smart_sharpe(sr2_aligned):.2f}")
        print(f"Sortino: {qs.stats.sortino(sr2_aligned):.2f}")
        print(f"Smart Sortino: {qs.stats.smart_sortino(sr2_aligned):.2f}")
        print(f"Sortino/√2: {qs.stats.sortino(sr2_aligned)/np.sqrt(2):.2f}")
        print(f"Smart Sortino/√2: {qs.stats.smart_sortino(sr2_aligned)/np.sqrt(2):.2f}")
        print(f"Omega: {qs.stats.omega(sr2_aligned):.2f}")
        print(f"Max Drawdown: {qs.stats.max_drawdown(sr2_aligned):.2%}")
        print(f"Longest DD Days: {qs.stats.longest_dd_days(sr2_aligned)}")
        print(f"Volatility (ann.): {qs.stats.volatility(sr2_aligned):.2%}")
        print(f"R²: {qs.stats.r_squared(sr2_aligned, bench_aligned):.2f}")
        print(f"Information Ratio: {qs.stats.information_ratio(sr2_aligned, bench_aligned):.2f}")
        print(f"Calmar: {qs.stats.calmar(sr2_aligned):.2f}")
        print(f"Skew: {qs.stats.skew(sr2_aligned):.2f}")
        print(f"Kurtosis: {qs.stats.kurtosis(sr2_aligned):.2f}")
        print(f"Expected Daily: {qs.stats.expected_return(sr2_aligned, aggregate_returns='daily'):.2%}")
        print(f"Expected Monthly: {qs.stats.expected_return(sr2_aligned, aggregate_returns='monthly'):.2%}")
        print(f"Expected Yearly: {qs.stats.expected_return(sr2_aligned, aggregate_returns='yearly'):.2%}")
        print(f"Kelly Criterion: {qs.stats.kelly_criterion(sr2_aligned):.2%}")
        print(f"Risk of Ruin: {qs.stats.risk_of_ruin(sr2_aligned):.1%}")
        print(f"Daily Value-at-Risk: {qs.stats.value_at_risk(sr2_aligned):.2%}")
        print(f"Expected Shortfall (cVaR): {qs.stats.conditional_value_at_risk(sr2_aligned):.2%}")
        
        print(f"Generating Strategy 2 tearsheet: {strategy2_path}")
        qs.reports.html(
            sr2_aligned, 
            benchmark=bench_aligned,
            output=str(strategy2_path),
            title="Tactical Asset Allocation Strategy 2 - Top-3 1/3/12m (CORRECTED)"
        )
        
        # Print QuantStats metrics for Benchmark
        print(f"\n📊 QUANTSTATS METRICS - BENCHMARK:")
        print("=" * 60)
        
        print(f"Start Period: {bench_aligned.index[0].date()}")
        print(f"End Period: {bench_aligned.index[-1].date()}")
        print(f"Risk-Free Rate: 0.0%")
        print(f"Time in Market: {qs.stats.exposure(bench_aligned):.1%}")
        print(f"Cumulative Return: {qs.stats.comp(bench_aligned):.2%}")
        print(f"CAGR: {qs.stats.cagr(bench_aligned):.2%}")
        print(f"Sharpe: {qs.stats.sharpe(bench_aligned):.2f}")
        print(f"Prob. Sharpe Ratio: {qs.stats.probabilistic_sharpe_ratio(bench_aligned):.1%}")
        print(f"Smart Sharpe: {qs.stats.smart_sharpe(bench_aligned):.2f}")
        print(f"Sortino: {qs.stats.sortino(bench_aligned):.2f}")
        print(f"Smart Sortino: {qs.stats.smart_sortino(bench_aligned):.2f}")
        print(f"Sortino/√2: {qs.stats.sortino(bench_aligned)/np.sqrt(2):.2f}")
        print(f"Smart Sortino/√2: {qs.stats.smart_sortino(bench_aligned)/np.sqrt(2):.2f}")
        print(f"Omega: {qs.stats.omega(bench_aligned):.2f}")
        print(f"Max Drawdown: {qs.stats.max_drawdown(bench_aligned):.2%}")
        print(f"Longest DD Days: {qs.stats.longest_dd_days(bench_aligned)}")
        print(f"Volatility (ann.): {qs.stats.volatility(bench_aligned):.2%}")
        print(f"Calmar: {qs.stats.calmar(bench_aligned):.2f}")
        print(f"Skew: {qs.stats.skew(bench_aligned):.2f}")
        print(f"Kurtosis: {qs.stats.kurtosis(bench_aligned):.2f}")
        print(f"Expected Daily: {qs.stats.expected_return(bench_aligned, aggregate_returns='daily'):.2%}")
        print(f"Expected Monthly: {qs.stats.expected_return(bench_aligned, aggregate_returns='monthly'):.2%}")
        print(f"Expected Yearly: {qs.stats.expected_return(bench_aligned, aggregate_returns='yearly'):.2%}")
        print(f"Kelly Criterion: {qs.stats.kelly_criterion(bench_aligned):.2%}")
        print(f"Risk of Ruin: {qs.stats.risk_of_ruin(bench_aligned):.1%}")
        print(f"Daily Value-at-Risk: {qs.stats.value_at_risk(bench_aligned):.2%}")
        print(f"Expected Shortfall (cVaR): {qs.stats.conditional_value_at_risk(bench_aligned):.2%}")
        
        print(f"Generating Benchmark tearsheet: {benchmark_path}")
        qs.reports.html(
            bench_aligned, 
            output=str(benchmark_path),
            title=f"Benchmark - {benchmark}"
        )
        
        print("\n✅ QuantStats tearsheets generated successfully!")
        print(f"📁 Files saved in: {outputs_dir}")
        print(f"   - {strategy1_path.name}")
        print(f"   - {strategy2_path.name}")
        print(f"   - {benchmark_path.name}")
        
        # Summary comparison table
        print(f"\n📋 SUMMARY COMPARISON TABLE:")
        print("=" * 80)
        print(f"{'Metric':<25} {'Strategy 1':<15} {'Strategy 2':<15} {'Benchmark':<15}")
        print("-" * 80)
        print(f"{'CAGR':<25} {qs.stats.cagr(sr1_aligned):<14.2%} {qs.stats.cagr(sr2_aligned):<14.2%} {qs.stats.cagr(bench_aligned):<14.2%}")
        print(f"{'Volatility':<25} {qs.stats.volatility(sr1_aligned):<14.2%} {qs.stats.volatility(sr2_aligned):<14.2%} {qs.stats.volatility(bench_aligned):<14.2%}")
        print(f"{'Sharpe Ratio':<25} {qs.stats.sharpe(sr1_aligned):<14.2f} {qs.stats.sharpe(sr2_aligned):<14.2f} {qs.stats.sharpe(bench_aligned):<14.2f}")
        print(f"{'Max Drawdown':<25} {qs.stats.max_drawdown(sr1_aligned):<14.2%} {qs.stats.max_drawdown(sr2_aligned):<14.2%} {qs.stats.max_drawdown(bench_aligned):<14.2%}")
        print(f"{'Calmar Ratio':<25} {qs.stats.calmar(sr1_aligned):<14.2f} {qs.stats.calmar(sr2_aligned):<14.2f} {qs.stats.calmar(bench_aligned):<14.2f}")
        print(f"{'Sortino Ratio':<25} {qs.stats.sortino(sr1_aligned):<14.2f} {qs.stats.sortino(sr2_aligned):<14.2f} {qs.stats.sortino(bench_aligned):<14.2f}")
        print(f"{'Total Return':<25} {qs.stats.comp(sr1_aligned):<14.2%} {qs.stats.comp(sr2_aligned):<14.2%} {qs.stats.comp(bench_aligned):<14.2%}")
        
    except Exception as e:
        print(f"Error creating VectorBT portfolios: {e}")
        print("Falling back to manual stats calculation...")
        
        # Manual stats as fallback
        def calculate_stats(returns, name):
            total_return = (1 + returns).prod() - 1
            annualized_return = (1 + returns).prod() ** (252 / len(returns)) - 1
            annualized_vol = returns.std() * np.sqrt(252)
            sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
            
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdown = (cum_returns / rolling_max - 1)
            max_dd = drawdown.min()
            
            calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0
            
            print(f"\n{name}")
            print("=" * 50)
            print(f"Total Return: {total_return:.2%}")
            print(f"Annualized Return: {annualized_return:.2%}")
            print(f"Annualized Volatility: {annualized_vol:.2%}")
            print(f"Sharpe Ratio: {sharpe:.3f}")
            print(f"Max Drawdown: {max_dd:.2%}")
            print(f"Calmar Ratio: {calmar:.3f}")
        
        calculate_stats(sr1, "STRATEGY 1")
        calculate_stats(sr2, "STRATEGY 2")
        calculate_stats(benchmark_returns, f"BENCHMARK - {benchmark}")
        
        # Generate QuantStats tearsheets even with manual stats
        print("\n" + "="*80)
        print("GENERATING QUANTSTATS TEARSHEETS")
        print("="*80)
        
        # Create outputs directory
        outputs_dir = script_dir / "outputs"
        outputs_dir.mkdir(exist_ok=True)
        
        # Generate tearsheets for each strategy
        strategy1_path = outputs_dir / "tactical_strategy1_quantstats_tearsheet.html"
        strategy2_path = outputs_dir / "tactical_strategy2_quantstats_tearsheet.html"
        benchmark_path = outputs_dir / "tactical_benchmark_quantstats_tearsheet.html"
        
        try:
            # Debug: Check data alignment and values
            print(f"\n🔍 DEBUGGING STRATEGY 1 DATA:")
            print(f"Strategy 1 returns shape: {sr1_aligned.shape}")
            print(f"Strategy 1 date range: {sr1_aligned.index[0]} to {sr1_aligned.index[-1]}")
            print(f"Strategy 1 total return (manual): {((1 + sr1_aligned).prod() - 1):.4f}")
            print(f"Strategy 1 annualized return (manual): {(((1 + sr1_aligned).prod()) ** (252 / len(sr1_aligned)) - 1):.4f}")
            print(f"Strategy 1 first 5 values: {sr1_aligned.head().values}")
            print(f"Strategy 1 last 5 values: {sr1_aligned.tail().values}")
            print(f"Strategy 1 mean daily return: {sr1_aligned.mean():.6f}")
            print(f"Strategy 1 std daily return: {sr1_aligned.std():.6f}")
            
            print(f"Generating Strategy 1 tearsheet: {strategy1_path}")
            qs.reports.html(
                sr1_aligned, 
                benchmark=bench_aligned,
                output=str(strategy1_path),
                title="Tactical Asset Allocation Strategy 1 - Top-2 3/6/12m"
            )
            
            print(f"Generating Strategy 2 tearsheet: {strategy2_path}")
            qs.reports.html(
                sr2_aligned, 
                benchmark=bench_aligned,
                output=str(strategy2_path),
                title="Tactical Asset Allocation Strategy 2 - Top-3 1/3/12m"
            )
            
            print(f"Generating Benchmark tearsheet: {benchmark_path}")
            qs.reports.html(
                bench_aligned, 
                output=str(benchmark_path),
                title=f"Benchmark - {benchmark}"
            )
            
            print("\n✅ QuantStats tearsheets generated successfully!")
            print(f"📁 Files saved in: {outputs_dir}")
            print(f"   - {strategy1_path.name}")
            print(f"   - {strategy2_path.name}")
            print(f"   - {benchmark_path.name}")
            
        except Exception as qs_error:
            print(f"❌ Error generating QuantStats tearsheets: {qs_error}")

if __name__ == "__main__":
    main()
