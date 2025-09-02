#!/usr/bin/env python3
"""
Audit to compare results between original code and VectorBT implementation.
This will help identify why total returns differ between the two approaches.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import math
from pathlib import Path
from scipy.stats import ttest_1samp, binomtest

# VectorBT import
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except Exception:
    VECTORBT_AVAILABLE = False
    print("Warning: VectorBT not available")

def original_approach(file_path='../data_pipelines/data_processed/with_er_daily.csv'):
    """Original approach from the provided code"""
    print("="*80)
    print("ORIGINAL APPROACH ANALYSIS")
    print("="*80)
    
    # === DATA PREPARATION ===
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    weekly = df.resample('W-FRI').last().dropna(how='all')
    weekly['cad_ret'] = weekly['cad_ig_er_index'].pct_change()
    
    print(f"Data loaded: {weekly.shape}")
    print(f"Date range: {weekly.index[0]} to {weekly.index[-1]}")
    
    def eval_strategy(signal, returns):
        strat_ret = signal * returns
        valid = strat_ret.dropna()
        if len(valid) < 10: return None
        
        cum = (valid+1).prod()-1
        years = len(valid)/52
        cagr = (1+cum)**(1/years)-1 if years>0 else np.nan
        vol = valid.std()*math.sqrt(52)
        sharpe = (valid.mean()*52)/vol if vol!=0 else np.nan
        curve = (valid+1).cumprod()
        peak = curve.cummax()
        maxdd = ((curve-peak)/peak).min()
        
        return {
            'Cumulative_Return_%': round(cum*100,2),
            'CAGR_%': round(cagr*100,2),
            'AnnVol_%': round(vol*100,2),
            'Sharpe': round(sharpe,2),
            'MaxDD_%': round(maxdd*100,2),
            'Avg_Exposure_%': round(signal.mean()*100,2)
        }
    
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
    
    optimal_lookbacks = {}
    for strat, (op, col) in strategies.items():
        best_cum, best_lb = -999, None
        for lb in range(2, 21):
            ma = weekly[col].rolling(lb, min_periods=lb).mean()
            cond = (weekly[col] > ma) if op == '>' else (weekly[col] < ma)
            signal = cond.shift(1).fillna(False).astype(int)  # 1-week lag
            result = eval_strategy(signal, weekly['cad_ret'])
            if result and result['Cumulative_Return_%'] > best_cum:
                best_cum = result['Cumulative_Return_%']
                best_lb = lb
        optimal_lookbacks[strat] = (best_lb, op, col)
    
    print("\nOptimal lookbacks (Original):")
    for strat, (lb, op, col) in optimal_lookbacks.items():
        print(f"  {strat}: {lb} weeks ({col} {op} MA)")
    
    # === STEP 2: BUILD COMPOSITE STRATEGY ===
    top5 = ['CAD_on_CAD', 'CAD_on_US_HY', 'CAD_on_US_IG', 'CAD_on_CAD_OAS', 'CAD_on_US_IG_OAS']
    
    signals = {}
    for strat in top5:
        lb, op, col = optimal_lookbacks[strat]
        ma = weekly[col].rolling(lb, min_periods=lb).mean()
        cond = (weekly[col] > ma) if op == '>' else (weekly[col] < ma)
        signals[strat] = cond.shift(1).fillna(False).astype(int)
    
    # Composite: at least 3 of 5 signals bullish
    composite_signal = (sum(signals.values()) >= 3).astype(int)
    
    # === STEP 3: PERFORMANCE COMPARISON ===
    results = {}
    results['Composite_Top5_AtLeast3'] = eval_strategy(composite_signal, weekly['cad_ret'])
    
    # Buy & Hold
    bh_ret = weekly['cad_ret'].dropna()
    results['Buy_and_Hold'] = eval_strategy(pd.Series(1, index=bh_ret.index), bh_ret)
    
    print("\nOriginal Results:")
    df_results = pd.DataFrame(results).T
    print(df_results.to_string())
    
    return weekly, composite_signal, results, optimal_lookbacks

def vectorbt_approach(weekly_data, optimal_lookbacks_orig):
    """VectorBT approach using same data and lookbacks as original"""
    print("\n" + "="*80)
    print("VECTORBT APPROACH ANALYSIS")
    print("="*80)
    
    if not VECTORBT_AVAILABLE:
        print("VectorBT not available, skipping comparison")
        return None
    
    weekly = weekly_data.copy()
    
    # Use the same optimal lookbacks from original approach
    top5 = ['CAD_on_CAD', 'CAD_on_US_HY', 'CAD_on_US_IG', 'CAD_on_CAD_OAS', 'CAD_on_US_IG_OAS']
    
    signals = {}
    print("\nUsing same optimal lookbacks as original:")
    for strat in top5:
        lb, op, col = optimal_lookbacks_orig[strat]
        print(f"  {strat}: {lb} weeks ({col} {op} MA)")
        ma = weekly[col].rolling(lb, min_periods=lb).mean()
        cond = (weekly[col] > ma) if op == '>' else (weekly[col] < ma)
        signals[strat] = cond.shift(1).fillna(False)
    
    # Composite: at least 3 of 5 signals bullish
    composite_bool = (pd.DataFrame(signals).sum(axis=1) >= 3).astype(bool)
    
    # Entries/exits for vbt
    entries = composite_bool & ~composite_bool.shift(1).fillna(False)
    exits = ~composite_bool & composite_bool.shift(1).fillna(False)
    
    print(f"\nVectorBT Signal Analysis:")
    print(f"  Composite signal periods: {composite_bool.sum()}")
    print(f"  Entry signals: {entries.sum()}")
    print(f"  Exit signals: {exits.sum()}")
    
    # === VECTORBT PORTFOLIOS ===
    pf = vbt.Portfolio.from_signals(
        close=weekly['cad_ig_er_index'],
        entries=entries,
        exits=exits,
        freq='W'
    )
    pf_bh = vbt.Portfolio.from_holding(
        close=weekly['cad_ig_er_index'],
        freq='W'
    )
    
    # Get stats
    comp_stats = pf.stats()
    bh_stats = pf_bh.stats()
    
    print(f"\nVectorBT Results:")
    print(f"Strategy Total Return: {comp_stats['Total Return [%]']:.2f}%")
    print(f"Benchmark Total Return: {bh_stats['Total Return [%]']:.2f}%")
    print(f"Strategy Sharpe: {comp_stats['Sharpe Ratio']:.2f}")
    print(f"Benchmark Sharpe: {bh_stats['Sharpe Ratio']:.2f}")
    print(f"Strategy Max DD: {comp_stats['Max Drawdown [%]']:.2f}%")
    print(f"Benchmark Max DD: {bh_stats['Max Drawdown [%]']:.2f}%")
    
    return pf, pf_bh, composite_bool

def detailed_comparison(weekly_data, composite_signal_orig, vectorbt_signal):
    """Detailed comparison of signals and returns"""
    print("\n" + "="*80)
    print("DETAILED SIGNAL COMPARISON")
    print("="*80)
    
    weekly = weekly_data.copy()
    
    # Compare signals
    orig_positions = composite_signal_orig.astype(bool)
    vbt_positions = vectorbt_signal
    
    print(f"Original signal periods: {orig_positions.sum()}")
    print(f"VectorBT signal periods: {vbt_positions.sum()}")
    print(f"Signal agreement: {(orig_positions == vbt_positions).sum()} / {len(orig_positions)} ({(orig_positions == vbt_positions).mean()*100:.1f}%)")
    
    # Check differences
    differences = orig_positions != vbt_positions
    if differences.any():
        print(f"\nSignal differences found: {differences.sum()} periods")
        diff_periods = weekly[differences].index[:10]  # Show first 10
        print(f"First few difference dates: {list(diff_periods)}")
    
    # Calculate returns using both approaches
    orig_strategy_returns = (composite_signal_orig * weekly['cad_ret']).dropna()
    vbt_strategy_returns = (vbt_positions.astype(int) * weekly['cad_ret']).dropna()
    
    print(f"\nReturn Statistics Comparison:")
    print(f"Original approach cumulative return: {((1 + orig_strategy_returns).cumprod().iloc[-1] - 1)*100:.2f}%")
    print(f"VectorBT approach cumulative return: {((1 + vbt_strategy_returns).cumprod().iloc[-1] - 1)*100:.2f}%")
    
    print(f"Original approach mean return: {orig_strategy_returns.mean()*52*100:.2f}% annually")
    print(f"VectorBT approach mean return: {vbt_strategy_returns.mean()*52*100:.2f}% annually")
    
    # Check price data used
    print(f"\nPrice Data Check:")
    print(f"CAD IG ER Index - First 5 values: {weekly['cad_ig_er_index'].head().values}")
    print(f"CAD IG ER Index - Last 5 values: {weekly['cad_ig_er_index'].tail().values}")
    print(f"CAD returns - First 5 non-null values: {weekly['cad_ret'].dropna().head().values}")
    
    return orig_strategy_returns, vbt_strategy_returns

def main():
    """Main audit function"""
    print("AUDIT: Comparing Original vs VectorBT Implementation Results")
    print("="*80)
    
    # Run original approach
    weekly_data, composite_signal_orig, orig_results, optimal_lookbacks_orig = original_approach()
    
    # Run VectorBT approach with same parameters
    if VECTORBT_AVAILABLE:
        pf, pf_bh, vectorbt_signal = vectorbt_approach(weekly_data, optimal_lookbacks_orig)
        
        # Detailed comparison
        orig_returns, vbt_returns = detailed_comparison(weekly_data, composite_signal_orig, vectorbt_signal)
        
        print("\n" + "="*80)
        print("SUMMARY OF DIFFERENCES")
        print("="*80)
        
        orig_total = orig_results['Composite_Top5_AtLeast3']['Cumulative_Return_%']
        vbt_total = pf.stats()['Total Return [%]']
        
        print(f"Original Total Return: {orig_total}%")
        print(f"VectorBT Total Return: {vbt_total:.2f}%")
        print(f"Difference: {abs(orig_total - vbt_total):.2f} percentage points")
        
        if abs(orig_total - vbt_total) > 0.1:
            print("\n⚠️  SIGNIFICANT DIFFERENCE DETECTED!")
            print("Possible causes:")
            print("1. Different signal calculation logic")
            print("2. Different return calculation methods")
            print("3. Different handling of entry/exit timing")
            print("4. Different portfolio construction approach")
        else:
            print("\n✅ Results are very similar - difference likely due to rounding/methodology")
    
    else:
        print("VectorBT not available - cannot perform comparison")

if __name__ == "__main__":
    main()
