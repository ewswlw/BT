#!/usr/bin/env python3
"""
Deep diagnostic to understand VectorBT portfolio calculation differences.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except Exception:
    VECTORBT_AVAILABLE = False
    print("VectorBT not available")
    exit(1)

def analyze_vectorbt_portfolio_construction():
    """Analyze how VectorBT constructs portfolios vs manual calculation"""
    
    # Load data
    df = pd.read_csv('../data_pipelines/data_processed/with_er_daily.csv', parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    weekly = df.resample('W-FRI').last().dropna(how='all')
    weekly['cad_ret'] = weekly['cad_ig_er_index'].pct_change()
    
    # Create simple test signal (first approach from audit)
    strategies = {
        'CAD_on_CAD': ('>', 'cad_ig_er_index', 3),
        'CAD_on_US_HY': ('>', 'us_hy_er_index', 3),
        'CAD_on_US_IG': ('>', 'us_ig_er_index', 5),
        'CAD_on_CAD_OAS': ('<', 'cad_oas', 3),
        'CAD_on_US_IG_OAS': ('<', 'us_ig_oas', 4),
    }
    
    signals = {}
    for strat, (op, col, lb) in strategies.items():
        ma = weekly[col].rolling(lb, min_periods=lb).mean()
        cond = (weekly[col] > ma) if op == '>' else (weekly[col] < ma)
        signals[strat] = cond.shift(1).fillna(False)
    
    composite_bool = (pd.DataFrame(signals).sum(axis=1) >= 3).astype(bool)
    
    # Manual calculation
    manual_returns = (composite_bool.astype(int) * weekly['cad_ret']).dropna()
    manual_cumret = ((1 + manual_returns).cumprod().iloc[-1] - 1) * 100
    
    print("="*80)
    print("VECTORBT PORTFOLIO CONSTRUCTION ANALYSIS")
    print("="*80)
    print(f"Manual calculation total return: {manual_cumret:.2f}%")
    
    # === DIFFERENT VECTORBT APPROACHES ===
    
    # Approach 1: from_signals with entries/exits
    entries = composite_bool & ~composite_bool.shift(1).fillna(False)
    exits = ~composite_bool & composite_bool.shift(1).fillna(False)
    
    print(f"\nApproach 1: from_signals with entries/exits")
    print(f"Entries: {entries.sum()}, Exits: {exits.sum()}")
    
    pf1 = vbt.Portfolio.from_signals(
        close=weekly['cad_ig_er_index'],
        entries=entries,
        exits=exits,
        freq='W'
    )
    print(f"VBT Total Return: {pf1.total_return()*100:.2f}%")
    print(f"VBT Stats Total Return: {pf1.stats()['Total Return [%]']:.2f}%")
    
    # Approach 2: from_signals with just entries (no exits)
    print(f"\nApproach 2: from_signals with entries only")
    pf2 = vbt.Portfolio.from_signals(
        close=weekly['cad_ig_er_index'],
        entries=entries,
        freq='W'
    )
    print(f"VBT Total Return: {pf2.total_return()*100:.2f}%")
    
    # Approach 3: from_orders using composite signal directly
    print(f"\nApproach 3: from_orders using signal size")
    
    # Create order sizes based on signal
    order_size = composite_bool.astype(float)
    order_size = order_size.diff().fillna(order_size.iloc[0])  # Only trade on changes
    
    pf3 = vbt.Portfolio.from_orders(
        close=weekly['cad_ig_er_index'],
        size=order_size,
        freq='W'
    )
    print(f"VBT Total Return: {pf3.total_return()*100:.2f}%")
    
    # Approach 4: Simple position-based portfolio
    print(f"\nApproach 4: Manual position tracking")
    
    # Create a simple equity curve manually using VectorBT's approach
    prices = weekly['cad_ig_er_index'].dropna()
    positions = composite_bool.reindex(prices.index).fillna(False)
    
    # Initialize portfolio
    initial_cash = 100.0
    equity_curve = [initial_cash]
    position = 0  # shares held
    cash = initial_cash
    
    for i in range(1, len(prices)):
        prev_pos = positions.iloc[i-1]
        curr_pos = positions.iloc[i]
        prev_price = prices.iloc[i-1]
        curr_price = prices.iloc[i]
        
        # Update value from price changes
        portfolio_value = cash + position * curr_price
        
        # Check for position changes
        if prev_pos != curr_pos:
            if curr_pos and not prev_pos:  # Enter position
                # Buy with all cash
                shares_to_buy = cash / curr_price
                position += shares_to_buy
                cash = 0
            elif prev_pos and not curr_pos:  # Exit position
                # Sell all shares
                cash = position * curr_price
                position = 0
        
        portfolio_value = cash + position * curr_price
        equity_curve.append(portfolio_value)
    
    manual_portfolio_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
    print(f"Manual portfolio return: {manual_portfolio_return:.2f}%")
    
    # === DETAILED DIAGNOSTICS ===
    print(f"\n" + "="*60)
    print("DETAILED DIAGNOSTICS")
    print("="*60)
    
    # Check portfolio records
    records = pf1.orders.records_readable
    if len(records) > 0:
        print(f"First 5 trades:")
        print(records.head()[['Column', 'Timestamp', 'Side', 'Size', 'Price', 'Fees']])
        
        print(f"\nTotal orders: {len(records)}")
        print(f"Buy orders: {len(records[records['Side'] == 'Buy'])}")
        print(f"Sell orders: {len(records[records['Side'] == 'Sell'])}")
    
    # Check returns calculation
    pf1_returns = pf1.returns()
    print(f"\nVectorBT returns stats:")
    print(f"Mean return: {pf1_returns.mean():.6f}")
    print(f"Cumulative return (manual): {((1 + pf1_returns).cumprod().iloc[-1] - 1)*100:.2f}%")
    
    # Compare with our signal-based returns
    print(f"\nSignal-based returns stats:")
    print(f"Mean return: {manual_returns.mean():.6f}")
    print(f"Returns correlation: {manual_returns.corr(pf1_returns.reindex(manual_returns.index)):.4f}")
    
    # Show some sample periods
    print(f"\nSample comparison (first 10 periods with positions):")
    pos_periods = composite_bool[composite_bool].head(10).index
    comparison_df = pd.DataFrame({
        'Position': composite_bool.reindex(pos_periods),
        'CAD_Return': weekly['cad_ret'].reindex(pos_periods),
        'Manual_Strat_Return': (composite_bool.astype(int) * weekly['cad_ret']).reindex(pos_periods),
        'VBT_Return': pf1_returns.reindex(pos_periods)
    })
    print(comparison_df)

if __name__ == "__main__":
    analyze_vectorbt_portfolio_construction()
