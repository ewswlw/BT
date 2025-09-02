#!/usr/bin/env python3
"""
Investigate VectorBT parameters to control signal timing behavior.
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

def investigate_vectorbt_parameters():
    """Investigate different VectorBT parameters for signal timing"""
    
    # Load minimal test data
    df = pd.read_csv('../data_pipelines/data_processed/with_er_daily.csv', parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    weekly = df.resample('W-FRI').last().dropna(how='all')
    weekly['cad_ret'] = weekly['cad_ig_er_index'].pct_change()
    
    # Create simple test signal
    ma = weekly['cad_ig_er_index'].rolling(3, min_periods=3).mean()
    cond = weekly['cad_ig_er_index'] > ma
    signal = cond.shift(1).fillna(False)  # 1-week lag
    
    # Manual calculation (our reference)
    manual_returns = (signal.astype(int) * weekly['cad_ret']).dropna()
    manual_total = ((1 + manual_returns).cumprod().iloc[-1] - 1) * 100
    
    print("="*80)
    print("VECTORBT PARAMETER INVESTIGATION")
    print("="*80)
    print(f"Reference (manual calculation): {manual_total:.2f}%")
    print("\nTesting different VectorBT parameters...")
    
    # Test 1: Basic from_signals
    entries = signal & ~signal.shift(1).fillna(False)
    exits = ~signal & signal.shift(1).fillna(False)
    
    print(f"\n1. Basic from_signals (entries/exits):")
    pf1 = vbt.Portfolio.from_signals(
        close=weekly['cad_ig_er_index'],
        entries=entries,
        exits=exits,
        freq='W'
    )
    print(f"   Result: {pf1.total_return()*100:.2f}%")
    
    # Test 2: Check if there's a signal_type parameter
    try:
        print(f"\n2. Testing signal_type parameter:")
        pf2 = vbt.Portfolio.from_signals(
            close=weekly['cad_ig_er_index'],
            entries=entries,
            exits=exits,
            freq='W',
            signal_type='entries_exits'  # Try explicit signal type
        )
        print(f"   Result: {pf2.total_return()*100:.2f}%")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Try upon_* parameters
    try:
        print(f"\n3. Testing upon_long_conflict parameter:")
        pf3 = vbt.Portfolio.from_signals(
            close=weekly['cad_ig_er_index'],
            entries=entries,
            exits=exits,
            freq='W',
            upon_long_conflict='keep'  # or 'ignore', 'remove'
        )
        print(f"   Result: {pf3.total_return()*100:.2f}%")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 4: Try different signal forms
    print(f"\n4. Testing signal as position sizes:")
    try:
        # Use signal directly as position indicator
        pf4 = vbt.Portfolio.from_signals(
            close=weekly['cad_ig_er_index'],
            entries=signal,  # Use signal directly
            freq='W'
        )
        print(f"   Result: {pf4.total_return()*100:.2f}%")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 5: Try from_orders with immediate execution
    print(f"\n5. Testing from_orders with position changes:")
    try:
        # Create order sizes that change position immediately
        position_changes = signal.astype(float).diff().fillna(signal.iloc[0])
        
        pf5 = vbt.Portfolio.from_orders(
            close=weekly['cad_ig_er_index'],
            size=position_changes,
            freq='W'
        )
        print(f"   Result: {pf5.total_return()*100:.2f}%")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 6: Try from_orders with all cash on signal
    print(f"\n6. Testing from_orders with 'all' cash:")
    try:
        # When signal is true, invest all cash; when false, exit
        order_sizes = pd.Series(index=signal.index, dtype=float)
        order_sizes[signal & ~signal.shift(1).fillna(False)] = 1.0  # Buy all
        order_sizes[~signal & signal.shift(1).fillna(False)] = -1.0  # Sell all
        order_sizes = order_sizes.fillna(0)
        
        pf6 = vbt.Portfolio.from_orders(
            close=weekly['cad_ig_er_index'],
            size=order_sizes,
            size_type='percent',  # Percentage of current value
            freq='W'
        )
        print(f"   Result: {pf6.total_return()*100:.2f}%")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 7: Check VectorBT documentation for timing parameters
    print(f"\n7. Checking Portfolio.from_signals parameters:")
    try:
        # Get the function signature
        import inspect
        sig = inspect.signature(vbt.Portfolio.from_signals)
        print(f"   Parameters: {list(sig.parameters.keys())}")
        
        # Check for timing-related parameters
        timing_params = [p for p in sig.parameters.keys() if 'time' in p.lower() or 'delay' in p.lower() or 'lag' in p.lower()]
        if timing_params:
            print(f"   Timing-related parameters found: {timing_params}")
        else:
            print(f"   No obvious timing-related parameters found")
            
    except Exception as e:
        print(f"   Error inspecting parameters: {e}")
    
    # Test 8: Try call_seq parameter
    try:
        print(f"\n8. Testing call_seq parameter:")
        pf8 = vbt.Portfolio.from_signals(
            close=weekly['cad_ig_er_index'],
            entries=entries,
            exits=exits,
            freq='W',
            call_seq='default'  # or 'auto'
        )
        print(f"   Result: {pf8.total_return()*100:.2f}%")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 9: Try different accumulation modes
    try:
        print(f"\n9. Testing accumulate parameter:")
        pf9 = vbt.Portfolio.from_signals(
            close=weekly['cad_ig_er_index'],
            entries=entries,
            exits=exits,
            freq='W',
            accumulate=True  # Accumulate positions
        )
        print(f"   Result: {pf9.total_return()*100:.2f}%")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 10: Manual adjustment - shift signals forward
    print(f"\n10. Manual timing adjustment - shift signals forward:")
    try:
        # Shift entries/exits forward by 1 period to capture same-day returns
        adj_entries = entries.shift(-1).fillna(False)
        adj_exits = exits.shift(-1).fillna(False)
        
        pf10 = vbt.Portfolio.from_signals(
            close=weekly['cad_ig_er_index'],
            entries=adj_entries,
            exits=adj_exits,
            freq='W'
        )
        print(f"   Result: {pf10.total_return()*100:.2f}%")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 11: Use price data shifted backward
    print(f"\n11. Manual timing adjustment - shift price data backward:")
    try:
        # Use previous period's price for execution
        shifted_prices = weekly['cad_ig_er_index'].shift(1)
        
        pf11 = vbt.Portfolio.from_signals(
            close=shifted_prices,
            entries=entries,
            exits=exits,
            freq='W'
        )
        print(f"   Result: {pf11.total_return()*100:.2f}%")
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    investigate_vectorbt_parameters()
