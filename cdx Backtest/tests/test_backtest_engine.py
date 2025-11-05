"""
Quick test script for backtesting engine without ML training.
Tests the engine with simple mock signals for fast iteration.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from core import DataLoader, CSVDataProvider, BacktestEngine, MetricsCalculator


def load_sample_data():
    """Load a sample of data for testing."""
    print("Loading sample data...")
    data_loader = DataLoader(CSVDataProvider())
    
    # Try to load the actual data file
    data_file = script_dir.parent / "data_pipelines" / "data_processed" / "cdx_related.csv"
    
    if not data_file.exists():
        print(f"Data file not found at: {data_file}")
        print("Creating synthetic data for testing...")
        # Create synthetic data
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        data = pd.DataFrame({
            'us_ig_cdx_er_index': 100 + np.cumsum(np.random.randn(500) * 0.5),
        }, index=dates)
        return data
    else:
        # Load real data but use only last 500 rows for speed
        data = data_loader.load_and_prepare(str(data_file))
        return data.tail(500)


def create_mock_signals(data, price_column='us_ig_cdx_er_index'):
    """Create simple mock signals for testing."""
    print("Creating mock signals...")
    
    # Simple momentum-based signals
    price_series = data[price_column]
    returns = price_series.pct_change(20)  # 20-day momentum
    
    # Entry: positive momentum
    entry_signals = (returns > 0.001).astype(bool)
    
    # Exit: negative momentum (but with 7-day holding period, this will be overridden)
    exit_signals = (returns < -0.001).astype(bool)
    
    print(f"  Entry signals: {entry_signals.sum()}")
    print(f"  Exit signals: {exit_signals.sum()}")
    
    return entry_signals, exit_signals


def test_backtest_engine():
    """Test the backtesting engine."""
    print("="*80)
    print("TESTING BACKTEST ENGINE")
    print("="*80)
    
    # Load sample data
    data = load_sample_data()
    print(f"Data loaded: {len(data)} periods")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Create mock signals
    entry_signals, exit_signals = create_mock_signals(data)
    
    # Initialize backtest engine
    print("\nInitializing backtest engine...")
    backtest_engine = BacktestEngine(
        initial_capital=100000,
        fees=0.0,
        slippage=0.0,
        holding_period_days=7
    )
    
    # Run backtest
    print("\nRunning backtest...")
    price_series = data['us_ig_cdx_er_index']
    
    result = backtest_engine.run_backtest(
        price_series,
        entry_signals,
        exit_signals,
        apply_holding_period=True
    )
    
    # Display results
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    print(f"Total Return: {result.metrics['total_return']:.2%}")
    print(f"CAGR: {result.metrics['cagr']:.2%}")
    print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {result.metrics['sortino_ratio']:.2f}")
    print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
    print(f"Volatility: {result.metrics['volatility']:.2%}")
    print(f"Time in Market: {result.time_in_market:.2%}")
    print(f"Number of Trades: {result.trades_count}")
    print(f"Initial Value: ${result.metrics['initial_value']:,.2f}")
    print(f"Final Value: ${result.metrics['final_value']:,.2f}")
    
    # Test buy-and-hold for comparison
    print("\n" + "="*80)
    print("BUY-AND-HOLD COMPARISON")
    print("="*80)
    buy_hold_return = (price_series.iloc[-1] / price_series.iloc[0]) - 1
    years = len(data) / 252
    buy_hold_cagr = (1 + buy_hold_return) ** (1/years) - 1 if years > 0 else 0
    
    print(f"Buy-and-Hold Return: {buy_hold_return:.2%}")
    print(f"Buy-and-Hold CAGR: {buy_hold_cagr:.2%}")
    print(f"Strategy Outperformance: {result.metrics['cagr'] - buy_hold_cagr:.2%}")
    
    # Verify holding period was enforced
    print("\n" + "="*80)
    print("HOLDING PERIOD VERIFICATION")
    print("="*80)
    
    # Check positions
    positions = pd.Series(0, index=price_series.index)
    in_position = False
    days_held = 0
    
    for i in range(len(entry_signals)):
        if entry_signals.iloc[i] and not in_position:
            in_position = True
            days_held = 0
        
        if in_position:
            positions.iloc[i] = 1
            days_held += 1
            
            # Check if we can exit (holding period satisfied)
            if days_held >= 7 and exit_signals.iloc[i]:
                in_position = False
                days_held = 0
    
    # Count consecutive holding periods
    min_holding_periods = []
    in_pos = False
    hold_start = None
    
    for i in range(len(positions)):
        if positions.iloc[i] == 1 and not in_pos:
            in_pos = True
            hold_start = i
        elif positions.iloc[i] == 0 and in_pos:
            in_pos = False
            if hold_start is not None:
                min_holding_periods.append(i - hold_start)
    
    if min_holding_periods:
        min_holding = min(min_holding_periods)
        max_holding = max(min_holding_periods)
        avg_holding = np.mean(min_holding_periods)
        
        print(f"Minimum holding period observed: {min_holding} days")
        print(f"Maximum holding period observed: {max_holding} days")
        print(f"Average holding period: {avg_holding:.1f} days")
        
        if min_holding < 7:
            print(f"⚠ WARNING: Minimum holding period ({min_holding}) is less than 7 days!")
        else:
            print("✓ Holding period constraint verified (minimum >= 7 days)")
    else:
        print("No positions found to verify holding period")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    
    return result


if __name__ == '__main__':
    test_backtest_engine()

