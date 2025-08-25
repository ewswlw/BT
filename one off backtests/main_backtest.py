"""
main_backtest.py

A self-contained, configurable backtesting script to test trading strategies,
compare them against a benchmark, and generate a comprehensive performance report.

Requirements:
- pandas
- numpy
- vectorbt
- quantstats

Install dependencies using pip:
pip install pandas numpy vectorbt quantstats
"""
import pandas as pd
import numpy as np
import vectorbt as vbt
import quantstats as qs
from datetime import datetime
import warnings

# Suppress vectorbt warnings about missing benchmark returns
warnings.filterwarnings('ignore', message='Metric.*requires benchmark_rets to be set')
warnings.filterwarnings('ignore', message='Metric.*raised an exception')

# Set vectorbt settings for annualization
vbt.settings.returns['year_freq'] = '252 days'

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
# Centralized configuration for the backtesting script.
# Modify this dictionary to change data sources, strategy parameters, and report settings.
# =============================================================================
CONFIG = {
    "data": {
        "file_path": "one off backtests/market_data.csv",
        "date_column": "Date",
        "price_column": "spx_close",
        "vix_column": "vix_close"
    },
    "portfolio": {
        "initial_capital": 100.0,
        "fees": 0.0,
        "slippage": 0.0,
        "freq": "D"  # 'D' for daily frequency
    },
    "benchmark": {
        "ticker": "spx_close",
        "name": "Buy & Hold"
    },
    "strategies": {
        "SMA_Crossover": {
            "enabled": True,
            "function": "sma_crossover_strategy",
            "params": {
                "fast_window": 50,
                "slow_window": 200,
            },
            "description": "A classic momentum strategy. Buys when a short-term moving average crosses above a long-term one."
        },
        "VIX_Volatility": {
            "enabled": True,
            "function": "vix_volatility_strategy",
            "params": {
                "vix_threshold": 20,
            },
            "description": "A mean-reversion strategy. Buys into market fear (high VIX) and sells into calm (low VIX)."
        }
    },
    "reporting": {
        "output_file": "one off backtests/backtest_results.txt",
        "title": "SIMPLE STRATEGIES VS. BUY & HOLD",
        "metrics_precision": 4
    }
}

# =============================================================================
# 2. DATA LOADING
# =============================================================================

def load_data(config: dict) -> pd.DataFrame:
    """
    Loads and prepares the market data using the DataPipeline class.
    
    Args:
        config (dict): The data configuration section.
        
    Returns:
        pd.DataFrame: A DataFrame with a DatetimeIndex and specified columns.
    """
    try:
        # Add project root to path to find data_pipelines module
        import os
        import sys
        # Assumes this script is in a subdirectory of the project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.append(project_root)
        
        from data_pipelines.fetch_data import DataPipeline
        
        # Initialize the pipeline and use it to load the dataset from CSV
        pipeline = DataPipeline()
        data = pipeline.load_dataset(config['file_path'])

        # Ensure the required columns exist
        required_cols = [config['price_column']]
        if config.get('vix_column'):
            required_cols.append(config['vix_column'])
        
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in the data file.")

        print(f"Data loaded successfully via DataPipeline. Date range: {data.index.min().date()} to {data.index.max().date()}")
        return data

    except ImportError:
        print("Error: Could not import DataPipeline. Make sure 'data_pipelines/fetch_data.py' exists.")
        return pd.DataFrame()
    except FileNotFoundError:
        print(f"Error: Data file not found at {config['file_path']}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return pd.DataFrame()

# =============================================================================
# 3. STRATEGY DEFINITIONS
# =============================================================================

def sma_crossover_strategy(data: pd.DataFrame, price_col: str, fast_window: int, slow_window: int, **kwargs) -> tuple:
    """
    Generates trading signals for a simple moving average crossover strategy.
    
    Args:
        data (pd.DataFrame): DataFrame containing the price data.
        price_col (str): The name of the price column to use.
        fast_window (int): The lookback period for the fast SMA.
        slow_window (int): The lookback period for the slow SMA.
        
    Returns:
        tuple: A tuple containing the entry and exit signal series (boolean).
    """
    fast_sma = vbt.MA.run(data[price_col], window=fast_window, short_name='fast_sma').ma
    slow_sma = vbt.MA.run(data[price_col], window=slow_window, short_name='slow_sma').ma
    
    entries = fast_sma.vbt.crossed_above(slow_sma)
    exits = fast_sma.vbt.crossed_below(slow_sma)
    
    return entries, exits

def vix_volatility_strategy(data: pd.DataFrame, vix_col: str, vix_threshold: int, **kwargs) -> tuple:
    """
    Generates trading signals based on a VIX threshold.
    
    Args:
        data (pd.DataFrame): DataFrame containing the VIX data.
        vix_col (str): The name of the VIX column.
        vix_threshold (int): The VIX level to trigger trades.
        
    Returns:
        tuple: A tuple containing the entry and exit signal series (boolean).
    """
    entries = data[vix_col] > vix_threshold
    exits = data[vix_col] < vix_threshold
    
    return entries, exits

# Mapping strategy names from config to their function objects
STRATEGY_MAPPING = {
    "sma_crossover_strategy": sma_crossover_strategy,
    "vix_volatility_strategy": vix_volatility_strategy,
}

# =============================================================================
# 4. BACKTESTING ENGINE
# =============================================================================

def run_backtest(price_data: pd.Series, entries: pd.Series, exits: pd.Series, config: dict) -> vbt.Portfolio:
    """
    Runs a backtest using vectorbt's Portfolio.from_signals.
    
    Args:
        price_data (pd.Series): The price series for the asset to be traded.
        entries (pd.Series): The entry signals.
        exits (pd.Series): The exit signals.
        config (dict): The portfolio configuration section.
        
    Returns:
        vbt.Portfolio: The resulting vectorbt portfolio object.
    """
    portfolio = vbt.Portfolio.from_signals(
        close=price_data,
        entries=entries,
        exits=exits,
        init_cash=config['initial_capital'],
        fees=config['fees'],
        slippage=config['slippage'],
        freq=config['freq']
    )
    return portfolio

def create_benchmark(price_data: pd.Series, config: dict) -> vbt.Portfolio:
    """Creates a buy-and-hold benchmark portfolio."""
    entries = pd.Series(False, index=price_data.index)
    entries.iloc[0] = True
    exits = pd.Series(False, index=price_data.index)
    
    benchmark_pf = vbt.Portfolio.from_signals(
        close=price_data,
        entries=entries,
        exits=exits,
        init_cash=config['portfolio']['initial_capital'],
        freq=config['portfolio']['freq']
    )
    return benchmark_pf

# =============================================================================
# 5. REPORTING ENGINE
# =============================================================================

def _section_break(char="=", length=80) -> str:
    return char * length + "\n"

def _format_vbt_stats(stats: pd.Series, precision: int) -> str:
    """Formats a vectorbt stats Series into a clean string."""
    report = ""
    for idx, val in stats.items():
        # Clean up the index name for better readability
        label = str(idx).replace('_', ' ').title().replace(' Pct', ' [%]')
        
        # Format the value based on its type
        if isinstance(val, (int, np.integer)):
            formatted_val = f"{val:d}"
        elif isinstance(val, (float, np.floating)):
            # Check if it's a percentage and format accordingly
            if 'Return' in label or 'Rate' in label or 'Drawdown' in label or 'Gross Exposure' in label or 'Trade' in label:
                 formatted_val = f"{val:>{15}.{precision}f}"
            else:
                 formatted_val = f"{val:>{15}.{precision}f}"
        else:
            formatted_val = str(val)
            
        report += f"{label:<30}: {formatted_val:>20}\n"
    return report

def generate_report_header(config: dict, results: dict) -> str:
    """Generates the main header for the report."""
    header = _section_break()
    header += f"{config['reporting']['title']}\n"
    header += _section_break()
    header += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    header += f"Total Strategies Analyzed: {len(results)}\n"
    header += f"Benchmark Asset: {config['benchmark']['name']} ({config['benchmark']['ticker']})\n"
    header += _section_break()
    return header

def generate_vbt_portfolio_stats_section(results: dict, config: dict) -> str:
    """Generates the vectorbt portfolio stats section of the report."""
    precision = config['reporting']['metrics_precision']
    report = "\n" + _section_break()
    report += "1. VECTORBT PORTFOLIO STATS (pf.stats())\n"
    report += _section_break()
    
    for name, pf in results.items():
        report += "-" * 50 + "\n"
        report += f"{name.upper()} - VectorBT Portfolio Stats\n"
        report += "-" * 50 + "\n"
        # The main portfolio stats are generated with no arguments
        report += _format_vbt_stats(pf.stats(), precision) + "\n"
        
    return report

def generate_vbt_returns_stats_section(results: dict, config: dict) -> str:
    """Generates the vectorbt returns stats section of the report."""
    precision = config['reporting']['metrics_precision']

    report = "\n" + _section_break()
    report += "2. VECTORBT RETURNS STATS (pf.returns().vbt.returns(freq='D').stats())\n"
    report += _section_break()

    for name, pf in results.items():
        report += "-" * 50 + "\n"
        report += f"{name.upper()} - VectorBT Returns Stats\n"
        report += "-" * 50 + "\n"
        
        # Correctly call stats by first specifying the frequency to the returns accessor.
        # This enables the calculation of annualized metrics.
        returns_stats = pf.returns().vbt.returns(freq='D').stats()
            
        report += _format_vbt_stats(returns_stats, precision) + "\n"
        
    return report

def generate_quantstats_section(results: dict, config: dict) -> str:
    """Generates QuantStats metrics comparison section."""
    benchmark_returns = results[config['benchmark']['name']].returns()
    benchmark_name = config['benchmark']['name']
    
    report = "\n" + _section_break()
    report += "3. QUANTSTATS STYLE COMPREHENSIVE COMPARISON\n"
    report += _section_break()
    
    for name, pf in results.items():
        if name == benchmark_name:
            continue
        
        strategy_returns = pf.returns()
        report += "-" * 80 + "\n"
        report += f"{name.upper()} vs BENCHMARK\n"
        report += "-" * 80 + "\n"
        
        # Temporarily capture print output from quantstats
        from io import StringIO
        import sys
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        qs.reports.metrics(strategy_returns, benchmark_returns, output=False, display=True)
        
        sys.stdout = old_stdout
        report += captured_output.getvalue() + "\n\n"
        
    return report

def generate_summary_table_section(results: dict, config: dict) -> str:
    """Generates a final summary table of key metrics."""
    benchmark_name = config['benchmark']['name']
    report = "\n" + _section_break()
    report += "4. STRATEGY SUMMARY COMPARISON\n"
    report += _section_break()
    
    header = (
        f"{'Strategy':<25} "
        f"{'Total Return':>15} "
        f"{'CAGR':>10} "
        f"{'Sharpe':>10} "
        f"{'Sortino':>10} "
        f"{'Max DD':>12} "
        f"{'Volatility':>12}"
    )
    report += header + "\n"
    report += "-" * len(header) + "\n"
    
    for name, pf in results.items():
        stats = pf.stats()
        returns_stats = pf.returns_stats()
        
        row = (
            f"{name:<25} "
            f"{stats['Total Return [%]']:>14.2f}% "
            f"{returns_stats['Annualized Return [%]']:>9.2f}% "
            f"{stats['Sharpe Ratio']:>10.2f} "
            f"{stats['Sortino Ratio']:>10.2f} "
            f"{stats['Max Drawdown [%]']:>11.2f}% "
            f"{returns_stats['Annualized Volatility [%]']:>11.2f}%"
        )
        report += row + "\n"
        
    return report
    
def generate_rules_section(config: dict) -> str:
    """Dynamically generates the trading rules section from the config."""
    report = "\n" + _section_break()
    report += "5. DYNAMIC TRADING RULES\n"
    report += _section_break()
    report += "Detailed trading rules and parameters for each strategy (extracted dynamically from config):\n\n"
    
    for name, strat_config in config['strategies'].items():
        if not strat_config['enabled']:
            continue
            
        report += "-" * 80 + "\n"
        report += f"{name.upper()} TRADING RULES\n"
        report += "-" * 80 + "\n"
        report += f"{strat_config['description']}\n\n"
        report += "TRADING RULES:\n"
        
        if name == 'SMA_Crossover':
            report += f"1. Asset: {config['data']['price_column']}\n"
            report += f"2. Entry: When the {strat_config['params']['fast_window']}-day SMA crosses above the {strat_config['params']['slow_window']}-day SMA.\n"
            report += f"3. Exit: When the {strat_config['params']['fast_window']}-day SMA crosses below the {strat_config['params']['slow_window']}-day SMA.\n\n"
        
        elif name == 'VIX_Volatility':
            report += f"1. Asset: {config['data']['price_column']}\n"
            report += f"2. Volatility Index: {config['data']['vix_column']}\n"
            report += f"3. Entry: When {config['data']['vix_column']} rises above {strat_config['params']['vix_threshold']}.\n"
            report += f"4. Exit: When {config['data']['vix_column']} falls below {strat_config['params']['vix_threshold']}.\n\n"
            
        report += "PARAMETERS:\n"
        for param, value in strat_config['params'].items():
            report += f"- {param}: {value}\n"
        report += "\n"
        
    return report

def generate_report_footer() -> str:
    """Generates the footer notes for the report."""
    report = "\n" + _section_break()
    report += "IMPORTANT NOTES\n"
    report += _section_break()
    report += "* All calculations are based on daily data.\n"
    report += f"* Fees and slippage are assumed to be {CONFIG['portfolio']['fees']} and {CONFIG['portfolio']['slippage']} respectively.\n"
    report += "* Results represent theoretical performance and may not be achievable in practice.\n"
    report += "* Past performance does not guarantee future results.\n"
    report += _section_break()
    report += "END OF REPORT\n"
    report += _section_break()
    return report

def generate_full_report(results: dict, config: dict) -> str:
    """Assembles all sections into a single report string."""
    full_report = ""
    full_report += generate_report_header(config, results)
    full_report += generate_vbt_portfolio_stats_section(results, config)
    full_report += generate_vbt_returns_stats_section(results, config)
    full_report += generate_quantstats_section(results, config)
    full_report += generate_summary_table_section(results, config)
    full_report += generate_rules_section(config)
    full_report += generate_report_footer()
    return full_report

# =============================================================================
# 6. MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """
    Main function to run the backtesting process.
    """
    print(_section_break())
    print("--- Starting Backtest Script ---")
    
    # Load data
    data = load_data(CONFIG['data'])
    if data.empty:
        print("--- Halting execution due to data loading error. ---")
        return

    price_series = data[CONFIG['data']['price_column']]
    results = {}

    # Run strategies
    print("\n--- Running Strategies ---")
    for name, strat_config in CONFIG['strategies'].items():
        if strat_config['enabled']:
            print(f"Processing strategy: {name}...")
            
            # Get the strategy function from the mapping
            strategy_func = STRATEGY_MAPPING.get(strat_config['function'])
            if not strategy_func:
                print(f"  Warning: Strategy function '{strat_config['function']}' not found. Skipping.")
                continue

            # Prepare arguments for the strategy function
            params = {
                "data": data,
                "price_col": CONFIG['data']['price_column'],
                "vix_col": CONFIG['data'].get('vix_column'),
                **strat_config['params']
            }
            
            # Generate signals
            entries, exits = strategy_func(**params)
            
            # Run backtest and store results
            portfolio = run_backtest(price_series, entries, exits, CONFIG['portfolio'])
            results[name] = portfolio
            print(f"  - {name} backtest complete. Total Return: {portfolio.total_return() * 100:.2f}%")

    # Run benchmark
    print("\n--- Running Benchmark ---")
    benchmark_name = CONFIG['benchmark']['name']
    benchmark_pf = create_benchmark(price_series, CONFIG)
    results[benchmark_name] = benchmark_pf
    print(f"  - {benchmark_name} backtest complete. Total Return: {benchmark_pf.total_return() * 100:.2f}%")



    # Generate and save report
    print("\n--- Generating Comprehensive Report ---")
    report_content = generate_full_report(results, CONFIG)
    output_path = CONFIG['reporting']['output_file']
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"Report saved successfully to: {output_path}")
    except Exception as e:
        print(f"Error: Could not save report file. {e}")

    print("\n--- Backtest Script Finished Successfully ---")
    print(_section_break())

if __name__ == "__main__":
    main() 