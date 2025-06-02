#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Kept as quantstats might use it
import vectorbt as vbt
import quantstats as qs
import os
import warnings

warnings.filterwarnings('ignore')

# === CONFIGURATION ===
CONFIG = {
    "input_file_path": r"c:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Eddy\Python Projects\BT\data_pipelines\data_processed\with_er_daily.csv",
    "date_column": "Date",
    "resample_frequency": "W",
    "momentum_assets_map": { # Renamed for clarity
        "tsx": "tsx",
        "us_hy": "us_hy_er_index",
        "cad_ig": "cad_ig_er_index"
    },
    "momentum_lookback_periods": 4, # e.g., 4 weeks
    "signal_threshold": -0.005,
    "exit_signal_shift_periods": 1, # e.g., exit after 1 week
    "trading_asset_column": "cad_ig_er_index", # Asset to trade for the portfolio
    "portfolio_settings": {
        "freq": "W",
        "init_cash": 100,
        "fees": 0.0,
        "slippage": 0.0
    },
    "report_output_dir": "ai backtests/tearsheets",
    "report_filename_html": "Multi_asset_momentum_refactored.html",
    "report_title": "Multi-asset Momentum vs Buy and Hold (Refactored)"
}

# === CORE LOGIC FUNCTIONS ===

def load_and_prepare_data(file_path: str, date_col: str, resample_freq: str) -> pd.DataFrame:
    """
    Loads data from a CSV file, sets the DatetimeIndex, and resamples to the specified frequency.

    Args:
        file_path (str): Path to the input CSV file.
        date_col (str): Name of the column to be used as the date index.
        resample_freq (str): Resampling frequency (e.g., 'W' for weekly).

    Returns:
        pd.DataFrame: The prepared (resampled) DataFrame.
    """
    df = pd.read_csv(file_path, parse_dates=[date_col]).set_index(date_col)
    weekly_df = df.resample(resample_freq).last()
    print(f"Data loaded. Strategy Period: {weekly_df.index[0]} to {weekly_df.index[-1]}")
    print(f"Total periods ({resample_freq}): {len(weekly_df)}")
    return weekly_df

def calculate_multi_asset_momentum(
    weekly_df: pd.DataFrame, 
    asset_columns_map: dict, 
    lookback_periods: int
) -> pd.Series:
    """
    Calculates momentum for multiple assets and their combined average momentum.

    Args:
        weekly_df (pd.DataFrame): DataFrame with weekly asset prices.
        asset_columns_map (dict): Mapping of internal asset names to DataFrame column names.
        lookback_periods (int): Number of periods for momentum calculation.

    Returns:
        pd.Series: Combined momentum signal.
    """
    momentums = {}
    for asset_key, col_name in asset_columns_map.items():
        momentums[asset_key] = weekly_df[col_name] / weekly_df[col_name].shift(lookback_periods) - 1
    
    combined_momentum = sum(momentums.values()) / len(momentums)
    return combined_momentum

def generate_buy_sell_signals(
    combined_momentum_series: pd.Series, 
    threshold: float, 
    exit_shift_periods: int
) -> tuple[pd.Series, pd.Series]:
    """
    Generates entry and exit signals based on combined momentum and a threshold.

    Args:
        combined_momentum_series (pd.Series): The combined momentum signal.
        threshold (float): Threshold for entry signal.
        exit_shift_periods (int): Number of periods to hold before exiting (shifts entry for exit).

    Returns:
        tuple[pd.Series, pd.Series]: entry_signals, exit_signals
    """
    entry_signals = combined_momentum_series > threshold
    exit_signals = entry_signals.shift(exit_shift_periods).fillna(False)
    return entry_signals, exit_signals

def run_vectorbt_backtest(
    price_series: pd.Series, 
    entry_signals: pd.Series, 
    exit_signals: pd.Series, 
    portfolio_config: dict
) -> vbt.Portfolio:
    """
    Runs a backtest using vectorbt.

    Args:
        price_series (pd.Series): The price series of the asset to be traded.
        entry_signals (pd.Series): Boolean series indicating entry points.
        exit_signals (pd.Series): Boolean series indicating exit points.
        portfolio_config (dict): Configuration for the portfolio (init_cash, fees, slippage, freq).

    Returns:
        vbt.Portfolio: The backtested portfolio object.
    """
    portfolio = vbt.Portfolio.from_signals(
        price_series,
        entries=entry_signals,
        exits=exit_signals,
        freq=portfolio_config["freq"],
        init_cash=portfolio_config["init_cash"],
        fees=portfolio_config["fees"],
        slippage=portfolio_config["slippage"]
    )
    return portfolio

def generate_quantstats_report(
    portfolio: vbt.Portfolio,
    full_weekly_data: pd.DataFrame,
    benchmark_asset_column: str,
    report_title: str,
    output_directory: str,
    html_report_filename: str,
    report_frequency: str
) -> str:
    """
    Generates and saves a QuantStats HTML performance report.

    Args:
        portfolio (vbt.Portfolio): The backtested portfolio object.
        full_weekly_data (pd.DataFrame): Weekly data to derive benchmark returns.
        benchmark_asset_column (str): Column name for the benchmark asset in full_weekly_data.
        report_title (str): Title for the report.
        output_directory (str): Directory to save the report.
        html_report_filename (str): Filename for the HTML report.
        report_frequency (str): Frequency for QuantStats reporting (e.g., 'W').

    Returns:
        str: Path to the saved HTML report.
    """
    os.makedirs(output_directory, exist_ok=True)
    report_path = os.path.join(output_directory, html_report_filename)

    strategy_returns = portfolio.returns()
    
    # Benchmark returns: pct_change of the traded asset
    benchmark_returns = full_weekly_data[benchmark_asset_column].pct_change().reindex(strategy_returns.index).fillna(0)
    benchmark_returns = benchmark_returns.replace([np.inf, -np.inf], np.nan).dropna() # defensive cleaning

    print("\nGenerating QuantStats report...")
    qs.reports.full(
        strategy_returns,
        benchmark=benchmark_returns,
        title=report_title,
        freq=report_frequency
    )

    qs.reports.html(
        strategy_returns,
        benchmark=benchmark_returns,
        output=report_path,
        title=report_title,
        freq=report_frequency
    )
    print(f"Strategy vs Buy and Hold tearsheet saved to: {report_path}")
    return report_path

# === MAIN WORKFLOW ===

def main(config: dict):
    """
    Main function to run the multi-asset momentum strategy.
    """
    print("--- Running Multi-Asset Momentum Strategy (Refactored) ---")
    
    # 1. Load and Prepare Data
    weekly_df = load_and_prepare_data(
        config["input_file_path"],
        config["date_column"],
        config["resample_frequency"]
    )

    # 2. Calculate Momentum
    combined_momentum = calculate_multi_asset_momentum(
        weekly_df,
        config["momentum_assets_map"],
        config["momentum_lookback_periods"]
    )

    # 3. Generate Buy/Sell Signals
    entry_signals, exit_signals = generate_buy_sell_signals(
        combined_momentum,
        config["signal_threshold"],
        config["exit_signal_shift_periods"]
    )

    # 4. Define Price Series for Trading
    price_for_trading = weekly_df[config["trading_asset_column"]]

    # 5. Run Backtest
    portfolio = run_vectorbt_backtest(
        price_for_trading,
        entry_signals,
        exit_signals,
        config["portfolio_settings"]
    )
    
    # 6. Generate Performance Report
    generate_quantstats_report(
        portfolio,
        weekly_df, # Used for benchmark calculation
        config["trading_asset_column"], # Benchmark is the traded asset itself
        config["report_title"],
        config["report_output_dir"],
        config["report_filename_html"],
        config["portfolio_settings"]["freq"]
    )
    
    print("--- Strategy Execution Finished ---")

if __name__ == "__main__":
    main(CONFIG)

# === MULTI-ASSET MOMENTUM STRATEGY WITH VECTORBT ===
# (Original descriptive comment kept for context if needed)

# # Expert Analysis: Multi-Asset Momentum Strategy
# 
# ## Potential Statistical Integrity Concerns
# 
# ### 1. Look-Ahead Bias Risk
# - Your combined momentum signal uses the current period's data to make decisions for that same period
# - Ensure entry signals at time t only use data available at time t-1
# - Verify signal generation doesn't peek into future data
# 
# ### 2. Overfitting Risk
# - Threshold parameter (-0.005) appears hard-coded
# - Without proper walk-forward or cross-validation, this value may be optimized for the specific dataset
# - Could lead to poor out-of-sample performance
# 
# ### 3. Transaction Costs
# - Current implementation sets fees=0.0, which is unrealistic
# - Even low-cost implementations face some transaction costs
# - Weekly rebalancing would compound these costs
# 
# ### 4. Execution Slippage
# - slippage=0.0 assumes perfect execution at weekly close prices
# - Unrealistic in real trading scenarios
# - Should be modeled based on asset liquidity
# 
# ### 5. Weekly Rebalancing Frequency
# - Weekly position evaluation might lead to higher turnover
# - Consider if transaction costs justify this frequency
# - Test less frequent rebalancing for comparison
# 
# ### 6. Data Quality
# - Weekly resampled data might smooth over market volatility
# - Examine if critical information is lost in the resampling process
# - Consider sensitivity to resampling method (last vs. mean vs. median)
# 
# ### 7. Regime Dependency
# - Multi-asset momentum strategies perform differently across market regimes
# - No regime analysis in current implementation
# - Consider testing performance in different market conditions
# 
# ## Recommended Statistical Robustness Improvements
# 
# ### 1. Out-of-Sample Testing
# - Split dataset into training/testing periods
# - Validate performance out-of-sample
# - Consider time-series cross-validation approaches
# 
# ### 2. Parameter Sensitivity Analysis
# - Test sensitivity to small changes in threshold parameter (-0.005)
# - Create surface plots of performance metrics across parameter ranges
# - Identify regions of parameter stability
# 
# ### 3. Monte Carlo Simulation
# - Implement bootstrap resampling
# - Generate statistical confidence intervals for performance metrics
# - Assess probability distribution of returns and drawdowns
# 
# ### 4. Realistic Costs
# - Implement realistic transaction costs and slippage models
# - Include market impact for larger positions
# - Adjust historical performance estimates accordingly
# 
# ### 5. Drawdown Analysis
# - Perform detailed analysis of drawdown periods
# - Examine recovery times and drawdown clustering
# - Identify market conditions associated with poor performance
# 
# ### 6. Return Distribution Analysis
# - Examine skewness, kurtosis, and normality of returns
# - Test for autocorrelation in returns
# - Assess impact of outliers on performance metrics
# 
# ### 7. Trade Statistics
# - Add metrics like average win/loss, profit factor
# - Calculate maximum consecutive losses
# - Measure average holding period and turnover ratio

# 
