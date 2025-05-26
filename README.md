# Modular Backtesting Framework

A comprehensive and modular backtesting framework for testing and comparing market-timing strategies on financial time series data.

## Overview

This backtesting framework provides a flexible and extensible system for developing, testing, and analyzing trading strategies. It is designed to work with the `vectorbt` library and includes pre-built implementations for three different types of strategies:

1. **Technical Indicator Strategy** - Based on moving averages and RSI
2. **Economic Regime Strategy** - Based on economic indicators and regime detection
3. **Statistical Strategy** - Based on z-scores, percentiles, and other statistical features

## Project Structure

```
BT/
├── backtest/                     # Main backtesting framework
│   ├── config/                   # Configuration classes
│   ├── strategies/               # Strategy implementations
│   ├── performance/              # Performance evaluation tools
│   ├── utils/                    # Utility functions
│   └── engine.py                 # Core backtesting engine
├── data_pipelines/               # Data handling and processing
│   └── data_processed/           # Processed data files
├── ai_instructions/              # Framework documentation
└── run_backtest.py               # Main script to run backtests
```

## Requirements

- Python 3.7+
- pandas
- numpy
- vectorbt
- matplotlib
- seaborn

## Setup

1. Clone the repository
2. Install required packages: `pip install pandas numpy vectorbt matplotlib seaborn`
3. Make sure you have the data file in the correct location (`data_pipelines/data_processed/with_er_daily.csv`)

## Usage

Run the main backtesting script:

```bash
python run_backtest.py
```

This will:
1. Load the data
2. Initialize the three strategies
3. Run backtests for all strategies
4. Calculate and display performance metrics
5. Generate visualization plots

## Strategies

### 1. Technical Indicator Strategy

A strategy based on technical analysis indicators:
- Moving average crossovers (short MA vs long MA)
- Relative Strength Index (RSI)

The strategy generates buy signals when the short-term moving average is above the long-term moving average and the RSI indicates oversold conditions.

### 2. Economic Regime Strategy

A strategy based on macroeconomic regimes and indicators:
- Economic regime detection
- Growth and inflation surprises
- Volatility measures (VIX)

The strategy identifies different economic regimes and takes positions accordingly:
- Expansion: Long
- Recovery: Long
- Contraction: No position (unless growth improving)
- Stagflation: No position

### 3. Statistical Strategy

A strategy based on statistical features and transformations:
- Z-scores of credit spreads and market indicators
- Percentile rankings
- Momentum metrics

The strategy uses a voting mechanism combining multiple signals from different statistical features to determine entry and exit points.

## Performance Evaluation

The framework includes a performance evaluation module with:
- Standard portfolio metrics (returns, volatility, drawdowns)
- Risk-adjusted metrics (Sharpe, Sortino, Calmar ratios)
- Comparative metrics against a buy-and-hold benchmark
- Visualization tools for equity curves, drawdowns, and return distributions

## Extension

You can create custom strategies by inheriting from the `Strategy` base class and implementing the `generate_signals` method. See the existing strategy implementations for examples.

## Tips for Optimal Use

1. **Data Preparation**: Ensure data is clean and properly formatted with a datetime index
2. **Parameter Tuning**: Experiment with different strategy parameters to improve performance
3. **Cross-Validation**: Test strategies on different time periods to check for robustness
4. **Combined Approaches**: Create new strategies that combine elements of the existing ones
5. **Performance Analysis**: Always analyze performance across multiple metrics, not just total return

## Important Notes

- The framework assumes no transaction costs, leverage, or short-selling
- Performance metrics should be considered as theoretical maximums
- Always check for data quality issues before interpreting results
