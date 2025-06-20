# Modular Financial Backtesting Framework

A flexible and powerful Python-based framework for developing, testing, and comparing quantitative trading strategies.

## Overview

This framework provides a structured environment to backtest complex, multi-asset trading strategies using daily time-series data. It is built to be extensible, allowing for the rapid implementation of new ideas while providing detailed performance analytics and reporting. The system can run multiple strategies simultaneously and generate a comprehensive comparison report, giving you a clear view of relative performance.

## Key Features

- **Strategy-Agnostic Engine**: A core backtesting engine that can run any strategy conforming to the base class.
- **Multiple Pre-Built Strategies**: Comes with four distinct, ready-to-run strategies.
- **Daily Data with Weekly Rebalancing**: All strategies are designed to operate on daily data but make trading decisions on a weekly (Monday) basis.
- **Flexible Configuration**: A central `config.yaml` file to control all parameters for data, features, portfolio construction, and strategies.
- **Command-Line Control**: A powerful CLI allows you to list available strategies and run specific subsets without touching the configuration file.
- **Comprehensive Reporting**: Automatically generates detailed reports including portfolio statistics, risk metrics, and human-readable trading rules for each strategy.

## Project Structure

```
backtesting_framework/
├── configs/
│   ├── config.yaml             # Primary configuration for the entire framework
│   └── custom_config.yaml      # An example template for alternate scenarios
├── core/                       # Core components of the backtesting engine
│   ├── data_loader.py
│   ├── portfolio.py
│   └── ...
├── data/                       # Placeholder for data files
├── notebooks/                  # Jupyter notebooks for analysis and experimentation
├── outputs/                    # Default directory for reports and results
│   └── results/
├── strategies/                 # Strategy implementations
│   ├── base_strategy.py        # Abstract base class for all strategies
│   ├── vol_adaptive_momentum.py
│   └── ...
├── tests/                      # Unit and integration tests
├── .gitignore
├── main.py                     # Main execution script
├── poetry.lock
├── pyproject.toml
└── README.md                   # This file
```

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1.  **Install Poetry**: Follow the instructions on the official Poetry website.
2.  **Install Dependencies**: From the project root directory, run:
    ```bash
    poetry install
    ```
3.  **Verify Data Path**: Ensure the `file_path` in `configs/config.yaml` points to your data source.

## Usage

All backtests are run from the command line using `main.py`.

#### 1. List Available Strategies

To see a list of all strategies you can run:
```bash
poetry run python backtesting_framework/main.py --list-strategies
```

#### 2. Run a Specific Subset of Strategies

To run one or more specific strategies, use the `--strategies` flag followed by the names of the strategies. This will override the `enabled` list in the config file.

*Example: Run only `vol_adaptive_momentum` and `genetic_algorithm`*
```bash
poetry run python backtesting_framework/main.py --strategies vol_adaptive_momentum genetic_algorithm
```

#### 3. Run Strategies from the Config File

To run all strategies listed in the `strategies.enabled` section of your `config.yaml`, simply run the script without the `--strategies` flag.

```bash
poetry run python backtesting_framework/main.py
```

#### 4. Use a Custom Configuration File

To run a backtest with a different configuration (e.g., `custom_config.yaml`), use the `--config` flag.

```bash
poetry run python backtesting_framework/main.py --config backtesting_framework/configs/custom_config.yaml
```

## Available Strategies

The framework comes with four pre-built strategies, all of which operate on daily data and rebalance weekly (on Mondays).

1.  **`cross_asset_momentum`**: A market-breadth strategy that enters a position based on the number of assets in a universe that are showing positive momentum. It's a confirmation-based model.
2.  **`multi_asset_momentum`**: This strategy calculates the momentum for several assets and averages them into a single "combined" momentum signal. It enters a position if this combined signal is strong enough.
3.  **`vol_adaptive_momentum`**: A more complex strategy that adjusts its behavior based on the market's volatility regime (measured by the VIX). It uses a stricter entry threshold in high-volatility environments and adapts its holding period based on momentum strength.
4.  **`genetic_algorithm`**: This strategy doesn't have fixed rules. Instead, it uses an evolutionary algorithm to "discover" the best trading rule from a large pool of technical indicators. It's designed to find novel patterns in the data.

## Configuration

The main `configs/config.yaml` file is the central hub for controlling the framework. Here you can adjust:
- Data sources and date ranges.
- Portfolio settings like initial capital, fees, and slippage.
- Parameters for the feature engineering engine.
- Specific parameters for each of the trading strategies.

## Extending the Framework

To create your own strategy:
1.  Create a new Python file in the `strategies/` directory.
2.  Define a new class that inherits from `BaseStrategy` (found in `strategies/base_strategy.py`).
3.  Implement the required methods: `__init__`, `get_required_features`, and `generate_signals`.
4.  Register your new strategy in `strategies/strategy_factory.py`.
5.  Add its configuration to `config.yaml`.
Your new strategy will then be available to run from the command line.
