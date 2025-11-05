# CDX Backtesting Framework

Comprehensive backtesting framework for `us_ig_cdx_er_index` trading strategies.

## Objective

Develop trading strategies that beat buy-and-hold by **2.5%+ annualized return** with:
- 7-day minimum holding period
- Binary positioning (100% long or 100% cash, no leverage)
- No transaction costs
- Walk-forward validation
- Statistical bias checking

## Features

- **150+ Engineered Features**: CDX spreads, yield curves, macro indicators, equity/volatility, commodities, interactions
- **Multiple Strategy Types**: ML ensemble, rule-based, hybrid approaches
- **Walk-Forward Validation**: Expanding window validation with 6 test periods
- **Statistical Validation**: Probabilistic Sharpe Ratio (PSR), Deflated Sharpe Ratio (DSR), bias checking
- **Comprehensive Metrics**: Performance metrics using quantstats

## Requirements

Uses existing poetry environment. All dependencies are in `pyproject.toml`:
- pandas, numpy, scipy
- scikit-learn, lightgbm, xgboost, catboost
- vectorbt, quantstats
- optuna, shap
- pyyaml, matplotlib, plotly

## Usage

### Quick Testing (Fast - No ML Training)

For quick testing of the backtesting engine without ML training:

```bash
# From project root
poetry run python "cdx Backtest/test_backtest_engine.py"
```

This script:
- Loads sample data (last 500 rows)
- Creates simple mock signals
- Tests backtest engine with 7-day holding period
- Verifies metrics calculation
- Runs in seconds (no ML training)

### Full Execution

```bash
# Run all strategies with walk-forward validation
poetry run python "cdx Backtest/main.py" --config config.yaml

# Run specific strategy
poetry run python "cdx Backtest/main.py" --config config.yaml --strategy ml
poetry run python "cdx Backtest/main.py" --config config.yaml --strategy rule
poetry run python "cdx Backtest/main.py" --config config.yaml --strategy hybrid

# Skip walk-forward validation (faster)
poetry run python "cdx Backtest/main.py" --config config.yaml --no-walk-forward
```

### Configuration

Edit `config.yaml` to customize:
- Data path and date ranges
- Feature engineering parameters
- Strategy configurations
- Walk-forward validation settings
- Target performance thresholds

## Data Source

Data is loaded from:
```
../../data_pipelines/data_processed/cdx_related.csv
```

The dataset includes:
- `us_ig_cdx_er_index`: Target trading asset
- CDX spreads (IG, HY)
- Treasury yields and yield curves
- Macro indicators (LEI, economic regime, surprises)
- Equity indices (SPX, NDX)
- Volatility (VIX, VVIX)
- Commodities (Gold, Dollar, WTI)
- Financial conditions (Bloomberg FCI)

## Strategy Types

### ML Strategy
- **Ensemble**: Random Forest + LightGBM + XGBoost
- **Features**: 150+ engineered features
- **Target**: Binary classification (positive forward return)
- **Probability Threshold**: 0.55 (configurable)

### Rule-Based Strategy
- **Momentum**: Multi-asset momentum with confirmation
- **Mean Reversion**: Z-score based mean reversion
- **Regime**: Volatility/credit regime detection
- **Combination**: Multiple filters combined

### Hybrid Strategy
- **ML for Entry**: ML predictions for entry signals
- **Rules for Filter**: Rule-based filters for confirmation
- **Weighted Ensemble**: Configurable ML/rule weights

## Output

Results are saved to `outputs/`:
- `outputs/results/summary.json`: Summary of all strategies
- `outputs/reports/`: HTML reports (if enabled)
- `outputs/models/`: Saved ML models (if enabled)
- `outputs/features/`: Feature importance plots (if enabled)

## Walk-Forward Validation

The framework uses expanding window walk-forward validation:
- Initial 70% of data for training
- Remaining 30% split into 6 sequential test periods
- Each test period uses all data up to that point for training
- Tracks performance across all periods

## Statistical Validation

- **Probabilistic Sharpe Ratio (PSR)**: Probability that true SR > benchmark SR
- **Deflated Sharpe Ratio (DSR)**: SR adjusted for multiple testing
- **Bias Checking**: Look-ahead bias, survivorship bias, selection bias
- **Monte Carlo Simulation**: Robustness testing

## Success Criteria

A strategy is considered successful if:
- Outperforms buy-and-hold by **2.5%+ annualized return**
- PSR > 0.95
- Walk-forward consistency > 50%
- No significant bias detected

## Example Output

```
================================================================================
CDX BACKTESTING FRAMEWORK
================================================================================

Buy-and-Hold Results:
  Total Return: 25.25%
  CAGR: 1.40%
  Sharpe Ratio: 0.25
  Max Drawdown: -15.30%

Target CAGR: 3.90% (beat buy-and-hold by 2.50%)

ML Strategy Results:
  CAGR: 4.20%
  Outperformance: 2.80%
  Sharpe Ratio: 0.85
  Max Drawdown: -12.50%
  Time in Market: 65.20%
  Trades: 45
  PSR: 0.97
  DSR: 0.78

Walk-Forward Summary:
  Avg CAGR: 3.95%
  Win Rate: 83.33%
  Consistency: 83.33%

BEST STRATEGY: ML
Outperformance: 2.80%
Meets 2.5% target: YES
```

## File Structure

```
cdx Backtest/
├── README.md                    # This file
├── config.yaml                  # Configuration
├── main.py                      # Main execution script
├── core/                        # Core modules
│   ├── data_loader.py          # Data loading
│   ├── feature_engineering.py  # Feature creation
│   ├── backtest_engine.py      # Backtesting
│   ├── validation.py           # Walk-forward & validation
│   └── metrics.py              # Performance metrics
├── strategies/                  # Trading strategies
│   ├── base_strategy.py        # Base class
│   ├── ml_strategy.py          # ML ensemble
│   ├── rule_based_strategy.py  # Rule-based
│   └── hybrid_strategy.py      # Hybrid approach
├── outputs/                     # Results
│   ├── results/                # CSV/JSON results
│   ├── reports/                # HTML reports
│   ├── models/                 # Saved models
│   └── features/               # Feature plots
└── tests/                       # Test suite
    └── test_strategies.py      # Tests
```

## Notes

- All code uses the existing poetry environment
- Data is forward-filled for monthly/weekly columns (LEI, economic regime, equity revisions)
- 7-day holding period is strictly enforced
- Binary positioning: 100% long or 100% cash (no partial positions)
- No transaction costs or leverage assumed

