# Backtesting Framework - Complete Strategy Guide

## Table of Contents
1. [Overview](#overview)
2. [Running main.py From Different Locations](#running-mainpy-from-different-locations)
3. [Available Strategies](#available-strategies)
4. [Single Strategy Commands](#single-strategy-commands)
5. [Multiple Strategy Comparisons](#multiple-strategy-comparisons)
6. [Command Line Options](#command-line-options)
7. [Expected Outputs](#expected-outputs)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The backtesting framework supports 6 different trading strategies:
1. **RF Ensemble Strategy** (Random Forest ML) - 3.86% annualized return
2. **LightGBM Strategy** (Gradient Boosting ML) - 144%+ total return
3. **Cross Asset Momentum** - Multi-asset momentum strategy
4. **Multi Asset Momentum** - Enhanced momentum across assets
5. **Genetic Algorithm** - Evolutionary optimization strategy
6. **Vol Adaptive Momentum** - Volatility-adjusted momentum

---

## Running main.py From Different Locations

The `main.py` script **automatically detects its location** and adjusts paths accordingly. You can run it from anywhere.

### Method 1: From the `cad_ig_er_index_backtesting` Directory (Recommended)

```powershell
# Navigate to the directory
cd "C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Eddy\Python Projects\BT\cad_ig_er_index_backtesting"

# Run commands
poetry run python main.py --examples
poetry run python main.py --list-strategies
poetry run python main.py --config configs/rf_ensemble_config.yaml
```

**Advantages:**
- ✅ Shortest commands
- ✅ Easiest config path references
- ✅ Most intuitive

---

### Method 2: From the Parent `BT` Directory

```powershell
# Navigate to BT directory
cd "C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Eddy\Python Projects\BT"

# Run with relative path
poetry run python cad_ig_er_index_backtesting/main.py --examples
poetry run python cad_ig_er_index_backtesting/main.py --list-strategies
poetry run python cad_ig_er_index_backtesting/main.py --config cad_ig_er_index_backtesting/configs/rf_ensemble_config.yaml
```

**Advantages:**
- ✅ Can manage multiple sub-projects
- ✅ Works from parent directory

---

### Method 3: From ANY Directory (Using Full Path)

```powershell
# From anywhere on your system
poetry run python "C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Eddy\Python Projects\BT\cad_ig_er_index_backtesting\main.py" --examples

# With config
poetry run python "C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Eddy\Python Projects\BT\cad_ig_er_index_backtesting\main.py" --config configs/rf_ensemble_config.yaml
```

**Advantages:**
- ✅ Works from absolutely anywhere
- ✅ Can be scripted/automated
- ✅ No need to change directories

---

### Method 4: Using PowerShell Alias (Optional Setup)

You can create an alias in your PowerShell profile:

```powershell
# Add to your PowerShell profile ($PROFILE)
function Run-Backtest {
    poetry run python "C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Eddy\Python Projects\BT\cad_ig_er_index_backtesting\main.py" @args
}

Set-Alias bt Run-Backtest
```

Then use it:
```powershell
bt --examples
bt --list-strategies
bt --config configs/rf_ensemble_config.yaml
```

---

## Available Strategies

### 1. **RF Ensemble Strategy** (NEW - Recommended)
- **Type:** Machine Learning (Random Forest Ensemble)
- **Performance:** 3.86% annualized return
- **Outperformance:** 2.9x vs buy-and-hold (1.33%)
- **Features:** 96 engineered features
- **Models:** 4 Random Forest models with weighted ensemble
- **Time Invested:** ~80%
- **Trade Frequency:** ~6 trades/year
- **Holding Period:** Minimum 7 days
- **Best For:** Long-term systematic trading with ML predictions

### 2. **LightGBM Strategy**
- **Type:** Machine Learning (Gradient Boosting)
- **Performance:** 144.6%+ total return
- **Features:** 100+ comprehensive features
- **Models:** LightGBM with threshold optimization
- **Best For:** High-performance ML-driven trading

### 3. **Cross Asset Momentum**
- **Type:** Momentum-based
- **Logic:** Long when ≥3 of 4 indices show positive momentum
- **Rebalancing:** Weekly on Mondays
- **Best For:** Multi-asset trend following

### 4. **Multi Asset Momentum**
- **Type:** Enhanced momentum
- **Logic:** Multiple asset momentum with confirmations
- **Best For:** Diversified momentum strategies

### 5. **Genetic Algorithm**
- **Type:** Evolutionary optimization
- **Logic:** Genetically evolved trading rules
- **Best For:** Parameter optimization and rule discovery

### 6. **Vol Adaptive Momentum**
- **Type:** Volatility-adjusted momentum
- **Logic:** Momentum strategy with volatility scaling
- **Best For:** Risk-adjusted momentum trading

---

## Single Strategy Commands

All commands assume you're in the `cad_ig_er_index_backtesting` directory unless otherwise noted.

### Run RF Ensemble Strategy (Your New 3.86% Strategy)

```powershell
# Using dedicated config
poetry run python main.py --config configs/rf_ensemble_config.yaml

# Or specify by name
poetry run python main.py --config configs/config.yaml --strategies rf_ensemble_strategy
```

**Expected Runtime:** 5-10 minutes (4 RF models training)

---

### Run LightGBM Strategy

```powershell
poetry run python main.py --config configs/lightgbm_config.yaml

# Or by name
poetry run python main.py --config configs/config.yaml --strategies lightgbm_strategy
```

**Expected Runtime:** 10-15 minutes (LightGBM training)

---

### Run Cross Asset Momentum

```powershell
poetry run python main.py --config configs/config.yaml --strategies cross_asset_momentum
```

**Expected Runtime:** 1-2 minutes

---

### Run Multi Asset Momentum

```powershell
poetry run python main.py --config configs/config.yaml --strategies multi_asset_momentum
```

**Expected Runtime:** 1-2 minutes

---

### Run Genetic Algorithm

```powershell
poetry run python main.py --config configs/config.yaml --strategies genetic_algorithm
```

**Expected Runtime:** 5-10 minutes (genetic evolution)

---

### Run Vol Adaptive Momentum

```powershell
poetry run python main.py --config configs/config.yaml --strategies vol_adaptive_momentum
```

**Expected Runtime:** 1-2 minutes

---

## Multiple Strategy Comparisons

### Compare 2 Strategies: RF Ensemble vs LightGBM (ML Battle)

```powershell
poetry run python main.py --config configs/config.yaml --strategies rf_ensemble_strategy lightgbm_strategy
```

**Purpose:** Compare two ML approaches
**Runtime:** 15-20 minutes

---

### Compare 3 Strategies: All ML Models

```powershell
poetry run python main.py --config configs/config.yaml --strategies rf_ensemble_strategy lightgbm_strategy genetic_algorithm
```

**Purpose:** Comprehensive ML comparison
**Runtime:** 20-30 minutes

---

### Compare 2 Strategies: ML vs Traditional

```powershell
poetry run python main.py --config configs/config.yaml --strategies rf_ensemble_strategy cross_asset_momentum
```

**Purpose:** ML vs rule-based comparison
**Runtime:** 6-12 minutes

---

### Compare All Momentum Strategies

```powershell
poetry run python main.py --config configs/config.yaml --strategies cross_asset_momentum multi_asset_momentum vol_adaptive_momentum
```

**Purpose:** Compare momentum variations
**Runtime:** 3-6 minutes

---

### Compare 4 Strategies: Mixed Approaches

```powershell
poetry run python main.py --config configs/config.yaml --strategies rf_ensemble_strategy lightgbm_strategy cross_asset_momentum vol_adaptive_momentum
```

**Purpose:** Diverse strategy comparison
**Runtime:** 16-22 minutes

---

### Run ALL 6 Strategies (Full Benchmark)

```powershell
poetry run python main.py --config configs/config.yaml --strategies cross_asset_momentum multi_asset_momentum genetic_algorithm vol_adaptive_momentum lightgbm_strategy rf_ensemble_strategy
```

**Purpose:** Complete performance benchmark
**Runtime:** 25-40 minutes
**Output:** Comprehensive comparison report with all strategies

---

## Command Line Options

### `--config CONFIG_PATH`
Specify the configuration file to use.

```powershell
poetry run python main.py --config configs/rf_ensemble_config.yaml
```

**Available Configs:**
- `configs/rf_ensemble_config.yaml` - RF Ensemble strategy
- `configs/lightgbm_config.yaml` - LightGBM strategy
- `configs/config.yaml` - General config for all strategies
- `configs/base_config.yaml` - Base configuration template
- `configs/custom_config.yaml` - Custom user configurations

---

### `--strategies STRATEGY1 STRATEGY2 ...`
Override config and run specific strategies by name.

```powershell
# Single strategy
poetry run python main.py --config configs/config.yaml --strategies rf_ensemble_strategy

# Multiple strategies
poetry run python main.py --config configs/config.yaml --strategies rf_ensemble_strategy lightgbm_strategy
```

**Valid Strategy Names:**
- `rf_ensemble_strategy`
- `lightgbm_strategy`
- `cross_asset_momentum`
- `multi_asset_momentum`
- `genetic_algorithm`
- `vol_adaptive_momentum`

---

### `--list-strategies`
List all available strategies and exit.

```powershell
poetry run python main.py --list-strategies
```

**Output:**
```
Available strategies:
- cross_asset_momentum
- multi_asset_momentum
- genetic_algorithm
- vol_adaptive_momentum
- lightgbm_strategy
- rf_ensemble_strategy
```

---

### `--examples`
Show comprehensive usage examples (this document in terminal form).

```powershell
poetry run python main.py --examples
```

**Output:** Formatted guide with copy-paste ready commands

---

### `--help` or `-h`
Show argparse help message.

```powershell
poetry run python main.py --help
```

---

## Expected Outputs

### Output Directory Structure

After running strategies, outputs are saved to:

```
outputs/
├── reports/               # HTML and text reports
│   ├── strategy_name_report.html
│   └── strategy_name_report.txt
├── results/               # Detailed results and trade blotters
│   ├── trade_blotter.csv
│   └── strategy_results.json
└── plots/                 # Performance charts (if enabled)
    ├── equity_curve.png
    ├── drawdown.png
    └── returns_distribution.png
```

---

### Console Output Example

```
====================================================================================================
Running user-specified strategies: rf_ensemble_strategy
====================================================================================================

Loading data from: data_pipelines/data_processed/with_er_daily.csv
Data loaded: 5767 rows

=== Engineering 96 Features ===
  Creating momentum features...
  Creating z-score features...
  Creating macro features...
  ✓ Feature engineering complete: 96 features created

=== Training 4 Random Forest Models ===
  Training samples: 3856, Test samples: 1653

  Training Model 1/4...
    Test AUC: 0.6368
  Training Model 2/4...
    Test AUC: 0.6401
  [...]

=== Generating Ensemble Predictions ===
  Probability threshold: 0.55

✓ Signal Generation Complete:
  Total entry signals: 37
  Time in market: 80.0%

=== Performance Metrics ===
Strategy Return:      3.86% annualized
Buy & Hold Return:    1.33% annualized
Outperformance:       +2.53%
Sharpe Ratio:         0.85
Max Drawdown:         -8.2%

Reports saved to: outputs/reports/rf_ensemble_strategy_report.html
```

---

### Report Contents

Each strategy generates:

1. **HTML Report** (interactive)
   - Strategy overview
   - Performance metrics table
   - Equity curve chart
   - Drawdown analysis
   - Trade statistics
   - Benchmark comparison

2. **Text Report** (for terminals/logs)
   - Same metrics in text format
   - Suitable for automation/logging

3. **Trade Blotter CSV**
   - Entry dates and prices
   - Exit dates and prices
   - Position sizes
   - P&L per trade
   - Cumulative returns

4. **Strategy Results JSON**
   - Machine-readable results
   - All metrics and statistics
   - Trade details
   - Suitable for further analysis

---

## Troubleshooting

### Issue: "Can't open file main.py"

**Problem:** Running from wrong directory

**Solution:**
```powershell
# Check your current directory
pwd

# Navigate to correct location
cd "C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Eddy\Python Projects\BT\cad_ig_er_index_backtesting"

# Or use full path
poetry run python "C:\Users\Eddy\...\cad_ig_er_index_backtesting\main.py" --examples
```

---

### Issue: "Config file not found"

**Problem:** Incorrect config path

**Solution:**
```powershell
# From cad_ig_er_index_backtesting directory
poetry run python main.py --config configs/rf_ensemble_config.yaml

# From BT directory (parent)
poetry run python cad_ig_er_index_backtesting/main.py --config cad_ig_er_index_backtesting/configs/rf_ensemble_config.yaml

# Always use relative paths from cad_ig_er_index_backtesting
```

---

### Issue: "Unknown strategy type"

**Problem:** Strategy name typo or not registered

**Solution:**
```powershell
# Check available strategies
poetry run python main.py --list-strategies

# Use exact names (with underscores)
poetry run python main.py --config configs/config.yaml --strategies rf_ensemble_strategy
# NOT: rf-ensemble-strategy or RFEnsembleStrategy
```

---

### Issue: "Module not found" errors

**Problem:** Missing dependencies

**Solution:**
```powershell
# Reinstall dependencies
poetry install

# If scikit-learn missing (for RF Ensemble)
poetry add scikit-learn

# If lightgbm missing (for LightGBM strategy)
poetry add lightgbm
```

---

### Issue: Strategy takes too long

**Problem:** ML models training on large dataset

**Solutions:**
1. **Reduce train/test split:** Edit config to use less training data
2. **Reduce model complexity:** Lower `n_estimators` or `max_depth`
3. **Use fewer models:** Comment out models in config
4. **Run simpler strategies first:** Test with Cross Asset Momentum (fast)

**Example - Speed up RF Ensemble:**
Edit `configs/rf_ensemble_config.yaml`:
```yaml
rf_ensemble_strategy:
  train_test_split: 0.60  # Use 60% instead of 70% for training
  models:
    - n_estimators: 300   # Reduce from 600
      max_depth: 10       # Reduce from 15
```

---

### Issue: Out of memory errors

**Problem:** Too much data or too many features

**Solutions:**
1. Reduce date range in config
2. Downsample data (use weekly instead of daily)
3. Reduce number of features
4. Run one strategy at a time instead of comparisons

---

### Issue: Different results each run

**Problem:** Random seed not set or not consistent

**Solution:**
Check config has consistent `random_seed`:
```yaml
random_seed: 42  # Set at top level

rf_ensemble_strategy:
  models:
    - random_state: 42  # Same seed for all models
```

---

## Quick Reference Card

### Most Common Commands

```powershell
# 1. Navigate (do this first)
cd "C:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Eddy\Python Projects\BT\cad_ig_er_index_backtesting"

# 2. See all options
poetry run python main.py --examples

# 3. List strategies
poetry run python main.py --list-strategies

# 4. Run your RF Ensemble strategy
poetry run python main.py --config configs/rf_ensemble_config.yaml

# 5. Compare RF Ensemble vs LightGBM
poetry run python main.py --config configs/config.yaml --strategies rf_ensemble_strategy lightgbm_strategy

# 6. Run all strategies
poetry run python main.py --config configs/config.yaml --strategies cross_asset_momentum multi_asset_momentum genetic_algorithm vol_adaptive_momentum lightgbm_strategy rf_ensemble_strategy
```

---

## Performance Expectations

| Strategy | Annualized Return | Time Invested | Trades/Year | Runtime |
|----------|-------------------|---------------|-------------|---------|
| **RF Ensemble** | **3.86%** | ~80% | ~6 | 5-10 min |
| **LightGBM** | 144%+ total | High | Varies | 10-15 min |
| **Cross Asset** | Varies | ~50% | 12-24 | 1-2 min |
| **Multi Asset** | Varies | ~60% | 15-30 | 1-2 min |
| **Genetic Algorithm** | Varies | Variable | Variable | 5-10 min |
| **Vol Adaptive** | Varies | Variable | Variable | 1-2 min |

**Benchmark (Buy & Hold):** 1.33% annualized

---

## Additional Resources

- **Strategy Details:** See individual strategy `.py` files in `strategies/` folder
- **Configuration Guide:** See `configs/*.yaml` files for parameter explanations
- **ML Strategy Documentation:** See `ML_Strategy_Summary.md` in this folder
- **Framework Documentation:** See project README.md

---

*Last Updated: 2025-10-29*
*Framework Version: 2.0*
*Strategies Available: 6*
