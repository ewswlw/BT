# Iterative ML Strategy - 4% Rule CAD-IG-ER Index

## Overview

This iterative machine learning strategy is designed to beat the buy-and-hold strategy for the CAD-IG-ER index by at least **2.5% annualized return** (no leverage, long only, binary positioning, no transaction costs).

## Strategy Architecture

### Key Features

1. **Advanced Feature Engineering** (200+ features)
   - Multi-timeframe momentum indicators (18 windows: 1d to 252d)
   - Volatility regimes and measures (7 windows)
   - Cross-asset relationships (VIX, OAS spreads)
   - Statistical features (z-scores, percentiles, skew, kurtosis)
   - Regime detection
   - Interaction features
   - Lagged features
   - Macro economic indicators

2. **Ensemble Machine Learning Models**
   - Random Forest (RF)
   - LightGBM
   - XGBoost
   - CatBoost
   - Weighted ensemble based on walk-forward validation performance

3. **Statistical Validation**
   - Walk-forward validation to prevent overfitting
   - Bias checking (look-ahead bias, data leakage, survivorship bias)
   - Minimum Sharpe ratio requirement (0.5)
   - Maximum drawdown limit (-20%)
   - Minimum trades requirement (20)

4. **Iterative Improvement**
   - Learns from past iterations
   - Feature importance analysis
   - Threshold optimization
   - Performance tracking

## Files Created

### 1. Strategy Implementation
**File:** `cad_ig_er_index_backtesting/strategies/iterative_ml_strategy.py`
- Complete iterative ML strategy implementation
- Advanced feature engineering
- Walk-forward validation
- Bias checking
- Ensemble model training

### 2. Configuration File
**File:** `cad_ig_er_index_backtesting/configs/iterative_ml_config.yaml`
- Strategy configuration
- Feature engineering parameters
- Model selection options
- Validation thresholds

### 3. Optimization Script
**File:** `cad_ig_er_index_backtesting/run_iterative_optimization.py`
- Runs iterative optimization loop
- Tracks performance across iterations
- Stops when target outperformance is achieved
- Generates reports for each iteration

### 4. Strategy Factory Update
**File:** `cad_ig_er_index_backtesting/strategies/strategy_factory.py`
- Registered `iterative_ml_strategy` in the factory

## How to Run

### Prerequisites

Ensure you have all dependencies installed:
```bash
poetry install
# or
pip install pandas numpy scikit-learn lightgbm xgboost catboost scipy
```

### Run Iterative Optimization

```bash
cd cad_ig_er_index_backtesting
poetry run python run_iterative_optimization.py --config configs/iterative_ml_config.yaml
```

Or using Python directly:
```bash
cd cad_ig_er_index_backtesting
python run_iterative_optimization.py --config configs/iterative_ml_config.yaml
```

### Run Single Iteration via Main Script

```bash
poetry run python main.py --config configs/iterative_ml_config.yaml
```

## Strategy Parameters

### Feature Engineering
- **Momentum Windows:** [1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 40, 50, 60, 90, 120, 180, 252] days
- **Volatility Windows:** [5, 10, 20, 40, 60, 120, 252] days
- **Z-Score Windows:** [5, 10, 20, 40, 60, 120, 252] days
- **Correlation Windows:** [20, 60, 120, 252] days

### Model Configuration
- **Random Forest:** 500 trees, max_depth=15, min_samples_leaf=5
- **LightGBM:** 200 iterations, learning_rate=0.05, early stopping
- **XGBoost:** 200 estimators, max_depth=7, learning_rate=0.05
- **CatBoost:** 200 iterations, depth=7, learning_rate=0.05

### Walk-Forward Validation
- **Train Size:** 70% of data
- **Test Size:** 15% of data
- **Step Size:** 5% of data

### Risk Management
- **Prediction Horizon:** 7 days ahead
- **Minimum Holding Period:** 5 days
- **Maximum Holding Period:** 60 days
- **Probability Threshold:** Optimized per iteration

### Validation Thresholds
- **Minimum Outperformance:** 2.5% annualized
- **Minimum Sharpe Ratio:** 0.5
- **Maximum Drawdown:** -20%
- **Minimum Trades:** 20

## Expected Performance

The strategy iteratively improves until it achieves:
- **Target:** Beat buy-and-hold by at least 2.5% annualized return
- **Methodology:** Statistical validation at each iteration
- **Bias Prevention:** Walk-forward validation, bias checking
- **Adaptive:** Learns from past iterations

## Output Files

Results are saved to:
- **Reports:** `outputs/reports/`
- **Iteration Results:** `outputs/iterative_ml/iteration_N.json`
- **Artifacts:** `outputs/reports/artifacts/`

Each iteration saves:
- Model performance metrics
- Feature importance
- Ensemble weights
- Validation results
- Trading signals and returns

## Iteration Process

1. **Feature Engineering:** Create 200+ features from raw data
2. **Bias Checking:** Verify no look-ahead bias or data leakage
3. **Model Training:** Train ensemble with walk-forward validation
4. **Threshold Optimization:** Find optimal prediction threshold
5. **Signal Generation:** Generate entry/exit signals
6. **Performance Validation:** Check if target outperformance met
7. **Learning:** Analyze results to guide next iteration

## Statistical Validation

The strategy includes comprehensive statistical validation:

1. **Walk-Forward Analysis:** Prevents overfitting by using time-series cross-validation
2. **Bias Detection:** Checks for look-ahead bias, data leakage, survivorship bias
3. **Performance Metrics:** Tracks CAGR, Sharpe ratio, max drawdown, trade count
4. **Significance Testing:** Validates that outperformance is statistically meaningful

## Customization

You can customize the strategy by editing `configs/iterative_ml_config.yaml`:

- Adjust feature engineering windows
- Enable/disable specific models
- Change validation thresholds
- Modify risk management parameters
- Set maximum iterations

## Notes

- The strategy uses **daily data** (not resampled)
- **No transaction costs** or slippage assumed
- **Binary positioning:** 100% long or 100% cash
- **No leverage** used
- **Long only** strategy

## Troubleshooting

If you encounter issues:

1. **Missing Dependencies:** Install required packages (scikit-learn, lightgbm, xgboost, catboost)
2. **Data Path:** Ensure data file path in config is correct
3. **Memory Issues:** Reduce feature windows or number of models
4. **Model Availability:** Strategy will use available models (RF always available)

## Next Steps

1. Run the iterative optimization script
2. Review iteration results in `outputs/iterative_ml/`
3. Analyze feature importance to understand what drives performance
4. Adjust parameters based on insights
5. Continue iterations until target outperformance is achieved

## Performance Tracking

Each iteration tracks:
- Strategy CAGR vs Benchmark CAGR
- Outperformance (target: +2.5% annualized)
- Sharpe ratio
- Maximum drawdown
- Number of trades
- Time in market
- Feature importance
- Ensemble model weights

The strategy stops automatically when the target outperformance is achieved.
