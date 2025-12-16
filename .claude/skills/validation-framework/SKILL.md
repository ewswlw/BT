---
name: validation-framework
description: López de Prado validation methodologies including purged K-fold, CPCV, walk-forward analysis, deflated Sharpe ratio, probabilistic Sharpe ratio, and backtest overfitting detection
---

# Validation Framework Guide

## Overview
This skill provides guidance on using the comprehensive validation framework based on López de Prado's methodologies from "Advances in Financial Machine Learning". Essential for preventing overfitting and ensuring strategy robustness.

## When to Use This Skill
Claude should use this when:
- Validating trading strategy performance
- Implementing cross-validation for ML strategies
- Calculating deflated or probabilistic Sharpe ratios
- Detecting backtest overfitting
- Setting up walk-forward analysis
- Working with the validation module

## Validation Framework Architecture

### Directory Structure
```
cad_ig_er_index_backtesting/core/validation/
├── __init__.py
├── cross_validation.py          # Purged K-fold, CPCV
├── walk_forward.py               # Walk-forward analysis
├── sample_weights.py             # Uniqueness, time decay, sequential bootstrap
├── performance_metrics.py        # Deflated SR, Probabilistic SR
├── backtest_overfitting.py       # Probability of Backtest Overfitting
├── min_backtest_length.py        # Minimum backtest length calculation
├── validation_runner.py          # Orchestrates all validations
└── utils.py                      # Helper functions
```

## Validation Configuration

### Configuration Template
```yaml
validation:
  enabled: true

  # Cross-validation method
  cv_method: 'walk_forward'  # Options: walk_forward, purged_kfold, cpcv

  # Walk-forward parameters
  walk_forward:
    train_period: 252  # Trading days (~1 year)
    test_period: 63    # Trading days (~3 months)
    step_size: 21      # Trading days (~1 month)

  # Purged K-fold parameters
  purged_kfold:
    n_splits: 5
    embargo_pct: 0.01  # 1% embargo period

  # CPCV parameters
  cpcv:
    n_splits: 5
    n_test_splits: 2
    embargo_pct: 0.01

  # Sample weights
  sample_weights:
    method: 'uniqueness'  # Options: uniqueness, time_decay, sequential_bootstrap
    decay_factor: 0.95    # For time_decay

  # Performance metrics
  deflated_sharpe:
    n_trials: 100         # Number of trials for deflation
    skewness: 0.0         # Expected skewness
    kurtosis: 3.0         # Expected kurtosis

  probabilistic_sharpe:
    benchmark_sr: 0.0     # Benchmark Sharpe ratio
    n_observations: 252   # Minimum observations

  # Backtest overfitting
  pbo:
    n_splits: 16          # CPCV splits for PBO
    n_test_splits: 8      # Test splits

  # Minimum backtest length
  min_backtest_length:
    target_sharpe: 1.0
    confidence_level: 0.95
```

## Validation Methods

### 1. Purged K-Fold Cross-Validation

**Purpose**: Prevents leakage in time series by purging training samples that overlap with test period.

```python
from validation.cross_validation import PurgedKFold

def run_purged_kfold(
    data: pd.DataFrame,
    strategy: BaseStrategy,
    n_splits: int = 5,
    embargo_pct: float = 0.01
) -> List[float]:
    """
    Run purged K-fold cross-validation.

    Args:
        data: Time series data
        strategy: Trading strategy instance
        n_splits: Number of folds
        embargo_pct: Embargo period as % of total samples

    Returns:
        List of Sharpe ratios for each fold
    """
    pkf = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)

    sharpe_ratios = []
    for train_idx, test_idx in pkf.split(data):
        # Train on train_idx, test on test_idx
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        # Generate signals
        entry, exit = strategy.generate_signals(train_data)

        # Backtest on test data
        portfolio = run_backtest(test_data, entry[test_idx], exit[test_idx])
        sharpe_ratios.append(portfolio.sharpe_ratio())

    return sharpe_ratios
```

**Key Concepts:**
- **Purging**: Remove training samples that are "informationally close" to test samples
- **Embargo**: Add buffer period after purging to prevent leakage
- **Time series respect**: Maintain temporal order

**Best Practices:**
- Use embargo_pct = 0.01 (1%) as starting point
- Increase embargo for strategies with longer signal horizons
- Always purge when using ML models
- Document purging logic in code

### 2. Combinatorial Purged Cross-Validation (CPCV)

**Purpose**: Generate multiple combinations of test paths to assess strategy robustness.

```python
from validation.cross_validation import CombinatorialPurgedCV

def run_cpcv(
    data: pd.DataFrame,
    strategy: BaseStrategy,
    n_splits: int = 5,
    n_test_splits: int = 2,
    embargo_pct: float = 0.01
) -> Dict[str, Any]:
    """
    Run Combinatorial Purged Cross-Validation.

    Args:
        data: Time series data
        strategy: Trading strategy
        n_splits: Total number of splits
        n_test_splits: Number of splits in each test combination
        embargo_pct: Embargo period

    Returns:
        Dictionary with CPCV results and statistics
    """
    cpcv = CombinatorialPurgedCV(
        n_splits=n_splits,
        n_test_splits=n_test_splits,
        embargo_pct=embargo_pct
    )

    results = []
    for train_idx, test_idx in cpcv.split(data):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        entry, exit = strategy.generate_signals(train_data)
        portfolio = run_backtest(test_data, entry[test_idx], exit[test_idx])

        results.append({
            'sharpe_ratio': portfolio.sharpe_ratio(),
            'total_return': portfolio.total_return(),
            'max_drawdown': portfolio.max_drawdown()
        })

    return {
        'mean_sharpe': np.mean([r['sharpe_ratio'] for r in results]),
        'std_sharpe': np.std([r['sharpe_ratio'] for r in results]),
        'all_results': results
    }
```

**Key Advantages:**
- Tests all possible combinations of test paths
- Provides distribution of performance metrics
- More robust than single train/test split
- Used in Probability of Backtest Overfitting (PBO)

**Best Practices:**
- Use n_splits = 5-10 for reasonable computation time
- Use n_test_splits = 2-3 to balance coverage and sample size
- Analyze distribution of results, not just mean
- Look for consistency across all combinations

### 3. Walk-Forward Analysis

**Purpose**: Simulate real-world trading by repeatedly training on past data and testing on future data.

```python
from validation.walk_forward import WalkForwardAnalysis

def run_walk_forward(
    data: pd.DataFrame,
    strategy: BaseStrategy,
    train_period: int = 252,
    test_period: int = 63,
    step_size: int = 21
) -> pd.DataFrame:
    """
    Run walk-forward analysis.

    Args:
        data: Time series data
        strategy: Trading strategy
        train_period: Training window size (trading days)
        test_period: Test window size (trading days)
        step_size: Step size for rolling window

    Returns:
        DataFrame with results for each walk-forward window
    """
    wfa = WalkForwardAnalysis(
        train_period=train_period,
        test_period=test_period,
        step_size=step_size
    )

    results = []
    for train_start, train_end, test_start, test_end in wfa.split(data):
        train_data = data.iloc[train_start:train_end]
        test_data = data.iloc[test_start:test_end]

        # Retrain strategy on each window
        if hasattr(strategy, 'train_model'):
            strategy.train_model(train_data)

        entry, exit = strategy.generate_signals(test_data)
        portfolio = run_backtest(test_data, entry, exit)

        results.append({
            'test_start': test_data.index[0],
            'test_end': test_data.index[-1],
            'sharpe_ratio': portfolio.sharpe_ratio(),
            'total_return': portfolio.total_return(),
            'max_drawdown': portfolio.max_drawdown()
        })

    return pd.DataFrame(results)
```

**Typical Parameters:**
- **train_period**: 252 days (1 year)
- **test_period**: 63 days (3 months)
- **step_size**: 21 days (1 month)

**Best Practices:**
- Use rolling window for ML strategies (retrain each period)
- Use expanding window for stable strategies
- Monitor performance degradation over time
- Check for regime changes in test periods

### 4. Sample Weights

**Purpose**: Weight observations by uniqueness, time relevance, or bootstrap sampling.

```python
from validation.sample_weights import (
    calculate_uniqueness_weights,
    calculate_time_decay_weights,
    sequential_bootstrap
)

# Uniqueness weights (based on label overlap)
def get_uniqueness_weights(events: pd.DataFrame) -> pd.Series:
    """
    Calculate weights based on label uniqueness.

    Args:
        events: DataFrame with 't1' (end time) and 'label' columns

    Returns:
        Series of weights for each sample
    """
    return calculate_uniqueness_weights(events)

# Time decay weights
def get_time_decay_weights(
    timestamps: pd.DatetimeIndex,
    decay_factor: float = 0.95
) -> pd.Series:
    """
    Calculate exponentially decaying weights.

    Args:
        timestamps: Time index
        decay_factor: Decay rate (0-1)

    Returns:
        Series of weights
    """
    return calculate_time_decay_weights(timestamps, decay_factor)

# Sequential bootstrap
def get_bootstrap_weights(
    data: pd.DataFrame,
    n_samples: int = None
) -> pd.Series:
    """
    Generate bootstrap sample indices respecting sequential structure.

    Args:
        data: Time series data
        n_samples: Number of bootstrap samples

    Returns:
        Bootstrap sample weights
    """
    return sequential_bootstrap(data, n_samples)
```

**When to Use:**
- **Uniqueness weights**: When labels have varying overlap
- **Time decay weights**: To emphasize recent data
- **Sequential bootstrap**: To preserve time series structure

### 5. Deflated Sharpe Ratio

**Purpose**: Adjust Sharpe ratio for multiple testing and non-normality.

```python
from validation.performance_metrics import DeflatedSharpeRatio

def calculate_deflated_sharpe(
    returns: pd.Series,
    n_trials: int = 100,
    skewness: float = 0.0,
    kurtosis: float = 3.0
) -> float:
    """
    Calculate deflated Sharpe ratio.

    Args:
        returns: Strategy returns
        n_trials: Number of trials tested
        skewness: Expected skewness
        kurtosis: Expected excess kurtosis

    Returns:
        Deflated Sharpe ratio
    """
    dsr = DeflatedSharpeRatio(
        n_trials=n_trials,
        skewness=skewness,
        kurtosis=kurtosis
    )
    return dsr.calculate(returns)
```

**Interpretation:**
- **DSR > 0.95**: Very likely strategy is genuinely profitable
- **DSR > 0.50**: More likely than not to be profitable
- **DSR < 0.50**: Likely due to luck/overfitting

**Best Practices:**
- Always report DSR alongside regular Sharpe ratio
- Use conservative n_trials estimate (how many strategies tested)
- Account for non-normality via skewness/kurtosis
- DSR is probability that Sharpe ratio is statistically significant

### 6. Probabilistic Sharpe Ratio

**Purpose**: Calculate probability that strategy's Sharpe ratio exceeds a benchmark.

```python
from validation.performance_metrics import ProbabilisticSharpeRatio

def calculate_probabilistic_sharpe(
    returns: pd.Series,
    benchmark_sr: float = 0.0,
    n_observations: int = None
) -> float:
    """
    Calculate probabilistic Sharpe ratio.

    Args:
        returns: Strategy returns
        benchmark_sr: Benchmark Sharpe ratio to beat
        n_observations: Number of observations (defaults to len(returns))

    Returns:
        Probability that true SR > benchmark SR
    """
    psr = ProbabilisticSharpeRatio(
        benchmark_sr=benchmark_sr,
        n_observations=n_observations or len(returns)
    )
    return psr.calculate(returns)
```

**Interpretation:**
- **PSR > 0.95**: 95% confidence strategy beats benchmark
- **PSR > 0.50**: More likely than not to beat benchmark
- **PSR < 0.50**: Unlikely to beat benchmark

### 7. Probability of Backtest Overfitting (PBO)

**Purpose**: Detect overfitting by comparing in-sample vs out-of-sample performance.

```python
from validation.backtest_overfitting import calculate_pbo

def detect_overfitting(
    data: pd.DataFrame,
    strategy_configs: List[dict],
    n_splits: int = 16,
    n_test_splits: int = 8
) -> Dict[str, float]:
    """
    Calculate Probability of Backtest Overfitting.

    Args:
        data: Time series data
        strategy_configs: List of strategy configurations to test
        n_splits: CPCV splits
        n_test_splits: Test splits

    Returns:
        Dictionary with PBO and related metrics
    """
    pbo_result = calculate_pbo(
        data=data,
        strategy_configs=strategy_configs,
        n_splits=n_splits,
        n_test_splits=n_test_splits
    )

    return {
        'pbo': pbo_result.pbo,
        'median_logit': pbo_result.median_logit,
        'stdev_logit': pbo_result.stdev_logit
    }
```

**Interpretation:**
- **PBO < 0.50**: Low overfitting risk
- **PBO > 0.70**: High overfitting risk, strategy unlikely to generalize
- **PBO ≈ 1.0**: Severe overfitting

**Best Practices:**
- Test multiple strategy configurations
- Use CPCV with n_splits = 16, n_test_splits = 8
- Report PBO for all optimized strategies
- If PBO > 0.70, redesign strategy or reduce complexity

### 8. Minimum Backtest Length

**Purpose**: Calculate minimum backtest duration for statistical significance.

```python
from validation.min_backtest_length import calculate_min_backtest_length

def get_min_backtest_length(
    target_sharpe: float = 1.0,
    confidence_level: float = 0.95
) -> int:
    """
    Calculate minimum backtest length in observations.

    Args:
        target_sharpe: Expected Sharpe ratio
        confidence_level: Statistical confidence (0-1)

    Returns:
        Minimum number of observations required
    """
    return calculate_min_backtest_length(
        target_sharpe=target_sharpe,
        confidence_level=confidence_level
    )
```

**Example Results:**
- Sharpe = 1.0, 95% confidence: ~270 observations (13 months daily)
- Sharpe = 0.5, 95% confidence: ~1080 observations (4+ years daily)
- Sharpe = 2.0, 95% confidence: ~68 observations (3 months daily)

## Running Validation

### CLI Usage
```bash
# Run validation for specific strategy
python main.py --strategies lightgbm_strategy --run-validation

# Run validation for all strategies
python main.py --run-validation
```

### Validation Runner
```python
from validation.validation_runner import ValidationRunner

def run_full_validation(
    data: pd.DataFrame,
    strategy: BaseStrategy,
    config: dict
) -> Dict[str, Any]:
    """
    Run complete validation suite.

    Returns comprehensive validation report.
    """
    runner = ValidationRunner(config)
    return runner.run_validation(data, strategy)
```

### Validation Report Structure
```python
{
    'walk_forward': {
        'mean_sharpe': 1.2,
        'std_sharpe': 0.3,
        'all_periods': [...]
    },
    'purged_kfold': {
        'mean_sharpe': 1.1,
        'fold_sharpes': [...]
    },
    'cpcv': {
        'mean_sharpe': 1.15,
        'std_sharpe': 0.25,
        'all_combinations': [...]
    },
    'deflated_sharpe': 0.92,
    'probabilistic_sharpe': 0.88,
    'pbo': 0.35,
    'min_backtest_length': 270
}
```

## Validation Checklist

Before deploying any strategy:

- [ ] Run walk-forward analysis (252/63/21 split)
- [ ] Calculate purged K-fold Sharpe (n_splits=5)
- [ ] Run CPCV for robustness (n_splits=5, n_test_splits=2)
- [ ] Calculate Deflated Sharpe Ratio (DSR > 0.50)
- [ ] Calculate Probabilistic Sharpe Ratio (PSR > 0.50)
- [ ] Check PBO (PBO < 0.50)
- [ ] Verify minimum backtest length met
- [ ] Review sample weights appropriateness
- [ ] Document all validation results
- [ ] Compare in-sample vs out-of-sample performance

## Common Pitfalls

1. **Not Using Purged K-Fold**: Standard K-fold leaks information in time series
2. **Ignoring Embargo Period**: Leads to inflated performance estimates
3. **Cherry-Picking Results**: Report all validation metrics, not just good ones
4. **Insufficient Data**: Ensure minimum backtest length is met
5. **Multiple Testing**: Use Deflated Sharpe to adjust for trials
6. **High PBO**: Strategy won't generalize, reduce complexity
7. **Walk-Forward Decay**: Performance degrading over time indicates regime change
8. **Not Retraining**: ML models must be retrained in walk-forward

## References

Based on methodologies from:
- López de Prado, M. (2018). *Advances in Financial Machine Learning*
- Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Bailey, D. H., et al. (2014). "The Probability of Backtest Overfitting"

## Next Steps

After validation:
1. Document all validation metrics in strategy report
2. If PBO > 0.50, reduce strategy complexity or add regularization
3. If DSR < 0.50, increase sample size or reconsider strategy
4. Compare validation results across multiple strategies
5. Monitor out-of-sample performance in production
