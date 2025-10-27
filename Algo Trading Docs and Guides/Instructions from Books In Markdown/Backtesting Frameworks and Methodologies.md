# Backtesting Frameworks and Methodologies

**Comprehensive Guide to Robust Backtesting in Financial Machine Learning**

---

## Introduction

> "Backtesting is not a research tool. It is a validation tool, and one that is fraught with peril." - Marcos López de Prado

This document covers the critical challenges of backtesting and presents advanced methodologies to avoid the common pitfalls that lead to strategy failure.

---

## The Dangers of Backtest Overfitting

### Selection Bias Under Multiple Testing

The fundamental problem: Given enough trials, any researcher will eventually find a strategy that works on historical data purely by chance.

**Probability of False Discovery:**

P(False Positive) = 1 - (1 - α)^N

Where:
- α = significance level (e.g., 0.05)
- N = number of trials

**Example:**
- Test 100 strategies at α = 0.05
- P(at least one false positive) = 1 - (0.95)^100 ≈ 99.4%

### Implementation

```python
import numpy as np

def prob_false_positive(n_trials, alpha=0.05):
    """Calculate probability of at least one false positive"""
    return 1 - (1 - alpha) ** n_trials

# Example
for n in [10, 50, 100, 500]:
    prob = prob_false_positive(n)
    print(f"{n} trials: {prob:.2%} chance of false positive")
```

---

## Deflated Sharpe Ratio

The Deflated Sharpe Ratio (DSR) adjusts for multiple testing.

### Formula

DSR = Φ((SR - E[SR_max]) / √Var[SR_max])

Where:
- SR = estimated Sharpe ratio
- E[SR_max] = expected maximum Sharpe ratio under null
- Φ = standard normal CDF

### Implementation

```python
from scipy.stats import norm
import numpy as np

def deflated_sharpe_ratio(sharpe, n_trials, n_observations, skew=0, kurt=3):
    """
    Calculate Deflated Sharpe Ratio
    
    Parameters:
    -----------
    sharpe : float
        Estimated Sharpe ratio
    n_trials : int
        Number of strategies tested
    n_observations : int
        Number of observations
    skew : float
        Skewness of returns
    kurt : float
        Kurtosis of returns
    
    Returns:
    --------
    float : Deflated Sharpe ratio
    """
    # Expected maximum Sharpe ratio under null
    euler_mascheroni = 0.5772156649
    expected_max_sr = (1 - euler_mascheroni) * norm.ppf(1 - 1.0/n_trials) +                       euler_mascheroni * norm.ppf(1 - 1.0/(n_trials * np.e))
    
    # Variance of Sharpe ratio
    var_sr = (1 + (1 - skew * sharpe + (kurt - 1)/4 * sharpe**2)) / (n_observations - 1)
    
    # Deflated Sharpe ratio
    dsr = norm.cdf((sharpe - expected_max_sr) / np.sqrt(var_sr))
    
    return dsr

# Example
sr = 2.0  # Estimated Sharpe ratio
n_trials = 100  # Tested 100 strategies
n_obs = 1000  # 1000 observations

dsr = deflated_sharpe_ratio(sr, n_trials, n_obs)
print(f"Sharpe Ratio: {sr:.2f}")
print(f"Deflated Sharpe Ratio: {dsr:.2f}")
print(f"Interpretation: {dsr:.1%} confidence this is not a false positive")
```

---

## Combinatorial Purged Cross-Validation for Backtesting

CPCV provides a more robust backtest by testing all possible train/test combinations.

### Implementation

```python
from itertools import combinations
import pandas as pd

def cpcv_backtest(returns, n_splits=10, n_test_splits=2):
    """
    Perform CPCV backtesting
    
    Parameters:
    -----------
    returns : pd.Series
        Strategy returns
    n_splits : int
        Number of groups to split data
    n_test_splits : int
        Number of groups for test set
    
    Returns:
    --------
    dict : Backtest statistics
    """
    # Split returns into groups
    groups = np.array_split(returns, n_splits)
    
    # All combinations of test groups
    test_combos = list(combinations(range(n_splits), n_test_splits))
    
    results = []
    
    for test_ids in test_combos:
        # Get test returns
        test_returns = pd.concat([groups[i] for i in test_ids])
        
        # Calculate performance
        sharpe = test_returns.mean() / test_returns.std() * np.sqrt(252)
        
        results.append({
            'sharpe': sharpe,
            'mean_return': test_returns.mean(),
            'volatility': test_returns.std(),
            'test_groups': test_ids
        })
    
    results_df = pd.DataFrame(results)
    
    return {
        'mean_sharpe': results_df['sharpe'].mean(),
        'std_sharpe': results_df['sharpe'].std(),
        'min_sharpe': results_df['sharpe'].min(),
        'max_sharpe': results_df['sharpe'].max(),
        'all_results': results_df
    }
```

---

## Walk-Forward Analysis

### Implementation

```python
def walk_forward_analysis(data, train_period=252, test_period=63, step=21):
    """
    Perform walk-forward analysis
    
    Parameters:
    -----------
    data : pd.DataFrame
        Price/feature data
    train_period : int
        Training window size
    test_period : int
        Testing window size
    step : int
        Step size for rolling window
    
    Returns:
    --------
    pd.DataFrame : Walk-forward results
    """
    results = []
    
    for i in range(0, len(data) - train_period - test_period, step):
        # Define windows
        train_start = i
        train_end = i + train_period
        test_start = train_end
        test_end = test_start + test_period
        
        # Get data
        train_data = data.iloc[train_start:train_end]
        test_data = data.iloc[test_start:test_end]
        
        # Train model (placeholder)
        # model = train_model(train_data)
        
        # Test model (placeholder)
        # performance = test_model(model, test_data)
        
        results.append({
            'train_start': data.index[train_start],
            'train_end': data.index[train_end-1],
            'test_start': data.index[test_start],
            'test_end': data.index[test_end-1],
            # 'performance': performance
        })
    
    return pd.DataFrame(results)
```

---

## Minimum Backtest Length

López de Prado provides a formula for minimum backtest length:

### Formula

MinBTL = ((SR* / SR)^2 - 1) × N

Where:
- SR* = target Sharpe ratio
- SR = expected Sharpe ratio
- N = number of observations in original backtest

### Implementation

```python
def min_backtest_length(target_sr, expected_sr, n_observations):
    """
    Calculate minimum backtest length
    
    Parameters:
    -----------
    target_sr : float
        Target Sharpe ratio
    expected_sr : float
        Expected Sharpe ratio
    n_observations : int
        Number of observations in backtest
    
    Returns:
    --------
    int : Minimum backtest length
    """
    min_length = ((target_sr / expected_sr)**2 - 1) * n_observations
    return int(np.ceil(min_length))

# Example
target = 2.0
expected = 1.5
n_obs = 1000

min_len = min_backtest_length(target, expected, n_obs)
print(f"Minimum backtest length: {min_len} observations")
print(f"That's {min_len/252:.1f} years of daily data")
```

---

## Probability of Backtest Overfitting (PBO)

PBO measures the probability that backtest performance is due to overfitting.

### Implementation

```python
def probability_backtest_overfitting(returns_matrix):
    """
    Calculate Probability of Backtest Overfitting
    
    Parameters:
    -----------
    returns_matrix : pd.DataFrame
        Returns for multiple strategy configurations
        Rows = time periods, Columns = configurations
    
    Returns:
    --------
    float : PBO estimate
    """
    n_configs = returns_matrix.shape[1]
    
    # Split into IS and OOS
    split_point = len(returns_matrix) // 2
    is_returns = returns_matrix.iloc[:split_point]
    oos_returns = returns_matrix.iloc[split_point:]
    
    # Calculate Sharpe ratios
    is_sharpes = is_returns.mean() / is_returns.std()
    oos_sharpes = oos_returns.mean() / oos_returns.std()
    
    # Find best IS configuration
    best_is_config = is_sharpes.idxmax()
    
    # Count configs where OOS performance < median OOS
    median_oos = oos_sharpes.median()
    n_worse = (oos_sharpes < median_oos).sum()
    
    # PBO
    pbo = n_worse / n_configs
    
    return pbo, {
        'is_sharpes': is_sharpes,
        'oos_sharpes': oos_sharpes,
        'best_is_config': best_is_config,
        'best_is_sharpe': is_sharpes[best_is_config],
        'best_is_oos_sharpe': oos_sharpes[best_is_config]
    }
```

---

## Synthetic Data Generation

Test strategies on synthetic data to assess robustness.

### Implementation

```python
def generate_synthetic_prices(real_returns, n_simulations=1000):
    """
    Generate synthetic price paths
    
    Parameters:
    -----------
    real_returns : pd.Series
        Historical returns
    n_simulations : int
        Number of synthetic paths
    
    Returns:
    --------
    pd.DataFrame : Synthetic price paths
    """
    # Estimate parameters
    mu = real_returns.mean()
    sigma = real_returns.std()
    
    # Generate synthetic returns
    synthetic_returns = np.random.normal(
        mu, sigma, 
        size=(len(real_returns), n_simulations)
    )
    
    # Convert to prices
    initial_price = 100
    synthetic_prices = initial_price * (1 + synthetic_returns).cumprod(axis=0)
    
    return pd.DataFrame(
        synthetic_prices,
        index=real_returns.index,
        columns=[f'sim_{i}' for i in range(n_simulations)]
    )
```

---

## Best Practices

1. **Calculate Deflated Sharpe Ratio** to account for multiple testing
2. **Use CPCV** for robust performance estimation
3. **Perform walk-forward analysis** to test adaptability
4. **Check minimum backtest length** before drawing conclusions
5. **Test on synthetic data** to assess robustness
6. **Monitor PBO** to detect overfitting

---

## References

1. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapters 11-14.
2. Bailey, D. H., et al. (2014). "The Probability of Backtest Overfitting." *Journal of Computational Finance*.
