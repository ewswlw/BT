# Performance Metrics and Risk Management

## When to Use

- Use this document when evaluating strategy performance beyond vanilla Sharpe ratios, especially for presentations that require statistical confidence and risk-tail metrics.
- Apply it before finalizing risk reporting so you incorporate probabilistic Sharpe, deflated Sharpe, drawdown analytics, and failure probabilities.
- Reference it while designing position sizing or risk overlays; many formulas link model confidence to capital allocation.
- Consult it during post-trade analysis to diagnose whether deviations stem from volatility clustering, tail events, or implementation shortfall.
- If you only need quick summary stats, lighter references may suffice; for institutional-level risk assessment, follow this guide.

**Advanced Metrics for Evaluating Trading Strategies**

---

## Introduction

Traditional performance metrics like Sharpe ratio have significant limitations. This document presents advanced metrics specifically designed for financial applications.

---

## Limitations of the Sharpe Ratio

### Problems

1. **Assumes normality:** Returns are not normally distributed
2. **Ignores higher moments:** Skewness and kurtosis matter
3. **Selection bias:** Easy to inflate through multiple testing
4. **Time aggregation:** Annual SR ≠ √12 × Monthly SR for non-IID returns

---

## Probabilistic Sharpe Ratio (PSR)

PSR computes the probability that the estimated Sharpe ratio exceeds a benchmark.

### Formula

PSR(SR*) = Φ(((SR - SR*) √(N-1)) / √(1 - γ₃SR + (γ₄-1)/4 SR²))

Where:
- SR = estimated Sharpe ratio
- SR* = benchmark Sharpe ratio
- N = number of observations
- γ₃ = skewness
- γ₄ = kurtosis
- Φ = standard normal CDF

### Implementation

```python
from scipy.stats import norm, skew, kurtosis
import numpy as np

def probabilistic_sharpe_ratio(returns, benchmark_sr=0):
    """
    Calculate Probabilistic Sharpe Ratio
    
    Parameters:
    -----------
    returns : pd.Series
        Strategy returns
    benchmark_sr : float
        Benchmark Sharpe ratio
    
    Returns:
    --------
    float : PSR value
    """
    # Calculate statistics
    sr = returns.mean() / returns.std() * np.sqrt(252)
    n = len(returns)
    skewness = skew(returns)
    kurt = kurtosis(returns)
    
    # PSR formula
    numerator = (sr - benchmark_sr) * np.sqrt(n - 1)
    denominator = np.sqrt(1 - skewness * sr + (kurt - 1) / 4 * sr**2)
    
    psr = norm.cdf(numerator / denominator)
    
    return psr

# Example
returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
psr = probabilistic_sharpe_ratio(returns, benchmark_sr=1.0)
print(f"PSR: {psr:.2%}")
print(f"Interpretation: {psr:.1%} confidence SR > 1.0")
```

---

## Drawdown Analysis

### Maximum Drawdown

```python
def calculate_drawdown(prices):
    """
    Calculate drawdown series
    
    Parameters:
    -----------
    prices : pd.Series
        Price series
    
    Returns:
    --------
    pd.DataFrame : Drawdown statistics
    """
    # Calculate running maximum
    running_max = prices.expanding().max()
    
    # Calculate drawdown
    drawdown = (prices - running_max) / running_max
    
    # Maximum drawdown
    max_dd = drawdown.min()
    
    # Time under water
    underwater = drawdown < 0
    
    return pd.DataFrame({
        'drawdown': drawdown,
        'running_max': running_max,
        'underwater': underwater,
        'max_dd': max_dd
    })
```

### Time Under Water

```python
def time_under_water(drawdown_series):
    """
    Calculate time under water periods
    
    Parameters:
    -----------
    drawdown_series : pd.Series
        Drawdown series (negative values)
    
    Returns:
    --------
    pd.DataFrame : Underwater periods
    """
    underwater = drawdown_series < 0
    
    # Find underwater periods
    periods = []
    start = None
    
    for date, is_underwater in underwater.items():
        if is_underwater and start is None:
            start = date
        elif not is_underwater and start is not None:
            periods.append({
                'start': start,
                'end': date,
                'duration': (date - start).days
            })
            start = None
    
    return pd.DataFrame(periods)
```

---

## Bet Sizing from Probabilities

Size positions based on model confidence.

### Kelly Criterion

f* = (p × b - q) / b

Where:
- f* = fraction of capital to bet
- p = probability of winning
- q = 1 - p
- b = odds (win/loss ratio)

### Implementation

```python
def kelly_criterion(win_prob, win_loss_ratio):
    """
    Calculate Kelly fraction
    
    Parameters:
    -----------
    win_prob : float
        Probability of winning (0 to 1)
    win_loss_ratio : float
        Ratio of win size to loss size
    
    Returns:
    --------
    float : Kelly fraction
    """
    loss_prob = 1 - win_prob
    kelly_fraction = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio
    
    # Ensure non-negative
    kelly_fraction = max(0, kelly_fraction)
    
    return kelly_fraction

# Example
prob = 0.55  # 55% win rate
ratio = 1.5  # Win 1.5x what you lose

kelly = kelly_criterion(prob, ratio)
print(f"Kelly fraction: {kelly:.2%}")
print(f"Recommendation: Bet {kelly:.1%} of capital")
```

### Dynamic Position Sizing

```python
def dynamic_position_sizing(prediction_prob, base_size=1.0, kelly_fraction=0.5):
    """
    Size position based on model confidence
    
    Parameters:
    -----------
    prediction_prob : float
        Model's predicted probability (0.5 to 1.0)
    base_size : float
        Base position size
    kelly_fraction : float
        Fraction of Kelly to use (for safety)
    
    Returns:
    --------
    float : Position size
    """
    # Convert probability to bet size
    # Assume 1:1 win/loss ratio for simplicity
    kelly = 2 * prediction_prob - 1
    
    # Apply Kelly fraction for safety
    size = base_size * kelly * kelly_fraction
    
    return max(0, size)
```

---

## Implementation Shortfall

Measure the cost of execution.

### Formula

IS = (Decision Price - Execution Price) / Decision Price

### Implementation

```python
def implementation_shortfall(decision_prices, execution_prices, side):
    """
    Calculate implementation shortfall
    
    Parameters:
    -----------
    decision_prices : pd.Series
        Prices when decision was made
    execution_prices : pd.Series
        Actual execution prices
    side : pd.Series
        Trade side (+1 for buy, -1 for sell)
    
    Returns:
    --------
    pd.Series : Implementation shortfall
    """
    # Calculate shortfall
    shortfall = (execution_prices - decision_prices) / decision_prices
    
    # Adjust for side (shortfall is cost, so flip sign for sells)
    shortfall = shortfall * side
    
    return shortfall
```

---

## Probability of Strategy Failure

Estimate probability of losing a certain amount.

### Implementation

```python
from scipy.stats import norm

def probability_of_loss(returns, loss_threshold=-0.10, horizon=252):
    """
    Calculate probability of loss exceeding threshold
    
    Parameters:
    -----------
    returns : pd.Series
        Historical returns
    loss_threshold : float
        Loss threshold (e.g., -0.10 for -10%)
    horizon : int
        Time horizon in periods
    
    Returns:
    --------
    float : Probability of loss
    """
    # Estimate parameters
    mu = returns.mean()
    sigma = returns.std()
    
    # Expected return and volatility over horizon
    mu_horizon = mu * horizon
    sigma_horizon = sigma * np.sqrt(horizon)
    
    # Probability of loss
    z_score = (loss_threshold - mu_horizon) / sigma_horizon
    prob_loss = norm.cdf(z_score)
    
    return prob_loss

# Example
returns = pd.Series(np.random.normal(0.0005, 0.02, 1000))
prob = probability_of_loss(returns, loss_threshold=-0.20, horizon=252)
print(f"Probability of 20% loss over 1 year: {prob:.2%}")
```

---

## Best Practices

1. **Use PSR instead of SR** for statistical significance
2. **Calculate deflated SR** to account for multiple testing
3. **Monitor drawdown and time under water** for risk management
4. **Size positions dynamically** based on model confidence
5. **Measure implementation shortfall** to account for execution costs
6. **Estimate probability of failure** for risk assessment

---

## References

1. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapters 14-16.
2. Bailey, D. H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier." *Journal of Risk*.
