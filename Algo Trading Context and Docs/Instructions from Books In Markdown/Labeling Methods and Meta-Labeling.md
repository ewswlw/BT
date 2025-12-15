# Labeling Methods and Meta-Labeling

## When to Use

- Use this guide whenever you need to construct supervised learning labels for trading models, especially when working with triple-barrier or meta-labeling frameworks.
- Apply it before delegating labeling tasks so agents avoid naive fixed-horizon approaches and respect volatility-adjusted thresholds.
- Reference it when evaluating model performance; poor labels often explain disappointing results, and this document provides diagnostics and best practices.
- Consult it while designing position-sizing logic—the meta-labeling sections clarify how to separate direction from size decisions.
- If your strategy does not require supervised learning labels, you can skip this file; otherwise treat these methods as foundational prerequisites.

**Comprehensive Guide to Supervised Learning Labels in Financial Machine Learning**

---

## Table of Contents

1. [Introduction](#introduction)
2. [The Fixed-Time Horizon Method](#the-fixed-time-horizon-method)
3. [Computing Dynamic Thresholds](#computing-dynamic-thresholds)
4. [The Triple-Barrier Method](#the-triple-barrier-method)
5. [Learning Side and Size](#learning-side-and-size)
6. [Meta-Labeling](#meta-labeling)
7. [Quantamental Strategies](#quantamental-strategies)
8. [Implementation Examples](#implementation-examples)
9. [Best Practices](#best-practices)

---

## Introduction

Labeling is one of the most critical and often overlooked aspects of financial machine learning. While the ML community has devoted enormous resources to developing sophisticated algorithms, relatively little attention has been paid to the question of how to properly label financial data for supervised learning.

As López de Prado emphasizes, **"The quality of your labels determines the ceiling of your model's performance."** No amount of algorithmic sophistication can overcome poorly constructed labels.

This document provides a comprehensive treatment of labeling methods specifically designed for financial applications, with particular focus on the triple-barrier method and meta-labeling—two innovations that address the unique challenges of financial time series.

---

## The Fixed-Time Horizon Method

### Overview

The fixed-time horizon method is by far the most common labeling approach in the financial ML literature. Despite its popularity, it has several critical flaws that make it unsuitable for most real-world applications.

### Mathematical Formulation

Given a features matrix **X** with *I* rows {X_i} drawn from bars with index t = 1, ..., T, an observation X_i is assigned a label y_i ∈ {-1, 0, 1}:

```
y_i = {
    -1  if r_{t_{i,0}, t_{i,0}+h} < -τ
     0  if |r_{t_{i,0}, t_{i,0}+h}| ≤ τ
    +1  if r_{t_{i,0}, t_{i,0}+h} > τ
}
```

Where:
- τ is a pre-defined constant threshold
- t_{i,0} is the index of the bar immediately after X_i
- t_{i,0}+h is the index of the h-th bar after t_{i,0}
- r_{t_{i,0}, t_{i,0}+h} is the price return over horizon h

The return is calculated as:

r_{t_{i,0}, t_{i,0}+h} = (P_{t_{i,0}+h} / P_{t_{i,0}}) - 1

### Example Implementation

```python
import pandas as pd
import numpy as np

def fixed_time_horizon_labels(prices, horizon=10, threshold=0.01):
    """
    Create labels using fixed-time horizon method
    
    Parameters:
    -----------
    prices : pd.Series
        Price series with datetime index
    horizon : int
        Number of bars to look ahead
    threshold : float
        Return threshold for labeling (e.g., 0.01 for 1%)
    
    Returns:
    --------
    pd.Series : Labels {-1, 0, +1}
    """
    labels = pd.Series(index=prices.index, dtype=int)
    
    for i in range(len(prices) - horizon):
        current_price = prices.iloc[i]
        future_price = prices.iloc[i + horizon]
        ret = (future_price / current_price) - 1
        
        if ret > threshold:
            labels.iloc[i] = 1
        elif ret < -threshold:
            labels.iloc[i] = -1
        else:
            labels.iloc[i] = 0
    
    # Last 'horizon' observations cannot be labeled
    labels.iloc[-horizon:] = np.nan
    
    return labels
```

### Critical Flaws

The fixed-time horizon method has three major problems that make it unsuitable for most financial applications.

---

## Computing Dynamic Thresholds

To address the volatility problem, we need dynamic thresholds that adapt to market conditions.

### Daily Volatility Estimation

```python
def get_daily_volatility(prices, span=100):
    """
    Compute daily volatility at intraday estimation points
    
    Parameters:
    -----------
    prices : pd.Series
        Price series with datetime index
    span : int
        Span for exponentially weighted moving average
    
    Returns:
    --------
    pd.Series : Daily volatility estimates
    """
    # Find the index of prices 1 day ago
    df0 = prices.index.searchsorted(prices.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    
    # Create series of prices from 1 day ago
    df0 = pd.Series(
        prices.index[df0 - 1],
        index=prices.index[prices.shape[0] - df0.shape[0]:]
    )
    
    # Calculate daily returns
    daily_returns = prices.loc[df0.index] / prices.loc[df0.values].values - 1
    
    # Apply EWMA to get volatility
    daily_vol = daily_returns.ewm(span=span).std()
    
    return daily_vol
```

---

## The Triple-Barrier Method

The triple-barrier method is López de Prado's solution to the path-dependence problem. It is arguably the most important innovation in financial ML labeling.

### Complete Implementation

```python
def apply_triple_barrier(prices, events, pt_sl, molecule):
    """
    Apply triple-barrier method to label observations
    
    Parameters:
    -----------
    prices : pd.Series
        Price series with datetime index
    events : pd.DataFrame
        DataFrame with columns:
        - 't1': timestamp of vertical barrier
        - 'trgt': unit width of horizontal barriers (volatility)
        - 'side': position side (optional)
    pt_sl : list
        [profit_taking_multiplier, stop_loss_multiplier]
    molecule : list
        Subset of event indices to process
    
    Returns:
    --------
    pd.DataFrame : Barrier touch timestamps
    """
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    
    # Define profit-taking and stop-loss levels
    if pt_sl[0] > 0:
        pt = pt_sl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index)
    
    if pt_sl[1] > 0:
        sl = -pt_sl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index)
    
    # For each event, find first barrier touch
    for loc, t1 in events_['t1'].fillna(prices.index[-1]).items():
        df0 = prices[loc:t1]
        df0 = df0 / prices[loc] - 1
        
        if 'side' in events_.columns:
            df0 = df0 * events_.at[loc, 'side']
        
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()
    
    return out
```

---

## Meta-Labeling

Meta-labeling is one of López de Prado's most important contributions. It addresses how to size positions based on model confidence.

### Implementation

```python
def create_meta_labels(primary_predictions, actual_outcomes):
    """
    Create meta-labels from primary model predictions
    
    Parameters:
    -----------
    primary_predictions : pd.Series
        Primary model's predictions {-1, +1}
    actual_outcomes : pd.Series
        Actual outcomes {-1, +1}
    
    Returns:
    --------
    pd.Series : Meta-labels {0, 1}
    """
    meta_labels = (
        np.sign(primary_predictions) == np.sign(actual_outcomes)
    ).astype(int)
    
    return meta_labels
```

---

## Best Practices

1. **Always Use Triple-Barrier Method** for realistic labeling
2. **Set Realistic Barriers** - pt_sl = [2, 2] is a good starting point
3. **Use Volatility-Adjusted Barriers** - never use fixed thresholds
4. **Consider Meta-Labeling** for position sizing
5. **Monitor Label Distribution** to avoid class imbalance

---

## References

1. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapter 3.
2. Jansen, S. (2020). *Machine Learning for Algorithmic Trading*. Packt Publishing.

