# Sample Weights and Data Sampling

**Comprehensive Guide to Handling Non-IID Data in Financial Machine Learning**

---

## Table of Contents

1. [Introduction](#introduction)
2. [The Problem of Overlapping Outcomes](#the-problem-of-overlapping-outcomes)
3. [Concurrent Labels and Uniqueness](#concurrent-labels-and-uniqueness)
4. [Sequential Bootstrap](#sequential-bootstrap)
5. [Return Attribution](#return-attribution)
6. [Time Decay Weighting](#time-decay-weighting)
7. [Fractional Differentiation](#fractional-differentiation)
8. [Implementation Examples](#implementation-examples)
9. [Best Practices](#best-practices)

---

## Introduction

One of the most fundamental assumptions in machine learning is that training samples are **independently and identically distributed (IID)**. This assumption underpins virtually all statistical learning theory and is critical for the validity of techniques like cross-validation, bootstrap, and ensemble methods.

However, financial data violates this assumption in profound ways. As López de Prado emphasizes:

> "The assumption that observations are IID is almost never satisfied in finance. Ignoring this fact is one of the primary reasons why financial ML models fail in production."

This document provides a comprehensive treatment of techniques for handling non-IID financial data through proper sample weighting and sampling methods.

---

## The Problem of Overlapping Outcomes

### Why Financial Data is Non-IID

Financial observations are non-IID for several reasons:

1. **Overlapping Labels:** When using the triple-barrier method, labels can span multiple bars, creating dependencies
2. **Serial Correlation:** Returns exhibit autocorrelation, especially at high frequencies
3. **Volatility Clustering:** Periods of high/low volatility persist
4. **Regime Changes:** Market dynamics shift over time

### Example of Overlapping Outcomes

Consider three observations with labels determined by the triple-barrier method:

```
Observation 1: Entry at t=0, Exit at t=10
Observation 2: Entry at t=5, Exit at t=15  
Observation 3: Entry at t=8, Exit at t=18

Timeline:
t=0  t=5  t=8  t=10  t=15  t=18
|----1----|
     |----2----|
          |----3----|
```

Observations 1, 2, and 3 all overlap during the period [t=8, t=10]. Any market event during this period affects all three labels simultaneously, violating independence.

### Mathematical Formulation

Let:
- I = total number of observations
- t_{i,0} = entry time for observation i
- t_{i,1} = exit time for observation i (from triple-barrier)

The number of concurrent labels at time t is:

c_t = Σ_{i=1}^I 1_{[t_{i,0} ≤ t ≤ t_{i,1}]}

Where 1_{[condition]} is the indicator function.

### Implementation

```python
import pandas as pd
import numpy as np

def compute_concurrent_labels(label_times):
    """
    Compute number of concurrent labels at each point in time
    
    Parameters:
    -----------
    label_times : pd.DataFrame
        DataFrame with columns 't0' (entry) and 't1' (exit)
    
    Returns:
    --------
    pd.Series : Number of concurrent labels indexed by time
    """
    # Get all unique timestamps
    all_times = pd.DatetimeIndex(
        sorted(set(label_times['t0'].values) | set(label_times['t1'].values))
    )
    
    # Count concurrent labels at each time
    concurrent = pd.Series(0, index=all_times)
    
    for idx, row in label_times.iterrows():
        # Increment count for all times in [t0, t1]
        mask = (all_times >= row['t0']) & (all_times <= row['t1'])
        concurrent[mask] += 1
    
    return concurrent


# Example
label_times = pd.DataFrame({
    't0': pd.to_datetime(['2024-01-01 09:00', '2024-01-01 09:30', '2024-01-01 09:45']),
    't1': pd.to_datetime(['2024-01-01 10:00', '2024-01-01 10:30', '2024-01-01 10:45'])
})

concurrent = compute_concurrent_labels(label_times)
print(f"Max concurrent labels: {concurrent.max()}")
print(f"Mean concurrent labels: {concurrent.mean():.2f}")
```

---

## Concurrent Labels and Uniqueness

### Average Uniqueness

The **uniqueness** of a label quantifies how much it overlaps with other labels. López de Prado defines the uniqueness of label i at time t as:

u_{t,i} = 1_{t,i} / c_t

Where:
- 1_{t,i} = 1 if t ∈ [t_{i,0}, t_{i,1}], 0 otherwise
- c_t = number of concurrent labels at time t

The **average uniqueness** of label i is:

ū_i = (Σ_t u_{t,i}) / (Σ_t 1_{t,i})

This can be interpreted as the reciprocal of the harmonic mean of concurrent labels over the label's lifespan.

### Implementation

```python
def compute_label_uniqueness(label_times, price_index):
    """
    Compute average uniqueness for each label
    
    Parameters:
    -----------
    label_times : pd.DataFrame
        DataFrame with 't0' and 't1' columns
    price_index : pd.DatetimeIndex
        Full price series index
    
    Returns:
    --------
    pd.Series : Average uniqueness for each label
    """
    # First, compute concurrent labels at each bar
    concurrent = pd.Series(0, index=price_index)
    
    for idx, row in label_times.iterrows():
        mask = (price_index >= row['t0']) & (price_index <= row['t1'])
        concurrent[mask] += 1
    
    # Now compute average uniqueness for each label
    uniqueness = pd.Series(index=label_times.index, dtype=float)
    
    for idx, row in label_times.iterrows():
        # Get times during this label's lifespan
        mask = (price_index >= row['t0']) & (price_index <= row['t1'])
        label_times_subset = price_index[mask]
        
        if len(label_times_subset) == 0:
            uniqueness[idx] = 0
            continue
        
        # Compute uniqueness at each time
        u_t = 1.0 / concurrent[label_times_subset]
        
        # Average uniqueness
        uniqueness[idx] = u_t.mean()
    
    return uniqueness


# Example
prices = pd.Series(
    np.random.randn(100).cumsum() + 100,
    index=pd.date_range('2024-01-01', periods=100, freq='1H')
)

label_times = pd.DataFrame({
    't0': prices.index[::10],
    't1': prices.index[::10] + pd.Timedelta(hours=20)
})

uniqueness = compute_label_uniqueness(label_times, prices.index)
print(f"Uniqueness statistics:")
print(uniqueness.describe())
```

---

## Sequential Bootstrap

Standard bootstrap sampling assumes IID data. For financial data with overlapping labels, we need a modified approach.

### The Sequential Bootstrap Algorithm

The sequential bootstrap ensures that sampled observations have, on average, the same uniqueness as the original dataset.

**Algorithm:**

1. Compute average uniqueness ū_i for each observation
2. Draw observations with probability proportional to ū_i
3. After each draw, reduce the uniqueness of overlapping observations
4. Repeat until desired sample size is reached

### Complete Implementation

```python
def sequential_bootstrap(label_times, sample_size=None):
    """
    Perform sequential bootstrap sampling
    
    Parameters:
    -----------
    label_times : pd.DataFrame
        DataFrame with 't0', 't1', and 'uniqueness' columns
    sample_size : int
        Number of samples to draw (default: len(label_times))
    
    Returns:
    --------
    list : Indices of sampled observations
    """
    if sample_size is None:
        sample_size = len(label_times)
    
    # Initialize
    sampled_indices = []
    remaining_uniqueness = label_times['uniqueness'].copy()
    
    for _ in range(sample_size):
        # Compute sampling probabilities (proportional to uniqueness)
        probs = remaining_uniqueness / remaining_uniqueness.sum()
        
        # Sample one observation
        sampled_idx = np.random.choice(
            label_times.index,
            p=probs.values
        )
        sampled_indices.append(sampled_idx)
        
        # Reduce uniqueness of overlapping observations
        sampled_t0 = label_times.loc[sampled_idx, 't0']
        sampled_t1 = label_times.loc[sampled_idx, 't1']
        
        # Find overlapping observations
        overlaps = (
            (label_times['t0'] <= sampled_t1) &
            (label_times['t1'] >= sampled_t0)
        )
        
        # Reduce their uniqueness
        # The reduction factor depends on the overlap duration
        for idx in label_times.index[overlaps]:
            if idx in remaining_uniqueness.index:
                overlap_start = max(label_times.loc[idx, 't0'], sampled_t0)
                overlap_end = min(label_times.loc[idx, 't1'], sampled_t1)
                overlap_duration = (overlap_end - overlap_start).total_seconds()
                label_duration = (
                    label_times.loc[idx, 't1'] - label_times.loc[idx, 't0']
                ).total_seconds()
                
                reduction = overlap_duration / label_duration
                remaining_uniqueness[idx] *= (1 - reduction)
    
    return sampled_indices


# Example usage
label_times_with_uniqueness = label_times.copy()
label_times_with_uniqueness['uniqueness'] = uniqueness

# Perform sequential bootstrap
bootstrap_sample = sequential_bootstrap(
    label_times_with_uniqueness,
    sample_size=len(label_times)
)

print(f"Bootstrap sample indices: {bootstrap_sample}")
print(f"Unique observations in sample: {len(set(bootstrap_sample))}")
```

---

## Return Attribution

When multiple labels overlap, we need to attribute returns fairly across observations.

### Time-Weighted Attribution

The simplest approach is to weight each observation's contribution by its uniqueness:

w_i = ū_i / Σ_j ū_j

### Implementation

```python
def compute_sample_weights(uniqueness):
    """
    Compute sample weights from uniqueness
    
    Parameters:
    -----------
    uniqueness : pd.Series
        Average uniqueness for each observation
    
    Returns:
    --------
    pd.Series : Normalized sample weights
    """
    # Weight proportional to uniqueness
    weights = uniqueness / uniqueness.sum()
    
    # Normalize to sum to number of samples
    weights = weights * len(weights)
    
    return weights


weights = compute_sample_weights(uniqueness)
print(f"Weight statistics:")
print(weights.describe())
```

---

## Time Decay Weighting

More recent observations are often more relevant for prediction. We can combine uniqueness with time decay.

### Exponential Decay

w_i = ū_i × exp(-λ × (T - t_i))

Where:
- T = current time
- t_i = time of observation i
- λ = decay parameter

### Implementation

```python
def compute_time_decay_weights(label_times, uniqueness, decay_factor=0.01):
    """
    Compute weights with time decay
    
    Parameters:
    -----------
    label_times : pd.DataFrame
        Label times with 't0' column
    uniqueness : pd.Series
        Average uniqueness
    decay_factor : float
        Exponential decay rate
    
    Returns:
    --------
    pd.Series : Time-decayed weights
    """
    # Get time since most recent observation (in days)
    most_recent = label_times['t0'].max()
    time_diff = (most_recent - label_times['t0']).dt.total_seconds() / 86400
    
    # Compute decay
    decay = np.exp(-decay_factor * time_diff)
    
    # Combine with uniqueness
    weights = uniqueness * decay
    
    # Normalize
    weights = weights / weights.sum() * len(weights)
    
    return weights


time_weights = compute_time_decay_weights(
    label_times,
    uniqueness,
    decay_factor=0.05
)

print(f"Time-decayed weights:")
print(time_weights.describe())
```

---

## Fractional Differentiation

To achieve stationarity while preserving memory, López de Prado introduces fractional differentiation.

### Mathematical Foundation

Standard differentiation:
Δ^1 X_t = X_t - X_{t-1}

Fractional differentiation:
Δ^d X_t = Σ_{k=0}^∞ w_k X_{t-k}

Where the weights are:
w_k = (-1)^k × Γ(d+1) / (Γ(k+1) × Γ(d-k+1))

### Implementation

```python
def get_weights_ffd(d, threshold=1e-5):
    """
    Compute weights for fractional differentiation
    
    Parameters:
    -----------
    d : float
        Differentiation order (0 < d < 1)
    threshold : float
        Cutoff threshold for weights
    
    Returns:
    --------
    np.array : Weights for fractional differentiation
    """
    w = [1.0]
    k = 1
    
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        
        if abs(w_k) < threshold:
            break
        
        w.append(w_k)
        k += 1
    
    return np.array(w)


def frac_diff_ffd(series, d, threshold=1e-5):
    """
    Apply fractional differentiation with fixed window
    
    Parameters:
    -----------
    series : pd.Series
        Time series to differentiate
    d : float
        Differentiation order
    threshold : float
        Weight cutoff threshold
    
    Returns:
    --------
    pd.Series : Fractionally differentiated series
    """
    # Get weights
    w = get_weights_ffd(d, threshold)
    width = len(w) - 1
    
    # Apply convolution
    output = pd.Series(index=series.index, dtype=float)
    
    for iloc in range(width, len(series)):
        output.iloc[iloc] = np.dot(w, series.iloc[iloc-width:iloc+1][::-1])
    
    return output


# Example
prices = pd.Series(
    np.random.lognormal(0, 0.02, 1000).cumprod() * 100,
    index=pd.date_range('2024-01-01', periods=1000, freq='1H')
)

# Apply fractional differentiation
frac_diff = frac_diff_ffd(prices, d=0.5)

print(f"Original series - ADF statistic: {adf_test(prices)}")
print(f"Frac diff series - ADF statistic: {adf_test(frac_diff.dropna())}")
```

---

## Best Practices

1. **Always compute uniqueness** for overlapping labels
2. **Use sequential bootstrap** instead of standard bootstrap
3. **Apply sample weights** during model training
4. **Consider time decay** for recent observations
5. **Use fractional differentiation** to achieve stationarity

---

## References

1. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapter 4.
2. Jansen, S. (2020). *Machine Learning for Algorithmic Trading*. Packt Publishing.
