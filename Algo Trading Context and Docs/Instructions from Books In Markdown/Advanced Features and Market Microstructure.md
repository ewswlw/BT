# Advanced Features and Market Microstructure

## When to Use

- Use this document when engineering features rooted in market microstructure, especially for high-frequency or liquidity-sensitive strategies.
- Apply it after you have baseline price/volume features and need deeper structural signals (tick rule, VPIN, entropy, structural breaks).
- Reference it when validating whether a dataset supports microstructure analysis—many functions assume trade-level timestamps, volumes, and bid/ask info.
- Consult it for code snippets and economic context before delegating feature creation to agents, ensuring they understand generation prerequisites and risks.
- If you're working on broad ML workflows without microstructure data, other topic files may suffice; once microstructure enters scope, use this guide.

**Sophisticated Feature Engineering for Financial ML**

---

## Introduction

Market microstructure provides valuable signals about price formation, information flow, and market quality. This document covers advanced feature engineering techniques based on microstructure theory.

---

## Three Generations of Microstructure Features

### First Generation: Price Sequences

Basic features derived from price and trade sequences.

#### Tick Rule

Classify trades as buyer or seller initiated:

```python
def tick_rule(prices):
    """
    Classify ticks using tick rule
    
    Parameters:
    -----------
    prices : pd.Series
        Trade prices
    
    Returns:
    --------
    pd.Series : Trade direction (+1 buy, -1 sell)
    """
    direction = pd.Series(index=prices.index, dtype=int)
    prev_price = None
    prev_direction = 1
    
    for idx, price in prices.items():
        if prev_price is None:
            direction[idx] = 1
        elif price > prev_price:
            direction[idx] = 1
            prev_direction = 1
        elif price < prev_price:
            direction[idx] = -1
            prev_direction = -1
        else:
            direction[idx] = prev_direction
        
        prev_price = price
    
    return direction
```

#### Roll Model

Estimate effective spread from price changes:

```python
def roll_spread(prices):
    """
    Estimate effective spread using Roll model
    
    Parameters:
    -----------
    prices : pd.Series
        Transaction prices
    
    Returns:
    --------
    float : Estimated spread
    """
    price_changes = prices.diff()
    
    # Roll's estimator
    spread = 2 * np.sqrt(-price_changes.autocorr(lag=1) * price_changes.var())
    
    return spread
```

### Second Generation: Strategic Trade Models

Models that capture strategic trading behavior.

#### Kyle's Lambda

Measures price impact of trades:

```python
def kyles_lambda(price_changes, volumes):
    """
    Estimate Kyle's lambda (price impact)
    
    Parameters:
    -----------
    price_changes : pd.Series
        Price changes
    volumes : pd.Series
        Trade volumes
    
    Returns:
    --------
    float : Kyle's lambda
    """
    from sklearn.linear_model import LinearRegression
    
    # Regression: price_change = lambda * volume
    model = LinearRegression()
    model.fit(volumes.values.reshape(-1, 1), price_changes.values)
    
    return model.coef_[0]
```

#### Amihud's Lambda

Illiquidity measure:

```python
def amihud_lambda(returns, dollar_volumes, window=20):
    """
    Calculate Amihud illiquidity measure
    
    Parameters:
    -----------
    returns : pd.Series
        Returns
    dollar_volumes : pd.Series
        Dollar volumes
    window : int
        Rolling window size
    
    Returns:
    --------
    pd.Series : Amihud lambda
    """
    illiquidity = (returns.abs() / dollar_volumes).rolling(window).mean()
    
    return illiquidity
```

### Third Generation: Sequential Trade Models

Advanced models that estimate informed trading.

#### VPIN (Volume-Synchronized Probability of Informed Trading)

```python
def calculate_vpin(prices, volumes, n_buckets=50):
    """
    Calculate VPIN
    
    Parameters:
    -----------
    prices : pd.Series
        Trade prices
    volumes : pd.Series
        Trade volumes
    n_buckets : int
        Number of volume buckets
    
    Returns:
    --------
    pd.Series : VPIN values
    """
    # Classify trades
    direction = tick_rule(prices)
    
    # Calculate buy and sell volumes
    buy_volume = (direction == 1) * volumes
    sell_volume = (direction == -1) * volumes
    
    # Create volume buckets
    total_volume = volumes.sum()
    bucket_size = total_volume / n_buckets
    
    vpin_values = []
    cumulative_volume = 0
    bucket_buy_vol = 0
    bucket_sell_vol = 0
    
    for i in range(len(volumes)):
        bucket_buy_vol += buy_volume.iloc[i]
        bucket_sell_vol += sell_volume.iloc[i]
        cumulative_volume += volumes.iloc[i]
        
        if cumulative_volume >= bucket_size:
            # Calculate VPIN for this bucket
            vpin = abs(bucket_buy_vol - bucket_sell_vol) / (bucket_buy_vol + bucket_sell_vol)
            vpin_values.append(vpin)
            
            # Reset
            cumulative_volume = 0
            bucket_buy_vol = 0
            bucket_sell_vol = 0
    
    return pd.Series(vpin_values)
```

---

## Entropy Features

Entropy measures the randomness or information content of a series.

### Shannon Entropy

```python
def shannon_entropy(series, n_bins=10):
    """
    Calculate Shannon entropy
    
    Parameters:
    -----------
    series : pd.Series
        Data series
    n_bins : int
        Number of bins for discretization
    
    Returns:
    --------
    float : Entropy value
    """
    # Discretize into bins
    counts, _ = np.histogram(series, bins=n_bins)
    
    # Calculate probabilities
    probs = counts / counts.sum()
    
    # Remove zeros
    probs = probs[probs > 0]
    
    # Calculate entropy
    entropy = -np.sum(probs * np.log2(probs))
    
    return entropy
```

### Plug-In Entropy

```python
def plugin_entropy(returns, window=100):
    """
    Calculate plug-in entropy estimator
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    window : int
        Rolling window size
    
    Returns:
    --------
    pd.Series : Rolling entropy
    """
    entropy = returns.rolling(window).apply(
        lambda x: shannon_entropy(x, n_bins=10)
    )
    
    return entropy
```

---

## Structural Breaks

Detecting regime changes in financial time series.

### CUSUM Test

```python
def cusum_test(series, threshold=None):
    """
    CUSUM test for structural breaks
    
    Parameters:
    -----------
    series : pd.Series
        Time series
    threshold : float
        Detection threshold
    
    Returns:
    --------
    pd.DatetimeIndex : Break points
    """
    if threshold is None:
        threshold = 3 * series.std()
    
    # Calculate cumulative sum of deviations
    mean = series.mean()
    cusum_pos = pd.Series(0.0, index=series.index)
    cusum_neg = pd.Series(0.0, index=series.index)
    
    for i in range(1, len(series)):
        cusum_pos.iloc[i] = max(0, cusum_pos.iloc[i-1] + series.iloc[i] - mean)
        cusum_neg.iloc[i] = min(0, cusum_neg.iloc[i-1] + series.iloc[i] - mean)
    
    # Detect breaks
    breaks = series.index[(cusum_pos > threshold) | (cusum_neg < -threshold)]
    
    return breaks
```

### SADF Test (Explosive Behavior)

```python
from statsmodels.tsa.stattools import adfuller

def sadf_test(series, min_window=20):
    """
    Supremum Augmented Dickey-Fuller test
    
    Parameters:
    -----------
    series : pd.Series
        Price series
    min_window : int
        Minimum window size
    
    Returns:
    --------
    dict : Test results
    """
    adf_stats = []
    
    for end in range(min_window, len(series)):
        subseries = series.iloc[:end]
        adf_stat = adfuller(subseries, regression='c')[0]
        adf_stats.append(adf_stat)
    
    sadf_stat = max(adf_stats)
    
    return {
        'sadf_statistic': sadf_stat,
        'adf_sequence': adf_stats
    }
```

---

## Fractals and Rescaled Range

### Hurst Exponent

```python
def hurst_exponent(series, max_lag=20):
    """
    Calculate Hurst exponent
    
    Parameters:
    -----------
    series : pd.Series
        Time series
    max_lag : int
        Maximum lag for R/S calculation
    
    Returns:
    --------
    float : Hurst exponent
    """
    lags = range(2, max_lag)
    rs_values = []
    
    for lag in lags:
        # Split series into subseries of length lag
        subseries = [series.iloc[i:i+lag] for i in range(0, len(series)-lag, lag)]
        
        rs_lag = []
        for sub in subseries:
            if len(sub) < lag:
                continue
            
            # Calculate R/S for this subseries
            mean = sub.mean()
            cumdev = (sub - mean).cumsum()
            r = cumdev.max() - cumdev.min()
            s = sub.std()
            
            if s > 0:
                rs_lag.append(r / s)
        
        if len(rs_lag) > 0:
            rs_values.append(np.mean(rs_lag))
    
    # Fit log(R/S) = H * log(lag) + c
    log_lags = np.log(lags[:len(rs_values)])
    log_rs = np.log(rs_values)
    
    hurst = np.polyfit(log_lags, log_rs, 1)[0]
    
    return hurst
```

---

## Best Practices

1. **Use VPIN** to detect toxic flow and potential flash crashes
2. **Monitor entropy** to assess market efficiency
3. **Detect structural breaks** before deploying strategies
4. **Calculate Hurst exponent** to identify mean-reversion vs momentum regimes
5. **Combine multiple generations** of microstructure features

---

## References

1. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapters 17-19.
2. Easley, D., López de Prado, M., & O'Hara, M. (2012). "Flow Toxicity and Liquidity in a High-Frequency World." *Review of Financial Studies*.
