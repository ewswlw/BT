---
name: python-financial
description: Python best practices for financial computing including numerical stability, vectorization, time series handling, performance optimization, and common pitfalls in quantitative finance
---

# Python Financial Computing Guide

## Overview
Best practices for writing robust, efficient Python code for financial applications and quantitative trading systems.

## When to Use This Skill
Claude should use this when:
- Writing financial calculations or analytics
- Optimizing performance-critical trading code
- Handling time series data
- Implementing numerical algorithms
- Working with financial data pipelines

## Numerical Stability

### Avoid Catastrophic Cancellation
```python
# BAD: Direct subtraction of similar numbers
def calculate_return_bad(price_t1: float, price_t0: float) -> float:
    return (price_t1 - price_t0) / price_t0

# GOOD: Use logarithmic returns for better numerical stability
def calculate_log_return(price_t1: float, price_t0: float) -> float:
    return np.log(price_t1 / price_t0)

# GOOD: For small changes, use np.expm1 and np.log1p
def calculate_return_stable(price_t1: float, price_t0: float) -> float:
    return np.expm1(np.log(price_t1 / price_t0))
```

### Sharpe Ratio Calculation
```python
# BAD: Can have numerical issues with small denominators
def sharpe_ratio_bad(returns: np.ndarray, risk_free: float = 0.0) -> float:
    return (returns.mean() - risk_free) / returns.std()

# GOOD: Handle edge cases and use degrees of freedom
def sharpe_ratio_robust(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Calculate annualized Sharpe ratio with proper error handling."""
    excess_returns = returns - risk_free

    if len(excess_returns) < 2:
        return np.nan

    std = excess_returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan

    return (excess_returns.mean() / std) * np.sqrt(periods_per_year)
```

### Covariance and Correlation
```python
# GOOD: Use pandas built-ins for robustness
def calculate_correlation(
    returns1: pd.Series,
    returns2: pd.Series
) -> float:
    """Calculate correlation with proper alignment and NaN handling."""
    # Align series by index
    aligned = pd.DataFrame({'r1': returns1, 'r2': returns2}).dropna()

    if len(aligned) < 2:
        return np.nan

    return aligned['r1'].corr(aligned['r2'])

# GOOD: Exponentially weighted covariance
def ewma_covariance(
    returns1: pd.Series,
    returns2: pd.Series,
    span: int = 60
) -> pd.Series:
    """Calculate exponentially weighted moving average covariance."""
    aligned = pd.DataFrame({'r1': returns1, 'r2': returns2})
    return aligned['r1'].ewm(span=span).cov(aligned['r2'])
```

## Vectorization

### Use Pandas/NumPy Operations
```python
# BAD: Python loops are slow
def calculate_returns_slow(prices: pd.Series) -> pd.Series:
    returns = pd.Series(index=prices.index[1:], dtype=float)
    for i in range(1, len(prices)):
        returns.iloc[i-1] = (prices.iloc[i] / prices.iloc[i-1]) - 1
    return returns

# GOOD: Vectorized pandas operations
def calculate_returns_fast(prices: pd.Series) -> pd.Series:
    return prices.pct_change()

# BETTER: For log returns
def calculate_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1))
```

### Moving Averages
```python
# BAD: Manual calculation with loops
def sma_slow(prices: pd.Series, window: int) -> pd.Series:
    sma = pd.Series(index=prices.index, dtype=float)
    for i in range(window, len(prices)):
        sma.iloc[i] = prices.iloc[i-window:i].mean()
    return sma

# GOOD: Pandas rolling
def sma_fast(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window=window).mean()

# GOOD: Exponentially weighted moving average
def ewma(prices: pd.Series, span: int) -> pd.Series:
    return prices.ewm(span=span, adjust=False).mean()
```

### Conditional Operations
```python
# BAD: Using loops for conditional logic
def apply_stop_loss_slow(
    prices: pd.Series,
    entry_price: float,
    stop_pct: float
) -> pd.Series:
    stops = pd.Series(index=prices.index, dtype=bool)
    for i in range(len(prices)):
        stops.iloc[i] = (prices.iloc[i] / entry_price - 1) < -stop_pct
    return stops

# GOOD: Vectorized boolean operations
def apply_stop_loss_fast(
    prices: pd.Series,
    entry_price: float,
    stop_pct: float
) -> pd.Series:
    return (prices / entry_price - 1) < -stop_pct

# GOOD: np.where for conditional values
def calculate_position_size(
    signals: pd.Series,
    volatility: pd.Series,
    target_vol: float = 0.15
) -> pd.Series:
    """Scale positions by inverse volatility."""
    base_size = np.where(signals > 0, 1.0, 0.0)
    vol_scalar = target_vol / volatility.clip(lower=0.01)
    return base_size * vol_scalar.clip(upper=2.0)
```

## Time Series Best Practices

### Index Management
```python
# GOOD: Always use DatetimeIndex for time series
def prepare_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare time series with proper datetime index."""
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Sort by time
    df = df.sort_index()

    # Remove duplicates, keeping last
    df = df[~df.index.duplicated(keep='last')]

    return df

# GOOD: Handle timezone awareness
def make_timezone_aware(
    df: pd.DataFrame,
    timezone: str = 'UTC'
) -> pd.DataFrame:
    """Convert to timezone-aware datetime index."""
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone)
    else:
        df.index = df.index.tz_convert(timezone)
    return df
```

### Resampling
```python
# GOOD: Proper resampling with aggregation
def resample_ohlc(
    prices: pd.DataFrame,
    freq: str = 'W'
) -> pd.DataFrame:
    """Resample to weekly OHLC bars."""
    return prices.resample(freq).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

# GOOD: Business day resampling
def resample_business_days(
    returns: pd.Series
) -> pd.Series:
    """Resample to business days, forward-filling gaps."""
    return returns.resample('B').ffill()
```

### Forward-Looking Data Prevention
```python
# BAD: Look-ahead bias - using future data
def generate_signals_bad(data: pd.DataFrame) -> pd.Series:
    # This uses future data to normalize!
    normalized = (data['price'] - data['price'].mean()) / data['price'].std()
    return normalized > 1.0

# GOOD: Only use past data (expanding window)
def generate_signals_good(data: pd.DataFrame) -> pd.Series:
    """Generate signals using only historical data."""
    # Expanding mean/std uses only past data
    normalized = (
        (data['price'] - data['price'].expanding().mean()) /
        data['price'].expanding().std()
    )
    return normalized > 1.0

# GOOD: Proper shift for signal execution
def apply_signal_lag(signals: pd.Series) -> pd.Series:
    """Shift signals forward to represent next-day execution."""
    # Signal generated today, executed tomorrow
    return signals.shift(1)
```

## Memory Optimization

### Efficient Data Types
```python
# GOOD: Use appropriate dtypes
def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce DataFrame memory usage."""
    for col in df.select_dtypes(include=['int']).columns:
        col_min = df[col].min()
        col_max = df[col].max()

        if col_min >= 0:
            if col_max < 255:
                df[col] = df[col].astype(np.uint8)
            elif col_max < 65535:
                df[col] = df[col].astype(np.uint16)
        else:
            if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)

    for col in df.select_dtypes(include=['float']).columns:
        df[col] = df[col].astype(np.float32)

    return df

# GOOD: Use categorical for repeated strings
def optimize_categorical(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Convert columns to categorical."""
    for col in cols:
        if df[col].nunique() / len(df[col]) < 0.5:  # Less than 50% unique
            df[col] = df[col].astype('category')
    return df
```

### Chunking Large Data
```python
# GOOD: Process data in chunks
def calculate_returns_chunked(
    csv_path: str,
    chunksize: int = 10000
) -> pd.Series:
    """Calculate returns for large CSV in chunks."""
    returns_list = []

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk_returns = chunk['Close'].pct_change()
        returns_list.append(chunk_returns)

    return pd.concat(returns_list)
```

## Performance Monitoring

### Timing Operations
```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    """Context manager for timing operations."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{name}: {elapsed:.4f} seconds")

# Usage
with timer("Calculate signals"):
    signals = strategy.generate_signals(data)
```

### Profiling
```python
# Use line_profiler for bottleneck identification
# Install: pip install line_profiler

# Add @profile decorator to function (no import needed)
# Run: kernprof -l -v script.py

@profile
def calculate_features(data: pd.DataFrame) -> pd.DataFrame:
    """This function will be profiled."""
    # Feature calculations
    data['sma_20'] = data['Close'].rolling(20).mean()
    data['rsi'] = calculate_rsi(data['Close'])
    return data
```

## Common Financial Calculations

### Returns
```python
def calculate_returns(
    prices: pd.Series,
    method: str = 'simple'
) -> pd.Series:
    """
    Calculate returns.

    Args:
        prices: Price series
        method: 'simple', 'log', or 'pct'
    """
    if method == 'simple':
        return prices / prices.shift(1) - 1
    elif method == 'log':
        return np.log(prices / prices.shift(1))
    elif method == 'pct':
        return prices.pct_change()
    else:
        raise ValueError(f"Unknown method: {method}")
```

### Drawdown
```python
def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
    """Calculate drawdown from equity curve."""
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown."""
    return calculate_drawdown(equity_curve).min()
```

### Volatility
```python
def calculate_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252
) -> pd.Series:
    """Calculate rolling volatility."""
    vol = returns.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol

def calculate_ewma_volatility(
    returns: pd.Series,
    span: int = 60,
    annualize: bool = True,
    periods_per_year: int = 252
) -> pd.Series:
    """Calculate EWMA volatility."""
    vol = returns.ewm(span=span).std()
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol
```

### Beta and Alpha
```python
def calculate_beta(
    returns: pd.Series,
    market_returns: pd.Series,
    window: int = 252
) -> pd.Series:
    """Calculate rolling beta."""
    # Align series
    aligned = pd.DataFrame({
        'returns': returns,
        'market': market_returns
    }).dropna()

    # Rolling covariance and variance
    cov = aligned['returns'].rolling(window).cov(aligned['market'])
    var = aligned['market'].rolling(window).var()

    return cov / var

def calculate_alpha(
    returns: pd.Series,
    market_returns: pd.Series,
    risk_free: float = 0.0,
    window: int = 252
) -> pd.Series:
    """Calculate rolling alpha (Jensen's alpha)."""
    beta = calculate_beta(returns, market_returns, window)

    # Excess returns
    excess_returns = returns - risk_free
    excess_market = market_returns - risk_free

    # Alpha = Portfolio return - (Risk-free + Beta * Market excess return)
    return excess_returns - beta * excess_market
```

## Error Handling

### Robust Division
```python
def safe_divide(
    numerator: Union[float, pd.Series, np.ndarray],
    denominator: Union[float, pd.Series, np.ndarray],
    fill_value: float = 0.0
) -> Union[float, pd.Series, np.ndarray]:
    """Safely divide, handling zeros and NaNs."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator

    if isinstance(result, pd.Series):
        result = result.replace([np.inf, -np.inf], fill_value)
        result = result.fillna(fill_value)
    elif isinstance(result, np.ndarray):
        result[~np.isfinite(result)] = fill_value
    elif not np.isfinite(result):
        result = fill_value

    return result
```

### Data Validation
```python
def validate_price_data(df: pd.DataFrame) -> None:
    """Validate price data integrity."""
    # Check for required columns
    required = ['Open', 'High', 'Low', 'Close']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Check for negative prices
    if (df[required] < 0).any().any():
        raise ValueError("Found negative prices")

    # Check High >= Low
    if (df['High'] < df['Low']).any():
        raise ValueError("High < Low found")

    # Check for excessive gaps (> 3 standard deviations)
    returns = df['Close'].pct_change()
    if (np.abs(returns) > 3 * returns.std()).any():
        print("Warning: Detected extreme price movements")
```

## Testing Financial Code

### Property-Based Testing
```python
import hypothesis
from hypothesis import given, strategies as st

@given(
    prices=st.lists(
        st.floats(min_value=1.0, max_value=1000.0),
        min_size=10,
        max_size=100
    )
)
def test_returns_inverse_property(prices):
    """Test that compounding returns reconstructs prices."""
    prices_series = pd.Series(prices)
    returns = prices_series.pct_change().fillna(0)

    # Reconstruct prices
    reconstructed = prices[0] * (1 + returns).cumprod()

    # Should match original (within floating point error)
    assert np.allclose(prices_series, reconstructed, rtol=1e-10)
```

## Common Pitfalls

1. **Look-Ahead Bias**: Using `.mean()` instead of `.expanding().mean()`
2. **Survivor Bias**: Only testing on assets that still exist today
3. **Division by Zero**: Not handling zero volatility or zero prices
4. **Data Alignment**: Forgetting to align indices when combining series
5. **Timezone Issues**: Mixing timezone-aware and naive datetimes
6. **Floating Point**: Comparing floats with `==` instead of `np.isclose()`
7. **Missing Data**: Not handling NaN values before calculations
8. **Pandas Copy Warning**: Modifying slices without `.copy()`

## Performance Checklist

- [ ] Use vectorized pandas/numpy operations instead of loops
- [ ] Use appropriate dtypes (float32 vs float64, categorical for strings)
- [ ] Profile code to find bottlenecks
- [ ] Use `.loc` and `.iloc` instead of chained indexing
- [ ] Avoid repeatedly calling expensive operations in loops
- [ ] Use `.values` or `.to_numpy()` for NumPy operations on pandas objects
- [ ] Preallocate arrays when size is known
- [ ] Use `.rolling()` instead of manual window calculations
- [ ] Consider using Numba or Cython for critical inner loops
- [ ] Cache expensive calculations when possible

## Libraries to Know

**Core:**
- pandas: Data manipulation
- numpy: Numerical computing
- scipy: Statistical functions

**Financial:**
- vectorbt: Backtesting and portfolio analysis
- quantstats: Performance metrics
- ta-lib: Technical indicators (C library wrapper)

**ML:**
- scikit-learn: ML models and validation
- lightgbm, xgboost, catboost: Gradient boosting

**Performance:**
- numba: JIT compilation
- dask: Parallel computing for large datasets
- vaex: Out-of-core DataFrames
