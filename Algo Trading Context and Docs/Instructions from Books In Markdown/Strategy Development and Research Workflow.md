# Strategy Development and Research Workflow

## When to Use

- Use this workflow when launching a new strategy project or auditing an existing one to ensure every phase—from hypothesis to deployment—is covered.
- Apply it before delegating tasks to agents so they follow a disciplined sequence of idea vetting, labeling, validation, and monitoring.
- Reference it during project reviews to track progress, confirm documentation completeness, and identify missing approvals or QA steps.
- Consult it whenever stakeholders request explanations of your research methodology; the guide provides an end-to-end narrative and supporting code patterns.
- If you are only making minor tweaks to a well-documented strategy, you may not need the entire pipeline; otherwise treat this workflow as mandatory governance.

**End-to-End Process for Developing Trading Strategies**

---

## Introduction

Developing a robust trading strategy requires a systematic workflow that minimizes overfitting while maximizing the probability of success. This document outlines the complete process from idea generation to deployment.

---

## Strategy Research Workflow

### 1. Idea Generation

Sources of strategy ideas:
- **Academic research:** Published papers
- **Market observations:** Anomalies, inefficiencies
- **Economic theory:** Fundamental relationships
- **Alternative data:** New data sources
- **Existing strategies:** Improvements, combinations

### 2. Hypothesis Formation

Formulate a testable hypothesis:

```
"I believe that [SIGNAL] predicts [OUTCOME] because [ECONOMIC RATIONALE]"
```

**Example:**
"I believe that unusual options activity predicts short-term stock price movements because informed traders use options for leverage."

### 3. Data Collection

Gather necessary data:
- Price data
- Fundamental data
- Alternative data
- Benchmark data

### 4. Feature Engineering

Create predictive features:

```python
def engineer_features(prices, volumes, fundamentals):
    """
    Create feature set for strategy
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data
    volumes : pd.DataFrame
        Volume data
    fundamentals : pd.DataFrame
        Fundamental data
    
    Returns:
    --------
    pd.DataFrame : Feature matrix
    """
    features = pd.DataFrame(index=prices.index)
    
    # Price-based features
    features['returns_1d'] = prices.pct_change()
    features['returns_5d'] = prices.pct_change(5)
    features['volatility'] = prices.pct_change().rolling(20).std()
    
    # Volume-based features
    features['volume_ratio'] = volumes / volumes.rolling(20).mean()
    features['volume_trend'] = volumes.rolling(5).mean() / volumes.rolling(20).mean()
    
    # Fundamental features
    if fundamentals is not None:
        features = features.join(fundamentals, how='left')
    
    return features
```

### 5. Label Generation

Use triple-barrier method (see File 02):

```python
from labeling import get_events, get_labels

# Generate labels
events = get_events(
    prices=prices['close'],
    timestamps=prices.index,
    pt_sl=[2, 2],
    target_volatility=volatility,
    vertical_barrier_days=5
)

labels = get_labels(events, prices['close'])
```

### 6. Model Training

Train model with proper cross-validation:

```python
from cross_validation import PurgedKFold
from sklearn.ensemble import RandomForestClassifier

# Purged k-fold CV
cv = PurgedKFold(n_splits=5, pct_embargo=0.01)

# Train model
model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=10,
    min_samples_leaf=100
)

# Cross-validation
scores = cross_val_score(
    model, features, labels,
    cv=cv, scoring='f1'
)

print(f"CV Score: {scores.mean():.3f} +/- {scores.std():.3f}")
```

### 7. Feature Importance Analysis

```python
from feature_importance import mean_decrease_accuracy

# Train final model
model.fit(features_train, labels_train)

# Calculate feature importance
importance = mean_decrease_accuracy(
    model, features_test, labels_test
)

print("Top 10 features:")
print(importance.head(10))
```

### 8. Backtesting

```python
from backtesting import backtest_strategy

results = backtest_strategy(
    model=model,
    features=features,
    prices=prices,
    initial_capital=1000000
)

print(f"Sharpe Ratio: {results['sharpe']:.2f}")
print(f"Max Drawdown: {results['max_dd']:.2%}")
print(f"Win Rate: {results['win_rate']:.2%}")
```

### 9. Robustness Checks

- **Out-of-sample testing:** Reserve recent data
- **Walk-forward analysis:** Rolling windows
- **Synthetic data:** Test on generated data
- **Parameter sensitivity:** Vary hyperparameters
- **Regime analysis:** Performance across market regimes

### 10. Production Deployment

- **Paper trading:** Test with live data, no real money
- **Gradual scale-up:** Start with small capital
- **Monitoring:** Track performance vs backtest
- **Risk management:** Implement stop-losses, position limits

---

## Feature Engineering Best Practices

### Technical Features

```python
def technical_features(prices):
    """Create technical indicator features"""
    features = pd.DataFrame(index=prices.index)
    
    # Moving averages
    features['sma_20'] = prices.rolling(20).mean()
    features['sma_50'] = prices.rolling(50).mean()
    features['sma_ratio'] = features['sma_20'] / features['sma_50']
    
    # Momentum
    features['rsi'] = calculate_rsi(prices, 14)
    features['macd'] = calculate_macd(prices)
    
    # Volatility
    features['bbands_width'] = calculate_bollinger_width(prices)
    features['atr'] = calculate_atr(prices, 14)
    
    return features
```

### Fundamental Features

```python
def fundamental_features(financials):
    """Create fundamental features"""
    features = pd.DataFrame(index=financials.index)
    
    # Valuation
    features['pe_ratio'] = financials['price'] / financials['eps']
    features['pb_ratio'] = financials['price'] / financials['book_value']
    
    # Growth
    features['revenue_growth'] = financials['revenue'].pct_change(4)
    features['earnings_growth'] = financials['earnings'].pct_change(4)
    
    # Quality
    features['roe'] = financials['earnings'] / financials['equity']
    features['debt_ratio'] = financials['debt'] / financials['assets']
    
    return features
```

### Alternative Data Features

```python
def alternative_data_features(sentiment, web_traffic):
    """Create alternative data features"""
    features = pd.DataFrame()
    
    # Sentiment
    features['sentiment_score'] = sentiment['polarity']
    features['sentiment_change'] = sentiment['polarity'].diff()
    
    # Web traffic
    features['traffic_ratio'] = web_traffic / web_traffic.rolling(30).mean()
    features['traffic_trend'] = web_traffic.rolling(7).mean() / web_traffic.rolling(30).mean()
    
    return features
```

---

## Regime Detection

Identify market regimes and adapt strategy:

```python
from sklearn.mixture import GaussianMixture

def detect_regimes(returns, n_regimes=3):
    """
    Detect market regimes using Gaussian Mixture Model
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    n_regimes : int
        Number of regimes
    
    Returns:
    --------
    pd.Series : Regime labels
    """
    # Prepare features for regime detection
    features = pd.DataFrame({
        'returns': returns,
        'volatility': returns.rolling(20).std(),
        'volume': returns.rolling(20).mean()
    }).dropna()
    
    # Fit GMM
    gmm = GaussianMixture(n_components=n_regimes, random_state=42)
    regimes = gmm.fit_predict(features)
    
    return pd.Series(regimes, index=features.index)
```

---

## Multi-Asset Strategies

Extend to multiple assets:

```python
def multi_asset_strategy(prices_dict, model):
    """
    Apply strategy across multiple assets
    
    Parameters:
    -----------
    prices_dict : dict
        Dictionary of price DataFrames {asset: prices}
    model : trained model
        ML model for predictions
    
    Returns:
    --------
    pd.DataFrame : Positions for each asset
    """
    positions = pd.DataFrame()
    
    for asset, prices in prices_dict.items():
        # Generate features
        features = engineer_features(prices)
        
        # Predict
        predictions = model.predict(features)
        
        # Store positions
        positions[asset] = predictions
    
    return positions
```

---

## Best Practices

1. **Start with economic rationale** - Don't data mine
2. **Use proper labeling** - Triple-barrier method
3. **Apply sample weights** - Account for overlapping labels
4. **Use purged CV** - Avoid data leakage
5. **Test robustness** - Multiple validation methods
6. **Monitor in production** - Track vs backtest
7. **Adapt to regimes** - Detect and respond to market changes
8. **Diversify across assets** - Reduce strategy-specific risk

---

## References

1. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
2. Jansen, S. (2020). *Machine Learning for Algorithmic Trading*. Packt Publishing.
