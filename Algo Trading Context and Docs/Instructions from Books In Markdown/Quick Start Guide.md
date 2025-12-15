# Quick Start Guide

## When to Use

- Use this quick start when you need an orientation to the knowledge base and want curated jump-off points for strategy building, enhancement, or deployment.
- Apply it during onboarding sessions so teammates know which files to read first for their specific goal.
- Reference it when triaging tasks; the guide maps common workflows (new strategy, improvement, production) to the relevant documentation sequence.
- Consult it for reminders of the highest-impact practices (information-driven bars, triple-barrier labeling, purged CV) before kicking off a build.
- If you already know the documentation landscape well, you can skip to individual files; otherwise start here.

**Algorithmic Trading & Financial ML Knowledge Base**

---

## What You Have

A comprehensive, production-ready knowledge base with **12 detailed documentation files** covering all aspects of algorithmic trading, backtesting, and financial machine learning.

**Total:** 5,400+ lines of documentation, 180KB of content

---

## Start Here

### If you're building a new trading strategy:

1. **Read File 01** - Learn about proper data structures (dollar bars, volume bars)
2. **Read File 02** - Understand the triple-barrier labeling method
3. **Read File 10** - Follow the systematic strategy development workflow

### If you're improving an existing strategy:

1. **File 05** - Implement purged k-fold cross-validation
2. **File 06** - Calculate deflated Sharpe ratio for your backtest
3. **File 09** - Add advanced microstructure features

### If you're deploying to production:

1. **File 08** - Implement Hierarchical Risk Parity for portfolio construction
2. **File 11** - Optimize code with vectorization and multiprocessing
3. **File 07** - Set up performance monitoring

---

## Most Important Concepts

### 1. Use Information-Driven Bars (File 01)
```python
# Don't use time bars
prices_time = data.resample('1H').last()

# Use dollar bars instead
dollar_bars = create_dollar_bars(ticks, threshold=1e6)
```

### 2. Apply Triple-Barrier Method (File 02)
```python
# Proper labeling with dynamic barriers
events = get_events(
    prices=prices,
    pt_sl=[2, 2],  # 2x volatility for profit/stop
    target_volatility=volatility
)
labels = get_labels(events, prices)
```

### 3. Use Sample Weights (File 03)
```python
# Account for overlapping labels
uniqueness = compute_label_uniqueness(label_times, prices.index)
weights = compute_sample_weights(uniqueness)

# Use in model training
model.fit(X_train, y_train, sample_weight=weights)
```

### 4. Implement Purged K-Fold CV (File 05)
```python
# Prevent data leakage
cv = PurgedKFold(n_splits=5, pct_embargo=0.01)
scores = cross_val_score(model, X, y, cv=cv)
```

### 5. Calculate Deflated Sharpe Ratio (File 06)
```python
# Account for multiple testing
dsr = deflated_sharpe_ratio(
    sharpe=2.0,
    n_trials=100,  # Tested 100 strategies
    n_observations=1000
)
```

---

## File-by-File Overview

| File | Topic | Key Takeaway |
|------|-------|--------------|
| 01 | Data Structures | Use dollar/volume bars, not time bars |
| 02 | Labeling | Triple-barrier method is essential |
| 03 | Sample Weights | Correct for overlapping labels |
| 04 | ML Models | Random Forests are the default choice |
| 05 | Cross-Validation | Use purged k-fold CV |
| 06 | Backtesting | Calculate deflated Sharpe ratio |
| 07 | Performance | Monitor PSR and drawdown |
| 08 | Portfolio | Use HRP, not mean-variance |
| 09 | Features | Add microstructure features (VPIN) |
| 10 | Workflow | Follow systematic development process |
| 11 | Implementation | Vectorize and parallelize |
| 12 | Alternative Data | Incorporate sentiment and alt data |

---

## Common Mistakes to Avoid

### ❌ Don't Do This
```python
# Time bars
data.resample('1H')

# Fixed-time horizon labels
labels = (prices.shift(-10) > prices).astype(int)

# Standard k-fold CV
cv = KFold(n_splits=5)

# Standard bootstrap
bootstrap_sample = np.random.choice(indices, size=len(indices))

# Mean-variance optimization
weights = mean_variance_optimize(returns)
```

### ✅ Do This Instead
```python
# Dollar bars
dollar_bars = create_dollar_bars(ticks, threshold=1e6)

# Triple-barrier labels
labels = apply_triple_barrier(prices, events, pt_sl=[2, 2])

# Purged k-fold CV
cv = PurgedKFold(n_splits=5, pct_embargo=0.01)

# Sequential bootstrap
bootstrap_sample = sequential_bootstrap(label_times)

# Hierarchical Risk Parity
weights = HierarchicalRiskParity(returns).optimize()
```

---

## Recommended Reading Order

### Week 1: Foundations
- Day 1-2: File 01 (Data Structures)
- Day 3-4: File 02 (Labeling)
- Day 5-7: File 03 (Sample Weights)

### Week 2: Machine Learning
- Day 1-3: File 04 (ML Models)
- Day 4-7: File 05 (Cross-Validation)

### Week 3: Backtesting
- Day 1-4: File 06 (Backtesting)
- Day 5-7: File 07 (Performance Metrics)

### Week 4: Advanced Topics
- Day 1-2: File 08 (Portfolio Construction)
- Day 3-4: File 09 (Microstructure)
- Day 5: File 10 (Strategy Development)
- Day 6: File 11 (Implementation)
- Day 7: File 12 (Alternative Data)

---

## Code Templates

### Complete Strategy Template

```python
# 1. Load and structure data
dollar_bars = create_dollar_bars(ticks, threshold=1e6)

# 2. Generate labels
volatility = get_daily_volatility(dollar_bars['close'])
events = get_events(dollar_bars['close'], pt_sl=[2, 2], target=volatility)
labels = get_labels(events, dollar_bars['close'])

# 3. Engineer features
features = engineer_features(dollar_bars)

# 4. Calculate sample weights
uniqueness = compute_label_uniqueness(events, dollar_bars.index)
weights = compute_sample_weights(uniqueness)

# 5. Train model with purged CV
cv = PurgedKFold(n_splits=5, pct_embargo=0.01)
model = RandomForestClassifier(n_estimators=1000)
scores = cross_val_score(model, features, labels, cv=cv, sample_weight=weights)

# 6. Backtest
results = backtest_strategy(model, features, dollar_bars)

# 7. Calculate deflated Sharpe ratio
dsr = deflated_sharpe_ratio(results['sharpe'], n_trials=100, n_obs=len(results))

print(f"Deflated Sharpe Ratio: {dsr:.2f}")
```

---

## Next Steps

1. **Read the README** for complete overview
2. **Pick a file** based on your current need
3. **Implement the code** in your project
4. **Iterate and improve** using the techniques

---

## Questions?

Refer to the original books for complete theoretical foundations:
- *Advances in Financial Machine Learning* by Marcos López de Prado
- *Machine Learning for Algorithmic Trading* by Stefan Jansen

---

**Good luck with your algorithmic trading projects!**
