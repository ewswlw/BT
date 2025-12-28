---
name: strategy-development
description: Guidelines for developing new trading strategies including signal generation, ML models, rule-based systems, and integration with the backtesting framework
---

# Strategy Development Guide

## Overview
This skill provides comprehensive guidance on developing new trading strategies for the backtesting framework. Use this when creating or modifying strategies.

## When to Use This Skill
Claude should use this when:
- Creating new trading strategies
- Modifying existing strategy implementations
- Debugging strategy signal generation
- Optimizing strategy parameters
- Integrating ML models into strategies

## BaseStrategy Pattern

### Abstract Base Class
All strategies must inherit from `BaseStrategy`:

```python
from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, config: dict):
        """
        Initialize strategy with configuration.

        Args:
            config: Strategy-specific configuration dictionary
        """
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate trading signals.

        Args:
            data: DataFrame with price and feature data

        Returns:
            Tuple of (entry_signals, exit_signals) as boolean Series
        """
        pass

    def validate_signals(
        self,
        entry: pd.Series,
        exit: pd.Series
    ) -> None:
        """Validate signal integrity."""
        assert isinstance(entry, pd.Series), "Entry signals must be Series"
        assert isinstance(exit, pd.Series), "Exit signals must be Series"
        assert entry.dtype == bool, "Entry signals must be boolean"
        assert exit.dtype == bool, "Exit signals must be boolean"
        assert not (entry & exit).any(), "Entry and exit cannot overlap"
```

## Strategy Types

### 1. Rule-Based Strategies

#### Example: Cross-Asset Momentum
```python
class CrossAssetMomentumStrategy(BaseStrategy):
    """Weekly rebalancing momentum across multiple assets."""

    def generate_signals(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """Generate momentum signals with weekly rebalancing."""

        # Extract configuration
        lookback = self.config.get('lookback_period', 20)
        rebalance_freq = self.config.get('rebalance_frequency', 'W')

        # Calculate momentum
        momentum = data['Close'].pct_change(lookback)

        # Weekly rebalancing logic
        rebalance_dates = data.resample(rebalance_freq).last().index

        # Entry: positive momentum on rebalance dates
        entry_signals = pd.Series(False, index=data.index)
        entry_signals.loc[rebalance_dates] = (
            momentum.loc[rebalance_dates] > 0
        )

        # Exit: negative momentum on rebalance dates
        exit_signals = pd.Series(False, index=data.index)
        exit_signals.loc[rebalance_dates] = (
            momentum.loc[rebalance_dates] < 0
        )

        return entry_signals, exit_signals
```

#### Example: Volatility-Adaptive Momentum
```python
class VolAdaptiveMomentumStrategy(BaseStrategy):
    """VIX-based volatility-adaptive momentum strategy."""

    def generate_signals(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """Generate signals adjusted for volatility regime."""

        # Configuration
        mom_lookback = self.config.get('momentum_lookback', 60)
        vix_threshold = self.config.get('vix_threshold', 20)

        # Calculate momentum
        momentum = data['Close'].pct_change(mom_lookback)

        # Volatility regime (assuming VIX in data)
        low_vol_regime = data['VIX'] < vix_threshold

        # Entry: positive momentum in low volatility
        entry_signals = (momentum > 0) & low_vol_regime

        # Exit: negative momentum or high volatility
        exit_signals = (momentum < 0) | (~low_vol_regime)

        return entry_signals, exit_signals
```

**Rule-Based Best Practices:**
- Use simple, interpretable rules
- Make all parameters configurable
- Document regime logic clearly
- Support multiple timeframes
- Avoid overfitting to specific periods

### 2. Machine Learning Strategies

#### Example: LightGBM Strategy
```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split

class LightGBMStrategy(BaseStrategy):
    """LightGBM gradient boosting ML strategy."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.model = None
        self.feature_cols = None

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix for training."""
        # Remove non-feature columns
        exclude_cols = ['Close', 'Open', 'High', 'Low', 'Volume', 'target']
        self.feature_cols = [
            col for col in data.columns
            if col not in exclude_cols
        ]
        return data[self.feature_cols]

    def create_target(
        self,
        data: pd.DataFrame,
        forward_periods: int = 5
    ) -> pd.Series:
        """Create forward-looking target variable."""
        future_return = (
            data['Close'].shift(-forward_periods) / data['Close'] - 1
        )
        # Binary classification: 1 if positive return, 0 otherwise
        return (future_return > 0).astype(int)

    def train_model(self, data: pd.DataFrame) -> None:
        """Train LightGBM model."""
        # Prepare features and target
        X = self.prepare_features(data)
        y = self.create_target(data, self.config.get('forward_periods', 5))

        # Remove NaN rows
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]

        # Train/test split (use walk-forward in production)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # LightGBM parameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': self.config.get('num_leaves', 31),
            'learning_rate': self.config.get('learning_rate', 0.05),
            'feature_fraction': self.config.get('feature_fraction', 0.8),
            'verbose': -1
        }

        # Train model
        train_data = lgb.Dataset(X_train, label=y_train)
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.get('num_boost_round', 100)
        )

    def generate_signals(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """Generate ML-based signals."""
        # Train model if not already trained
        if self.model is None:
            self.train_model(data)

        # Prepare features
        X = self.prepare_features(data)

        # Predict probabilities
        predictions = pd.Series(
            self.model.predict(X),
            index=data.index
        )

        # Generate signals based on threshold
        threshold = self.config.get('prediction_threshold', 0.5)

        entry_signals = predictions > threshold
        exit_signals = predictions < (1 - threshold)

        return entry_signals, exit_signals
```

#### Example: Random Forest Ensemble
```python
from sklearn.ensemble import RandomForestClassifier

class RFEnsembleStrategy(BaseStrategy):
    """4-model Random Forest ensemble strategy."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.models = []
        self.n_models = 4

    def train_ensemble(self, data: pd.DataFrame) -> None:
        """Train ensemble of Random Forest models."""
        X = self.prepare_features(data)
        y = self.create_target(data)

        # Remove NaN
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]

        # Train multiple models with different random states
        self.models = []
        for i in range(self.n_models):
            rf = RandomForestClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 10),
                random_state=i,
                n_jobs=-1
            )
            rf.fit(X, y)
            self.models.append(rf)

    def generate_signals(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """Generate ensemble predictions."""
        if not self.models:
            self.train_ensemble(data)

        X = self.prepare_features(data)

        # Average predictions across ensemble
        ensemble_predictions = np.mean([
            model.predict_proba(X)[:, 1]
            for model in self.models
        ], axis=0)

        predictions = pd.Series(ensemble_predictions, index=data.index)

        threshold = self.config.get('threshold', 0.5)
        entry_signals = predictions > threshold
        exit_signals = predictions < (1 - threshold)

        return entry_signals, exit_signals
```

**ML Strategy Best Practices:**
- Use walk-forward analysis, not simple train/test split
- Avoid look-ahead bias (never use future data)
- Validate with purged K-fold cross-validation
- Monitor for overfitting using validation framework
- Use feature importance analysis
- Implement early stopping
- Save and version models
- Document hyperparameter choices

### 3. Evolutionary/Genetic Algorithm Strategies

#### Example: Genetic Algorithm Strategy
```python
from deap import base, creator, tools, algorithms
import numpy as np

class GeneticAlgorithmStrategy(BaseStrategy):
    """Genetic algorithm-optimized trading rules."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.best_rules = None

    def fitness_function(
        self,
        individual: list,
        data: pd.DataFrame
    ) -> tuple:
        """
        Evaluate fitness of trading rule individual.

        Args:
            individual: List of rule parameters
            data: Price and feature data

        Returns:
            Tuple with single fitness value (Sharpe ratio)
        """
        # Decode individual into trading rules
        # individual = [ma_short, ma_long, rsi_threshold, ...]
        ma_short = int(individual[0])
        ma_long = int(individual[1])
        rsi_threshold = individual[2]

        # Generate signals based on rules
        sma_short = data['Close'].rolling(ma_short).mean()
        sma_long = data['Close'].rolling(ma_long).mean()
        rsi = data.get('RSI', pd.Series(50, index=data.index))

        entry = (sma_short > sma_long) & (rsi < rsi_threshold)
        exit = (sma_short < sma_long) | (rsi > (100 - rsi_threshold))

        # Quick backtest to calculate Sharpe ratio
        returns = data['Close'].pct_change()
        strategy_returns = returns.where(entry.shift(1), 0)

        if strategy_returns.std() == 0:
            return (0.0,)

        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        return (sharpe,)

    def optimize_rules(self, data: pd.DataFrame) -> list:
        """Run genetic algorithm optimization."""
        # Setup DEAP
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # Gene ranges: [ma_short (5-50), ma_long (50-200), rsi_threshold (20-80)]
        toolbox.register("ma_short", np.random.randint, 5, 50)
        toolbox.register("ma_long", np.random.randint, 50, 200)
        toolbox.register("rsi_threshold", np.random.uniform, 20, 80)

        toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            (toolbox.ma_short, toolbox.ma_long, toolbox.rsi_threshold),
            n=1
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.fitness_function, data=data)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Run evolution
        pop = toolbox.population(n=self.config.get('population_size', 50))
        hof = tools.HallOfFame(1)

        algorithms.eaSimple(
            pop, toolbox,
            cxpb=0.5, mutpb=0.2,
            ngen=self.config.get('generations', 20),
            halloffame=hof,
            verbose=False
        )

        return hof[0]

    def generate_signals(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """Generate signals using optimized rules."""
        if self.best_rules is None:
            self.best_rules = self.optimize_rules(data)

        # Apply best rules
        ma_short = int(self.best_rules[0])
        ma_long = int(self.best_rules[1])
        rsi_threshold = self.best_rules[2]

        sma_short = data['Close'].rolling(ma_short).mean()
        sma_long = data['Close'].rolling(ma_long).mean()
        rsi = data.get('RSI', pd.Series(50, index=data.index))

        entry_signals = (sma_short > sma_long) & (rsi < rsi_threshold)
        exit_signals = (sma_short < sma_long) | (rsi > (100 - rsi_threshold))

        return entry_signals, exit_signals
```

**Evolutionary Strategy Best Practices:**
- Define clear fitness functions (Sharpe, Sortino, Calmar)
- Use constraints to prevent unrealistic parameters
- Run multiple generations with different seeds
- Validate on out-of-sample data
- Monitor for overfitting
- Document evolved parameters

## Strategy Configuration

### Configuration Template
```yaml
strategies:
  my_new_strategy:
    enabled: true
    # Strategy-specific parameters
    lookback_period: 20
    threshold: 0.5
    rebalance_frequency: 'W'

    # ML-specific (if applicable)
    model_type: 'lightgbm'
    num_leaves: 31
    learning_rate: 0.05

    # Risk management
    max_position_size: 1.0
    stop_loss: 0.05
    take_profit: 0.10
```

## Strategy Registration

### Adding to StrategyFactory
```python
# In strategies/strategy_factory.py

class StrategyFactory:
    STRATEGY_MAP = {
        'cross_asset_momentum': CrossAssetMomentumStrategy,
        'vol_adaptive_momentum': VolAdaptiveMomentumStrategy,
        'lightgbm_strategy': LightGBMStrategy,
        'rf_ensemble_strategy': RFEnsembleStrategy,
        'genetic_algorithm': GeneticAlgorithmStrategy,
        # Add your new strategy here:
        'my_new_strategy': MyNewStrategy,
    }
```

## Testing Strategies

### Unit Test Template
```python
# In tests/test_strategies.py

def test_my_new_strategy(sample_data):
    """Test MyNewStrategy signal generation."""
    config = {
        'lookback_period': 20,
        'threshold': 0.5
    }

    strategy = MyNewStrategy(config)
    entry, exit = strategy.generate_signals(sample_data)

    # Assertions
    assert isinstance(entry, pd.Series)
    assert isinstance(exit, pd.Series)
    assert entry.dtype == bool
    assert exit.dtype == bool
    assert not (entry & exit).any()  # No overlap
    assert len(entry) == len(sample_data)
```

## Common Pitfalls

1. **Look-Ahead Bias**: Never use `.shift(-n)` in signal generation
2. **Data Snooping**: Don't optimize on full dataset
3. **Overfitting**: Use cross-validation and validation framework
4. **Signal Leakage**: Ensure signals based only on past data
5. **Position Conflicts**: Entry and exit cannot be True simultaneously
6. **Index Alignment**: Always align signals with data index
7. **NaN Handling**: Handle missing values before signal generation
8. **Rebalancing Logic**: Document and test rebalancing frequency

## Performance Targets

Based on existing strategies:
- **LightGBM Strategy**: 144.6%+ annualized return target
- **RF Ensemble**: 3.86% annualized return (conservative)
- **Momentum Strategies**: 15-30% annualized return range

## Validation Requirements

All strategies must be validated using:
1. Walk-forward analysis (252-day train, 63-day test)
2. Purged K-fold cross-validation
3. Combinatorial Purged Cross-Validation (CPCV)
4. Deflated Sharpe Ratio
5. Probability of Backtest Overfitting (PBO)

See `validation-framework` skill for details.

## Next Steps After Development

1. Add strategy to `strategy_factory.py`
2. Create configuration in `config.yaml`
3. Write unit tests in `tests/test_strategies.py`
4. Run backtest: `python main.py --strategies my_new_strategy`
5. Run validation: `python main.py --strategies my_new_strategy --run-validation`
6. Generate reports and analyze results
7. Document strategy rationale and parameters
