# Machine Learning Models and Ensemble Methods

## When to Use

- Use this document when selecting or configuring ML algorithms for trading problems, particularly when deciding between ensemble methods, deep learning, or reinforcement learning.
- Apply it before delegating model-building work so collaborators understand finance-specific hyperparameters, sample-weight usage, and why bagging often wins.
- Reference it when diagnosing model performance; the bias-variance-noise framing helps determine whether to escalate to more complex architectures.
- Consult it while documenting model choices for stakeholders—the guide includes rationale, code templates, and economic interpretations.
- If your project does not employ ML (e.g., pure rules-based systems), other guides may be more relevant; otherwise treat these recommendations as default best practices.

**Comprehensive Guide to ML Algorithms for Financial Applications**

---

## Introduction

Machine learning models in finance must contend with unique challenges: low signal-to-noise ratios, non-stationarity, and the adversarial nature of markets. This document covers the most effective ML approaches for financial applications, with emphasis on ensemble methods that are particularly well-suited to noisy financial data.

---

## Bias-Variance-Noise Decomposition

The prediction error of any model can be decomposed into three components:

**E[(y - ŷ)²] = Bias² + Variance + Irreducible Error**

### Bias
Bias measures how far off the model's average prediction is from the true value. High bias indicates underfitting.

### Variance  
Variance measures how much the model's predictions vary for different training sets. High variance indicates overfitting.

### Noise
Irreducible error from the inherent randomness in the data.

### Financial Implications

In finance:
- **High noise:** Financial data has very low signal-to-noise ratio
- **Non-stationarity:** Relationships change over time, increasing effective variance
- **Adversarial environment:** Patterns decay as they become known

**Solution:** Ensemble methods, particularly bagging, are ideal for high-noise environments.

---

## Random Forests for Finance

Random Forests are López de Prado's recommended default algorithm for financial ML.

### Why Random Forests Work in Finance

1. **Robust to overfitting:** Bagging reduces variance
2. **Handles non-linearity:** No assumptions about functional form
3. **Feature interactions:** Automatically captures complex relationships
4. **Missing data:** Can handle NaN values
5. **Interpretable:** Feature importance provides insights

### Implementation

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

def train_random_forest(X_train, y_train, sample_weights=None):
    """
    Train Random Forest for financial prediction
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Feature matrix
    y_train : pd.Series
        Labels {-1, +1}
    sample_weights : pd.Series
        Sample weights (from uniqueness calculation)
    
    Returns:
    --------
    Trained Random Forest model
    """
    rf = RandomForestClassifier(
        n_estimators=1000,
        criterion='entropy',  # Better for finance than 'gini'
        max_depth=None,  # Grow full trees
        min_samples_leaf=100,  # Prevent overfitting
        max_features='sqrt',  # Feature bagging
        class_weight='balanced_subsample',  # Handle class imbalance
        n_jobs=-1,  # Use all cores
        random_state=42
    )
    
    rf.fit(X_train, y_train, sample_weight=sample_weights)
    
    return rf
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [500, 1000, 2000],
    'max_depth': [None, 10, 20, 30],
    'min_samples_leaf': [50, 100, 200],
    'max_features': ['sqrt', 'log2', 0.5]
}

rf_random = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,  # Use purged k-fold in practice
    scoring='f1',
    n_jobs=-1
)
```

---

## Bagging vs Boosting

### Bagging (Bootstrap Aggregating)

**Mechanism:** Train multiple models on bootstrap samples, average predictions

**Advantages:**
- Reduces variance
- Parallelizable
- Robust to outliers
- Works well with high-noise data

**Best for:** Financial data (recommended)

### Boosting

**Mechanism:** Sequentially train models, each focusing on previous errors

**Advantages:**
- Reduces bias
- Can achieve very high accuracy

**Disadvantages:**
- Prone to overfitting in noisy data
- Sequential (slower)
- Sensitive to outliers

**Conclusion:** Bagging is generally superior for finance due to high noise levels.

---

## Feature Bagging

Beyond bagging observations, we can bag features to reduce overfitting.

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging_clf = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=1000,
    max_samples=0.8,  # Sample 80% of observations
    max_features=0.8,  # Sample 80% of features
    bootstrap=True,
    bootstrap_features=True,
    n_jobs=-1
)
```

---

## Deep Learning Applications

Deep learning can be effective for specific financial applications:

### Time Series: LSTMs and GRUs

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape, num_classes=2):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

### Autoencoders for Feature Extraction

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def build_autoencoder(input_dim, encoding_dim=32):
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    
    # Models
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, encoder
```

---

## Reinforcement Learning

RL can be used for optimal execution and portfolio management.

### Q-Learning for Trading

```python
import numpy as np

class QLearningTrader:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount=0.95):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.q_table.shape[1])
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state, action] = new_q
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
```

---

## Model Selection Criteria

### Cross-Validation Score
Use purged k-fold CV (see File 05)

### Out-of-Sample Performance
Reserve recent data for final validation

### Sharpe Ratio
Risk-adjusted returns in backtest

### Deflated Sharpe Ratio
Accounts for multiple testing (see File 07)

### Feature Importance Stability
Consistent feature importance across folds

---

## Best Practices

1. **Default to Random Forests** for most applications
2. **Use bagging over boosting** in high-noise environments
3. **Apply sample weights** from uniqueness calculation
4. **Tune hyperparameters** using purged k-fold CV
5. **Monitor feature importance** for stability
6. **Avoid deep learning** unless you have massive datasets

---

## References

1. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
2. Jansen, S. (2020). *Machine Learning for Algorithmic Trading*. Packt Publishing.
