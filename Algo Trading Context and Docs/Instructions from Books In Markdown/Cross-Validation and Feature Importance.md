# Cross-Validation and Feature Importance

## When to Use

- Use this guide whenever you validate ML models on financial data or need defensible feature-importance metrics tailored to overlapping labels.
- Apply it before delegating CV work so agents adopt purged k-fold, embargo techniques, and CPCV rather than default scikit-learn splits.
- Reference it when comparing variable importance methods (MDA, SFI, clustered) to ensure noise and correlation effects are handled correctly.
- Consult it during model audits to confirm hyperparameter tuning respected financial CV constraints and that feature rankings support your investment thesis.
- For quick experiments you might start with standard CV, but no production-grade model should bypass the methods documented here.

**Advanced Techniques for Model Validation and Feature Selection in Finance**

---

## Introduction

Standard cross-validation fails in finance due to data leakage from overlapping labels and serial correlation. This document presents advanced CV techniques specifically designed for financial data.

---

## The Failure of Standard K-Fold CV

Standard k-fold cross-validation assumes:
1. **IID data:** Observations are independent
2. **No leakage:** Training and test sets don't overlap

Both assumptions are violated in financial data with overlapping labels.

---

## Purged K-Fold Cross-Validation

### Algorithm

1. Split data into k folds
2. For each fold as test set:
   - **Purge:** Remove training observations that overlap with test set
   - **Embargo:** Remove training observations immediately after test set
   - Train model on purged training set
   - Evaluate on test set

### Implementation

```python
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

class PurgedKFold:
    def __init__(self, n_splits=5, samples_info_sets=None, pct_embargo=0.01):
        self.n_splits = n_splits
        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo
    
    def split(self, X, y=None, groups=None):
        indices = np.arange(X.shape[0])
        embargo_size = int(X.shape[0] * self.pct_embargo)
        
        test_ranges = [
            (i * len(indices) // self.n_splits, 
             (i + 1) * len(indices) // self.n_splits)
            for i in range(self.n_splits)
        ]
        
        for start, end in test_ranges:
            test_indices = indices[start:end]
            
            # Purge training indices that overlap with test
            if self.samples_info_sets is not None:
                train_indices = self._purge_train_indices(
                    indices, test_indices
                )
            else:
                train_indices = np.concatenate([
                    indices[:start],
                    indices[end + embargo_size:]
                ])
            
            yield train_indices, test_indices
    
    def _purge_train_indices(self, all_indices, test_indices):
        # Remove training samples that overlap with test samples
        train_indices = []
        
        for idx in all_indices:
            if idx in test_indices:
                continue
            
            # Check if this training sample overlaps with any test sample
            overlaps = False
            for test_idx in test_indices:
                if self._check_overlap(idx, test_idx):
                    overlaps = True
                    break
            
            if not overlaps:
                train_indices.append(idx)
        
        return np.array(train_indices)
    
    def _check_overlap(self, idx1, idx2):
        # Check if two samples overlap based on their time spans
        if self.samples_info_sets is None:
            return False
        
        t1_start, t1_end = self.samples_info_sets[idx1]
        t2_start, t2_end = self.samples_info_sets[idx2]
        
        return (t1_start <= t2_end) and (t2_start <= t1_end)
```

---

## Combinatorial Purged Cross-Validation (CPCV)

CPCV tests all possible combinations of training/test splits.

### Benefits
- More robust performance estimates
- Better for small datasets
- Reduces variance of CV score

### Implementation

```python
from itertools import combinations

def combinatorial_purged_cv(X, y, n_splits=5, n_test_groups=2):
    """
    Perform combinatorial purged cross-validation
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Labels
    n_splits : int
        Number of groups to split data into
    n_test_groups : int
        Number of groups to use as test set
    
    Returns:
    --------
    list : CV scores for each combination
    """
    # Split indices into groups
    indices = np.arange(len(X))
    groups = np.array_split(indices, n_splits)
    
    # Generate all combinations of test groups
    test_combinations = list(combinations(range(n_splits), n_test_groups))
    
    scores = []
    
    for test_group_ids in test_combinations:
        # Get test indices
        test_indices = np.concatenate([groups[i] for i in test_group_ids])
        
        # Get train indices (all other groups)
        train_group_ids = [i for i in range(n_splits) if i not in test_group_ids]
        train_indices = np.concatenate([groups[i] for i in train_group_ids])
        
        # Train and evaluate
        # ... (model training code)
        
        scores.append(score)
    
    return scores
```

---

## Mean Decrease Accuracy (MDA)

MDA measures feature importance by permuting feature values and measuring the decrease in model accuracy.

### Implementation

```python
def mean_decrease_accuracy(model, X_test, y_test, n_repeats=5):
    """
    Calculate feature importance using MDA
    
    Parameters:
    -----------
    model : trained model
        Model with predict method
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    n_repeats : int
        Number of permutation repeats
    
    Returns:
    --------
    pd.Series : Feature importance scores
    """
    # Baseline accuracy
    baseline_score = model.score(X_test, y_test)
    
    importance = pd.Series(index=X_test.columns, dtype=float)
    
    for col in X_test.columns:
        scores = []
        
        for _ in range(n_repeats):
            # Permute this feature
            X_permuted = X_test.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col].values)
            
            # Calculate score with permuted feature
            permuted_score = model.score(X_permuted, y_test)
            scores.append(permuted_score)
        
        # Importance is decrease in accuracy
        importance[col] = baseline_score - np.mean(scores)
    
    return importance.sort_values(ascending=False)
```

---

## Single Feature Importance (SFI)

SFI trains a separate model for each feature individually.

### Implementation

```python
def single_feature_importance(X_train, y_train, X_test, y_test, model_class):
    """
    Calculate SFI for each feature
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Train and test features
    y_train, y_test : pd.Series
        Train and test labels
    model_class : class
        Model class to instantiate
    
    Returns:
    --------
    pd.Series : Feature importance scores
    """
    importance = pd.Series(index=X_train.columns, dtype=float)
    
    for col in X_train.columns:
        # Train model on single feature
        model = model_class()
        model.fit(X_train[[col]], y_train)
        
        # Evaluate
        score = model.score(X_test[[col]], y_test)
        importance[col] = score
    
    return importance.sort_values(ascending=False)
```

---

## Clustered Feature Importance

Group correlated features and calculate importance at cluster level.

### Implementation

```python
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def clustered_feature_importance(X, y, model, n_clusters=10):
    """
    Calculate feature importance at cluster level
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Labels
    model : trained model
    n_clusters : int
        Number of feature clusters
    
    Returns:
    --------
    dict : Cluster importance and member features
    """
    # Calculate feature correlation
    corr_matrix = X.corr().abs()
    
    # Convert to distance matrix
    dist_matrix = 1 - corr_matrix
    
    # Hierarchical clustering
    linkage_matrix = linkage(squareform(dist_matrix), method='ward')
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # Calculate importance for each cluster
    cluster_importance = {}
    
    for cluster_id in range(1, n_clusters + 1):
        cluster_features = X.columns[clusters == cluster_id].tolist()
        
        # Calculate MDA for this cluster
        importance = mean_decrease_accuracy(
            model, X[cluster_features], y
        ).mean()
        
        cluster_importance[cluster_id] = {
            'importance': importance,
            'features': cluster_features
        }
    
    return cluster_importance
```

---

## Best Practices

1. **Always use purged k-fold CV** for financial data
2. **Apply embargo** to account for serial correlation
3. **Use MDA over MDI** for feature importance
4. **Cluster correlated features** before importance calculation
5. **Validate on multiple CV schemes** for robustness

---

## References

1. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapters 7-8.
2. Jansen, S. (2020). *Machine Learning for Algorithmic Trading*. Packt Publishing.
