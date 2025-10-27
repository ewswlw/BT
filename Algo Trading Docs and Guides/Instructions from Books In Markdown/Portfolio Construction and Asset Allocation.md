# Portfolio Construction and Asset Allocation

**Modern Portfolio Optimization Using Machine Learning**

---

## Introduction

Traditional mean-variance optimization suffers from instability and concentration. This document presents modern approaches, particularly Hierarchical Risk Parity (HRP).

---

## Markowitz's Curse

Mean-variance optimization has fundamental problems:

1. **Ill-conditioned covariance matrix:** Small changes in inputs lead to large changes in outputs
2. **Concentration:** Tends to allocate large weights to few assets
3. **Estimation error:** Requires accurate estimates of means and covariances
4. **Instability:** Optimal weights change dramatically over time

---

## Hierarchical Risk Parity (HRP)

HRP is López de Prado's solution to the problems of mean-variance optimization.

### Three-Step Algorithm

1. **Tree Clustering:** Group similar assets hierarchically
2. **Quasi-Diagonalization:** Reorder covariance matrix
3. **Recursive Bisection:** Allocate weights top-down

### Complete Implementation

```python
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

class HierarchicalRiskParity:
    def __init__(self, returns):
        """
        Initialize HRP with return data
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns (rows = time, columns = assets)
        """
        self.returns = returns
        self.cov = returns.cov()
        self.corr = returns.corr()
    
    def get_quasi_diag(self, link):
        """
        Reorder covariance matrix based on hierarchical clustering
        
        Parameters:
        -----------
        link : np.array
            Linkage matrix from hierarchical clustering
        
        Returns:
        --------
        list : Sorted asset indices
        """
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        
        return sort_ix.tolist()
    
    def get_cluster_var(self, cov, c_items):
        """
        Calculate variance of a cluster
        
        Parameters:
        -----------
        cov : pd.DataFrame
            Covariance matrix
        c_items : list
            Assets in cluster
        
        Returns:
        --------
        float : Cluster variance
        """
        cov_slice = cov.loc[c_items, c_items]
        w = self.get_ivp(cov_slice)
        c_var = np.dot(np.dot(w.T, cov_slice), w)
        return c_var
    
    def get_ivp(self, cov):
        """
        Get inverse-variance portfolio weights
        
        Parameters:
        -----------
        cov : pd.DataFrame
            Covariance matrix
        
        Returns:
        --------
        np.array : Portfolio weights
        """
        ivp = 1.0 / np.diag(cov)
        ivp /= ivp.sum()
        return ivp
    
    def get_rec_bipart(self, cov, sort_ix):
        """
        Recursive bisection to allocate weights
        
        Parameters:
        -----------
        cov : pd.DataFrame
            Covariance matrix
        sort_ix : list
            Sorted asset indices
        
        Returns:
        --------
        pd.Series : Asset weights
        """
        w = pd.Series(1, index=sort_ix)
        c_items = [sort_ix]
        
        while len(c_items) > 0:
            c_items = [i[j:k] for i in c_items 
                      for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) 
                      if len(i) > 1]
            
            for i in range(0, len(c_items), 2):
                c_items0 = c_items[i]
                c_items1 = c_items[i + 1]
                
                c_var0 = self.get_cluster_var(cov, c_items0)
                c_var1 = self.get_cluster_var(cov, c_items1)
                
                alpha = 1 - c_var0 / (c_var0 + c_var1)
                
                w[c_items0] *= alpha
                w[c_items1] *= 1 - alpha
        
        return w
    
    def optimize(self):
        """
        Run HRP optimization
        
        Returns:
        --------
        pd.Series : Optimal weights
        """
        # Step 1: Tree clustering
        dist = ((1 - self.corr) / 2.0) ** 0.5
        link = linkage(squareform(dist.values), method='single')
        
        # Step 2: Quasi-diagonalization
        sort_ix = self.get_quasi_diag(link)
        sort_ix = self.corr.index[sort_ix].tolist()
        
        # Step 3: Recursive bisection
        weights = self.get_rec_bipart(self.cov, sort_ix)
        
        return weights

# Example usage
returns = pd.DataFrame(
    np.random.randn(1000, 10),
    columns=[f'Asset_{i}' for i in range(10)]
) * 0.02

hrp = HierarchicalRiskParity(returns)
weights = hrp.optimize()

print("HRP Weights:")
print(weights.sort_values(ascending=False))
```

---

## Comparison with Mean-Variance Optimization

```python
from scipy.optimize import minimize

def mean_variance_optimization(returns, target_return=None):
    """
    Traditional mean-variance optimization
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns
    target_return : float
        Target portfolio return (optional)
    
    Returns:
    --------
    np.array : Optimal weights
    """
    n_assets = returns.shape[1]
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # Objective: minimize variance
    def objective(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
    ]
    
    if target_return is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.dot(w, mean_returns) - target_return
        })
    
    # Bounds
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess
    x0 = np.array([1.0 / n_assets] * n_assets)
    
    # Optimize
    result = minimize(
        objective, x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x
```

---

## ML-Based Asset Allocation

Use ML to predict returns, then optimize.

```python
def ml_asset_allocation(features, returns, model):
    """
    ML-based asset allocation
    
    Parameters:
    -----------
    features : pd.DataFrame
        Features for prediction
    returns : pd.DataFrame
        Historical returns
    model : trained model
        Model to predict returns
    
    Returns:
    --------
    pd.Series : Optimal weights
    """
    # Predict expected returns
    predicted_returns = model.predict(features)
    
    # Use HRP with predicted returns
    # (Modify HRP to accept expected returns)
    
    # Or use mean-variance with predicted returns
    weights = mean_variance_optimization(
        returns,
        target_return=predicted_returns.mean()
    )
    
    return pd.Series(weights, index=returns.columns)
```

---

## Best Practices

1. **Use HRP as default** for most applications
2. **Avoid mean-variance optimization** unless you have very accurate estimates
3. **Rebalance periodically** but not too frequently
4. **Consider transaction costs** in optimization
5. **Monitor concentration** to ensure diversification

---

## References

1. López de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out of Sample." *Journal of Portfolio Management*.
2. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapter 16.
