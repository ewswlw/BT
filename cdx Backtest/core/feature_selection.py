"""
Feature selection to identify most predictive features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    """Feature selection for trading strategy."""
    
    def __init__(self, method: str = 'combined', top_k: int = 100):
        """
        Initialize feature selector.
        
        Args:
            method: Selection method ('mutual_info', 'f_test', 'rfe', 'rf_importance', 'combined')
            top_k: Number of top features to select
        """
        self.method = method
        self.top_k = top_k
        self.selected_features = []
        self.selector = None
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: Optional[str] = None) -> pd.DataFrame:
        """
        Select top features using specified method.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Override default method
            
        Returns:
            DataFrame with selected features
        """
        method = method or self.method
        
        # Remove infinite and NaN values
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Remove constant features
        constant_features = X_clean.columns[X_clean.nunique() <= 1].tolist()
        X_clean = X_clean.drop(columns=constant_features)
        
        # Align X and y
        common_index = X_clean.index.intersection(y.index)
        X_aligned = X_clean.loc[common_index]
        y_aligned = y.loc[common_index]
        
        # Remove NaN from y
        valid_mask = ~y_aligned.isnull()
        X_aligned = X_aligned[valid_mask]
        y_aligned = y_aligned[valid_mask]
        
        if len(X_aligned) < 100:
            return X_clean  # Return all features if not enough data
        
        # Select method
        if method == 'mutual_info':
            selected_features = self._mutual_info_selection(X_aligned, y_aligned)
        elif method == 'f_test':
            selected_features = self._f_test_selection(X_aligned, y_aligned)
        elif method == 'rfe':
            selected_features = self._rfe_selection(X_aligned, y_aligned)
        elif method == 'rf_importance':
            selected_features = self._rf_importance_selection(X_aligned, y_aligned)
        elif method == 'combined':
            selected_features = self._combined_selection(X_aligned, y_aligned)
        else:
            return X_clean  # Return all if unknown method
        
        # Return selected features
        self.selected_features = selected_features
        return X_clean[selected_features]
    
    def _mutual_info_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using mutual information."""
        try:
            selector = SelectKBest(score_func=mutual_info_classif, k=min(self.top_k, len(X.columns)))
            X_scaled = StandardScaler().fit_transform(X)
            selector.fit(X_scaled, y)
            
            feature_scores = pd.Series(selector.scores_, index=X.columns)
            selected = feature_scores.nlargest(min(self.top_k, len(feature_scores))).index.tolist()
            return selected
        except:
            return X.columns.tolist()[:self.top_k]
    
    def _f_test_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using F-test."""
        try:
            selector = SelectKBest(score_func=f_classif, k=min(self.top_k, len(X.columns)))
            X_scaled = StandardScaler().fit_transform(X)
            selector.fit(X_scaled, y)
            
            feature_scores = pd.Series(selector.scores_, index=X.columns)
            # Handle NaN scores
            feature_scores = feature_scores.fillna(0)
            selected = feature_scores.nlargest(min(self.top_k, len(feature_scores))).index.tolist()
            return selected
        except:
            return X.columns.tolist()[:self.top_k]
    
    def _rfe_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using Recursive Feature Elimination."""
        try:
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            selector = RFE(estimator, n_features_to_select=min(self.top_k, len(X.columns)), step=10)
            X_scaled = StandardScaler().fit_transform(X)
            selector.fit(X_scaled, y)
            
            selected = X.columns[selector.support_].tolist()
            return selected
        except:
            return X.columns.tolist()[:self.top_k]
    
    def _rf_importance_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using Random Forest importance."""
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            X_scaled = StandardScaler().fit_transform(X)
            rf.fit(X_scaled, y)
            
            feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
            selected = feature_importance.nlargest(min(self.top_k, len(feature_importance))).index.tolist()
            return selected
        except:
            return X.columns.tolist()[:self.top_k]
    
    def _combined_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Combine multiple selection methods."""
        try:
            # Get features from multiple methods
            mi_features = set(self._mutual_info_selection(X, y))
            f_test_features = set(self._f_test_selection(X, y))
            rf_features = set(self._rf_importance_selection(X, y))
            
            # Combine: features that appear in at least 2 methods get priority
            all_features = mi_features | f_test_features | rf_features
            priority_features = (mi_features & f_test_features) | (mi_features & rf_features) | (f_test_features & rf_features)
            
            # Start with priority features, then add others
            selected = list(priority_features)
            remaining = list(all_features - priority_features)
            
            # Fill up to top_k
            while len(selected) < self.top_k and remaining:
                selected.append(remaining.pop(0))
            
            # If still not enough, add from all features
            if len(selected) < self.top_k:
                all_remaining = set(X.columns) - set(selected)
                selected.extend(list(all_remaining)[:self.top_k - len(selected)])
            
            return selected[:self.top_k]
        except:
            return X.columns.tolist()[:self.top_k]
    
    def get_selected_features(self) -> List[str]:
        """Return list of selected feature names."""
        return self.selected_features

