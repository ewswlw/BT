"""
Purged Cross-Validation for financial time series.

Implements purged k-fold cross-validation that prevents data leakage
from overlapping labels in financial data.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Generator
from sklearn.model_selection import BaseCrossValidator


class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold Cross-Validation for financial time series.
    
    Prevents data leakage by:
    1. Purging training samples that overlap with test samples
    2. Applying embargo period after test set
    
    Based on López de Prado's "Advances in Financial Machine Learning"
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        samples_info_sets: Optional[List[Tuple]] = None,
        pct_embargo: float = 0.01
    ):
        """
        Initialize PurgedKFold.
        
        Args:
            n_splits: Number of folds
            samples_info_sets: List of tuples (start_time, end_time) for each sample.
                               If None, assumes no overlapping labels.
            pct_embargo: Percentage of data to embargo after test set (0.0 to 1.0)
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if pct_embargo < 0 or pct_embargo > 1:
            raise ValueError("pct_embargo must be between 0 and 1")
        
        self.n_splits = n_splits
        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo
    
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test set.
        
        Args:
            X: Features DataFrame
            y: Target Series (optional)
            groups: Group labels (optional, not used)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        indices = np.arange(len(X))
        
        # Split indices into folds
        fold_size = len(indices) // self.n_splits
        test_starts = [(i * fold_size) for i in range(self.n_splits)]
        
        for test_start in test_starts:
            test_end = min(test_start + fold_size, len(indices))
            test_indices = indices[test_start:test_end]
            
            # Calculate embargo size
            embargo_size = int(len(indices) * self.pct_embargo)
            embargo_end = min(test_end + embargo_size, len(indices))
            
            # Get training indices (before test and after embargo)
            train_indices = np.concatenate([
                indices[:test_start],
                indices[embargo_end:]
            ])
            
            # Purge overlapping samples if samples_info_sets provided
            if self.samples_info_sets is not None:
                train_indices = self._purge_overlaps(
                    train_indices,
                    test_indices,
                    self.samples_info_sets
                )
            
            yield train_indices, test_indices
    
    def _purge_overlaps(
        self,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        samples_info_sets: List[Tuple]
    ) -> np.ndarray:
        """
        Remove training samples that overlap with test samples.
        
        Args:
            train_indices: Training sample indices
            test_indices: Test sample indices
            samples_info_sets: List of (start_time, end_time) tuples
            
        Returns:
            Purged training indices
        """
        if len(samples_info_sets) != len(train_indices) + len(test_indices):
            # If lengths don't match, return original train_indices
            return train_indices
        
        # Get test sample time ranges
        test_ranges = []
        for idx in test_indices:
            if idx < len(samples_info_sets):
                test_ranges.append(samples_info_sets[idx])
        
        # Purge training samples that overlap with test ranges
        purged_train = []
        for idx in train_indices:
            if idx >= len(samples_info_sets):
                continue
            
            train_start, train_end = samples_info_sets[idx]
            overlaps = False
            
            for test_start, test_end in test_ranges:
                # Check if ranges overlap
                if train_start <= test_end and test_start <= train_end:
                    overlaps = True
                    break
            
            if not overlaps:
                purged_train.append(idx)
        
        return np.array(purged_train)
    
    def get_n_splits(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Return the number of splitting iterations."""
        return self.n_splits

