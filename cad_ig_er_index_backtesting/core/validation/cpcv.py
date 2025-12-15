"""
Combinatorial Purged Cross-Validation (CPCV).

Tests all possible combinations of train/test splits for more robust
performance estimation.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Generator
from itertools import combinations
from .purged_cv import PurgedKFold


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation.
    
    Tests all possible combinations of train/test splits, providing
    more robust performance estimates with reduced variance.
    
    Based on López de Prado's "Advances in Financial Machine Learning"
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        n_test_groups: int = 2,
        samples_info_sets: Optional[List[Tuple]] = None,
        pct_embargo: float = 0.01
    ):
        """
        Initialize CombinatorialPurgedCV.
        
        Args:
            n_splits: Number of groups to split data into
            n_test_groups: Number of groups to use as test set
            samples_info_sets: List of tuples (start_time, end_time) for each sample
            pct_embargo: Percentage of data to embargo after test set
        """
        if n_test_groups >= n_splits:
            raise ValueError("n_test_groups must be less than n_splits")
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo
    
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate all combinations of train/test splits.
        
        Args:
            X: Features DataFrame
            y: Target Series (optional)
            
        Yields:
            Tuple of (train_indices, test_indices) for each combination
        """
        indices = np.arange(len(X))
        
        # Split indices into groups
        group_size = len(indices) // self.n_splits
        groups = []
        for i in range(self.n_splits):
            start_idx = i * group_size
            if i == self.n_splits - 1:
                # Last group gets remaining indices
                end_idx = len(indices)
            else:
                end_idx = (i + 1) * group_size
            groups.append(indices[start_idx:end_idx])
        
        # Generate all combinations of test groups
        test_combinations = list(combinations(range(self.n_splits), self.n_test_groups))
        
        for test_group_ids in test_combinations:
            # Get test indices from selected groups
            test_indices = np.concatenate([groups[i] for i in test_group_ids])
            
            # Get train indices from remaining groups
            train_group_ids = [i for i in range(self.n_splits) if i not in test_group_ids]
            train_indices = np.concatenate([groups[i] for i in train_group_ids])
            
            # Apply embargo
            if len(test_indices) > 0:
                max_test_idx = test_indices.max()
                embargo_size = int(len(indices) * self.pct_embargo)
                embargo_end = min(max_test_idx + embargo_size, len(indices))
                
                # Remove indices in embargo period from training
                train_indices = train_indices[train_indices >= embargo_end]
            
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
                if train_start <= test_end and test_start <= train_end:
                    overlaps = True
                    break
            
            if not overlaps:
                purged_train.append(idx)
        
        return np.array(purged_train)
    
    def get_n_splits(self) -> int:
        """
        Return the number of splitting iterations.
        
        This is the number of combinations: C(n_splits, n_test_groups)
        """
        from math import comb
        return comb(self.n_splits, self.n_test_groups)

