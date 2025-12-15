"""
Sample weighting for overlapping labels in financial time series.

Handles non-IID data by calculating label uniqueness and applying
appropriate sample weights.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class LabelInfo:
    """Information about a label."""
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    index: int


class LabelUniquenessCalculator:
    """
    Calculate label uniqueness for overlapping labels.
    
    Based on López de Prado's methodology for handling non-IID data
    in financial machine learning.
    """
    
    def __init__(self):
        """Initialize the calculator."""
        pass
    
    def compute_uniqueness(
        self,
        label_times: List[Tuple[pd.Timestamp, pd.Timestamp]],
        price_index: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Compute average uniqueness for each label.
        
        Args:
            label_times: List of (start_time, end_time) tuples for each label
            price_index: Full price series index
            
        Returns:
            Series of average uniqueness values (indexed by label position)
        """
        n_labels = len(label_times)
        uniqueness = pd.Series(index=range(n_labels), dtype=float)
        
        # First, compute concurrent labels at each bar
        concurrent = pd.Series(0, index=price_index)
        
        for start_time, end_time in label_times:
            mask = (price_index >= start_time) & (price_index <= end_time)
            concurrent[mask] += 1
        
        # Now compute average uniqueness for each label
        for i, (start_time, end_time) in enumerate(label_times):
            # Get times during this label's lifespan
            mask = (price_index >= start_time) & (price_index <= end_time)
            label_times_subset = price_index[mask]
            
            if len(label_times_subset) == 0:
                uniqueness.iloc[i] = 0.0
                continue
            
            # Compute uniqueness at each time: 1 / (number of concurrent labels)
            u_t = 1.0 / concurrent[label_times_subset]
            
            # Average uniqueness
            uniqueness.iloc[i] = u_t.mean()
        
        return uniqueness


def compute_label_uniqueness(
    label_times: List[Tuple[pd.Timestamp, pd.Timestamp]],
    price_index: pd.DatetimeIndex
) -> pd.Series:
    """
    Compute average uniqueness for each label.
    
    Convenience function wrapper around LabelUniquenessCalculator.
    
    Args:
        label_times: List of (start_time, end_time) tuples
        price_index: Full price series index
        
    Returns:
        Series of average uniqueness values
    """
    calculator = LabelUniquenessCalculator()
    return calculator.compute_uniqueness(label_times, price_index)


def compute_sample_weights(
    uniqueness: pd.Series,
    normalize: bool = True
) -> pd.Series:
    """
    Compute sample weights from uniqueness.
    
    Args:
        uniqueness: Average uniqueness for each observation
        normalize: If True, normalize weights to sum to number of samples
        
    Returns:
        Series of sample weights
    """
    # Weight proportional to uniqueness
    weights = uniqueness.copy()
    
    # Avoid division by zero
    if weights.sum() == 0:
        weights = pd.Series(1.0, index=weights.index)
    else:
        weights = weights / weights.sum()
    
    # Normalize to sum to number of samples
    if normalize:
        weights = weights * len(weights)
    
    return weights


def compute_time_decay_weights(
    label_times: List[Tuple[pd.Timestamp, pd.Timestamp]],
    uniqueness: pd.Series,
    decay_factor: float = 0.01
) -> pd.Series:
    """
    Compute weights with time decay.
    
    More recent observations get higher weights.
    
    Args:
        label_times: List of (start_time, end_time) tuples
        uniqueness: Average uniqueness for each label
        decay_factor: Exponential decay rate
        
    Returns:
        Series of time-decayed weights
    """
    # Get time since most recent observation (in days)
    if len(label_times) == 0:
        return pd.Series(dtype=float)
    
    most_recent = max(end_time for _, end_time in label_times)
    
    time_diff = pd.Series(index=range(len(label_times)), dtype=float)
    for i, (start_time, _) in enumerate(label_times):
        time_diff.iloc[i] = (most_recent - start_time).total_seconds() / 86400
    
    # Compute decay
    decay = np.exp(-decay_factor * time_diff.values)
    
    # Combine with uniqueness
    weights = uniqueness.values * decay
    
    # Normalize
    if weights.sum() > 0:
        weights = weights / weights.sum() * len(weights)
    else:
        weights = np.ones_like(weights)
    
    return pd.Series(weights, index=uniqueness.index)


def sequential_bootstrap(
    label_times: List[Tuple[pd.Timestamp, pd.Timestamp]],
    uniqueness: pd.Series,
    sample_size: Optional[int] = None
) -> List[int]:
    """
    Perform sequential bootstrap sampling.
    
    Ensures sampled observations have, on average, the same uniqueness
    as the original dataset.
    
    Args:
        label_times: List of (start_time, end_time) tuples
        uniqueness: Average uniqueness for each observation
        sample_size: Number of samples to draw (default: len(label_times))
        
    Returns:
        List of sampled indices
    """
    if sample_size is None:
        sample_size = len(label_times)
    
    if sample_size == 0:
        return []
    
    # Initialize
    sampled_indices = []
    remaining_uniqueness = uniqueness.copy()
    
    for _ in range(sample_size):
        # Compute sampling probabilities (proportional to uniqueness)
        if remaining_uniqueness.sum() == 0:
            # If all uniqueness is zero, sample uniformly
            probs = np.ones(len(remaining_uniqueness)) / len(remaining_uniqueness)
        else:
            probs = remaining_uniqueness / remaining_uniqueness.sum()
        
        # Sample one observation
        sampled_idx = np.random.choice(
            remaining_uniqueness.index,
            p=probs.values
        )
        sampled_indices.append(int(sampled_idx))
        
        # Reduce uniqueness of overlapping observations
        sampled_t0, sampled_t1 = label_times[sampled_idx]
        
        # Find overlapping observations
        for idx in remaining_uniqueness.index:
            if idx == sampled_idx:
                continue
            
            t0, t1 = label_times[idx]
            
            # Check if overlaps
            if t0 <= sampled_t1 and sampled_t0 <= t1:
                # Calculate overlap duration
                overlap_start = max(t0, sampled_t0)
                overlap_end = min(t1, sampled_t1)
                overlap_duration = (overlap_end - overlap_start).total_seconds()
                label_duration = (t1 - t0).total_seconds()
                
                if label_duration > 0:
                    reduction = overlap_duration / label_duration
                    remaining_uniqueness.iloc[idx] *= (1 - reduction)
        
        # Set sampled observation uniqueness to zero
        remaining_uniqueness.iloc[sampled_idx] = 0
    
    return sampled_indices


def count_overlapping_labels(
    label_times: List[Tuple[pd.Timestamp, pd.Timestamp]]
) -> int:
    """
    Count number of overlapping label pairs.
    
    Args:
        label_times: List of (start_time, end_time) tuples
        
    Returns:
        Number of overlapping pairs
    """
    count = 0
    n = len(label_times)
    
    for i in range(n):
        for j in range(i + 1, n):
            t0_i, t1_i = label_times[i]
            t0_j, t1_j = label_times[j]
            
            # Check if overlaps
            if t0_i <= t1_j and t0_j <= t1_i:
                count += 1
    
    return count

