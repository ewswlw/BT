"""
Container for validation results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


@dataclass
class ValidationResults:
    """Container for all validation results."""
    
    # Strategy identification
    strategy_name: str = ""
    
    # Cross-validation results
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    cv_method: str = ""
    
    # Sample weights
    sample_weights: Optional[pd.Series] = None
    label_uniqueness: Optional[pd.Series] = None
    overlapping_labels_count: Optional[int] = None
    
    # Deflated metrics
    deflated_sharpe: Optional[float] = None
    probabilistic_sharpe: Optional[float] = None
    n_trials: Optional[int] = None
    estimated_sharpe: Optional[float] = None
    
    # Walk-forward
    walk_forward_results: Optional[List[Dict]] = None
    
    # PBO
    pbo: Optional[float] = None
    is_sharpe: Optional[float] = None
    oos_sharpe: Optional[float] = None
    
    # Robustness
    synthetic_data_results: Optional[Dict] = None
    
    # Minimum backtest length
    min_backtest_length: Optional[int] = None
    current_length: Optional[int] = None
    
    # Data quality
    data_quality_metrics: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert results to dictionary."""
        result = {
            'cv_scores': self.cv_scores,
            'cv_mean': self.cv_mean,
            'cv_std': self.cv_std,
            'cv_method': self.cv_method,
            'deflated_sharpe': self.deflated_sharpe,
            'probabilistic_sharpe': self.probabilistic_sharpe,
            'n_trials': self.n_trials,
            'estimated_sharpe': self.estimated_sharpe,
            'pbo': self.pbo,
            'is_sharpe': self.is_sharpe,
            'oos_sharpe': self.oos_sharpe,
            'min_backtest_length': self.min_backtest_length,
            'current_length': self.current_length,
            'overlapping_labels_count': self.overlapping_labels_count,
        }
        
        if self.sample_weights is not None:
            result['sample_weights_mean'] = float(self.sample_weights.mean())
            result['sample_weights_std'] = float(self.sample_weights.std())
        
        if self.label_uniqueness is not None:
            result['label_uniqueness_mean'] = float(self.label_uniqueness.mean())
            result['label_uniqueness_min'] = float(self.label_uniqueness.min())
        
        if self.data_quality_metrics:
            result['data_quality'] = self.data_quality_metrics
        
        return result

