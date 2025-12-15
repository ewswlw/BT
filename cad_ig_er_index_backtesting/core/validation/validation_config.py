"""
Configuration for validation framework.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class ValidationConfig:
    """Configuration for validation framework."""
    
    # Cross-validation settings
    cv_method: Literal["purged_kfold", "cpcv", "walk_forward"] = "purged_kfold"
    n_splits: int = 5
    embargo_pct: float = 0.01  # 1% embargo period
    
    # CPCV settings
    n_test_groups: int = 2  # Number of groups for test set in CPCV
    
    # Sample weights
    use_sample_weights: bool = True
    weight_method: Literal["uniqueness", "time_decay", "sequential_bootstrap"] = "uniqueness"
    time_decay_factor: float = 0.01
    
    # Deflated Sharpe Ratio
    n_trials: int = 100  # Number of strategies tested (for deflated SR)
    
    # PBO settings
    calculate_pbo: bool = True
    n_configurations: int = 50  # Number of configurations for PBO calculation
    
    # Walk-forward settings
    walk_forward_train_period: int = 252  # Days
    walk_forward_test_period: int = 63    # Days
    walk_forward_step: int = 21           # Days
    
    # Synthetic data testing (disabled by default - use actual strategy data)
    synthetic_test_enabled: bool = False
    n_simulations: int = 1000
    
    # Minimum backtest length
    min_backtest_length_enabled: bool = True
    target_sharpe: float = 1.5
    
    # Output settings
    output_dir: str = "outputs/validation"
    generate_report: bool = True
    
    # Auto-run validation during normal strategy execution
    auto_run: bool = True  # Set to false to disable automatic validation
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.embargo_pct < 0 or self.embargo_pct > 1:
            raise ValueError("embargo_pct must be between 0 and 1")
        if self.n_test_groups >= self.n_splits:
            raise ValueError("n_test_groups must be less than n_splits")
        if self.n_trials < 1:
            raise ValueError("n_trials must be at least 1")
        if self.walk_forward_train_period < 1:
            raise ValueError("walk_forward_train_period must be at least 1")
        if self.walk_forward_test_period < 1:
            raise ValueError("walk_forward_test_period must be at least 1")
        if self.n_simulations < 1:
            raise ValueError("n_simulations must be at least 1")

