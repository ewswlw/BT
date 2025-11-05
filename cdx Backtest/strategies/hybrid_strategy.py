"""
Hybrid strategy combining ML and rule-based approaches.
"""

from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base_strategy import BaseStrategy
from .ml_strategy import MLStrategy
from .rule_based_strategy import RuleBasedStrategy


class HybridStrategy(BaseStrategy):
    """
    Hybrid strategy combining ML predictions with rule-based filters.
    
    Uses ML for entry signals and rule-based filters for confirmation/filtering.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Initialize ML and rule-based components
        ml_config = config.get('ml_config', {})
        ml_config['trading_asset'] = config.get('trading_asset', 'us_ig_cdx_er_index')
        ml_config['holding_period_days'] = config.get('holding_period_days', 7)
        ml_config['name'] = f"{config.get('name', 'Hybrid')}_ML"
        
        rule_config = config.get('rule_config', {})
        rule_config['trading_asset'] = config.get('trading_asset', 'us_ig_cdx_er_index')
        rule_config['holding_period_days'] = config.get('holding_period_days', 7)
        rule_config['strategy_type'] = config.get('rule_strategy_type', 'combination')
        rule_config['name'] = f"{config.get('name', 'Hybrid')}_Rule"
        
        self.ml_strategy = MLStrategy(ml_config)
        self.rule_strategy = RuleBasedStrategy(rule_config)
        
        # Hybrid parameters
        self.ml_weight = config.get('ml_weight', 0.7)
        self.rule_weight = config.get('rule_weight', 0.3)
        self.use_ml_for_entry = config.get('use_ml_for_entry', True)
        self.use_rules_for_filter = config.get('use_rules_for_filter', True)
        self.require_both_signals = config.get('require_both_signals', False)
        
        # Normalize weights
        total_weight = self.ml_weight + self.rule_weight
        if total_weight > 0:
            self.ml_weight = self.ml_weight / total_weight
            self.rule_weight = self.rule_weight / total_weight
    
    def get_required_features(self) -> List[str]:
        """Return required features from both strategies."""
        ml_features = self.ml_strategy.get_required_features()
        rule_features = self.rule_strategy.get_required_features()
        return list(set(ml_features + rule_features))
    
    def generate_signals(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                        train_features: pd.DataFrame, test_features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate trading signals using hybrid approach.
        
        Args:
            train_data: Training price data
            test_data: Test price data
            train_features: Training features
            test_features: Test features
            
        Returns:
            Tuple of (entry_signals, exit_signals) for test_data
        """
        print(f"\n=== Hybrid Strategy: {self.name} ===")
        print(f"ML weight: {self.ml_weight:.2f}, Rule weight: {self.rule_weight:.2f}")
        
        # Generate ML signals
        print("Generating ML signals...")
        ml_entry_signals, ml_exit_signals = self.ml_strategy.generate_signals(
            train_data, test_data, train_features, test_features
        )
        
        # Generate rule-based signals
        print("Generating rule-based signals...")
        rule_entry_signals, rule_exit_signals = self.rule_strategy.generate_signals(
            train_data, test_data, train_features, test_features
        )
        
        # Combine signals based on strategy
        if self.use_ml_for_entry and self.use_rules_for_filter:
            # Use ML for entry, filter with rules
            if self.require_both_signals:
                # Require both ML and rule signals
                combined_entry = ml_entry_signals & rule_entry_signals
            else:
                # Use ML signals, but filter out if rules disagree strongly
                combined_entry = ml_entry_signals & rule_entry_signals
        elif self.use_ml_for_entry:
            # Use only ML signals
            combined_entry = ml_entry_signals
        else:
            # Use only rule signals
            combined_entry = rule_entry_signals
        
        # For exits, use weighted combination or ML exits
        if self.use_ml_for_entry:
            combined_exit = ml_exit_signals
        else:
            combined_exit = rule_exit_signals
        
        # Apply holding period
        print(f"Applying {self.holding_period_days}-day holding period...")
        raw_signals = combined_entry.astype(int)
        adjusted_signals = self.apply_holding_period(raw_signals, self.holding_period_days)
        
        # Convert to entry/exit signals
        positions = adjusted_signals.astype(int)
        positions_shifted = positions.shift(1).fillna(0).astype(int)
        
        final_entry_signals = (positions == 1) & (positions_shifted == 0)
        final_exit_signals = (positions == 0) & (positions_shifted == 1)
        
        print(f"  Entry signals: {final_entry_signals.sum()}")
        print(f"  Exit signals: {final_exit_signals.sum()}")
        print(f"  Time in market: {positions.mean():.2%}")
        
        return final_entry_signals, final_exit_signals

