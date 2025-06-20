"""
Strategy factory for creating strategy instances.
"""

from typing import Dict, Type
from .base_strategy import BaseStrategy
from .cross_asset_momentum import CrossAssetMomentumStrategy
from .multi_asset_momentum import MultiAssetMomentumStrategy
from .genetic_algorithm import GeneticAlgorithmStrategy
from .vol_adaptive_momentum import VolAdaptiveMomentumStrategy


class StrategyFactory:
    """Factory for creating strategy instances."""
    
    _strategies: Dict[str, Type[BaseStrategy]] = {
        'cross_asset_momentum': CrossAssetMomentumStrategy,
        'multi_asset_momentum': MultiAssetMomentumStrategy,
        'genetic_algorithm': GeneticAlgorithmStrategy,
        'vol_adaptive_momentum': VolAdaptiveMomentumStrategy,
    }
    
    @classmethod
    def create_strategy(cls, config: Dict) -> BaseStrategy:
        """
        Create a strategy instance from configuration.
        
        Args:
            config: Strategy configuration dictionary
            
        Returns:
            Strategy instance
            
        Raises:
            ValueError: If strategy type is not registered
        """
        strategy_type = config.get('type', config.get('name', '')).lower()
        
        if strategy_type not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(f"Unknown strategy type: {strategy_type}. Available: {available}")
        
        strategy_class = cls._strategies[strategy_type]
        return strategy_class(config)
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[BaseStrategy]):
        """Register a new strategy type."""
        cls._strategies[name] = strategy_class
    
    @classmethod
    def get_available_strategies(cls) -> Dict[str, Type[BaseStrategy]]:
        """Get all available strategy types."""
        return cls._strategies.copy()
    
    @classmethod
    def get_strategy_info(cls, strategy_type: str) -> Dict:
        """Get information about a strategy type."""
        if strategy_type not in cls._strategies:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        strategy_class = cls._strategies[strategy_type]
        
        # Create a dummy instance to get info
        dummy_config = {'type': strategy_type}
        try:
            dummy_instance = strategy_class(dummy_config)
            return {
                'name': strategy_type,
                'class': strategy_class.__name__,
                'description': strategy_class.__doc__ or 'No description available',
                'required_features': dummy_instance.get_required_features()
            }
        except Exception as e:
            return {
                'name': strategy_type,
                'class': strategy_class.__name__,
                'description': strategy_class.__doc__ or 'No description available',
                'error': f"Could not instantiate: {e}"
            } 