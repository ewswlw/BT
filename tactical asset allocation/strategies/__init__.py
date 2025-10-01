"""
Tactical Asset Allocation Strategies

This package implements institutional-grade TAA strategies from academic papers:
1. VIX Timing Strategy (2013-2025)
2. Defense First Base (2008-2025) 
3. Defense First Leveraged (2008-2025)
4. Sector Rotation (1999-2025)
"""

from .base_strategy import BaseTAAStrategy
from .vix_timing_strategy import VIXTimingStrategy
from .defense_first_base_strategy import DefenseFirstBaseStrategy
from .defense_first_levered_strategy import DefenseFirstLeveredStrategy
from .sector_rotation_strategy import SectorRotationStrategy

__all__ = [
    'BaseTAAStrategy',
    'VIXTimingStrategy',
    'DefenseFirstBaseStrategy',
    'DefenseFirstLeveredStrategy',
    'SectorRotationStrategy',
]

