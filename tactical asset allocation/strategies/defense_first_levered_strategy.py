"""
Defense First Leveraged Strategy

Paper: "A Simple and Effective Tactical Allocation Strategy" (QuantSeeker, 2025)
Period: 2008-11-05 to present (SPXL inception)
Rebalance: Monthly (end-of-month)

Strategy Logic:
- Identical to Defense First Base
- BUT uses SPXL (3x leveraged SPY) instead of SPY as fallback
- Higher returns, similar volatility to SPY
"""

from typing import Dict
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from .defense_first_base_strategy import DefenseFirstBaseStrategy


class DefenseFirstLeveredStrategy(DefenseFirstBaseStrategy):
    """
    Defense First tactical allocation with SPXL (3x leveraged) fallback.
    
    Identical logic to base strategy but substitutes SPXL for SPY.
    This amplifies returns when equity exposure is warranted.
    
    Attributes:
        Inherits all from DefenseFirstBaseStrategy
        fallback_asset: 'SPXL' (overridden from 'SPY')
    """
    
    def __init__(
        self,
        start_date: str = '2008-11-05',  # SPXL inception date
        end_date: str = '2025-07-31',  # Matches paper end date
        **kwargs
    ):
        """
        Initialize Defense First Leveraged Strategy.
        
        Args:
            start_date: Strategy start date (SPXL inception: Nov 5, 2008)
            end_date: Strategy end date
            **kwargs: Additional arguments passed to parent class
        """
        # Initialize parent class
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )
        
        # Override name and fallback asset
        self.name = 'DEFENSE_FIRST_LEVERED'
        self.fallback_asset = 'SPXL'  # Key difference from base strategy
    
    def get_strategy_info(self) -> Dict:
        """
        Return strategy metadata and rules.
        
        Returns:
            Dictionary with strategy information
        """
        info = super().get_strategy_info()
        
        # Update metadata for leveraged version
        info.update({
            'name': 'Defense First - Leveraged (SPXL)',
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'fallback_asset': 'SPXL',
            'expected_performance': {
                'CAGR': '~21%',
                'Sharpe': 0.99,
                'Max_DD': 'Not specified in paper',
                'Volatility': 'Similar to SPY',
                '2025_YTD': '~23%'
            },
            'rules': self._get_trading_rules()
        })
        
        return info
    
    def _get_trading_rules(self) -> str:
        """Get detailed trading rules as formatted string."""
        rules = """
DEFENSE FIRST LEVERAGED STRATEGY - SPXL FALLBACK

ASSET UNIVERSE:
- Defensive (dynamic): TLT, GLD, DBC, UUP (+ BTAL from May 2011)
- Fallback: SPXL (Direxion Daily S&P 500 Bull 3X Shares)

MONTHLY REBALANCING LOGIC (T-1 signals):

★ IDENTICAL TO BASE STRATEGY, BUT WITH SPXL INSTEAD OF SPY ★

1. Calculate Momentum Scores:
   For each defensive asset:
   - Lookback periods: 21, 63, 126, 252 days (~1, 3, 6, 12 months)
   - For each lookback:
       return = (price[t] / price[t-lookback] - 1)
       annualized_return = return * (252 / lookback)
   - Momentum score = average of annualized returns

2. Rank Assets:
   - Sort defensive assets by momentum score (highest to lowest)
   
3. Assign Fixed Weights:
   - Rank 1 (highest momentum): 40%
   - Rank 2: 30%
   - Rank 3: 20%
   - Rank 4: 10%
   - Rank 5+: 0% (only top 4 get weight)

4. Apply Momentum Filter:
   - Get current 3-month T-bill rate
   - For each asset:
       IF momentum_score < T-bill rate:
           Reallocate its weight to SPXL  ← KEY DIFFERENCE
       ELSE:
           Keep defensive allocation

5. Final Portfolio:
   - Defensive allocations (if momentum > T-bill)
   - SPXL allocation = sum of reallocated weights  ← KEY DIFFERENCE
   - Total always sums to 100%

KEY ADVANTAGE:
- When market conditions favor equities (defensive momentum weak),
  the strategy automatically increases equity exposure through 3x leverage
- This amplifies returns during bull markets
- Defensive assets still protect during downturns

EXPECTED PERFORMANCE (Nov 2008 - 2025):
- CAGR: ~21% (vs SPY ~11%)
- Sharpe: 0.99 (vs SPY 0.79)
- Volatility: Similar to SPY
- 2025 YTD: ~23%

COMPARISON TO BASE STRATEGY:
- Base (SPY): Lower returns, lower risk
- Leveraged (SPXL): Higher returns, similar risk to SPY
- Choice depends on investor risk tolerance
"""
        
        return rules.strip()
    
    def __repr__(self):
        return f"{self.name} - SPXL Fallback ({self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')})"

