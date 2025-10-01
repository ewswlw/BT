"""
Defense First Base Strategy

Paper: "A Simple and Effective Tactical Allocation Strategy" (QuantSeeker, 2025)
Period: 2008-02-01 to present
Rebalance: Monthly (end-of-month)

Strategy Logic:
- Rank 4-5 defensive assets by multi-period momentum
- Assign fixed weights: 40%, 30%, 20%, 10% (top 4 only)
- If momentum < T-bill rate → reallocate weight to SPY
- SPY allocation can range from 0% to 100%
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from .base_strategy import BaseTAAStrategy


class DefenseFirstBaseStrategy(BaseTAAStrategy):
    """
    Defense First tactical allocation with SPY fallback.
    
    Monthly rotation among defensive assets with dynamic SPY allocation
    based on momentum filter vs risk-free rate.
    
    Attributes:
        defensive_assets: List of defensive asset tickers
        fallback_asset: Ticker for fallback asset (default: 'SPY')
        lookbacks: Momentum calculation lookback periods
        fixed_weights: Weight allocation by rank [40%, 30%, 20%, 10%]
    """
    
    def __init__(
        self,
        start_date: str = '2008-02-01',
        end_date: str = '2025-07-31',  # Matches paper end date
        lookbacks: List[int] = [21, 63, 126, 252],
        fixed_weights: List[float] = [0.40, 0.30, 0.20, 0.10]
    ):
        """
        Initialize Defense First Base Strategy.
        
        Args:
            start_date: Strategy start date (default matches paper)
            end_date: Strategy end date
            lookbacks: Momentum lookback periods (days)
            fixed_weights: Allocation weights by rank
        """
        super().__init__(
            name='DEFENSE_FIRST_BASE',
            start_date=start_date,
            end_date=end_date
        )
        
        self.defensive_assets = ['TLT', 'GLD', 'DBC', 'UUP', 'BTAL']
        self.fallback_asset = 'SPY'
        self.lookbacks = lookbacks
        self.fixed_weights = fixed_weights
        
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate monthly allocation weights.
        
        Returns:
            DataFrame with dates as index and asset weights as columns
        """
        if self.prices is None or self.tbill is None:
            raise ValueError("Data not set. Call set_data() first.")
        
        # Get month-end rebalance dates
        rebalance_dates = self.get_monthly_rebalance_dates(self.prices)
        
        # Store weights for each rebalance date
        all_weights = {}
        
        for reb_date in rebalance_dates:
            # Use T-1 signal to avoid lookahead
            signal_date = self.get_signal_date(reb_date)
            
            # Calculate weights for this month
            weights = self._calculate_monthly_weights(signal_date)
            all_weights[reb_date] = weights
        
        # Convert to DataFrame
        weight_df = self.format_weight_df(all_weights)
        
        return weight_df
    
    def _calculate_monthly_weights(self, date: pd.Timestamp) -> Dict[str, float]:
        """
        Calculate portfolio weights for a given date.
        
        Args:
            date: Signal generation date
            
        Returns:
            Dictionary of asset -> weight
        """
        # Get available defensive assets for this date
        available_defensives = self.asset_manager.get_available_assets(
            date, self.defensive_assets
        )
        
        if len(available_defensives) == 0:
            # No defensives available - all to SPY
            return {self.fallback_asset: 1.0}
        
        # Calculate momentum scores for available assets
        momentum_scores = self._calculate_momentum_scores(
            available_defensives, date
        )
        
        # Rank assets by momentum (highest first)
        ranked_assets = sorted(
            momentum_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get T-bill rate for this date
        rf_rate = self.tbill.loc[date]
        
        # Assign weights and apply momentum filter
        weights = {}
        spy_allocation = 0.0
        
        # Only top 4 assets get weights (or fewer if less than 4 available)
        top_n = min(len(self.fixed_weights), len(ranked_assets))
        
        for i, (asset, momentum) in enumerate(ranked_assets[:top_n]):
            weight = self.fixed_weights[i]
            
            # Momentum filter: if momentum < risk-free rate, allocate to SPY
            if momentum < rf_rate:
                spy_allocation += weight
            else:
                weights[asset] = weight
        
        # Add SPY allocation if any
        if spy_allocation > 0:
            weights[self.fallback_asset] = spy_allocation
        
        return weights
    
    def _calculate_momentum_scores(
        self,
        assets: List[str],
        date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Calculate multi-period momentum score for each asset.
        
        Uses exact calendar month lookbacks (1, 3, 6, 12 months) per paper.
        
        Args:
            assets: List of asset tickers
            date: Date to calculate momentum up to
            
        Returns:
            Dictionary of asset -> momentum score (annualized)
        """
        scores = {}
        
        # Exact calendar month lookbacks (matching paper)
        month_lookbacks = [1, 3, 6, 12]
        
        for asset in assets:
            if asset not in self.prices.columns:
                scores[asset] = -np.inf  # Asset not in data
                continue
            
            # Get price series up to date
            prices = self.prices[asset].loc[:date]
            
            momentum_values = []
            for months in month_lookbacks:
                # Calculate date N months ago
                lookback_date = date - pd.DateOffset(months=months)
                
                # Find closest available trading day
                valid_dates = prices.index[prices.index <= lookback_date]
                if len(valid_dates) == 0 or len(prices) < 2:
                    continue
                
                start_date = valid_dates[-1]
                start_price = prices.loc[start_date]
                current_price = prices.iloc[-1]
                
                # Calculate return
                ret = (current_price / start_price) - 1
                
                # Annualize
                days_elapsed = (date - start_date).days
                if days_elapsed > 0:
                    ann_ret = ret * (365.25 / days_elapsed)
                    momentum_values.append(ann_ret)
            
            # Average across periods
            scores[asset] = np.mean(momentum_values) if momentum_values else 0.0
        
        return scores
    
    def get_strategy_info(self) -> Dict:
        """
        Return strategy metadata and rules.
        
        Returns:
            Dictionary with strategy information
        """
        return {
            'name': 'Defense First - Base (SPY)',
            'paper': 'A Simple and Effective Tactical Allocation Strategy',
            'author': 'QuantSeeker (based on Carlson, 2025)',
            'year': 2025,
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'rebalance_frequency': 'Monthly (end-of-month)',
            'signal_timing': 'T-1 (previous day close)',
            'lookbacks': self.lookbacks,
            'fixed_weights': self.fixed_weights,
            'defensive_assets': self.defensive_assets,
            'fallback_asset': self.fallback_asset,
            'expected_performance': {
                'CAGR': '9.5%',
                'Sharpe': 0.83,
                'Max_DD': '~12%',
                'Volatility': 'Half of SPY'
            },
            'rules': self._get_trading_rules()
        }
    
    def _get_trading_rules(self) -> str:
        """Get detailed trading rules as formatted string."""
        rules = """
DEFENSE FIRST BASE STRATEGY - SPY FALLBACK

ASSET UNIVERSE:
- Defensive (dynamic): TLT, GLD, DBC, UUP (+ BTAL from May 2011)
- Fallback: SPY (S&P 500 ETF)

MONTHLY REBALANCING LOGIC (T-1 signals):

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
           Reallocate its weight to SPY
       ELSE:
           Keep defensive allocation

5. Final Portfolio:
   - Defensive allocations (if momentum > T-bill)
   - SPY allocation = sum of reallocated weights
   - Total always sums to 100%

EXAMPLE SCENARIOS:

Scenario 1 - All defensives strong (2021):
- All momentum > T-bill → 40% TLT, 30% GLD, 20% DBC, 10% UUP, 0% SPY

Scenario 2 - Mixed signals (2022 bear market):
- TLT momentum > T-bill → 40% TLT
- Others < T-bill → 60% SPY (30%+20%+10% reallocated)

Scenario 3 - All defensives weak (2023 recovery):
- All momentum < T-bill → 100% SPY

EXPECTED PERFORMANCE (2008-2025):
- CAGR: 9.5% (vs SPY 11.3%)
- Sharpe: 0.83 (vs SPY 0.59)
- Max DD: ~12% (vs SPY ~24%)
- Volatility: ~Half of SPY
- 2022 bear market: +5%
"""
        
        return rules.strip()

