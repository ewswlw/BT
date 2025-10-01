"""
Sector Rotation Strategy

Paper: "Replicating an Asset Allocation Model" (QuantSeeker, 2025)
Period: 1999-12-01 to present
Rebalance: Monthly (end-of-month)

Strategy Logic:
- Rank 9-11 sector ETFs using 4 signals: momentum, volatility, correlation, trend
- Select top 5 sectors, equal-weight
- Apply absolute momentum filter (must be positive to qualify)
- Cash allocation for sectors with negative momentum
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from .base_strategy import BaseTAAStrategy


class SectorRotationStrategy(BaseTAAStrategy):
    """
    Multi-signal sector rotation strategy.
    
    Uses 4 ranking signals to select best sectors:
    1. Momentum (higher is better)
    2. Volatility (lower is better)
    3. Correlation (lower is better - diversification)
    4. Trend (closer to highs is better)
    
    Attributes:
        lookbacks: Lookback periods for all calculations
        top_n_sectors: Number of sectors to select (default: 5)
    """
    
    def __init__(
        self,
        start_date: str = '1999-12-01',
        end_date: str = '2025-08-31',  # Matches paper end date (August 2025)
        lookbacks: List[int] = [21, 63, 126, 252],
        top_n_sectors: int = 5
    ):
        """
        Initialize Sector Rotation Strategy.
        
        Args:
            start_date: Strategy start date (default matches paper)
            end_date: Strategy end date
            lookbacks: Lookback periods for all signals
            top_n_sectors: Number of sectors to hold
        """
        super().__init__(
            name='SECTOR_ROTATION',
            start_date=start_date,
            end_date=end_date
        )
        
        self.lookbacks = lookbacks
        self.top_n_sectors = top_n_sectors
        
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate monthly allocation weights.
        
        Returns:
            DataFrame with dates as index and sector weights as columns
        """
        if self.prices is None:
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
            Dictionary of sector -> weight
        """
        # Get available sectors for this date
        available_sectors = self.asset_manager.get_sector_universe(date)
        
        # Filter to only sectors in our data
        available_sectors = [s for s in available_sectors if s in self.prices.columns]
        
        if len(available_sectors) == 0:
            return {'CASH': 1.0}
        
        # Calculate composite ranks for all available sectors
        composite_ranks = self._calculate_composite_ranks(available_sectors, date)
        
        # Select top N sectors
        top_n = min(self.top_n_sectors, len(available_sectors))
        top_sectors = sorted(
            composite_ranks.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        # Equal weight allocation
        equal_weight = 1.0 / top_n
        
        # Apply absolute momentum filter
        weights = {}
        cash_allocation = 0.0
        
        for sector, _ in top_sectors:
            # Check if sector has positive momentum
            if self._has_positive_momentum(sector, date):
                weights[sector] = equal_weight
            else:
                cash_allocation += equal_weight
        
        # Add cash if any
        if cash_allocation > 0:
            weights['CASH'] = cash_allocation
        
        return weights
    
    def _calculate_composite_ranks(
        self,
        sectors: List[str],
        date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Calculate composite rank for each sector.
        
        Combines 4 signals: momentum, volatility, correlation, trend.
        Each signal is ranked, then average rank is composite score.
        
        Args:
            sectors: List of sector tickers
            date: Date to calculate ranks
            
        Returns:
            Dictionary of sector -> composite rank score
        """
        # Calculate all 4 signals
        momentum_scores = self._calculate_momentum_signals(sectors, date)
        volatility_scores = self._calculate_volatility_signals(sectors, date)
        correlation_scores = self._calculate_correlation_signals(sectors, date)
        trend_scores = self._calculate_trend_signals(sectors, date)
        
        # Rank each signal
        momentum_ranks = self._rank_dict(momentum_scores, ascending=False)  # Higher is better
        volatility_ranks = self._rank_dict(volatility_scores, ascending=True)  # Lower is better
        correlation_ranks = self._rank_dict(correlation_scores, ascending=True)  # Lower is better
        trend_ranks = self._rank_dict(trend_scores, ascending=False)  # Higher is better
        
        # Combine ranks (average)
        composite_ranks = {}
        for sector in sectors:
            avg_rank = np.mean([
                momentum_ranks.get(sector, 0),
                volatility_ranks.get(sector, 0),
                correlation_ranks.get(sector, 0),
                trend_ranks.get(sector, 0)
            ])
            composite_ranks[sector] = avg_rank
        
        return composite_ranks
    
    def _calculate_momentum_signals(
        self,
        sectors: List[str],
        date: pd.Timestamp
    ) -> Dict[str, float]:
        """Calculate momentum score for each sector."""
        scores = {}
        for sector in sectors:
            prices = self.prices[sector].loc[:date]
            momentum = self.calculate_momentum(prices, self.lookbacks)
            scores[sector] = momentum
        return scores
    
    def _calculate_volatility_signals(
        self,
        sectors: List[str],
        date: pd.Timestamp
    ) -> Dict[str, float]:
        """Calculate volatility score for each sector."""
        scores = {}
        for sector in sectors:
            returns = self.prices[sector].pct_change().loc[:date]
            
            # Calculate volatility over multiple lookbacks
            vols = []
            for lookback in self.lookbacks:
                if len(returns) >= lookback:
                    vol = returns.iloc[-lookback:].std() * np.sqrt(252)
                    vols.append(vol)
            
            scores[sector] = np.mean(vols) if vols else np.inf
        
        return scores
    
    def _calculate_correlation_signals(
        self,
        sectors: List[str],
        date: pd.Timestamp
    ) -> Dict[str, float]:
        """Calculate average correlation with other sectors."""
        scores = {}
        returns = self.calculate_returns(self.prices).loc[:date]
        
        for sector in sectors:
            if sector not in returns.columns:
                scores[sector] = 1.0
                continue
            
            # Calculate correlation with all other sectors over multiple lookbacks
            avg_corrs = []
            
            for lookback in self.lookbacks:
                if len(returns) >= lookback:
                    lookback_returns = returns.iloc[-lookback:]
                    
                    # Get correlations with other sectors
                    corrs = []
                    for other in sectors:
                        if other != sector and other in lookback_returns.columns:
                            corr = lookback_returns[sector].corr(lookback_returns[other])
                            if not pd.isna(corr):
                                corrs.append(corr)
                    
                    if corrs:
                        avg_corrs.append(np.mean(corrs))
            
            scores[sector] = np.mean(avg_corrs) if avg_corrs else 0.5
        
        return scores
    
    def _calculate_trend_signals(
        self,
        sectors: List[str],
        date: pd.Timestamp
    ) -> Dict[str, float]:
        """Calculate trend score (price to high ratio)."""
        scores = {}
        
        for sector in sectors:
            prices = self.prices[sector].loc[:date]
            current_price = prices.iloc[-1]
            
            # Calculate ratio of current price to rolling high over multiple lookbacks
            ratios = []
            for lookback in self.lookbacks:
                if len(prices) >= lookback:
                    rolling_high = prices.iloc[-lookback:].max()
                    ratio = current_price / rolling_high
                    ratios.append(ratio)
            
            scores[sector] = np.mean(ratios) if ratios else 0.0
        
        return scores
    
    def _has_positive_momentum(self, sector: str, date: pd.Timestamp) -> bool:
        """Check if sector has positive 1-month momentum."""
        prices = self.prices[sector].loc[:date]
        
        if len(prices) < 22:
            return False
        
        # Simple 21-day (1-month) return
        ret = (prices.iloc[-1] / prices.iloc[-22] - 1)
        return ret > 0
    
    def _rank_dict(
        self,
        scores: Dict[str, float],
        ascending: bool = False
    ) -> Dict[str, int]:
        """
        Rank dictionary values.
        
        Args:
            scores: Dictionary to rank
            ascending: If True, lower values get higher ranks
            
        Returns:
            Dictionary with ranks (1 = best)
        """
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=not ascending)
        return {item[0]: rank + 1 for rank, item in enumerate(sorted_items)}
    
    def get_strategy_info(self) -> Dict:
        """
        Return strategy metadata and rules.
        
        Returns:
            Dictionary with strategy information
        """
        return {
            'name': 'Sector Rotation',
            'paper': 'Replicating an Asset Allocation Model',
            'author': 'QuantSeeker (based on Giordano 2018, 2019)',
            'year': 2025,
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'rebalance_frequency': 'Monthly (end-of-month)',
            'signal_timing': 'T-1 (previous day close)',
            'lookbacks': self.lookbacks,
            'top_n_sectors': self.top_n_sectors,
            'sector_universe': '9-11 sectors (evolves over time)',
            'expected_performance': {
                'Combined_Signals_Sharpe': 0.60,
                'Max_DD': '~20-25%',
                'Note': 'Standalone version. Better when combined with Defense First.'
            },
            'rules': self._get_trading_rules()
        }
    
    def _get_trading_rules(self) -> str:
        """Get detailed trading rules as formatted string."""
        rules = """
SECTOR ROTATION STRATEGY - MULTI-SIGNAL

SECTOR UNIVERSE (dynamic):
- 1999-2015: 9 sectors (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY)
- 2015-2018: 10 sectors (+ XLRE Real Estate)
- 2018+: 11 sectors (+ XLC Communication Services)

MONTHLY REBALANCING LOGIC (T-1 signals):

1. Calculate 4 Ranking Signals for Each Sector:

   A. MOMENTUM SIGNAL:
      - Lookbacks: 21, 63, 126, 252 days
      - For each: return = (price[t]/price[t-lb] - 1) * (252/lb)
      - Score = average annualized return
      - Rank: Higher is better

   B. VOLATILITY SIGNAL:
      - Lookbacks: 21, 63, 126, 252 days
      - For each: vol = std(returns) * sqrt(252)
      - Score = average volatility
      - Rank: Lower is better (inverse rank)

   C. CORRELATION SIGNAL:
      - Lookbacks: 21, 63, 126, 252 days
      - For each: avg_corr = mean(correlations with all other sectors)
      - Score = average correlation
      - Rank: Lower is better (diversification benefit)

   D. TREND SIGNAL:
      - Lookbacks: 21, 63, 126, 252 days
      - For each: trend = current_price / max(prices over lookback)
      - Score = average trend ratio
      - Rank: Higher is better (closer to highs)

2. Calculate Composite Rank:
   - For each sector: composite_rank = mean([rank_A, rank_B, rank_C, rank_D])
   - Sort sectors by composite rank

3. Select Top 5 Sectors:
   - Take top 5 ranked sectors
   - Equal weight: 20% each

4. Apply Absolute Momentum Filter:
   - For each top 5 sector:
       IF 21-day return > 0:
           Allocate 20%
       ELSE:
           Reallocate 20% to Cash

5. Final Portfolio:
   - Up to 5 sectors at 20% each
   - Cash for any sectors with negative momentum

EXAMPLE ALLOCATION:
- Strong market: 5 sectors × 20% = 100% equity
- Mixed market: 3 sectors × 20% + 40% cash
- Weak market: 0% sectors + 100% cash

EXPECTED PERFORMANCE (1999-2025):
- Individual signals: Sharpe 0.50-0.58
- Combined signals: Sharpe 0.60
- Max DD: ~20-25%
- Outperforms SPY and equal-weight sectors

NOTE: Works best when combined with Defense First model
"""
        
        return rules.strip()

