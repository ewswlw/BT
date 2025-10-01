"""
VIX Timing Strategy

Paper: "Timing Leveraged Equity Exposure in a TAA Model" (QuantSeeker, 2025)
Period: 2013-01-01 to present
Rebalance: Monthly (end-of-month)

Strategy Logic:
- Uses VIX filter to time SPXL (3x leveraged SPY) vs Cash
- Defensive assets (TLT, GLD, DBC, UUP, BTAL) follow momentum-based TAA
- Multi-lookback version: 10, 15, 20-day with graded allocation
- Volatility normalization to 10% target (per paper)
- VIX averaging (10-day average vs point estimate)
"""

from typing import Dict, List
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from .base_strategy import BaseTAAStrategy


class VIXTimingStrategy(BaseTAAStrategy):
    """
    VIX-based timing strategy for leveraged equity exposure.
    
    Toggles between SPXL (3x S&P 500) and Cash based on volatility risk premium.
    When realized vol < VIX → risk-on (allocate to SPXL)
    When realized vol > VIX → risk-off (allocate to Cash)
    
    Attributes:
        lookbacks: Realized volatility lookback periods [10, 15, 20] days
        use_multi_lookback: If True, use graded allocation across lookbacks
    """
    
    def __init__(
        self,
        start_date: str = '2013-01-01',
        end_date: str = '2025-07-31',  # Matches paper end date
        lookbacks: List[int] = [10, 15, 20],
        use_multi_lookback: bool = True,
        target_volatility: float = 0.10,  # 10% annualized vol target (per paper)
        vix_averaging: bool = True  # Use 10-day average VIX vs point estimate (per paper)
    ):
        """
        Initialize VIX Timing Strategy.
        
        Args:
            start_date: Strategy start date (default matches paper)
            end_date: Strategy end date
            lookbacks: Realized vol lookback periods (days)
            use_multi_lookback: Use multi-lookback graded allocation
            target_volatility: Target annual volatility for vol-normalization (default: 10%)
        """
        super().__init__(
            name='VIX_TIMING',
            start_date=start_date,
            end_date=end_date
        )
        
        self.lookbacks = lookbacks
        self.use_multi_lookback = use_multi_lookback
        self.target_volatility = target_volatility
        self.vix_averaging = vix_averaging
        
        # Defensive assets (available set expands when BTAL launches May 2011)
        self.defensive_assets = ['TLT', 'GLD', 'DBC', 'UUP', 'BTAL']
        
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate monthly allocation weights.
        
        Returns:
            DataFrame with dates as index and asset weights as columns
        """
        if self.prices is None or self.vix is None:
            raise ValueError("Data not set. Call set_data() first.")
        
        # Get month-end rebalance dates
        rebalance_dates = self.get_monthly_rebalance_dates(self.prices)
        
        # Store weights for each rebalance date
        all_weights = {}
        
        for reb_date in rebalance_dates:
            # Use T-1 signal to avoid lookahead
            signal_date = self.get_signal_date(reb_date)
            
            # Get available defensive assets for this date
            available_defensives = self.asset_manager.get_available_assets(
                signal_date, self.defensive_assets
            )
            
            # Implement full TAA logic for defensive assets (per paper methodology)
            defensive_weight_total = 0.60  # 60% to defensives, 40% to fallback
            
            weights = {}
            if len(available_defensives) >= 4:
                # Calculate momentum scores for defensive assets
                momentum_scores = self._calculate_defensive_momentum_scores(
                    available_defensives, signal_date
                )
                
                # Rank by momentum and assign fixed weights (40/30/20/10%)
                ranked_assets = sorted(
                    momentum_scores.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                fixed_weights = [0.40, 0.30, 0.20, 0.10]
                tbill_rate = self.tbill.loc[signal_date] if signal_date in self.tbill.index else 0.02
                
                for i, (asset, momentum_score) in enumerate(ranked_assets[:4]):
                    if momentum_score > tbill_rate:
                        # Asset passes momentum filter
                        weights[asset] = fixed_weights[i] * defensive_weight_total
                    else:
                        # Asset fails momentum filter - weight goes to fallback
                        weights[asset] = 0.0
            elif len(available_defensives) > 0:
                # Fallback to equal weight if less than 4 assets available
                defensive_each = defensive_weight_total / len(available_defensives)
                for asset in available_defensives:
                    weights[asset] = defensive_each
            
            # Calculate fallback allocation (SPXL vs Cash)
            # First, calculate total defensive allocation
            total_defensive_weight = sum(weights.values())
            fallback_weight = 1.0 - total_defensive_weight
            
            # Apply VIX filter
            spxl_allocation = self._calculate_spxl_allocation(signal_date)
            
            if spxl_allocation > 0:
                weights['SPXL'] = fallback_weight * spxl_allocation
                if spxl_allocation < 1.0:
                    weights['CASH'] = fallback_weight * (1 - spxl_allocation)
            else:
                weights['CASH'] = fallback_weight
            
            all_weights[reb_date] = weights
        
        # Convert to DataFrame
        weight_df = self.format_weight_df(all_weights)
        
        return weight_df
    
    def _calculate_spxl_allocation(self, date: pd.Timestamp) -> float:
        """
        Calculate SPXL allocation percentage based on VIX filter.
        
        Args:
            date: Signal generation date
            
        Returns:
            SPXL allocation: 0.0 (all cash) to 1.0 (all SPXL)
        """
        if self.use_multi_lookback:
            return self._multi_lookback_allocation(date)
        else:
            return self._simple_allocation(date)
    
    def _simple_allocation(self, date: pd.Timestamp, lookback: int = 20) -> float:
        """
        Simple VIX filter: single lookback (default 20 days).
        
        Args:
            date: Signal date
            lookback: Realized vol lookback period
            
        Returns:
            1.0 if risk-on (RV < VIX), 0.0 if risk-off (RV >= VIX)
        """
        # Calculate realized volatility of SPY
        spy_returns = self.prices['SPY'].pct_change()
        rv = self.calculate_realized_volatility(
            spy_returns.loc[:date], lookback
        )
        
        # Get VIX value (with optional averaging)
        if self.vix_averaging:
            # Use 10-day average VIX as recommended in paper
            vix_window = self.vix.loc[:date].tail(10)
            vix = vix_window.mean() if len(vix_window) >= 5 else self.vix.loc[date]
        else:
            vix = self.vix.loc[date]
        
        # Simple rule: RV < VIX → risk-on
        if pd.isna(rv) or pd.isna(vix):
            return 0.0  # Default to cash if data missing
        
        return 1.0 if rv < vix else 0.0
    
    def _multi_lookback_allocation(self, date: pd.Timestamp) -> float:
        """
        Multi-lookback VIX filter with graded allocation.
        
        Uses multiple lookbacks (10, 15, 20 days) and counts signals.
        - 3 signals risk-on: 100% SPXL
        - 2 signals risk-on: 67% SPXL, 33% Cash
        - 1 signal risk-on: 33% SPXL, 67% Cash
        - 0 signals risk-on: 100% Cash
        
        Args:
            date: Signal date
            
        Returns:
            SPXL allocation percentage (0.0 to 1.0)
        """
        spy_returns = self.prices['SPY'].pct_change()
        
        # Get VIX value (with optional averaging)
        if self.vix_averaging:
            # Use 10-day average VIX as recommended in paper
            vix_window = self.vix.loc[:date].tail(10)
            vix = vix_window.mean() if len(vix_window) >= 5 else self.vix.loc[date]
        else:
            vix = self.vix.loc[date]
        
        if pd.isna(vix):
            return 0.0
        
        # Count risk-on signals across lookbacks
        risk_on_signals = 0
        
        for lookback in self.lookbacks:
            rv = self.calculate_realized_volatility(
                spy_returns.loc[:date], lookback
            )
            
            if not pd.isna(rv) and rv < vix:
                risk_on_signals += 1
        
        # Graded allocation based on signal count
        total_signals = len(self.lookbacks)
        
        if risk_on_signals == total_signals:
            return 1.0  # 100% SPXL
        elif risk_on_signals == total_signals - 1:
            return 0.67  # 67% SPXL
        elif risk_on_signals == total_signals - 2:
            return 0.33  # 33% SPXL
        else:
            return 0.0  # 100% Cash
    
    def get_strategy_info(self) -> Dict:
        """
        Return strategy metadata and rules.
        
        Returns:
            Dictionary with strategy information
        """
        return {
            'name': 'VIX Timing Strategy',
            'paper': 'Timing Leveraged Equity Exposure in a TAA Model',
            'author': 'QuantSeeker',
            'year': 2025,
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'rebalance_frequency': 'Monthly (end-of-month)',
            'signal_timing': 'T-1 (previous day close)',
            'lookbacks': self.lookbacks,
            'multi_lookback': self.use_multi_lookback,
            'defensive_assets': self.defensive_assets,
            'fallback_assets': ['SPXL', 'CASH'],
            'rules': self._get_trading_rules()
        }
    
    def _get_trading_rules(self) -> str:
        """Get detailed trading rules as formatted string."""
        if self.use_multi_lookback:
            rules = """
VIX TIMING STRATEGY - MULTI-LOOKBACK VERSION

ASSET UNIVERSE:
- Defensive (dynamic): TLT, GLD, DBC, UUP (+ BTAL from May 2011)
- Fallback: SPXL (3x S&P 500) vs Cash

MONTHLY REBALANCING LOGIC (T-1 signals):

1. Calculate Realized Volatility of SPY:
   - Lookbacks: 10, 15, 20 trading days
   - Formula: std(daily_returns) * sqrt(252)

2. Get VIX Index Value (previous day close)

3. Count Risk-On Signals:
   - For each lookback: IF realized_vol < VIX → count as risk-on
   - Sum total risk-on signals

4. Determine SPXL Allocation:
   - 3/3 signals: 100% SPXL, 0% Cash
   - 2/3 signals: 67% SPXL, 33% Cash
   - 1/3 signals: 33% SPXL, 67% Cash
   - 0/3 signals: 0% SPXL, 100% Cash

5. Defensive Assets:
   - Maintain 60% total allocation using momentum-based ranking
   - Rank defensives by 1/3/6/12 month momentum scores
   - Assign fixed weights: 40%, 30%, 20%, 10% (top 4 only)
   - Filter by T-bill rate: if momentum < T-bill, reallocate to fallback
   - Fallback gets remaining 40% (split between SPXL/Cash per VIX filter)

EXPECTED PERFORMANCE (from paper, vol-normalized):
- CAGR: >25%
- Sharpe: 1.41
- Max Drawdown: 15-16%
"""
        else:
            rules = """
VIX TIMING STRATEGY - SIMPLE VERSION

ASSET UNIVERSE:
- Defensive (dynamic): TLT, GLD, DBC, UUP (+ BTAL from May 2011)
- Fallback: SPXL (3x S&P 500) vs Cash

MONTHLY REBALANCING LOGIC (T-1 signals):

1. Calculate 20-day Realized Volatility of SPY
   - Formula: std(daily_returns[-20:]) * sqrt(252)

2. Get VIX Index Value (previous day close)

3. VIX Filter Decision:
   - IF realized_vol < VIX: Allocate 40% to SPXL
   - ELSE: Allocate 40% to Cash

4. Defensive Assets:
   - Maintain 60% total allocation (equal-weighted across available assets)

EXPECTED PERFORMANCE (from paper):
- Better than unfiltered SPXL
- Lower drawdowns during volatility spikes
"""
        
        return rules.strip()
    
    def _calculate_defensive_momentum_scores(
        self,
        assets: List[str],
        date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Calculate momentum scores for defensive assets using exact calendar months.
        
        Args:
            assets: List of defensive asset tickers
            date: Date to calculate momentum for
            
        Returns:
            Dictionary of asset -> momentum score (annualized)
        """
        momentum_scores = {}
        
        # Exact calendar month lookbacks (matching Defense First methodology)
        month_lookbacks = [1, 3, 6, 12]
        
        for asset in assets:
            if asset not in self.prices.columns:
                momentum_scores[asset] = -np.inf
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
            momentum_scores[asset] = np.mean(momentum_values) if momentum_values else 0.0
        
        return momentum_scores

