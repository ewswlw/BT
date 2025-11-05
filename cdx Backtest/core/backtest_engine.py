"""
Backtesting engine with 7-day minimum holding period enforcement.
"""

from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    print("Warning: vectorbt not available, using manual calculations")


@dataclass
class BacktestResult:
    """Container for backtest results."""
    strategy_name: str
    portfolio: Optional[object]  # vbt.Portfolio or manual portfolio
    entry_signals: pd.Series
    exit_signals: pd.Series
    returns: pd.Series
    equity_curve: pd.Series
    metrics: Dict
    config: Dict
    trades_count: int = 0
    time_in_market: float = 0.0


class PortfolioManager:
    """Manages portfolio positions with 7-day minimum holding period."""
    
    def __init__(self, initial_capital: float = 100000, fees: float = 0.0, slippage: float = 0.0):
        self.initial_capital = initial_capital
        self.fees = fees
        self.slippage = slippage
    
    def apply_holding_period(self, raw_signals: pd.Series, holding_days: int = 7) -> pd.Series:
        """
        Apply minimum holding period constraint to signals.
        
        Once entered, position is held for exactly 'holding_days' regardless of signals.
        
        Args:
            raw_signals: Raw binary signals (1=enter, 0=no position)
            holding_days: Minimum days to hold position
            
        Returns:
            Adjusted signals with holding period enforced
        """
        adjusted_signals = pd.Series(0, index=raw_signals.index, dtype=int)
        
        i = 0
        while i < len(raw_signals):
            if raw_signals.iloc[i] == 1:
                # Enter position and hold for 'holding_days'
                end_idx = min(i + holding_days, len(raw_signals))
                adjusted_signals.iloc[i:end_idx] = 1
                i = end_idx  # Skip ahead to after holding period
            else:
                i += 1
        
        return adjusted_signals


class BacktestEngine:
    """Portfolio backtesting engine."""
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 fees: float = 0.0,
                 slippage: float = 0.0,
                 holding_period_days: int = 7):
        self.initial_capital = initial_capital
        self.fees = fees
        self.slippage = slippage
        self.holding_period_days = holding_period_days
        self.portfolio_manager = PortfolioManager(initial_capital, fees, slippage)
    
    def run_backtest(self, 
                     price_data: pd.Series,
                     entry_signals: pd.Series,
                     exit_signals: pd.Series,
                     apply_holding_period: bool = True) -> BacktestResult:
        """
        Run backtest with given signals.
        
        Args:
            price_data: Price series for the asset
            entry_signals: Boolean series indicating entry signals
            exit_signals: Boolean series indicating exit signals
            apply_holding_period: Whether to enforce 7-day minimum holding period
            
        Returns:
            BacktestResult object
        """
        # Remove duplicate indices from price data
        price_data = price_data[~price_data.index.duplicated(keep='first')]
        
        # Align signals with price data
        entry_signals = entry_signals.loc[~entry_signals.index.duplicated(keep='first')]
        exit_signals = exit_signals.loc[~exit_signals.index.duplicated(keep='first')]
        
        # Reindex to match price data
        entry_signals = entry_signals.reindex(price_data.index, fill_value=False)
        exit_signals = exit_signals.reindex(price_data.index, fill_value=False)
        
        # Generate positions from entry/exit signals
        positions = pd.Series(0, index=price_data.index, dtype=int)
        
        # Create position series
        in_position = False
        days_held = 0
        
        for i in range(len(positions)):
            if not in_position and entry_signals.iloc[i]:
                # Enter position
                in_position = True
                days_held = 0
                positions.iloc[i] = 1
            elif in_position:
                # Check if we should exit
                if exit_signals.iloc[i] and days_held >= self.holding_period_days:
                    # Exit position (holding period satisfied)
                    in_position = False
                    days_held = 0
                    positions.iloc[i] = 0
                elif days_held < self.holding_period_days:
                    # Must hold for minimum period
                    positions.iloc[i] = 1
                    days_held += 1
                else:
                    # Continue holding
                    positions.iloc[i] = 1
                    days_held += 1
        
        # Apply holding period if requested
        if apply_holding_period:
            # Convert positions to raw signals for holding period logic
            raw_signals = positions.copy()
            positions = self.portfolio_manager.apply_holding_period(raw_signals, self.holding_period_days)
        
        # Generate entry/exit signals from positions
        positions_shifted = positions.shift(1).fillna(0).astype(int)
        final_entry_signals = (positions == 1) & (positions_shifted == 0)
        final_exit_signals = (positions == 0) & (positions_shifted == 1)
        
        if VECTORBT_AVAILABLE:
            portfolio = self._run_vectorbt_backtest(price_data, final_entry_signals, final_exit_signals)
            returns = portfolio.returns()
            equity_curve = portfolio.value()
            trades_count = portfolio.trades.count() if hasattr(portfolio, 'trades') else 0
        else:
            portfolio, returns, equity_curve, trades_count = self._run_manual_backtest(
                price_data, final_entry_signals, final_exit_signals
            )
        
        # Calculate time in market
        time_in_market = positions.mean()
        
        # Calculate basic metrics
        metrics = self._calculate_basic_metrics(returns, equity_curve)
        
        return BacktestResult(
            strategy_name="Strategy",
            portfolio=portfolio,
            entry_signals=final_entry_signals,
            exit_signals=final_exit_signals,
            returns=returns,
            equity_curve=equity_curve,
            metrics=metrics,
            config={
                'initial_capital': self.initial_capital,
                'fees': self.fees,
                'slippage': self.slippage,
                'holding_period_days': self.holding_period_days
            },
            trades_count=trades_count,
            time_in_market=time_in_market
        )
    
    def _run_vectorbt_backtest(self, 
                              price_series: pd.Series, 
                              entry_signals: pd.Series, 
                              exit_signals: pd.Series) -> 'vbt.Portfolio':
        """Run backtest using vectorbt."""
        portfolio = vbt.Portfolio.from_signals(
            price_series,
            entries=entry_signals,
            exits=exit_signals,
            freq='D',  # Daily frequency
            init_cash=self.initial_capital,
            fees=self.fees,
            slippage=self.slippage
        )
        return portfolio
    
    def _run_manual_backtest(self, 
                           price_series: pd.Series,
                           entry_signals: pd.Series,
                           exit_signals: pd.Series) -> Tuple[None, pd.Series, pd.Series, int]:
        """Run manual backtest when vectorbt is not available."""
        cash = self.initial_capital
        position = 0
        shares = 0
        portfolio_value = []
        trades_count = 0
        
        for i, (date, price) in enumerate(price_series.items()):
            if pd.isna(price):
                portfolio_value.append(cash + shares * price if not pd.isna(price) else cash)
                continue
            
            # Check for entry signal
            if entry_signals.iloc[i] and shares == 0:
                # Buy with available cash (minus fees)
                shares_to_buy = cash / price * (1 - self.fees - self.slippage)
                shares = shares_to_buy
                cash = 0
                trades_count += 1
            
            # Check for exit signal
            elif exit_signals.iloc[i] and shares > 0:
                # Sell position
                cash = shares * price * (1 - self.fees - self.slippage)
                shares = 0
                trades_count += 1
            
            # Calculate portfolio value
            current_value = cash + shares * price
            portfolio_value.append(current_value)
        
        # Convert to pandas Series
        equity_curve = pd.Series(portfolio_value, index=price_series.index)
        returns = equity_curve.pct_change().fillna(0)
        
        return None, returns, equity_curve, trades_count
    
    def _calculate_basic_metrics(self, returns: pd.Series, equity_curve: pd.Series) -> Dict:
        """Calculate basic performance metrics."""
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        # Use standard trading days per year (252)
        annual_factor = 252
        
        # Annualized metrics
        years = len(returns) / annual_factor
        cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(annual_factor)
        sharpe_ratio = (returns.mean() * annual_factor) / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(annual_factor) if len(downside_returns) > 0 else 0
        sortino_ratio = (returns.mean() * annual_factor) / downside_std if downside_std > 0 else 0
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'final_value': equity_curve.iloc[-1],
            'initial_value': equity_curve.iloc[0]
        }

