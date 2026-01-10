"""
Strategy Template for BT Backtesting Framework

This template provides a standardized structure for implementing trading strategies.
Copy this template and fill in the specific logic for your strategy.

Author: [Your Name]
Created: [Date]
Strategy Type: [Momentum/Mean Reversion/Factor/etc.]
"""

from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class StrategyConfig:
    """
    Configuration for strategy parameters.

    Attributes:
        name: Strategy name
        lookback_days: Lookback period for calculations
        rebalance_frequency: How often to rebalance ('D', 'W', 'M', 'Q')
        long_only: Whether strategy is long-only
        max_positions: Maximum number of positions
        transaction_cost_bps: Transaction cost in basis points
    """
    name: str = "MyStrategy"
    lookback_days: int = 126
    rebalance_frequency: str = "M"
    long_only: bool = True
    max_positions: int = 20
    transaction_cost_bps: float = 10.0


class Strategy:
    """
    Base strategy implementation.

    This class provides the structure for a trading strategy including:
    - Signal generation
    - Position sizing
    - Risk management
    - Performance tracking
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize strategy with configuration.

        Args:
            config: StrategyConfig object with parameters
        """
        self.config = config
        self.name = config.name

    def calculate_signals(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculate trading signals.

        IMPORTANT: Ensure no look-ahead bias! Only use data available at each point in time.
        Use .shift(1) to prevent using current period data for current period signals.

        Args:
            prices: DataFrame with OHLCV data (index: date, columns: symbol)
            fundamentals: Optional fundamental data
            **kwargs: Additional data sources

        Returns:
            DataFrame of signals (1 = long, 0 = neutral, -1 = short)

        Example:
            >>> signals = strategy.calculate_signals(prices)
            >>> print(signals.head())
                        AAPL  MSFT  GOOGL
            2020-01-01   1.0   0.0   1.0
            2020-01-02   1.0   1.0   0.0
        """
        # TODO: Implement your signal logic here

        # Example: Simple momentum strategy
        # Calculate returns over lookback period
        returns = prices.pct_change(self.config.lookback_days)

        # CRITICAL: Shift to avoid look-ahead bias
        returns = returns.shift(1)

        # Generate signals: Long top quartile, short bottom quartile
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

        for date in prices.index:
            if date in returns.index:
                date_returns = returns.loc[date]

                # Remove NaN values
                valid_returns = date_returns.dropna()

                if len(valid_returns) > 0:
                    # Long top 25%
                    top_threshold = valid_returns.quantile(0.75)
                    signals.loc[date, valid_returns >= top_threshold] = 1.0

                    # Short bottom 25% (if not long-only)
                    if not self.config.long_only:
                        bottom_threshold = valid_returns.quantile(0.25)
                        signals.loc[date, valid_returns <= bottom_threshold] = -1.0

        return signals

    def calculate_position_sizes(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        portfolio_value: float = 1_000_000,
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculate position sizes based on signals.

        Args:
            signals: DataFrame of trading signals
            prices: Current prices
            portfolio_value: Total portfolio value
            **kwargs: Additional context (volatility, risk, etc.)

        Returns:
            DataFrame of position sizes (in shares or contract values)

        Example:
            >>> positions = strategy.calculate_position_sizes(signals, prices)
        """
        # TODO: Implement position sizing logic

        # Example: Equal-weight positions
        positions = signals.copy()

        for date in signals.index:
            date_signals = signals.loc[date]
            n_positions = (date_signals != 0).sum()

            if n_positions > 0:
                # Equal weight for each position
                weight_per_position = 1.0 / n_positions

                # Limit to max_positions
                if n_positions > self.config.max_positions:
                    weight_per_position = 1.0 / self.config.max_positions

                # Apply weights
                positions.loc[date] = date_signals * weight_per_position

        return positions

    def apply_risk_controls(
        self,
        positions: pd.DataFrame,
        prices: pd.DataFrame,
        current_portfolio: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Apply risk management rules to positions.

        Args:
            positions: Proposed positions
            prices: Current prices
            current_portfolio: Current holdings

        Returns:
            Adjusted positions after risk controls

        Example:
            >>> safe_positions = strategy.apply_risk_controls(positions, prices)
        """
        # TODO: Implement risk controls

        adjusted_positions = positions.copy()

        # Example risk controls:

        # 1. Maximum position size (% of portfolio)
        max_position_pct = 0.10  # 10% max per position
        adjusted_positions = adjusted_positions.clip(-max_position_pct, max_position_pct)

        # 2. Maximum sector exposure (if sector data available)
        # [Implement sector limits]

        # 3. Maximum leverage
        max_leverage = 2.0 if not self.config.long_only else 1.0
        total_exposure = adjusted_positions.abs().sum(axis=1)
        scale_factor = max_leverage / total_exposure.where(total_exposure > max_leverage, 1.0)
        adjusted_positions = adjusted_positions.multiply(scale_factor, axis=0)

        return adjusted_positions

    def generate_trades(
        self,
        target_positions: pd.DataFrame,
        current_positions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate trades to move from current to target positions.

        Args:
            target_positions: Desired positions
            current_positions: Current positions

        Returns:
            DataFrame of trades (positive = buy, negative = sell)
        """
        trades = target_positions - current_positions
        return trades

    def calculate_transaction_costs(
        self,
        trades: pd.DataFrame,
        prices: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate transaction costs for trades.

        Args:
            trades: Trade sizes
            prices: Execution prices

        Returns:
            Series of transaction costs by date
        """
        # Cost in dollars
        trade_values = (trades.abs() * prices).sum(axis=1)
        costs = trade_values * (self.config.transaction_cost_bps / 10000)
        return costs

    def backtest(
        self,
        prices: pd.DataFrame,
        initial_capital: float = 1_000_000,
        **kwargs
    ) -> Dict:
        """
        Run full backtest of the strategy.

        Args:
            prices: Historical price data
            initial_capital: Starting capital
            **kwargs: Additional data and parameters

        Returns:
            Dictionary with backtest results including:
            - equity_curve: Portfolio value over time
            - positions: Position history
            - trades: Trade history
            - metrics: Performance metrics
        """
        # Calculate signals
        signals = self.calculate_signals(prices, **kwargs)

        # Calculate positions
        positions = self.calculate_position_sizes(
            signals, prices, initial_capital, **kwargs
        )

        # Apply risk controls
        safe_positions = self.apply_risk_controls(positions, prices)

        # TODO: Implement full backtest simulation
        # This would include:
        # - Tracking portfolio value over time
        # - Recording all trades
        # - Calculating returns
        # - Applying transaction costs
        # - Generating performance metrics

        results = {
            "signals": signals,
            "positions": safe_positions,
            "equity_curve": None,  # TODO: Calculate
            "trades": None,  # TODO: Generate
            "metrics": None  # TODO: Calculate
        }

        return results


def calculate_performance_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None
) -> Dict:
    """
    Calculate comprehensive performance metrics.

    Args:
        returns: Strategy returns (daily)
        benchmark_returns: Optional benchmark returns for comparison

    Returns:
        Dictionary of performance metrics
    """
    # Annualization factor (252 trading days)
    annual_factor = 252

    # Basic metrics
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / annual_factor
    cagr = (1 + total_return) ** (1 / n_years) - 1

    # Risk metrics
    annual_vol = returns.std() * np.sqrt(annual_factor)
    sharpe = (returns.mean() * annual_factor) / annual_vol if annual_vol > 0 else 0

    # Downside metrics
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(annual_factor)
    sortino = (returns.mean() * annual_factor) / downside_vol if downside_vol > 0 else 0

    # Drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    metrics = {
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": cagr / abs(max_drawdown) if max_drawdown != 0 else 0,
    }

    # Benchmark comparison
    if benchmark_returns is not None:
        correlation = returns.corr(benchmark_returns)
        tracking_error = (returns - benchmark_returns).std() * np.sqrt(annual_factor)
        information_ratio = (
            (returns.mean() - benchmark_returns.mean()) * annual_factor / tracking_error
            if tracking_error > 0 else 0
        )

        metrics.update({
            "correlation": correlation,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio
        })

    return metrics


if __name__ == "__main__":
    """
    Example usage and testing.
    """
    # Create configuration
    config = StrategyConfig(
        name="Example Momentum Strategy",
        lookback_days=126,
        rebalance_frequency="M",
        long_only=True,
        max_positions=20
    )

    # Initialize strategy
    strategy = Strategy(config)

    # Load data (example)
    # prices = pd.read_parquet("data/processed/prices.parquet")

    # Run backtest
    # results = strategy.backtest(prices)

    # Calculate metrics
    # metrics = calculate_performance_metrics(results["returns"])

    print(f"Strategy '{strategy.name}' initialized successfully")
    print(f"Configuration: {config}")
