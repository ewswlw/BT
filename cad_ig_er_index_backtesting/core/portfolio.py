"""
Portfolio management and backtesting engine.
"""

from typing import Optional, Dict, Tuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
from .config import PortfolioConfig

try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
    # Configure vectorbt for weekly data
    vbt.settings.array_wrapper['freq'] = 'W'
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


class PortfolioEngine:
    """Portfolio backtesting engine."""
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        
    def run_backtest(self, 
                     price_data: pd.DataFrame,
                     entry_signals: pd.Series,
                     exit_signals: pd.Series,
                     asset_column: Optional[str] = None) -> BacktestResult:
        """Run backtest with given signals."""
        
        # Select trading asset
        if asset_column is None:
            asset_column = price_data.columns[0]
        
        if asset_column not in price_data.columns:
            raise ValueError(f"Asset {asset_column} not found in price data")
        
        price_series = price_data[asset_column]
        
        # Align signals with price data
        entry_signals = entry_signals.reindex(price_series.index, fill_value=False)
        exit_signals = exit_signals.reindex(price_series.index, fill_value=False)
        
        # Generate positions series for time in market calculation
        positions = pd.Series(np.nan, index=price_series.index, dtype=float)
        positions[entry_signals] = 1.0
        positions[exit_signals] = 0.0
        positions = positions.ffill().fillna(0.0) # Forward fill positions and fill leading NaNs with 0

        if VECTORBT_AVAILABLE:
            portfolio = self._run_vectorbt_backtest(price_series, entry_signals, exit_signals)
            returns = portfolio.returns()
            equity_curve = portfolio.value()
            trades_count = portfolio.trades.count()
        else:
            portfolio, returns, equity_curve, trades_count = self._run_manual_backtest(
                price_series, entry_signals, exit_signals
            )
        
        # Calculate time in market
        time_in_market = positions.mean()
        
        # Calculate basic metrics
        metrics = self._calculate_basic_metrics(returns, equity_curve)
        
        return BacktestResult(
            strategy_name="Strategy",
            portfolio=portfolio,
            entry_signals=entry_signals,
            exit_signals=exit_signals,
            returns=returns,
            equity_curve=equity_curve,
            metrics=metrics,
            config=self.config.__dict__,
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
            freq=self.config.frequency,
            init_cash=self.config.initial_capital,
            fees=self.config.fees,
            slippage=self.config.slippage
        )
        return portfolio
    
    def _run_manual_backtest(self, 
                           price_series: pd.Series,
                           entry_signals: pd.Series,
                           exit_signals: pd.Series) -> Tuple[None, pd.Series, pd.Series, int]:
        """Run manual backtest when vectorbt is not available."""
        cash = self.config.initial_capital
        position = 0
        portfolio_value = []
        trades_count = 0
        
        for i, (date, price) in enumerate(price_series.items()):
            if pd.isna(price):
                portfolio_value.append(cash + position * price if not pd.isna(price) else cash)
                continue
            
            # Check for entry signal
            if entry_signals.iloc[i] and position == 0:
                # Buy with available cash (minus fees)
                shares_to_buy = cash / price * (1 - self.config.fees)
                position = shares_to_buy
                cash = 0
                trades_count += 1
            
            # Check for exit signal
            elif exit_signals.iloc[i] and position > 0:
                # Sell position
                cash = position * price * (1 - self.config.fees)
                position = 0
                trades_count += 1
            
            # Calculate portfolio value
            current_value = cash + position * price
            portfolio_value.append(current_value)
        
        # Convert to pandas Series
        equity_curve = pd.Series(portfolio_value, index=price_series.index)
        returns = equity_curve.pct_change().fillna(0)
        
        return None, returns, equity_curve, trades_count
    
    def _calculate_basic_metrics(self, returns: pd.Series, equity_curve: pd.Series) -> Dict:
        """Calculate basic performance metrics."""
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        # Use standard trading days per year (252) for financial returns
        # This excludes weekends and holidays, standard in finance industry
        annual_factor = 252

        # Annualized metrics
        cagr = (1 + total_return) ** (annual_factor / len(returns)) - 1
        
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
    
    def run_multi_asset_backtest(self, 
                                price_data: pd.DataFrame,
                                signals_dict: Dict[str, Tuple[pd.Series, pd.Series]]) -> Dict[str, BacktestResult]:
        """Run backtest across multiple assets."""
        results = {}
        
        for asset, (entry_signals, exit_signals) in signals_dict.items():
            if asset in price_data.columns:
                result = self.run_backtest(price_data, entry_signals, exit_signals, asset)
                result.strategy_name = f"Strategy_{asset}"
                results[asset] = result
            else:
                print(f"Warning: Asset {asset} not found in price data")
        
        return results
    
    def run_walk_forward_backtest(self, 
                                 price_data: pd.DataFrame,
                                 strategy_func,
                                 train_window: int = 252,  # 1 year for weekly data
                                 test_window: int = 52,    # 1 quarter for weekly data
                                 step_size: int = 13) -> Dict:   # Monthly steps
        """Run walk-forward analysis."""
        results = []
        
        for start_idx in range(0, len(price_data) - train_window - test_window, step_size):
            # Define train and test periods
            train_end = start_idx + train_window
            test_end = train_end + test_window
            
            train_data = price_data.iloc[start_idx:train_end]
            test_data = price_data.iloc[train_end:test_end]
            
            # Train strategy and generate signals
            try:
                entry_signals, exit_signals = strategy_func(train_data, test_data)
                
                # Run backtest on test data
                result = self.run_backtest(test_data, entry_signals, exit_signals)
                
                results.append({
                    'train_start': train_data.index[0],
                    'train_end': train_data.index[-1],
                    'test_start': test_data.index[0],
                    'test_end': test_data.index[-1],
                    'result': result
                })
                
            except Exception as e:
                print(f"Error in walk-forward period {start_idx}: {e}")
                continue
        
        return {
            'individual_results': results,
            'summary': self._summarize_walk_forward_results(results)
        }
    
    def _summarize_walk_forward_results(self, results: list) -> Dict:
        """Summarize walk-forward analysis results."""
        if not results:
            return {}
        
        # Extract metrics from all periods
        total_returns = [r['result'].metrics['total_return'] for r in results]
        sharpe_ratios = [r['result'].metrics['sharpe_ratio'] for r in results]
        max_drawdowns = [r['result'].metrics['max_drawdown'] for r in results]
        
        return {
            'num_periods': len(results),
            'avg_return': np.mean(total_returns),
            'std_return': np.std(total_returns),
            'avg_sharpe': np.mean(sharpe_ratios),
            'avg_max_dd': np.mean(max_drawdowns),
            'win_rate': sum(1 for r in total_returns if r > 0) / len(total_returns),
            'best_period_return': max(total_returns),
            'worst_period_return': min(total_returns)
        } 