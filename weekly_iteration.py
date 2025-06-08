"""
AGGRESSIVE Weekly CAD IG Backtesting - TARGET: >64.07% TOTAL RETURN
==================================================================

CRITICAL MISSION: Achieve >64.07% total return (2x buy-and-hold 32.04%)
Current Best Known: 18.10% - NEED 3.5x IMPROVEMENT

CONSTRAINTS:
- No leverage allowed
- No short selling
- Long-only strategies (0-100% allocation)
- Weekly rebalancing

APPROACH: Maximum aggression within constraints using:
- Extreme position sizing (0-100% range)
- Binary risk-on/risk-off strategies  
- Volatility breakout timing
- Multi-signal ensembles
"""

import pandas as pd
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')

# Import VectorBT and QuantStats
try:
    import vectorbt as vbt
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "vectorbt"])
    import vectorbt as vbt

try:
    import quantstats as qs
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "quantstats"])
    import quantstats as qs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CADIGAggressiveBacktester:
    """
    AGGRESSIVE CAD IG Backtester targeting >64.07% total return
    """
    
    def __init__(self):
        logger.info("="*80)
        logger.info("AGGRESSIVE CAD IG BACKTESTER")
        logger.info("TARGET: >64.07% TOTAL RETURN (2x buy-and-hold)")
        logger.info("CURRENT BEST: 18.10% - NEED 3.5x IMPROVEMENT")
        logger.info("="*80)
        
        self.target_total_return = 64.07
        self.best_total_return = -np.inf
        self.best_strategy = None
        self.results = []
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load and prepare data"""
        try:
            logger.info("Loading data...")
            self.data = pd.read_csv('../data_pipelines/data_processed/with_er_daily.csv')
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data.set_index('Date', inplace=True)
            self.data.fillna(method='ffill', inplace=True)
            
            # Convert to weekly
            self.weekly_data = self.data.resample('W-FRI').last().iloc[1:]
            self.returns = self.weekly_data['cad_ig_er_index'].pct_change().dropna()
            
            logger.info(f"Data loaded: {len(self.weekly_data)} weeks")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def binary_extreme_strategy(self, vix_threshold=75, oas_threshold=75, 
                               low_alloc=0.05, high_alloc=0.95):
        """
        Binary extreme allocation strategy
        
        Strategy: Switch between very low and very high allocations
        based on market conditions
        """
        try:
            # Calculate percentiles
            vix_pct = self.weekly_data['vix'].rolling(52).rank(pct=True) * 100
            oas_pct = self.weekly_data['cad_oas'].rolling(52).rank(pct=True) * 100
            
            # Risk-on condition: Low VIX and High OAS (attractive spreads)
            risk_on = (vix_pct < (100 - vix_threshold)) & (oas_pct > oas_threshold)
            
            # Binary allocation
            positions = np.where(risk_on, high_alloc, low_alloc)
            
            return pd.Series(positions, index=self.weekly_data.index)
            
        except Exception as e:
            logger.error(f"Error in binary extreme strategy: {e}")
            return pd.Series(0.5, index=self.weekly_data.index)

    def volatility_carry_strategy(self, vol_threshold=1.5, carry_alloc=0.9):
        """
        Volatility carry strategy - high allocation in low vol periods
        """
        try:
            # Calculate rolling volatility
            vol_ratio = (self.returns.rolling(4).std() / 
                        self.returns.rolling(52).std())
            
            # High allocation when volatility is low (carry trade)
            positions = np.where(vol_ratio < (1/vol_threshold), carry_alloc, 0.1)
            
            return pd.Series(positions, index=self.weekly_data.index)
            
        except Exception as e:
            logger.error(f"Error in volatility carry strategy: {e}")
            return pd.Series(0.5, index=self.weekly_data.index)

    def momentum_breakout_strategy(self, lookback=4, amplification=10.0):
        """
        Extreme momentum strategy with high amplification
        """
        try:
            # Calculate credit spread momentum (negative = bullish for credit)
            momentum = -self.weekly_data['cad_oas'].pct_change(lookback)
            
            # Normalize and amplify
            momentum_norm = momentum / momentum.rolling(52).std()
            
            # Apply extreme amplification
            raw_positions = 0.5 + (amplification * momentum_norm * 0.05)
            positions = np.clip(raw_positions, 0, 1)
            
            return pd.Series(positions, index=self.weekly_data.index)
            
        except Exception as e:
            logger.error(f"Error in momentum breakout strategy: {e}")
            return pd.Series(0.5, index=self.weekly_data.index)

    def economic_regime_strategy(self, growth_threshold=0.5):
        """
        Economic regime-based extreme allocation
        """
        try:
            # Economic growth momentum
            growth_mom = self.weekly_data['us_growth_surprises'].rolling(8).mean()
            
            # Binary allocation based on growth regime
            positions = np.where(growth_mom > growth_threshold, 0.9, 0.1)
            
            return pd.Series(positions, index=self.weekly_data.index)
            
        except Exception as e:
            logger.error(f"Error in economic regime strategy: {e}")
            return pd.Series(0.5, index=self.weekly_data.index)

    def run_backtest(self, positions, strategy_name):
        """Run VectorBT backtest"""
        try:
            # Align data
            common_index = positions.index.intersection(self.returns.index)
            if len(common_index) == 0:
                return None
                
            aligned_positions = positions.loc[common_index]
            aligned_returns = self.returns.loc[common_index]
            
            # Create price series
            prices = (1 + aligned_returns).cumprod()
            
            # Create signals from position changes
            position_changes = aligned_positions.diff().fillna(0)
            entries = position_changes > 0.1  # Large increases
            exits = position_changes < -0.1   # Large decreases
            
            # Run backtest
            portfolio = vbt.Portfolio.from_signals(
                close=prices,
                entries=entries,
                exits=exits,
                size=aligned_positions,
                size_type='percent',
                init_cash=100,
                freq='W'
            )
            
            return portfolio
            
        except Exception as e:
            logger.error(f"Error running backtest for {strategy_name}: {e}")
            return None

    def analyze_performance(self, portfolio, strategy_name):
        """Analyze portfolio performance"""
        try:
            if portfolio is None:
                return {}
            
            returns = portfolio.returns()
            if len(returns) == 0 or returns.isnull().all():
                return {}
            
            # Calculate total return
            total_return = (1 + returns).prod() - 1
            total_return_pct = total_return * 100
            
            # Other metrics
            cagr = qs.stats.cagr(returns, periods=52)
            sharpe = qs.stats.sharpe(returns, periods=52)
            max_dd = qs.stats.max_drawdown(returns)
            
            metrics = {
                'strategy_name': strategy_name,
                'total_return_pct': total_return_pct,
                'cagr': cagr,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'target_ratio': total_return_pct / self.target_total_return
            }
            
            # Check target achievement
            if total_return_pct >= self.target_total_return:
                logger.info(f"🎯 TARGET ACHIEVED! {strategy_name}: {total_return_pct:.2f}%")
            else:
                shortfall = self.target_total_return - total_return_pct
                logger.info(f"{strategy_name}: {total_return_pct:.2f}% (Shortfall: {shortfall:.2f}%)")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing {strategy_name}: {e}")
            return {}

    def run_aggressive_iteration(self):
        """Run aggressive strategy iteration"""
        try:
            logger.info("STARTING AGGRESSIVE ITERATION...")
            all_results = []
            
            # Strategy 1: Binary Extreme with various parameters
            logger.info("Testing Binary Extreme strategies...")
            binary_params = [
                (70, 70, 0.05, 0.95), (75, 75, 0.03, 0.97), (80, 80, 0.02, 0.98),
                (85, 85, 0.01, 0.99), (90, 90, 0.01, 0.99), (95, 95, 0.01, 0.99)
            ]
            
            for vix_t, oas_t, low_a, high_a in binary_params:
                try:
                    strategy_name = f"binary_extreme_{vix_t}_{oas_t}_{low_a}_{high_a}"
                    positions = self.binary_extreme_strategy(vix_t, oas_t, low_a, high_a)
                    portfolio = self.run_backtest(positions, strategy_name)
                    
                    if portfolio is not None:
                        metrics = self.analyze_performance(portfolio, strategy_name)
                        if metrics:
                            all_results.append(metrics)
                            
                            if metrics['total_return_pct'] > self.best_total_return:
                                self.best_total_return = metrics['total_return_pct']
                                self.best_strategy = strategy_name
                                logger.info(f"NEW BEST: {strategy_name} - {metrics['total_return_pct']:.2f}%")
                                
                except Exception as e:
                    logger.warning(f"Failed binary extreme {vix_t}_{oas_t}: {e}")
                    continue
            
            # Strategy 2: Volatility Carry
            logger.info("Testing Volatility Carry strategies...")
            vol_params = [(1.2, 0.8), (1.5, 0.9), (2.0, 0.95), (2.5, 0.98), (3.0, 0.99)]
            
            for vol_t, carry_a in vol_params:
                try:
                    strategy_name = f"vol_carry_{vol_t}_{carry_a}"
                    positions = self.volatility_carry_strategy(vol_t, carry_a)
                    portfolio = self.run_backtest(positions, strategy_name)
                    
                    if portfolio is not None:
                        metrics = self.analyze_performance(portfolio, strategy_name)
                        if metrics:
                            all_results.append(metrics)
                            
                            if metrics['total_return_pct'] > self.best_total_return:
                                self.best_total_return = metrics['total_return_pct']
                                self.best_strategy = strategy_name
                                logger.info(f"NEW BEST: {strategy_name} - {metrics['total_return_pct']:.2f}%")
                                
                except Exception as e:
                    logger.warning(f"Failed vol carry {vol_t}_{carry_a}: {e}")
                    continue
            
            # Strategy 3: Momentum Breakout
            logger.info("Testing Momentum Breakout strategies...")
            momentum_params = [
                (2, 15.0), (3, 20.0), (4, 25.0), (6, 30.0), (8, 35.0)
            ]
            
            for lookback, amp in momentum_params:
                try:
                    strategy_name = f"momentum_breakout_{lookback}_{amp}"
                    positions = self.momentum_breakout_strategy(lookback, amp)
                    portfolio = self.run_backtest(positions, strategy_name)
                    
                    if portfolio is not None:
                        metrics = self.analyze_performance(portfolio, strategy_name)
                        if metrics:
                            all_results.append(metrics)
                            
                            if metrics['total_return_pct'] > self.best_total_return:
                                self.best_total_return = metrics['total_return_pct']
                                self.best_strategy = strategy_name
                                logger.info(f"NEW BEST: {strategy_name} - {metrics['total_return_pct']:.2f}%")
                                
                except Exception as e:
                    logger.warning(f"Failed momentum breakout {lookback}_{amp}: {e}")
                    continue
            
            # Strategy 4: Economic Regime
            logger.info("Testing Economic Regime strategies...")
            regime_params = [0.0, 0.2, 0.5, 0.8, 1.0]
            
            for threshold in regime_params:
                try:
                    strategy_name = f"economic_regime_{threshold}"
                    positions = self.economic_regime_strategy(threshold)
                    portfolio = self.run_backtest(positions, strategy_name)
                    
                    if portfolio is not None:
                        metrics = self.analyze_performance(portfolio, strategy_name)
                        if metrics:
                            all_results.append(metrics)
                            
                            if metrics['total_return_pct'] > self.best_total_return:
                                self.best_total_return = metrics['total_return_pct']
                                self.best_strategy = strategy_name
                                logger.info(f"NEW BEST: {strategy_name} - {metrics['total_return_pct']:.2f}%")
                                
                except Exception as e:
                    logger.warning(f"Failed economic regime {threshold}: {e}")
                    continue
            
            # Compile results
            if all_results:
                self.results = pd.DataFrame(all_results)
                self.results = self.results.sort_values('total_return_pct', ascending=False)
                
                logger.info("\n" + "="*80)
                logger.info(f"AGGRESSIVE ITERATION COMPLETED!")
                logger.info(f"Tested {len(all_results)} strategies")
                logger.info(f"Best Total Return: {self.best_total_return:.2f}%")
                logger.info(f"Target: {self.target_total_return:.2f}%")
                
                # Check target achievement
                if self.best_total_return >= self.target_total_return:
                    logger.info("🎯 SUCCESS! TARGET ACHIEVED!")
                else:
                    shortfall = self.target_total_return - self.best_total_return
                    improvement_needed = self.target_total_return / max(self.best_total_return, 0.01)
                    logger.info(f"Target not achieved. Shortfall: {shortfall:.2f}%")
                    logger.info(f"Need {improvement_needed:.2f}x improvement")
                
                # Show top 10
                logger.info("\nTOP 10 STRATEGIES:")
                logger.info("="*80)
                for i, (_, row) in enumerate(self.results.head(10).iterrows()):
                    status = "✅" if row['total_return_pct'] >= self.target_total_return else "❌"
                    logger.info(f"{i+1:2d}. {status} {row['strategy_name']:30s} | "
                              f"Return: {row['total_return_pct']:7.2f}% | "
                              f"Sharpe: {row['sharpe_ratio']:6.3f} | "
                              f"Target: {row['target_ratio']:6.3f}x")
                
                return self.results
            else:
                logger.warning("No successful strategies completed")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error in aggressive iteration: {e}")
            raise


def main():
    """Main execution"""
    try:
        logger.info("="*80)
        logger.info("AGGRESSIVE CAD IG BACKTESTING")
        logger.info("TARGET: >64.07% TOTAL RETURN")
        logger.info("="*80)
        
        # Run aggressive backtesting
        backtester = CADIGAggressiveBacktester()
        results = backtester.run_aggressive_iteration()
        
        # Final summary
        if not results.empty:
            best_return = backtester.best_total_return
            target = backtester.target_total_return
            
            logger.info("\n" + "="*80)
            logger.info("FINAL RESULTS SUMMARY")
            logger.info("="*80)
            logger.info(f"Best Strategy: {backtester.best_strategy}")
            logger.info(f"Best Total Return: {best_return:.2f}%")
            logger.info(f"Target: {target:.2f}%")
            logger.info(f"Achievement Ratio: {best_return/target:.3f}")
            
            if best_return >= target:
                logger.info("🎯 MISSION ACCOMPLISHED!")
            else:
                shortfall = target - best_return
                logger.info(f"Mission incomplete. Need {shortfall:.2f}% more.")
        
        logger.info("\n" + "="*80)
        logger.info("AGGRESSIVE BACKTESTING COMPLETED")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main() 