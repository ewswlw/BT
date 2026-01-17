"""
Comprehensive Multi-Iteration CDX Trading Strategy Development
Goal: Beat IG CDX buy-and-hold by at least 2.5% annualized return (3.81% CAGR target)
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from pathlib import Path
import sys
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from strategies.base_strategy import BaseStrategy
from core.backtest_engine import BacktestEngine

OUTPUT_DIR = Path(__file__).parent / "outputs" / "comprehensive_iterations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import from previous script
exec(open('iterative_strategy_development.py').read().split('def main():')[0])

class Iteration2Strategy(BaseStrategy):
    """
    Iteration 2: Aggressive Momentum + Mean Reversion Hybrid
    Lessons from Iteration 1: Too conservative, need more active trading
    New approach: Combine momentum and mean reversion with lower thresholds
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration2_MomentumMeanReversion"
        self.description = "Aggressive momentum + mean reversion hybrid"

    def get_required_features(self):
        return []  # Will check manually

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """Aggressive momentum + mean reversion signals"""

        features = test_features.copy()

        # Signal 1: IG CDX spread momentum
        # Buy when spreads are tightening (index improving)
        if 'ig_cdx_ret_21d' in features.columns:
            spread_momentum = features['ig_cdx_ret_21d'] < 0  # Tightening
        else:
            spread_momentum = pd.Series(False, index=features.index)

        # Signal 2: Short-term mean reversion
        # Buy when spreads widened short-term but are mean reverting
        if 'ig_cdx_zscore_21d' in features.columns:
            mean_reversion = (features['ig_cdx_zscore_21d'] > 0.3) & (features['ig_cdx_ret_5d'] < 0)
        else:
            mean_reversion = pd.Series(False, index=features.index)

        # Signal 3: VIX regime (low volatility)
        if 'vix' in features.columns:
            vix_ma = features['vix'].rolling(21).mean()
            low_vol = features['vix'] < vix_ma
        else:
            low_vol = pd.Series(True, index=features.index)

        # Signal 4: Equity momentum (SPX rising)
        if 'spx_ret_21d' in features.columns:
            equity_momentum = features['spx_ret_21d'] > 0
        else:
            equity_momentum = pd.Series(True, index=features.index)

        # Signal 5: Yield curve (steepening or flat)
        if 'us_2s10s_spread' in features.columns:
            curve_signal = features['us_2s10s_spread'] > 0.5
        else:
            curve_signal = pd.Series(True, index=features.index)

        # Combine signals - require 2 out of 5 for entry (more aggressive)
        signal_sum = (spread_momentum.astype(int) +
                     mean_reversion.astype(int) +
                     low_vol.astype(int) +
                     equity_momentum.astype(int) +
                     curve_signal.astype(int))

        entry_signals = signal_sum >= 2

        # Exit when only 0-1 signals
        exit_signals = signal_sum <= 1

        return entry_signals.fillna(False), exit_signals.fillna(False)


class Iteration3Strategy(BaseStrategy):
    """
    Iteration 3: High Frequency Tactical - VIX and Spread Regime
    Focus on volatility regime switching and credit spread dynamics
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration3_VIXSpreadRegime"
        self.description = "VIX regime + spread dynamics tactical"

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """VIX regime and spread signals"""

        features = test_features.copy()

        # Primary signal: VIX regime
        if 'vix' in features.columns:
            vix_percentile_21 = features['vix'].rolling(63).apply(
                lambda x: stats.percentileofscore(x, x.iloc[-1]) if len(x) > 0 else 50
            ) / 100

            # Buy when VIX in bottom 60% (low vol regime)
            vix_signal = vix_percentile_21 < 0.6
        else:
            vix_signal = pd.Series(True, index=features.index)

        # Secondary signal: Spread not widening rapidly
        if 'ig_cdx' in features.columns:
            spread_change = features['ig_cdx'].pct_change(5)
            spread_signal = spread_change < 0.05  # Not widening > 5%
        else:
            spread_signal = pd.Series(True, index=features.index)

        # Tertiary signal: Economic growth not collapsing
        if 'us_growth_surprises' in features.columns:
            growth_signal = features['us_growth_surprises'] > -0.5
        else:
            growth_signal = pd.Series(True, index=features.index)

        # Financial conditions not tightening sharply
        if 'us_bloomberg_fci' in features.columns:
            fci_change = features['us_bloomberg_fci'].diff(5)
            fci_signal = fci_change > -0.2
        else:
            fci_signal = pd.Series(True, index=features.index)

        # Combine: all 4 must be true (defensive)
        entry_signals = vix_signal & spread_signal & growth_signal & fci_signal

        # Exit when VIX spikes OR spreads widen
        if 'vix' in features.columns and 'ig_cdx' in features.columns:
            vix_spike = vix_percentile_21 > 0.8
            spread_widen = spread_change > 0.1
            exit_signals = vix_spike | spread_widen
        else:
            exit_signals = ~entry_signals

        return entry_signals.fillna(False), exit_signals.fillna(False)


class Iteration4Strategy(BaseStrategy):
    """
    Iteration 4: Economic Regime Rotation
    Focus on economic cycle and credit sensitivity
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration4_EconomicRegime"
        self.description = "Economic regime-based rotation"

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """Economic regime rotation signals"""

        features = test_features.copy()

        # Regime 1: LEI growth (leading economic indicator)
        if 'us_lei_yoy' in features.columns:
            lei_improving = features['us_lei_yoy'] > features['us_lei_yoy'].shift(21)
        else:
            lei_improving = pd.Series(True, index=features.index)

        # Regime 2: Economic surprises positive
        if 'us_growth_surprises' in features.columns and 'us_inflation_surprises' in features.columns:
            growth_positive = features['us_growth_surprises'] > 0
            inflation_controlled = features['us_inflation_surprises'] < 0.5
            econ_signal = growth_positive & inflation_controlled
        else:
            econ_signal = pd.Series(True, index=features.index)

        # Regime 3: Yield curve
        if 'us_2s10s_spread' in features.columns:
            curve_steepening = features['us_2s10s_spread'] > features['us_2s10s_spread'].shift(21)
            not_inverted = features['us_2s10s_spread'] > -0.2
            curve_signal = curve_steepening | not_inverted
        else:
            curve_signal = pd.Series(True, index=features.index)

        # Regime 4: Credit spread environment
        if 'ig_cdx' in features.columns and 'hy_cdx' in features.columns:
            # IG vs HY ratio - when IG relatively attractive
            ig_hy_ratio = features['ig_cdx'] / features['hy_cdx']
            ig_hy_ma = ig_hy_ratio.rolling(63).mean()
            credit_signal = ig_hy_ratio < ig_hy_ma * 1.1  # IG not too expensive vs HY
        else:
            credit_signal = pd.Series(True, index=features.index)

        # Entry when 3 out of 4 regimes favorable
        signal_sum = (lei_improving.astype(int) +
                     econ_signal.astype(int) +
                     curve_signal.astype(int) +
                     credit_signal.astype(int))

        entry_signals = signal_sum >= 3

        # Exit when less than 2 regimes favorable
        exit_signals = signal_sum < 2

        return entry_signals.fillna(False), exit_signals.fillna(False)


class Iteration5Strategy(BaseStrategy):
    """
    Iteration 5: Carry + Momentum Combo
    Focus on capturing carry with momentum overlay
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration5_CarryMomentum"
        self.description = "Carry strategy with momentum overlay"

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """Carry and momentum signals"""

        features = test_features.copy()

        # Carry signal: Index return in past month
        if 'us_ig_cdx_er_index' in test_data.columns:
            monthly_return = test_data['us_ig_cdx_er_index'].pct_change(21)
            carry_positive = monthly_return > 0
        else:
            carry_positive = pd.Series(True, index=features.index)

        # Momentum: Trend over 3 months
        if 'us_ig_cdx_er_index' in test_data.columns:
            momentum_3m = test_data['us_ig_cdx_er_index'].pct_change(63)
            momentum_positive = momentum_3m > 0
        else:
            momentum_positive = pd.Series(True, index=features.index)

        # Volatility filter: Low volatility regime
        if 'ig_cdx' in features.columns:
            spread_vol = features['ig_cdx'].pct_change().rolling(21).std()
            spread_vol_ma = spread_vol.rolling(63).mean()
            low_vol_regime = spread_vol < spread_vol_ma * 1.2
        else:
            low_vol_regime = pd.Series(True, index=features.index)

        # Market conditions: Equity market not crashing
        if 'spx_tr' in features.columns:
            spx_dd = (features['spx_tr'] / features['spx_tr'].rolling(252).max()) - 1
            equity_ok = spx_dd > -0.15  # Not in 15% drawdown
        else:
            equity_ok = pd.Series(True, index=features.index)

        # Entry: Carry + Momentum + low vol + equity ok
        entry_signals = carry_positive & momentum_positive & low_vol_regime & equity_ok

        # Exit: Any signal turns negative
        exit_signals = ~entry_signals

        return entry_signals.fillna(False), exit_signals.fillna(False)


class Iteration6Strategy(BaseStrategy):
    """
    Iteration 6: Pure Trend Following
    Simple but effective trend following on the index itself
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration6_TrendFollowing"
        self.description = "Pure trend following on IG CDX index"

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """Pure trend following"""

        # Use the index price directly
        if 'us_ig_cdx_er_index' in test_data.columns:
            price = test_data['us_ig_cdx_er_index']

            # Multiple timeframe MAs
            ma_20 = price.rolling(20).mean()
            ma_50 = price.rolling(50).mean()
            ma_100 = price.rolling(100).mean()

            # Entry: All trends aligned
            fast_above_med = ma_20 > ma_50
            fast_above_slow = ma_20 > ma_100
            med_above_slow = ma_50 > ma_100

            entry_signals = fast_above_med & fast_above_slow & med_above_slow

            # Exit: Fast below medium
            exit_signals = ma_20 < ma_50

        else:
            entry_signals = pd.Series(False, index=test_data.index)
            exit_signals = pd.Series(False, index=test_data.index)

        return entry_signals.fillna(False), exit_signals.fillna(False)


def run_all_iterations():
    """Run all iterations and compare"""

    print("\n" + "="*80)
    print("COMPREHENSIVE MULTI-ITERATION STRATEGY DEVELOPMENT")
    print("="*80)

    # Load data
    data = DataLoader.load_data()
    baseline_results, baseline_prices = BaselineAnalysis.calculate_baseline(data)
    features = FeatureEngineering.create_features(data)
    features = features.fillna(method='ffill').fillna(method='bfill')

    common_idx = data.index.intersection(features.index)
    data = data.loc[common_idx]
    features = features.loc[common_idx]

    # Define all iterations
    iterations = [
        (2, Iteration2Strategy, {'name': 'Iteration2_MomentumMeanReversion', 'holding_period_days': 7}),
        (3, Iteration3Strategy, {'name': 'Iteration3_VIXSpreadRegime', 'holding_period_days': 7}),
        (4, Iteration4Strategy, {'name': 'Iteration4_EconomicRegime', 'holding_period_days': 7}),
        (5, Iteration5Strategy, {'name': 'Iteration5_CarryMomentum', 'holding_period_days': 7}),
        (6, Iteration6Strategy, {'name': 'Iteration6_TrendFollowing', 'holding_period_days': 7}),
    ]

    results = []

    for iter_num, strategy_class, config in iterations:
        try:
            result, validation, output_dir = run_iteration(
                iter_num, strategy_class, config, data, features,
                baseline_results, baseline_prices
            )

            results.append({
                'iteration': iter_num,
                'name': config['name'],
                'cagr': result.metrics['cagr'],
                'outperformance': validation['outperformance'],
                'sharpe': result.metrics['sharpe_ratio'],
                'max_dd': result.metrics['max_drawdown'],
                'trades': result.trades_count,
                'target_achieved': validation['target_achieved']
            })
        except Exception as e:
            print(f"\n❌ Iteration {iter_num} failed: {e}")
            continue

    # Summary report
    print("\n" + "="*80)
    print("SUMMARY OF ALL ITERATIONS")
    print("="*80)
    print(f"\nBaseline CAGR: {baseline_results['cagr']*100:.2f}%")
    print(f"Target CAGR: {(baseline_results['cagr'] + 0.025)*100:.2f}%\n")

    summary_df = pd.DataFrame(results)

    if len(summary_df) > 0:
        summary_df['cagr_pct'] = summary_df['cagr'] * 100
        summary_df['outperf_pct'] = summary_df['outperformance'] * 100
        summary_df['max_dd_pct'] = summary_df['max_dd'] * 100

        print(summary_df[['iteration', 'name', 'cagr_pct', 'outperf_pct',
                         'sharpe', 'max_dd_pct', 'trades', 'target_achieved']].to_string(index=False))

        # Save summary
        summary_df.to_csv(OUTPUT_DIR / "all_iterations_summary.csv", index=False)

        # Check if any achieved target
        winners = summary_df[summary_df['target_achieved']]

        if len(winners) > 0:
            print("\n" + "="*80)
            print("🎯 SUCCESS! The following strategies achieved the target:")
            print("="*80)
            for idx, row in winners.iterrows():
                print(f"  Iteration {row['iteration']}: {row['name']}")
                print(f"    CAGR: {row['cagr']*100:.2f}%")
                print(f"    Outperformance: {row['outperformance']*100:.2f}%\n")
        else:
            print("\n⚠️ No strategy achieved the target yet. Need more iterations...")

            # Show best performer
            best_idx = summary_df['outperformance'].idxmax()
            best = summary_df.loc[best_idx]
            print(f"\nBest performer:")
            print(f"  Iteration {best['iteration']}: {best['name']}")
            print(f"  CAGR: {best['cagr']*100:.2f}%")
            print(f"  Outperformance: {best['outperformance']*100:.2f}%")
            print(f"  Still need: {(0.025 - best['outperformance'])*100:.2f}% more")

    print("\n" + "="*80)

if __name__ == "__main__":
    run_all_iterations()
