"""
Advanced CDX Trading Strategy Iterations
New approach: Stay invested more, avoid only the worst periods
Goal: 3.81% CAGR (beat baseline 1.31% by 2.5%)
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from pathlib import Path
import sys
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from strategies.base_strategy import BaseStrategy
from core.backtest_engine import BacktestEngine

OUTPUT_DIR = Path(__file__).parent / "outputs" / "advanced_iterations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import from previous script
exec(open('iterative_strategy_development.py').read().split('def main():')[0])


class Iteration7Strategy(BaseStrategy):
    """
    Iteration 7: Inverse Logic - Stay Invested Except Crisis
    Buy and hold EXCEPT during clear crisis periods
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration7_InverseLogic"
        self.description = "Stay invested except during clear crises"

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """Stay invested except during crisis"""

        features = test_features.copy()

        # Define crisis conditions (when to EXIT)
        crisis_signals = []

        # Crisis 1: VIX spike above 90th percentile
        if 'vix' in features.columns:
            vix_percentile = features['vix'].rolling(252).apply(
                lambda x: stats.percentileofscore(x, x.iloc[-1]) if len(x) > 20 else 50
            ) / 100
            vix_crisis = vix_percentile > 0.9
            crisis_signals.append(vix_crisis)

        # Crisis 2: Spread widening rapidly (top 95th percentile)
        if 'ig_cdx' in features.columns:
            spread_change_5d = features['ig_cdx'].pct_change(5)
            spread_percentile = spread_change_5d.rolling(252).apply(
                lambda x: stats.percentileofscore(x, x.iloc[-1]) if len(x) > 20 else 50
            ) / 100
            spread_crisis = spread_percentile > 0.95
            crisis_signals.append(spread_crisis)

        # Crisis 3: SPX drawdown > 10%
        if 'spx_tr' in features.columns:
            spx_dd = (features['spx_tr'] / features['spx_tr'].rolling(252).max()) - 1
            equity_crisis = spx_dd < -0.10
            crisis_signals.append(equity_crisis)

        # Crisis 4: Financial conditions tightening severely
        if 'us_bloomberg_fci' in features.columns:
            fci_change = features['us_bloomberg_fci'].diff(21)
            fci_crisis = fci_change < features['us_bloomberg_fci'].diff(21).rolling(252).quantile(0.05)
            crisis_signals.append(fci_crisis)

        # Exit during crisis (any 2 crisis signals)
        if len(crisis_signals) > 0:
            crisis_count = sum(s.astype(int) for s in crisis_signals)
            in_crisis = crisis_count >= 2
        else:
            in_crisis = pd.Series(False, index=features.index)

        # Entry = NOT in crisis (stay invested by default)
        entry_signals = ~in_crisis

        # Exit = in crisis
        exit_signals = in_crisis

        return entry_signals.fillna(True), exit_signals.fillna(False)


class Iteration8Strategy(BaseStrategy):
    """
    Iteration 8: Contrarian - Buy the Dip
    Buy when spreads widen (contrarian to panic)
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration8_Contrarian"
        self.description = "Buy when spreads widen (contrarian)"

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """Contrarian - buy when others sell"""

        features = test_features.copy()

        # Buy when spreads have widened significantly
        if 'ig_cdx' in features.columns:
            # Z-score of spreads
            spread_mean = features['ig_cdx'].rolling(126).mean()
            spread_std = features['ig_cdx'].rolling(126).std()
            spread_z = (features['ig_cdx'] - spread_mean) / spread_std

            # Buy when spreads wide (z > 0.5) - credit is cheap
            buy_signal = spread_z > 0.3

        else:
            buy_signal = pd.Series(True, index=features.index)

        # But not during extreme crisis
        if 'vix' in features.columns:
            vix_extreme = features['vix'] > features['vix'].rolling(252).quantile(0.95)
            buy_signal = buy_signal & ~vix_extreme

        # Exit when spreads compress back to normal
        if 'ig_cdx' in features.columns:
            exit_signal = spread_z < -0.2  # Spreads tight
        else:
            exit_signal = ~buy_signal

        return buy_signal.fillna(False), exit_signal.fillna(False)


class Iteration9Strategy(BaseStrategy):
    """
    Iteration 9: Volatility Targeting
    Increase exposure during low vol, decrease during high vol
    (Implemented as binary: IN during low vol, OUT during high vol)
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration9_VolatilityTargeting"
        self.description = "Stay in during low realized volatility"

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """Volatility targeting"""

        features = test_features.copy()

        # Calculate realized volatility of the index
        if 'us_ig_cdx_er_index' in test_data.columns:
            returns = test_data['us_ig_cdx_er_index'].pct_change()
            realized_vol_21d = returns.rolling(21).std() * np.sqrt(252)

            # Calculate volatility percentile
            vol_percentile = realized_vol_21d.rolling(252).apply(
                lambda x: stats.percentileofscore(x, x.iloc[-1]) if len(x) > 20 else 50
            ) / 100

            # Stay invested during low to medium vol (bottom 70%)
            entry_signals = vol_percentile < 0.70

            # Exit during high vol (top 30%)
            exit_signals = vol_percentile > 0.70

        else:
            entry_signals = pd.Series(True, index=features.index)
            exit_signals = pd.Series(False, index=features.index)

        return entry_signals.fillna(True), exit_signals.fillna(False)


class Iteration10Strategy(BaseStrategy):
    """
    Iteration 10: Risk Parity Inspired - Balance based on recent volatility
    Stay out when recent sharp moves suggest instability
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration10_RiskParity"
        self.description = "Risk parity approach - avoid unstable periods"

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """Risk parity signals"""

        features = test_features.copy()

        stable_signals = []

        # Stability signal 1: Spread volatility not elevated
        if 'ig_cdx' in features.columns:
            spread_vol = features['ig_cdx'].pct_change().rolling(21).std()
            spread_vol_ma = spread_vol.rolling(63).mean()
            spread_stable = spread_vol < spread_vol_ma * 1.5
            stable_signals.append(spread_stable)

        # Stability signal 2: VIX not spiking
        if 'vix' in features.columns:
            vix_change = features['vix'].diff(5)
            vix_stable = vix_change < features['vix'].diff(5).rolling(63).quantile(0.80)
            stable_signals.append(vix_stable)

        # Stability signal 3: Rate volatility contained
        if 'rate_vol' in features.columns:
            rate_vol_high = features['rate_vol'] > features['rate_vol'].rolling(63).quantile(0.80)
            stable_signals.append(~rate_vol_high)

        # Stability signal 4: No sharp equity drawdowns
        if 'spx_tr' in features.columns:
            spx_dd = (features['spx_tr'] / features['spx_tr'].rolling(63).max()) - 1
            equity_stable = spx_dd > -0.07
            stable_signals.append(equity_stable)

        # Stay invested when at least 3 out of 4 stable
        if len(stable_signals) >= 3:
            stable_count = sum(s.astype(int) for s in stable_signals)
            entry_signals = stable_count >= 3
            exit_signals = stable_count < 2
        else:
            entry_signals = pd.Series(True, index=features.index)
            exit_signals = pd.Series(False, index=features.index)

        return entry_signals.fillna(True), exit_signals.fillna(False)


class Iteration11Strategy(BaseStrategy):
    """
    Iteration 11: Macro Momentum
    Follow macro trends that favor credit
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration11_MacroMomentum"
        self.description = "Follow favorable macro trends"

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """Macro momentum signals"""

        features = test_features.copy()

        macro_signals = []

        # Signal 1: Growth momentum positive
        if 'us_growth_surprises' in features.columns:
            growth_momentum = features['us_growth_surprises'].rolling(21).mean() > -0.3
            macro_signals.append(growth_momentum)

        # Signal 2: Not in recession (LEI not collapsing)
        if 'us_lei_yoy' in features.columns:
            lei_ok = features['us_lei_yoy'] > -5
            macro_signals.append(lei_ok)

        # Signal 3: Yield curve not deeply inverted
        if 'us_2s10s_spread' in features.columns:
            curve_ok = features['us_2s10s_spread'] > -0.5
            macro_signals.append(curve_ok)

        # Signal 4: Equity market in uptrend (above 200-day MA)
        if 'spx_tr' in features.columns:
            spx_ma200 = features['spx_tr'].rolling(200).mean()
            equity_uptrend = features['spx_tr'] > spx_ma200 * 0.95
            macro_signals.append(equity_uptrend)

        # Signal 5: Dollar not strengthening rapidly (risk-off)
        if 'dollar_index' in features.columns:
            dollar_change = features['dollar_index'].pct_change(21)
            dollar_ok = dollar_change < 0.05
            macro_signals.append(dollar_ok)

        # Entry when majority of signals positive
        if len(macro_signals) >= 3:
            macro_count = sum(s.astype(int) for s in macro_signals)
            entry_signals = macro_count >= 3
            exit_signals = macro_count < 2
        else:
            entry_signals = pd.Series(True, index=features.index)
            exit_signals = pd.Series(False, index=features.index)

        return entry_signals.fillna(True), exit_signals.fillna(False)


class Iteration12Strategy(BaseStrategy):
    """
    Iteration 12: Kitchen Sink - Combine best elements
    Use multiple filters, stay invested unless clear danger
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "Iteration12_KitchenSink"
        self.description = "Combined multi-factor approach"

    def get_required_features(self):
        return []

    def generate_signals(self, train_data, test_data, train_features, test_features):
        """Kitchen sink - everything"""

        features = test_features.copy()

        danger_signals = []

        # Danger 1: VIX extreme
        if 'vix' in features.columns:
            vix_extreme = features['vix'] > features['vix'].rolling(252).quantile(0.85)
            danger_signals.append(vix_extreme)

        # Danger 2: Spreads widening fast
        if 'ig_cdx' in features.columns:
            spread_widen = features['ig_cdx'].pct_change(5) > features['ig_cdx'].pct_change(5).rolling(252).quantile(0.90)
            danger_signals.append(spread_widen)

        # Danger 3: Financial conditions tightening
        if 'us_bloomberg_fci' in features.columns:
            fci_tighten = features['us_bloomberg_fci'].diff(10) < features['us_bloomberg_fci'].diff(10).rolling(252).quantile(0.10)
            danger_signals.append(fci_tighten)

        # Danger 4: Growth collapsing
        if 'us_growth_surprises' in features.columns:
            growth_collapse = features['us_growth_surprises'] < -0.7
            danger_signals.append(growth_collapse)

        # Danger 5: Equity crash
        if 'spx_tr' in features.columns:
            spx_dd = (features['spx_tr'] / features['spx_tr'].rolling(252).max()) - 1
            equity_crash = spx_dd < -0.12
            danger_signals.append(equity_crash)

        # Exit only if 3+ danger signals
        if len(danger_signals) >= 3:
            danger_count = sum(s.astype(int) for s in danger_signals)
            in_danger = danger_count >= 3

            entry_signals = ~in_danger
            exit_signals = in_danger
        else:
            entry_signals = pd.Series(True, index=features.index)
            exit_signals = pd.Series(False, index=features.index)

        return entry_signals.fillna(True), exit_signals.fillna(False)


def run_advanced_iterations():
    """Run advanced iterations"""

    print("\n" + "="*80)
    print("ADVANCED ITERATION STRATEGY DEVELOPMENT")
    print("="*80)

    # Load data
    data = DataLoader.load_data()
    baseline_results, baseline_prices = BaselineAnalysis.calculate_baseline(data)
    features = FeatureEngineering.create_features(data)
    features = features.fillna(method='ffill').fillna(method='bfill')

    common_idx = data.index.intersection(features.index)
    data = data.loc[common_idx]
    features = features.loc[common_idx]

    # Define advanced iterations
    iterations = [
        (7, Iteration7Strategy, {'name': 'Iteration7_InverseLogic', 'holding_period_days': 7}),
        (8, Iteration8Strategy, {'name': 'Iteration8_Contrarian', 'holding_period_days': 7}),
        (9, Iteration9Strategy, {'name': 'Iteration9_VolatilityTargeting', 'holding_period_days': 7}),
        (10, Iteration10Strategy, {'name': 'Iteration10_RiskParity', 'holding_period_days': 7}),
        (11, Iteration11Strategy, {'name': 'Iteration11_MacroMomentum', 'holding_period_days': 7}),
        (12, Iteration12Strategy, {'name': 'Iteration12_KitchenSink', 'holding_period_days': 7}),
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
                'time_in_market': result.time_in_market,
                'target_achieved': validation['target_achieved']
            })
        except Exception as e:
            print(f"\n❌ Iteration {iter_num} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary report
    print("\n" + "="*80)
    print("SUMMARY OF ADVANCED ITERATIONS")
    print("="*80)
    print(f"\nBaseline CAGR: {baseline_results['cagr']*100:.2f}%")
    print(f"Target CAGR: {(baseline_results['cagr'] + 0.025)*100:.2f}%\n")

    summary_df = pd.DataFrame(results)

    if len(summary_df) > 0:
        summary_df['cagr_pct'] = summary_df['cagr'] * 100
        summary_df['outperf_pct'] = summary_df['outperformance'] * 100
        summary_df['max_dd_pct'] = summary_df['max_dd'] * 100
        summary_df['time_in_mkt_pct'] = summary_df['time_in_market'] * 100

        print(summary_df[['iteration', 'name', 'cagr_pct', 'outperf_pct',
                         'sharpe', 'max_dd_pct', 'trades', 'time_in_mkt_pct', 'target_achieved']].to_string(index=False))

        # Save summary
        summary_df.to_csv(OUTPUT_DIR / "advanced_iterations_summary.csv", index=False)

        # Check if any achieved target
        winners = summary_df[summary_df['target_achieved']]

        if len(winners) > 0:
            print("\n" + "="*80)
            print("🎯 SUCCESS! The following strategies achieved the target:")
            print("="*80)
            for idx, row in winners.iterrows():
                print(f"  Iteration {row['iteration']}: {row['name']}")
                print(f"    CAGR: {row['cagr']*100:.2f}%")
                print(f"    Outperformance: {row['outperformance']*100:.2f}%")
                print(f"    Time in Market: {row['time_in_market']*100:.2f}%\n")
        else:
            print("\n⚠️ No strategy achieved the target yet.")

            # Show best performer
            if len(summary_df) > 0:
                best_idx = summary_df['outperformance'].idxmax()
                best = summary_df.loc[best_idx]
                print(f"\nBest performer:")
                print(f"  Iteration {best['iteration']}: {best['name']}")
                print(f"  CAGR: {best['cagr']*100:.2f}%")
                print(f"  Outperformance: {best['outperformance']*100:.2f}%")
                print(f"  Still need: {(0.025 - best['outperformance'])*100:.2f}% more")

    print("\n" + "="*80)

if __name__ == "__main__":
    run_advanced_iterations()
