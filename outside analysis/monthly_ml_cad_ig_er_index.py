#!/usr/bin/env python3
"""
monthly_ml_cad_ig_er_index.py

Monthly ML Strategy with RandomForest + GradientBoosting ensemble models.
Uses VectorBT for backtesting and generates comprehensive outputs.

Strategy Overview:
1. Monthly resampling of daily data (month-end)
2. Feature engineering with momentum, volatility, and composite indicators
3. RandomForest + GradientBoosting ensemble for signal generation
4. VectorBT portfolio construction and analysis

Key Features:
- Uses with_er_daily.csv data source
- VectorBT for portfolio backtesting
- Comprehensive output artifacts matching existing codebase patterns
- Pre-processed monthly data with monthly VectorBT frequency

Outputs saved to: outputs/
"""

import warnings
warnings.filterwarnings('ignore')

import os
from pathlib import Path
import pandas as pd
import numpy as np
import re
import math
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import spearmanr, ttest_1samp, binomtest

# VectorBT and QuantStats
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except Exception as e:
    VECTORBT_AVAILABLE = False
    raise RuntimeError("vectorbt is required for this script.") from e

try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except Exception:
    QUANTSTATS_AVAILABLE = False

# Strategy configuration
STRATEGY_NAME = "monthly_ml_cad_ig_er_index"
WARMUP_MONTHS = 60  # 5 years warmup
TOPK = 20  # Top features to select
BASE_THR = 0.60  # Base threshold for signals
META_THR = 0.65  # Meta threshold for filtering

def _resolve_input_csv(file_path: str, script_dir: Path) -> Path:
    """Resolve input CSV file path with multiple fallback options."""
    # 1) as given
    candidate = Path(file_path)
    if candidate.exists():
        return candidate
    
    # 2) relative to script dir
    candidate = script_dir / file_path
    if candidate.exists():
        return candidate
    
    # 3) relative to project root
    project_root = script_dir.parent
    candidate = project_root / file_path
    if candidate.exists():
        return candidate
    
    raise FileNotFoundError(f"Could not find {file_path} in any expected location")

def load_and_prepare_data(file_path: str = '../data_pipelines/data_processed/with_er_daily.csv'):
    """Load and prepare data for monthly ML strategy."""
    script_dir = Path(__file__).parent.absolute()
    data_path = _resolve_input_csv(file_path, script_dir)
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Find date column
    date_col = None
    for c in df.columns:
        if re.search(r"(date|time|timestamp)", c, re.I):
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]
    
    # Convert date and set index
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    
    # Clean numeric columns
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",","").str.replace(" ",""), errors="ignore")
    df = df.dropna(axis=1, how="all")
    
    # Find target column
    cands = [c for c in df.columns if re.search(r'cad.*ig.*er|cad_ig_er|ig_er|enhanced.*return', c, re.I)]
    target_col = cands[0] if cands else df.columns[0]
    print(f"Using target column: {target_col}")
    
    # Convert to price series
    series = df[target_col].astype(float)
    is_returnish = (np.mean(series.dropna().between(-0.5,0.5)) > 0.8 and 
                   series.dropna().abs().median() < 0.2)
    
    if is_returnish:
        price = (1 + series.fillna(0)).cumprod() * 100.0
    else:
        price = series.copy()
    
    # Resample to monthly (month-end)
    m_px = price.resample("M").last().dropna()
    m_df = df.resample("M").last().ffill()
    m_df["PX"] = m_px
    m_df["fwd_ret_1m"] = m_df["PX"].pct_change().shift(-1)
    m_df = m_df.dropna(subset=["fwd_ret_1m"])
    
    print(f"Data prepared: {len(m_df)} monthly observations from {m_df.index[0]} to {m_df.index[-1]}")
    return m_df, target_col

def create_monthly_features(frame):
    """Create monthly features (exact implementation from ManualBacktestAnalysis)."""
    # Resample to monthly
    monthly = frame.resample('M').agg({
        'cad_ig_er_index': 'last',
        'vix': 'last',
        'us_3m_10y': 'last',
        'cad_oas': 'last',
        'us_hy_oas': 'last',
        'us_ig_oas': 'last',
        'tsx': 'last',
        'us_economic_regime': 'last',
        'us_growth_surprises': 'last',
        'us_inflation_surprises': 'last',
        'us_lei_yoy': 'last',
        'us_hard_data_surprises': 'last',
        'us_equity_revisions': 'last',
        'spx_1bf_eps': 'last',
        'spx_1bf_sales': 'last',
        'tsx_1bf_eps': 'last',
        'tsx_1bf_sales': 'last'
    })
    
    # Basic monthly return
    monthly['monthly_return'] = monthly['cad_ig_er_index'].pct_change()
    
    # Key momentum features
    for window in [1, 3, 6, 12]:
        monthly[f'momentum_{window}m'] = monthly['cad_ig_er_index'].pct_change(window)
        monthly[f'momentum_{window}m_rank'] = monthly[f'momentum_{window}m'].rolling(36).rank(pct=True)
    
    # Key volatility features
    for window in [3, 6, 12]:
        monthly[f'volatility_{window}m'] = monthly['monthly_return'].rolling(window).std()
        monthly[f'volatility_{window}m_rank'] = monthly[f'volatility_{window}m'].rolling(36).rank(pct=True)
    
    # VIX features
    monthly['vix_rank_12m'] = monthly['vix'].rolling(12).rank(pct=True)
    
    # Economic features
    monthly['econ_regime_rank'] = monthly['us_economic_regime'].rolling(24).rank(pct=True)
    
    # Composite indicators
    monthly['momentum_composite'] = (
        monthly['momentum_1m_rank'].fillna(0.5) * 0.25 + 
        monthly['momentum_3m_rank'].fillna(0.5) * 0.35 + 
        monthly['momentum_6m_rank'].fillna(0.5) * 0.25 +
        monthly['momentum_12m_rank'].fillna(0.5) * 0.15
    )
    
    monthly['risk_composite'] = (
        monthly['vix_rank_12m'].fillna(0.5) * 0.5 + 
        monthly['volatility_6m_rank'].fillna(0.5) * 0.5
    )
    
    # Target variable
    monthly['monthly_forward_return'] = monthly['cad_ig_er_index'].pct_change().shift(-1)
    monthly['monthly_target'] = (monthly['monthly_forward_return'] > 0).astype(int)
    
    return monthly.dropna()

def build_features(frame):
    """Build features with expanding window normalization."""
    # Use the monthly features from the provided implementation
    monthly_features = create_monthly_features(frame)
    
    # Select feature columns
    feature_cols = [col for col in monthly_features.columns 
                   if any(keyword in col for keyword in ['momentum', 'volatility', 'rank', 'composite'])
                   and col not in ['monthly_forward_return', 'monthly_target']
                   and not pd.isna(monthly_features[col]).all()]
    
    X = monthly_features[feature_cols].copy()
    
    # Replace inf/nan and normalize
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for c in X.columns:
        mu = X[c].expanding().mean()
        sd = X[c].expanding().std().replace(0, np.nan)
        X[c] = (X[c] - mu) / sd
    
    return X, monthly_features

def generate_optimized_ml_v2_signals(monthly_df):
    """Generate Optimized ML v2 signals (exact implementation from ManualBacktestAnalysis)."""
    
    # Use key features
    feature_cols = [col for col in monthly_df.columns 
                   if any(keyword in col for keyword in ['momentum', 'volatility', 'rank', 'composite'])
                   and col not in ['monthly_forward_return', 'monthly_target']
                   and not pd.isna(monthly_df[col]).all()]
    
    print(f"Using {len(feature_cols)} features for signal generation")
    
    # Split data
    split_idx = int(len(monthly_df) * 0.7)
    train_data = monthly_df.iloc[:split_idx]
    
    X_train = train_data[feature_cols].fillna(method='ffill').fillna(0)
    y_train = train_data['monthly_target']
    
    # Train ensemble models
    models = {
        'rf1': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'rf2': RandomForestClassifier(n_estimators=300, max_depth=8, random_state=43),
        'gb1': GradientBoostingClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42),
        'gb2': GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=43)
    }
    
    # Train models
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    # Generate predictions
    X_full = monthly_df[feature_cols].fillna(method='ffill').fillna(0)
    
    # Ensemble predictions
    predictions = np.zeros(len(X_full))
    predictions += models['rf1'].predict_proba(X_full)[:, 1] * 0.3
    predictions += models['rf2'].predict_proba(X_full)[:, 1] * 0.3
    predictions += models['gb1'].predict_proba(X_full)[:, 1] * 0.25
    predictions += models['gb2'].predict_proba(X_full)[:, 1] * 0.15
    
    # Dynamic threshold
    signals = pd.Series(0, index=monthly_df.index)
    
    for i in range(len(monthly_df)):
        current_vol = monthly_df['volatility_6m'].iloc[i]
        current_regime = monthly_df['us_economic_regime'].iloc[i]
        current_vix_rank = monthly_df['vix_rank_12m'].iloc[i]
        
        # Adaptive threshold
        if pd.isna(current_vol) or pd.isna(current_regime) or pd.isna(current_vix_rank):
            threshold = 0.6
        elif current_vol < 0.01 and current_regime > 0.8:
            threshold = 0.55
        elif current_vol > 0.02 or current_vix_rank > 0.8:
            threshold = 0.75
        else:
            threshold = 0.65
        
        signals.iloc[i] = 1 if predictions[i] > threshold else 0
    
    return signals, predictions


def create_vectorbt_portfolio(price_data, signals, strategy_name="Strategy"):
    """Create VectorBT portfolio."""
    # Convert signals to boolean entries/exits
    signals_bool = signals.astype(bool)
    entries = signals_bool & ~signals_bool.shift(1).fillna(False)
    exits = ~signals_bool & signals_bool.shift(1).fillna(False)
    
    # Create portfolio with monthly frequency
    portfolio = vbt.Portfolio.from_signals(
        close=price_data,
        entries=entries,
        exits=exits,
        freq='30D',  # 30-day frequency for monthly data
        init_cash=100.0,
        fees=0.0,
        slippage=0.0
    )
    
    return portfolio

def comprehensive_backtest_analysis(file_path='../data_pipelines/data_processed/with_er_daily.csv'):
    """Main analysis function."""
    print("="*70)
    print("MONTHLY ML CAD IG ER INDEX STRATEGY")
    print("Optimized ML v2 - RandomForest + GradientBoosting Ensemble")
    print("="*70)
    
    # Load and prepare data
    m_df, target_col = load_and_prepare_data(file_path)
    
    # Build features
    print("\nBuilding features...")
    monthly_features = create_monthly_features(m_df)
    
    # Generate optimized ML v2 signals
    print("Generating optimized ML v2 signals...")
    final_sig, predictions = generate_optimized_ml_v2_signals(monthly_features)
    
    print(f"Optimized ML v2: {final_sig.sum()} signals out of {len(final_sig)} periods ({final_sig.mean()*100:.1f}% exposure)")
    
    # Create VectorBT portfolios
    print("\nCreating VectorBT portfolios...")
    
    # Strategy portfolio
    strategy_pf = create_vectorbt_portfolio(monthly_features["cad_ig_er_index"], final_sig, "OptimizedMLv2")
    
    # Find first trade date
    first_trade_date = None
    if len(strategy_pf.trades.records_readable) > 0:
        first_trade_date = pd.to_datetime(strategy_pf.trades.records_readable.iloc[0]['Entry Timestamp'])
        print(f"• First Trade Date: {first_trade_date.strftime('%Y-%m-%d')}")
    else:
        print("• First Trade Date: No trades found")
    
    # Buy & Hold portfolio
    bh_signals = pd.Series(1, index=monthly_features.index, dtype=int)
    bh_pf = create_vectorbt_portfolio(monthly_features["cad_ig_er_index"], bh_signals, "Buy & Hold")
    
    # Perfect Foresight portfolio
    pf_signals = (monthly_features["monthly_forward_return"] > 0).astype(int)
    pf_pf = create_vectorbt_portfolio(monthly_features["cad_ig_er_index"], pf_signals, "Perfect Foresight")
    
    # Use existing outputs directory in outside analysis
    script_dir = Path(__file__).parent.absolute()
    outputs_dir = script_dir / "outputs"
    
    # Generate comprehensive outputs
    print(f"\nGenerating outputs in {outputs_dir}/...")
    
    # 1. Strategy statistics
    strategy_stats = strategy_pf.stats()
    bh_stats = bh_pf.stats()
    pf_stats = pf_pf.stats()
    
    # Save stats
    strategy_df = pd.DataFrame({'Metric': strategy_stats.index, 'Value': strategy_stats.values})
    bh_df = pd.DataFrame({'Metric': bh_stats.index, 'Value': bh_stats.values})
    pf_df = pd.DataFrame({'Metric': pf_stats.index, 'Value': pf_stats.values})
    
    strategy_df.to_csv(outputs_dir / f"{STRATEGY_NAME}_composite_stats.csv", index=False)
    bh_df.to_csv(outputs_dir / f"{STRATEGY_NAME}_buy_and_hold_stats.csv", index=False)
    pf_df.to_csv(outputs_dir / f"{STRATEGY_NAME}_perfect_foresight_stats.csv", index=False)
    
    # 2. Signals and returns
    final_sig.rename('signal').to_csv(outputs_dir / f"{STRATEGY_NAME}_signals.csv")
    
    strategy_returns = strategy_pf.returns().dropna()
    bh_returns = bh_pf.returns().dropna()
    pf_returns = pf_pf.returns().dropna()
    
    strategy_returns.rename('strategy_returns').to_csv(outputs_dir / f"{STRATEGY_NAME}_returns.csv")
    bh_returns.rename('benchmark_returns').to_csv(outputs_dir / f"{STRATEGY_NAME}_benchmark_returns.csv")
    pf_returns.rename('perfect_foresight_returns').to_csv(outputs_dir / f"{STRATEGY_NAME}_perfect_foresight_returns.csv")
    
    # 3. Equity curves
    strategy_equity = (1.0 + strategy_returns).cumprod()
    bh_equity = (1.0 + bh_returns).cumprod()
    pf_equity = (1.0 + pf_returns).cumprod()
    
    strategy_equity.rename('equity_curve').to_csv(outputs_dir / f"{STRATEGY_NAME}_equity_curve.csv")
    bh_equity.rename('benchmark_equity').to_csv(outputs_dir / f"{STRATEGY_NAME}_benchmark_equity.csv")
    pf_equity.rename('perfect_foresight_equity').to_csv(outputs_dir / f"{STRATEGY_NAME}_perfect_foresight_equity.csv")
    
    # 4. Trades analysis
    trades = strategy_pf.trades.records_readable
    trades.to_csv(outputs_dir / f"{STRATEGY_NAME}_trades.csv", index=False)
    
    # 5. Monthly signals with probabilities
    monthly_signals = pd.DataFrame({
        "Date": monthly_features.index,
        "EnsembleProb": predictions,
        "FinalSig": final_sig.values,
        "FwdRet_1m": monthly_features["monthly_forward_return"].values
    })
    try:
        monthly_signals.to_csv(outputs_dir / f"{STRATEGY_NAME}_monthly_signals.csv", index=False)
    except PermissionError:
        print(f"⚠️  Could not save monthly_signals.csv - file may be open in IDE")
        # Save with a different name
        monthly_signals.to_csv(outputs_dir / f"{STRATEGY_NAME}_monthly_signals_backup.csv", index=False)
    
    # 6. Statistical validation
    common_idx = strategy_returns.index.intersection(bh_returns.index)
    sr = strategy_returns.reindex(common_idx)
    br = bh_returns.reindex(common_idx)
    
    # t-tests
    t_stat, t_pval = ttest_1samp(sr, 0.0, nan_policy='omit')
    t_stat_excess, t_pval_excess = ttest_1samp(sr - br, 0.0, nan_policy='omit')
    
    # Trade metrics
    if len(trades) > 0:
        wins = trades[trades['PnL'] > 0]
        win_rate = len(wins) / len(trades)
        binom = binomtest(len(wins), len(trades), 0.5, alternative='greater')
    else:
        win_rate = np.nan
        binom = type('obj', (), {'pvalue': np.nan})()
    
    validation = pd.DataFrame([{
        't_stat': float(t_stat),
        't_pval': float(t_pval),
        't_stat_excess': float(t_stat_excess),
        't_pval_excess': float(t_pval_excess),
        'win_rate': float(win_rate) if pd.notna(win_rate) else np.nan,
        'binom_pval': float(binom.pvalue) if hasattr(binom, 'pvalue') else np.nan,
        'n_trades': int(len(trades))
    }])
    validation.to_csv(outputs_dir / f"{STRATEGY_NAME}_validation.csv", index=False)
    
    # 7. QuantStats tearsheet
    if QUANTSTATS_AVAILABLE and len(sr) > 0 and len(br) > 0:
        qs_html = outputs_dir / f"{STRATEGY_NAME}_quantstats_tearsheet.html"
        try:
            # Ensure data is clean and has proper index
            sr_clean = sr.dropna()
            br_clean = br.dropna()
            
            if len(sr_clean) > 0 and len(br_clean) > 0:
                qs.reports.html(
                    sr_clean, 
                    benchmark=br_clean, 
                    output=str(qs_html), 
                    title="Monthly ML CAD IG ER Index vs Benchmark",
                    benchmark_title="Buy & Hold"
                )
                print(f"✓ QuantStats tearsheet saved: {qs_html}")
            else:
                print("⚠️  QuantStats tearsheet skipped: no clean data after dropna")
        except Exception as e:
            print(f"⚠️  QuantStats tearsheet failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️  QuantStats tearsheet skipped: insufficient data")
    
    # Console output - Comprehensive Strategy Analysis
    print("\n" + "="*80)
    print("🤖 MONTHLY ML CAD IG ER INDEX STRATEGY - COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    # 1. STRATEGY IMPLEMENTATION DETAILS
    print("\n📊 STRATEGY IMPLEMENTATION")
    print("-" * 50)
    feature_cols = [col for col in monthly_features.columns 
                   if any(keyword in col for keyword in ['momentum', 'volatility', 'rank', 'composite'])
                   and col not in ['monthly_forward_return', 'monthly_target']
                   and not pd.isna(monthly_features[col]).all()]
    print(f"• Feature Engineering: {len(feature_cols)} total features created")
    print(f"• Selected Features: {len(feature_cols)} features used")
    print(f"• Models: RandomForest + GradientBoosting Ensemble")
    print(f"• Adaptive Thresholds: Based on volatility and economic regime")
    print(f"• Training Period: 70% of data for training")
    
    # 2. SIGNAL GENERATION PROCESS
    print("\n🎯 SIGNAL GENERATION PROCESS")
    print("-" * 50)
    print(f"• Ensemble Signals Generated: {final_sig.sum()} out of {len(final_sig)} periods ({final_sig.mean()*100:.1f}%)")
    print(f"• Signal Frequency: Monthly (month-end)")
    print(f"• Adaptive Thresholding: Yes (based on market conditions)")
    
    # 3. CURRENT STATE INFORMATION
    print("\n📈 CURRENT STATE & RECENT SIGNALS")
    print("-" * 50)
    
    # Get recent data
    recent_data = monthly_features.tail(5)
    recent_signals = pd.DataFrame({
        'Date': recent_data.index,
        'Ensemble_Prob': predictions[-5:],
        'Final_Sig': final_sig.reindex(recent_data.index).values,
        'Fwd_Return': recent_data['monthly_forward_return'].values
    })
    
    print("Recent Periods Analysis:")
    print(recent_signals.to_string(index=False, float_format='%.4f'))
    
    # Current signal status
    latest_date = monthly_features.index[-1]
    latest_prob = predictions[-1] if not pd.isna(predictions[-1]) else 0
    latest_signal = final_sig.iloc[-1]
    
    print(f"\n🔍 CURRENT SIGNAL STATUS ({latest_date.strftime('%Y-%m-%d')})")
    print(f"• Ensemble Probability: {latest_prob:.4f} ({'✅ ABOVE' if latest_prob >= 0.6 else '❌ BELOW'} 0.6 threshold)")
    print(f"• Final Signal: {'🟢 LONG' if latest_signal == 1 else '🔴 NO POSITION'}")
    
    # 4. ACTIONABLE INFORMATION
    print("\n⚡ ACTIONABLE RECOMMENDATIONS")
    print("-" * 50)
    
    if latest_signal == 1:
        print("🟢 ACTION: ENTER LONG POSITION")
        print(f"   • Signal Strength: {latest_prob:.1%}")
        print(f"   • Expected Hold: Monthly rebalancing")
        print(f"   • Risk Management: Adaptive thresholds based on market conditions")
    else:
        print("🔴 ACTION: NO POSITION")
        print(f"   • Reason: Ensemble probability {latest_prob:.1%} below 0.6 threshold")
    
    # Portfolio performance summary
    print("\n📊 PORTFOLIO PERFORMANCE SUMMARY")
    print("-" * 50)
    print(f"• Total Return: {strategy_stats['Total Return [%]']:.2f}%")
    print(f"• Sharpe Ratio: {strategy_stats['Sharpe Ratio']:.2f}")
    print(f"• Max Drawdown: {strategy_stats['Max Drawdown [%]']:.2f}%")
    print(f"• Win Rate: {strategy_stats['Win Rate [%]']:.1f}%")
    print(f"• Total Trades: {strategy_stats['Total Trades']}")
    print(f"• Current Exposure: {final_sig.mean()*100:.1f}%")
    
    # Risk metrics
    print("\n⚠️  RISK MANAGEMENT")
    print("-" * 50)
    
    # Calculate correct monthly trade duration
    avg_duration = strategy_stats['Avg Winning Trade Duration']
    if hasattr(avg_duration, 'days'):
        # For monthly frequency, duration should be in months
        avg_duration_months = avg_duration.days / 30
        print(f"• Average Trade Duration: {avg_duration_months:.1f} months")
    elif hasattr(avg_duration, 'total_seconds'):
        # Handle timedelta objects
        avg_duration_months = avg_duration.total_seconds() / (30 * 24 * 3600)
        print(f"• Average Trade Duration: {avg_duration_months:.1f} months")
    else:
        print(f"• Average Trade Duration: {avg_duration}")
    
    print(f"• Best Trade: {strategy_stats['Best Trade [%]']:.2f}%")
    print(f"• Worst Trade: {strategy_stats['Worst Trade [%]']:.2f}%")
    print(f"• Profit Factor: {strategy_stats['Profit Factor']:.1f}")
    
    # Additional monthly context
    print(f"• Signal Frequency: Monthly (month-end)")
    print(f"• Rebalancing: Monthly")
    
    # TRADE ANALYSIS SECTION
    print("\n" + "="*80)
    print("📊 TRADE ANALYSIS - DETAILED PERFORMANCE REVIEW")
    print("="*80)
    
    # Get trades data
    trades_df = strategy_pf.trades.records_readable
    
    if len(trades_df) > 0:
        # Calculate duration in days
        trades_df['Duration_Days'] = (pd.to_datetime(trades_df['Exit Timestamp']) - 
                                    pd.to_datetime(trades_df['Entry Timestamp'])).dt.days
        
        # Format the data for display
        trades_display = trades_df.copy()
        trades_display['Entry Date'] = pd.to_datetime(trades_display['Entry Timestamp']).dt.strftime('%m/%d/%Y')
        trades_display['Exit Date'] = pd.to_datetime(trades_display['Exit Timestamp']).dt.strftime('%m/%d/%Y')
        trades_display['Return %'] = (trades_display['Return'] * 100).round(2)
        trades_display['PnL $'] = trades_display['PnL'].round(2)
        trades_display['Entry Price'] = trades_display['Avg Entry Price'].round(2)
        trades_display['Exit Price'] = trades_display['Avg Exit Price'].round(2)
        
        # Show last 20 trades
        recent_trades = trades_display.tail(20)
        
        print("\n🔍 RECENT TRADES (Last 20)")
        print("-" * 80)
        print(f"{'#':<3} {'Entry Date':<12} {'Exit Date':<12} {'Duration':<8} {'Entry $':<8} {'Exit $':<8} {'Return %':<8} {'PnL $':<8} {'Status':<8}")
        print("-" * 80)
        
        for idx, (_, trade) in enumerate(recent_trades.iterrows(), 1):
            print(f"{idx:<3} {trade['Entry Date']:<12} {trade['Exit Date']:<12} {trade['Duration_Days']:<8} "
                  f"{trade['Entry Price']:<8} {trade['Exit Price']:<8} {trade['Return %']:<8} "
                  f"{trade['PnL $']:<8} {trade['Status']:<8}")
        
        # Trade Statistics Summary
        print(f"\n📈 TRADE STATISTICS SUMMARY")
        print("-" * 50)
        
        # Basic stats
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['PnL'] > 0])
        losing_trades = len(trades_df[trades_df['PnL'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Profit/Loss stats
        total_pnl = trades_df['PnL'].sum()
        avg_pnl = trades_df['PnL'].mean()
        best_trade = trades_df['PnL'].max()
        worst_trade = trades_df['PnL'].min()
        
        # Return stats
        avg_return = (trades_df['Return'].mean() * 100)
        best_return = (trades_df['Return'].max() * 100)
        worst_return = (trades_df['Return'].min() * 100)
        
        # Duration stats
        avg_duration = trades_df['Duration_Days'].mean()
        max_duration = trades_df['Duration_Days'].max()
        min_duration = trades_df['Duration_Days'].min()
        
        print(f"• Total Trades: {total_trades}")
        print(f"• Win Rate: {win_rate:.1f}% ({winning_trades}W / {losing_trades}L)")
        print(f"• Total P&L: ${total_pnl:.2f}")
        print(f"• Average P&L: ${avg_pnl:.2f}")
        print(f"• Best Trade: ${best_trade:.2f} ({best_return:.2f}%)")
        print(f"• Worst Trade: ${worst_trade:.2f} ({worst_return:.2f}%)")
        print(f"• Average Duration: {avg_duration:.1f} days")
        print(f"• Duration Range: {min_duration} - {max_duration} days")
        
        # Profit Factor
        gross_profit = trades_df[trades_df['PnL'] > 0]['PnL'].sum()
        gross_loss = abs(trades_df[trades_df['PnL'] < 0]['PnL'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        print(f"• Profit Factor: {profit_factor:.2f}")
        
    else:
        print("⚠️  No trades found in the strategy")
    
    # PERFORMANCE DRILLDOWN SECTION
    print("\n" + "="*80)
    print("📊 PERFORMANCE DRILLDOWN")
    print("="*80)
    
    # 1. Performance by Timeframe
    print("\n📈 PERFORMANCE BY TIMEFRAME")
    print("-" * 60)
    print(f"{'Timeframe':<15} {'Strategy %':<12} {'Buy & Hold %':<12}")
    print("-" * 60)
    
    # Calculate timeframe returns
    strategy_equity = (1.0 + strategy_returns).cumprod()
    bh_equity = (1.0 + bh_returns).cumprod()
    
    # Align data to start from first trade date for fair comparison
    if first_trade_date is not None:
        # Find the first trade date in the equity data
        first_trade_idx = strategy_equity.index.get_indexer([first_trade_date], method='nearest')[0]
        if first_trade_idx >= 0:
            start_date = strategy_equity.index[first_trade_idx]
            strategy_equity = strategy_equity.loc[start_date:]
            bh_equity = bh_equity.loc[start_date:]
            print(f"• Aligned comparisons start from: {start_date.strftime('%Y-%m-%d')}")
    
    # Get current date and calculate various timeframes
    end_date = strategy_equity.index[-1]
    
    timeframes = {
        'MTD': 1,  # Month to date
        '3M': 3,   # 3 months
        '6M': 6,   # 6 months
        'YTD': 12, # Year to date
        '1Y': 12,  # 1 year
        '3Y': 36,  # 3 years
        '5Y': 60,  # 5 years
        '10Y': 120, # 10 years
        'All-time': 999 # All time
    }
    
    for tf_name, tf_months in timeframes.items():
        if tf_name == 'MTD':
            # Month to date
            start_date = end_date.replace(day=1)
        elif tf_name == 'YTD':
            # Year to date
            start_date = end_date.replace(month=1, day=1)
        elif tf_name == 'All-time':
            # All time
            start_date = strategy_equity.index[0]
        else:
            # Other timeframes
            start_date = end_date - pd.DateOffset(months=tf_months)
        
        # Get data for the timeframe
        tf_strategy = strategy_equity.loc[start_date:end_date]
        tf_bh = bh_equity.loc[start_date:end_date]
        
        if len(tf_strategy) > 1 and len(tf_bh) > 1:
            strategy_ret = ((tf_strategy.iloc[-1] / tf_strategy.iloc[0]) - 1) * 100
            bh_ret = ((tf_bh.iloc[-1] / tf_bh.iloc[0]) - 1) * 100
            
            # Annualize for longer periods
            if tf_name in ['3Y', '5Y', '10Y', 'All-time']:
                years = (end_date - start_date).days / 365.25
                if years > 0:
                    strategy_ret = ((tf_strategy.iloc[-1] / tf_strategy.iloc[0]) ** (1/years) - 1) * 100
                    bh_ret = ((tf_bh.iloc[-1] / tf_bh.iloc[0]) ** (1/years) - 1) * 100
                    tf_name += ' (ann.)'
            
            print(f"{tf_name:<15} {strategy_ret:>10.2f}% {bh_ret:>10.2f}%")
        else:
            print(f"{tf_name:<15} {'N/A':>10} {'N/A':>10}")
    
    # 2. EOY Returns vs Benchmark
    print(f"\n📅 EOY RETURNS VS BENCHMARK")
    print(f"Note: Strategy has {WARMUP_MONTHS} month warmup period (~{WARMUP_MONTHS/12:.1f} years)")
    print("-" * 70)
    print(f"{'Year':<6} {'Buy & Hold %':<12} {'Strategy %':<12} {'Multiplier':<10} {'Won':<6}")
    print("-" * 70)
    
    # Calculate yearly returns (using aligned data)
    strategy_yearly = strategy_equity.resample('Y').last().pct_change() * 100
    bh_yearly = bh_equity.resample('Y').last().pct_change() * 100
    
    # Align the data
    common_years = strategy_yearly.index.intersection(bh_yearly.index)
    
    for year in common_years:
        if pd.notna(strategy_yearly[year]) and pd.notna(bh_yearly[year]):
            strategy_ret = strategy_yearly[year]
            bh_ret = bh_yearly[year]
            
            # Calculate multiplier (use absolute value for negative returns)
            if abs(bh_ret) < 0.01:  # Very close to zero
                multiplier = "N/A"
            else:
                # Use absolute value of buy & hold return for multiplier
                multiplier = f"{strategy_ret / abs(bh_ret):.2f}"
            
            # Determine if won (strategy > buy & hold, or strategy = 0% and buy & hold < 0%)
            if strategy_ret > bh_ret:
                won = "+"
            elif strategy_ret == 0.0 and bh_ret < 0.0:
                won = "+"  # Strategy avoided losses during warmup period
            else:
                won = "-"
            
            print(f"{year.year:<6} {bh_ret:>10.2f}% {strategy_ret:>10.2f}% {multiplier:>10} {won:>6}")
    
    # 3. Worst 10 Drawdowns - Strategy
    print(f"\n📉 WORST 10 DRAWDOWNS - STRATEGY")
    print("-" * 60)
    print(f"{'Started':<12} {'Recovered':<12} {'Drawdown %':<12} {'Days':<6}")
    print("-" * 60)
    
    # Get strategy drawdowns
    try:
        strategy_drawdowns = strategy_pf.drawdowns.records_readable
        if len(strategy_drawdowns) > 0:
            # Calculate drawdown percentage from peak and valley values
            strategy_drawdowns['Drawdown_Pct'] = ((strategy_drawdowns['Valley Value'] - strategy_drawdowns['Peak Value']) / strategy_drawdowns['Peak Value']) * 100
            
            # Sort by drawdown (most negative first) and take top 10
            strategy_drawdowns_sorted = strategy_drawdowns.sort_values('Drawdown_Pct').head(10)
            
            for _, dd in strategy_drawdowns_sorted.iterrows():
                started = pd.to_datetime(dd['Start Timestamp']).strftime('%m/%d/%Y')
                recovered = pd.to_datetime(dd['End Timestamp']).strftime('%m/%d/%Y')
                drawdown_pct = dd['Drawdown_Pct']
                days = (pd.to_datetime(dd['End Timestamp']) - pd.to_datetime(dd['Start Timestamp'])).days
                
                print(f"{started:<12} {recovered:<12} {drawdown_pct:>10.2f}% {days:>6}")
        else:
            print("No drawdowns found")
    except Exception as e:
        print(f"Error getting strategy drawdowns: {e}")
        print("No drawdowns found")
    
    # 4. Worst 10 Drawdowns - Buy & Hold
    print(f"\n📉 WORST 10 DRAWDOWNS - BUY & HOLD")
    print("-" * 60)
    print(f"{'Started':<12} {'Recovered':<12} {'Drawdown %':<12} {'Days':<6}")
    print("-" * 60)
    
    # Get buy & hold drawdowns
    try:
        bh_drawdowns = bh_pf.drawdowns.records_readable
        if len(bh_drawdowns) > 0:
            # Calculate drawdown percentage from peak and valley values
            bh_drawdowns['Drawdown_Pct'] = ((bh_drawdowns['Valley Value'] - bh_drawdowns['Peak Value']) / bh_drawdowns['Peak Value']) * 100
            
            # Sort by drawdown (most negative first) and take top 10
            bh_drawdowns_sorted = bh_drawdowns.sort_values('Drawdown_Pct').head(10)
            
            for _, dd in bh_drawdowns_sorted.iterrows():
                started = pd.to_datetime(dd['Start Timestamp']).strftime('%m/%d/%Y')
                recovered = pd.to_datetime(dd['End Timestamp']).strftime('%m/%d/%Y')
                drawdown_pct = dd['Drawdown_Pct']
                days = (pd.to_datetime(dd['End Timestamp']) - pd.to_datetime(dd['Start Timestamp'])).days
                
                print(f"{started:<12} {recovered:<12} {drawdown_pct:>10.2f}% {days:>6}")
        else:
            print("No drawdowns found")
    except Exception as e:
        print(f"Error getting buy & hold drawdowns: {e}")
        print("No drawdowns found")
    
    print("\n" + "="*80)
    print("✅ MONTHLY ML STRATEGY ANALYSIS COMPLETE - READY FOR TRADING DECISIONS")
    print("="*80)
    
    # VectorBT detailed stats - Full Range vs Aligned
    print("\n📊 VECTORBT DETAILED STATISTICS COMPARISON")
    print("="*80)
    
    # Show full range statistics first
    print("🔍 FULL RANGE STATISTICS (2003-2025)")
    print("-" * 60)
    print("STRATEGY (FULL RANGE):")
    print(strategy_stats)
    print()
    print("BUY & HOLD (FULL RANGE):")
    print(bh_stats)
    
    # Create aligned portfolios for fair comparison
    if first_trade_date is not None:
        # Find the first trade date in the data
        first_trade_idx = monthly_features.index.get_indexer([first_trade_date], method='nearest')[0]
        if first_trade_idx >= 0:
            start_date = monthly_features.index[first_trade_idx]
            
            # Align data to first trade date
            aligned_monthly_features = monthly_features.loc[start_date:]
            aligned_final_sig = final_sig.loc[start_date:]
            
            # Create aligned portfolios
            aligned_strategy_pf = create_vectorbt_portfolio(aligned_monthly_features["cad_ig_er_index"], aligned_final_sig, "OptimizedMLv2")
            aligned_bh_signals = pd.Series(1, index=aligned_monthly_features.index, dtype=int)
            aligned_bh_pf = create_vectorbt_portfolio(aligned_monthly_features["cad_ig_er_index"], aligned_bh_signals, "Buy & Hold")
            
            # Get aligned statistics
            aligned_strategy_stats = aligned_strategy_pf.stats()
            aligned_bh_stats = aligned_bh_pf.stats()
            
            print("\n" + "="*80)
            print("🎯 ALIGNED STATISTICS (2010-2025 - FAIR COMPARISON)")
            print("-" * 60)
            print(f"Aligned Period: {start_date.strftime('%Y-%m-%d')} to {aligned_monthly_features.index[-1].strftime('%Y-%m-%d')}")
            print(f"Aligned Duration: {(aligned_monthly_features.index[-1] - start_date).days} days")
            print()
            print("STRATEGY (ALIGNED):")
            print(aligned_strategy_stats)
            print()
            print("BUY & HOLD (ALIGNED):")
            print(aligned_bh_stats)
        else:
            print("\nCould not align data - showing full range only")
    else:
        print("\nNo first trade date - showing full range only")
    
    # Manual backtest calculation for comparison
    print("\n🔍 MANUAL BACKTEST VERIFICATION")
    print("="*80)
    
    # Manual calculation - use aligned data for fair comparison
    if first_trade_date is not None:
        # Use aligned data (from first trade date)
        first_trade_idx = monthly_features.index.get_indexer([first_trade_date], method='nearest')[0]
        if first_trade_idx >= 0:
            start_date = monthly_features.index[first_trade_idx]
            aligned_monthly_features = monthly_features.loc[start_date:]
            aligned_final_sig = final_sig.loc[start_date:]
            manual_returns = (aligned_final_sig * aligned_monthly_features["monthly_forward_return"]).dropna()
            manual_equity = (1 + manual_returns).cumprod()
            years = (aligned_monthly_features.index[-1] - aligned_monthly_features.index[0]).days / 365.25
            print(f"• Using aligned data from: {start_date.strftime('%Y-%m-%d')} to {aligned_monthly_features.index[-1].strftime('%Y-%m-%d')}")
        else:
            # Fallback to full data
            manual_returns = (final_sig * monthly_features["monthly_forward_return"]).dropna()
            manual_equity = (1 + manual_returns).cumprod()
            years = (monthly_features.index[-1] - monthly_features.index[0]).days / 365.25
            print("• Using full data range (2003-2025)")
    else:
        # Fallback to full data
        manual_returns = (final_sig * monthly_features["monthly_forward_return"]).dropna()
        manual_equity = (1 + manual_returns).cumprod()
        years = (monthly_features.index[-1] - monthly_features.index[0]).days / 365.25
        print("• Using full data range (2003-2025)")
    
    # Calculate metrics
    manual_total_return = (manual_equity.iloc[-1] - 1) * 100
    manual_annualized_return = ((manual_equity.iloc[-1] / manual_equity.iloc[0]) ** (1/years) - 1) * 100
    
    # Max drawdown
    manual_peak = manual_equity.cummax()
    manual_drawdown = ((manual_equity - manual_peak) / manual_peak * 100)
    manual_max_drawdown = manual_drawdown.min()
    
    # Time in market
    manual_time_in_market = final_sig.mean() * 100
    
    # Number of trades
    manual_trades = (final_sig.diff() != 0).sum() // 2  # Divide by 2 for entry/exit pairs
    
    print(f"• Manual Total Return: {manual_total_return:.2f}%")
    print(f"• Manual Annualized Return: {manual_annualized_return:.2f}%")
    print(f"• Manual Max Drawdown: {manual_max_drawdown:.2f}%")
    print(f"• Manual Time in Market: {manual_time_in_market:.1f}%")
    print(f"• Manual Number of Trades: {manual_trades}")
    
    # Comparison with VectorBT
    print(f"\n📈 VECTORBT vs MANUAL COMPARISON")
    print("-" * 50)
    print(f"• Total Return: VBT {strategy_stats['Total Return [%]']:.2f}% vs Manual {manual_total_return:.2f}%")
    print(f"• Max Drawdown: VBT {strategy_stats['Max Drawdown [%]']:.2f}% vs Manual {manual_max_drawdown:.2f}%")
    print(f"• Time in Market: VBT {final_sig.mean()*100:.1f}% vs Manual {manual_time_in_market:.1f}%")
    print(f"• Number of Trades: VBT {strategy_stats['Total Trades']} vs Manual {manual_trades}")
    
    # Calculate differences
    tr_diff = abs(strategy_stats['Total Return [%]'] - manual_total_return)
    dd_diff = abs(strategy_stats['Max Drawdown [%]'] - manual_max_drawdown)
    
    print(f"\n✅ VALIDATION STATUS")
    print("-" * 50)
    if tr_diff < 0.1 and dd_diff < 0.1:
        print("✅ Manual and VectorBT calculations match within 0.1%")
    else:
        print(f"⚠️  Differences detected: TR diff {tr_diff:.2f}%, DD diff {dd_diff:.2f}%")
    
    print(f"\nOutput files saved to: {outputs_dir}/")
    print(f"- {STRATEGY_NAME}_composite_stats.csv")
    print(f"- {STRATEGY_NAME}_buy_and_hold_stats.csv") 
    print(f"- {STRATEGY_NAME}_perfect_foresight_stats.csv")
    print(f"- {STRATEGY_NAME}_signals.csv")
    print(f"- {STRATEGY_NAME}_returns.csv")
    print(f"- {STRATEGY_NAME}_benchmark_returns.csv")
    print(f"- {STRATEGY_NAME}_perfect_foresight_returns.csv")
    print(f"- {STRATEGY_NAME}_equity_curve.csv")
    print(f"- {STRATEGY_NAME}_benchmark_equity.csv")
    print(f"- {STRATEGY_NAME}_perfect_foresight_equity.csv")
    print(f"- {STRATEGY_NAME}_trades.csv")
    print(f"- {STRATEGY_NAME}_monthly_signals.csv")
    print(f"- {STRATEGY_NAME}_validation.csv")
    if QUANTSTATS_AVAILABLE:
        print(f"- {STRATEGY_NAME}_quantstats_tearsheet.html")
    
    return strategy_pf, bh_pf, pf_pf

if __name__ == "__main__":
    try:
        strategy_pf, bh_pf, pf_pf = comprehensive_backtest_analysis()
        print("\n✓ Monthly ML analysis completed successfully!")
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        import traceback
        traceback.print_exc()
