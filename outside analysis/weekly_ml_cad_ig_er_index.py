#!/usr/bin/env python3
"""
weekly_ml_cad_ig_er_index.py

Weekly ML Strategy with Base (Logit) + Meta-Labeling (Logit) using Triple-Barrier labels.
Uses VectorBT for backtesting and generates comprehensive outputs.

Strategy Overview:
1. Weekly resampling of daily data
2. Feature engineering with expanding window normalization
3. Base logistic regression model for directional prediction
4. Triple-barrier labeling for meta-labeling
5. Meta logistic regression for trade filtering
6. VectorBT portfolio construction and analysis

Key Features:
- Uses with_er_daily.csv data source
- VectorBT for portfolio backtesting
- Comprehensive output artifacts matching existing codebase patterns
- Pre-processed weekly data with daily VectorBT frequency

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
from sklearn.linear_model import LogisticRegression
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
STRATEGY_NAME = "weekly_ml_cad_ig_er_index"
WARMUP_WEEKS = 260
TOPK = 40
BASE_THR = 0.50
META_THR = 0.60
H_WEEKS = 4
PT_MULT = 1.0
SL_MULT = 1.0

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
    """Load and prepare data for weekly ML strategy."""
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
    
    # Resample to weekly (Friday close)
    w_px = price.resample("W-FRI").last().dropna()
    w_df = df.resample("W-FRI").last().ffill()
    w_df["PX"] = w_px
    w_df["fwd_ret_1w"] = w_df["PX"].pct_change().shift(-1)
    w_df = w_df.dropna(subset=["fwd_ret_1w"])
    
    print(f"Data prepared: {len(w_df)} weekly observations from {w_df.index[0]} to {w_df.index[-1]}")
    return w_df, target_col

def build_features(frame):
    """Build features with expanding window normalization (exact approach)."""
    num_cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    X = pd.DataFrame(index=frame.index)
    
    for c in num_cols:
        s = frame[c].astype(float)
        r = s.pct_change()
        X[f"{c}"] = s
        X[f"{c}_chg1w"] = s.diff()
        X[f"{c}_ret1w"] = r
        X[f"{c}_mom4w"] = s/s.shift(4) - 1
        X[f"{c}_mom12w"] = s/s.shift(12) - 1
        mu = r.rolling(52).mean()
        sd = r.rolling(52).std()
        X[f"{c}_zret"] = (r - mu) / (sd + 1e-12)
    
    # Replace inf/nan and normalize
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for c in X.columns:
        mu = X[c].expanding().mean()
        sd = X[c].expanding().std().replace(0, np.nan)
        X[c] = (X[c] - mu) / sd
    
    return X

def train_base_model(X_all, y_dir, warmup_weeks=WARMUP_WEEKS, topk=TOPK):
    """Train base logistic regression model."""
    logit_base = LogisticRegression(max_iter=300, solver="liblinear")
    p_base = pd.Series(index=X_all.index, dtype=float)
    selected = None
    
    for i in range(len(X_all)):
        if i < warmup_weeks:
            continue
            
        if selected is None:
            # Select top features based on correlation
            cors = []
            for c in X_all.columns:
                rho = spearmanr(X_all.iloc[:i][c].values, y_dir.iloc[:i].values, nan_policy='omit')[0]
                rho = 0.0 if np.isnan(rho) else abs(rho)
                cors.append((c, rho))
            selected = [c for c, _ in sorted(cors, key=lambda z: z[1], reverse=True)[:topk]]
        
        # Train on past data
        Xtr = X_all.iloc[:i][selected].fillna(0.0).values
        ytr = y_dir.iloc[:i].values
        
        if len(np.unique(ytr)) < 2:
            continue
            
        logit_base.fit(Xtr, ytr)
        p_base.iloc[i] = logit_base.predict_proba(X_all.iloc[[i]][selected].fillna(0.0).values)[0,1]
    
    return p_base, selected

def create_triple_barrier_labels(base_sig, w_df, h_weeks=H_WEEKS, pt_mult=PT_MULT, sl_mult=SL_MULT):
    """Create triple-barrier labels for meta-labeling."""
    r = w_df["fwd_ret_1w"]
    vol26 = r.rolling(26).std().fillna(r.std())
    entries = base_sig[base_sig == 1].index
    
    def tb_label(t):
        i = r.index.get_loc(t)
        end = min(i + h_weeks, len(r) - 1)
        cum = 0.0
        pt = pt_mult * float(vol26.loc[t])
        sl = sl_mult * float(vol26.loc[t])
        
        for j in range(i, end):
            cum += float(r.iloc[j])
            if cum >= pt:
                return 1
            if cum <= -sl:
                return -1
        
        return 1 if cum > 0 else -1 if cum < 0 else 0
    
    tb = pd.Series({t: tb_label(t) for t in entries})
    meta_y = (tb.replace(0, -1) > 0).astype(int)
    
    return meta_y, entries

def train_meta_model(X_all, selected, meta_y, entries):
    """Train meta logistic regression model."""
    X_meta = X_all[selected].reindex(entries).fillna(0.0)
    logit_meta = LogisticRegression(max_iter=300, solver="liblinear")
    p_meta = pd.Series(index=entries, dtype=float)
    
    for i, t in enumerate(entries):
        past = entries[:i]
        if len(past) < 52:
            continue
            
        Xtr = X_meta.loc[past].values
        ytr = meta_y.loc[past].values
        
        if len(np.unique(ytr)) < 2:
            continue
            
        logit_meta.fit(Xtr, ytr)
        p_meta.loc[t] = logit_meta.predict_proba(X_meta.loc[[t]].values)[0,1]
    
    return p_meta

def create_vectorbt_portfolio(price_data, signals, strategy_name="Strategy"):
    """Create VectorBT portfolio."""
    # Convert signals to boolean entries/exits
    signals_bool = signals.astype(bool)
    entries = signals_bool & ~signals_bool.shift(1).fillna(False)
    exits = ~signals_bool & signals_bool.shift(1).fillna(False)
    
    # Create portfolio with weekly frequency for weekly data
    portfolio = vbt.Portfolio.from_signals(
        close=price_data,
        entries=entries,
        exits=exits,
        freq='W',  # Weekly frequency for weekly data
        init_cash=100.0,
        fees=0.0,
        slippage=0.0
    )
    
    return portfolio

def comprehensive_backtest_analysis(file_path='../data_pipelines/data_processed/with_er_daily.csv'):
    """Main analysis function."""
    print("="*70)
    print("WEEKLY ML CAD IG ER INDEX STRATEGY")
    print("Base (Logit) + Meta-Labeling (Logit) with Triple-Barrier Labels")
    print("="*70)
    
    # Load and prepare data
    w_df, target_col = load_and_prepare_data(file_path)
    
    # Build features
    print("\nBuilding features...")
    X_all = build_features(w_df)
    y_dir = (w_df["fwd_ret_1w"] > 0).astype(int)
    
    # Train base model
    print("Training base logistic regression model...")
    p_base, selected = train_base_model(X_all, y_dir)
    base_sig = (p_base >= BASE_THR).astype(int).fillna(0)
    
    print(f"Base model: {base_sig.sum()} signals out of {len(base_sig)} periods ({base_sig.mean()*100:.1f}% exposure)")
    
    # Create triple-barrier labels
    print("Creating triple-barrier labels...")
    meta_y, entries = create_triple_barrier_labels(base_sig, w_df)
    
    # Train meta model
    print("Training meta logistic regression model...")
    p_meta = train_meta_model(X_all, selected, meta_y, entries)
    meta_accept = (p_meta >= META_THR).astype(float).reindex(w_df.index).fillna(0.0)
    final_sig = ((base_sig == 1) & (meta_accept == 1)).astype(int)
    
    print(f"Meta-filtered: {final_sig.sum()} signals out of {len(final_sig)} periods ({final_sig.mean()*100:.1f}% exposure)")
    
    # Create VectorBT portfolios
    print("\nCreating VectorBT portfolios...")
    
    # Strategy portfolio
    strategy_pf = create_vectorbt_portfolio(w_df["PX"], final_sig, "MetaLabeled")
    
    # Find first trade date
    first_trade_date = None
    if len(strategy_pf.trades.records_readable) > 0:
        first_trade_date = pd.to_datetime(strategy_pf.trades.records_readable.iloc[0]['Entry Timestamp'])
        print(f"• First Trade Date: {first_trade_date.strftime('%Y-%m-%d')}")
    else:
        print("• First Trade Date: No trades found")
    
    # Buy & Hold portfolio
    bh_signals = pd.Series(1, index=w_df.index, dtype=int)
    bh_pf = create_vectorbt_portfolio(w_df["PX"], bh_signals, "Buy & Hold")
    
    # Perfect Foresight portfolio
    pf_signals = (w_df["fwd_ret_1w"] > 0).astype(int)
    pf_pf = create_vectorbt_portfolio(w_df["PX"], pf_signals, "Perfect Foresight")
    
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
    
    # 5. Weekly signals with probabilities
    weekly_signals = pd.DataFrame({
        "Date": w_df.index,
        "BaseProb": p_base.reindex(w_df.index).values,
        "MetaProb": p_meta.reindex(w_df.index).values,
        "BaseSig": base_sig.values,
        "MetaAccept": meta_accept.values,
        "FinalSig": final_sig.values,
        "FwdRet_1w": w_df["fwd_ret_1w"].values
    })
    try:
        weekly_signals.to_csv(outputs_dir / f"{STRATEGY_NAME}_weekly_signals.csv", index=False)
    except PermissionError:
        print(f"⚠️  Could not save weekly_signals.csv - file may be open in IDE")
        # Save with a different name
        weekly_signals.to_csv(outputs_dir / f"{STRATEGY_NAME}_weekly_signals_backup.csv", index=False)
    
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
                    title="Weekly ML CAD IG ER Index vs Benchmark",
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
    print("🤖 WEEKLY ML CAD IG ER INDEX STRATEGY - COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    # 1. STRATEGY IMPLEMENTATION DETAILS
    print("\n📊 STRATEGY IMPLEMENTATION")
    print("-" * 50)
    print(f"• Feature Engineering: {len(X_all.columns)} total features created")
    print(f"• Selected Features: {len(selected)} top features used")
    print(f"• Base Model: Logistic Regression (threshold: {BASE_THR})")
    print(f"• Meta Model: Logistic Regression (threshold: {META_THR})")
    print(f"• Triple-Barrier: {H_WEEKS} week horizon, {PT_MULT}x profit, {SL_MULT}x stop")
    print(f"• Training Period: {WARMUP_WEEKS} weeks warmup")
    
    # 2. SIGNAL GENERATION PROCESS
    print("\n🎯 SIGNAL GENERATION PROCESS")
    print("-" * 50)
    print(f"• Base Signals Generated: {base_sig.sum()} out of {len(base_sig)} periods ({base_sig.mean()*100:.1f}%)")
    print(f"• Meta-Labeling Entries: {len(entries)} base signals for labeling")
    print(f"• Triple-Barrier Labels: {len(meta_y)} labels created")
    print(f"• Meta-Filtered Signals: {final_sig.sum()} out of {len(final_sig)} periods ({final_sig.mean()*100:.1f}%)")
    print(f"• Signal Reduction: {((base_sig.sum() - final_sig.sum()) / base_sig.sum() * 100):.1f}% filtered out")
    
    # 3. CURRENT STATE INFORMATION
    print("\n📈 CURRENT STATE & RECENT SIGNALS")
    print("-" * 50)
    
    # Get recent data
    recent_data = w_df.tail(5)
    recent_signals = pd.DataFrame({
        'Date': recent_data.index,
        'Base_Prob': p_base.reindex(recent_data.index).values,
        'Meta_Prob': p_meta.reindex(recent_data.index).values,
        'Base_Sig': base_sig.reindex(recent_data.index).values,
        'Meta_Accept': meta_accept.reindex(recent_data.index).values,
        'Final_Sig': final_sig.reindex(recent_data.index).values,
        'Fwd_Return': recent_data['fwd_ret_1w'].values
    })
    
    print("Recent Periods Analysis:")
    print(recent_signals.to_string(index=False, float_format='%.4f'))
    
    # Current signal status
    latest_date = w_df.index[-1]
    latest_base_prob = p_base.iloc[-1] if not pd.isna(p_base.iloc[-1]) else 0
    latest_meta_prob = p_meta.iloc[-1] if not pd.isna(p_meta.iloc[-1]) else 0
    latest_signal = final_sig.iloc[-1]
    
    print(f"\n🔍 CURRENT SIGNAL STATUS ({latest_date.strftime('%Y-%m-%d')})")
    print(f"• Base Probability: {latest_base_prob:.4f} ({'✅ ABOVE' if latest_base_prob >= BASE_THR else '❌ BELOW'} {BASE_THR} threshold)")
    print(f"• Meta Probability: {latest_meta_prob:.4f} ({'✅ ABOVE' if latest_meta_prob >= META_THR else '❌ BELOW'} {META_THR} threshold)")
    print(f"• Final Signal: {'🟢 LONG' if latest_signal == 1 else '🔴 NO POSITION'}")
    
    # 4. ACTIONABLE INFORMATION
    print("\n⚡ ACTIONABLE RECOMMENDATIONS")
    print("-" * 50)
    
    if latest_signal == 1:
        print("🟢 ACTION: ENTER LONG POSITION")
        print(f"   • Signal Strength: Base {latest_base_prob:.1%} + Meta {latest_meta_prob:.1%}")
        print(f"   • Expected Hold: Up to {H_WEEKS} weeks")
        print(f"   • Risk Management: {SL_MULT}x volatility stop loss")
        print(f"   • Profit Target: {PT_MULT}x volatility take profit")
    else:
        print("🔴 ACTION: NO POSITION")
        if latest_base_prob < BASE_THR:
            print(f"   • Reason: Base model probability {latest_base_prob:.1%} below {BASE_THR:.1%} threshold")
        elif latest_meta_prob < META_THR:
            print(f"   • Reason: Meta model probability {latest_meta_prob:.1%} below {META_THR:.1%} threshold")
        else:
            print("   • Reason: Signal not generated")
    
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
    
    # Calculate correct weekly trade duration
    avg_duration = strategy_stats['Avg Winning Trade Duration']
    if hasattr(avg_duration, 'days'):
        # For weekly frequency, duration should be in weeks
        avg_duration_weeks = avg_duration.days / 7
        print(f"• Average Trade Duration: {avg_duration_weeks:.1f} weeks")
    elif hasattr(avg_duration, 'total_seconds'):
        # Handle timedelta objects
        avg_duration_weeks = avg_duration.total_seconds() / (7 * 24 * 3600)
        print(f"• Average Trade Duration: {avg_duration_weeks:.1f} weeks")
    else:
        print(f"• Average Trade Duration: {avg_duration}")
    
    print(f"• Best Trade: {strategy_stats['Best Trade [%]']:.2f}%")
    print(f"• Worst Trade: {strategy_stats['Worst Trade [%]']:.2f}%")
    print(f"• Profit Factor: {strategy_stats['Profit Factor']:.1f}")
    
    # Additional weekly context
    print(f"• Expected Hold Period: {H_WEEKS} weeks maximum")
    print(f"• Signal Frequency: Weekly (Friday close)")
    
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
    print(f"Note: Strategy has {WARMUP_WEEKS} week warmup period (~{WARMUP_WEEKS/52:.1f} years)")
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
    print("✅ STRATEGY ANALYSIS COMPLETE - READY FOR TRADING DECISIONS")
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
        first_trade_idx = w_df.index.get_indexer([first_trade_date], method='nearest')[0]
        if first_trade_idx >= 0:
            start_date = w_df.index[first_trade_idx]
            
            # Align data to first trade date
            aligned_w_df = w_df.loc[start_date:]
            aligned_final_sig = final_sig.loc[start_date:]
            
            # Create aligned portfolios
            aligned_strategy_pf = create_vectorbt_portfolio(aligned_w_df["PX"], aligned_final_sig, "MetaLabeled")
            aligned_bh_signals = pd.Series(1, index=aligned_w_df.index, dtype=int)
            aligned_bh_pf = create_vectorbt_portfolio(aligned_w_df["PX"], aligned_bh_signals, "Buy & Hold")
            
            # Get aligned statistics
            aligned_strategy_stats = aligned_strategy_pf.stats()
            aligned_bh_stats = aligned_bh_pf.stats()
            
            print("\n" + "="*80)
            print("🎯 ALIGNED STATISTICS (2010-2025 - FAIR COMPARISON)")
            print("-" * 60)
            print(f"Aligned Period: {start_date.strftime('%Y-%m-%d')} to {aligned_w_df.index[-1].strftime('%Y-%m-%d')}")
            print(f"Aligned Duration: {(aligned_w_df.index[-1] - start_date).days} days")
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
        first_trade_idx = w_df.index.get_indexer([first_trade_date], method='nearest')[0]
        if first_trade_idx >= 0:
            start_date = w_df.index[first_trade_idx]
            aligned_w_df = w_df.loc[start_date:]
            aligned_final_sig = final_sig.loc[start_date:]
            manual_returns = (aligned_final_sig * aligned_w_df["fwd_ret_1w"]).dropna()
            manual_equity = (1 + manual_returns).cumprod()
            years = (aligned_w_df.index[-1] - aligned_w_df.index[0]).days / 365.25
            print(f"• Using aligned data from: {start_date.strftime('%Y-%m-%d')} to {aligned_w_df.index[-1].strftime('%Y-%m-%d')}")
        else:
            # Fallback to full data
            manual_returns = (final_sig * w_df["fwd_ret_1w"]).dropna()
            manual_equity = (1 + manual_returns).cumprod()
            years = (w_df.index[-1] - w_df.index[0]).days / 365.25
            print("• Using full data range (2003-2025)")
    else:
        # Fallback to full data
        manual_returns = (final_sig * w_df["fwd_ret_1w"]).dropna()
        manual_equity = (1 + manual_returns).cumprod()
        years = (w_df.index[-1] - w_df.index[0]).days / 365.25
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
    print(f"- {STRATEGY_NAME}_weekly_signals.csv")
    print(f"- {STRATEGY_NAME}_validation.csv")
    if QUANTSTATS_AVAILABLE:
        print(f"- {STRATEGY_NAME}_quantstats_tearsheet.html")
    
    return strategy_pf, bh_pf, pf_pf

if __name__ == "__main__":
    try:
        strategy_pf, bh_pf, pf_pf = comprehensive_backtest_analysis()
        print("\n✓ Analysis completed successfully!")
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        import traceback
        traceback.print_exc()
