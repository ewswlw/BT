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
    
    print("\n" + "="*80)
    print("✅ STRATEGY ANALYSIS COMPLETE - READY FOR TRADING DECISIONS")
    print("="*80)
    
    # VectorBT detailed stats
    print("\n📊 VECTORBT DETAILED STATISTICS")
    print("="*80)
    print(strategy_stats)
    
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
