# ----------------------------------------------------------------------------------
# Trender Meta-Label Strategy vs Buy-and-Hold
# Runs on the user-supplied Excel file:  "spx tr daily trend.xlsx"
# Expected runtime 20-30 s.  Produces:
#   • Performance table (CAGR, Sharpe, Max-DD)
#   • Log-scale equity-curve plot
# ----------------------------------------------------------------------------------

import pandas as pd, numpy as np, math, warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')

# ───────────────────────────────────────────────────────────────────────────────
# 1. Load & basic series
FILE = "outside data/spx tr daily trend.xlsx"                 # using local data file
df   = pd.read_excel(FILE, parse_dates=['Date'])\
          .set_index('Date').sort_index()

df['trend']     = np.where(~df['TrndrUp'].isna(),  1,
                    np.where(~df['TrndrDn'].isna(), -1, 0))
df['Open_next'] = df['Open'].shift(-1)
df['ret_open']  = df['Open_next'] / df['Open'] - 1            # open→open return

# ───────────────────────────────────────────────────────────────────────────────
# 2. Features exactly as used in the winning config
df['gap_pct']    = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
df['dist_tr_up'] = ((df['Open'] - df['TrndrUp']) / df['Open']).fillna(0)
df['mom_20']     = df['Close'].pct_change(20)
df['mom_60']     = df['Close'].pct_change(60)

rv20             = df['Close'].pct_change().rolling(20).std()*np.sqrt(252)
rv60             = df['Close'].pct_change().rolling(60).std()*np.sqrt(252)
df['real_vol_20'] = rv20
df['vol_chg']     = rv20 / rv60 - 1

true_rng         = np.maximum(df['High']-df['Low'],
                      np.maximum(abs(df['High']-df['Close'].shift(1)),
                                 abs(df['Low'] - df['Close'].shift(1))))
df['ATR20']      = true_rng.rolling(20).mean()

df['dd_60']      = df['Close'] / df['Close'].rolling(60 , min_periods=60 ).max() - 1
df['dd_120']     = df['Close'] / df['Close'].rolling(120, min_periods=120).max() - 1

# seasonality (continuous – avoids 11 dummies)
df['sin_month']  = np.sin(2*np.pi*df.index.month/12)
df['cos_month']  = np.cos(2*np.pi*df.index.month/12)

FEATS = ['gap_pct','dist_tr_up','mom_20','mom_60','real_vol_20','vol_chg',
         'ATR20','dd_60','dd_120','sin_month','cos_month']

# 30- & 60-day forward open-to-open returns (labels)
for h in (30, 60):
    df[f'fwd{h}'] = df['Open'].shift(-h) / df['Open'] - 1

# ───────────────────────────────────────────────────────────────────────────────
# 3.  Candidate entry dates  = 1st day of new up-trend + positive momentum
up        = df['trend'] == 1
up_cnt    = up.groupby((up != up.shift()).cumsum()).cumsum()
entry_day = up & (up_cnt == 1) & (df['mom_20'] > 0)

# Build labelled feature sets for horizon-30 & horizon-60
def make_label_df(h):
    rows = [[dt, int(df.loc[dt,f'fwd{h}'] > 0)]
            + list(df.loc[dt, FEATS].fillna(0).values)
            for dt in df.index[entry_day] if not pd.isna(df.loc[dt,f'fwd{h}'])]
    cols = ['date','label'] + FEATS
    return pd.DataFrame(rows, columns=cols).set_index('date').sort_index()

lab30 = make_label_df(30)
lab60 = make_label_df(60)
common_dates = sorted(set(lab30.index) & set(lab60.index))    # dual-horizon filter

# ───────────────────────────────────────────────────────────────────────────────
# 4.  Helper: adaptive threshold (rolling 504-day window)
def adaptive_thr(probs, win=504):
    arr = np.asarray(probs[-win:])
    return np.median(arr) + 0.5*np.std(arr)

# Hyper-parameters (winning micro-grid)
GAIN_STEP = 0.04      # tighten stop after each +4 %
SHRINK    = 0.75      # k = k·0.75 on ratchet
K_FLOOR   = 0.10      # stop multiplier never below 0.10 × ATR20
MIN_P     = 0.60      # absolute floor for probability

# ───────────────────────────────────────────────────────────────────────────────
# 5.  Walk-forward meta-label filter  + trade generation
clf30, clf60 = LogisticRegression(max_iter=400), LogisticRegression(max_iter=400)
p30_hist, p60_hist, entries = [], [], []

for i, dt in enumerate(common_dates):
    if i < 40:                                         # need ≥40 prior samples
        continue
    # train on *past* data only
    clf30.fit(lab30.loc[lab30.index < dt, FEATS], lab30.loc[lab30.index < dt, 'label'])
    clf60.fit(lab60.loc[lab60.index < dt, FEATS], lab60.loc[lab60.index < dt, 'label'])
    # predict today
    p30 = clf30.predict_proba([lab30.loc[dt, FEATS]])[0][1]
    p60 = clf60.predict_proba([lab60.loc[dt, FEATS]])[0][1]
    p30_hist.append(p30); p60_hist.append(p60)
    thr30 = max(MIN_P, adaptive_thr(p30_hist))
    thr60 = max(MIN_P, adaptive_thr(p60_hist))
    if (p30 >= thr30) and (p60 >= thr60):              # dual-horizon agreement
        entries.append(dt)

# ───────────────────────────────────────────────────────────────────────────────
# 6.  Back-test with dynamic ATR trailing-stop
stop_k_init = 0.25                                    # initial k
pos, k_now, entry_px, stop_px = 0, np.nan, np.nan, np.nan
position = pd.Series(0, index=df.index)

for t in df.index[:-1]:                               # skip last day (no ret_open)
    if pos == 0:                                      # flat
        if t in entries:
            pos = 1
            entry_px = df.loc[t, 'Open']
            k_now    = stop_k_init
            stop_px  = entry_px - k_now * df.loc[t, 'ATR20']
    else:                                             # in position
        # tighten k after each +4 % advance
        if df.loc[t, 'Open'] >= entry_px * (1 + GAIN_STEP):
            k_now   = max(K_FLOOR, k_now * SHRINK)
            entry_px = df.loc[t, 'Open']              # reset advance anchor
        # trail stop
        stop_px = max(stop_px, df.loc[t, 'Open'] - k_now * df.loc[t, 'ATR20'])
        # stop breach or trend ends → exit at next open
        if (df.loc[t, 'Low'] < stop_px) or (df.loc[t, 'trend'] != 1):
            pos = 0
    position.loc[t] = pos

strategy_eq  = (1 + position * df['ret_open'].fillna(0)).cumprod()
buyhold_eq   = df['Close'] / df['Close'].iloc[0]

# ───────────────────────────────────────────────────────────────────────────────
# 7.  Performance summary helper
def metrics(eq):
    total = eq.iloc[-2]; yrs = (eq.index[-2] - eq.index[0]).days / 365.25
    cagr  = total ** (1/yrs) - 1
    daily = eq.pct_change().dropna()
    sharpe= (daily.mean()/daily.std()) * np.sqrt(252)
    mdd   = (eq / eq.cummax() - 1).min()
    return cagr, sharpe, mdd

c_s, sh_s, dd_s   = metrics(strategy_eq)
c_bh, sh_bh, dd_bh = metrics(buyhold_eq)

print("\n=========== Performance 1989-09-11 → 2025-06-03 ==========")
print(f"{'Metric':<15} |  Strategy   |  Buy-&-Hold")
print("-"*45)
print(f"{'CAGR':<15} | {c_s:>9.2%}  | {c_bh:>9.2%}")
print(f"{'Sharpe':<15} | {sh_s:>9.2f}    | {sh_bh:>9.2f}")
print(f"{'Max DD':<15} | {dd_s:>9.2%}  | {dd_bh:>9.2%}")

# ───────────────────────────────────────────────────────────────────────────────
# 8.  Equity-curve plot (log scale, single axes)
plt.figure(figsize=(10,5))
plt.plot(strategy_eq.index, strategy_eq, label="Trender Strategy")
plt.plot(buyhold_eq.index , buyhold_eq , label="Buy & Hold (SPX)")
plt.yscale('log'); plt.title("Equity Curve (log scale)")
plt.legend(); plt.tight_layout(); plt.show()
