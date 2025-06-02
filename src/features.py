import pandas as pd

def build_features(df, price, extra_cols=None):
    """
    Computes all technical indicators used in the Multi-asset momentum notebook, with debug prints for output matching.
    """
    feat = pd.DataFrame(index=price.index)
    for lag in [1,2,3,4,6,8,12,13,26,52]:
        feat[f"mom_{lag}"] = price.pct_change(lag)
    for w in [4,8,13,26]:
        feat[f"vol_{w}"] = price.pct_change().rolling(w).std()
    for w in [4,8,13,26]:
        feat[f"sma_{w}_dev"] = price / price.rolling(w).mean() - 1
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    feat["macd_diff"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    low14 = price.rolling(14).min()
    high14 = price.rolling(14).max()
    feat["stoch_k"] = 100 * (price - low14) / (high14 - low14 + 1e-8)
    if extra_cols:
        for col in extra_cols:
            if col in df.columns:
                feat[f"{col}_mom4"] = df[col].resample("W-FRI").last().ffill().pct_change(4)
            else:
                feat[f"{col}_mom4"] = 0.0
    feat = feat.fillna(0)
    print("[FEATURES] Built features:", feat.columns.tolist())
    print("[FEATURES] Feature sample:\n", feat.head())
    return feat

def get_feature_list(feat):
    """Returns a list of feature names for GA."""
    return feat.columns.tolist()
