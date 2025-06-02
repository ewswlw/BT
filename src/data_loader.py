import pandas as pd
import yaml

def load_config(config_path="src/config.yaml"):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_data(csv_path, date_column):
    """
    Loads CSV, parses dates, sorts, and returns DataFrame.
    Matches notebook logic exactly.
    """
    print(f"[DATA] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=[date_column])
    df = df.set_index(date_column).sort_index()
    print(f"[DATA] Loaded {len(df)} rows, columns: {df.columns.tolist()}")
    return df

def get_weekly_data(df, price_col, resample_freq):
    """
    Resamples daily prices to weekly, forward-fills missing values. Matches notebook logic.
    """
    print(f"[DATA] Resampling {price_col} to {resample_freq}...")
    price = df[price_col].resample(resample_freq).last().ffill()
    print(f"[DATA] Weekly price series: {price.shape}")
    return price
