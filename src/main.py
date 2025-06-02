import os
import sys
import traceback
import yaml
import pandas as pd
import numpy as np
import matplotlib
# Force non-interactive mode to avoid plots blocking
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

# SUPER AGGRESSIVE PRINT
print("\n\n" + "!" * 50)
print("SCRIPT STARTED - PYTHON VERSION:", sys.version)
print("CURRENT DIRECTORY:", os.getcwd())
print("!" * 50 + "\n\n")

# Fix imports for src modules
sys.path.insert(0, os.path.abspath('.'))  # Add current dir to path
from src.data_loader import load_config, load_data, get_weekly_data
from src.features import build_features, get_feature_list
from src.strategy import momentum_strategy
from src.backtest import run_backtest
from src.reporting import print_metrics, plot_results

def main():
    try:
        print("\n[MAIN] Pipeline started\n" + "-"*50)
        
        # Get config path
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        print(f"[MAIN] Loading config from: {config_path}")
        
        # Load config
        try:
            config = load_config(config_path)
            print(f"[MAIN] Config loaded successfully: {config}")
        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}")
            return
            
        # Load data
        try:
            csv_path = config['data']['csv_path']
            print(f"[MAIN] Looking for data at: {csv_path} (relative to {os.getcwd()})")
            if not os.path.isabs(csv_path):
                csv_path = os.path.join(os.getcwd(), csv_path)
            print(f"[MAIN] Full path: {csv_path}")
            
            # Check if file exists
            if not os.path.exists(csv_path):
                print(f"[ERROR] Data file not found: {csv_path}")
                # Try to list available files
                data_dir = os.path.dirname(csv_path)
                if os.path.exists(data_dir):
                    print(f"[INFO] Files in {data_dir}: {os.listdir(data_dir)}")
                return
                
            df = load_data(csv_path, config['data']['date_column'])
            price = get_weekly_data(df, config['data']['price_column'], config['data']['resample_freq'])
        except Exception as e:
            print(f"[ERROR] Data loading failed: {e}")
            traceback.print_exc()
            return
            
        # Build features
        try:
            feat = build_features(df, price, extra_cols=['cad_oas', 'us_ig_oas', 'us_hy_oas', 'vix'])
        except Exception as e:
            print(f"[ERROR] Feature building failed: {e}")
            traceback.print_exc()
            return
            
        # Run momentum strategy (default lag=4)
        try:
            print("[MAIN] Running momentum strategy...")
            signal = momentum_strategy(feat, lag=4)
        except Exception as e:
            print(f"[ERROR] Strategy execution failed: {e}")
            traceback.print_exc()
            return
            
        # Backtest
        try:
            print("[MAIN] Running backtest...")
            equity, strat_ret, drawdown, metrics = run_backtest(price, signal)
        except Exception as e:
            print(f"[ERROR] Backtest failed: {e}")
            traceback.print_exc()
            return
            
        # Reporting
        try:
            print("[MAIN] Reporting results...")
            print_metrics(metrics)
            plot_results(equity, drawdown)
            print("\n[MAIN] Pipeline completed successfully!")
        except Exception as e:
            print(f"[ERROR] Reporting failed: {e}")
            traceback.print_exc()
    
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
