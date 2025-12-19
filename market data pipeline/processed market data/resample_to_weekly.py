#!/usr/bin/env python3
"""Resample daily CAD-IG-ER data to weekly frequency."""

import pandas as pd
import numpy as np

# Load daily data
print("Loading daily data...")
df = pd.read_csv('cad_ig_er_index_data_for_backtest.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)

print(f"Daily data shape: {df.shape}")
print(f"Daily data period: {df.index[0]} to {df.index[-1]}")

# Resample to weekly (Friday anchor)
print("\nResampling to weekly frequency (W-FRI)...")
weekly = df.resample('W-FRI').last()  # Use last value of week

# Forward fill any missing values
weekly = weekly.fillna(method='ffill')

print(f"Weekly data shape: {weekly.shape}")
print(f"Weekly data period: {weekly.index[0]} to {weekly.index[-1]}")
print(f"Total weeks: {len(weekly)}")

# Save weekly data
output_file = 'cad_ig_er_index_data_weekly.csv'
weekly.to_csv(output_file)
print(f"\n✓ Weekly data saved to: {output_file}")

# Show sample
print("\nFirst 5 rows:")
print(weekly.head())
print("\nLast 5 rows:")
print(weekly.tail())
