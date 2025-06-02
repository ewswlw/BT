#!/usr/bin/env python
# coding: utf-8

# # Genetic Algorithm Weekly Trading Strategy — Beginner's Deep Dive
# 
# ---
# 
# ## Table of Contents
# 1. What is a Genetic Algorithm? (Beginner's Analogy)
# 2. Why Use GAs for Trading Strategies?
# 3. Step-by-Step: How This Script Works
#     - 3.1. Setting the Stage: Parameters & Randomness
#     - 3.2. Loading and Preparing Data
#     - 3.3. Building Features: Turning Prices into Signals
#     - 3.4. The GA Building Blocks: Rules, Mutation, Crossover
#     - 3.5. The Evolution Loop: How Strategies Compete & Improve
#     - 3.6. Measuring Success: Backtesting and Fitness
#     - 3.7. Results, Visualization, and Next Steps
# 4. Error Handling, Robustness, and Reproducibility
# 5. Common Pitfalls & Practical Tips
# 6. Glossary
# 7. Further Reading
# 
# ---
# 
# ## 1. What is a Genetic Algorithm? (Beginner's Analogy)
# 
# Imagine you're trying to breed the fastest racehorse. You start with a bunch of horses (strategies), race them, and pick the fastest ones. Then you mix their genes (combine rules), sometimes introduce random changes (mutations), and repeat. Over time, you get faster horses.
# 
# A **Genetic Algorithm (GA)** is a computer method inspired by this process:
# - **Population:** A group of candidate solutions (trading rules).
# - **Selection:** Pick the best performers.
# - **Crossover:** Combine parts of two good solutions to create new ones.
# - **Mutation:** Randomly tweak parts of a solution to explore new possibilities.
# - **Fitness:** A score measuring how good each solution is (e.g., how much money it makes, how risky it is).
# - **Generations:** Repeat the process, letting the best solutions survive and evolve.
# 
# ---
# 
# ## 2. Why Use GAs for Trading Strategies?
# 
# - **Exploration:** Financial markets are complex. There are millions of possible rules for when to buy/sell. GAs help explore this huge space efficiently.
# - **Adaptability:** GAs can discover creative, non-obvious strategies by combining and mutating rules.
# - **Automation:** Instead of hand-crafting rules, let the algorithm search for you.
# 
# ---
# 
# ## 3. Step-by-Step: How This Script Works
# 
# ### 3.1. Setting the Stage: Parameters & Randomness
# 
# ```python
# POP_SIZE, MAX_GENS = 120, 120
# MUT_RATE, CROSS_RATE = 0.40, 0.40
# TARGET_RET = 0.70
# RANDOM_SEED = 7
# np.random.seed(RANDOM_SEED)
# random.seed(RANDOM_SEED)
# ```
# - **POP_SIZE:** Number of strategies (rules) in each generation (think: number of racehorses).
# - **MAX_GENS:** How many generations (rounds of evolution) to run.
# - **MUT_RATE/CROSS_RATE:** Controls how often mutation/crossover happens — keeps the search diverse.
# - **TARGET_RET:** If a strategy achieves this return, stop early (success!).
# - **RANDOM_SEED:** Ensures the experiment is repeatable (important for science and debugging).
# 
# ### 3.2. Loading and Preparing Data
# 
# ```python
# CSV_PATH = r"...with_er_daily.csv"
# df = pd.read_csv(CSV_PATH, parse_dates=True)
# df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
# df = df.set_index(df.columns[0]).sort_index()
# price_col = "cad_ig_er_index" if "cad_ig_er_index" in df.columns else df.columns[1]
# price = df[price_col].resample("W-FRI").last().ffill()
# df_w = price.to_frame("price")
# df_w["ret_fwd"] = price.pct_change().shift(-1)
# for col in ["cad_oas", "us_ig_oas", "us_hy_oas", "vix"]:
#     df_w[col] = df[col].resample("W-FRI").last().ffill() if col in df.columns else 0.0
# df_w = df_w.dropna(subset=["ret_fwd"])
# ```
# - **Reads the data:** Loads daily market data from a CSV file.
# - **Date handling:** Ensures the first column is a date and sorts it.
# - **Resampling:** Converts daily prices to weekly (Friday close), filling missing values.
# - **Forward return:** Calculates the return for the next week (used to judge if a rule is good).
# - **Features:** Adds extra columns (OAS, VIX) if available, else fills with zeros (robust for missing data).
# 
# ### 3.3. Building Features: Turning Prices into Signals
# 
# Features are like clues for the strategy — e.g., is the price trending up? Is volatility high?
# 
# ```python
# feat = pd.DataFrame(index=df_w.index)
# for lag in [1,2,3,4,6,8,12,13,26,52]:
#     feat[f"mom_{lag}"] = price.pct_change(lag)
# for w in [4,8,13,26]:
#     feat[f"vol_{w}"] = price.pct_change().rolling(w).std()
# for w in [4,8,13,26]:
#     feat[f"sma_{w}_dev"] = price/price.rolling(w).mean()-1
# ema12, ema26 = price.ewm(span=12, adjust=False).mean(), price.ewm(span=26, adjust=False).mean()
# feat["macd_diff"] = (ema12-ema26) - (ema12-ema26).ewm(span=9, adjust=False).mean()
# low14, high14 = price.rolling(14).min(), price.rolling(14).max()
# feat["stoch_k"] = 100*(price-low14)/(high14-low14+1e-8)
# for col in ["cad_oas","us_ig_oas","us_hy_oas","vix"]:
#     feat[f"{col}_mom4"] = df_w[col].pct_change(4)
# feat = feat.fillna(0)
# FEATURES = feat.columns.tolist()
# ```
# - **Momentum:** How much the price has changed over various timeframes (1 to 52 weeks).
# - **Volatility:** How much the price fluctuates (rolling standard deviation).
# - **SMA Deviation:** How far the price is from its average (trend detection).
# - **MACD Diff:** A popular momentum indicator.
# - **Stochastic K:** Shows where the price is within its recent range (overbought/oversold).
# - **OAS/VIX Momentum:** How much these risk indicators have changed recently.
# - **NaN Handling:** Any missing values are filled with 0 (prevents errors later).
# 
# ### 3.4. The GA Building Blocks: Rules, Mutation, Crossover
# 
# #### a. Creating a Random Clause
# A **clause** is a simple rule like "momentum over 4 weeks is greater than X".
# 
# ```python
# def rand_clause():
#     f = random.choice(FEATURES)
#     thr = np.percentile(feat[f], random.uniform(10,90))
#     op  = ">" if random.random()<0.5 else "<"
#     return f"(feat['{f}'] {op} {thr:.6f})"
# ```
# - **Randomly picks a feature** (e.g., 4-week momentum).
# - **Chooses a threshold** (between the 10th and 90th percentile of that feature's values).
# - **Randomly picks > or <** (greater or less than the threshold).
# - **Returns a string** representing this clause.
# 
# #### b. Creating a Full Rule
# A **rule** is a combination of clauses, joined by AND (`&`) or OR (`|`).
# 
# ```python
# def gen_rule(max_c=4):
#     n = random.randint(1,max_c)
#     join = " & " if random.random()<0.5 else " | "
#     return join.join(rand_clause() for _ in range(n))
# ```
# - **Chooses 1 to 4 clauses**.
# - **Joins them** with AND (all must be true) or OR (any can be true).
# - Example: `(feat['mom_4'] > 0.03) & (feat['vol_8'] < 0.02)`
# 
# #### c. Evaluating a Rule (Fitness)
# 
# ```python
# def evaluate(rule):
#     try:
#         mask = eval(rule)
#         mask = mask.reindex(price.index).fillna(False)
#         pf = vbt.Portfolio.from_signals(
#             price,
#             entries=mask,
#             exits=~mask,
#             freq='W',
#             init_cash=10000,
#             fees=0.0,
#             slippage=0.0
#         )
#         ret = pf.total_return()
#         dd = pf.max_drawdown()
#         fitness = ret - 0.1*abs(dd)
#         return fitness, ret, dd, pf
#     except Exception as e:
#         print(f"Error evaluating rule: {rule}\n{e}")
#         return -1,-1,0,None
# ```
# - **Evaluates the rule:** Turns the rule string into a series of True/False signals (when to be in the market).
# - **Backtests:** Uses vectorbt to simulate trading (buy when rule is True, sell otherwise).
# - **Measures performance:**
#     - **Total return** (how much money made).
#     - **Max drawdown** (biggest loss from a peak).
#     - **Fitness:** Return minus 10% of drawdown (prefers high return, low risk).
# - **Error Handling:** If the rule is invalid, prints the error and gives a bad score (so it won't be selected).
# 
# #### d. Mutation (Random Change)
# 
# ```python
# def mutate(rule):
#     if random.random()<0.5 and ("&" in rule or "|" in rule):
#         parts = rule.split("&" if "&" in rule else "|")
#         parts[random.randrange(len(parts))] = rand_clause()
#         join = "&" if "&" in rule else "|"
#         return join.join(p.strip() for p in parts)
#     return f"({rule}) {'&' if random.random()<0.5 else '|'} {rand_clause()}"
# ```
# - **With 50% chance:** Replaces a random clause in the rule.
# - **Otherwise:** Adds a new clause with AND/OR.
# - **Purpose:** Keeps the population diverse, helps discover new strategies.
# 
# #### e. Crossover (Breeding)
# 
# ```python
# def crossover(r1,r2):
#     p1 = r1.split("&" if "&" in r1 else "|")
#     p2 = r2.split("&" if "&" in r2 else "|")
#     return random.choice(p1).strip()+" & "+random.choice(p2).strip()
# ```
# - **Combines parts of two rules:** Takes a clause from each and joins with AND.
# - **Purpose:** Mixes good ideas from different strategies.
# 
# ### 3.5. The Evolution Loop: How Strategies Compete & Improve
# 
# This is where the magic happens. The algorithm runs for many generations, each time:
# - **Evaluating all strategies** in the current population.
# - **Selecting the best ones** (the "elite").
# - **Breeding new strategies** by mutation, crossover, or creating new random rules.
# - **Replacing the population** with the new generation.
# 
# ```python
# pop=[gen_rule() for _ in range(POP_SIZE)]
# best_rule,best_ret,best_pf=None,-1,None
# for gen in range(MAX_GENS):
#     scored=[(*evaluate(r)[:2],r) for r in pop]
#     scored.sort(key=lambda x:x[0], reverse=True)
#     elite=[r for _,_,r in scored[:25]]
#     if scored[0][1]>best_ret:
#         best_rule, best_ret = scored[0][2], scored[0][1]
#         best_pf = evaluate(best_rule)[3]
#         print(f"Gen {gen:03d}  best return {best_ret*100:5.1f}%")
#         if best_ret>=TARGET_RET: break
#     next_pop = elite.copy()
#     while len(next_pop)<POP_SIZE:
#         roll=random.random()
#         if roll<MUT_RATE: next_pop.append(mutate(random.choice(elite)))
#         elif roll<MUT_RATE+CROSS_RATE: next_pop.append(crossover(*random.sample(elite,2)))
#         else: next_pop.append(gen_rule())
#     pop=next_pop
# ```
# - **Initial population:** Random rules.
# - **For each generation:**
#     - **Evaluate:** Score all rules.
#     - **Sort:** Best to worst.
#     - **Elite:** Top 25 rules survive.
#     - **Track the best:** If new high score, save it and print progress.
#     - **Early stop:** If target return is reached, break.
#     - **Fill next population:**
#         - By mutation, crossover, or new random rules.
# - **Repeat:** Until max generations or target is reached.
# 
# ### 3.6. Measuring Success: Backtesting and Fitness
# 
# - **Backtesting:** Simulates trading with each rule, using historical data.
# - **Fitness:** Combines return and drawdown into a single score.
# - **Selection:** Only the best (highest fitness) strategies survive to the next round.
# 
# ### 3.7. Results, Visualization, and Next Steps
# 
# ```python
# if best_pf is not None:
#     print("\nBest rule:", best_rule)
#     print(f"Cumulative return {best_ret*100:.1f}%")
#     print(best_pf.stats())
#     best_pf.plot().show()
# else:
#     print("No valid solution found.")
# ```
# - **Prints the best rule** and its performance.
# - **Shows stats** (returns, drawdowns, etc.).
# - **Plots the equity curve** (how your money would have grown).
# - **Handles failure:** If no valid rule is found, prints a warning.
# 
# ---
# 
# ## 4. Error Handling, Robustness, and Reproducibility
# 
# - **Missing Data:** If features are missing, fills with zeros (prevents crashes).
# - **Invalid Rules:** If a rule can't be evaluated, prints the error and gives a low score.
# - **NaNs:** All missing values filled with 0.
# - **Random Seed:** Ensures you get the same results every run (unless you change the seed).
# - **Logging:** Prints progress and errors for transparency.
# 
# ---
# 
# ## 5. Common Pitfalls & Practical Tips
# 
# - **Overfitting:** The GA may find rules that work great on historical data but fail in the future. Always test on unseen data.
# - **Randomness:** Results can vary if you change the seed. Try multiple runs for robustness.
# - **Complexity:** Simpler rules are often more robust. Too many clauses can overfit.
# - **Interpretability:** The best rules may be hard to interpret. Consider adding constraints for clarity.
# 
# ---
# 
# ## 6. Glossary
# 
# - **Genetic Algorithm (GA):** A method inspired by evolution to search for good solutions.
# - **Feature:** A numerical clue or signal derived from market data.
# - **Clause:** A simple condition (e.g., momentum > 0.02).
# - **Rule:** A combination of clauses (e.g., clause1 AND clause2).
# - **Mutation:** Randomly changing part of a rule.
# - **Crossover:** Combining parts of two rules.
# - **Fitness:** Score measuring how good a rule is.
# - **Backtest:** Simulate trading using historical data.
# - **Drawdown:** Largest drop from a peak in portfolio value.
# - **Elite:** The best-performing rules in a generation.
# 
# ---
# 
# ## 7. Further Reading
# 
# - [Wikipedia: Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm)
# - [vectorbt Documentation](https://vectorbt.dev/)
# - [Quantitative Trading Strategies](https://www.investopedia.com/terms/q/quantitative-trading.asp)
# - [Overfitting in Finance](https://www.investopedia.com/terms/o/overfitting.asp)

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import random
import os
import quantstats as qs
import vectorbt as vbt

warnings.filterwarnings("ignore")   # to keep notebook output tidy

# === CONFIGURATION ===
CONFIG = {
    # --- Environment ---
    "random_seed": 7,

    # --- Data ---
    "csv_path": r"c:\Users\Eddy\YTM Capital Dropbox\Eddy Winiarz\Trading\COF\Models\Unfinished Models\Eddy\Python Projects\BT\data_pipelines\data_processed\with_er_daily.csv",
    "date_column_name": None, # If None, assumes first column is date
    "price_column_name": "cad_ig_er_index", # Primary asset for trading and some features
    "resample_frequency": "W-FRI", # Weekly, Friday last
    "forward_return_column": "ret_fwd",
    "additional_weekly_feature_cols": ["cad_oas", "us_ig_oas", "us_hy_oas", "vix"], # Will be resampled weekly

    # --- Feature Engineering ---
    "feature_momentum_lags": [1, 2, 3, 4, 6, 8, 12, 13, 26, 52],
    "feature_volatility_windows": [4, 8, 13, 26],
    "feature_sma_dev_windows": [4, 8, 13, 26],
    "feature_macd_ema_short": 12,
    "feature_macd_ema_long": 26,
    "feature_macd_signal": 9,
    "feature_stochastic_k_window": 14,
    "feature_factor_momentum_lag": 4, # For additional_weekly_feature_cols
    "fill_na_value_for_features": 0.0,

    # --- Genetic Algorithm ---
    "ga_population_size": 120,
    "ga_max_generations": 120,
    "ga_mutation_rate": 0.40,
    "ga_crossover_rate": 0.40,
    "ga_target_return_early_stop": 0.70, # 70%
    "ga_max_clauses_per_rule": 4,
    "ga_elite_size": 25, # Top N individuals to carry over
    "ga_fitness_drawdown_penalty_factor": 0.1, # fitness = ret - (factor * abs(dd))

    # --- Backtesting (vectorbt) ---
    "vbt_initial_cash": 10000,
    "vbt_fees": 0.0,
    "vbt_slippage": 0.0,
    "vbt_freq": "W", # Matches resample_frequency, but 'W' is vbt's standard for weekly

    # --- Reporting ---
    "report_output_dir": "ai backtests/tearsheets",
    "report_filename_html_ga": "Genetic_Algo_Weekly_vs_BuyHold_Refactored.html",
    "report_title_ga": "Genetic Algorithm Weekly vs Buy and Hold (Refactored)"
}

# === ENVIRONMENT SETUP ===
def setup_environment(config: dict):
    """Sets up the environment, primarily random seeds."""
    np.random.seed(config["random_seed"])
    random.seed(config["random_seed"])
    warnings.filterwarnings("ignore") # To keep output tidy
    print("Environment setup: Random seeds set, warnings ignored.")

# === DATA HANDLING ===
def load_and_prepare_data(config: dict) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Loads daily data, prepares weekly price series, forward returns, and other weekly features.
    Returns:
        price_series (pd.Series): Weekly prices of the main trading asset.
        features_df (pd.DataFrame): DataFrame pre-populated with weekly resampled additional features.
        raw_daily_df (pd.DataFrame): Original daily dataframe, for reference if needed.
    """
    print(f"Loading data from: {config['csv_path']}")
    df = pd.read_csv(config["csv_path"], parse_dates=True)
    
    date_col_idx = 0 if config["date_column_name"] is None else df.columns.get_loc(config["date_column_name"])
    df.iloc[:, date_col_idx] = pd.to_datetime(df.iloc[:, date_col_idx], errors="coerce")
    df = df.set_index(df.columns[date_col_idx]).sort_index()

    price_col = config["price_column_name"] if config["price_column_name"] in df.columns else df.columns[1]
    print(f"Using price column: {price_col}")
    
    price_series = df[price_col].resample(config["resample_frequency"]).last().ffill()
    
    # Base for feature dataframe, starting with price and forward returns
    weekly_features_df = price_series.to_frame("price") # "price" is used by feature engineering
    weekly_features_df[config["forward_return_column"]] = price_series.pct_change().shift(-1)

    # Add other specified features, resampled weekly
    for col in config["additional_weekly_feature_cols"]:
        if col in df.columns:
            weekly_features_df[col] = df[col].resample(config["resample_frequency"]).last().ffill()
        else:
            weekly_features_df[col] = config["fill_na_value_for_features"] # Fill with 0 or specified value
            print(f"Warning: Column '{col}' not found in CSV, filled with {config['fill_na_value_for_features']}.")

    weekly_features_df = weekly_features_df.dropna(subset=[config["forward_return_column"]]) # Crucial for training/evaluation
    
    print(f"Data prepared. Weekly price series from {price_series.index.min()} to {price_series.index.max()}")
    print(f"Weekly features shape: {weekly_features_df.shape}")
    return price_series, weekly_features_df, df # Return raw_daily_df as well

# === FEATURE ENGINEERING ===
def engineer_technical_features(
    price_series: pd.Series, 
    weekly_data_df: pd.DataFrame, # Contains 'price' and other resampled columns
    config: dict
) -> tuple[pd.DataFrame, list]:
    """
    Generates a matrix of technical features.
    'weekly_data_df' is expected to have a 'price' column (from price_series) and
    other factor columns specified in config['additional_weekly_feature_cols'].
    """
    print("Engineering technical features...")
    feat_df = pd.DataFrame(index=weekly_data_df.index) # Align index with weekly data

    # Momentum features from primary price_series
    for lag in config["feature_momentum_lags"]:
        feat_df[f"mom_{lag}"] = price_series.pct_change(lag)
    
    # Volatility features from primary price_series
    for w in config["feature_volatility_windows"]:
        feat_df[f"vol_{w}"] = price_series.pct_change().rolling(w).std()
        
    # SMA deviation features from primary price_series
    for w in config["feature_sma_dev_windows"]:
        feat_df[f"sma_{w}_dev"] = price_series / price_series.rolling(w).mean() - 1
        
    # MACD from primary price_series
    ema_short = price_series.ewm(span=config["feature_macd_ema_short"], adjust=False).mean()
    ema_long = price_series.ewm(span=config["feature_macd_ema_long"], adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=config["feature_macd_signal"], adjust=False).mean()
    feat_df["macd_diff"] = macd_line - signal_line
    
    # Stochastic Oscillator K from primary price_series
    low_k = price_series.rolling(config["feature_stochastic_k_window"]).min()
    high_k = price_series.rolling(config["feature_stochastic_k_window"]).max()
    feat_df["stoch_k"] = 100 * (price_series - low_k) / (high_k - low_k + 1e-8) # Adding 1e-8 to avoid division by zero

    # Momentum of additional factors (e.g., OAS, VIX)
    for col_name in config["additional_weekly_feature_cols"]:
        if col_name in weekly_data_df.columns: # Check if column exists in the input df
             feat_df[f"{col_name}_mom{config['feature_factor_momentum_lag']}"] = weekly_data_df[col_name].pct_change(config["feature_factor_momentum_lag"])
        else: # Should not happen if load_and_prepare_data added it (even with zeros)
             feat_df[f"{col_name}_mom{config['feature_factor_momentum_lag']}"] = config["fill_na_value_for_features"]


    feat_df = feat_df.fillna(config["fill_na_value_for_features"])
    feature_names_list = feat_df.columns.tolist()
    print(f"Features engineered. Shape: {feat_df.shape}. Number of features: {len(feature_names_list)}")
    return feat_df, feature_names_list

# === GENETIC ALGORITHM OPERATORS ===
# These functions now need features_df and feature_names_list passed to them.
# The 'feat' global variable is replaced by 'features_df' passed as argument.

def _generate_random_clause(features_df: pd.DataFrame, feature_names_list: list, config: dict) -> str:
    """Generates a single random trading rule clause.
    Note: Uses 'features_df' passed as argument instead of global 'feat'.
    """
    selected_feature = random.choice(feature_names_list)
    # Ensure the chosen feature column is not all NaN or identical values before percentile calculation
    if features_df[selected_feature].nunique() < 2: # Not enough unique values for percentile
        threshold_val = features_df[selected_feature].iloc[0] if not features_df[selected_feature].empty else 0
    else:
        threshold_val = np.percentile(features_df[selected_feature].dropna(), random.uniform(10, 90))
    
    operator = ">" if random.random() < 0.5 else "<"
    # The rule string will refer to 'features_df' when evaluated by eval() in the calling scope
    return f"(features_df['{selected_feature}'] {operator} {threshold_val:.6f})"

def generate_initial_rule(features_df: pd.DataFrame, feature_names_list: list, config: dict) -> str:
    """Generates a full random trading rule (combination of clauses)."""
    num_clauses = random.randint(1, config["ga_max_clauses_per_rule"])
    join_operator = " & " if random.random() < 0.5 else " | "
    clauses = [_generate_random_clause(features_df, feature_names_list, config) for _ in range(num_clauses)]
    return join_operator.join(clauses)

def mutate_rule(rule_string: str, features_df: pd.DataFrame, feature_names_list: list, config: dict) -> str:
    """Applies mutation to an existing rule string."""
    # Simplified: 50% chance to replace a part, otherwise add a new clause
    if random.random() < 0.5 and ("&" in rule_string or "|" in rule_string):
        join_operator = "&" if "&" in rule_string else "|"
        parts = rule_string.split(f" {join_operator} ") # Split by " & " or " | "
        
        # Ensure parts are correctly formed (e.g. "(clause1) & (clause2)")
        # This simple split might need refinement for complex nested rules.
        # For now, assuming simple structure like "clause1 & clause2 & clause3"
        
        if parts: # If splitting produced non-empty list
            idx_to_mutate = random.randrange(len(parts))
            parts[idx_to_mutate] = _generate_random_clause(features_df, feature_names_list, config)
            return f" {join_operator} ".join(p.strip() for p in parts) # Ensure spaces around operator
        # Fallback if split was not as expected or parts is empty
        return _generate_random_clause(features_df, feature_names_list, config) 
    else: # Add a new clause
        new_clause = _generate_random_clause(features_df, feature_names_list, config)
        join_operator = " & " if random.random() < 0.5 else " | "
        return f"({rule_string}) {join_operator} {new_clause}"


def crossover_rules(rule1_string: str, rule2_string: str, features_df: pd.DataFrame, feature_names_list: list, config: dict) -> str:
    """Combines two parent rule strings to produce a child. Simplified crossover."""
    # Simplistic crossover: take a random clause from each and combine with AND.
    # This might not always produce logically sound or diverse offspring if clauses are complex.
    # A more robust crossover would parse the rule structure.
    
    # Try to extract clauses. This is a heuristic.
    clauses1 = [c.strip() for c in rule1_string.replace("(", "").replace(")", "").split(" & ") if c.strip()]
    if not clauses1: clauses1 = [c.strip() for c in rule1_string.replace("(", "").replace(")", "").split(" | ") if c.strip()]
    
    clauses2 = [c.strip() for c in rule2_string.replace("(", "").replace(")", "").split(" & ") if c.strip()]
    if not clauses2: clauses2 = [c.strip() for c in rule2_string.replace("(", "").replace(")", "").split(" | ") if c.strip()]

    if not clauses1: # If r1 had no clear clauses (e.g. was a single clause itself)
        clause_from_r1 = rule1_string 
    else:
        clause_from_r1 = random.choice(clauses1)

    if not clauses2: # If r2 had no clear clauses
        clause_from_r2 = rule2_string
    else:
        clause_from_r2 = random.choice(clauses2)

    # Ensure they are not empty strings if split failed badly
    if not clause_from_r1.strip(): clause_from_r1 = _generate_random_clause(features_df, feature_names_list, config)
    if not clause_from_r2.strip(): clause_from_r2 = _generate_random_clause(features_df, feature_names_list, config)
        
    return f"({clause_from_r1}) & ({clause_from_r2})" # Default to AND combination


# === FITNESS EVALUATION ===
def evaluate_rule_fitness(
    rule_string: str, 
    price_series_to_trade: pd.Series, 
    features_df: pd.DataFrame, # features_df must be in scope for eval()
    config: dict
) -> tuple[float, float, float, vbt.Portfolio | None]:
    """
    Evaluates a rule string's fitness using vectorbt.
    'features_df' is passed explicitly and used by eval().
    """
    try:
        # `eval` will use the 'features_df' in the local scope of this function.
        signal_mask = eval(rule_string, {"pd": pd, "np": np}, {"features_df": features_df})
        signal_mask = signal_mask.reindex(price_series_to_trade.index).fillna(False)

        portfolio = vbt.Portfolio.from_signals(
            price_series_to_trade,
            entries=signal_mask,
            exits=~signal_mask, # Exit when signal is False
            freq=config["vbt_freq"],
            init_cash=config["vbt_initial_cash"],
            fees=config["vbt_fees"],
            slippage=config["vbt_slippage"]
        )
        total_return = portfolio.total_return()
        max_drawdown = portfolio.max_drawdown()
        
        fitness = total_return - config["ga_fitness_drawdown_penalty_factor"] * abs(max_drawdown)
        return fitness, total_return, max_drawdown, portfolio
    
    except Exception as e:
        # print(f"Error evaluating rule: '{rule_string}'\n{e}") # Can be too verbose
        return -float('inf'), -float('inf'), 1.0, None # Worst possible fitness, high DD


# === GENETIC ALGORITHM CORE ===
def run_genetic_algorithm(
    price_series_to_trade: pd.Series, 
    features_df: pd.DataFrame, 
    feature_names_list: list, 
    config: dict
) -> tuple[str | None, float, vbt.Portfolio | None]:
    """
    Runs the main genetic algorithm loop.
    Uses features_df and feature_names_list for rule generation and evaluation.
    """
    print("\n--- Starting Genetic Algorithm ---")
    population = [generate_initial_rule(features_df, feature_names_list, config) for _ in range(config["ga_population_size"])]
    
    best_overall_rule = None
    best_overall_return = -float('inf')
    best_overall_portfolio = None

    for gen in range(config["ga_max_generations"]):
        # Evaluate current population
        # The list comprehension now passes features_df to evaluate_rule_fitness
        scored_population = []
        for rule_str in population:
            fitness, ret, dd, pf = evaluate_rule_fitness(rule_str, price_series_to_trade, features_df, config)
            scored_population.append({'rule': rule_str, 'fitness': fitness, 'return': ret, 'pf': pf})

        scored_population.sort(key=lambda x: x['fitness'], reverse=True)

        # Track best individual in this generation and overall
        current_gen_best = scored_population[0]
        if current_gen_best['return'] > best_overall_return: # Using return for tracking best, fitness for selection
            best_overall_return = current_gen_best['return']
            best_overall_rule = current_gen_best['rule']
            best_overall_portfolio = current_gen_best['pf'] # Get the portfolio object
            print(f"Gen {gen:03d}: New best return = {best_overall_return*100:6.2f}%, Fitness = {current_gen_best['fitness']:.4f}")
            if best_overall_return >= config["ga_target_return_early_stop"]:
                print(f"Target return of {config['ga_target_return_early_stop']*100:.1f}% reached. Stopping early.")
                break
        else:
            print(f"Gen {gen:03d}: Best fitness this gen = {current_gen_best['fitness']:.4f} (Return: {current_gen_best['return']*100:6.2f}%)")


        # Create next generation
        elite_rules = [indiv['rule'] for indiv in scored_population[:config["ga_elite_size"]]]
        next_population = elite_rules.copy()

        while len(next_population) < config["ga_population_size"]:
            roll = random.random()
            parent1 = random.choice(elite_rules) # Select from elite for breeding
            
            if roll < config["ga_mutation_rate"]:
                next_population.append(mutate_rule(parent1, features_df, feature_names_list, config))
            elif roll < config["ga_mutation_rate"] + config["ga_crossover_rate"]:
                parent2 = random.choice(elite_rules)
                if parent1 == parent2 and len(elite_rules) > 1: # Avoid crossover with self if possible
                    parent2 = random.choice([r for r in elite_rules if r != parent1])
                if parent2 is None : parent2 = parent1 # fallback if only one elite
                next_population.append(crossover_rules(parent1, parent2, features_df, feature_names_list, config))
            else:
                next_population.append(generate_initial_rule(features_df, feature_names_list, config))
        
        population = next_population

    print("--- Genetic Algorithm Finished ---")
    if best_overall_rule:
        print(f"Best rule found: {best_overall_rule}")
        print(f"Best total return: {best_overall_return*100:.2f}%")
    else:
        print("No valid solution improving initial state was found by GA.")
        
    return best_overall_rule, best_overall_return, best_overall_portfolio


# === RESULTS DISPLAY & REPORTING ===
def display_and_report_results(
    best_rule_string: str | None, 
    best_return: float, 
    best_portfolio: vbt.Portfolio | None,
    price_series_for_benchmark: pd.Series,
    config: dict
):
    """Displays results, plots, and generates QuantStats report for the best GA strategy."""
    print("\n--- GA Strategy Results ---")
    if best_portfolio is not None and best_rule_string is not None:
        print(f"Best rule: {best_rule_string}")
        print(f"Cumulative return from GA: {best_return*100:.2f}%")
        
        # --- QuantStats Backtest & Report ---
        print("\n--- Generating QuantStats Report for GA Strategy ---")
        output_dir = config["report_output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        report_path_ga = os.path.join(output_dir, config["report_filename_html_ga"])

        strategy_returns = best_portfolio.returns()
        if strategy_returns is None or strategy_returns.empty:
            print("No valid strategy returns found for QuantStats report.")
        else:
            benchmark_returns = price_series_for_benchmark.pct_change().reindex(strategy_returns.index).fillna(0)
            benchmark_returns = benchmark_returns.replace([np.inf, -np.inf], np.nan).dropna()

            # Generate the full QuantStats tearsheet in console (like Multi-asset momentum)
            print("\n--- QuantStats Full Report (Strategy vs Buy & Hold) ---")
            qs.reports.full(
                strategy_returns,
                benchmark=benchmark_returns,
                title=config["report_title_ga"],
                freq=config["vbt_freq"]
            )

            # Save the HTML report for later review or sharing
            print(f"\nSaving QuantStats HTML report to: {report_path_ga}")
            qs.reports.html(
                strategy_returns,
                benchmark=benchmark_returns,
                output=report_path_ga,
                title=config["report_title_ga"],
                freq=config["vbt_freq"] # Use vbt_freq for consistency
            )
            print("QuantStats report saved.")

    else:
        print("No valid GA solution found to display or report.")

# === MAIN WORKFLOW ===
def main(app_config: dict):
    """Main function to run the Genetic Algorithm trading strategy discovery."""
    
    setup_environment(app_config)
    
    price_series, weekly_features_base_df, _ = load_and_prepare_data(app_config) # underscore for raw_daily_df
    
    # Align price_series index with weekly_features_base_df if they differ after dropna in load_and_prepare
    price_series = price_series.reindex(weekly_features_base_df.index).ffill().bfill()

    features_df, feature_names_list = engineer_technical_features(
        price_series, # Use the reindexed price_series
        weekly_features_base_df, 
        app_config
    )
    
    # Ensure features_df index aligns with price_series for backtesting
    # This is critical because 'eval(rule_string)' uses features_df, and vbt uses price_series
    # Their indices must match for signal alignment.
    # `engineer_technical_features` already creates `feat_df` with `weekly_data_df.index`
    # `weekly_data_df` is derived from `price_series` and then `dropna` on `ret_fwd`
    # So, `price_series` needs to be reindexed to match `features_df` for `evaluate_rule_fitness`
    price_series_for_ga_eval = price_series.reindex(features_df.index).ffill().bfill()


    best_rule, best_ga_return, best_ga_portfolio = run_genetic_algorithm(
        price_series_for_ga_eval, # Use aligned price series
        features_df, 
        feature_names_list, 
        app_config
    )
    
    display_and_report_results(
        best_rule, 
        best_ga_return, 
        best_ga_portfolio,
        price_series_for_ga_eval, # Use the same aligned price series for benchmark calc
        app_config
    )
    
    print("\n--- Genetic Algorithm Script Finished ---")

if __name__ == "__main__":
    main(CONFIG)

# --- Original extensive markdown comments and explanations follow ---
# (These are preserved from the notebook and provide excellent context)

# # Genetic Algorithm Weekly Trading Strategy — Beginner's Deep Dive
# 
# ... (rest of the original markdown comments) ...
# ## 1. What is a Genetic Algorithm? (Beginner's Analogy)
# ...
# ## 2. Why Use GAs for Trading Strategies?
# ...
# ## 3. Step-by-Step: How This Script Works
# ...
# ### 3.1. Setting the Stage: Parameters & Randomness
# ...
# ### 3.2. Loading and Preparing Data
# ...
# ### 3.3. Building Features: Turning Prices into Signals
# ...
# ### 3.4. The GA Building Blocks: Rules, Mutation, Crossover
# ...
# #### a. Creating a Random Clause
# ...
# #### b. Creating a Full Rule
# ...
# #### c. Evaluating a Rule (Fitness)
# ...
# #### d. Mutation (Random Change)
# ...
# #### e. Crossover (Breeding)
# ...
# ### 3.5. The Evolution Loop: How Strategies Compete & Improve
# ...
# ### 3.6. Measuring Success: Backtesting and Fitness
# ...
# ### 3.7. Results, Visualization, and Next Steps
# ...
# ## 4. Error Handling, Robustness, and Reproducibility
# ...
# ## 5. Common Pitfalls & Practical Tips
# ...
# ## 6. Glossary
# ...
# ## 7. Further Reading
...

