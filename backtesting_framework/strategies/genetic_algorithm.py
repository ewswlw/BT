"""
Genetic Algorithm trading strategy implementation.
Matches the logic from genetic algo weekly.py exactly.
"""

from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
import random
from .base_strategy import BaseStrategy


class GeneticAlgorithmStrategy(BaseStrategy):
    """
    Genetic Algorithm-based trading strategy.
    Evolves trading rules and applies them on a weekly basis (Mondays).
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # GA Parameters matching genetic algo weekly.py exactly
        self.population_size = config.get('population_size', 120)
        self.max_generations = config.get('max_generations', 120) 
        self.mutation_rate = config.get('mutation_rate', 0.40)
        self.crossover_rate = config.get('crossover_rate', 0.40)
        self.target_return_early_stop = config.get('target_return_early_stop', 0.70)
        self.max_clauses_per_rule = config.get('max_clauses_per_rule', 4)
        self.elite_size = config.get('elite_size', 25)
        self.fitness_drawdown_penalty_factor = config.get('fitness_drawdown_penalty_factor', 0.1)
        self.initial_cash_ga = config.get('initial_cash_ga', 10000)
        
        # Trading parameters
        self.trading_asset = config.get('trading_asset', 'cad_ig_er_index')
        
        # Best evolved rule (will be set after evolution)
        self.best_rule = None
        self.best_fitness = -np.inf
        
    def get_required_features(self) -> List[str]:
        """Returns an empty list as features are generated dynamically."""
        return []
    
    def engineer_ga_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer technical features exactly matching genetic algo weekly.py.
        
        Args:
            data: Weekly price data DataFrame
            
        Returns:
            pd.DataFrame: Comprehensive technical features
        """
        # Get the main price series
        if self.trading_asset not in data.columns:
            raise ValueError(f"Trading asset {self.trading_asset} not found in data")
        
        price = data[self.trading_asset]
        feat = pd.DataFrame(index=data.index)
        
        # Momentum features (matching genetic algo weekly.py exactly)
        momentum_lags = [1, 2, 3, 4, 6, 8, 12, 13, 26, 52]
        for lag in momentum_lags:
            feat[f"mom_{lag}"] = price.pct_change(lag)
        
        # Volatility features (matching genetic algo weekly.py exactly)
        vol_windows = [4, 8, 13, 26]
        for w in vol_windows:
            feat[f"vol_{w}"] = price.pct_change().rolling(w).std()
        
        # SMA deviation features (matching genetic algo weekly.py exactly)
        sma_windows = [4, 8, 13, 26]
        for w in sma_windows:
            feat[f"sma_{w}_dev"] = price / price.rolling(w).mean() - 1
        
        # MACD (matching genetic algo weekly.py exactly)
        ema12 = price.ewm(span=12, adjust=False).mean()
        ema26 = price.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        feat["macd_diff"] = macd_line - signal_line
        
        # Stochastic K (matching genetic algo weekly.py exactly)
        low14 = price.rolling(14).min()
        high14 = price.rolling(14).max()
        feat["stoch_k"] = 100 * (price - low14) / (high14 - low14 + 1e-8)
        
        # OAS/VIX momentum features (matching genetic algo weekly.py exactly)
        factor_cols = ["cad_oas", "us_ig_oas", "us_hy_oas", "vix"]
        for col in factor_cols:
            if col in data.columns:
                feat[f"{col}_mom4"] = data[col].pct_change(4)
            else:
                feat[f"{col}_mom4"] = 0.0
        
        # Fill NaN values (matching genetic algo weekly.py exactly)
        feat = feat.fillna(0.0)
        
        print(f"Engineered {len(feat.columns)} technical features for GA evolution")
        return feat
    
    def generate_signals(self, data: pd.DataFrame, features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Evolves a trading rule and generates signals based on it,
        rebalancing weekly on Mondays.
        
        Args:
            data: Daily price data
            features: DataFrame of pre-computed features
            
        Returns:
            Tuple of (entry_signals, exit_signals)
        """
        # Evolve the best trading rule based on the provided data and features
        best_rule = self._evolve_best_rule(data, features)
        print(f"Best evolved rule: {best_rule}")

        # Apply the best rule to the features to get raw signals for all days
        try:
            # Use eval in a controlled way to apply the rule string
            raw_signals = eval(best_rule, {'features_df': features})
        except Exception as e:
            print(f"Error evaluating evolved rule: {e}")
            raw_signals = pd.Series(False, index=data.index)

        # --- Monday-Only Trading Logic ---
        # 1. Identify Mondays for rebalancing
        is_monday = data.index.dayofweek == 0
        
        # 2. Get the signal from the evolved rule ONLY on Mondays
        monday_signal = raw_signals[is_monday]
        
        # 3. Create a Series that only has signal values on Mondays
        signals_on_mondays = pd.Series(np.nan, index=data.index)
        signals_on_mondays[is_monday] = monday_signal
        
        # 4. Forward-fill the signal to hold position until the next Monday
        final_signals = signals_on_mondays.ffill().fillna(False)
        
        # --- Convert positions to entry/exit signals ---
        positions = final_signals.astype(int)
        positions_shifted = positions.shift(1).fillna(0)
        
        entry_signals = (positions == 1) & (positions_shifted == 0)
        exit_signals = (positions == 0) & (positions_shifted == 1)

        print(f"Generated signals using evolved rule: {entry_signals.sum()} entry periods ({entry_signals.mean():.2%})")
        return entry_signals, exit_signals
    
    def _evolve_best_rule(self, data: pd.DataFrame, features: pd.DataFrame) -> str:
        """
        Evolve trading rules using genetic algorithm.
        Exactly matches the GA logic from genetic algo weekly.py
        """
        print(f"\n--- Starting Genetic Algorithm Evolution ---")
        print(f"Population: {self.population_size}, Generations: {self.max_generations}")
        print(f"Mutation Rate: {self.mutation_rate}, Crossover Rate: {self.crossover_rate}")
        
        # Get feature names for rule generation
        feature_names = features.columns.tolist()
        price_series = data[self.trading_asset]
        
        # Initialize population with random rules
        population = []
        for _ in range(self.population_size):
            rule = self._generate_random_rule(features, feature_names)
            population.append(rule)
        
        best_overall_rule = None
        best_overall_return = -np.inf
        
        # Evolution loop
        for generation in range(self.max_generations):
            # Evaluate all rules in current population
            scored_population = []
            for rule in population:
                fitness, total_return, max_dd = self._evaluate_rule_fitness(rule, price_series, features)
                scored_population.append({
                    'rule': rule,
                    'fitness': fitness, 
                    'return': total_return,
                    'drawdown': max_dd
                })
            
            # Sort by fitness (best first)
            scored_population.sort(key=lambda x: x['fitness'], reverse=True)
            
            # Track best overall
            current_best = scored_population[0]
            if current_best['return'] > best_overall_return:
                best_overall_return = current_best['return']
                best_overall_rule = current_best['rule']
                print(f"Gen {generation:03d}: New best return = {best_overall_return*100:6.2f}%, Fitness = {current_best['fitness']:.4f}")
                
                # Early stopping if target reached
                if best_overall_return >= self.target_return_early_stop:
                    print(f"Target return of {self.target_return_early_stop*100:.1f}% reached. Stopping early.")
                    break
            else:
                print(f"Gen {generation:03d}: Best fitness this gen = {current_best['fitness']:.4f} (Return: {current_best['return']*100:6.2f}%)")
            
            # Create next generation
            elite_rules = [indiv['rule'] for indiv in scored_population[:self.elite_size]]
            next_population = elite_rules.copy()
            
            # Fill rest of population with mutations, crossovers, and new random rules
            while len(next_population) < self.population_size:
                roll = random.random()
                
                if roll < self.mutation_rate:
                    # Mutation
                    parent = random.choice(elite_rules)
                    child = self._mutate_rule(parent, features, feature_names)
                    next_population.append(child)
                    
                elif roll < self.mutation_rate + self.crossover_rate:
                    # Crossover
                    parent1, parent2 = random.sample(elite_rules, 2)
                    child = self._crossover_rules(parent1, parent2, features, feature_names)
                    next_population.append(child)
                    
                else:
                    # New random rule
                    new_rule = self._generate_random_rule(features, feature_names)
                    next_population.append(new_rule)
            
            population = next_population
        
        # Save best evolved rule
        self.best_rule = best_overall_rule
        self.best_fitness = best_overall_return
        
        print(f"--- Genetic Algorithm Evolution Complete ---")
        if best_overall_rule:
            print(f"Best evolved rule: {best_overall_rule}")
            print(f"Best return: {best_overall_return*100:.2f}%")
        else:
            print("No valid rule evolved")
        
        return best_overall_rule
    
    def _generate_random_rule(self, features_df: pd.DataFrame, feature_names: List[str]) -> str:
        """Generate a random trading rule (matching genetic algo weekly.py)."""
        num_clauses = random.randint(1, self.max_clauses_per_rule)
        clauses = []
        
        for _ in range(num_clauses):
            clause = self._generate_random_clause(features_df, feature_names)
            clauses.append(clause)
        
        # Join with AND or OR
        join_op = " & " if random.random() < 0.5 else " | "
        return join_op.join(clauses)
    
    def _generate_random_clause(self, features_df: pd.DataFrame, feature_names: List[str]) -> str:
        """Generate a single random clause (matching genetic algo weekly.py)."""
        feature = random.choice(feature_names)
        
        # Get threshold from percentile of feature values
        if features_df[feature].nunique() < 2:
            threshold = features_df[feature].iloc[0] if not features_df[feature].empty else 0
        else:
            threshold = np.percentile(features_df[feature].dropna(), random.uniform(10, 90))
        
        operator = ">" if random.random() < 0.5 else "<"
        return f"(features_df['{feature}'] {operator} {threshold:.6f})"
    
    def _mutate_rule(self, rule: str, features_df: pd.DataFrame, feature_names: List[str]) -> str:
        """Mutate a trading rule (matching genetic algo weekly.py)."""
        if random.random() < 0.5 and ("&" in rule or "|" in rule):
            # Replace a random clause
            join_op = "&" if "&" in rule else "|"
            parts = rule.split(f" {join_op} ")
            
            if parts:
                idx = random.randrange(len(parts))
                parts[idx] = self._generate_random_clause(features_df, feature_names)
                return f" {join_op} ".join(p.strip() for p in parts)
        
        # Add a new clause
        new_clause = self._generate_random_clause(features_df, feature_names)
        join_op = " & " if random.random() < 0.5 else " | "
        return f"({rule}) {join_op} {new_clause}"
    
    def _crossover_rules(self, rule1: str, rule2: str, features_df: pd.DataFrame, feature_names: List[str]) -> str:
        """Crossover two trading rules (matching genetic algo weekly.py)."""
        # Extract clauses from both rules
        clauses1 = self._extract_clauses(rule1)
        clauses2 = self._extract_clauses(rule2)
        
        if not clauses1:
            clause1 = rule1
        else:
            clause1 = random.choice(clauses1)
            
        if not clauses2:
            clause2 = rule2
        else:
            clause2 = random.choice(clauses2)
            
        return f"({clause1}) & ({clause2})"
    
    def _extract_clauses(self, rule: str) -> List[str]:
        """Extract individual clauses from a rule."""
        clauses = []
        for sep in [" & ", " | "]:
            if sep in rule:
                parts = rule.replace("(", "").replace(")", "").split(sep)
                clauses.extend([p.strip() for p in parts if p.strip()])
                break
        return clauses
    
    def _evaluate_rule_fitness(self, rule: str, price_series: pd.Series, features_df: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Evaluate fitness of a trading rule (matching genetic algo weekly.py).
        
        Returns:
            Tuple of (fitness, total_return, max_drawdown)
        """
        try:
            # Evaluate rule to get trading signals
            signal_mask = eval(rule, {"pd": pd, "np": np}, {"features_df": features_df})
            signal_mask = signal_mask.reindex(price_series.index).fillna(False)
            
            # Simple backtest calculation (without vectorbt for compatibility)
            returns = price_series.pct_change().fillna(0)
            strategy_returns = returns * signal_mask.shift(1).fillna(False)  # Lag signals by 1 period
            
            # Calculate performance metrics
            total_return = (1 + strategy_returns).prod() - 1
            
            # Calculate max drawdown
            cumulative = (1 + strategy_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
            
            # Fitness = return - penalty * abs(drawdown)
            fitness = total_return - self.fitness_drawdown_penalty_factor * abs(max_drawdown)
            
            return fitness, total_return, max_drawdown
            
        except Exception as e:
            # Return worst possible fitness for invalid rules
            return -np.inf, -np.inf, 1.0
    
    def get_signal_statistics(self, data: pd.DataFrame, features: pd.DataFrame) -> Dict:
        """Get detailed signal statistics for reporting."""
        entry_signals, exit_signals = self.generate_signals(data, features)
        
        return {
            'total_signals': entry_signals.sum(),
            'signal_frequency': entry_signals.mean(),
            'time_in_market': entry_signals.mean(),
            'best_evolved_rule': self.best_rule,
            'best_fitness': self.best_fitness,
            'population_size': self.population_size,
            'max_generations': self.max_generations
        }
    
    def get_strategy_description(self) -> str:
        """Get human-readable strategy description."""
        return f"""Genetic Algorithm Evolved Trading Strategy
        
Strategy Logic (matching genetic algo weekly.py):
- Uses evolutionary algorithm to discover optimal trading rules
- Population Size: {self.population_size}
- Max Generations: {self.max_generations}
- Mutation Rate: {self.mutation_rate:.1%}
- Crossover Rate: {self.crossover_rate:.1%}
- Elite Size: {self.elite_size}
- Trading Asset: {self.trading_asset}

Technical Features Used:
- Momentum: Multiple timeframes (1-52 weeks)
- Volatility: Rolling standard deviations
- SMA Deviations: Price vs moving averages
- MACD: Momentum indicator
- Stochastic K: Overbought/oversold
- Risk Factors: OAS and VIX momentum

Evolved Rule: {self.best_rule if self.best_rule else 'Not yet evolved'}
Best Fitness: {self.best_fitness:.4f}
""" 