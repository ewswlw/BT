import random
import numpy as np

def momentum_strategy(feat, lag=4):
    """
    Simple momentum: go long if momentum over 'lag' weeks is positive.
    Matches notebook logic exactly.
    """
    signal = feat[f"mom_{lag}"] > 0
    print(f"[STRATEGY] Signal (first 10):\n{signal.head(10)}")
    return signal

def rand_clause(FEATURES, feat):
    f = random.choice(FEATURES)
    thr = np.percentile(feat[f], random.uniform(10,90))
    op = ">" if random.random() < 0.5 else "<"
    return f"(feat['{f}'] {op} {thr:.6f})"

def gen_rule(FEATURES, feat, max_c=4):
    n = random.randint(1, max_c)
    join = " & " if random.random() < 0.5 else " | "
    return join.join(rand_clause(FEATURES, feat) for _ in range(n))

def evaluate(rule, feat, price, vbt):
    """
    Evaluates a rule string, runs backtest, returns fitness, return, drawdown, pf.
    """
    try:
        mask = eval(rule)
        mask = mask.reindex(price.index).fillna(False)
        pf = vbt.Portfolio.from_signals(
            price,
            entries=mask,
            exits=~mask,
            freq='W',
            init_cash=10000,
            fees=0.0,
            slippage=0.0
        )
        ret = pf.total_return()
        dd = pf.max_drawdown()
        fitness = ret - 0.1 * abs(dd)
        return fitness, ret, dd, pf
    except Exception as e:
        print(f"Error evaluating rule: {rule}\n{e}")
        return -1, -1, 0, None

def mutate(rule, FEATURES, feat):
    # Placeholder for mutation logic
    pass

def crossover(rule1, rule2):
    # Placeholder for crossover logic
    pass
