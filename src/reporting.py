import matplotlib.pyplot as plt
import os

def print_metrics(metrics):
    print("\nPerformance Metrics\n-------------------")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:8.4%}")
    print("\n[DEBUG] Metrics printed successfully!")


def plot_results(equity_curve, drawdown=None):
    print("[DEBUG] Creating plot with data shapes:", 
          f"equity_curve: {equity_curve.shape}, "
          f"drawdown: {None if drawdown is None else drawdown.shape}")
    
    plt.figure(figsize=(10,6))
    plt.plot(equity_curve, label="Strategy Equity")
    if drawdown is not None:
        plt.plot(drawdown, label="Drawdown")
    plt.legend()
    plt.title("Equity Curve and Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.tight_layout()
    
    # Save plot to file instead of displaying
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "momentum_strategy_plot.png")
    plt.savefig(output_path)
    print(f"[DEBUG] Plot saved to {output_path}")
    
    # Still try show() in case environment supports it
    try:
        plt.show(block=False)
    except Exception as e:
        print(f"[DEBUG] Show plot failed (this may be normal): {e}")
    finally:
        plt.close()
