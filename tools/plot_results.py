"""
Plot training results from CSV log.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_results(csv_path: str, save_path: str):
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss curves
    ax1.plot(df['step'], df['train_loss'], label='Train Loss')
    ax1.plot(df['step'], df['val_loss'], label='Val Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Perplexity curve
    ax2.plot(df['step'], df['perplexity'], color='orange')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Perplexity Over Time')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    plot_results("research/logs/training_log.csv", "research/logs/training_curve.png")