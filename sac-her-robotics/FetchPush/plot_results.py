"""Generate training curve plots for the FetchPush README."""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
OUT_DIR     = RESULTS_DIR

EVAL_FREQ   = 5_000   # steps between evaluations
STYLE       = 'seaborn-v0_8-darkgrid'

# ── helpers ──────────────────────────────────────────────────────────────────

def read_csv_column(path: str, column: str) -> list[float]:
    """Return a single column from a CSV file as a list of floats."""
    with open(path, newline='') as f:
        return [float(row[column]) for row in csv.DictReader(f)]

def smooth(values: list[float], window: int) -> np.ndarray:
    """Apply a uniform moving-average with the given window size."""
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')

# ── figure 1: success rate ────────────────────────────────────────────────────

def plot_success_rate(run_dir: str):
    values = read_csv_column(os.path.join(run_dir, 'success_ratio.csv'), 'Success Ratio')
    steps  = [i * EVAL_FREQ / 1_000 for i in range(len(values))]

    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(steps, values, color='#2196F3', linewidth=2, marker='o', markersize=3)

    ax.set_xlabel('Environment steps (×10³)', fontsize=12)
    ax.set_ylabel('Success rate',             fontsize=12)
    ax.set_title('FetchPush-v4 – Success Rate during Training', fontsize=13, fontweight='bold')
    ax.set_xlim(0, steps[-1])
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    fig.tight_layout()

    out_path = os.path.join(run_dir, 'success_rate.png')
    fig.savefig(out_path, dpi=150)
    print(f'Saved: {out_path}')
    plt.close(fig)

# ── figure 2: training losses ────────────────────────────────────────────────

def plot_losses(run_dir: str):
    critic_raw = read_csv_column(os.path.join(run_dir, 'losses.csv'), 'Critic Loss')
    actor_raw  = read_csv_column(os.path.join(run_dir, 'losses.csv'), 'Actor Loss')
    alpha_raw  = read_csv_column(os.path.join(run_dir, 'losses.csv'), 'Log_Alpha Loss')

    window = 500
    critic = smooth(critic_raw, window)
    actor  = smooth(actor_raw,  window)
    alpha  = smooth(alpha_raw,  window)

    steps = np.arange(len(critic)) / 1_000

    plt.style.use(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    configs = [
        (axes[0], critic, '#2196F3', 'Critic Loss'),
        (axes[1], actor,  '#4CAF50', 'Actor Loss'),
        (axes[2], alpha,  '#FF9800', 'Alpha Loss'),
    ]

    for ax, data, color, title in configs:
        ax.plot(steps, data, color=color, linewidth=1.5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Updates (×10³)', fontsize=10)
        ax.set_ylabel('Loss',           fontsize=10)
        ax.set_xlim(0, steps[-1])

    fig.suptitle('FetchPush-v4 – Training Losses (smoothed, window=500)', fontsize=13)
    fig.tight_layout()

    out_path = os.path.join(run_dir, 'losses.png')
    fig.savefig(out_path, dpi=150)
    print(f'Saved: {out_path}')
    plt.close(fig)

# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    run_dir = sys.argv[1] if len(sys.argv) > 1 else RESULTS_DIR
    plot_success_rate(run_dir)
    plot_losses(run_dir)
    print('Done.')
