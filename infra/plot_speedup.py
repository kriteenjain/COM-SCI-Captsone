"""Plot speedup curve from benchmark results.

Usage:
    python3 infra/plot_speedup.py
    python3 infra/plot_speedup.py --results-dir ./benchmark_results
"""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_wall_time(csv_path: str) -> float:
    """Extract final cumulative wall time from a metrics CSV."""
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0.0
    return float(rows[-1]["elapsed_time_s"])


def main():
    parser = argparse.ArgumentParser(description="Plot ElasTF speedup curve")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "benchmark_results"),
    )
    parser.add_argument("--output", default="speedup_curve.png")
    args = parser.parse_args()

    configs = [1, 2, 4]
    wall_times = {}

    for n in configs:
        path = os.path.join(args.results_dir, f"metrics_{n}w.csv")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping {n}-worker config")
            continue
        wall_times[n] = load_wall_time(path)
        print(f"  {n} worker(s): {wall_times[n]:.1f}s")

    if 1 not in wall_times or len(wall_times) < 2:
        print("Need at least the 1-worker baseline and one multi-worker run.")
        sys.exit(1)

    baseline = wall_times[1]
    workers = sorted(wall_times.keys())
    speedups = [baseline / wall_times[n] for n in workers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar([str(n) for n in workers], [wall_times[n] for n in workers], color="#4A90D9")
    ax1.set_xlabel("Number of Workers (GPUs)")
    ax1.set_ylabel("Wall Time (seconds)")
    ax1.set_title("Training Wall Time vs Workers")
    for i, n in enumerate(workers):
        ax1.text(i, wall_times[n] + 5, f"{wall_times[n]:.0f}s", ha="center", fontweight="bold")

    ax2.plot(workers, speedups, "o-", color="#E74C3C", linewidth=2, markersize=8, label="Actual")
    ax2.plot(workers, workers, "--", color="#95A5A6", label="Ideal (linear)")
    ax2.set_xlabel("Number of Workers (GPUs)")
    ax2.set_ylabel("Speedup (x)")
    ax2.set_title("Speedup Curve")
    ax2.legend()
    ax2.set_xticks(workers)

    plt.suptitle("ElasTF Distributed Training: ResNet-50 on CIFAR-10", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = os.path.join(args.results_dir, args.output)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSpeedup curve saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
