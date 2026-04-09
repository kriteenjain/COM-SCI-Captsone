"""Plot elastic scaling benchmark results.

Produces a bar chart comparing total training time across 4 scenarios:
  - Baseline (2 workers, static)
  - Scale-down (2 → 1 worker mid-training)
  - Scale-up to 3 (2 → 3 workers mid-training)
  - Scale-up to 4 (2 → 4 workers mid-training)

Also produces a per-epoch timeline showing worker count transitions.

Usage:
    python3 infra/plot_elastic.py
    python3 infra/plot_elastic.py --results-dir ./elastic_results
"""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


SCENARIOS = [
    ("baseline_2w", "Baseline\n(2 workers)"),
    ("scale_down_2to1", "Scale-down\n(2→1)"),
    ("scale_up_2to3", "Scale-up\n(2→3)"),
    ("scale_up_2to4", "Scale-up\n(2→4)"),
]

COLORS = {
    "baseline_2w": "#4A90D9",
    "scale_down_2to1": "#E74C3C",
    "scale_up_2to3": "#27AE60",
    "scale_up_2to4": "#2ECC71",
}


def load_scenario_data(csv_path: str) -> list[dict]:
    with open(csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Plot ElasTF elastic benchmark")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "elastic_results"),
    )
    parser.add_argument("--output", default="elastic_comparison.png")
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)

    scenario_data = {}
    wall_times = {}
    for key, _ in SCENARIOS:
        path = os.path.join(results_dir, f"{key}.csv")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        data = load_scenario_data(path)
        if not data:
            print(f"  WARNING: {path} is empty, skipping")
            continue
        scenario_data[key] = data
        wall_times[key] = float(data[-1]["elapsed_time_s"])
        print(f"  {key}: {wall_times[key]:.1f}s ({len(data)} epochs)")

    if len(wall_times) < 2:
        print("Need at least 2 scenarios to compare. Run the benchmark first.")
        sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- Left panel: total training time bar chart ---
    ax1 = axes[0]
    present = [(k, l) for k, l in SCENARIOS if k in wall_times]
    labels = [l for _, l in present]
    keys = [k for k, _ in present]
    times = [wall_times[k] for k in keys]
    colors = [COLORS.get(k, "#888") for k in keys]

    bars = ax1.bar(labels, times, color=colors, edgecolor="white", linewidth=1.2)
    ax1.set_ylabel("Total Training Time (seconds)", fontsize=12)
    ax1.set_title("Total Training Time by Scenario", fontsize=13, fontweight="bold")

    for bar, t in zip(bars, times):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(times) * 0.02,
            f"{t:.0f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    if "baseline_2w" in wall_times:
        baseline = wall_times["baseline_2w"]
        ax1.axhline(y=baseline, color="#4A90D9", linestyle="--", alpha=0.5, linewidth=1)
        ax1.text(
            len(labels) - 0.5, baseline * 1.02,
            "baseline", color="#4A90D9", alpha=0.7, fontsize=9,
        )

    ax1.set_ylim(0, max(times) * 1.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- Right panel: per-epoch timeline ---
    ax2 = axes[1]
    for key, label in present:
        if key not in scenario_data:
            continue
        data = scenario_data[key]
        epochs = [int(row["epoch"]) for row in data]
        elapsed = [float(row["elapsed_time_s"]) for row in data]
        num_workers = [int(row["num_workers"]) for row in data]

        ax2.plot(
            epochs, elapsed,
            "o-", color=COLORS.get(key, "#888"),
            linewidth=2, markersize=5,
            label=label.replace("\n", " "),
        )

        for i, (e, t, nw) in enumerate(zip(epochs, elapsed, num_workers)):
            if i > 0 and nw != int(data[i - 1]["num_workers"]):
                ax2.annotate(
                    f"{nw}w",
                    (e, t),
                    textcoords="offset points",
                    xytext=(8, 8),
                    fontsize=8,
                    fontweight="bold",
                    color=COLORS.get(key, "#888"),
                    arrowprops=dict(arrowstyle="->", color=COLORS.get(key, "#888"), lw=0.8),
                )

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Cumulative Wall Time (seconds)", fontsize=12)
    ax2.set_title("Training Progress per Epoch", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.suptitle(
        "ElasTF: Elastic Scaling Impact on Training Time",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    output_path = os.path.join(results_dir, args.output)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {output_path}")

    summary_path = os.path.join(results_dir, "summary.csv")
    if os.path.exists(summary_path):
        print("\nSummary:")
        with open(summary_path) as f:
            for line in f:
                print(f"  {line.strip()}")


if __name__ == "__main__":
    main()
