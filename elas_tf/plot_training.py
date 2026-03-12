#!/usr/bin/env python3
"""plot_training.py — Live plot of Global Training Steps vs Epochs and Wall Time.

Usage (from the ElasTF project root):
    python3 plot_training.py                         # auto-finds CSV in shared/checkpoints
    python3 plot_training.py shared/checkpoints      # explicit checkpoint dir
    python3 plot_training.py --once                  # plot once and exit (no live update)

The script reads `training_metrics.csv` written by the chief worker inside the
checkpoint directory. It refreshes every 5 seconds while training is running.

Requires: matplotlib (pip install matplotlib)
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import List, Dict


                                                                             
              
                                                                             

CSV_FILENAME = "training_metrics.csv"


def _find_csv(checkpoint_dir: str) -> Path:
    return Path(checkpoint_dir) / CSV_FILENAME


def _load_rows(csv_path: Path) -> List[Dict]:
    if not csv_path.exists():
        return []
    try:
        with open(csv_path, "r", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _parse(rows: List[Dict]):
    """Return (global_steps, epochs, elapsed_times, num_workers_series, train_acc, val_acc)."""
    steps, epochs, times, workers, train_acc, val_acc = [], [], [], [], [], []
    for r in rows:
        try:
            steps.append(int(r["global_step"]))
            epochs.append(int(r["epoch"]))
            times.append(float(r["elapsed_time_s"]))
            workers.append(int(r["num_workers"]))
            train_acc.append(float(r["train_accuracy"]) if r.get("train_accuracy") else None)
            val_acc.append(float(r["val_accuracy"]) if r.get("val_accuracy") else None)
        except (KeyError, ValueError):
            continue
    return steps, epochs, times, workers, train_acc, val_acc


                                                                             
          
                                                                             

def _build_fig(steps, epochs, times, workers, train_acc, val_acc):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D

                                                                              
    BG        = "#0d1117"
    PANEL     = "#161b22"
    GRID      = "#21262d"
    ACCENT1   = "#58a6ff"                           
    ACCENT2   = "#3fb950"                          
    ACCENT3   = "#f78166"                           
    ACCENT4   = "#d2a8ff"                          
    TEXT      = "#e6edf3"
    SUBTEXT   = "#8b949e"
    MARKER    = "#ffa657"                                        

    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    PANEL,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   TEXT,
        "xtick.color":       SUBTEXT,
        "ytick.color":       SUBTEXT,
        "grid.color":        GRID,
        "text.color":        TEXT,
        "font.family":       "monospace",
    })

    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    fig.suptitle(
        "ElasTF  ·  Global Training Steps",
        fontsize=15, fontweight="bold", color=TEXT, y=0.97,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.08, right=0.97, top=0.91, bottom=0.09)

                                                                             
    def _mark_scale_changes(ax, x_vals):
        prev = workers[0] if workers else None
        for i, w in enumerate(workers):
            if i > 0 and w != prev:
                ax.axvline(x=x_vals[i], color=MARKER, linewidth=1.2,
                           linestyle="--", alpha=0.7)
                ax.text(x_vals[i], ax.get_ylim()[1] * 0.95,
                        f" ×{w}w", color=MARKER, fontsize=7, va="top")
            prev = w

                                                                              
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, steps, color=ACCENT1, linewidth=2, zorder=3)
    ax1.scatter(epochs, steps, color=ACCENT1, s=50, zorder=4)
    ax1.set_xlabel("Epoch", fontsize=9)
    ax1.set_ylabel("Global Step", fontsize=9)
    ax1.set_title("Global Step  vs  Epoch", fontsize=10, color=TEXT, pad=8)
    ax1.grid(True, linewidth=0.4)
    _mark_scale_changes(ax1, epochs)

                                                                              
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(times, steps, color=ACCENT2, linewidth=2, zorder=3)
    ax2.scatter(times, steps, color=ACCENT2, s=50, zorder=4)
    ax2.set_xlabel("Wall Time (s)", fontsize=9)
    ax2.set_ylabel("Global Step", fontsize=9)
    ax2.set_title("Global Step  vs  Wall Time", fontsize=10, color=TEXT, pad=8)
    ax2.grid(True, linewidth=0.4)
    _mark_scale_changes(ax2, times)

                                                                              
    ax3 = fig.add_subplot(gs[1, 0])
    clean_train = [(e, a) for e, a in zip(epochs, train_acc) if a is not None]
    clean_val   = [(e, a) for e, a in zip(epochs, val_acc)   if a is not None]
    if clean_train:
        ex, ax_ = zip(*clean_train)
        ax3.plot(ex, ax_, color=ACCENT3, linewidth=2, label="train", zorder=3)
        ax3.scatter(ex, ax_, color=ACCENT3, s=40, zorder=4)
    if clean_val:
        ex, ax_ = zip(*clean_val)
        ax3.plot(ex, ax_, color=ACCENT4, linewidth=2, label="val", zorder=3)
        ax3.scatter(ex, ax_, color=ACCENT4, s=40, zorder=4)
    ax3.set_xlabel("Epoch", fontsize=9)
    ax3.set_ylabel("Accuracy", fontsize=9)
    ax3.set_title("Train / Val Accuracy  vs  Epoch", fontsize=10, color=TEXT, pad=8)
    ax3.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax3.grid(True, linewidth=0.4)
    ax3.set_ylim(0, 1.05)

                                                                              
    ax4 = fig.add_subplot(gs[1, 1])
    if len(steps) >= 2:
        step_deltas = [steps[0]] + [steps[i] - steps[i-1] for i in range(1, len(steps))]
        bar_colors  = [ACCENT1 if d > 0 else MARKER for d in step_deltas]
        bars = ax4.bar(epochs, step_deltas, color=bar_colors, zorder=3, edgecolor=BG, linewidth=0.5)
    ax4.set_xlabel("Epoch", fontsize=9)
    ax4.set_ylabel("Steps in Epoch", fontsize=9)
    ax4.set_title("Steps per Epoch  (cluster size changes → bar color)", fontsize=9, color=TEXT, pad=8)
    ax4.grid(True, axis="y", linewidth=0.4)

                                                                             
    legend_els = [
        Line2D([0], [0], color=MARKER, linewidth=1.5, linestyle="--",
               label="cluster size change"),
    ]
    fig.legend(handles=legend_els, loc="lower right", fontsize=8,
               facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

                                                                              
    if rows := len(steps):
        summary = (
            f"epochs={epochs[-1]}  |  global_step={steps[-1]}  |  "
            f"workers={workers[-1]}  |  elapsed={times[-1]:.0f}s"
        )
        fig.text(0.5, 0.01, summary, ha="center", fontsize=8, color=SUBTEXT)

    return fig


                                                                             
     
                                                                             

def _default_checkpoint_dir() -> str:
                                                                                 
    here = Path(__file__).resolve().parent
    for directory in [here, here.parent, here.parent.parent]:
        candidate = directory / "shared" / "checkpoints"
        if candidate.exists():
            return str(candidate)
                                                                  
    return str(here.parent / "shared" / "checkpoints")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint_dir", nargs="?",
        default=_default_checkpoint_dir(),
        help="Path to the shared/checkpoints directory (default: auto-detected)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Plot once and exit instead of live-updating every 5s",
    )
    parser.add_argument(
        "--interval", type=float, default=5.0,
        help="Refresh interval in seconds for live mode (default: 5)",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found. Install it with:  pip install matplotlib")
        sys.exit(1)

    csv_path = _find_csv(args.checkpoint_dir)
    print(f"[plot] Watching: {csv_path}")

    if args.once:
        rows = _load_rows(csv_path)
        if not rows:
            print("[plot] No data yet — CSV is empty or missing.")
            sys.exit(0)
        steps, epochs, times, workers, train_acc, val_acc = _parse(rows)
        fig = _build_fig(steps, epochs, times, workers, train_acc, val_acc)
        plt.tight_layout()
        plt.show()
        return

                                                    
    plt.ion()
    fig = None
    last_row_count = -1

    print(f"[plot] Live mode — refreshing every {args.interval}s. Ctrl+C to stop.")
    try:
        while True:
            rows = _load_rows(csv_path)
            if len(rows) != last_row_count:
                last_row_count = len(rows)
                if rows:
                    steps, epochs, times, workers, train_acc, val_acc = _parse(rows)
                    if fig is not None:
                        plt.close(fig)
                    fig = _build_fig(steps, epochs, times, workers, train_acc, val_acc)
                    plt.pause(0.1)
                else:
                    print("[plot] Waiting for data…")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[plot] Stopped.")
    finally:
        if fig is not None:
                                                    
            out = csv_path.parent / "training_steps_plot.png"
            fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"[plot] Final plot saved → {out}")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
