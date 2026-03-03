"""
Basic functional scenarios for ElasTF.

These are lightweight entrypoints you can run manually (or wire into CI) to
exercise the main project scenarios:
- Normal static training with two workers
- Simulated worker drop/add while the controller observes membership
"""

import os

from .training import run_baseline_training


def scenario_normal_training() -> None:
    """Run a short training job as a smoke test."""
    os.environ.setdefault("EPOCHS", "1")
    run_baseline_training(epochs=int(os.getenv("EPOCHS", "1")))


def main() -> None:
    scenario_normal_training()


if __name__ == "__main__":
    main()

