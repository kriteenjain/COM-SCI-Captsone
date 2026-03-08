import csv
import glob
import json
import os
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from .checkpointing import ensure_dir

TOTAL_TRAIN_SAMPLES = 60000

# CSV header written once per run (chief worker only).
_METRICS_HEADER = [
    "global_step",
    "epoch",
    "elapsed_time_s",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "num_workers",
    "worker_index",
    "samples_per_worker",
]


def _metrics_csv_path(checkpoint_dir: str) -> str:
    """Shared CSV file written by the chief worker for live plotting."""
    return os.path.join(checkpoint_dir, "training_metrics.csv")


def _load_mnist() -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    batch_size = 64
    train_ds = train_ds.shuffle(10000).batch(batch_size)
    test_ds = test_ds.batch(batch_size)
    train_size = x_train.shape[0]
    return train_ds, test_ds, int(train_size)


def _build_model() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )


def _get_worker_info() -> Tuple[int, int]:
    """Return (worker_index, num_workers) from TF_CONFIG."""
    tf_config = os.getenv("TF_CONFIG")
    if not tf_config:
        return 0, 1
    try:
        cfg = json.loads(tf_config)
        task = cfg.get("task", {})
        index = int(task.get("index", 0))
        num_workers = len(cfg.get("cluster", {}).get("worker", []))
        return index, max(num_workers, 1)
    except Exception:
        return 0, 1


def _is_chief() -> bool:
    idx, _ = _get_worker_info()
    return idx == 0


def _write_checkpoint_dir_for_worker(checkpoint_dir: str) -> str:
    if _is_chief():
        return checkpoint_dir
    task_dir = os.path.join(checkpoint_dir, "temp_worker")
    ensure_dir(task_dir)
    return task_dir


def _find_latest_checkpoint(checkpoint_dir: str) -> Tuple[Optional[str], int]:
    pattern = os.path.join(checkpoint_dir, "ckpt-*.index")
    files = glob.glob(pattern)
    if not files:
        return None, 0

    best_epoch = 0
    best_path = None
    for f in files:
        match = re.search(r"ckpt-(\d+)\.index$", f)
        if match:
            epoch = int(match.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_path = f.replace(".index", "")

    return best_path, best_epoch


def _count_batches_per_epoch(train_size: int, batch_size: int = 64) -> int:
    """Number of gradient steps (batches) per epoch."""
    return (train_size + batch_size - 1) // batch_size


def run_baseline_training(epochs: int = 5) -> None:
    worker_index, num_workers = _get_worker_info()

    print("")
    print("=" * 60)
    print(f"[training] WORKER {worker_index} of {num_workers} starting up")
    print("=" * 60)

    tf_config = os.getenv("TF_CONFIG")
    if tf_config:
        print(f"[training] Using TF_CONFIG for distributed training.")
        print(f"[training] Cluster has {num_workers} workers. I am worker {worker_index}.")
    else:
        print(f"[training] No TF_CONFIG set. Running as single worker.")

    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "/shared/checkpoints")
    ensure_dir(checkpoint_dir)

    print(f"[training] Setting up MultiWorkerMirroredStrategy...")
    strategy = _get_strategy()

    with strategy.scope():
        model = _build_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    print(f"[training] Model compiled under distributed strategy.")

    # Restore from the latest checkpoint if one exists.
    latest_ckpt, completed_epochs = _find_latest_checkpoint(checkpoint_dir)
    if latest_ckpt:
        model.load_weights(latest_ckpt)
        print("")
        print("+" * 60)
        print(f"[training] CHECKPOINT RESTORED: {latest_ckpt}")
        print(f"[training] Already completed {completed_epochs} of {epochs} epochs.")
        print(f"[training] Resuming training from epoch {completed_epochs + 1}.")
        print("+" * 60)
        print("")
    else:
        print(f"[training] No checkpoint found. Training from scratch (epoch 1).")

    if completed_epochs >= epochs:
        print(f"[training] Already completed all {epochs} epochs. Nothing to do.")
        return

    train_ds, test_ds, train_size = _load_mnist()

    # Data sharding info.
    samples_per_worker = train_size // num_workers
    steps_per_epoch = _count_batches_per_epoch(train_size)

    print("")
    print("-" * 60)
    print(f"[training] DATA SHARDING:")
    print(f"[training]   Total training samples: {train_size}")
    print(f"[training]   Number of workers:      {num_workers}")
    print(f"[training]   Samples per worker:     ~{samples_per_worker}")
    print(f"[training]   This worker (index {worker_index}) processes ~{samples_per_worker} samples/epoch")
    print(f"[training]   Steps per epoch:        {steps_per_epoch}")
    print("-" * 60)
    print("")

    write_dir = _write_checkpoint_dir_for_worker(checkpoint_dir)
    ckpt_path_template = os.path.join(write_dir, "ckpt-{epoch:02d}")
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path_template,
        save_weights_only=True,
    )

    # ------------------------------------------------------------------
    # Metrics CSV setup (chief only — avoids races in multi-worker runs).
    # global_step counts optimizer steps across the entire training run,
    # including steps from previous generations (after elastic recovery).
    # ------------------------------------------------------------------
    csv_path = _metrics_csv_path(checkpoint_dir)
    is_chief = _is_chief()

    # Load the last recorded global_step and elapsed_time from a prior generation
    # so both accumulate monotonically across elastic restarts.
    global_step_offset = 0
    elapsed_time_offset = 0.0
    if is_chief and os.path.exists(csv_path):
        try:
            with open(csv_path, "r", newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                global_step_offset = int(rows[-1]["global_step"])
                elapsed_time_offset = float(rows[-1]["elapsed_time_s"])
        except Exception:
            pass

    if is_chief and not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(_METRICS_HEADER)

    # Wall-clock reference for this generation (seconds since training start).
    run_start_time = time.time()

    # Epoch-level start time for per-epoch elapsed tracking.
    epoch_start_times: dict = {}

    def on_epoch_begin(epoch, logs=None):
        epoch_start_times[epoch] = time.time()
        print("")
        print(f"[checkpoint] --- Epoch {epoch + 1}/{epochs} starting ---")
        print(f"[checkpoint]   Workers in cluster: {num_workers}")
        print(f"[checkpoint]   Samples this worker will process: ~{samples_per_worker}")

    def on_epoch_end(epoch, logs=None):
        loss = logs.get("loss", "?")
        acc = logs.get("accuracy", "?")
        val_loss = logs.get("val_loss", "?")
        val_acc = logs.get("val_accuracy", "?")
        ckpt_file = ckpt_path_template.format(epoch=epoch + 1)

        # global_step = steps already done before this run
        #             + steps completed in this run up to and including this epoch.
        # "epoch" here is 0-indexed (Keras convention), but completed_epochs
        # is the number of epochs already done when this generation started.
        epochs_done_this_run = (epoch + 1) - completed_epochs
        current_global_step = global_step_offset + epochs_done_this_run * steps_per_epoch
        elapsed = elapsed_time_offset + (time.time() - run_start_time)

        print("")
        print("+" * 60)
        print(f"[checkpoint] EPOCH {epoch + 1}/{epochs} COMPLETE")
        print(f"[checkpoint]   Global step:    {current_global_step}")
        print(f"[checkpoint]   Elapsed time:   {elapsed:.1f}s")
        print(f"[checkpoint]   Train loss:     {loss}")
        print(f"[checkpoint]   Train accuracy: {acc}")
        print(f"[checkpoint]   Val loss:       {val_loss}")
        print(f"[checkpoint]   Val accuracy:   {val_acc}")
        if is_chief:
            print(f"[checkpoint]   Checkpoint saved to: {ckpt_file}")
        else:
            print(f"[checkpoint]   (Non-chief worker, checkpoint written by chief)")
        print("+" * 60)
        print("")

        # Write a row to the shared CSV (chief only).
        if is_chief:
            row = [
                current_global_step,
                epoch + 1,
                round(elapsed, 3),
                loss if isinstance(loss, float) else "",
                acc if isinstance(acc, float) else "",
                val_loss if isinstance(val_loss, float) else "",
                val_acc if isinstance(val_acc, float) else "",
                num_workers,
                worker_index,
                samples_per_worker,
            ]
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            print(f"[metrics]  Row appended → {csv_path}")

    epoch_log_cb = tf.keras.callbacks.LambdaCallback(
        on_epoch_begin=on_epoch_begin,
        on_epoch_end=on_epoch_end,
    )

    remaining_epochs = epochs - completed_epochs
    print(f"[training] Starting training: epoch {completed_epochs + 1} to {epochs} ({remaining_epochs} epochs remaining)")
    start_time = time.time()

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        initial_epoch=completed_epochs,
        callbacks=[checkpoint_cb, epoch_log_cb],
    )

    wall_time = time.time() - start_time
    total_samples = train_size * remaining_epochs
    throughput = total_samples / wall_time if wall_time > 0 else 0.0

    train_acc = history.history.get("accuracy", [None])[-1]
    val_acc = history.history.get("val_accuracy", [None])[-1]

    print("")
    print("=" * 60)
    print(f"[training] TRAINING COMPLETE (worker {worker_index})")
    print(f"[training]   Epochs trained:        {remaining_epochs} (epoch {completed_epochs + 1} to {epochs})")
    print(f"[training]   Wall time:             {wall_time:.2f}s")
    print(f"[training]   Throughput:             {throughput:.1f} samples/sec")
    print(f"[training]   Final train accuracy:   {train_acc:.4f}" if train_acc else "[training]   Final train accuracy:   N/A")
    print(f"[training]   Final val accuracy:     {val_acc:.4f}" if val_acc else "[training]   Final val accuracy:     N/A")
    print(f"[training]   Workers in cluster:     {num_workers}")
    print(f"[training]   Samples/worker/epoch:   ~{samples_per_worker}")
    print("=" * 60)
    print("")


def _get_strategy() -> tf.distribute.Strategy:
    return tf.distribute.MultiWorkerMirroredStrategy()


def main() -> None:
    epochs = int(os.getenv("EPOCHS", "3"))
    run_baseline_training(epochs=epochs)


if __name__ == "__main__":
    main()
