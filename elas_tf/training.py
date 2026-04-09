import csv
import glob
import json
import os
import re
import time
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from .checkpointing import ensure_dir

TOTAL_TRAIN_SAMPLES = 50000
BATCH_SIZE = 128
WALL_TIME_FILE = "cumulative_wall_time"
USE_LIGHT_MODEL = os.getenv("LIGHT_MODEL", "0") == "1"

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
    return os.path.join(checkpoint_dir, "training_metrics.csv")


def _load_cifar10() -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE)
    test_ds = test_ds.batch(BATCH_SIZE)
    train_size = x_train.shape[0]
    return train_ds, test_ds, int(train_size)


def _build_model() -> tf.keras.Model:
    if USE_LIGHT_MODEL:
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
    return tf.keras.applications.ResNet50(
        weights=None, input_shape=(32, 32, 3), classes=10
    )


def _get_worker_info() -> Tuple[int, int]:
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


def _count_batches_per_epoch(train_size: int, batch_size: int = BATCH_SIZE) -> int:
    return (train_size + batch_size - 1) // batch_size


def _load_cumulative_wall_time(checkpoint_dir: str) -> float:
    path = os.path.join(checkpoint_dir, WALL_TIME_FILE)
    try:
        with open(path, "r") as f:
            return float(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0.0


def _save_cumulative_wall_time(checkpoint_dir: str, wall_time: float) -> None:
    path = os.path.join(checkpoint_dir, WALL_TIME_FILE)
    with open(path, "w") as f:
        f.write(f"{wall_time:.4f}\n")


def _maybe_upload_checkpoint_to_gcs(checkpoint_dir: str) -> None:
    """If GCS_BUCKET is set, upload checkpoint files after each epoch."""
    bucket_name = os.getenv("GCS_BUCKET")
    if not bucket_name:
        return
    try:
        from .gcs_storage import upload_checkpoint
        upload_checkpoint(checkpoint_dir, bucket_name, "checkpoints")
        print(f"[training] Checkpoint uploaded to gs://{bucket_name}/checkpoints/")
    except Exception as e:
        print(f"[training] WARNING: GCS upload failed: {e}")


def _maybe_download_checkpoint_from_gcs(checkpoint_dir: str) -> None:
    """If GCS_BUCKET is set, download latest checkpoint before training."""
    bucket_name = os.getenv("GCS_BUCKET")
    if not bucket_name:
        return
    try:
        from .gcs_storage import download_latest_checkpoint
        download_latest_checkpoint(bucket_name, "checkpoints", checkpoint_dir)
        print(f"[training] Checkpoint downloaded from gs://{bucket_name}/checkpoints/")
    except Exception as e:
        print(f"[training] WARNING: GCS download failed: {e}")


def _maybe_upload_metrics_to_gcs(csv_path: str) -> None:
    """If GCS_BUCKET is set, upload metrics CSV after each epoch."""
    bucket_name = os.getenv("GCS_BUCKET")
    if not bucket_name:
        return
    try:
        from .gcs_storage import upload_file
        upload_file(csv_path, bucket_name, "metrics/training_metrics.csv")
    except Exception as e:
        print(f"[training] WARNING: GCS metrics upload failed: {e}")


def run_baseline_training(epochs: int = 5) -> None:
    worker_index, num_workers = _get_worker_info()

    print("")
    print("=" * 60)
    print(f"[training] WORKER {worker_index} of {num_workers} starting up")
    print(f"[training] Model: ResNet-50 on CIFAR-10 ({TOTAL_TRAIN_SAMPLES} samples)")
    print("=" * 60)

    tf_config = os.getenv("TF_CONFIG")
    if tf_config:
        print(f"[training] Using TF_CONFIG for distributed training.")
        print(f"[training] Cluster has {num_workers} workers. I am worker {worker_index}.")
    else:
        print(f"[training] No TF_CONFIG set. Running as single worker.")

    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "/shared/checkpoints")
    ensure_dir(checkpoint_dir)

    _maybe_download_checkpoint_from_gcs(checkpoint_dir)

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
    print(f"[training] Model parameters: {model.count_params():,}")

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

    train_ds, test_ds, train_size = _load_cifar10()

    samples_per_worker = train_size // num_workers
    steps_per_epoch = _count_batches_per_epoch(train_size)

    print("")
    print("-" * 60)
    print(f"[training] DATA SHARDING:")
    print(f"[training]   Total training samples: {train_size}")
    print(f"[training]   Number of workers:      {num_workers}")
    print(f"[training]   Samples per worker:     ~{samples_per_worker}")
    print(f"[training]   This worker (index {worker_index}) processes ~{samples_per_worker} samples/epoch")
    print(f"[training]   Batch size:             {BATCH_SIZE}")
    print(f"[training]   Steps per epoch:        {steps_per_epoch}")
    print("-" * 60)
    print("")

    write_dir = _write_checkpoint_dir_for_worker(checkpoint_dir)
    ckpt_path_template = os.path.join(write_dir, "ckpt-{epoch:02d}")
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path_template,
        save_weights_only=True,
    )

    csv_path = _metrics_csv_path(checkpoint_dir)
    is_chief = _is_chief()

    global_step_offset = 0
    if is_chief and os.path.exists(csv_path):
        try:
            with open(csv_path, "r", newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                global_step_offset = int(rows[-1]["global_step"])
        except Exception:
            pass

    if is_chief and not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(_METRICS_HEADER)

    prev_wall_time = _load_cumulative_wall_time(checkpoint_dir)
    gen_start_time = time.time()

    def on_epoch_begin(epoch, logs=None):
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

        epochs_done_this_run = (epoch + 1) - completed_epochs
        current_global_step = global_step_offset + epochs_done_this_run * steps_per_epoch
        elapsed = time.time() - gen_start_time
        cumulative_elapsed = prev_wall_time + elapsed

        _save_cumulative_wall_time(checkpoint_dir, cumulative_elapsed)

        print("")
        print("+" * 60)
        print(f"[checkpoint] EPOCH {epoch + 1}/{epochs} COMPLETE")
        print(f"[checkpoint]   Global step:    {current_global_step}")
        print(f"[checkpoint]   Elapsed time:   {cumulative_elapsed:.1f}s")
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

        if is_chief:
            row = [
                current_global_step,
                epoch + 1,
                round(cumulative_elapsed, 3),
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
            print(f"[metrics]  Row appended -> {csv_path}")

            _maybe_upload_checkpoint_to_gcs(checkpoint_dir)
            _maybe_upload_metrics_to_gcs(csv_path)

    epoch_log_cb = tf.keras.callbacks.LambdaCallback(
        on_epoch_begin=on_epoch_begin,
        on_epoch_end=on_epoch_end,
    )

    remaining_epochs = epochs - completed_epochs
    print(f"[training] Starting training: epoch {completed_epochs + 1} to {epochs} ({remaining_epochs} epochs remaining)")

    if prev_wall_time > 0:
        print(f"[training] Previous cumulative wall time: {prev_wall_time:.2f}s")

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        initial_epoch=completed_epochs,
        callbacks=[checkpoint_cb, epoch_log_cb],
    )

    generation_wall_time = time.time() - gen_start_time
    cumulative_wall_time = prev_wall_time + generation_wall_time
    _save_cumulative_wall_time(checkpoint_dir, cumulative_wall_time)

    total_samples = train_size * remaining_epochs
    throughput = total_samples / generation_wall_time if generation_wall_time > 0 else 0.0
    overall_throughput = (train_size * epochs) / cumulative_wall_time if cumulative_wall_time > 0 else 0.0

    train_acc = history.history.get("accuracy", [None])[-1]
    val_acc = history.history.get("val_accuracy", [None])[-1]

    print("")
    print("=" * 60)
    print(f"[training] TRAINING COMPLETE (worker {worker_index})")
    print(f"[training]   Model:                 ResNet-50 on CIFAR-10")
    print(f"[training]   Epochs trained:        {remaining_epochs} (epoch {completed_epochs + 1} to {epochs})")
    print(f"[training]   This generation time:  {generation_wall_time:.2f}s")
    print(f"[training]   Cumulative wall time:  {cumulative_wall_time:.2f}s")
    print(f"[training]   Generation throughput:  {throughput:.1f} samples/sec")
    print(f"[training]   Overall throughput:     {overall_throughput:.1f} samples/sec")
    print(f"[training]   Final train accuracy:   {train_acc:.4f}" if train_acc else "[training]   Final train accuracy:   N/A")
    print(f"[training]   Final val accuracy:     {val_acc:.4f}" if val_acc else "[training]   Final val accuracy:     N/A")
    print(f"[training]   Workers in cluster:     {num_workers}")
    print(f"[training]   Samples/worker/epoch:   ~{samples_per_worker}")
    print("=" * 60)
    print("")


def _get_strategy() -> tf.distribute.Strategy:
    return tf.distribute.MultiWorkerMirroredStrategy()


def main() -> None:
    epochs = int(os.getenv("EPOCHS", "10"))
    run_baseline_training(epochs=epochs)


if __name__ == "__main__":
    main()
