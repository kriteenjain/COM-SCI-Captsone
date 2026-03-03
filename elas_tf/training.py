import os
import time
from typing import Tuple

import numpy as np
import tensorflow as tf

from .checkpointing import create_checkpoint_objects, ensure_dir, restore_latest_if_available


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


def _get_strategy() -> tf.distribute.Strategy:
    # MultiWorkerMirroredStrategy uses TF_CONFIG for cluster coordination.
    return tf.distribute.MultiWorkerMirroredStrategy()


def _is_chief() -> bool:
    """Return True if this process is the chief worker for checkpointing."""
    tf_config = os.getenv("TF_CONFIG")
    if not tf_config:
        # Single-worker case.
        return True

    # Parse TF_CONFIG directly to determine task index.
    import json

    try:
        cfg = json.loads(tf_config)
        task = cfg.get("task", {})
        task_type = task.get("type")
        task_index = int(task.get("index", 0))
        return task_type == "worker" and task_index == 0
    except Exception:
        # If we cannot reliably determine, default to True to avoid skipping checkpoints.
        return True


def run_baseline_training(epochs: int = 5) -> None:
    """Run a baseline MultiWorkerMirroredStrategy training job on MNIST with checkpointing."""
    tf_config = os.getenv("TF_CONFIG")
    if tf_config:
        print(f"[training] Using TF_CONFIG environment.")
    else:
        print("[training] WARNING: TF_CONFIG not set, running as single-worker.")

    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "/shared/checkpoints")
    ensure_dir(checkpoint_dir)

    strategy = _get_strategy()

    with strategy.scope():
        model = _build_model()
        optimizer = tf.keras.optimizers.Adam()
        ckpt, manager, global_step = create_checkpoint_objects(model, optimizer, checkpoint_dir)

        restored = restore_latest_if_available(manager)
        if restored:
            print(f"[training] Resumed training at global_step={int(global_step.numpy())}.")

        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    train_ds, test_ds, train_size = _load_mnist()

    print("[training] Starting training...")
    start_time = time.time()

    def on_epoch_end(epoch, logs=None):
        # Only the chief worker writes checkpoints to avoid contention.
        if _is_chief():
            global_step.assign_add(1)
            save_path = manager.save()
            print(f"[training] Epoch {epoch} complete. Saved checkpoint to {save_path}")

    callbacks = [tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)]

    history = model.fit(train_ds, validation_data=test_ds, epochs=epochs, callbacks=callbacks)

    wall_time = time.time() - start_time
    total_samples = train_size * epochs
    throughput = total_samples / wall_time if wall_time > 0 else 0.0

    # Extract final accuracies if present.
    train_acc = history.history.get("accuracy", [None])[-1]
    val_acc = history.history.get("val_accuracy", [None])[-1]

    print("[training] Training complete.")
    print(f"[training] Wall time: {wall_time:.2f}s, throughput: {throughput:.1f} samples/sec (approx).")
    print(f"[training] Final train accuracy: {train_acc}, final val accuracy: {val_acc}")


def main() -> None:
    epochs = int(os.getenv("EPOCHS", "3"))
    run_baseline_training(epochs=epochs)


if __name__ == "__main__":
    main()

