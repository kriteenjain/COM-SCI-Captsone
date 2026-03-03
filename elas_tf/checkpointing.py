import os
from typing import Tuple

import tensorflow as tf


def create_checkpoint_objects(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    checkpoint_dir: str,
) -> Tuple[tf.train.Checkpoint, tf.train.CheckpointManager, tf.Variable]:
    """Create checkpoint, checkpoint manager, and global step for training."""
    global_step = tf.Variable(0, dtype=tf.int64, name="global_step")
    ckpt = tf.train.Checkpoint(step=global_step, optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
    return ckpt, manager, global_step


def restore_latest_if_available(manager: tf.train.CheckpointManager) -> bool:
    """Restore from the latest checkpoint if one exists. Returns True if restored."""
    latest_path = manager.latest_checkpoint
    if latest_path:
        manager.checkpoint.restore(latest_path).expect_partial()
        print(f"[checkpointing] Restored from checkpoint: {latest_path}")
        return True
    print("[checkpointing] No checkpoint found, initializing from scratch.")
    return False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

