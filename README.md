# ElasTF

Elastic Distributed Training with TensorFlow (CS 214 Big Data Systems).

This repository implements the ElasTF project plan:

- **Baseline static distributed training** using `tf.distribute.MultiWorkerMirroredStrategy`
- **Heartbeat-based worker monitoring** to detect joins and failures
- **Checkpointing and restart** using `tf.train.CheckpointManager`
- **Elastic reconfiguration** on worker add/drop via dynamic `TF_CONFIG` regeneration

### Project layout

- `requirements.txt`: Python dependencies
- `elas_tf/`: Python package
  - `training.py`: Baseline distributed training script
  - `controller.py`: Elastic controller and TF_CONFIG manager
  - `worker.py`: Worker entrypoint that wraps the training loop
  - `heartbeat.py`: Heartbeat client/server utilities
  - `checkpointing.py`: Checkpoint manager helpers
  - `tests_functional.py`: Simple functional smoke tests
- `run_cluster.py`: Convenience script to launch controller + 3 workers locally
- `shared/`: Checkpoints and TF_CONFIG state (not for committing)

### Quick start (bare metal, 3 workers)

1. Create and activate a virtualenv, install deps:

   ```bash
   cd ElasTF
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Launch controller + 3 workers in one terminal:

   ```bash
   python run_cluster.py
   ```

   This will start one controller and three worker processes on your machine,
   train MNIST for a few epochs, and print wall time, throughput, and accuracy
   metrics at the end.

Elastic behavior (detecting worker failures/joins and regenerating TF_CONFIG)
is implemented in the controller and can be exercised by killing or adding
worker processes between runs while reusing checkpoints.