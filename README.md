# ElasTF

Elastic Distributed Training with TensorFlow (CS 214 Big Data Systems).

ElasTF demonstrates fault-tolerant, elastic distributed deep learning on a local machine. A central **controller** monitors worker health via heartbeats, dynamically generates `TF_CONFIG` for TensorFlow's `MultiWorkerMirroredStrategy`, and a **supervisor** script automatically recovers from failures and adapts to scaling events — resuming training from the latest checkpoint with however many workers remain (or are added).

## Features

- **Distributed training** using `tf.distribute.MultiWorkerMirroredStrategy` on MNIST
- **Heartbeat-based failure detection** — TCP heartbeat protocol with a standalone sender process that survives TF cascade crashes, enabling the controller to distinguish intentionally killed workers from crash victims
- **Automatic checkpointing** via `tf.keras.callbacks.ModelCheckpoint` with chief-worker write coordination
- **Scale down** — kill any number of workers (Ctrl+C or `kill_workers.sh`); the controller detects the failure via heartbeat timeout, and the supervisor restarts surviving workers in their existing terminals from the latest checkpoint
- **Scale up** — add a worker at any time (`add_worker.sh`); the controller registers it immediately and the cluster reconfigures without waiting for a heartbeat timeout
- **Dynamic `TF_CONFIG`** — the controller regenerates cluster configuration on every membership or port change, with generation tracking
- **In-terminal restart** — surviving workers restart inside their existing terminal windows (no new terminal spawning on recovery)
- **Stabilization window** — after a failure, the controller waits 10 seconds to collect all survivor heartbeats before reporting, preventing premature recovery decisions

## Architecture

```
┌─────────────┐     heartbeats (TCP)     ┌────────────┐
│   Worker 0  │ ──────────────────────── │            │
│  (Terminal) │                          │ Controller │
├─────────────┤                          │  (port     │
│   Worker 1  │ ──────────────────────── │   6000)    │
│  (Terminal) │                          │            │
├─────────────┤                          │  Writes:   │
│   Worker 2  │ ──────────────────────── │  tf_config │
│  (Terminal) │                          │  remaining │
└─────────────┘                          └─────┬──────┘
       ▲                                       │
       │  restart signals (file-based)         │ remaining_workers
       │                                       ▼
       │                                ┌──────────────┐
       └─────────────────────────────── │  Supervisor  │
                                        │  (launch.sh) │
                                        └──────────────┘
```

Each worker runs three processes managed by a bash wrapper:

1. **Bash wrapper** — loops through training generations, handles restart signals and Ctrl+C
2. **Heartbeat sender** (`heartbeat_sender.py`) — standalone process that sends TCP heartbeats every 2 seconds; ignores SIGINT/SIGTERM/SIGHUP so it survives TF cascade crashes and can only be stopped by `kill -9`
3. **Python training** (`worker.py` + `training.py`) — runs `MultiWorkerMirroredStrategy` training on MNIST

## Project Layout

```
ElasTF/
├── launch.sh               # Supervisor: orchestrates controller + workers
├── kill_workers.sh          # Kill specific workers to simulate failure
├── add_worker.sh            # Add a worker to a running cluster
├── requirements.txt         # Python dependencies
├── elas_tf/                 # Main Python package
│   ├── __init__.py
│   ├── controller.py        # Heartbeat monitor + TF_CONFIG writer
│   ├── worker.py            # Worker entrypoint (load config, launch training)
│   ├── training.py          # MNIST model + MultiWorkerMirroredStrategy loop
│   ├── heartbeat.py         # TCP heartbeat server/client protocol
│   ├── heartbeat_sender.py  # Standalone heartbeat sender (separate process)
│   └── checkpointing.py     # Checkpoint manager helpers
└── shared/                  # Runtime state (gitignored)
    ├── checkpoints/         # Model checkpoints
    └── config/              # Generated TF_CONFIG, signals, PID files
```

## Prerequisites

- **macOS** (the supervisor uses `open -a Terminal` / `osascript` to launch windows)
- **Python 3.10+** (tested with 3.10 on Apple Silicon)
- **TensorFlow 2.15** installed as `tensorflow-macos`

## Setup

1. Clone the repository and create a virtual environment:

   ```bash
   cd ElasTF
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install `tensorflow-macos` and other dependencies:

   ```bash
   pip install tensorflow-macos==2.15.0
   pip install -r requirements.txt
   ```

3. Fix SSL certificates so the MNIST dataset can be downloaded:

   ```bash
   pip install certifi
   ```

   Then add the following line to your `~/.zshrc` (or run it before launching):

   ```bash
   export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")
   ```

   Apply it immediately with `source ~/.zshrc` or open a new terminal.

## Running

### Start the cluster

```bash
chmod +x launch.sh kill_workers.sh add_worker.sh
./launch.sh
```

What happens:

1. The supervisor kills any leftover ElasTF processes (including zombie heartbeat senders) and clears old state.
2. A **controller** terminal opens — it listens for heartbeats on port 6000.
3. Three **worker** terminals open — each starts a heartbeat sender, registers with the controller, loads the generated `TF_CONFIG`, and begins distributed training on MNIST.
4. The supervisor prints commands you can use to simulate failures.

Configuration is at the top of `launch.sh`:

| Variable        | Default | Description                                        |
|-----------------|---------|----------------------------------------------------|
| `NUM_WORKERS`   | 3       | Initial number of workers                          |
| `EPOCHS`        | 5       | Training epochs per generation                     |
| `HEARTBEAT_PORT`| 6000    | Controller heartbeat listen port                   |
| `STARTUP_SLEEP` | 15      | Seconds workers wait for all peers before training |

### Simulate a failure (scale down)

**Option A — Ctrl+C in a worker terminal:**

Press Ctrl+C in any worker terminal window. That worker's bash wrapper traps the signal, kills both the heartbeat sender and python process, and exits. The controller detects the missing heartbeat after ~15 seconds, and the supervisor restarts surviving workers.

**Option B — `kill_workers.sh`:**

From a separate terminal:

```bash
cd ~/Downloads/comsci214/ElasTF

# Kill worker 0 (restart with workers 1 and 2):
./kill_workers.sh 0

# Kill workers 0 and 1 (restart with worker 2 only):
./kill_workers.sh 0 1
```

The script kills each worker's bash wrapper and all its children (heartbeat sender + python). Works with any worker IDs, including non-contiguous ones after previous scale events.

**Recovery timeline (scale down):**

| Phase                | Duration | What happens                                          |
|----------------------|----------|-------------------------------------------------------|
| Heartbeat timeout    | ~15s     | Controller waits for heartbeats that never arrive     |
| Stabilization window | ~10s     | Controller confirms no more workers are dropping      |
| Supervisor picks up  | ~2s      | Supervisor reads `remaining_workers`, sends signals   |
| Workers restart      | ~20s     | Surviving workers restart training from checkpoint    |

### Add a worker (scale up)

While training is running, from a separate terminal:

```bash
./add_worker.sh
```

What happens:

1. Reads `tf_config.json` to find current workers and determines the next ID.
2. Opens a new terminal for the new worker with a heartbeat sender.
3. Waits for the controller to register the new worker (~2-5s).
4. Stops existing workers' python training (heartbeat senders stay alive).
5. Writes restart signals so all workers pick up the new cluster config.
6. All workers (old + new) restart training together from the latest checkpoint.

**Recovery timeline (scale up):**

| Phase                  | Duration | What happens                                          |
|------------------------|----------|-------------------------------------------------------|
| New worker registered  | ~2-5s    | Controller sees join heartbeat immediately            |
| Existing workers stop  | ~1s      | Training processes killed, heartbeats stay alive      |
| Workers restart        | ~20s     | All workers restart training from checkpoint          |

Scale up is faster than scale down because detecting presence (a join heartbeat) is instant, while detecting absence (a missing heartbeat) requires a timeout.

### Running components manually

Start the controller:

```bash
source .venv/bin/activate
export CONFIG_DIR=shared/config CHECKPOINT_DIR=shared/checkpoints HEARTBEAT_PORT=6000
python3 -m elas_tf.controller
```

Start each worker in a separate terminal:

```bash
source .venv/bin/activate
export WORKER_ID=0 CONTROLLER_HOST=localhost WORKER_HOST=localhost
export HEARTBEAT_PORT=6000 CONFIG_DIR=shared/config CHECKPOINT_DIR=shared/checkpoints
export TF_PORT=30000 STARTUP_SLEEP_SECS=15 EPOCHS=5
python3 -m elas_tf.worker
```

Increment `WORKER_ID` and `TF_PORT` for each additional worker. You will also need to run a separate heartbeat sender for each worker:

```bash
python3 -m elas_tf.heartbeat_sender localhost 6000 <worker_id> localhost <tf_port> 2
```

## How It Works

### Heartbeat-based failure detection

Each worker runs a **standalone heartbeat sender** as a separate process. This sender:
- Sends a TCP "join" message on startup, then "heartbeat" messages every 2 seconds
- Ignores SIGINT, SIGTERM, and SIGHUP — can only be stopped by `kill -9`
- Survives TF cascade crashes (when one worker dies, TensorFlow kills all collective operations, but the heartbeat sender keeps running)

This design lets the controller distinguish between:
- **Intentionally killed workers** — heartbeat sender is dead, controller times out after 15s
- **TF cascade crash victims** — heartbeat sender is still alive, controller sees them as healthy survivors

### Controller logic

The controller (`controller.py`) runs a main loop that:
1. Polls heartbeat events (joins, heartbeats, failures)
2. Detects timeouts for workers whose heartbeats stopped
3. On membership change: writes a new `tf_config.json` with updated cluster topology
4. On membership shrink: starts a 10-second stabilization window, then writes `remaining_workers` listing survivor IDs
5. On port change (worker rejoin): rewrites `tf_config.json` with updated ports

### Supervisor logic

The supervisor (`launch.sh`) runs a main loop that:
1. Tracks alive worker IDs and their PIDs
2. Polls for worker PID exits (failure) or a `scale_up` file (from `add_worker.sh`)
3. On failure: waits up to 45s for the controller's `remaining_workers` file, then sends restart signals to survivors
4. On scale up: adds new worker IDs to its tracking and re-reads PIDs
5. On training completion: detects the `training_done` marker and shuts down cleanly

### Worker lifecycle

Each worker's bash wrapper runs an internal loop:
1. Check for a restart signal file; if found, source new environment variables
2. Start heartbeat sender as a background process
3. Run `python3 -m elas_tf.worker` and wait for it to exit
4. On exit code 0: training complete, write `training_done` marker, exit
5. On Ctrl+C: kill heartbeat sender and python, exit
6. On non-zero exit (TF crash): print "waiting for restart signal", poll for signal file up to 90s
7. If signal received: loop back to step 1; if not: this worker was removed, exit

## Troubleshooting

### `SSL: CERTIFICATE_VERIFY_FAILED` when downloading MNIST

Python cannot verify the HTTPS certificate for `storage.googleapis.com`. Fix it by installing `certifi`:

```bash
pip install certifi
export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")
```

Add the `export` line to `~/.zshrc` to make it permanent.

### `module 'tensorflow' has no attribute 'keras'`

The generic `tensorflow` PyPI package shadows `tensorflow-macos`. Fix:

```bash
pip uninstall tensorflow -y
pip install --force-reinstall tensorflow-macos==2.15.0
```

### `exec: python: not found`

macOS does not provide a bare `python` command. All scripts use `python3`.

### `numpy.core.umath failed to import`

Reinstall NumPy:

```bash
pip install --upgrade --force-reinstall numpy
```

### Zombie heartbeat senders from a previous run

If you see rapid generation increments or ghost workers, leftover heartbeat senders from a previous run may still be alive (they ignore normal signals). Kill them manually:

```bash
pkill -9 -f "elas_tf.heartbeat_sender"
```

`launch.sh` does this automatically on startup.
