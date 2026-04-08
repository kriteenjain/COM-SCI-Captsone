# COM SCI Capstone — ElasTF

Elastic distributed training with TensorFlow for CS 214.

ElasTF runs a controller and multiple workers on your laptop to train MNIST with `tf.distribute.MultiWorkerMirroredStrategy`. The `launch.sh` script starts everything and automatically recovers from worker failures using heartbeats and checkpoints.

## Project layout

```text
ElasTF/
├── launch.sh          # main supervisor script
├── kill_workers.sh    # kill workers to simulate failures
├── add_worker.sh      # add workers while training is running
├── requirements.txt
├── elas_tf/
│   ├── controller.py
│   ├── worker.py
│   ├── training.py
│   ├── heartbeat.py
│   ├── heartbeat_sender.py
│   └── checkpointing.py
└── shared/            # checkpoints + config (created at runtime)
```

## Setup

```bash
cd ElasTF
python3 -m venv .venv
source .venv/bin/activate
pip install tensorflow-macos==2.15.0
pip install -r requirements.txt
```

## Run the demo

```bash
chmod +x launch.sh kill_workers.sh add_worker.sh
./launch.sh
```

This opens one controller terminal and several worker terminals. Training starts automatically on MNIST and will resume from the latest checkpoint after failures.

## Testing elasticity

- **Simulate failures:** press Ctrl+C in a worker terminal, or run `./kill_workers.sh 0` from the project root.
- **Scale up:** run `./add_worker.sh` (or `./add_worker.sh 3`) while training is running to add more workers.
