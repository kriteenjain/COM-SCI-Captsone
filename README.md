# COM SCI Capstone — ElasTF

Elastic distributed GPU training with TensorFlow on GCP.

ElasTF trains a **ResNet-50 on CIFAR-10** across multiple GCP VMs, each with a dedicated T4 GPU. The controller coordinates workers via a heartbeat protocol and HTTP API, with checkpoints stored in Google Cloud Storage. Workers can join, leave, or crash — the cluster recovers automatically.

## Architecture

```text
┌───────────────────────────────────────────────────────┐
│                    GCP VPC Network                     │
│                                                        │
│   ┌──────────────────┐                                │
│   │  Controller VM    │  e2-medium (no GPU)           │
│   │  :5000 heartbeat  │                               │
│   │  :8080 HTTP API   │                               │
│   └────────┬─────────┘                                │
│            │  heartbeat + HTTP                        │
│   ┌────────┼──────────────────────────┐               │
│   │        │                          │               │
│   ▼        ▼                          ▼               │
│ ┌────────┐ ┌────────┐    ┌────────┐  ┌────────┐     │
│ │Worker 0│ │Worker 1│    │Worker 2│  │Worker 3│     │
│ │ T4 GPU │ │ T4 GPU │◄──►│ T4 GPU │  │ T4 GPU │     │
│ └────┬───┘ └────┬───┘    └────┬───┘  └────┬───┘     │
│      │          │             │            │          │
│      └──────────┴─────────────┴────────────┘          │
│                  TF gRPC gradient sync                │
│                                                        │
│   ┌────────────────┐                                  │
│   │  GCS Bucket     │  checkpoints + metrics          │
│   └────────────────┘                                  │
└───────────────────────────────────────────────────────┘
```

## Project layout

```text
ElasTF/
├── elas_tf/
│   ├── controller.py        # Heartbeat monitor + HTTP API
│   ├── worker.py            # Worker: loads TF_CONFIG, runs training
│   ├── worker_entrypoint.py # Cloud lifecycle manager (replaces bash)
│   ├── training.py          # ResNet-50 on CIFAR-10, distributed
│   ├── heartbeat.py         # Heartbeat server (TCP)
│   ├── heartbeat_sender.py  # Heartbeat client (survives TF crashes)
│   ├── gcs_storage.py       # GCS checkpoint upload/download
│   ├── checkpointing.py     # Checkpoint utilities
│   └── plot_training.py     # Metrics visualization
├── infra/
│   ├── create_cluster.sh    # Provision GCP cluster
│   ├── destroy_cluster.sh   # Tear down GCP cluster
│   ├── controller_startup.sh # Controller VM startup script
│   ├── worker_startup.sh    # Worker VM startup script
│   ├── run_benchmark.sh     # Automated 1/2/4 GPU benchmark
│   └── plot_speedup.py      # Plot speedup curve from results
├── launch.sh                # Local-mode supervisor (macOS)
├── add_worker.sh            # Local-mode: add workers
├── kill_workers.sh          # Local-mode: simulate failures
├── requirements.txt
└── shared/                  # Local-mode runtime state
```

## Prerequisites

- **GCP project** with billing enabled
- **gcloud CLI** authenticated (`gcloud auth login`)
- **GPU quota**: at least 4x NVIDIA T4 in your chosen zone
- **APIs enabled**: Compute Engine, Cloud Storage

## Quick Start (GCP Distributed)

### 1. Provision the cluster

```bash
cd ElasTF

# Default: 4 workers with T4 GPUs
./infra/create_cluster.sh

# Or specify worker count
./infra/create_cluster.sh 2
```

This creates:
- 1 controller VM (`e2-medium`, no GPU)
- N worker VMs (`n1-standard-4` + T4 GPU each)
- GCS bucket for checkpoints
- Firewall rules for internal communication

Workers auto-start training once they register with the controller.

### 2. Monitor training

```bash
# Controller logs
gcloud compute ssh elastf-controller --zone=us-central1-a -- tail -f /var/log/elastf.log

# Worker logs
gcloud compute ssh elastf-worker-0 --zone=us-central1-a -- tail -f /var/log/elastf.log

# Cluster status (from any machine with network access)
curl http://<controller-external-ip>:8080/status
```

### 3. Simulate failures

```bash
# Kill a worker VM
gcloud compute instances delete elastf-worker-2 --zone=us-central1-a --quiet

# Remaining workers detect the failure, restart, and resume from checkpoint
```

### 4. Scale up

```bash
# Add more workers to a running cluster
# (Re-run create with higher count; existing workers restart)
./infra/create_cluster.sh 6
```

### 5. Tear down

```bash
./infra/destroy_cluster.sh
```

## Running the Benchmark

The benchmark script provisions clusters with 1, 2, and 4 workers, runs 10 epochs each, and collects wall times.

```bash
./infra/run_benchmark.sh

# Plot the speedup curve
python3 infra/plot_speedup.py
```

### Expected Results (ResNet-50, CIFAR-10, 10 epochs)

| Workers | GPUs | Wall Time  | Speedup |
|---------|------|------------|---------|
| 1       | 1xT4 | ~700s     | 1.0x    |
| 2       | 2xT4 | ~400s     | 1.75x   |
| 4       | 4xT4 | ~220s     | 3.2x    |

Sub-linear speedup is expected due to gradient synchronization overhead.

## Local Mode (macOS)

The original local-only mode still works for development:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install tensorflow-macos==2.15.0
pip install -r requirements.txt

chmod +x launch.sh kill_workers.sh add_worker.sh
./launch.sh
```

## Cost Estimate

- Controller: ~$0.03/hr
- Each T4 worker: ~$0.95/hr
- Full benchmark (1+2+4 worker runs, ~1hr each): ~$6-8 total
