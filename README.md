# COM SCI Capstone — ElasTF

Elastic distributed TensorFlow training on GCP.

ElasTF trains a CNN on **CIFAR-10** across multiple GCP VMs. The controller coordinates workers via a heartbeat protocol and HTTP API, with checkpoints stored in Google Cloud Storage. Workers can join, leave, or crash mid-training — the cluster recovers automatically and resumes from the latest checkpoint.

By default the cluster runs on **CPU-only `e2-standard-8` VMs** (8 vCPU / 32 GB RAM each). An optional GPU path (`n1-standard-4` + NVIDIA T4) is gated behind a `USE_GPU=1` flag for users with GPU quota.

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
│ │  CPU   │ │  CPU   │◄──►│  CPU   │  │  CPU   │     │
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

Workers run `e2-standard-8` (CPU) by default. Pass `USE_GPU=1` to provision `n1-standard-4` + T4 instead.

## Project layout

```text
ElasTF/
├── elas_tf/
│   ├── controller.py          # Heartbeat monitor + HTTP API
│   ├── worker.py              # Worker: loads TF_CONFIG, runs training
│   ├── worker_entrypoint.py   # Cloud lifecycle manager
│   ├── training.py            # CNN on CIFAR-10, distributed
│   ├── heartbeat.py           # Heartbeat server (TCP)
│   ├── heartbeat_sender.py    # Heartbeat client (survives TF crashes)
│   ├── gcs_storage.py         # GCS checkpoint upload/download
│   ├── checkpointing.py       # Checkpoint utilities
│   └── plot_training.py       # Per-run metrics visualization
├── infra/
│   ├── create_cluster.sh      # Provision GCP cluster
│   ├── destroy_cluster.sh     # Tear down GCP cluster
│   ├── controller_startup.sh  # Controller VM startup script
│   ├── worker_startup.sh      # Worker VM startup script
│   ├── add_worker.sh          # Add a worker to a running cluster
│   ├── remove_worker.sh       # Remove a worker from a running cluster
│   ├── elastic_benchmark.sh   # Automated elastic-scaling benchmark
│   ├── plot_elastic.py        # Plot elastic-scaling results
│   └── plot_speedup.py        # Plot strong-scaling speedup curve
├── elastic_results/           # Benchmark CSVs + comparison plot
├── requirements.txt
└── README.md
```

## Models

`elas_tf/training.py` supports three model sizes, selected via env vars:

| Flag                 | Model                         | Params  | Default |
|----------------------|-------------------------------|---------|---------|
| (none)               | ResNet-50 (Keras Applications)| ~23.5M  |         |
| `MEDIUM_MODEL=1`     | Medium CNN (custom)           | ~3M     |  ✅     |
| `LIGHT_MODEL=1`      | Lightweight CNN (custom)      | ~600k   |         |

The medium CNN is the default because it has enough compute per step for distributed training to actually beat single-worker training on CPU — gradient-sync overhead is small relative to per-worker compute. The light CNN is used by `elastic_benchmark.sh` to keep iteration time short.

## Prerequisites

- **GCP project** with billing enabled
- **gcloud CLI** authenticated (`gcloud auth login`)
- **APIs enabled**: Compute Engine, Cloud Storage
- **GPU quota** is *only* required if you set `USE_GPU=1` (≥ N × NVIDIA T4 in your zone)

## Quick Start

### 1. Provision the cluster

```bash
cd ElasTF

# Default: 4 CPU workers (e2-standard-8)
./infra/create_cluster.sh

# Specify worker count
./infra/create_cluster.sh 2

# Use T4 GPUs instead (requires quota)
USE_GPU=1 ./infra/create_cluster.sh 2
```

This creates:
- 1 controller VM (`e2-medium`, no GPU)
- N worker VMs (CPU by default, GPU with `USE_GPU=1`)
- GCS bucket for checkpoints (`elastf-checkpoints-<project>`)
- Firewall rules for internal communication

Workers auto-start training as soon as they register with the controller.

### 2. Monitor training

```bash
# Controller logs
gcloud compute ssh elastf-controller --zone=us-west1-a -- tail -f /var/log/elastf.log

# Worker logs
gcloud compute ssh elastf-worker-0 --zone=us-west1-a -- tail -f /var/log/elastf.log

# Cluster status
curl http://<controller-external-ip>:8080/status
```

### 3. Scale at runtime

```bash
# Add a worker (scale up)
./infra/add_worker.sh 1

# Remove a worker (scale down) — simulates failure
./infra/remove_worker.sh 1

# Or kill a VM directly to simulate a crash
gcloud compute instances delete elastf-worker-2 --zone=us-west1-a --quiet
```

The remaining workers detect membership changes via heartbeat, restart the TF process, reshard the dataset, and resume from the latest checkpoint.

### 4. Tear down

```bash
./infra/destroy_cluster.sh
```

## Elastic-scaling benchmark

`infra/elastic_benchmark.sh` runs four scenarios (10 epochs each, lightweight CNN) and writes per-epoch metrics + a summary CSV to `elastic_results/`:

1. **Baseline** — 2 workers, static
2. **Scale-down** — 2 workers, kill 1 at epoch 3 → finish with 1
3. **Scale-up to 3** — 2 workers, add 1 at epoch 3 → finish with 3
4. **Scale-up to 4** — 2 workers, add 2 at epoch 3 → finish with 4

```bash
./infra/elastic_benchmark.sh
python3 infra/plot_elastic.py
```

### Measured results (CIFAR-10, 10 epochs, CPU workers)

From `elastic_results/*.csv`:

| Scenario              | Worker timeline    | Wall time | Final val acc |
|-----------------------|--------------------|-----------|---------------|
| Baseline (2 workers)  | 2 → 2              | 1864 s    | 0.790         |
| Scale-up 2→3          | 2 → 3 (at epoch 6) | 1327 s    | 0.727         |
| Scale-down 2→1        | 2 → 1 (at epoch 4) | 2979 s    | 0.819         |

Cross-run variance comes from VM cold-start time and shared-tenant CPU jitter on `e2-standard-8`. The scale-up run completes faster than the baseline because the extra worker arrives early; the scale-down run takes ~1.6× the baseline because the surviving worker now owns the full dataset shard. Final accuracies vary by ±0.05 across runs at this epoch budget.

`elastic_results/elastic_comparison.png` shows the bar chart and per-epoch timeline (with annotated worker-count transitions) generated by `plot_elastic.py`.

### Strong-scaling benchmark (optional)

For users with GPU quota, `infra/plot_speedup.py` plots a speedup curve from runs at different worker counts. Re-provision the cluster with `USE_GPU=1` and the desired worker count, run training, and feed the metrics CSVs into the plotter.

## Cost estimate (us-west1)

- Controller (`e2-medium`):       ~$0.03/hr
- Each CPU worker (`e2-standard-8`): ~$0.27/hr
- Each GPU worker (`n1-standard-4` + T4): ~$0.55/hr (T4) + ~$0.19/hr (VM)

A full elastic benchmark (4 scenarios, ~30–50 min each on CPU) runs for roughly $2–4 total.

## Dependencies

See `requirements.txt`:

- `tensorflow==2.15.0`
- `numpy`, `grpcio`, `protobuf`
- `flask`, `requests` (controller HTTP API)
- `google-cloud-storage` (checkpoint persistence)
- `matplotlib` (plotting)
