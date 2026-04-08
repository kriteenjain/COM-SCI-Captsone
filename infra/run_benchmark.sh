#!/usr/bin/env bash
#
# Run the ElasTF distributed training benchmark.
# Tests 1, 2, and 4 worker configurations and collects results.
#
# Usage:
#   ./infra/run_benchmark.sh
#
set -euo pipefail

ZONE=${ZONE:-us-central1-a}
PROJECT=$(gcloud config get-value project 2>/dev/null)
BUCKET="elastf-checkpoints-${PROJECT}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../benchmark_results"
mkdir -p "$RESULTS_DIR"

echo ""
echo "============================================================"
echo " ElasTF Distributed Training Benchmark"
echo "============================================================"
echo "  Configurations: 1 worker, 2 workers, 4 workers"
echo "  Model: ResNet-50 on CIFAR-10 (10 epochs)"
echo "  Results dir: $RESULTS_DIR"
echo "============================================================"
echo ""

for NUM_WORKERS in 1 2 4; do
    echo ""
    echo "########################################################"
    echo "  BENCHMARK RUN: $NUM_WORKERS worker(s)"
    echo "########################################################"
    echo ""

    # Clean GCS checkpoint state from previous run
    echo "[benchmark] Clearing previous checkpoints..."
    gsutil -m rm -r "gs://${BUCKET}/checkpoints/" 2>/dev/null || true
    gsutil -m rm -r "gs://${BUCKET}/metrics/" 2>/dev/null || true

    # Provision cluster
    echo "[benchmark] Provisioning cluster with $NUM_WORKERS workers..."
    bash "${SCRIPT_DIR}/create_cluster.sh" "$NUM_WORKERS"

    # Wait for training to complete by polling GCS for metrics
    echo "[benchmark] Waiting for training to complete..."
    MAX_WAIT=1800  # 30 minutes max
    WAITED=0
    POLL_INTERVAL=30

    while [ $WAITED -lt $MAX_WAIT ]; do
        sleep $POLL_INTERVAL
        WAITED=$((WAITED + POLL_INTERVAL))

        # Check if metrics file exists and has all epochs
        if gsutil -q stat "gs://${BUCKET}/metrics/training_metrics.csv" 2>/dev/null; then
            gsutil cp "gs://${BUCKET}/metrics/training_metrics.csv" "/tmp/elastf_check.csv" 2>/dev/null
            EPOCH_COUNT=$(tail -n +2 /tmp/elastf_check.csv | wc -l | tr -d ' ')
            echo "[benchmark]   ... $EPOCH_COUNT epochs complete ($WAITED/${MAX_WAIT}s)"
            if [ "$EPOCH_COUNT" -ge 10 ]; then
                echo "[benchmark] All 10 epochs complete!"
                break
            fi
        else
            echo "[benchmark]   ... waiting for metrics ($WAITED/${MAX_WAIT}s)"
        fi
    done

    # Download results
    RESULT_FILE="${RESULTS_DIR}/metrics_${NUM_WORKERS}w.csv"
    gsutil cp "gs://${BUCKET}/metrics/training_metrics.csv" "$RESULT_FILE" 2>/dev/null || true
    echo "[benchmark] Results saved to $RESULT_FILE"

    # Tear down cluster
    echo "[benchmark] Tearing down cluster..."
    bash "${SCRIPT_DIR}/destroy_cluster.sh"

    echo "[benchmark] Run with $NUM_WORKERS worker(s) complete."
    echo ""
done

echo ""
echo "============================================================"
echo " Benchmark Complete!"
echo "============================================================"
echo " Results:"
for NUM_WORKERS in 1 2 4; do
    RESULT_FILE="${RESULTS_DIR}/metrics_${NUM_WORKERS}w.csv"
    if [ -f "$RESULT_FILE" ]; then
        WALL_TIME=$(tail -1 "$RESULT_FILE" | cut -d',' -f3)
        echo "  $NUM_WORKERS worker(s): ${WALL_TIME}s total"
    fi
done
echo ""
echo " To plot the speedup curve:"
echo "   python3 infra/plot_speedup.py"
echo "============================================================"
