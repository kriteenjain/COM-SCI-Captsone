#!/usr/bin/env bash
#
# ElasTF Elastic Scaling Benchmark
#
# Runs 4 scenarios to measure the impact of adding/removing workers:
#   1. Baseline:      2 workers, static for all 10 epochs
#   2. Scale-down:    2 workers → kill 1 at epoch 3 → finish with 1 worker
#   3. Scale-up to 3: 2 workers → add 1 at epoch 3 → finish with 3 workers
#   4. Scale-up to 4: 2 workers → add 2 at epoch 3 → finish with 4 workers
#
# Usage:
#   ./infra/elastic_benchmark.sh
#   ZONE=us-west1-a ./infra/elastic_benchmark.sh
#
set -euo pipefail

ZONE=${ZONE:-us-west1-a}
PROJECT=$(gcloud config get-value project 2>/dev/null)
BUCKET="elastf-checkpoints-${PROJECT}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../elastic_results"
EPOCHS=10
SCALE_AT_EPOCH=3

mkdir -p "$RESULTS_DIR"

echo ""
echo "============================================================"
echo " ElasTF Elastic Scaling Benchmark"
echo "============================================================"
echo "  Scenarios:"
echo "    1) Baseline:   2 workers, static"
echo "    2) Scale-down: 2 → 1 worker at epoch $SCALE_AT_EPOCH"
echo "    3) Scale-up 3: 2 → 3 workers at epoch $SCALE_AT_EPOCH"
echo "    4) Scale-up 4: 2 → 4 workers at epoch $SCALE_AT_EPOCH"
echo "  Epochs:  $EPOCHS"
echo "  Model:   Lightweight CNN on CIFAR-10"
echo "  Zone:    $ZONE"
echo "  Results: $RESULTS_DIR"
echo "============================================================"
echo ""

wait_for_epoch() {
    local target_epoch=$1
    local max_wait=${2:-2400}
    local waited=0
    local poll=20

    echo "[bench] Waiting for epoch $target_epoch to complete..."
    while [ $waited -lt $max_wait ]; do
        sleep $poll
        waited=$((waited + poll))

        if gsutil -q stat "gs://${BUCKET}/metrics/training_metrics.csv" 2>/dev/null; then
            gsutil cp -q "gs://${BUCKET}/metrics/training_metrics.csv" "/tmp/elastf_bench_check.csv" 2>/dev/null
            EPOCH_COUNT=$(tail -n +2 /tmp/elastf_bench_check.csv 2>/dev/null | wc -l | tr -d ' ')
            echo "[bench]   ... $EPOCH_COUNT / $target_epoch epochs done (${waited}s elapsed)"
            if [ "$EPOCH_COUNT" -ge "$target_epoch" ]; then
                echo "[bench] Epoch $target_epoch reached!"
                return 0
            fi
        else
            echo "[bench]   ... waiting for metrics (${waited}s elapsed)"
        fi
    done
    echo "[bench] WARNING: Timed out waiting for epoch $target_epoch"
    return 1
}

run_scenario() {
    local scenario_name=$1
    local scenario_label=$2
    local scale_action=$3

    echo ""
    echo "########################################################"
    echo "  SCENARIO: $scenario_label"
    echo "########################################################"
    echo ""

    echo "[bench] Clearing GCS state..."
    gsutil -m rm -r "gs://${BUCKET}/checkpoints/" 2>/dev/null || true
    gsutil -m rm -r "gs://${BUCKET}/metrics/" 2>/dev/null || true

    echo "[bench] Creating cluster with 2 workers..."
    REAL_START=$(date +%s)
    ZONE=$ZONE LIGHT_MODEL=1 EPOCHS=$EPOCHS bash "${SCRIPT_DIR}/create_cluster.sh" 2

    echo "[bench] Waiting for training to start (~180s)..."
    sleep 180

    if [ "$scale_action" = "none" ]; then
        echo "[bench] Static run — no scaling action."
        wait_for_epoch $EPOCHS
    elif [ "$scale_action" = "kill_1" ]; then
        wait_for_epoch $SCALE_AT_EPOCH
        echo ""
        echo "[bench] >>> SCALING ACTION: Removing worker 1 <<<"
        ZONE=$ZONE bash "${SCRIPT_DIR}/remove_worker.sh" 1
        echo "[bench] Waiting for remaining epochs..."
        sleep 30
        wait_for_epoch $EPOCHS
    elif [ "$scale_action" = "add_1" ]; then
        wait_for_epoch $SCALE_AT_EPOCH
        echo ""
        echo "[bench] >>> SCALING ACTION: Adding 1 worker (2 → 3) <<<"
        ZONE=$ZONE LIGHT_MODEL=1 EPOCHS=$EPOCHS bash "${SCRIPT_DIR}/add_worker.sh" 1
        echo "[bench] Waiting for remaining epochs..."
        sleep 60
        wait_for_epoch $EPOCHS
    elif [ "$scale_action" = "add_2" ]; then
        wait_for_epoch $SCALE_AT_EPOCH
        echo ""
        echo "[bench] >>> SCALING ACTION: Adding 2 workers (2 → 4) <<<"
        ZONE=$ZONE LIGHT_MODEL=1 EPOCHS=$EPOCHS bash "${SCRIPT_DIR}/add_worker.sh" 2
        echo "[bench] Waiting for remaining epochs..."
        sleep 60
        wait_for_epoch $EPOCHS
    fi

    REAL_END=$(date +%s)
    REAL_WALL=$((REAL_END - REAL_START))

    RESULT_CSV="${RESULTS_DIR}/${scenario_name}.csv"
    gsutil cp "gs://${BUCKET}/metrics/training_metrics.csv" "$RESULT_CSV" 2>/dev/null || true

    TRAINING_TIME="unknown"
    if [ -f "$RESULT_CSV" ]; then
        TRAINING_TIME=$(tail -1 "$RESULT_CSV" | cut -d',' -f3)
    fi

    echo "$scenario_name,$scenario_label,$TRAINING_TIME,$REAL_WALL" >> "${RESULTS_DIR}/summary.csv"

    echo ""
    echo "[bench] Scenario complete: $scenario_label"
    echo "[bench]   Training wall time:  ${TRAINING_TIME}s"
    echo "[bench]   Real wall clock:     ${REAL_WALL}s"
    echo ""

    echo "[bench] Tearing down cluster..."
    ZONE=$ZONE bash "${SCRIPT_DIR}/destroy_cluster.sh"

    echo "[bench] Cooldown 30s before next scenario..."
    sleep 30
}

echo "scenario,label,training_time_s,real_wall_s" > "${RESULTS_DIR}/summary.csv"

run_scenario "baseline_2w" "Baseline: 2 workers (static)" "none"
run_scenario "scale_down_2to1" "Scale-down: 2→1 worker at epoch $SCALE_AT_EPOCH" "kill_1"
run_scenario "scale_up_2to3" "Scale-up: 2→3 workers at epoch $SCALE_AT_EPOCH" "add_1"
run_scenario "scale_up_2to4" "Scale-up: 2→4 workers at epoch $SCALE_AT_EPOCH" "add_2"

echo ""
echo "============================================================"
echo " Elastic Benchmark Complete!"
echo "============================================================"
echo ""
echo " Summary:"
echo " --------"
column -t -s',' "${RESULTS_DIR}/summary.csv"
echo ""
echo " Detailed results: $RESULTS_DIR/"
echo " To plot: python3 infra/plot_elastic.py"
echo "============================================================"
