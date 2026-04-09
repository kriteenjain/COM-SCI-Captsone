#!/usr/bin/env bash
#
# Add worker(s) to a running cluster (scale-up).
# Creates new worker VMs, waits for them to register, then restarts
# existing workers so they pick up the new cluster config.
#
# Usage:
#   ./infra/add_worker.sh                  # add 1 worker
#   ./infra/add_worker.sh 2                # add 2 workers
#   ZONE=us-west1-a ./infra/add_worker.sh  # specify zone
#
set -euo pipefail

NUM_TO_ADD=${1:-1}
ZONE=${ZONE:-us-west1-a}
PROJECT=$(gcloud config get-value project 2>/dev/null)
BUCKET="elastf-checkpoints-${PROJECT}"
REPO_URL="https://github.com/kriteenjain/COM-SCI-Captsone.git"
BRANCH="main"
USE_GPU=${USE_GPU:-0}
EPOCHS=${EPOCHS:-10}
LIGHT_MODEL=${LIGHT_MODEL:-0}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Get controller IP
CONTROLLER_IP=$(gcloud compute instances describe elastf-controller \
    --zone="$ZONE" --format='get(networkInterfaces[0].networkIP)')

echo ""
echo "============================================================"
echo " Scaling up: adding $NUM_TO_ADD worker(s)"
echo "============================================================"
echo ""

# Find current highest worker ID
CURRENT_WORKERS=$(gcloud compute instances list \
    --filter="name~'^elastf-worker-' AND zone:$ZONE" \
    --format="value(name)" 2>/dev/null | sort)
MAX_ID=-1
for vm in $CURRENT_WORKERS; do
    WID=$(echo "$vm" | sed 's/elastf-worker-//')
    [ "$WID" -gt "$MAX_ID" ] && MAX_ID="$WID"
done

echo "[scale-up] Current workers: $CURRENT_WORKERS"
echo "[scale-up] Highest worker ID: $MAX_ID"
echo "[scale-up] Controller IP: $CONTROLLER_IP"

echo ""
echo "[scale-up] Cluster status BEFORE scale-up:"
gcloud compute ssh elastf-controller --zone="$ZONE" \
    --command="curl -s http://localhost:8080/status | python3 -m json.tool" 2>/dev/null || true
echo ""

# Create new worker VMs
NEW_IDS=()
for n in $(seq 1 "$NUM_TO_ADD"); do
    NEW_ID=$((MAX_ID + n))
    NEW_IDS+=("$NEW_ID")
    TF_PORT=$((35000 + NEW_ID))
    echo "[scale-up] Creating elastf-worker-${NEW_ID} (TF port: $TF_PORT)..."

    if [ "$USE_GPU" -eq 1 ]; then
        gcloud compute instances create "elastf-worker-${NEW_ID}" \
            --zone="$ZONE" \
            --machine-type=n1-standard-4 \
            --accelerator=type=nvidia-tesla-t4,count=1 \
            --image-family=common-cu128-ubuntu-2204-nvidia-570 \
            --image-project=deeplearning-platform-release \
            --maintenance-policy=TERMINATE \
            --tags=elastf-worker \
            --scopes=storage-full \
            --metadata="worker_id=${NEW_ID},controller_ip=${CONTROLLER_IP},tf_port=${TF_PORT},repo_url=${REPO_URL},branch=${BRANCH},gcs_bucket=${BUCKET},epochs=${EPOCHS},light_model=${LIGHT_MODEL}" \
            --metadata-from-file=startup-script="${SCRIPT_DIR}/worker_startup.sh" \
            --quiet &
    else
        gcloud compute instances create "elastf-worker-${NEW_ID}" \
            --zone="$ZONE" \
            --machine-type=e2-standard-8 \
            --image-family=debian-12 \
            --image-project=debian-cloud \
            --tags=elastf-worker \
            --scopes=storage-full \
            --metadata="worker_id=${NEW_ID},controller_ip=${CONTROLLER_IP},tf_port=${TF_PORT},repo_url=${REPO_URL},branch=${BRANCH},gcs_bucket=${BUCKET},epochs=${EPOCHS},light_model=${LIGHT_MODEL}" \
            --metadata-from-file=startup-script="${SCRIPT_DIR}/worker_startup.sh" \
            --quiet &
    fi
done
wait

echo ""
echo "[scale-up] Waiting for new worker(s) to install deps and register (~150s)..."
sleep 150

echo ""
echo "[scale-up] Cluster status after new workers registered:"
gcloud compute ssh elastf-controller --zone="$ZONE" \
    --command="curl -s http://localhost:8080/status | python3 -m json.tool" 2>/dev/null || true

# Kill training processes on existing workers so they restart with new config
echo ""
echo "[scale-up] Restarting existing workers to pick up new cluster config..."
for vm in $CURRENT_WORKERS; do
    WID=$(echo "$vm" | sed 's/elastf-worker-//')
    echo "[scale-up]   Sending SIGUSR1 to entrypoint on $vm (triggers restart)..."
    gcloud compute ssh "$vm" --zone="$ZONE" \
        --command='sudo kill -USR1 $(cat /tmp/elastf_entrypoint.pid 2>/dev/null) 2>/dev/null && echo "Sent SIGUSR1" || echo "Failed to send signal"' 2>/dev/null || true
done

echo ""
echo "[scale-up] Existing workers will detect the crash, poll for restart signal,"
echo "           fetch new TF_CONFIG, and resume training with the larger cluster."
echo ""

sleep 20

echo "[scale-up] Cluster status AFTER scale-up:"
gcloud compute ssh elastf-controller --zone="$ZONE" \
    --command="curl -s http://localhost:8080/status | python3 -m json.tool" 2>/dev/null || true

echo ""
echo "============================================================"
echo " Scale-up complete! Added workers: ${NEW_IDS[*]}"
echo " All workers should restart and resume training."
echo "============================================================"
