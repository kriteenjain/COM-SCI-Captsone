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

# Count expected total workers
CURRENT_COUNT=$(echo "$CURRENT_WORKERS" | wc -w | tr -d ' ')
EXPECTED_TOTAL=$((CURRENT_COUNT + NUM_TO_ADD))
echo "[scale-up] Expected total workers after scale-up: $EXPECTED_TOTAL"

# Poll controller until all new workers have registered (instead of blind sleep)
echo ""
echo "[scale-up] Polling controller until $EXPECTED_TOTAL worker(s) are registered..."
POLL_DEADLINE=$((SECONDS + 300))
while [ "$SECONDS" -lt "$POLL_DEADLINE" ]; do
    NUM_REGISTERED=$(gcloud compute ssh elastf-controller --zone="$ZONE" \
        --command="curl -s http://localhost:8080/status" 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('num_workers',0))" 2>/dev/null || echo "0")
    echo "[scale-up]   Controller sees $NUM_REGISTERED worker(s)..."
    if [ "$NUM_REGISTERED" -ge "$EXPECTED_TOTAL" ]; then
        echo "[scale-up]   All $EXPECTED_TOTAL workers registered!"
        break
    fi
    sleep 15
done

echo ""
echo "[scale-up] Cluster status after new workers registered:"
gcloud compute ssh elastf-controller --zone="$ZONE" \
    --command="curl -s http://localhost:8080/status | python3 -m json.tool" 2>/dev/null || true

# Send SIGUSR1 to existing workers IMMEDIATELY so they restart and
# enter the stability wait at the same time as the new worker(s).
echo ""
echo "[scale-up] Sending SIGUSR1 to existing workers (triggers restart)..."
for vm in $CURRENT_WORKERS; do
    WID=$(echo "$vm" | sed 's/elastf-worker-//')
    echo "[scale-up]   Signaling $vm..."
    gcloud compute ssh "$vm" --zone="$ZONE" \
        --command='sudo kill -USR1 $(cat /tmp/elastf_entrypoint.pid 2>/dev/null) 2>/dev/null && echo "Sent SIGUSR1" || echo "Failed to send signal"' 2>/dev/null || true
done

echo ""
echo "[scale-up] All workers will now wait for cluster stability (30s of no changes)"
echo "           before starting training together with the new config."
echo ""

# Give workers time to restart and form the new cluster
echo "[scale-up] Waiting 60s for cluster to re-form..."
sleep 60

echo "[scale-up] Cluster status AFTER scale-up:"
gcloud compute ssh elastf-controller --zone="$ZONE" \
    --command="curl -s http://localhost:8080/status | python3 -m json.tool" 2>/dev/null || true

echo ""
echo "============================================================"
echo " Scale-up complete! Added workers: ${NEW_IDS[*]}"
echo " All workers should restart and resume training."
echo "============================================================"
