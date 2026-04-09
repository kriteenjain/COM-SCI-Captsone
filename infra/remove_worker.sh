#!/usr/bin/env bash
#
# Remove a worker from the running cluster (simulates failure / scale-down).
# The controller detects the missing heartbeat, updates the cluster config,
# and surviving workers restart from the latest checkpoint.
#
# Usage:
#   ./infra/remove_worker.sh 1                # remove worker 1
#   ZONE=us-west1-a ./infra/remove_worker.sh 2  # remove worker 2
#
set -euo pipefail

WORKER_ID=${1:?Usage: ./infra/remove_worker.sh <worker_id>}
ZONE=${ZONE:-us-west1-a}
VM_NAME="elastf-worker-${WORKER_ID}"

echo ""
echo "============================================================"
echo " Removing worker $WORKER_ID ($VM_NAME)"
echo "============================================================"
echo ""

# Check the VM exists
if ! gcloud compute instances describe "$VM_NAME" --zone="$ZONE" &>/dev/null; then
    echo "[remove] ERROR: VM $VM_NAME not found in zone $ZONE"
    exit 1
fi

# Get controller IP for status check
CONTROLLER_IP=$(gcloud compute instances describe elastf-controller \
    --zone="$ZONE" --format='get(networkInterfaces[0].networkIP)' 2>/dev/null || echo "")

echo "[remove] Cluster status BEFORE removal:"
if [ -n "$CONTROLLER_IP" ]; then
    gcloud compute ssh elastf-controller --zone="$ZONE" \
        --command="curl -s http://localhost:8080/status | python3 -m json.tool" 2>/dev/null || true
fi
echo ""

# Delete the VM (this kills the worker and its heartbeat)
echo "[remove] Deleting VM $VM_NAME..."
gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet

echo "[remove] VM deleted. Controller will detect heartbeat timeout in ~8s."
echo "[remove] Waiting 15s for controller to update cluster..."
sleep 15

echo ""
echo "[remove] Cluster status AFTER removal:"
if [ -n "$CONTROLLER_IP" ]; then
    gcloud compute ssh elastf-controller --zone="$ZONE" \
        --command="curl -s http://localhost:8080/status | python3 -m json.tool" 2>/dev/null || true
fi

echo ""
echo "============================================================"
echo " Worker $WORKER_ID removed. Surviving workers will restart"
echo " with the new cluster config and resume from checkpoint."
echo "============================================================"
