#!/usr/bin/env bash
#
# Provision the ElasTF distributed training cluster on GCP.
#
# Usage:
#   ./infra/create_cluster.sh            # default: 4 worker VMs (CPU)
#   ./infra/create_cluster.sh 2          # 2 worker VMs
#   USE_GPU=1 ./infra/create_cluster.sh  # use T4 GPUs (requires quota)
#
set -euo pipefail

NUM_WORKERS=${1:-4}
ZONE=${ZONE:-us-west1-a}
USE_GPU=${USE_GPU:-0}
EPOCHS=${EPOCHS:-10}
LIGHT_MODEL=${LIGHT_MODEL:-0}
PROJECT=$(gcloud config get-value project 2>/dev/null)
BUCKET="elastf-checkpoints-${PROJECT}"
REPO_URL="https://github.com/kriteenjain/COM-SCI-Captsone.git"
BRANCH="main"

if [ "$USE_GPU" -eq 1 ]; then
    WORKER_TYPE="n1-standard-4 + T4 GPU"
else
    WORKER_TYPE="e2-standard-8 (CPU only, 8 vCPU / 32GB RAM)"
fi

echo ""
echo "============================================================"
echo " ElasTF Cluster Provisioning"
echo "============================================================"
echo "  Project:      $PROJECT"
echo "  Zone:         $ZONE"
echo "  Workers:      $NUM_WORKERS ($WORKER_TYPE)"
echo "  GCS bucket:   gs://$BUCKET/"
echo "  Repo:         $REPO_URL ($BRANCH)"
echo "============================================================"
echo ""

# ── GCS bucket ───────────────────────────────────────────────────
echo "[infra] Creating GCS bucket (if not exists)..."
gsutil ls -b "gs://${BUCKET}" 2>/dev/null || gsutil mb -l us-central1 "gs://${BUCKET}/"

# ── Upload startup scripts to GCS ────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "[infra] Uploading startup scripts to GCS..."
gsutil cp "${SCRIPT_DIR}/controller_startup.sh" "gs://${BUCKET}/scripts/controller_startup.sh"
gsutil cp "${SCRIPT_DIR}/worker_startup.sh" "gs://${BUCKET}/scripts/worker_startup.sh"

# ── Firewall rules ───────────────────────────────────────────────
echo "[infra] Creating firewall rules..."
gcloud compute firewall-rules describe elastf-internal >/dev/null 2>&1 || \
gcloud compute firewall-rules create elastf-internal \
    --allow=tcp:5000,tcp:8080,tcp:30000-40000 \
    --source-tags=elastf-controller,elastf-worker \
    --target-tags=elastf-controller,elastf-worker \
    --description="ElasTF internal communication" \
    --quiet

# ── Controller VM ────────────────────────────────────────────────
echo "[infra] Creating controller VM..."
gcloud compute instances create elastf-controller \
    --zone="$ZONE" \
    --machine-type=e2-medium \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --tags=elastf-controller \
    --scopes=storage-full \
    --metadata="repo_url=${REPO_URL},branch=${BRANCH},gcs_bucket=${BUCKET}" \
    --metadata-from-file=startup-script="${SCRIPT_DIR}/controller_startup.sh" \
    --quiet

CONTROLLER_IP=$(gcloud compute instances describe elastf-controller \
    --zone="$ZONE" --format='get(networkInterfaces[0].networkIP)')
echo "[infra] Controller internal IP: $CONTROLLER_IP"

# ── Worker VMs ───────────────────────────────────────────────────
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    TF_PORT=$((35000 + i))
    echo "[infra] Creating worker VM elastf-worker-${i} (TF port: $TF_PORT)..."

    if [ "$USE_GPU" -eq 1 ]; then
        gcloud compute instances create "elastf-worker-${i}" \
            --zone="$ZONE" \
            --machine-type=n1-standard-4 \
            --accelerator=type=nvidia-tesla-t4,count=1 \
            --image-family=common-cu128-ubuntu-2204-nvidia-570 \
            --image-project=deeplearning-platform-release \
            --maintenance-policy=TERMINATE \
            --tags=elastf-worker \
            --scopes=storage-full \
            --metadata="worker_id=${i},controller_ip=${CONTROLLER_IP},tf_port=${TF_PORT},repo_url=${REPO_URL},branch=${BRANCH},gcs_bucket=${BUCKET},epochs=${EPOCHS},light_model=${LIGHT_MODEL}" \
            --metadata-from-file=startup-script="${SCRIPT_DIR}/worker_startup.sh" \
            --quiet &
    else
        gcloud compute instances create "elastf-worker-${i}" \
            --zone="$ZONE" \
            --machine-type=e2-standard-8 \
            --image-family=debian-12 \
            --image-project=debian-cloud \
            --tags=elastf-worker \
            --scopes=storage-full \
            --metadata="worker_id=${i},controller_ip=${CONTROLLER_IP},tf_port=${TF_PORT},repo_url=${REPO_URL},branch=${BRANCH},gcs_bucket=${BUCKET},epochs=${EPOCHS},light_model=${LIGHT_MODEL}" \
            --metadata-from-file=startup-script="${SCRIPT_DIR}/worker_startup.sh" \
            --quiet &
    fi
done
wait
echo ""

# ── Summary ──────────────────────────────────────────────────────
echo "============================================================"
echo " Cluster created!"
echo "============================================================"
echo "  Controller:  elastf-controller ($CONTROLLER_IP)"
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    WIP=$(gcloud compute instances describe "elastf-worker-${i}" \
        --zone="$ZONE" --format='get(networkInterfaces[0].networkIP)' 2>/dev/null || echo "pending")
    echo "  Worker $i:    elastf-worker-${i} ($WIP)"
done
echo ""
echo "  GCS bucket:  gs://$BUCKET/"
echo ""
echo "  Monitor logs:"
echo "    gcloud compute ssh elastf-controller --zone=$ZONE -- tail -f /var/log/elastf.log"
echo "    gcloud compute ssh elastf-worker-0 --zone=$ZONE -- tail -f /var/log/elastf.log"
echo ""
echo "  Workers will auto-start training once they register with the controller."
echo "============================================================"
