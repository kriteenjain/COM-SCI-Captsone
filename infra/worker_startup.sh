#!/usr/bin/env bash
#
# GCP VM startup script for an ElasTF worker.
# Runs automatically when the VM boots (Deep Learning VM image).
#
set -euo pipefail
exec > >(tee -a /var/log/elastf.log) 2>&1

echo "[startup] ElasTF Worker startup script running..."
echo "[startup] $(date)"

# Fetch metadata
WORKER_ID=$(curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker_id" || echo "0")
CONTROLLER_IP=$(curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/controller_ip" || echo "")
TF_PORT=$(curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/tf_port" || echo "35000")
REPO_URL=$(curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/repo_url" || echo "")
BRANCH=$(curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/branch" || echo "main")
GCS_BUCKET=$(curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/gcs_bucket" || echo "")

echo "[startup] Worker ID: $WORKER_ID"
echo "[startup] Controller IP: $CONTROLLER_IP"
echo "[startup] TF port: $TF_PORT"
echo "[startup] Repo: $REPO_URL ($BRANCH)"
echo "[startup] GCS bucket: $GCS_BUCKET"

if [ -z "$CONTROLLER_IP" ]; then
    echo "[startup] FATAL: No controller_ip in metadata. Exiting."
    exit 1
fi

# Clone repo (Deep Learning VM already has Python, TF, CUDA)
if [ -d /opt/elastf ]; then
    cd /opt/elastf && git pull origin "$BRANCH"
else
    git clone --branch "$BRANCH" "$REPO_URL" /opt/elastf
fi
cd /opt/elastf/ElasTF

# Install additional Python dependencies
pip install --quiet requests flask google-cloud-storage

# Start worker entrypoint
export WORKER_ID="$WORKER_ID"
export CONTROLLER_HOST="$CONTROLLER_IP"
export CONTROLLER_URL="http://${CONTROLLER_IP}:8080"
export HEARTBEAT_PORT=5000
export HTTP_PORT=8080
export TF_PORT="$TF_PORT"
export CHECKPOINT_DIR="/tmp/elastf_checkpoints"
export GCS_BUCKET="$GCS_BUCKET"
export EPOCHS=10

echo "[startup] Starting worker entrypoint (worker_id=$WORKER_ID, tf_port=$TF_PORT)..."
nohup python3 -m elas_tf.worker_entrypoint >> /var/log/elastf.log 2>&1 &
echo "[startup] Worker entrypoint started (pid=$!)"
