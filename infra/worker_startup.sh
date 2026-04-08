#!/usr/bin/env bash
#
# GCP VM startup script for an ElasTF worker.
# Works on both CPU-only (Debian) and GPU (Deep Learning) VMs.
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

# Install system dependencies
apt-get update -qq
apt-get install -y -qq git python3-pip python3-venv

# Clone repo
if [ -d /opt/elastf ]; then
    cd /opt/elastf && git pull origin "$BRANCH"
else
    git clone --branch "$BRANCH" "$REPO_URL" /opt/elastf
fi
cd /opt/elastf/ElasTF

# Set up venv and install Python dependencies
python3 -m venv /opt/elastf_venv
source /opt/elastf_venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet tensorflow==2.15.0 requests flask google-cloud-storage numpy grpcio protobuf

# Check for GPU
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "[startup] GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "[startup] No GPU detected. Running CPU-only training."
fi

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
nohup /opt/elastf_venv/bin/python3 -m elas_tf.worker_entrypoint >> /var/log/elastf.log 2>&1 &
echo "[startup] Worker entrypoint started (pid=$!)"
