#!/usr/bin/env bash
#
# GCP VM startup script for an ElasTF worker.
# Works on both CPU-only (Debian) and GPU (Deep Learning) VMs.
#
set -euo pipefail
exec > >(tee -a /var/log/elastf.log) 2>&1

echo "[startup] ElasTF Worker startup script running..."
echo "[startup] $(date)"

# Helper: fetch a metadata attribute with retry (the metadata server may not
# be ready immediately after VM boot). Returns empty string after max retries.
fetch_meta() {
    local key="$1"
    local default_val="${2:-}"
    local attempts=30
    for i in $(seq 1 $attempts); do
        local val
        val=$(curl -sf -H "Metadata-Flavor: Google" \
            "http://metadata.google.internal/computeMetadata/v1/instance/attributes/${key}" 2>/dev/null || echo "")
        if [ -n "$val" ]; then
            echo "$val"
            return 0
        fi
        sleep 2
    done
    echo "$default_val"
}

# Wait for critical metadata (controller_ip) to be available before continuing.
echo "[startup] Fetching instance metadata (with retry)..."
CONTROLLER_IP=$(fetch_meta controller_ip "")
WORKER_ID=$(fetch_meta worker_id "0")
TF_PORT=$(fetch_meta tf_port "35000")
REPO_URL=$(fetch_meta repo_url "")
BRANCH=$(fetch_meta branch "main")
GCS_BUCKET=$(fetch_meta gcs_bucket "")
EPOCHS=$(fetch_meta epochs "10")
LIGHT_MODEL=$(fetch_meta light_model "0")
MEDIUM_MODEL=$(fetch_meta medium_model "0")
BATCH_SIZE=$(fetch_meta batch_size "256")
EXPECTED_WORKERS=$(fetch_meta expected_workers "0")

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
cd /opt/elastf

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

# Disable mTLS for GCE metadata server (venv SSL context can't verify its certs)
export GCE_METADATA_MTLS_MODE=none

# Start worker entrypoint
export WORKER_ID="$WORKER_ID"
export CONTROLLER_HOST="$CONTROLLER_IP"
export CONTROLLER_URL="http://${CONTROLLER_IP}:8080"
export HEARTBEAT_PORT=5000
export HTTP_PORT=8080
export TF_PORT="$TF_PORT"
export CHECKPOINT_DIR="/tmp/elastf_checkpoints"
export GCS_BUCKET="$GCS_BUCKET"
export EPOCHS="$EPOCHS"
export LIGHT_MODEL="$LIGHT_MODEL"
export MEDIUM_MODEL="$MEDIUM_MODEL"
export BATCH_SIZE="$BATCH_SIZE"
export EXPECTED_WORKERS="$EXPECTED_WORKERS"

echo "[startup] Starting worker entrypoint (worker_id=$WORKER_ID, tf_port=$TF_PORT)..."
nohup /opt/elastf_venv/bin/python3 -m elas_tf.worker_entrypoint >> /var/log/elastf.log 2>&1 &
ENTRYPOINT_PID=$!
echo "$ENTRYPOINT_PID" > /tmp/elastf_entrypoint.pid
echo "[startup] Worker entrypoint started (pid=$ENTRYPOINT_PID)"
