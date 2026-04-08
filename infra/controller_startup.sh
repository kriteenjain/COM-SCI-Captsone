#!/usr/bin/env bash
#
# GCP VM startup script for the ElasTF controller.
# Runs automatically when the VM boots.
#
set -euo pipefail
exec > >(tee -a /var/log/elastf.log) 2>&1

echo "[startup] ElasTF Controller startup script running..."
echo "[startup] $(date)"

# Fetch metadata
REPO_URL=$(curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/repo_url" || echo "")
BRANCH=$(curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/branch" || echo "main")
GCS_BUCKET=$(curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/gcs_bucket" || echo "")

echo "[startup] Repo: $REPO_URL ($BRANCH)"
echo "[startup] GCS bucket: $GCS_BUCKET"

# Install dependencies
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv git

# Clone repo
if [ -d /opt/elastf ]; then
    cd /opt/elastf && git pull origin "$BRANCH"
else
    git clone --branch "$BRANCH" "$REPO_URL" /opt/elastf
fi
cd /opt/elastf/ElasTF

# Set up venv
python3 -m venv /opt/elastf_venv
source /opt/elastf_venv/bin/activate
pip install --quiet -r requirements.txt

# Create local dirs for backward compatibility
mkdir -p shared/config shared/checkpoints

# Start controller
export CONFIG_DIR=shared/config
export CHECKPOINT_DIR=shared/checkpoints
export HEARTBEAT_PORT=5000
export HTTP_PORT=8080
export GCS_BUCKET="$GCS_BUCKET"

echo "[startup] Starting controller (heartbeat=5000, http=8080)..."
nohup python3 -m elas_tf.controller >> /var/log/elastf.log 2>&1 &
echo "[startup] Controller started (pid=$!)"
