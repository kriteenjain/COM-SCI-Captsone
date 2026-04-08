#!/usr/bin/env bash
#
# Tear down the ElasTF cluster on GCP.
# Deletes all VMs and firewall rules. Keeps the GCS bucket for results.
#
set -euo pipefail

ZONE=${ZONE:-us-central1-a}

echo ""
echo "============================================================"
echo " ElasTF Cluster Teardown"
echo "============================================================"
echo ""

# Find and delete all elastf VMs
echo "[teardown] Finding ElasTF VMs..."
VMS=$(gcloud compute instances list --filter="name~'^elastf-'" --format="value(name)" --zones="$ZONE" 2>/dev/null || true)

if [ -z "$VMS" ]; then
    echo "[teardown] No ElasTF VMs found."
else
    echo "[teardown] Deleting VMs: $VMS"
    for vm in $VMS; do
        gcloud compute instances delete "$vm" --zone="$ZONE" --quiet &
    done
    wait
    echo "[teardown] All VMs deleted."
fi

# Delete firewall rule
echo "[teardown] Deleting firewall rules..."
gcloud compute firewall-rules delete elastf-internal --quiet 2>/dev/null || echo "[teardown] Firewall rule not found (already deleted)."

echo ""
echo "============================================================"
echo " Teardown complete."
echo " GCS bucket kept for results. To delete it:"
PROJECT=$(gcloud config get-value project 2>/dev/null)
echo "   gsutil -m rm -r gs://elastf-checkpoints-${PROJECT}/"
echo "============================================================"
