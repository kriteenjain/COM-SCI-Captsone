#!/usr/bin/env bash
#
# Tear down the ElasTF cluster on GCP.
# Deletes all VMs and firewall rules. Keeps the GCS bucket for results.
#
set -euo pipefail

echo ""
echo "============================================================"
echo " ElasTF Cluster Teardown"
echo "============================================================"
echo ""

# Find ElasTF VMs in ANY zone (not just one), so the script works regardless
# of which zone was used to create the cluster.
echo "[teardown] Finding ElasTF VMs across all zones..."
VMS_WITH_ZONES=$(gcloud compute instances list \
    --filter="name~'^elastf-'" \
    --format="value(name,zone)" 2>/dev/null || true)

if [ -z "$VMS_WITH_ZONES" ]; then
    echo "[teardown] No ElasTF VMs found."
else
    echo "[teardown] Deleting VMs:"
    echo "$VMS_WITH_ZONES" | while read -r name zone; do
        echo "  - $name (zone: $zone)"
    done
    echo "$VMS_WITH_ZONES" | while read -r name zone; do
        gcloud compute instances delete "$name" --zone="$zone" --quiet &
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
