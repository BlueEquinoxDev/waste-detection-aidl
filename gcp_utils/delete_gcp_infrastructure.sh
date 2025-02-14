#!/bin/bash

# Step 1: Source the config file
if [ -f "gcp_utils/config/config.env" ]; then
  source gcp_utils/config/config.env
else
  echo "Configuration file 'config.env' not found!"
  exit 1
fi

# Step 2: Delete everything

echo "Cleaning up..."

# Delete the firewall rule
if gcloud compute firewall-rules delete allow-tcp-5000 --quiet; then
    echo "Firewall rule allow-tcp-5000 deleted successfully."
else
    echo "Error: Failed to delete the firewall rule allow-tcp-5000." >&2
    exit 1
fi

# Delete the VM instance
if gcloud compute instances delete "$VM_NAME" --zone="${LOCATION}-b" --quiet; then
    echo "VM instance $VM_NAME deleted successfully."
else
    echo "Error: Failed to delete the VM instance $VM_NAME." >&2
    exit 1
fi

echo "Google Cloud infrastructure cleanup complete."