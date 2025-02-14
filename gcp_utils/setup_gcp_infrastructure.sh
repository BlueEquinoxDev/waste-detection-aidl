#!/bin/bash

# Step 1: Source the config file
if [ -f "gcp_utils/config/config.env" ]; then
  source gcp_utils/config/config.env
else
  echo "Configuration file 'config.env' not found!"
  exit 1
fi

# Step 2: Setting up the Google Cloud SDK

echo "Starting Google Cloud SDK setup..."

# Enable the required services:
gcloud services enable compute.googleapis.com

# Set the active project
gcloud config set project "$PROJECT_ID"

# Step 5: Create a Compute Engine VM instance

echo "Setting up Compute Engine VM instance..."

# Create a firewall rule to open port 5000
gcloud compute firewall-rules create allow-tcp-5000 \
  --direction=INGRESS \
  --priority=1000 \
  --network=default \
  --action=ALLOW \
  --rules=tcp:5000 \
  --source-ranges=0.0.0.0/0 \
  --target-tags=http-server

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Script directory: $SCRIPT_DIR"

# Create the VM instance
gcloud compute instances create "$VM_NAME" \
  --zone="$LOCATION"-b \
  --machine-type=e2-micro \
  --image-family=debian-11 \
  --image-project=debian-cloud \
  --boot-disk-size=30GB \
  --tags=http-server,https-server \
  --service-account="my-vm-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
  --metadata-from-file startup-script="$SCRIPT_DIR/startup_script.sh"

# Echo the public IP address of the VM
VM_PUBLIC_IP=$(gcloud compute instances describe "$VM_NAME" --zone="$LOCATION"-a --format="get(networkInterfaces[0].accessConfigs[0].natIP)")
echo "The public IP address of the VM is: $VM_PUBLIC_IP"