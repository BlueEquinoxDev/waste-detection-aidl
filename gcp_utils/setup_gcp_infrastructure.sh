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
# Create the VM instance
gcloud compute instances create "$VM_NAME" \
  --zone="$LOCATION"-b \
  --machine-type=g2-standard-4 \
  --accelerator=count=1,type=nvidia-l4 \
  --create-disk=auto-delete=yes,boot=yes,device-name=instance-20250215-111128,image=projects/ubuntu-os-accelerator-images/global/images/ubuntu-accelerator-2204-amd64-with-nvidia-550-v20250213,mode=rw,size=30,type=pd-balanced \
  --maintenance-policy=TERMINATE \
  --provisioning-model=STANDARD \
  --tags=http-server,https-server \
  --service-account="my-vm-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
  --metadata-from-file startup-script="$SCRIPT_DIR/startup_script.sh"

  # gcloud compute instances create instance-20250215-111128 \
  #   --project=aidl2025-waste \
  #   --zone=europe-west4-c \
  #   --machine-type=g2-standard-4 \
  #   --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
  #   --maintenance-policy=TERMINATE \
  #   --provisioning-model=STANDARD \
  #   --service-account=1014632772558-compute@developer.gserviceaccount.com \
  #   --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append \
  #   --accelerator=count=1,type=nvidia-l4 \
  #   --create-disk=auto-delete=yes,boot=yes,device-name=instance-20250215-111128,image=projects/ubuntu-os-accelerator-images/global/images/ubuntu-accelerator-2204-amd64-with-nvidia-550-v20250213,mode=rw,size=30,type=pd-balanced \
  #   --no-shielded-secure-boot \
  #   --shielded-vtpm \
  #   --shielded-integrity-monitoring \
  #   --labels=goog-ec-src=vm_add-gcloud \
  #   --reservation-affinity=any

# Echo the public IP address of the VM
VM_PUBLIC_IP=$(gcloud compute instances describe "$VM_NAME" --zone="$LOCATION"-b --format="get(networkInterfaces[0].accessConfigs[0].natIP)")
echo "The public IP address of the VM is: $VM_PUBLIC_IP"



#### END OF SCRIPT ####
# 1. SSH into the machine #
# gcloud compute ssh --zone "$LOCATION"-b "$VM_NAME"
#
# 2. Check logs
# cat /tmp/startup_script.log
#
# 3. Change to project directory
# cd /opt/docker-app
#
# 4. Build Docker container
# Build the image with:
# ```sudo docker build -t waste-detection-app .```

# 5. Run specific Python file:
# ```sudo docker run --rm -it waste-detection-app <FILE_NAME.py>```
# ```sudo docker run --rm -it waste-detection-app -m scripts.train_vit_classification```

