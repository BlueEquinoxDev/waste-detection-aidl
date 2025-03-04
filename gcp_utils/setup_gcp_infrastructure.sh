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

# Step 3: Generate SSH key if it doesn't exist
SSH_KEY_PATH="$HOME/.ssh/id_rsa"
if [ ! -f "$SSH_KEY_PATH" ]; then
  echo "Generating SSH key at $SSH_KEY_PATH"
  mkdir -p "$HOME/.ssh"
  ssh-keygen -t rsa -b 4096 -N "" -f "$SSH_KEY_PATH"
  chmod 600 "$SSH_KEY_PATH"
  chmod 600 "$SSH_KEY_PATH.pub"
fi

# Get username (assuming it's the same as in WSL2)
USERNAME=$(whoami)
SSH_PUBLIC_KEY=$(cat "$SSH_KEY_PATH.pub")

# Step 4: Create a Compute Engine VM instance
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

gcloud compute firewall-rules create allow-vscode-remote \
  --direction=INGRESS \
  --priority=1000 \
  --network=default \
  --action=ALLOW \
  --rules=tcp:52698-52699 \
  --source-ranges=0.0.0.0/0 \
  --target-tags=http-server

# Create the VM instance with SSH key in metadata
gcloud compute instances create "$VM_NAME" \
  --zone="$LOCATION"-b \
  --machine-type=g2-standard-4 \
  --accelerator=count=1,type=nvidia-l4 \
  --create-disk=auto-delete=yes,boot=yes,device-name="$VM_NAME"-disk,image=projects/ubuntu-os-accelerator-images/global/images/ubuntu-accelerator-2204-amd64-with-nvidia-550-v20250213,mode=rw,size=100,type=pd-balanced \
  --maintenance-policy=TERMINATE \
  --provisioning-model=STANDARD \
  --tags=http-server,https-server \
  --metadata="ssh-keys=$USERNAME:$SSH_PUBLIC_KEY" \
  --service-account="my-vm-service-account@$PROJECT_ID.iam.gserviceaccount.com"

# Echo the public IP address of the VM
VM_PUBLIC_IP=$(gcloud compute instances describe "$VM_NAME" --zone="$LOCATION"-b --format="get(networkInterfaces[0].accessConfigs[0].natIP)")
echo "The public IP address of the VM is: $VM_PUBLIC_IP"

# Step 5: Set up SSH config for easy access
SSH_CONFIG_PATH="$HOME/.ssh/config"
echo "Configuring SSH for easy access..."

# Check if the host is already in config, and remove it if so
if grep -q "Host $VM_NAME" "$SSH_CONFIG_PATH" 2>/dev/null; then
  # Create a temporary file without the VM_NAME host block
  sed "/Host $VM_NAME/,/StrictHostKeyChecking no/d" "$SSH_CONFIG_PATH" > "${SSH_CONFIG_PATH}.tmp"
  mv "${SSH_CONFIG_PATH}.tmp" "$SSH_CONFIG_PATH"
fi

# Add new SSH config
cat >> "$SSH_CONFIG_PATH" << EOF
Host $VM_NAME
  HostName $VM_PUBLIC_IP
  User $USERNAME
  IdentityFile $SSH_KEY_PATH
  StrictHostKeyChecking no
  ForwardAgent yes
  ForwardX11 no
  RemoteForward 52698 127.0.0.1:52698
  AllowTcpForwarding yes

EOF

chmod 600 "$SSH_CONFIG_PATH"

# Step 6: Test SSH connection
echo "Testing SSH connection..."
ssh -o ConnectTimeout=10 -o BatchMode=yes $VM_NAME echo "SSH connection successful" || echo "SSH connection failed, but configuration is set up. You may need to wait a minute for the VM to initialize."

# Step 7: Set up VS Code configuration
echo "
==============================================
VS Code Remote SSH Connection:
==============================================

1. In your WSL2 terminal, install VS Code's Remote Development extension:
   code --install-extension ms-vscode-remote.vscode-remote-extensionpack

2. Open VS Code from WSL2:
   code .

3. In VS Code:
   - Press F1 and type 'Remote-SSH: Connect to Host'
   - Select '$VM_NAME' from the list

4. Or connect directly from terminal:
   code --remote ssh-remote+$VM_NAME ~/dev/waste-management

==============================================
"

# Include port forwarding troubleshooting
echo "
==============================================
Port Forwarding Setup:
==============================================

If you encounter port forwarding issues:

1. Ensure the SSH connection has port forwarding enabled
   - Check ~/.ssh/config has 'AllowTcpForwarding yes'
   
2. When connecting, use VS Code command palette (F1):
   - Type 'Remote-SSH: Connect to Host... with Specific Settings'
   - Enable port forwarding when prompted

==============================================
"

# Done!
echo "Setup complete! Your VM '$VM_NAME' is ready for remote development."

# After the port forwarding troubleshooting section and before the "Done!" message

# Step 8: Set up password authentication (optional)
echo "
==============================================
Setting Up Password Authentication (Optional)
==============================================
"

read -p "Would you like to set up password authentication for this VM? (y/n): " setup_password
if [[ "$setup_password" == "y" || "$setup_password" == "Y" ]]; then
  echo "Enter a password for user $USERNAME on the VM:"
  read -s VM_PASSWORD
  echo "Confirm password:"
  read -s VM_PASSWORD_CONFIRM
  
  if [ "$VM_PASSWORD" != "$VM_PASSWORD_CONFIRM" ]; then
    echo "Passwords do not match. Password authentication setup skipped."
  else
    echo "Setting up password authentication..."
    ssh $VM_NAME "sudo bash -c '
    # Enable password authentication
    sed -i \"s/PasswordAuthentication no/PasswordAuthentication yes/\" /etc/ssh/sshd_config
    # Ensure password login is allowed
    echo \"ChallengeResponseAuthentication yes\" >> /etc/ssh/sshd_config
    # Set password for user
    echo \"$USERNAME:$VM_PASSWORD\" | chpasswd
    # Restart SSH service
    systemctl restart sshd
    '"
    
    echo "
==============================================
Password Authentication Setup Complete!
==============================================

You can now connect to your VM using:
- Username: $USERNAME
- Password: (the password you just set)

When connecting through VS Code:
1. Press F1 and type 'Remote-SSH: Connect to Host'
2. Select '$VM_NAME' from the list
3. Choose 'password' authentication when prompted
==============================================
"
  fi
else
  echo "Password authentication setup skipped. Using SSH key authentication only."
fi

# Done!
echo "Setup complete! Your VM '$VM_NAME' is ready for remote development."