#!/bin/bash

# 0. Variables setting
GIT_REPO_URL="https://github.com/fmirogea/flask-docker-hello-world.git"
DEST_DIR="/opt/docker-compose-app"
USER=$(whoami)

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting VM initialization script..." >> /tmp/startup-script.log 2>&1

# 1. Update and install necessary tools
echo "Updating package list and installing required tools..." >> /tmp/startup-script.log 2>&1
sudo apt-get update -y >> /tmp/startup-script.log 2>&1
sudo apt-get install -y git docker.io docker-compose >> /tmp/startup-script.log 2>&1

# Ensure Docker is enabled and started
echo "Starting and enabling Docker service..." >> /tmp/startup-script.log 2>&1
sudo systemctl start docker >> /tmp/startup-script.log 2>&1
sudo systemctl enable docker >> /tmp/startup-script.log 2>&1

# Add the current user to the Docker group (optional, requires re-login for effect)
echo "Adding current user to the Docker group..." >> /tmp/startup-script.log 2>&1
sudo usermod -aG docker $USER >> /tmp/startup-script.log 2>&1

# 2. Clone the Git repository
echo "Cloning the repository..." >> /tmp/startup-script.log 2>&1
if [ -d "$DEST_DIR" ]; then
  echo "Directory $DEST_DIR already exists. Pulling latest changes..." >> /tmp/startup-script.log 2>&1
  cd "$DEST_DIR" >> /tmp/startup-script.log 2>&1
  git pull >> /tmp/startup-script.log 2>&1
else
  echo "Cloning the repository into $DEST_DIR..." >> /tmp/startup-script.log 2>&1
  git clone "$GIT_REPO_URL" "$DEST_DIR" >> /tmp/startup-script.log 2>&1
fi

# 3. Navigate to the repository
cd "$DEST_DIR" >> /tmp/startup-script.log 2>&1

# 4. Build and start the Docker Compose application
echo "Starting Docker Compose application..." >> /tmp/startup-script.log 2>&1
docker-compose pull >> /tmp/startup-script.log 2>&1 # Pull prebuilt images if available >> /tmp/startup-script.log 2>&1
docker-compose up -d --build  >> /tmp/startup-script.log 2>&1  # Build and start in detached mode

echo "VM initialization script completed successfully."  >> /tmp/startup-script.log 2>&1