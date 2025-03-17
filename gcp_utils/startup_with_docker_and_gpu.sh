##### HOW TO USE #####
# git clone -b vit-delivery-ferran https://github.com/BlueEquinoxDev/waste-detection-aidl.git
# cd waste-detection-aidl
# chmod +x gcp_utils/startup_with_docker_and_gpu.sh
# ./gcp_utils/startup_with_docker_and_gpu.sh

# Once changes are made the the scripts
# Change the command to be executed at the docker-compose.yml file. Then:
# docker-compose build
# docker-compose up
######################

#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "Navigating to waste-detection-aidl directory..."
# cd waste-detection-aidl

echo "Installing required Python packages..."
sudo apt update
sudo apt install -y python3-pip
sudo pip3 install Pillow scikit-learn

echo "Running dataset scripts..."
sudo python3 -m scripts.download
sudo python3 -m scripts.split_dataset --dataset_dir data --dataset_type taco1

echo "Setting up NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "Updating system and installing NVIDIA toolkit..."
sudo apt update
sudo apt install -y nvidia-container-toolkit nvidia-docker2 docker-compose docker

echo "Restarting Docker..."
sudo systemctl restart docker

echo "Verifying NVIDIA GPU support in Docker..."
sudo docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi

echo "Running waste-detection-app with Docker Compose..."
sudo docker-compose build
sudo docker-compose up

echo "Setup completed successfully!"