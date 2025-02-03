#!/bin/bash

# Set Variables
BUCKET_PATH="gs://aidl2025-waste-models/models/202502012134_ViT_Viola_TACO"  # Full GCS path to where checkpoints are stored (e.g., gs://your-gcs-bucket-name/models)
LOCAL_SAVE_PATH="./results/202502012134_ViT_Viola_TACO"  # Where to extract the checkpoint

# Ensure BUCKET_PATH is provided
if [ -z "$BUCKET_PATH" ]; then
    echo "Error: Please provide the full GCS bucket path as an argument."
    echo "Usage: ./download_latest_checkpoint.sh gs://your-gcs-bucket-name/models"
    exit 1
fi

# Ensure gsutil is installed
if ! command -v gsutil &> /dev/null
then
    echo "Error: gsutil not found! Install Google Cloud SDK first."
    exit 1
fi

# Find the latest file in the given BUCKET_PATH
echo "Finding the latest file in $BUCKET_PATH ..."
LATEST_FILE=$(gsutil ls "$BUCKET_PATH" | sort | tail -n 1)

# Ensure a file exists
if [ -z "$LATEST_FILE" ]; then
    echo "Error: No files found in $BUCKET_PATH!"
    exit 1
fi

echo "Latest file found: $LATEST_FILE"

# Create local save directory if it doesn't exist
mkdir -p "$LOCAL_SAVE_PATH"

# Download the latest checkpoint file
echo "Downloading checkpoint file: $LATEST_FILE ..."
gsutil cp "$LATEST_FILE" "$LOCAL_SAVE_PATH/"

# If the file is a tar.gz archive, extract it
ARCHIVE_NAME=$(basename "$LATEST_FILE")
if [[ "$ARCHIVE_NAME" == *.tar.gz ]]; then
    echo "Extracting $ARCHIVE_NAME ..."
    tar -xzf "$LOCAL_SAVE_PATH/$ARCHIVE_NAME" -C "$LOCAL_SAVE_PATH"
    
    # Cleanup the archive after extraction
    rm "$LOCAL_SAVE_PATH/$ARCHIVE_NAME"
fi

echo "Checkpoint downloaded and extracted to: $LOCAL_SAVE_PATH"
