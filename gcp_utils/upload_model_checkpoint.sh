#!/bin/bash

# Check if all required arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <local_checkpoint_path> <model_name> <dataset_name>"
    echo "Example: $0 ./results/seg-mask2former-taco-20240220-123456 MaskFormer TACO"
    exit 1
fi

# Set Variables from arguments
LOCAL_CHECKPOINT_PATH="$1"  # Folder containing checkpoint files
MODEL_NAME="$2"            # Model name
DATASET_NAME="$3"          # Dataset name

# Set fixed Variables
BUCKET_NAME="aidl2025-waste-models"
GCS_DESTINATION_PATH="models"  # Base folder in GCS

# Generate timestamp (YYYYMMDDHHMM)
TIMESTAMP=$(date +"%Y%m%d%H%M")

# Construct the folder name: TIMESTAMP_MODELNAME_DATASETNAME
FOLDER_NAME="${TIMESTAMP}_${MODEL_NAME}_${DATASET_NAME}"

# Create a compressed archive of checkpoint files
ARCHIVE_NAME="checkpoint_${FOLDER_NAME}.tar.gz"
ARCHIVE_PATH="/tmp/$ARCHIVE_NAME"  # Temporary storage location

echo "Compressing checkpoint files from $LOCAL_CHECKPOINT_PATH into $ARCHIVE_PATH ..."
tar -czf "$ARCHIVE_PATH" -C "$LOCAL_CHECKPOINT_PATH" .

# Ensure gsutil is installed
if ! command -v gsutil &> /dev/null
then
    echo "Error: gsutil not found! Install Google Cloud SDK first."
    exit 1
fi

# Ensure the archive was created successfully
if [ ! -f "$ARCHIVE_PATH" ]; then
    echo "Error: Failed to create archive $ARCHIVE_PATH"
    exit 1
fi

# Construct full GCS path
FULL_GCS_PATH="gs://$BUCKET_NAME/$GCS_DESTINATION_PATH/$FOLDER_NAME/"

# Upload the archive to GCS
echo "Uploading archive to $FULL_GCS_PATH ..."
gsutil cp "$ARCHIVE_PATH" "$FULL_GCS_PATH"

# Verify upload
if [ $? -eq 0 ]; then
    echo "Upload successful! File saved in $FULL_GCS_PATH$ARCHIVE_NAME"
    # Cleanup the temporary archive file
    rm "$ARCHIVE_PATH"
else
    echo "Error: Failed to upload checkpoint archive."
    exit 1
fi
