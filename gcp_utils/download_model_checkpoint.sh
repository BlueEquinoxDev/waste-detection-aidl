#!/bin/bash

# Check if the model argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Model argument is required."
    echo "Usage: ./download_model_checkpoint.sh [model]"
    echo "Where [model] can be: resnet50, vit, mask2former, or maskrcnn"
    exit 1
fi

# Get the model argument and validate it
MODEL="$1"
VALID_MODELS=("resnet50" "vit" "mask2former" "maskrcnn")

# Check if the model argument is valid
VALID_MODEL=false
for valid_model in "${VALID_MODELS[@]}"; do
    if [ "$MODEL" == "$valid_model" ]; then
        VALID_MODEL=true
        break
    fi
done

if [ "$VALID_MODEL" = false ]; then
    echo "Error: Invalid model specified: $MODEL"
    echo "Valid models are: resnet50, vit, mask2former, maskrcnn"
    exit 1
fi

# Set model-specific paths
case "$MODEL" in
    "resnet50")
        BUCKET_PATH="gs://aidl2025-waste-models/models/resnet50"
        LOCAL_SAVE_PATH="./results/resnet50"
        ;;
    "vit")
        BUCKET_PATH="gs://aidl2025-waste-models/models/vit" 
        LOCAL_SAVE_PATH="./results/vit"
        ;;
    "mask2former")
        BUCKET_PATH="gs://aidl2025-waste-models/models/mask2former" 
        LOCAL_SAVE_PATH="./results/mask2former"
        ;;
    "maskrcnn")
        BUCKET_PATH="gs://aidl2025-waste-models/models/maskrcnn" 
        LOCAL_SAVE_PATH="./results/maskrcnn"
        ;;
esac

echo "Downloading $MODEL model checkpoint..."

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

# Create/update a symlink in the app/checkpoint directory (if it exists)
APP_CHECKPOINT_DIR="./app/checkpoint"
if [ -d "$APP_CHECKPOINT_DIR" ]; then
    # Create symlink based on model name
    ln -sf "../$LOCAL_SAVE_PATH/$ARCHIVE_NAME" "$APP_CHECKPOINT_DIR/${MODEL}_checkpoint.pt"
    echo "Created symlink in app/checkpoint directory"
fi

echo "Checkpoint for $MODEL downloaded and extracted to: $LOCAL_SAVE_PATH"
echo "You can use it with MODEL_NAME=\"${MODEL^^}\" environment variable"