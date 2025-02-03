# README for GCS Checkpoint Scripts

This repository contains two Bash scripts to manage ML model checkpoints stored in Google Cloud Storage (GCS).

## Prerequisites

- Google Cloud SDK installed (`gsutil`).
- Google Cloud Storage bucket set up.
- Access permissions to the bucket.

## Usage

### 1. **Uploading Checkpoints (`upload_model_checkpoint.sh`)**

- **Purpose**: Compresses all checkpoint files in a local directory and uploads them to GCS with a timestamped folder.
  
**Run the script**:
```
chmod +x upload_model_checkpoint.sh
./upload_model_checkpoint.sh
```

You have to modify the following variables in the script:

- `LOCAL_CHECKPOINT_PATH`: Path to your local checkpoint directory.
- `BUCKET_NAME`: Your GCS bucket name.
- `GCS_DESTINATION_PATH`: Base path in the GCS bucket.

### 2. **Downloading Model Checkpoint (`download_model_checkpoint.sh`)**

- **Purpose**: Downloads the checkpoint file from the specified GCS bucket and extracts it to the local system.

**Run the script**:
```
chmod +x download_model_checkpoint.sh
./download_model_checkpoint.sh
```
You have to modify the following variables in the script:

- `BUCKET_PATH`: Full GCS path to where checkpoints are stored
- `LOCAL_SAVE_PATH`: Where to extract the checkpoint

