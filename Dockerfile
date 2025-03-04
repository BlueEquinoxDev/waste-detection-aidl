FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Run the application
CMD ["python", "-m", "scripts.train_mask2former_segmentation"]