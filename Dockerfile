FROM python:3.10.12

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements_TACO.txt .
RUN pip install --upgrade pip && pip install -r requirements_TACO.txt

# Copy the rest of the application code
COPY . .

# Set the entry point to run Python files
ENTRYPOINT ["python"]
