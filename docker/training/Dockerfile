# Base image
FROM python:3.9-slim

# Setup working directory
WORKDIR /app
# Copy requirements.txt and install dependencies
COPY docker/training/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
# Copy the training script to the working directory
COPY docker/training/train.py /app/train.py
# Store model in a shared volume
VOLUME ["/app/models"]
ENV MODEL_DIR=/app/models
# TODO: Set the command to run the training script
CMD ["python", "/app/train.py"]
