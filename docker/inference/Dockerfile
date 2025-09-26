# Base image
FROM python:3.9-slim

# Setup working directory
WORKDIR /app
# Copy requirements.txt and install dependencies
COPY docker/training/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
# Copy the inference script to the working directory
COPY docker/inference/server.py /app/server.py
ENV MODEL_PATH=/app/models/iris_model.pkl
EXPOSE 8080
# Set the command to run the server
CMD ["python", "/app/server.py"]
