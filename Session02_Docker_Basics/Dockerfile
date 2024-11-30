docker login
docker tag mnist-pytorch <your-dockerhub-username>/mnist-pytorch:latest
docker push <your-dockerhub-username>/mnist-pytorch:latestFROM python:3.9-slim

WORKDIR /app

# Install only necessary dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training script
COPY train.py .

# Create directories for data and checkpoints
RUN mkdir -p /app/data /app/checkpoints

# Set volume for checkpoints
VOLUME /app/checkpoints

# Command to run the training
CMD ["python", "train.py"]
