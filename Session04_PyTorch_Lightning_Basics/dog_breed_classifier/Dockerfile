FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the source code
COPY src/ src/
COPY checkpoints/ checkpoints/

# Create necessary directories
RUN mkdir -p data logs predictions

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Default command to run inference
CMD ["python", "src/infer.py", "-input_folder", "src/samples", "-output_folder", "predictions", "-ckpt", "checkpoints/dog-breed-epoch=07-val_acc=1.00.ckpt"]
