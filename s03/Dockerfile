FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install additional dependencies
RUN pip install torchvision==0.16.0

# Copy the source code
COPY src/ /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1
