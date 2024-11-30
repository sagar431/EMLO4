# Docker Basics - MNIST Training Project

This session covers:
- PyTorch-based MNIST training in Docker
- Checkpoint saving and resumption
- Docker volume mounting for model persistence
- Lightweight Docker image (< 1.1GB)

## Quick Start

1. Build the Docker image:
```bash
docker build -t mnist-pytorch .
```

2. Run training (creates and saves checkpoint):
```bash
docker run -v $(pwd)/checkpoints:/app/checkpoints mnist-pytorch
```

3. Resume training from checkpoint:
```bash
docker run -v $(pwd)/checkpoints:/app/checkpoints mnist-pytorch python train.py --resume
```

## Project Structure
- `Dockerfile`: Defines the container environment
- `train.py`: MNIST training script with checkpoint support
- `requirements.txt`: Python dependencies
- `tests/grading.sh`: Test script to verify functionality

## Features
- Efficient PyTorch CPU training
- Checkpoint saving to host system
- Training resumption capability
- Progress bar with training metrics
- Test set evaluation

## Docker Hub
To pull the pre-built image:
```bash
docker pull <your-dockerhub-username>/mnist-pytorch:latest
