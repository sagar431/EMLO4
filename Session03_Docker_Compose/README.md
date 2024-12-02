# MNIST Training Pipeline with Docker Compose

This project implements a complete MNIST training pipeline using Docker Compose, featuring distributed training with Hogwild, model evaluation, and inference services.

## Project Structure

```
.
├── Dockerfile              # Base Dockerfile for all services
├── docker-compose.yml      # Docker Compose configuration
├── src/
│   ├── model.py           # CNN model architecture
│   ├── train.py           # Training script with Hogwild
│   ├── eval.py            # Model evaluation script
│   └── infer.py           # Inference script
└── tests/
    └── grading.sh         # Test script for validation
```

## Features

- **Distributed Training**: Uses PyTorch's Hogwild implementation with 2 processes
- **Model Evaluation**: Computes and saves test loss and accuracy metrics
- **Inference Service**: Generates predictions for random MNIST test images
- **Shared Volume**: Persistent storage for model checkpoints and results
- **GPU Support**: CUDA-enabled for all services

## Requirements

- Docker
- Docker Compose
- NVIDIA Container Toolkit (for GPU support)

## Quick Start

1. Build the Docker images:
```bash
docker compose build
```

2. Run the services:
```bash
# Run training
docker compose run train

# Run evaluation
docker compose run evaluate

# Run inference
docker compose run infer
```

3. Run all tests:
```bash
./tests/grading.sh
```

## Output Structure

The pipeline creates the following structure in the shared volume:

```
/mnist/
├── model/
│   ├── mnist_cnn.pt         # Model checkpoint
│   └── eval_results.json    # Evaluation metrics
└── results/
    └── predicted_*.png      # Inference results
```

## Service Details

### Train Service
- Implements Hogwild distributed training
- Trains for 1 epoch
- Saves model checkpoint
- Resumes from checkpoint if available

### Evaluate Service
- Loads trained model
- Computes test loss and accuracy
- Saves metrics to JSON file

### Inference Service
- Selects 5 random test images
- Generates predictions
- Saves images with predicted labels

## Testing

The `grading.sh` script validates:
1. Model checkpoint creation
2. Evaluation results generation
3. Inference output generation
4. File structure and naming conventions

## License

MIT

## Acknowledgments

Based on PyTorch's MNIST Hogwild example: https://github.com/pytorch/examples/tree/main/mnist_hogwild
