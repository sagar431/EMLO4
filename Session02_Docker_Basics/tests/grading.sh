#!/bin/bash
set -e

echo "Building Docker image..."
docker build -t mnist-pytorch .

echo "Running container with mounted volume..."
docker run -v $(pwd)/checkpoints:/app/checkpoints mnist-pytorch

echo "Checking if checkpoint was created..."
if [ -f "checkpoints/mnist_checkpoint.pt" ]; then
    echo "✅ Checkpoint file created successfully"
else
    echo "❌ Checkpoint file not found"
    exit 1
fi

echo "All tests passed! 🎉"
