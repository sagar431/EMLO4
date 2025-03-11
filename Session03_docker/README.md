# Docker Compose for MNIST Training, Evaluation, and Inference

This project demonstrates how to use Docker Compose to perform training, evaluation, and inference on the MNIST dataset using the MNIST Hogwild example from PyTorch.

## Project Structure

```
Session03_docker/
├── model.py                  # Shared model definition
├── docker-compose.yml        # Docker Compose configuration
├── model-train/
│   ├── train.py              # Training script
│   ├── check_train.py        # Script to check if training was successful
│   └── Dockerfile.train      # Dockerfile for training service
├── model-eval/
│   ├── evaluate.py           # Evaluation script
│   ├── check_eval.py         # Script to check if evaluation was successful
│   └── Dockerfile.eval       # Dockerfile for evaluation service
└── model-inference/
    ├── infer.py              # Inference script
    └── Dockerfile.infer      # Dockerfile for inference service
```

## Requirements

- Docker
- Docker Compose

## Implementation Details

- **Base Image**: Uses `satyajitghana/pytorch:2.3.1` which is a CPU-only PyTorch image
- **Dependencies**: Each service installs matplotlib for visualization and data handling
- **Training**: Uses MNIST Hogwild with 2 processes to train for 1 epoch. If a checkpoint is found, it resumes training from that checkpoint.
- **Evaluation**: Evaluates the trained model and saves the results in a JSON file.
- **Inference**: Runs inference on 5 random MNIST images and saves the results with the predicted number as the filename.

## Shared Volume

All services share a volume called `mnist` which is mounted at `/opt/mount` in each container. This volume contains:
- The MNIST dataset (downloaded during training) in `/opt/mount/data`
- The trained model checkpoint in `/opt/mount/model/mnist_cnn.pt`
- The evaluation results in `/opt/mount/model/eval_results.json`
- The inference results in `/opt/mount/results/` directory

## How to Use

1. Build all Docker images:
   ```
   docker compose build
   ```

2. Run the training service:
   ```
   docker compose run train
   ```

3. Run the evaluation service:
   ```
   docker compose run evaluate
   ```

4. Run the inference service:
   ```
   docker compose run infer
   ```

5. Check if the checkpoint file was created:
   ```
   docker compose run train python check_train.py
   ```

6. Check if the evaluation results were saved:
   ```
   docker compose run evaluate python check_eval.py
   ```

## Testing

You can run the provided test script to verify that all parts of the setup are working correctly:

```bash
./test.sh
```

The test script will:
1. Build all Docker images
2. Run all services in sequence
3. Check if the checkpoint file exists
4. Check if the evaluation results file exists
5. Print the evaluation results
6. Check if 5 inference result images were created

## Troubleshooting

If you encounter any issues with dependencies, each Dockerfile installs the necessary packages. The inference script also has a fallback mechanism if matplotlib is not available - it will save the raw tensor data instead of visualizing the images. 