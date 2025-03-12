# 🐕 Dog Breed Classifier

A deep learning model that classifies dog breeds using PyTorch Lightning and ResNet50. Features a web interface for real-time predictions and Docker support for easy deployment.

## Features

- ResNet50 model with pretrained weights
- PyTorch Lightning for efficient training
- FastAPI web service with modern UI
- Docker support for containerized deployment
- Real-time predictions with confidence scores
- Comprehensive logging with loguru
- Rich CLI output with progress bars
- Data augmentation for better generalization
- Model evaluation metrics in JSON format

## Project Structure
```
/root
├── src/
│   ├── train.py         # Training script
│   ├── eval.py          # Evaluation script
│   ├── infer.py         # Inference script
│   ├── app.py           # FastAPI web service
│   ├── datamodule.py    # Dataset and DataModule
│   ├── utils/
│   │   └── logging_config.py
│   └── samples/         # Sample images for inference
├── docker/              # Docker configuration
│   ├── base.Dockerfile
│   ├── train.Dockerfile
│   ├── eval.Dockerfile
│   └── infer.Dockerfile
├── data/                # Dataset directory
├── logs/                # Log files
├── checkpoints/         # Model checkpoints
├── predictions/         # Inference results
└── docker-compose.yml   # Docker Compose configuration

## Supported Dog Breeds

1. Beagle
2. Boxer
3. Bulldog
4. Dachshund
5. German Shepherd
6. Golden Retririver
7. Labrador Retriever
8. Poodle
9. Rottweiler
10. Yorkshire Terrier

## Model Performance

- Training Accuracy: >95%
- Validation Accuracy: 100%
- Inference Time: ~0.1s per image
- GPU Support: CUDA-enabled with automatic device selection

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd dog-breed-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Direct Usage

#### Training

Train the model on your dataset:
```bash
python src/train.py
```

The best model will be saved to `checkpoints/` based on validation accuracy.

#### Evaluation

Evaluate the model's performance:
```bash
python src/eval.py
```

Results will be saved to `predictions/eval_metrics.json`.

#### Inference CLI

Run inference on individual images:
```bash
python src/infer.py -input_folder path/to/images -output_folder predictions
```

#### Web Service

1. Start the web service:
```bash
PYTHONPATH=/root uvicorn src.app:app --host 0.0.0.0 --port 8080
```

2. Access the web interface:
   - Open http://localhost:8080 in your browser
   - Upload an image using the "Choose Image" button
   - Click "Classify Dog Breed" to get predictions

### Option 2: Docker Deployment

#### Prerequisites
- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- nvidia-docker2 installed

#### Running with Docker Compose

1. **Build all images:**
```bash
docker-compose build
```

2. **Training:**
```bash
docker-compose run train
```

3. **Evaluation:**
```bash
docker-compose run eval
```

4. **Inference:**
```bash
docker-compose run infer
```

5. **Web Service:**
```bash
docker-compose run -p 8080:8080 web
```

#### Volume Mounts
- `/app/data`: Dataset directory
- `/app/checkpoints`: Model checkpoints
- `/app/predictions`: Output predictions
- `/app/input`: Input images for inference

#### GPU Support
All services are configured to use NVIDIA GPUs through nvidia-docker2. Installation:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### API Endpoints

- `GET /`: Web interface
- `POST /predict`: Submit an image for prediction
  - Request: multipart/form-data with 'file' field
  - Response: JSON with breed prediction and confidence
- `/docs`: Interactive API documentation (Swagger UI)
- `/redoc`: Alternative API documentation (ReDoc)

## Development

### Project Components

1. **Model Architecture**
   - ResNet50 backbone with pretrained weights
   - Custom classifier head for 10 dog breeds
   - Cross-entropy loss function

2. **Training Pipeline**
   - Data augmentation (random flips, rotations)
   - Learning rate scheduling
   - Early stopping and model checkpointing
   - Rich progress visualization

3. **Web Service**
   - FastAPI backend for efficient API handling
   - Modern, responsive web interface
   - Real-time image processing
   - Confidence score visualization

### Adding New Features

1. **New Dog Breeds**
   - Add breed images to `data/dog-breed-dataset/`
   - Update class mapping in `datamodule.py`
   - Retrain the model

2. **Custom Model Architecture**
   - Modify `train.py` to use different backbone
   - Adjust hyperparameters as needed

3. **UI Customization**
   - Modify HTML/CSS/JS in `app.py`
   - Add new API endpoints as needed

## Logging

- Training logs: `logs/training.log`
- Evaluation logs: `logs/evaluation.log`
- Inference logs: `logs/inference.log`
- Web service logs: Standard output

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
