# 🐕 Dog Breed Classifier

A deep learning model that classifies dog breeds using PyTorch Lightning and ResNet50. Features a web interface for real-time predictions and Docker support for easy deployment.

## 🏆 CML Training Report

### 📊 Training Metrics

```json
{
  "best_val_acc": 0.9879940152168274,
  "test_acc": 0.9964007139205933,
  "test_loss": 0.010746844112873077
}
```

| Metric | Value |
|--------|-------|
| **Best Validation Accuracy** | 98.80% |
| **Test Accuracy** | **99.64%** |
| **Test Loss** | 0.0107 |

### 📈 Training & Validation Accuracy
![Training & Validation Accuracy](https://asset.cml.dev/866d451eecaf15f6f19a75d5ded0e91de36f8057?cml=png&cache-bypass=09059ad0-b0e2-4c91-a897-c9dd5129f147)

### 📉 Training & Validation Loss
![Training & Validation Loss](https://asset.cml.dev/6ffe361f831b71711eaa2435a8548b20071f4674?cml=png&cache-bypass=58b742ce-8370-4a36-9b8f-b51a2c9a3c1f)

### 🔢 Confusion Matrices

#### Training Set
![Training Confusion Matrix](https://asset.cml.dev/0ab7a3e46fe67be779635bee67e7750c5480ccec?cml=png&cache-bypass=3093bcad-daad-4438-a77e-cc38794d72fd)

#### Test Set
![Test Confusion Matrix](https://asset.cml.dev/681a3db28cfdb708d7ce7d00e94cddb1bb085aa4?cml=png&cache-bypass=91f8816e-6dff-46d3-a436-b618d561ecec)

### 🔍 Inference Results (Sample Test Images)

![Inference 1](https://asset.cml.dev/6a110fe7f6410fdc7b7a84a7e5bda3abe5b4165e?cml=png&cache-bypass=5ce0cc7f-f62b-4c61-bd99-f8464435819f)
![Inference 2](https://asset.cml.dev/68c0d6fe28e401825766b4076ba420e05ed439ff?cml=png&cache-bypass=8e2e8f02-b1cb-407e-9b24-38a094bc6c5b)
![Inference 3](https://asset.cml.dev/33f530bc247ad120eea5d32bf732f6321870dadf?cml=png&cache-bypass=0b61894b-08ba-445e-a0ef-19ebaea581d7)
![Inference 4](https://asset.cml.dev/de4efc4f90f9cc7dd235df0026b7188d6ca36051?cml=png&cache-bypass=a083f3dc-9261-4e2d-bd1a-cba0444a81d6)
![Inference 5](https://asset.cml.dev/aa2854df207ae038bbe235178e4bd75e745f68b4?cml=png&cache-bypass=3bafcb72-e70c-4ca3-b278-1d9f12512d7b)

---

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

#### Using `docker run` Directly

```bash
# Build the base image first
docker build -f docker/base.Dockerfile -t dog-breed-base:latest .

# Train
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  dog-breed-base:latest python src/train.py

# Eval
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/predictions:/app/predictions \
  dog-breed-base:latest python src/eval.py

# Infer
docker run --gpus all \
  -v $(pwd)/src/samples:/app/input \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/predictions:/app/predictions \
  dog-breed-base:latest python src/infer.py -input_folder /app/input -output_folder /app/predictions -ckpt checkpoints/dog-breed-epoch=07-val_acc=1.00.ckpt
```

#### Download Dataset with Kaggle API

```bash
# Install Kaggle CLI
pip install kaggle

# Set up Kaggle API credentials (download from kaggle.com/account)
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download and extract the dataset
kaggle datasets download -d khushikhushikhushi/dog-breed-image-dataset -p data/
unzip data/dog-breed-image-dataset.zip -d data/dog-breed-dataset
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
