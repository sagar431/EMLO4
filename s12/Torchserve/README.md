# ⚡ Z-Image-Turbo TorchServe Deployment

A production-ready deployment of the **Tongyi-MAI/Z-Image-Turbo** model using TorchServe with a beautiful FastAPI frontend.

![Z-Image-Turbo Frontend](frontend_screenshot.png)

## 🚀 Features

- **TorchServe Backend** - Scalable model serving with GPU support
- **FastAPI Middleware** - Fast API layer with health checks and gallery
- **Beautiful Web UI** - Modern dark theme with animated effects
- **Auto-rotating Prompts** - 20+ curated prompts that shuffle automatically
- **Image Gallery** - View and manage generated images
- **Real-time Status** - Live TorchServe and model status indicators

## 📁 Project Structure

```
Torchserve/
├── app.py                 # FastAPI server with web frontend
├── handler.py             # TorchServe custom handler
├── requirements.txt       # Python dependencies for handler
├── config.properties      # TorchServe configuration
├── test_endpoint.py       # Test script for TorchServe endpoint
├── download_model.py      # Script to download model from HuggingFace
├── gen_images.py          # Batch image generation script
├── prompts.txt            # Sample prompts for testing
├── z-image-model/         # Downloaded model weights (not in git)
├── models/                # .mar files directory
│   └── z-image.mar        # Model archive (not in git)
└── generated_images/      # Output directory for generated images
```

## 🛠️ Setup Instructions

### 1. Clone and Navigate
```bash
cd /home/ubuntu/EMLO4/s12/Torchserve
```

### 2. Download the Model
```bash
python download_model.py
```
This downloads the Z-Image-Turbo model (~20GB) from HuggingFace.

### 3. Create Model Archive (Light Version)
```bash
touch empty.txt
torch-model-archiver --model-name z-image \
    --version 1.0 \
    --handler handler.py \
    --extra-files empty.txt \
    -r requirements.txt \
    --archive-format zip-store
mkdir -p models && cp z-image.mar models/
```

### 4. Start TorchServe with Docker
```bash
sudo docker run --rm --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p8080:8080 -p8081:8081 -p8082:8082 \
    -p7070:7070 -p7071:7071 \
    --gpus all \
    -v $(pwd)/config.properties:/home/model-server/config.properties \
    -v $(pwd)/z-image-model:/home/model-server/weights \
    --mount type=bind,source=$(pwd)/models,target=/tmp/models \
    pytorch/torchserve:0.12.0-gpu torchserve --model-store=/tmp/models
```

### 5. Start FastAPI Frontend
```bash
pip install fastapi uvicorn python-multipart requests pillow
python app.py
```

### 6. Access the Application
- **Web UI**: http://localhost:8000
- **TorchServe Inference**: http://localhost:8080/predictions/z-image
- **TorchServe Management**: http://localhost:8081/models

## 🔧 API Endpoints

### FastAPI (Port 8000)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web frontend |
| `/health` | GET | Health check |
| `/generate` | POST | Generate image from prompt |
| `/gallery` | GET | List generated images |

### TorchServe (Ports 8080-8082)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ping` | GET | Health check |
| `/predictions/z-image` | POST | Generate image |
| `/models` | GET | List models |
| `/models/z-image` | GET | Model details |

## 📝 Example Usage

### Test with curl
```bash
# Check health
curl http://localhost:8080/ping

# Generate image
curl -X POST http://localhost:8080/predictions/z-image \
    -d "A majestic dragon flying over a mountain at sunset"
```

### Test with Python script
```bash
python test_endpoint.py -p "A cyberpunk city at night with neon lights"
```

## 🎨 Sample Prompts

The frontend includes these curated prompts (and more!):

- "Cinematic close-up of red grapes on a marble table, soft diffused lighting"
- "Renaissance style painting of a majestic stag in a forest clearing"
- "Expressionist painting of bioluminescent forest with bold brushstrokes"
- "Dramatic macro shot of a garden spider in its web with morning dew"
- "Portrait of an elderly man with weathered face, Rembrandt lighting"
- "Ancient tree city with leaves made of stained glass, Studio Ghibli style"

## ⚡ Architecture

```
┌──────────────────┐     ┌─────────────────┐     ┌───────────────────┐
│   Web Browser    │────▶│  FastAPI :8000  │────▶│ TorchServe :8080  │
│    (Frontend)    │◀────│   (API Layer)   │◀────│  (Model Server)   │
└──────────────────┘     └─────────────────┘     └───────────────────┘
                                                          │
                                                          ▼
                                                 ┌───────────────────┐
                                                 │   Z-Image-Turbo   │
                                                 │   (GPU: ~20GB)    │
                                                 └───────────────────┘
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| Model Size | ~20GB |
| GPU Memory | ~20GB VRAM |
| Inference Steps | 50 |
| Generation Time | ~30-60 seconds |
| Image Resolution | 1024x1024 |

## 🤝 Credits

- Model: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- Serving: [PyTorch TorchServe](https://pytorch.org/serve/)
- API: [FastAPI](https://fastapi.tiangolo.com/)

## 📜 License

This project is for educational purposes (EMLO4 Session 12).
