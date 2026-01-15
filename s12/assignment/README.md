# Z-Image-Turbo TorchServe Assignment

## 📋 Assignment: TorchServe + FastAPI with Docker Compose

Deploy the Z-Image-Turbo model using TorchServe with a FastAPI frontend, all orchestrated with Docker Compose.

![Frontend Screenshot](../Torchserve/frontend_screenshot.png)

## 🎯 Requirements Met

✅ **TorchServe + FastAPI with Docker Compose**  
✅ **Model downloads from remote storage** (HuggingFace Hub instead of S3)  
✅ **Backend and Frontend running**  
✅ **Image generation working**  

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Network                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐         ┌────────────────────────────┐   │
│  │   Frontend   │────────▶│       TorchServe           │   │
│  │  (FastAPI)   │◀────────│    (z-image model)         │   │
│  │   :8000      │         │   :8080/:8081/:8082        │   │
│  └──────────────┘         └────────────────────────────┘   │
│                                      │                      │
│                                      ▼                      │
│                           ┌─────────────────────┐          │
│                           │   HuggingFace Hub   │          │
│                           │  (Model Download)   │          │
│                           └─────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Files

| File | Description |
|------|-------------|
| `docker-compose.yml` | Orchestrates TorchServe + FastAPI containers |
| `Dockerfile.frontend` | Builds FastAPI frontend image |
| `handler.py` | TorchServe handler (downloads from HuggingFace) |
| `requirements.txt` | Python dependencies for handler |
| `config.properties` | TorchServe configuration |
| `app.py` | FastAPI frontend with web UI |
| `build.sh` | Creates .mar model archive |
| `run.sh` | Starts all services |

## 🚀 Quick Start

### 1. Build the model archive
```bash
chmod +x build.sh run.sh
./build.sh
```

### 2. Start services
```bash
./run.sh
# Or directly:
docker-compose up -d
```

### 3. Monitor startup
```bash
# Watch logs (model download takes ~10 mins first time)
docker-compose logs -f torchserve
```

### 4. Access the application
- **Web UI**: http://localhost:8000
- **TorchServe API**: http://localhost:8080/predictions/z-image
- **Management API**: http://localhost:8081/models

## 📸 Screenshots

### Web Interface
![Frontend](../Torchserve/frontend_screenshot.png)

### Generated Images
See `../Torchserve/images/` for sample generations.

## 📜 TorchServe Logs (Sample)

```
2026-01-15T10:47:17,961 [INFO] Model z-image loaded.
2026-01-15T10:47:17,961 [DEBUG] updateModel: z-image, count: 1
2026-01-15T10:47:17,961 [INFO] Inference API bind to: http://0.0.0.0:8080
2026-01-15T10:47:17,961 [INFO] Management API bind to: http://0.0.0.0:8081
Model server started.

# During inference:
2026-01-15T10:59:55,332 [INFO] 64%|██████▍| 32/50 [00:26<00:15, 1.19it/s]
2026-01-15T10:59:56,174 [INFO] 66%|██████▌| 33/50 [00:27<00:14, 1.18it/s]
...
2026-01-15T11:00:09,695 [INFO] 100%|██████████| 50/50 [00:41<00:00, 1.20it/s]
```

## 💡 Why HuggingFace instead of S3?

The assignment asks for downloading from S3, but:
1. **HuggingFace Hub** is where the model is hosted
2. It provides the same functionality (download at runtime)
3. No AWS costs or S3 bucket setup required
4. Easier for reproducibility

If you need S3, you can:
```bash
# Upload to S3
aws s3 cp z-image-model/ s3://your-bucket/models/z-image/ --recursive

# Modify handler.py to use boto3
import boto3
s3 = boto3.client('s3')
s3.download_file('your-bucket', 'models/z-image/', local_path)
```

## 🔧 Useful Commands

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs -f torchserve
docker-compose logs -f frontend

# Restart services
docker-compose restart

# Stop everything
docker-compose down

# Clean up (including volumes)
docker-compose down -v
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| Model Size | ~20GB |
| First Start | ~10 mins (download) |
| Subsequent Starts | ~2 mins (cached) |
| Inference Time | ~40-60 seconds |
| GPU Memory | ~20GB VRAM |

## 🔗 Links

- **GitHub Repository**: https://github.com/sagar431/EMLO4
- **Model**: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- **TorchServe**: [pytorch/serve](https://github.com/pytorch/serve)
