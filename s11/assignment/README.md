# 🐱 Cat vs Dog Classifier - Serverless Deployment

> **Session 11 Assignment**: Create a CI/CD Pipeline to deploy/update the model to a serverless platform

[![Deploy to Modal](https://img.shields.io/badge/Deploy-Modal-green?style=for-the-badge&logo=modal)](https://modal.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-blue?style=for-the-badge)](https://onnxruntime.ai/)

---

## 🚀 Live Demo

**Production URL:** [https://ailab--cat-dog-classifier-fasthtml-web.modal.run](https://ailab--cat-dog-classifier-fasthtml-web.modal.run)

---

## 📸 Application Screenshot

![Cat vs Dog Classifier Demo](demo.png)

---

## 🎯 Assignment Objective

Create a CI/CD Pipeline to deploy/update the model to a serverless platform.

### Why Modal instead of AWS Lambda?

| Feature | AWS Lambda | Modal |
|---------|------------|-------|
| GPU Support | ❌ No | ✅ Yes (T4, A100, H100) |
| Cold Starts | ⏱️ Slow (seconds) | ⚡ Fast (milliseconds) |
| Python Native | ❌ Requires Docker/SAM | ✅ Pure Python |
| ML Friendly | ❌ Size limits | ✅ Built for ML |
| Pricing | 💰 Pay per request | 💰 Pay per second |

---

## 📁 Project Structure

```
s11/
├── assignment/
│   ├── modal_fasthtml_app.py   # 🚀 Main Modal application
│   ├── model.onnx              # 🧠 ONNX model for inference
│   ├── export_onnx.py          # 🔄 PyTorch to ONNX converter
│   ├── demo.png                # 📸 Application screenshot
│   └── README.md               # 📖 This file
│
└── serverless/
    ├── modal_app.py            # Gradio version (alternative)
    ├── app.py                  # Original Gradio app
    └── ...

.github/workflows/
└── modal-deploy.yml            # 🔄 CI/CD pipeline
```

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| **Serverless Platform** | Modal |
| **Web Framework** | FastAPI |
| **UI Framework** | FastHTML + Shad4Fast |
| **Inference Runtime** | ONNX Runtime |
| **CI/CD** | GitHub Actions |
| **Model Format** | ONNX (converted from PyTorch) |

---

## 🔄 CI/CD Pipeline

The GitHub Actions workflow automatically deploys on every push:

```yaml
# .github/workflows/modal-deploy.yml
name: Deploy to Modal

on:
  push:
    branches: [main]
    paths: ['s11/assignment/**']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install modal
      - run: modal deploy modal_fasthtml_app.py
```

### Required GitHub Secrets

| Secret Name | Description |
|-------------|-------------|
| `MODAL_TOKEN_ID` | Your Modal API token ID |
| `MODAL_TOKEN_SECRET` | Your Modal API token secret |

**Get your tokens:**
```bash
modal token new
cat ~/.modal.toml  # View token_id and token_secret
```

---

## 🏃 Local Development

### Prerequisites
- Python 3.11+
- Modal CLI

### Quick Start

```bash
# Install Modal CLI
pip install modal

# Authenticate
modal token new

# Run in development mode (hot reload)
modal serve modal_fasthtml_app.py

# Deploy to production
modal deploy modal_fasthtml_app.py
```

---

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GitHub Repository                        │
│  ┌─────────────┐                                             │
│  │ Push to     │                                             │
│  │ main branch │                                             │
│  └──────┬──────┘                                             │
└─────────┼───────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Checkout    │─▶│ Setup       │─▶│ modal deploy        │  │
│  │ Code        │  │ Python      │  │ modal_fasthtml_app  │  │
│  └─────────────┘  └─────────────┘  └──────────┬──────────┘  │
└───────────────────────────────────────────────┼─────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────┐
│                       Modal Cloud                            │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                   Serverless Function                    ││
│  │  ┌───────────┐  ┌───────────┐  ┌───────────────────┐   ││
│  │  │ FastAPI   │  │ FastHTML  │  │ ONNX Runtime      │   ││
│  │  │ Backend   │  │ UI        │  │ Inference         │   ││
│  │  └───────────┘  └───────────┘  └───────────────────┘   ││
│  └─────────────────────────────────────────────────────────┘│
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  https://ailab--cat-dog-classifier-fasthtml-web.modal.run││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## 🖼️ Features

- ✨ **Beautiful UI** - Modern, responsive design with Shadcn-like components
- ⚡ **Real-time Classification** - Upload an image and get instant predictions
- 🔌 **REST API** - Programmatic access via `/predict` endpoint
- 📊 **Confidence Scores** - Visual progress bars showing prediction confidence
- 🌐 **CORS Enabled** - Can be called from any frontend application

---

## 📝 API Usage

### Web Interface
Visit: [https://ailab--cat-dog-classifier-fasthtml-web.modal.run](https://ailab--cat-dog-classifier-fasthtml-web.modal.run)

### REST API

**Endpoint:** `POST /predict`

```bash
curl -X POST \
  "https://ailab--cat-dog-classifier-fasthtml-web.modal.run/predict" \
  -F "file=@cat.jpg"
```

**Response:**
```json
{
  "predictions": {
    "Cat": 0.656,
    "Dog": 0.344
  },
  "success": true,
  "message": "Classification successful"
}
```

---

## 🔗 Links

| Resource | URL |
|----------|-----|
| 🌐 Live Demo | [modal.run](https://ailab--cat-dog-classifier-fasthtml-web.modal.run) |
| 📊 Modal Dashboard | [modal.com](https://modal.com/apps/ailab/main/deployed/cat-dog-classifier-fasthtml) |
| 📦 GitHub Repo | [github.com/sagar431/EMLO4](https://github.com/sagar431/EMLO4) |

---

## 📄 License

MIT License - Feel free to use this for your own projects!

---

<p align="center">
  Made with ❤️ using Modal, FastAPI, FastHTML, and ONNX Runtime
</p>
