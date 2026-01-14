# Session 11 Assignment - Serverless Deployment with CI/CD

## 🎯 Assignment Objective
Create a CI/CD Pipeline to deploy/update the model to a serverless platform.

Since AWS Lambda doesn't support GPU and has cold start issues, we use **Modal** as our serverless platform which offers:
- ✅ GPU support (T4, A100, etc.)
- ✅ Automatic scaling
- ✅ Pay-per-use pricing
- ✅ Fast cold starts
- ✅ Easy Python-native deployment

## 🚀 Deployed Function URL

**Production URL:** https://ailab--cat-dog-classifier-fasthtml-web.modal.run

## 📁 Project Structure

```
assignment/
├── modal_fasthtml_app.py   # Main Modal application with FastHTML UI
├── model.onnx              # ONNX model for inference
├── export_onnx.py          # Script to convert PyTorch model to ONNX
└── README.md               # This file

.github/workflows/
└── modal-deploy.yml        # CI/CD pipeline for Modal deployment
```

## 🔧 Tech Stack

- **Serverless Platform:** Modal
- **Web Framework:** FastAPI + FastHTML
- **UI Components:** Shad4Fast (Shadcn-like components)
- **Inference Runtime:** ONNX Runtime
- **CI/CD:** GitHub Actions

## 🔄 CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/modal-deploy.yml`) automatically:
1. Triggers on push to `main` branch (when files in `s11/assignment/` change)
2. Sets up Python environment
3. Installs Modal CLI
4. Authenticates with Modal using secrets
5. Deploys the application

### Required GitHub Secrets

To enable CI/CD, add these secrets to your GitHub repository:

1. `MODAL_TOKEN_ID` - Your Modal token ID
2. `MODAL_TOKEN_SECRET` - Your Modal token secret

To get these tokens:
```bash
modal token new
```

Then add them to GitHub: Settings → Secrets and variables → Actions → New repository secret

## 🏃 Local Development

### Prerequisites
- Python 3.11+
- Modal CLI (`pip install modal`)

### Run Locally
```bash
# Authenticate with Modal
modal token new

# Serve in development mode (hot reload)
modal serve modal_fasthtml_app.py

# Deploy to production
modal deploy modal_fasthtml_app.py
```

## 📊 Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   GitHub Push   │────▶│  GitHub Actions  │────▶│  Modal Deploy   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │  Modal Function │
                                                 │  (Serverless)   │
                                                 │                 │
                                                 │  FastAPI +      │
                                                 │  FastHTML UI    │
                                                 │  ONNX Runtime   │
                                                 └─────────────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │  Public URL     │
                                                 │  (HTTPS)        │
                                                 └─────────────────┘
```

## 🖼️ Features

- **Beautiful UI:** Modern, responsive design with Shadcn-like components
- **Real-time Classification:** Upload an image and get instant predictions
- **API Endpoint:** REST API at `/predict` for programmatic access
- **Confidence Scores:** Visual progress bars showing prediction confidence

## 📝 API Usage

### POST /predict
```bash
curl -X POST "https://ailab--cat-dog-classifier-fasthtml-web.modal.run/predict" \
  -F "file=@cat.jpg"
```

Response:
```json
{
  "predictions": {"Cat": 0.95, "Dog": 0.05},
  "success": true,
  "message": "Classification successful"
}
```

## 🔗 Links

- **Live Demo:** https://ailab--cat-dog-classifier-fasthtml-web.modal.run
- **Modal Dashboard:** https://modal.com/apps/ailab/main/deployed/cat-dog-classifier-fasthtml
