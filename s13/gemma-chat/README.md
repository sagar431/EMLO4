# ✨ TinyLlama Chat on Kubernetes

<div align="center">

![Kubernetes](https://img.shields.io/badge/kubernetes-%23326ce5.svg?style=for-the-badge&logo=kubernetes&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![TinyLlama](https://img.shields.io/badge/🦙%20TinyLlama-FFD21E?style=for-the-badge)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

**Chat with TinyLlama-1.1B-Chat model deployed on Kubernetes!**

</div>

---

## 🚀 Quick Start

### 1️⃣ Build Docker Image

```bash
# Use Minikube's Docker daemon
eval $(minikube docker-env)

# Build the image
docker build -t gemma-chat:latest .
```

### 2️⃣ Deploy to Kubernetes

```bash
kubectl apply -f k8s/
```

### 3️⃣ Access the App

**For Lambda Labs / Remote:**
Run this on your LOCAL machine:
```bash
ssh -L 8001:localhost:8001 ubuntu@<YOUR_LAMBDA_IP>
```

Then open: **http://localhost:8001**

---

## 📁 Project Structure

```
gemma-chat/
├── app.py              # FastAPI + TinyLlama Chat UI
├── Dockerfile          
├── requirements.txt    
└── k8s/
    ├── deployment.yaml # 1 Replica, 2GB RAM Request
    ├── service.yaml    # ClusterIP Service
    └── ingress.yaml    # Ingress (host: gemma-chat.localhost)
```

---

## 🎨 Features

- 💬 Modern chat interface with glassmorphism design
- 🦙 **TinyLlama-1.1B-Chat** model (Open Source, No Auth required!)
- ⚡ FastAPI backend with async support
- 🌊 Streaming-like user experience (typing indicators)

---

**Note:** The first time you deploy, it may take 2-3 minutes to download the model (2.2GB). Check status with:
`kubectl logs -l app=gemma-chat -f`
