# ✨ Gemma Chat on Kubernetes

<div align="center">

![Kubernetes](https://img.shields.io/badge/kubernetes-%23326ce5.svg?style=for-the-badge&logo=kubernetes&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-FFD21E?style=for-the-badge)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

**Chat with Google's Gemma-3-270M-IT model deployed on Kubernetes!**

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

```bash
kubectl port-forward service/gemma-chat-service 8001:80
```

Open: http://localhost:8001

---

## 📁 Project Structure

```
gemma-chat/
├── app.py              # FastAPI + Chat UI
├── Dockerfile          
├── requirements.txt    
└── k8s/
    ├── deployment.yaml 
    ├── service.yaml    
    └── ingress.yaml    
```

---

## 🎨 Features

- 💬 Modern chat interface with typing indicators
- 🤗 Pulls Gemma model from Hugging Face
- ⚡ FastAPI backend with async support
- 📚 Auto-generated API docs at `/docs`

---

**Just for fun! 🎉**
