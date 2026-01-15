# Dog Breed Classifier on Kubernetes

A FastAPI-based Dog Breed Classifier with FastHTML frontend, deployed on Minikube Kubernetes cluster.

## 📁 Project Structure

```
assignment/
├── app.py                 # FastAPI app with FastHTML frontend
├── Dockerfile             # Docker container configuration
├── requirements.txt       # Python dependencies
├── k8s/
│   ├── deployment.yaml    # Kubernetes Deployment (2 replicas)
│   ├── service.yaml       # Kubernetes Service
│   └── ingress.yaml       # Kubernetes Ingress
├── README.md              # This file
└── kubectl_outputs.md     # Output of kubectl commands
```

## 🚀 Prerequisites

- Docker installed and running
- Minikube installed
- kubectl installed

## 📋 Instructions

### 1. Start Minikube

```bash
# Start Minikube with Docker driver
minikube start --driver=docker

# Verify cluster is running
minikube status

# Enable required addons
minikube addons enable ingress
minikube addons enable metrics-server
```

### 2. Build the Docker Image

```bash
# Point shell to Minikube's Docker daemon
eval $(minikube docker-env)

# Build the image
docker build -t dog-classifier:latest .

# Verify image is built
docker images | grep dog-classifier
```

### 3. Create the Deployment

```bash
# Apply all Kubernetes manifests
kubectl apply -f k8s/

# Verify deployment
kubectl get all

# Wait for pods to be ready
kubectl get pods -w
```

### 4. Tunnel to the Ingress

```bash
# Option 1: Use minikube tunnel (requires sudo)
minikube tunnel

# Option 2: Port forward directly
kubectl port-forward service/dog-classifier-service 8000:80
```

### 5. Access the Application

#### Via Port Forward:
- **Frontend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

#### Via Ingress (after adding to /etc/hosts):
Add the following to `/etc/hosts`:
```
127.0.0.1 dog-classifier.localhost
```

Then access:
- **Frontend**: http://dog-classifier.localhost
- **API Docs**: http://dog-classifier.localhost/docs

## 🔧 Useful Commands

```bash
# View deployment status
kubectl describe deployment dog-classifier-deployment

# View pod logs
kubectl logs -l app=dog-classifier -f

# Check pod metrics
kubectl top pod

# Check node metrics
kubectl top node

# Scale deployment
kubectl scale deployment dog-classifier-deployment --replicas=3

# Delete deployment
kubectl delete -f k8s/
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | FastHTML frontend |
| `/docs` | GET | Swagger API documentation |
| `/predict` | POST | Classify an uploaded image |
| `/health` | GET | Health check endpoint |

## 🐳 Docker Commands

```bash
# Build image locally (outside minikube)
docker build -t dog-classifier:latest .

# Run container locally for testing
docker run -p 8000:8000 dog-classifier:latest

# Load image into minikube (alternative to eval)
minikube image load dog-classifier:latest
```

## 🏗️ Architecture

```
                    ┌─────────────────┐
                    │     Ingress     │
                    │  (nginx-based)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │     Service     │
                    │  (ClusterIP)    │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     ┌────────▼────────┐           ┌────────▼────────┐
     │      Pod 1      │           │      Pod 2      │
     │  dog-classifier │           │  dog-classifier │
     └─────────────────┘           └─────────────────┘
```

## 📝 Notes

- The deployment uses `imagePullPolicy: Never` since we're building the image directly in Minikube's Docker daemon
- Session affinity is enabled in Ingress to ensure consistent routing for the frontend
- Health checks are configured with appropriate delays to allow model loading time
- Resource limits are set to prevent pods from consuming too much cluster resources
