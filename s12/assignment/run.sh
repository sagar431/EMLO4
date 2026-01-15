#!/bin/bash
# Run script for Z-Image-Turbo TorchServe + FastAPI

set -e

echo "🚀 Starting Z-Image-Turbo services..."

# Check if .mar exists
if [ ! -f "models/z-image.mar" ]; then
    echo "⚠️  Model archive not found. Building..."
    ./build.sh
fi

# Start services
echo "🐳 Starting Docker Compose..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to start..."
echo "   (First run takes ~10 minutes to download model from HuggingFace)"
echo ""
echo "📊 Monitor logs with: docker-compose logs -f"
echo "🌐 Frontend will be available at: http://localhost:8000"
echo "🔧 TorchServe API: http://localhost:8080"
echo ""
echo "💡 Check model status:"
echo "   curl http://localhost:8081/models"
