#!/bin/bash
# Build script for Z-Image-Turbo TorchServe deployment
# Creates .mar file (lightweight - model downloads from HuggingFace)

set -e

echo "🔧 Building Z-Image-Turbo deployment..."

# Create directories
mkdir -p models model_store

# Create empty file for extra-files requirement
touch empty.txt

# Create .mar file (only contains handler + requirements, no model weights)
echo "📦 Creating model archive..."
torch-model-archiver \
    --model-name z-image \
    --version 1.0 \
    --handler handler.py \
    --extra-files empty.txt \
    -r requirements.txt \
    --archive-format zip-store \
    --force

# Move to models directory
mv z-image.mar models/

# Cleanup
rm -f empty.txt

echo "✅ Model archive created: models/z-image.mar"
echo ""
echo "📋 Next steps:"
echo "   1. Run: docker-compose up -d"
echo "   2. Wait for model to download from HuggingFace (~10 mins first time)"
echo "   3. Access frontend: http://localhost:8000"
echo "   4. Check logs: docker-compose logs -f torchserve"
