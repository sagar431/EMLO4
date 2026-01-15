#!/bin/bash
set -e

echo "📦 Zipping model artifacts..."
cd z-image-model
zip -0 -r ../z-image-model.zip *
cd ..
echo "✅ Created z-image-model.zip"
