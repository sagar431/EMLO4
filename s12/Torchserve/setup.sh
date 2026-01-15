#!/bin/bash
# Setup script for Z-Image-Turbo environment using uv

set -e

echo "🚀 Setting up Z-Image-Turbo environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create virtual environment and install dependencies
echo "📦 Creating virtual environment and installing dependencies..."
uv venv
uv pip install -e .

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the script:"
echo "  python z_image_turbo.py --prompt 'Your prompt here'"
echo ""
echo "Examples:"
echo "  # Local GPU inference"
echo "  python z_image_turbo.py -p 'Astronaut in a jungle, cold color palette, muted colors, detailed, 8k'"
echo ""
echo "  # Remote API inference (requires HF_TOKEN)"
echo "  export HF_TOKEN='your_token_here'"
echo "  python z_image_turbo.py -p 'Astronaut riding a horse' -m remote"
