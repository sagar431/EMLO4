# Base image with Python
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY .project-root .

# Install dependencies (CPU version of PyTorch for smaller image)
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN uv sync --no-dev

# Copy source code and configs
COPY src/ src/
COPY configs/ configs/

# Create necessary directories
RUN mkdir -p data logs predictions checkpoints outputs samples

# Set environment variables
ENV PYTHONPATH=/app
ENV PROJECT_ROOT=/app

# Default command - show help
CMD ["uv", "run", "python", "-c", "print('Dog Breed Classifier\\n\\nUsage:\\n  Eval: uv run python src/eval.py checkpoint_path=<path>\\n  Infer: uv run python src/infer.py input_folder=<path> checkpoint_path=<path>')"]
