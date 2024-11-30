FROM python:3.9-slim

WORKDIR /workspace

# Install only CPU version of PyTorch and core dependencies to reduce size
RUN pip install --no-cache-dir numpy==1.24.3 && \
    pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install other requirements
COPY requirements.txt .
RUN pip install --no-cache-dir tqdm black pillow && \
    rm -rf /root/.cache/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Command to run the training
CMD ["python", "train.py"]
