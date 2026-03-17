# NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Symlink python
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install PyTorch with CUDA support first
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Copy and install requirements
COPY api/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000

CMD python data_loader/load_and_clean.py && \
    python indexing/build_index.py && \
    python api/start.py
