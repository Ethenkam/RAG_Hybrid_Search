# Use Ubuntu 24.04 LTS as base image for Intel Arc support
FROM ubuntu:24.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    gnupg2 \
    software-properties-common \
    git \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Add Intel GPU repository for Arc drivers
# Using 'client' component for Intel Arc GPUs on Ubuntu 24.04 (noble)
RUN wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client" | tee /etc/apt/sources.list.d/intel-gpu-noble.list && \
    apt-get update && apt-get install -y \
    intel-opencl-icd \
    intel-level-zero-gpu \
    level-zero \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV WORKDIR=/app

WORKDIR /app

# Install Python dependencies
# We specify extra index URLs for:
# 1. PyTorch Nightly XPU builds (required for torch 2.7.0+xpu)
# 2. Intel Extension for PyTorch (stable release channel as fallback)
COPY api/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip --break-system-packages && \
    pip install --no-cache-dir -r requirements.txt --break-system-packages \
    --extra-index-url https://download.pytorch.org/whl/nightly/xpu \
    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Set the entrypoint to start the application
ENTRYPOINT ["python3", "api/start.py"]
