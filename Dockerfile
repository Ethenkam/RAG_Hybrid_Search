# Use the Intel Extension for PyTorch base image with XPU support
# This image includes PyTorch, IPEX, and necessary Intel drivers/libraries
FROM intel/intel-extension-for-pytorch:2.7.10-xpu

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies that might be missing in the base image
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    python3-venv \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY api/requirements.txt .

# Install Python dependencies
# We filter out torch, torchvision, torchaudio, and intel_extension_for_pytorch
# to use the pre-installed optimized versions from the base image.
# If strict version matching fails, pip might try to reinstall torch, so we use
# --no-deps for the filtered requirements if needed, but here we just remove the lines.
RUN grep -vE "torch|intel_extension_for_pytorch" requirements.txt > requirements_docker.txt && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_docker.txt

# Copy the application code
COPY . .

# Expose the application port
EXPOSE 8000

# Set the entrypoint
CMD python data_loader/load_and_clean.py && \
    python indexing/build_index.py && \
    python api/start.py
