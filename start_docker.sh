#!/bin/bash

# Ensure the script is executable
# chmod +x start_docker.sh

# Function to check if MISTRAL_API_KEY is set
check_api_key() {
    if [ -z "$MISTRAL_API_KEY" ]; then
        # Check if .env file exists and contains the key
        if [ -f .env ] && grep -q "MISTRAL_API_KEY" .env; then
            export $(grep -v '^#' .env | xargs)
        fi
    fi

    if [ -z "$MISTRAL_API_KEY" ]; then
        echo "❌ MISTRAL_API_KEY is not set."
        read -p "Please enter your Mistral API Key: " api_key_input
        if [ -z "$api_key_input" ]; then
            echo "❌ Key is required. Exiting."
            exit 1
        fi
        export MISTRAL_API_KEY="$api_key_input"
        echo "✅ API Key set temporarily for this session."
    else
        echo "✅ MISTRAL_API_KEY found."
    fi
}

# Check for Docker and Docker Compose
if ! command -v docker &> /dev/null; then
    echo "❌ Docker could not be found. Please install Docker first."
    exit 1
fi

# Check for NVIDIA Container Toolkit
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  nvidia-smi not found. Make sure NVIDIA drivers are installed."
    echo "   For NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

echo "🚀 Building and starting the container (NVIDIA GPU)..."
check_api_key

# Run docker-compose with NVIDIA GPU support
MISTRAL_API_KEY=$MISTRAL_API_KEY docker compose up --build
