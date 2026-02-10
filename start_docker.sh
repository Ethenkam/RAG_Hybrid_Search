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

echo "🚀 Building and starting the container..."
check_api_key

# Run docker-compose
# Pass the environment variable explicitly
MISTRAL_API_KEY=$MISTRAL_API_KEY docker compose up --build

