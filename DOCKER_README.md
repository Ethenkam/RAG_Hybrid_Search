# Docker Setup for Intel Arc B580

This project can be run inside a Docker container configured for Intel Arc GPUs (XPU). It uses the official `intel/intel-extension-for-pytorch:2.7.10-xpu` image.

## Prerequisites

1.  **Intel GPU Drivers**: Ensure your host machine has the latest Intel Arc drivers installed.
    *   Ubuntu 24.04: `sudo apt install intel-opencl-icd intel-level-zero-gpu level-zero`
    *   Windows (WSL2): Install Windows drivers; WSL2 usually passes them through automatically.
2.  **Docker**: Install Docker and Docker Compose.
3.  **Mistral API Key**: You need an API key from [Mistral AI](https://console.mistral.ai/).

## Quick Start

Run the helper script:

```bash
chmod +x start_docker.sh
./start_docker.sh
```

It will:
1.  Check for `MISTRAL_API_KEY` (prompting if missing).
2.  Build the Docker image.
3.  Start the container with GPU access.

## Manual Run

If you prefer `docker-compose` directly:

1.  Set the API key:
    ```bash
    export MISTRAL_API_KEY=your_key_here
    ```
2.  Run:
    ```bash
    docker-compose up --build
    ```

## File Structure

*   `Dockerfile`: Builds the image based on Intel's XPU image.
*   `docker-compose.yml`: Defines the service, GPU mapping (`/dev/dri`), and volume mounts.
*   `.dockerignore`: Excludes unnecessary files from the build context.

## Troubleshooting

*   **Permission Denied `/dev/dri`**: Ensure your user is in the `render` group: `sudo usermod -aG render $USER`.
*   **Missing Dependencies**: If you see `ModuleNotFoundError`, check `Dockerfile` filtering logic for `requirements.txt`.
