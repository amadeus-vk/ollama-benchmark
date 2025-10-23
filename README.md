# Ollama AMD GPU Benchmark

Benchmark container for comparing Ollama performance between CPU and AMD GPU (Vulkan) on Radeon RX Vega M GH.

## Features

- Automatic detection of AMD GPU and Vulkan support
- Performance comparison between CPU and GPU inference
- CSV results export and visualization
- Health checks and error handling
- Portainer-compatible deployment

## Usage

1. Ensure Ollama is running with Vulkan support on host
2. Clone and deploy:

```bash
git clone amadeus-vk/ollama-benchmark
cd ollama-amd-benchmark
docker-compose up -d
