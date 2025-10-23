#!/bin/bash
version='0.0.2'

# Build and deploy to Portainer
docker-compose build
docker-compose up -d

echo "Benchmark stack deployed!"
echo "Check Portainer for logs and results in ./results directory"