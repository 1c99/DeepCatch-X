# Docker Deployment

This directory contains Docker configuration for cross-platform deployment.

## Contents

- `Dockerfile` - Multi-stage build for creating optimized Docker images
- `docker-compose.yml` - Configuration for batch processing
- `run_docker.sh` - Helper script for Docker operations

## Usage

### Build Docker Image
```bash
# From project root
docker build -f deploy/docker/Dockerfile -t dcx:latest .

# Or use the helper script
cd deploy/docker
./run_docker.sh build
```

### Run with Docker
```bash
# Single file
docker run -v $(pwd)/input:/data -v $(pwd)/output:/results dcx:latest \
  --input /data/sample.dcm --output_dir /results --all_modules

# Batch processing
docker-compose -f deploy/docker/docker-compose.yml up
```

## Deployment

The Docker image can be:
- Pushed to Docker Hub or private registry
- Deployed to any cloud service (AWS, GCP, Azure)
- Run on any system with Docker installed