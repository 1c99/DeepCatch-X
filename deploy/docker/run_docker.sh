#!/bin/bash
# Helper script to run DCX in Docker

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}DCX Docker Runner${NC}"
echo "=================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Function to show usage
usage() {
    echo "Usage: $0 [build|run|inference|inference-ray] [options]"
    echo ""
    echo "Commands:"
    echo "  build         Build the Docker image"
    echo "  run           Run with docker-compose (processes input/ directory)"
    echo "  inference     Run inference.py with custom options"
    echo "  inference-ray Run inference_ray.py with custom options"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 run"
    echo "  $0 inference --input sample.dcm --output_dir results --module lung"
    echo "  $0 inference-ray --input /data/dicoms --output_dir results --all_modules"
    exit 1
}

# Parse command
if [ $# -eq 0 ]; then
    usage
fi

COMMAND=$1
shift

case $COMMAND in
    build)
        echo -e "${GREEN}Building DCX Docker image...${NC}"
        # Build from project root, using Dockerfile in deploy/docker
        (cd ../.. && docker build -f deploy/docker/Dockerfile -t dcx:latest .)
        ;;
    
    run)
        echo -e "${GREEN}Running DCX with docker-compose...${NC}"
        docker-compose up
        ;;
    
    inference)
        echo -e "${GREEN}Running inference.py...${NC}"
        docker run --rm \
            -v "$(pwd)/input:/data" \
            -v "$(pwd)/output:/results" \
            --entrypoint /app/inference \
            dcx:latest "$@"
        ;;
    
    inference-ray)
        echo -e "${GREEN}Running inference_ray.py...${NC}"
        docker run --rm \
            -v "$(pwd)/input:/data" \
            -v "$(pwd)/output:/results" \
            --entrypoint /app/inference_ray \
            dcx:latest "$@"
        ;;
    
    *)
        usage
        ;;
esac