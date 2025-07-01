# Deployment Directory

This directory contains deployment configurations and scripts for DCX.

## Structure

```
deploy/
├── docker/          # Docker-based deployment
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── run_docker.sh
└── nuitka/          # Native executables (created by build process)
```

## Deployment Methods

### 1. Docker Deployment (Recommended)
- Cross-platform compatibility
- Consistent environment
- Easy distribution
- See `docker/README.md` for details

### 2. Native Executables
- Platform-specific binaries created by Nuitka
- Smaller footprint
- No Docker required on target system
- Built using scripts in `../build/`

## Quick Start

### Docker
```bash
cd docker
./run_docker.sh build
./run_docker.sh inference-ray --input /path/to/data --output_dir results --all_modules
```

### Native
```bash
cd ../build
python compile_with_nuitka.py
# Executables will be in deploy/nuitka/
```