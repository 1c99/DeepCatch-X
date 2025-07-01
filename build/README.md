# Build Directory

This directory contains scripts and tools for building DCX executables.

## Contents

- `compile_with_nuitka.py` - Interactive Nuitka compilation script for creating standalone executables

## Usage

### Local Nuitka Build
```bash
cd build
python compile_with_nuitka.py
```

This will create native executables for your platform in the `dist/` directory.

## Build Output

Compiled executables will be placed in:
- `../dist/inference/` - Standard inference executable
- `../dist/inference_ray/` - Ray-enabled inference executable