# DCX Unified Inference System

A unified medical imaging inference system that consolidates 11 different X-ray analysis modules into a single, efficient architecture.

## Overview

DCX Unified is a complete refactoring of the original DCX_python_inference system, achieving:
- **76.6% code reduction** (34,266 → 8,017 lines)
- **86% fewer files** (110 → 15 files)
- **Single entry point** for all modules
- **Maintained 100% functionality**

## Features

### Segmentation Modules
1. **Lung** - Lung segmentation with area/volume calculations
2. **Heart** - Heart segmentation with area/volume calculations
3. **Airway** - Airway structure segmentation
4. **Bone** - Bone structure segmentation
5. **Aorta** - Aortic arch segmentation (ascending/descending)
6. **T12/L1** - Vertebrae detection and segmentation
7. **LAA** - Low attenuation area (emphysema) detection
8. **TB** - Tuberculosis detection
9. **Bone Suppression** - Bone structure suppression
10. **COVID** - COVID-19 pneumonia detection
11. **Vessel** - Vascular structure segmentation

### Post-Processing Modules
- **CTR** - Cardiothoracic ratio calculation
- **Peripheral** - Peripheral lung area analysis
- **Diameter** - Aorta diameter measurement

### Additional Features
- Batch processing support
- Multiple output formats (NIfTI, DICOM, NPZ, PNG)
- Automatic measurement collection
- Volume calculations
- JSON metadata export

## Installation

1. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python inference.py --input_file path/to/image.dcm --output_dir ./results --module lung
```

### Run All Modules
```bash
python inference.py --input_file path/to/image.dcm --output_dir ./results --all_modules
```

### Specific Modules
```bash
# Single module
python inference.py --input_file image.dcm --output_dir ./results --module heart

# Multiple modules
python inference.py --input_file image.dcm --output_dir ./results --lung --heart --airway
```

### Post-Processing
```bash
# CTR calculation
python inference.py --input_file image.dcm --output_dir ./results --ctr

# Peripheral area
python inference.py --input_file image.dcm --output_dir ./results --peripheral

# Aorta diameter
python inference.py --input_file image.dcm --output_dir ./results --diameter
```

### Batch Processing
```bash
python inference.py --input_dir ./input_folder --output_dir ./results --all_modules
```

### Advanced Options
```bash
# Custom output format
python inference.py --input_file image.dcm --output_dir ./results --module lung --output_format png

# Collect measurements
python inference.py --input_file image.dcm --output_dir ./results --all_modules --collect_measurements

# Generate metadata
python inference.py --input_file image.dcm --output_dir ./results --module heart --metadata
```

## Output Structure

```
results/
├── {filename}_lung.nii          # Lung segmentation mask
├── {filename}_heart.nii         # Heart segmentation mask
├── {filename}_measurements.json # Collected measurements
└── ...
```

## Module Dependencies

Some modules depend on others:
- **COVID** and **Vessel** require lung segmentation
- **TB** module uses lung segmentation for preprocessing
- Post-processing modules require corresponding segmentation masks

## Architecture Improvements

### Before (DCX_python_inference)
- 110 separate Python files
- Scattered across multiple directories
- Duplicate code across modules
- Inconsistent interfaces

### After (dcx_unified)
- Single unified `inference.py`
- Modular architecture with shared utilities
- Consistent error handling
- Clean separation of concerns

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA-capable GPU (recommended) or MPS (Apple Silicon)
- 8GB+ RAM

## Checkpoints

Model checkpoints should be placed in the `checkpoints/` directory:
```
checkpoints/
├── xray2lung.pth
├── xray2heart.pth
├── xray2airway.pth
└── ...
```

## Performance

- Processes single X-ray in ~30-60 seconds (all modules)
- GPU acceleration supported (CUDA/MPS)
- Efficient batch processing
- Memory-optimized for large datasets

## Contributing

When adding new modules:
1. Add module configuration to `module_configs`
2. Implement processing logic in appropriate method
3. Update command-line arguments
4. Add to documentation

## License

[License information here]

## Citation

[Citation information here]