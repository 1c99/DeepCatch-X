# DCX Unified Medical Imaging System

A unified architecture for all DCX medical imaging modules that eliminates code duplication while maintaining identical functionality.

## Architecture

```
dcx_unified/
├── core/                    # Shared core modules
│   ├── base_options.py      # Configuration and argument parsing
│   ├── utils.py            # Utility functions
│   ├── model.py            # Model architecture
│   ├── pix2pixHD_model.py  # Pix2PixHD generator
│   ├── data_loader.py      # Data loading utilities
│   ├── heartregression_model.py  # Heart volume regression
│   └── lungregression_model.py   # Lung volume regression
├── configs/                # Module-specific configurations
│   ├── lung.yaml
│   ├── heart.yaml
│   ├── airway.yaml
│   ├── covid.yaml
│   ├── vessel.yaml
│   ├── bone.yaml
│   ├── heart_volumetry.yaml
│   └── lung_volumetry.yaml
├── inference.py            # Unified inference script
├── test_unified.py         # Test script
└── modules/                # Checkpoints and data (created by test script)
```

## Usage

### Basic Inference

```bash
python inference.py --module lung --input input.dcm --output output.nii

python inference.py --module heart --input input.dcm --output output.nii

python inference.py --module covid --input input.dcm --output output.nii --lung_mask lung.nii
```

### Available Modules

- **lung**: Lung segmentation
- **heart**: Heart segmentation  
- **airway**: Airway segmentation
- **covid**: COVID-19 detection
- **vessel**: Vascular segmentation
- **bone**: Bone segmentation
- **heart_volumetry**: Heart segmentation + volume estimation
- **lung_volumetry**: Lung segmentation + volume estimation

### Modules Requiring Lung Masks

- **covid**: Requires pre-segmented lung mask
- **vessel**: Requires pre-segmented lung mask

### Testing

1. Setup test environment:
```bash
cd dcx_unified
python test_unified.py --setup
```

2. Test specific module:
```bash
python test_unified.py --module lung
```

3. Test all modules:
```bash
python test_unified.py --all
```

## Configuration

Each module has its own YAML configuration file in the `configs/` directory that specifies:

- Model checkpoint path
- Image resolution (512 or 2048) 
- Threshold values
- Normalization method
- Output processing options
- Whether lung mask is required

Example configuration (lung.yaml):
```yaml
name: "xray2lung"
checkpoint_path: "checkpoints/xray2lung.pth"
loadSize: 2048
threshold: -1015
output_range: [-1100, -500]
calculate_area: true
use_connected_regions: true
n_regions: 2
normalization_method: "percentile"
percentile_min: 0
percentile_max: 99
requires_lung_mask: false
```

## Benefits

- **Single codebase**: All functionality consolidated into one system
- **Easy maintenance**: Fix bugs once, benefit all modules
- **Consistent API**: Same interface for all modules
- **Configuration-driven**: Easy to add new modules or modify existing ones
- **Identical results**: Produces the same outputs as original separate modules

## Dependencies

- PyTorch
- PyDICOM  
- NiBabel
- NumPy
- PIL/Pillow
- PyYAML
- SciPy (for connected regions)

## Migration from Original Modules

The unified system produces identical results to the original separate modules. To migrate:

1. Use the test script to verify functionality matches
2. Update any existing scripts to use the new unified interface
3. Remove old duplicate modules once testing is complete

## Adding New Modules

1. Create a new YAML configuration file in `configs/`
2. Add any module-specific model files to `core/`
3. Test with the unified inference script
4. Update this README with the new module information