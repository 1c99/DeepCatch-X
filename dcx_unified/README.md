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
│   ├── lung_volumetry.yaml
│   ├── aorta.yaml
│   ├── t12l1.yaml
│   ├── laa.yaml
│   ├── tb.yaml
│   └── t12l1_regression.yaml
├── inference.py            # Unified inference script
├── postprocessing.py       # Unified postprocessing script
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

#### Basic Segmentation
- **lung**: Lung segmentation
- **heart**: Heart segmentation  
- **airway**: Airway segmentation
- **bone**: Bone segmentation
- **aorta**: Aorta segmentation (outputs both ascending and descending aorta as separate channels)
- **t12l1**: T12 and L1 vertebrae segmentation

#### Lung-based Processing
- **covid**: COVID-19 pattern detection (requires lung mask)
- **vessel**: Vascular segmentation (requires lung mask)

#### Volumetry (Segmentation + Regression)
- **heart_volumetry**: Heart segmentation + volume estimation
- **lung_volumetry**: Lung segmentation + volume estimation
- **t12l1_regression**: T12/L1 segmentation + bone density regression

#### Diffusion Models
- **bone_supp**: Bone suppression using diffusion model

#### Alternative Architectures
- **laa**: Left atrial appendage segmentation (segmentation_models_pytorch)
- **tb**: Tuberculosis classification (EfficientNet)

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

## Postprocessing

The unified system includes postprocessing capabilities for clinical measurements:

### Aorta Maximum Diameter
```bash
python postprocessing.py --function aorta_diameter --input1 aorta_segmentation.nii --pixel_spacing 0.18
```

### Cardiothoracic Ratio
```bash
python postprocessing.py --function cardiothoracic_ratio --input1 heart_segmentation.nii --input2 lung_segmentation.nii
```

### Peripheral Vessel Area
```bash
python postprocessing.py --function peripheral_vessels --input1 vessel_segmentation.nii --input2 lung_segmentation.nii --pixel_spacing 0.18
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